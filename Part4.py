import argparse
import cv2
import numpy as np
import math
import os
import copy
# Minimum number of matches that have to be found
# to consider the recognition valid
MIN_MATCHES = 8

class OBJ:
    def __init__(self, filename, swapyz=False):
        """Loads a Wavefront OBJ file. """
        self.vertices = []
        self.normals = []
        self.texcoords = []
        self.faces = []
        material = None
        for line in open(filename, "r"):
            if line.startswith('#'): continue
            values = line.split()
            if not values: continue
            if values[0] == 'v':
                v = tuple(map(float, values[1:4]))
                # for key, values in v.items():
                #     print(key)
                # for key in v:
                #     print(key)
                #     print(v[key])
                if swapyz:
                    v = v[0], v[2], v[1]
                self.vertices.append(v)
            elif values[0] == 'vn':
                v = tuple(map(float, values[1:4]))
                if swapyz:
                    v = v[0], v[2], v[1]
                self.normals.append(v)
            elif values[0] == 'vt':
                self.texcoords.append(map(float, values[1:3]))
            #elif values[0] in ('usemtl', 'usemat'):
                #material = values[1]
            #elif values[0] == 'mtllib':
                #self.mtl = MTL(values[1])
            elif values[0] == 'f':
                face = []
                texcoords = []
                norms = []
                for v in values[1:]:
                    w = v.split('/')
                    face.append(int(w[0]))
                    if len(w) >= 2 and len(w[1]) > 0:
                        texcoords.append(int(w[1]))
                    else:
                        texcoords.append(0)
                    if len(w) >= 3 and len(w[2]) > 0:
                        norms.append(int(w[2]))
                    else:
                        norms.append(0)
                #self.faces.append((face, norms, texcoords, material))
                self.faces.append((face, norms, texcoords))

def detectFeaturesKeys(image):
    descriptor = cv2.xfeatures2d.SIFT_create()
    (kps, features) = descriptor.detectAndCompute(image, None)
    return (kps, features)

def showImage(frame, time=1000):
    cv2.imshow('frame', frame)
    cv2.waitKey(time)
    return

def main():
    """
    This functions loads the target surface image,
    """
    homography = None 
    # matrix of camera parameters (made up but works quite well for me) 
    camera_parameters = np.array([[756.56499986, 0, 493.2992946 ], [ 0, 753.44416051, 304.00857278],[ 0, 0, 1]])
    # load the reference surface that will be searched in the video stream
    dir_name = os.getcwd()
    # starting model
    model1 = cv2.imread(os.path.join(dir_name, 'reference/model.png'), 0)
    # end model
    model2 = cv2.imread(os.path.join(dir_name, 'reference/model42.png'), 0)
    # Compute model keypoints and its descriptors
    kp_model1, des_model1 = detectFeaturesKeys(model1)
    kp_model2, des_model2 = detectFeaturesKeys(model2)
    # Load 3D model from OBJ file
    obj = OBJ(os.path.join(dir_name, 'models/pirate-ship-fat.obj'), swapyz=True)  
    # init video capture
    frame = cv2.imread('images4/10.jpg')
    # read the current frame
    kp_frame, des_frame = detectFeaturesKeys(frame)
    # match frame descriptors with model descriptors
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)   # or pass empty dictionary

    flann = cv2.FlannBasedMatcher(index_params,search_params)

    matches = flann.knnMatch(des_model1,des_frame,k=2)

    # ratio test as per Lowe's paper
    good = []
    for m,n in matches:
        if m.distance < 0.65*n.distance:
            good.append(m)
    # compute Homography if enough matches are found
    if len(good) > MIN_MATCHES:
        # differenciate between source points and destination points
        src_pts = np.float32([kp_model1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        # compute Homography
        homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()

    else:
        print("Not enough matches found - "+ str(len(matches)/MIN_MATCHES))

    #--------------------------    dest    ----------------------------
    
    frame = cv2.imread('images4/10.jpg')
    # read the current frame
    kp_frame, des_frame = detectFeaturesKeys(frame)
    # match frame descriptors with model descriptors
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)   # or pass empty dictionary

    flann = cv2.FlannBasedMatcher(index_params,search_params)

    matches = flann.knnMatch(des_model2,des_frame,k=2)

    # ratio test as per Lowe's paper
    good = []
    for m,n in matches:
        if m.distance < 0.65*n.distance:
            good.append(m)
    # compute Homography if enough matches are found
    if len(good) > MIN_MATCHES:
        # differenciate between source points and destination points
        src_pts = np.float32([kp_model2[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        # compute Homography
        homography2, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    else:
        print("Not enough matches found - "+ str(len(matches)/MIN_MATCHES))
    
    if homography is not None and homography2 is not None:
        try:
            # obtain 3D projection matrix from homography matrix and camera parameters
            projection = projection_matrix(camera_parameters, homography)  
            meanS = np.array([int(model1.shape[1]/2),int(model1.shape[0]/2), 1])
            meanD = np.array([int(model2.shape[1]/2),int(model2.shape[0]), 1])
            meanS = np.dot(homography, meanS)
            meanD = np.dot(homography2, meanD)
            meanS/=meanS[2]
            meanD/=meanD[2]
            meanS = np.int32(meanS)
            meanD = np.int32(meanD)
            # meanD = [382, 233, 1]
            # dest = [int(1.1*meanD[0]-0.1*meanS[0]), int(1.1*meanD[1]-0.1*meanS[1])]
            dest = [int(meanD[0]), int(meanD[1])]
            # project cube or model
            oldFrame = copy.deepcopy(frame)
            count = 0
            reached = False
            while not reached:
                count+=1
                frame = copy.deepcopy(oldFrame)
                frame, reached = render(frame, obj, projection, model1, 0.01*(meanD-meanS), count, dest, False)
                showImage(frame, 1)
                # print(reached)
            print("Reached destination!!!")
            showImage(frame, 1000)
        except Exception as e:
            print(e)
            print('Error')
            pass
    else:
        if homography is None:
            print('Homo 1 None')
        if homography2 is None:
            print('Homo 2 None')

    cv2.destroyAllWindows()
    return 0

def render(img, obj, projection, model, mov, count, dest, reached, color=False):
    """
    Render a loaded obj model into the current video frame
    """
    vertices = obj.vertices
    scale_matrix = np.eye(3) * 30
    h, w = model.shape
    z = 0
    for face in obj.faces:
        face_vertices = face[0]
        points = np.array([vertices[vertex - 1] for vertex in face_vertices])
        points = np.dot(points, scale_matrix)
        # render model in the middle of the reference surface. To do so,
        # model points must be displaced
        points = np.array([[int(p[0] + w / 2), int(p[1] + h / 2), p[2]] for p in points])
        second_matrix = np.array([[1,0,0.1*count*mov[0]], [0,1,0.1*count*mov[1]], [0,0,1]])
        dst = cv2.perspectiveTransform(points.reshape(-1, 1, 3), np.dot(second_matrix, projection))
        imgpts = np.int32(dst)
        if dest in imgpts:
            z+=1
        if color is False:
            cv2.fillConvexPoly(img, imgpts, (137, 27, 211))
        else:
            color = hex_to_rgb(face[-1])
            color = color[::-1]  # reverse
            cv2.fillConvexPoly(img, imgpts, color)
    if z>10:
        reached = True
    return img, reached

def projection_matrix(camera_parameters, homography):
    """
    From the camera calibration matrix and the estimated homography
    compute the 3D projection matrix
    """
    # Compute rotation along the x and y axis as well as the translation
    homography = homography * (-1)
    rot_and_transl = np.dot(np.linalg.inv(camera_parameters), homography)
    col_1 = rot_and_transl[:, 0]
    col_2 = rot_and_transl[:, 1]
    col_3 = rot_and_transl[:, 2]
    # normalise vectors
    l = math.sqrt(np.linalg.norm(col_1, 2) * np.linalg.norm(col_2, 2))
    rot_1 = col_1 / l
    rot_2 = col_2 / l
    translation = col_3 / l
    # compute the orthonormal basis
    c = rot_1 + rot_2
    p = np.cross(rot_1, rot_2)
    d = np.cross(c, p)
    rot_1 = np.dot(c / np.linalg.norm(c, 2) + d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
    rot_2 = np.dot(c / np.linalg.norm(c, 2) - d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
    rot_3 = np.cross(rot_1, rot_2)
    # finally, compute the 3D projection matrix from the model to the current frame
    projection = np.stack((rot_1, rot_2, rot_3, translation)).T
    return np.dot(camera_parameters, projection)

def hex_to_rgb(hex_color):
    """
    Helper function to convert hex strings to RGB
    """
    hex_color = hex_color.lstrip('#')
    h_len = len(hex_color)
    return tuple(int(hex_color[i:i + h_len // 3], 16) for i in range(0, h_len, h_len // 3))

# Command line argument parsing
# NOT ALL OF THEM ARE SUPPORTED YET
parser = argparse.ArgumentParser(description='Augmented reality application')

parser.add_argument('-ma','--matches', help = 'draw matches between keypoints', action = 'store_true')

args = parser.parse_args()

if __name__ == '__main__':
    main()
