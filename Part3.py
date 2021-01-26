import argparse

import cv2
import numpy as np
import math
import os
# from objloader_simple import *
# import objloader_simple
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
    descriptor = cv2.xfeatures2d.SURF_create()
    # descriptor = cv2.xfeatures2d.SIFT_create()
    (kps, features) = descriptor.detectAndCompute(image, None)
    
    # kps = np.float32([kp_.pt for kp_ in kps])
    return (kps, features)

def main():
    """
    This functions loads the target surface image,
    """
    homography = None 
    # matrix of camera parameters (made up but works quite well for me) 
    camera_parameters = np.array([[756.56499986, 0, 493.2992946 ], [ 0, 753.44416051, 304.00857278],[ 0, 0, 1]])
    # create ORB keypoint detector
    # orb = cv2.ORB_create()
    # create BFMatcher object based on hamming distance  
    # bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # load the reference surface that will be searched in the video stream
    dir_name = os.getcwd()
    model = cv2.imread(os.path.join(dir_name, 'reference/model42.png'), 0)
    # cv2.imshow('image',model)
    # Compute model keypoints and its descriptors
    kp_model, des_model = detectFeaturesKeys(model)
    # Load 3D model from OBJ file
    obj = OBJ(os.path.join(dir_name, 'models/fox.obj'), swapyz=True)  
    # init video capture
    cap = cv2.VideoCapture(0)

    while True:
        # read the current frame
        ret, frame = cap.read()
        if not ret:
            print("Unable to capture video")
            return 
        # find and draw the keypoints of the frame
        # cv2.imshow('video', frame)
        # cv2.imshow('im', model)
        newX,newY = frame.shape[1]*0.5, frame.shape[0]*0.5
        frame = cv2.resize(frame, (int(newX), int(newY)))
        # frame = cv2.
        kp_frame, des_frame = detectFeaturesKeys(frame)
        # match frame descriptors with model descriptors
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks=50)   # or pass empty dictionary

        flann = cv2.FlannBasedMatcher(index_params,search_params)

        matches = flann.knnMatch(des_model,des_frame,k=2)

        # Need to draw only good matches, so create a mask
        # matchesMask = [[0,0] for i in range(len(matches))]

        # ratio test as per Lowe's paper
        good = []
        for m,n in matches:
            if m.distance < 0.65*n.distance:
                good.append(m)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        # compute Homography if enough matches are found
        if len(good) > MIN_MATCHES:
            # differenciate between source points and destination points
            src_pts = np.float32([kp_model[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            # compute Homography
            homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            matchesMask = mask.ravel().tolist()
            print(homography)
            print(len(good))
            if args.rectangle:
                # Draw a rectangle that marks the found model in the frame
                h, w = model.shape
                pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
                # project corners into frame
                dst = cv2.perspectiveTransform(pts, homography)
                # connect them with lines  
                frame = cv2.polylines(frame, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)  
            # if a valid homography matrix was found render cube on model plane
            if homography is not None:
                try:
                    # obtain 3D projection matrix from homography matrix and camera parameters
                    projection = projection_matrix(camera_parameters, homography)  
                    # project cube or model
                    frame = render(frame, obj, projection, model, False)
                    #frame = render(frame, model, projection)
                except:
                    pass
            # draw first 10 matches.
            cv2.imshow('org', frame)
            if args.matches:
                # frame = cv2.drawMatches(model, kp_model, frame, kp_frame, matches[:10], 0, flags=2)
                # frame = cv2.resize(frame, (500, 500))   
                draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)

                frame = cv2.drawMatches(model,kp_model,frame,kp_frame,good,None,**draw_params)
                frame = cv2.resize(frame, (500, 500))  
# plt.imshow(img3, 'gray'),plt.show()     
            # show result
            cv2.imshow('frame', frame)
            # break
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        else:
            print("Not enough matches found - "+ str(len(matches)/MIN_MATCHES))

    cap.release()
    cv2.destroyAllWindows()
    return 0

def render(img, obj, projection, model, color=False):
    """
    Render a loaded obj model into the current video frame
    """
    vertices = obj.vertices
    scale_matrix = np.eye(3) * 3
    h, w = model.shape

    for face in obj.faces:
        face_vertices = face[0]
        points = np.array([vertices[vertex - 1] for vertex in face_vertices])
        points = np.dot(points, scale_matrix)
        # render model in the middle of the reference surface. To do so,
        # model points must be displaced
        points = np.array([[p[0] + w / 2, p[1] + h / 2, p[2]] for p in points])
        # print(points)
        dst = cv2.perspectiveTransform(points.reshape(-1, 1, 3), projection)
        # print(dst)
        imgpts = np.int32(dst)
        if color is False:
            cv2.fillConvexPoly(img, imgpts, (137, 27, 211))
        else:
            color = hex_to_rgb(face[-1])
            color = color[::-1]  # reverse
            cv2.fillConvexPoly(img, imgpts, color)

    return img

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
    print(projection)
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

parser.add_argument('-r','--rectangle', help = 'draw rectangle delimiting target surface on frame', action = 'store_true')
parser.add_argument('-mk','--model_keypoints', help = 'draw model keypoints', action = 'store_true')
parser.add_argument('-fk','--frame_keypoints', help = 'draw frame keypoints', action = 'store_true')
parser.add_argument('-ma','--matches', help = 'draw matches between keypoints', action = 'store_true')
# TODO jgallostraa -> add support for model specification
#parser.add_argument('-mo','--model', help = 'Specify model to be projected', action = 'store_true')

args = parser.parse_args()

if __name__ == '__main__':
    main()
