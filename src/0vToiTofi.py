# face detection from all the videos from a given directory
from os import listdir
from os.path import isdir
from os import mkdir
from PIL import Image
from matplotlib import pyplot
from numpy import savez_compressed
from numpy import asarray
from mtcnn.mtcnn import MTCNN
import cv2
import numpy as np
import csv

# extract a single face from a given photograph
def extract_face(cnt, image, required_size=(160, 160)):
    # convert to RGB, if needed
    image = image.convert('RGB')
    # convert to array
    pixels = asarray(image)
    # create the detector, using default weights
    detector = MTCNN()
    # detect faces in the image
    results = detector.detect_faces(pixels)
    # if no faces were detected.
    if not results:
        pass
    # if one or more faces were detected.
    elif len(results) >= 1:
        for result in results:
            if result['confidence'] >= 0.95:
                # extract the bounding box from the first face
                x1, y1, width, height = result['box']
                # bug fix
                x1, y1 = abs(x1), abs(y1)
                x2, y2 = x1 + width, y1 + height
                # extract the face
                face = pixels[y1:y2, x1:x2]
                # resize pixels to the model size
                image = Image.fromarray(face)
                image = image.resize(required_size)
                face_array = asarray(image)
                cv2.imwrite("/home/jbryants/project/0facenet/V/frames/faces/testVid%d/face%d.jpg"%(vidCnt, cnt), face_array) 
                yield face_array


# Function to extract frames 
def FrameCapture(path): 
      
    # Path to video file 
    vidObj = cv2.VideoCapture(path) 
     
    # reducing frames per sec
    sec = 100

    # Used as counter variable 
    count = 0
  
    # checks whether frames were extracted 
    success = 1
    
    faces = list()

    # making appropriate directory to store the faces detected for testing purposes
    try:
        mkdir('/home/jbryants/project/0facenet/V/frames/faces/testVid%d'%(vidCnt))
    except:
        print("An error occured while trying to create directory.")

    while success:
  
        # vidObj object calls read 
        # function extract frames 
        success, image = vidObj.read() 
 
        # Saves the frames with frame-count 
        try:
            img = Image.fromarray(image)

            # get face
            faces_gen = extract_face(count, img)
        
            # store
            for face in faces_gen:
                faces.append(face)
        except:
            pass
        print(count)
        count += 1

    return asarray(faces)


# Driver Code
if __name__ == '__main__':
    items = listdir("/home/jbryants/project/0facenet/videos/team_mates/.")
    print(items)

    vidList = []
    for item_name in items:
        if item_name.endswith(".mp4"):
            vidList.append(item_name)

    print(vidList)

    global vidCnt
    vidCnt = 0
    
    fname_dict = {}
    
    for vidName in vidList:
        # Calling the function
        testVidArray = FrameCapture("/home/jbryants/project/0facenet/videos/team_mates/%s"%vidName)
        # save arrays to one file in compressed format
        savez_compressed('/home/jbryants/project/0facenet/V/testVidArrays/testVidArray%d.npz'%vidCnt, testVidArray)
        fname_dict['testVidArray%d.npz-embeddings.npz'%(vidCnt)] = vidName
        vidCnt += 1

    w = csv.writer(open("fname_dict.csv", "w"))
    for key, val in fname_dict.items():
        w.writerow([key, val])
