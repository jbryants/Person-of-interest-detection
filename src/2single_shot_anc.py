from os import listdir
from os.path import isdir
from PIL import Image
from matplotlib import pyplot
from numpy import savez_compressed
from numpy import asarray
import numpy as np
from mtcnn.mtcnn import MTCNN
from keras.models import load_model
import pandas as pd

# extract a single face from a given photograph
def extract_face(image, required_size=(160, 160)):
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
            print(result)
            # extract the bounding box from the first face
            x1, y1, width, height = result['box']
            # bug fix
            x1, y1 = abs(x1), abs(y1)
            x2, y2 = x1 + width, y1 + height
            # extract the face
            face = pixels[y1:y2, x1:x2]
            pyplot.imshow(face)
            pyplot.show()
            # resize pixels to the model size
            image = Image.fromarray(face)
            image = image.resize(required_size)
            face_array = asarray(image)
            return face_array

# get the face embedding for one face
def get_embedding(model, face_pixels):
        # scale pixel values
        face_pixels = face_pixels.astype('float32')
        # standardize pixel values across channels (global)
        mean, std = face_pixels.mean(), face_pixels.std()
        face_pixels = (face_pixels - mean) / std
        # transform face into one sample
        samples = np.expand_dims(face_pixels, axis=0)
        # make prediction to get embedding
        yhat = model.predict(samples)
        return yhat[0]


def euclidean_distance(anchor_emb, row_emb):
    dist = np.sqrt(np.sum(np.square(np.subtract(anchor_emb, row_emb))))
    if dist < 10.1:
        #print('  %1.4f  ' % dist, end='')
        #print("\n")
        return True

anc = str(input("Enter anchor name: "))

# load anchor image
img = Image.open('/home/jbryants/project/0facenet/V/anchor_ss/%s.jpg'%(anc))

# extract anchor face
face = extract_face(img)
print(face.shape)

# load the facenet model
model = load_model('facenet_keras.h5')
print('Loaded Model')

# Creating the embedding for the anchor face image
anc_embedding = get_embedding(model, face)
print(anc_embedding.shape)

# load the faces embeddings array from the .npz files.
items = listdir("/home/jbryants/project/0facenet/V/testVidArrays-Embeddings/.")

npzList = []
for item_name in items:
    if item_name.endswith(".npz"):
        npzList.append(item_name)

print(npzList)

# Creating a dictionary to store the faces embeddings array
arrDict = {}
for npz in npzList:
    data = np.load('/home/jbryants/project/0facenet/V/testVidArrays-Embeddings/%s'%npz)
    arrDict[npz] = data['arr_0']

# Selecting the matching faces with the anchor based on euclidean distance measure.
sel = []
for k, v in arrDict.items():
    flag = False
    for row in v:
        flag = euclidean_distance(anc_embedding, row)
        if flag == True:
            sel.append(k)

# Reading the csv file where 
# the NpzFile test data file name to corresponding video file name 
# mapping is present.
data = pd.read_csv('fname_dict.csv')

# Showing the list of all videos in which we are looking for the person of interest.
print("The list of videos we have passed in for checking are:- ")
print(data['f_name'])

# The videos in which the person of interest was found.
print("\n\nThe videos in which %s is present are:- "%(anc))
for row in range(data.shape[0]):
    for col in range(data.shape[1]):
        if data.iloc[row][col] in sel:
            print(data.iloc[row][(col + 1)])
            
