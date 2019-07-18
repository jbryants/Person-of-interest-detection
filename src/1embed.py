# calculate a face embedding for each face in the dataset using facenet
import os
from numpy import load
from numpy import expand_dims
from numpy import asarray
from numpy import savez_compressed
from keras.models import load_model

# get the face embedding for one face
def get_embedding(model, face_pixels):
	# scale pixel values
	face_pixels = face_pixels.astype('float32')
	# standardize pixel values across channels (global)
	mean, std = face_pixels.mean(), face_pixels.std()
	face_pixels = (face_pixels - mean) / std
	# transform face into one sample
	samples = expand_dims(face_pixels, axis=0)
	# make prediction to get embedding
	yhat = model.predict(samples)
	return yhat[0]

# load the facenet model
model = load_model('facenet_keras.h5')
print('Loaded Model')

# load the faces array from the .npz files.
items = os.listdir("/home/jbryants/project/0facenet/V/testVidArrays/.")

npzList = []
for item_name in items:
    if item_name.endswith(".npz"):
        npzList.append(item_name)

print(npzList)

# Creating a dictionary to store the faces array
arrDict = {}

for npz in npzList:
    data = load('/home/jbryants/project/0facenet/V/testVidArrays/%s'%npz)
    arrDict[npz] = data['arr_0']

# convert each face in the train set to an embedding
embeddedTestDict = dict()
for k, v in arrDict.items():
    embeddedTestList = list()
    print(k)
    print(v.shape)
    for row in v:
        embedding = get_embedding(model, row)
        embeddedTestList.append(embedding)
    embeddedTestArr = asarray(embeddedTestList)
    print(embeddedTestArr.shape)
    embeddedTestDict[k] = embeddedTestArr

print(embeddedTestDict)

# save arrays to files in compressed format
for k, v in embeddedTestDict.items():
    savez_compressed('/home/jbryants/project/0facenet/V/testVidArrays-Embeddings/%s-embeddings.npz'%k, v)
