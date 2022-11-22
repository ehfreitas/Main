from glob import glob
from keras.preprocessing import image
import numpy as np
import cv2
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
import mtcnn
from matplotlib import pyplot

def extract_InceptionV3(tensor):
	from keras.applications.inception_v3 import InceptionV3, preprocess_input
	return InceptionV3(weights='imagenet', include_top=False).predict(preprocess_input(tensor))

def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def inceptionv3_predict_dogbreed(img_path, model):
    # load list of dog names
    dog_names = [item[20:-1] for item in sorted(glob("../data/dog_images/train/*/"))]
    # extract bottleneck features
    bottleneck_feature = extract_InceptionV3(path_to_tensor(img_path))
    # obtain predicted vector
    predicted_vector = model.predict(bottleneck_feature)
    # return dog breed that is predicted by the model
    return dog_names[np.argmax(predicted_vector)].split('.')[1].replace('_', ' ')

def face_detector(img_path):
    # extract pre-trained face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +'./models/haarcascade_frontalface_alt.xml')
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0

def face_detector2(img_path):
    # load image from file
    pixels = pyplot.imread(img_path)
    # create the detector, using default weights
    detector = mtcnn.MTCNN()
    # detect faces in the image
    faces = detector.detect_faces(pixels)
    return len(faces) > 0

def ResNet50_predict_labels(img_path):
    # define ResNet50 model
    ResNet50_model = ResNet50(weights='imagenet')
    # returns prediction vector for image located at img_path
    img = preprocess_input(path_to_tensor(img_path))
    return np.argmax(ResNet50_model.predict(img))

### returns "True" if a dog is detected in the image stored at img_path
def dog_detector(img_path):
    prediction = ResNet50_predict_labels(img_path)
    return ((prediction <= 268) & (prediction >= 151))

def detect_human_dog(img_path, model):
    img = cv2.imread(img_path)
    if dog_detector(img_path):        
        breed = inceptionv3_predict_dogbreed(img_path, model)        
        return 'Dog detected: This dog looks like a {}'.format(breed)
    elif face_detector2(img_path):
        breed = inceptionv3_predict_dogbreed(img_path, model)        
        return 'Person detected: This person looks like a {}'.format(breed)
    else:
        return 'Error: The image does not have a dog nor a person!'