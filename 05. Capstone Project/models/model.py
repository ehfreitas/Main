from inflection import humanize
from sklearn.datasets import load_files       
from keras.utils import np_utils
import numpy as np
from glob import glob
import random
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from keras.preprocessing import image
from tqdm import tqdm
from PIL import ImageFile
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint 
from extract_bottleneck_features import *
import sys
from tensorflow.keras.models import load_model
from matplotlib import pyplot
import mtcnn

def load_dataset(path):
    '''
    Loads the train, test and validation datasets

            Parameters:
                    path (str): path where the files are located

            Returns:
                    None
    '''
    data = load_files(path)
    dog_files = np.array(data['filenames'])
    dog_targets = np_utils.to_categorical(np.array(data['target']), 133)
    return dog_files, dog_targets

# returns "True" if face is detected in image stored at img_path
def face_detector(img_path):
    '''
    Use OpenCV's implementation of Haar feature-based cascade classifiers to detect human faces in images.

            Parameters:
                    img_path (str): image file path

            Returns:
                    Bool (bool): True if a human face is detected otherwise false
    '''
    # extract pre-trained face detector
    face_cascade = cv2.CascadeClassifier('./models/haarcascade_frontalface_alt.xml')
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0

def face_detector2(img_path):
    '''
    Use a deep cascaded multi-task framework using different features of “sub-models”  to detect human faces in images.

            Parameters:
                    img_path (str): image file path

            Returns:
                    Bool (bool): True if a human face is detected otherwise false
    '''
    # load image from file
    pixels = pyplot.imread(img_path)
    # create the detector, using default weights
    detector = mtcnn.MTCNN()
    # detect faces in the image
    faces = detector.detect_faces(pixels)
    return len(faces) > 0

def path_to_tensor(img_path):    
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)

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

### TODO: Write a function that takes a path to an image as input
### and returns the dog breed that is predicted by the model.
def inceptionv3_predict_dogbreed(img_path, model):
    '''
    Returns the dog breed that is predicted by an CNN network model.

            Parameters:
                    img_path (str): image file path
                    model (keras.models.Sequential): CNN model to be used to detect dog breed

            Returns:
                    dog_names (str): Returns the name of the identified dog breed
    '''
    # load list of dog names
    dog_names = [item[20:-1] for item in sorted(glob("./data/dog_images/train/*/"))]
    # extract bottleneck features
    bottleneck_feature = extract_InceptionV3(path_to_tensor(img_path))
    # obtain predicted vector
    predicted_vector = model.predict(bottleneck_feature)
    # return dog breed that is predicted by the model
    return dog_names[np.argmax(predicted_vector)].split('.')[1].replace('_', ' ')

def detect_human_dog(img_path, model):
    '''
    Returns if the identified animal is a person or a dog and what is each breed (or what breeds it resembles if a person)
    If it is not returns an error.

            Parameters:
                    img_path (str): image file path
                    model (keras.models.Sequential): CNN model to be used to detect dog breed

            Returns:
                    str: Returns the name of the identified dog breed
    '''
    img = cv2.imread(img_path)
    #cv_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #plt.imshow(cv_rgb)
    #plt.show()    
    if dog_detector(img_path):        
        breed = inceptionv3_predict_dogbreed(img_path, model)        
        return 'Dog detected: This dogs looks like a {}'.format(breed)
    elif face_detector2(img_path):
        breed = inceptionv3_predict_dogbreed(img_path, model)        
        return 'Person detected: This guys looks like a {}'.format(breed)
    else:
        return 'Error: The image does not has a dog nor a person!'

def train_test_model():
    # load train, test, and validation datasets
    train_files, train_targets = load_dataset('./data/dog_images/train')
    valid_files, valid_targets = load_dataset('./data/dog_images/valid')
    test_files, test_targets = load_dataset('./data/dog_images/test')

    # load list of dog names
    dog_names = [item[20:-1] for item in sorted(glob("./data/dog_images/train/*/"))]

    # load filenames in shuffled human dataset
    human_files = np.array(glob("./data/lfw/*/*"))
    random.shuffle(human_files)

    ImageFile.LOAD_TRUNCATED_IMAGES = True                 

    # pre-process the data for Keras
    train_tensors = paths_to_tensor(train_files).astype('float32')/255
    valid_tensors = paths_to_tensor(valid_files).astype('float32')/255
    test_tensors = paths_to_tensor(test_files).astype('float32')/255

    ### TODO: Obtain bottleneck features from another pre-trained CNN.
    bottleneck_features = np.load('./models/DogInceptionV3Data.npz')
    train_inceptionv3 = bottleneck_features['train']
    valid_inceptionv3 = bottleneck_features['valid']
    test_inceptionv3 = bottleneck_features['test']

    ### TODO: Define your architecture.
    inceptionv3_model = Sequential()
    inceptionv3_model.add(GlobalAveragePooling2D(input_shape=test_inceptionv3.shape[1:]))
    inceptionv3_model.add(Dense(133, activation='softmax'))

    inceptionv3_model.summary()

    ### TODO: Compile the model.
    inceptionv3_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    ### TODO: Train the model.
    print('Saving model checkpoint as ./models/weights.best.inceptionv3.hdf5')
    checkpointer = ModelCheckpoint(filepath='./models/weights.best.inceptionv3.hdf5', 
                                verbose=1, save_best_only=True)

    inceptionv3_model.fit(train_inceptionv3, train_targets, 
                    validation_data=(valid_inceptionv3, valid_targets),
                    epochs=20, batch_size=20, callbacks=[checkpointer], verbose=1)

    ### TODO: Load the model weights with the best validation loss.
    inceptionv3_model.load_weights('./models/weights.best.inceptionv3.hdf5')

    # save the model to disk
    print('Saving model as ./models/inceptionv3_model.h5')
    inceptionv3_model.save(filepath='./models/inceptionv3_model.h5')

    ### TODO: Calculate classification accuracy on the test dataset.
    # get index of predicted dog breed for each image in test set
    inceptionv3_predictions = [np.argmax(inceptionv3_model.predict(np.expand_dims(feature, axis=0))) for feature in test_inceptionv3]

    # report test accuracy
    test_accuracy = 100*np.sum(np.array(inceptionv3_predictions)==np.argmax(test_targets, axis=1))/len(inceptionv3_predictions)
    print('Test accuracy: %.4f%%' % test_accuracy)

    return inceptionv3_model

def main():
    if len(sys.argv) == 2:
        if sys.argv[1] == 'train_test_model':
            print('Training & testing model...')
            inceptionv3_model = train_test_model()
        elif sys.argv[1] == 'load_trained_model':
            print('Loading trained model...')
            inceptionv3_model = load_model('./models/inceptionv3_model.h5')
    else:
        print('Please provide the argument "train_test_model" to train, test and save the neural network'\
              'or "load_trained_model" to load the saved neural network')
        return None
    print(detect_human_dog('./models/amstaff.jpg', inceptionv3_model))

if __name__ == '__main__':
    main()