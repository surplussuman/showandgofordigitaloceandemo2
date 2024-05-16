import os
from face_recognition.face_recognition_cli import image_files_in_folder
import cv2
import face_recognition
import numpy as  np
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC



def train():

    training_dir = 'face_recognition_data/training_dataset'

    count = 0
    for person_name in os.listdir(training_dir):
        curr_directory = os.path.join(training_dir, person_name)
        if not os.path.isdir(curr_directory):
            continue
        for imagefile in image_files_in_folder(curr_directory):
            count += 1

    X = []
    y = []
    i = 0

    for person_name in os.listdir(training_dir):
        print(str(person_name))
        curr_directory = os.path.join(training_dir, person_name)
        if not os.path.isdir(curr_directory):
            continue
        for imagefile in image_files_in_folder(curr_directory):
            print(str(imagefile))
            image = cv2.imread(imagefile)
            try:
                X.append((face_recognition.face_encodings(image)[0]).tolist())

                y.append(person_name)
                i += 1
            except:
                print("removed")
                os.remove(imagefile)

    targets = np.array(y)
    encoder = LabelEncoder()
    encoder.fit(y)
    y = encoder.transform(y)
    X1 = np.array(X)
    print("shape: " + str(X1.shape))
    np.save('face_recognition_data/classes.npy', encoder.classes_)
    svc = SVC(kernel='linear', probability=True)
    svc.fit(X1, y)

    svc.save("trained_model.yml")


train()