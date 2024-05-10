from django.shortcuts import render, redirect
from .forms import usernameForm, DateForm, UsernameAndDateForm, DateForm_2
from django.contrib import messages
from django.contrib.auth.models import User
import cv2
#import dlib
import imutils
from imutils import face_utils
from imutils.video import VideoStream
from imutils.face_utils import rect_to_bb
from imutils.face_utils import FaceAligner
import time
from attendance_system_facial_recognition.settings import BASE_DIR
import os
import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import numpy as np
from django.contrib.auth.decorators import login_required
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import datetime
from django_pandas.io import read_frame
from users.models import Present, Time
import seaborn as sns
import pandas as pd
from django.db.models import Count
#import mpld3
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
from matplotlib import rcParams
import math
from twilio.rest import Client
from django.http import HttpResponse
import threading
from facenet_pytorch import MTCNN
import queue

mpl.use('Agg')


# utility functions:
def username_present(username):
    if User.objects.filter(username=username).exists():
        return True

    return False


'''def create_dataset(username):
    id = username
    if(os.path.exists('face_recognition_data/training_dataset/{}/'.format(id)) == False):
        os.makedirs('face_recognition_data/training_dataset/{}/'.format(id))
    directory = 'face_recognition_data/training_dataset/{}/'.format(id)

    # Detecting face
    # Loading the HOG face detector and the shape predictpr for allignment

    print("[INFO] Loading the facial detector")
    detector = dlib.get_frontal_face_detector()
    # Adding our shape predictor to the path
    predictor = dlib.shape_predictor(
        'face_recognition_data/shape_predictor_68_face_landmarks.dat')
    fa = FaceAligner(predictor, desiredFaceWidth=96)
    # Capturing images from the webcam and detect the face
    # Initialize the video stream
    print("[INFO] Initializing Video stream")
    vs = VideoStream(src=1).start()
    #time.sleep(2.0)

    # Our identifier
    # We will put the id here and we will store the id with a face, so that later we can identify whose face it is

    # Our dataset naming counter
    sampleNum = 0
    # Capturing the faces one by one and detect the faces and showing it on the window
    while(True):
        # Capturing the image
        frame = vs.read()
        # Resize each image
        frame = imutils.resize(frame, width=800)
        # COLOR to convert GRAY
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # To store the faces
        # This will detect all the images in the current frame, and it will return the coordinates of the faces
        # Takes in image and some other parameter for accurate result
        faces = detector(gray_frame, 0)
        # In above 'faces' variable there can be multiple faces so we have to get each and every face and draw a rectangle around it.

        for face in faces:
            print("inside for loop")
            (x, y, w, h) = face_utils.rect_to_bb(face)

            face_aligned = fa.align(frame, gray_frame, face)
            # Whenever the program captures the face, we will write that is a folder
            # Before capturing the face, we need to tell the script whose face it is
            # For that we will need an identifier, here we call it id
            # So now we captured a face, we need to write it in a file
            sampleNum = sampleNum+1
            # Saving the image dataset, but only the face part, cropping the rest

            if face is None:
                print("face is none")
                continue

            cv2.imwrite(directory+'/'+str(sampleNum)+'.jpg', face_aligned)
            face_aligned = imutils.resize(face_aligned, width=400)

            # @params the initial point of the rectangle will be x,y and
            # @params end point will be x+width and y+height
            # @params along with color of the rectangle
      # thickness of the rectangle
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 1)
            # Before continuing to the next loop, I want to give it a little pause
            # waitKey of 50 millisecond
            cv2.waitKey(1)

        # Showing the image in another window
        # Creates a window with window name "Face" and with the image img
        cv2.imshow("Add Images", frame)
        # Before closing it we need to give a wait command, otherwise the OpenCV won't work
        # @params with the millisecond of delay 1
        cv2.waitKey(1)
        # To get out of the loop
        if(sampleNum > 300):
            break

    # Stoping the videostream
    vs.stop()
    # destroying all the windows
    cv2.destroyAllWindows()'''

def create_dataset(username):
    id = username
    if not os.path.exists('face_recognition_data/training_dataset/{}/'.format(id)):
        os.makedirs('face_recognition_data/training_dataset/{}/'.format(id))
    directory = 'face_recognition_data/training_dataset/{}/'.format(id)

    print("[INFO] Initializing Video stream")
    vs = VideoStream(src=0).start()

    mtcnn = MTCNN()
    sampleNum = 0
    while True:
        frame = vs.read()
        frame = imutils.resize(frame, width=800)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect faces using MTCNN
        boxes, probs, landmarks = mtcnn.detect(rgb_frame, landmarks=True)
        if boxes is not None:
            percentage = 0.8
            for box, landmark in zip(boxes, landmarks):
                x, y, w, h = map(int, box)

                #face_aligned = frame[y:y+h, x:x+w]
                
                # Check if the landmarks array has the expected shape
                if len(landmark) != 5:
                    continue  # Skip this iteration if landmarks are not detected properly

                # Calculate the bounding box coordinates based on the landmarks
                min_x = int(min(landmark[:, 0]))
                max_x = int(max(landmark[:, 0]))
                min_y = int(min(landmark[:, 1]))
                max_y = int(max(landmark[:, 1]))

                # Calculate the width and height of the extended bounding box
                width = max_x - min_x
                height = max_y - min_y
                extend_width = int(percentage * width)
                extend_height = int(percentage * height)

                # Extend the bounding box coordinates
                min_x -= extend_width
                max_x += extend_width
                min_y -= extend_height
                max_y += extend_height

                # Ensure the extended bounding box is within the frame boundaries
                min_x = max(0, min_x)
                max_x = min(frame.shape[1], max_x)
                min_y = max(0, min_y)
                max_y = min(frame.shape[0], max_y)

                face_aligned = frame[min_y:max_y, min_x:max_x]

                sampleNum = sampleNum+1

                cv2.imwrite(directory + '/' + str(sampleNum) + '.jpg', face_aligned)
                face_aligned = imutils.resize(face_aligned, width=400)

                print(f'Image {sampleNum} saved!!!')

                # Draw the rectangle
                cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), (0, 255, 0), 2)
                for (x, y) in landmark.astype(int):
                    cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)

        cv2.imshow("Add Images", frame)
        cv2.waitKey(1)

        if sampleNum > 300:
            break

    vs.stop()
    cv2.destroyAllWindows()


def predict(face_aligned, svc, threshold=0.7):
    face_encodings = np.zeros((1, 128))
    try:
        x_face_locations = face_recognition.face_locations(face_aligned)
        faces_encodings = face_recognition.face_encodings(
            face_aligned, known_face_locations=x_face_locations)
        if(len(faces_encodings) == 0):
            return ([-1], [0])

    except:

        return ([-1], [0])

    prob = svc.predict_proba(faces_encodings)
    result = np.where(prob[0] == np.amax(prob[0]))
    if(prob[0][result[0]] <= threshold):
        return ([-1], prob[0][result[0]])

    return (result[0], prob[0][result[0]])

'''def predict(face, svc, threshold=0.7):
    # Resize face to a fixed size if necessary
    face = cv2.resize(face, (100, 100))
    # Flatten face to a 1D array
    face = face.flatten().reshape(1, -1)

    # Predict using SVC model
    pred = svc.predict(face)
    prob = svc.predict_proba(face)

    if prob.max() >= threshold:
        return pred[0], prob.max()
    else:
        return -1, 0.0
'''
'''def predict(face_roi, svc, threshold=0.7):
    try:
        # Detect faces and extract encodings
        boxes, probs, landmarks = mtcnn.detect(face_roi, landmarks=True)
        if boxes is not None:
            face_encodings = []
            for box in boxes:
                x, y, w, h = box.astype(int)
                face = face_roi[y:y+h, x:x+w]
                if face.size == 0:
                    continue  # Skip empty faces
                face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                face = cv2.resize(face, (100, 100))  # Resize face to a fixed size if necessary
                face = face.flatten()  # Flatten face to a 1D array
                face_encodings.append(face)

            if not face_encodings:
                return -1, 0.0

            face_encodings = [encoding[:128] for encoding in face_encodings]

            # Predict using SVC model
            prob = svc.predict_proba(face_encodings)
            if len(prob.shape) < 2:
                return -1, 0.0  # No valid predictions
            result = np.argmax(prob, axis=1)
            confidence = np.max(prob, axis=1)

            if confidence >= threshold:
                return result[0], confidence[0]
            else:
                return -1, 0.0


    except Exception as e:
        print(f"Error predicting face: {e}")
        return -1, 0.0
    return -1, 0.0  # Return default values if no faces are detected'''


def vizualize_Data(embedded, targets,):

    X_embedded = TSNE(n_components=2).fit_transform(embedded)

    for i, t in enumerate(set(targets)):
        idx = targets == t
        plt.scatter(X_embedded[idx, 0], X_embedded[idx, 1], label=t)

    plt.legend(bbox_to_anchor=(1, 1))
    rcParams.update({'figure.autolayout': True})
    plt.tight_layout()
    plt.savefig(
        './recognition/static/recognition/img/training_visualisation.png')
    plt.close()


def update_attendance_in_db_in(present):
    today = datetime.date.today()
    time = datetime.datetime.now()
    for person in present:
        user = User.objects.get(username=person)
        try:
            qs = Present.objects.get(user=user, date=today)
        except:
            qs = None

        if qs is None:
            if present[person] == True:
                a = Present(user=user, date=today, present=True)
                a.save()
            else:
                a = Present(user=user, date=today, present=False)
                a.save()
        else:
            if present[person] == True:
                qs.present = True
                qs.save(update_fields=['present'])
        if present[person] == True:
            a = Time(user=user, date=today, time=time, out=False)
            a.save()


def update_attendance_in_db_out(present):
    today = datetime.date.today()
    time = datetime.datetime.now()
    for person in present:
        user = User.objects.get(username=person)
        if present[person] == True:
            a = Time(user=user, date=today, time=time, out=True)
            a.save()


def check_validity_times(times_all):

    if(len(times_all) > 0):
        sign = times_all.first().out
    else:
        sign = True
    times_in = times_all.filter(out=False)
    times_out = times_all.filter(out=True)
    if(len(times_in) != len(times_out)):
        sign = True
    break_hourss = 0
    if(sign == True):
        check = False
        break_hourss = 0
        return (check, break_hourss)
    prev = True
    prev_time = times_all.first().time

    for obj in times_all:
        curr = obj.out
        if(curr == prev):
            check = False
            break_hourss = 0
            return (check, break_hourss)
        if(curr == False):
            curr_time = obj.time
            to = curr_time
            ti = prev_time
            break_time = ((to-ti).total_seconds())/3600
            break_hourss += break_time

        else:
            prev_time = obj.time

        prev = curr

    return (True, break_hourss)


def convert_hours_to_hours_mins(hours):

    h = int(hours)
    hours -= h
    m = hours*60
    m = math.ceil(m)
    return str(str(h) + " hrs " + str(m) + "  mins")


def hours_vs_date_given_employee(present_qs, time_qs, admin=True):
    register_matplotlib_converters()
    df_hours = []
    df_break_hours = []
    qs = present_qs

    for obj in qs:
        date = obj.date
        times_in = time_qs.filter(date=date).filter(out=False).order_by('time')
        times_out = time_qs.filter(date=date).filter(out=True).order_by('time')
        times_all = time_qs.filter(date=date).order_by('time')
        obj.time_in = None
        obj.time_out = None
        obj.hours = 0
        obj.break_hours = 0
        if (len(times_in) > 0):
            obj.time_in = times_in.first().time

        if (len(times_out) > 0):
            obj.time_out = times_out.last().time

        if(obj.time_in is not None and obj.time_out is not None):
            ti = obj.time_in
            to = obj.time_out
            hours = ((to-ti).total_seconds())/3600
            obj.hours = hours

        else:
            obj.hours = 0

        (check, break_hourss) = check_validity_times(times_all)
        if check:
            obj.break_hours = break_hourss

        else:
            obj.break_hours = 0

        df_hours.append(obj.hours)
        df_break_hours.append(obj.break_hours)
        obj.hours = convert_hours_to_hours_mins(obj.hours)
        obj.break_hours = convert_hours_to_hours_mins(obj.break_hours)

    df = read_frame(qs)

    df["hours"] = df_hours
    df["break_hours"] = df_break_hours

    print(df)

    sns.barplot(data=df, x='date', y='hours')
    plt.xticks(rotation='vertical')
    rcParams.update({'figure.autolayout': True})
    plt.tight_layout()
    if(admin):
        plt.savefig(
            './recognition/static/recognition/img/attendance_graphs/hours_vs_date/1.png')
        plt.close()
    else:
        plt.savefig(
            './recognition/static/recognition/img/attendance_graphs/employee_login/1.png')
        plt.close()
    return qs


def hours_vs_employee_given_date(present_qs, time_qs):
    register_matplotlib_converters()
    df_hours = []
    df_break_hours = []
    df_username = []
    qs = present_qs

    for obj in qs:
        user = obj.user
        times_in = time_qs.filter(user=user).filter(out=False)
        times_out = time_qs.filter(user=user).filter(out=True)
        times_all = time_qs.filter(user=user)
        obj.time_in = None
        obj.time_out = None
        obj.hours = 0
        obj.hours = 0
        if (len(times_in) > 0):
            obj.time_in = times_in.first().time
        if (len(times_out) > 0):
            obj.time_out = times_out.last().time
        if(obj.time_in is not None and obj.time_out is not None):
            ti = obj.time_in
            to = obj.time_out
            hours = ((to-ti).total_seconds())/3600
            obj.hours = hours
        else:
            obj.hours = 0
        (check, break_hourss) = check_validity_times(times_all)
        if check:
            obj.break_hours = break_hourss

        else:
            obj.break_hours = 0

        df_hours.append(obj.hours)
        df_username.append(user.username)
        df_break_hours.append(obj.break_hours)
        obj.hours = convert_hours_to_hours_mins(obj.hours)
        obj.break_hours = convert_hours_to_hours_mins(obj.break_hours)

    df = read_frame(qs)
    df['hours'] = df_hours
    df['username'] = df_username
    df["break_hours"] = df_break_hours

    sns.barplot(data=df, x='username', y='hours')
    plt.xticks(rotation='vertical')
    rcParams.update({'figure.autolayout': True})
    plt.tight_layout()
    plt.savefig(
        './recognition/static/recognition/img/attendance_graphs/hours_vs_employee/1.png')
    plt.close()
    return qs


def total_number_employees():
    qs = User.objects.all()
    return (len(qs) - 1)
    # -1 to account for Admin Panel


#to show how many present today
def employees_present_today():
    today = datetime.date.today()
    qs = Present.objects.filter(date=today).filter(present=True)
    return len(qs)


def this_week_emp_count_vs_date():
    today = datetime.date.today()
    some_day_last_week = today-datetime.timedelta(days=7)
    monday_of_last_week = some_day_last_week - \
        datetime.timedelta(days=(some_day_last_week.isocalendar()[2] - 1))
    monday_of_this_week = monday_of_last_week + datetime.timedelta(days=7)
    qs = Present.objects.filter(
        date__gte=monday_of_this_week).filter(date__lte=today)
    str_dates = []
    emp_count = []
    str_dates_all = []
    emp_cnt_all = []
    cnt = 0

    for obj in qs:
        date = obj.date
        str_dates.append(str(date))
        qs = Present.objects.filter(date=date).filter(present=True)
        emp_count.append(len(qs))

    while(cnt < 5):

        date = str(monday_of_this_week+datetime.timedelta(days=cnt))
        cnt += 1
        str_dates_all.append(date)
        if(str_dates.count(date)) > 0:
            idx = str_dates.index(date)

            emp_cnt_all.append(emp_count[idx])
        else:
            emp_cnt_all.append(0)

    df = pd.DataFrame()
    df["date"] = str_dates_all
    df["Number of employees"] = emp_cnt_all

    sns.lineplot(data=df, x='date', y='Number of employees')
    plt.savefig(
        './recognition/static/recognition/img/attendance_graphs/this_week/1.png')
    plt.close()


# used
def last_week_emp_count_vs_date():
    today = datetime.date.today()
    some_day_last_week = today-datetime.timedelta(days=7)
    monday_of_last_week = some_day_last_week - \
        datetime.timedelta(days=(some_day_last_week.isocalendar()[2] - 1))
    monday_of_this_week = monday_of_last_week + datetime.timedelta(days=7)
    qs = Present.objects.filter(date__gte=monday_of_last_week).filter(
        date__lt=monday_of_this_week)
    str_dates = []
    emp_count = []

    str_dates_all = []
    emp_cnt_all = []
    cnt = 0

    for obj in qs:
        date = obj.date
        str_dates.append(str(date))
        qs = Present.objects.filter(date=date).filter(present=True)
        emp_count.append(len(qs))

    while(cnt < 5):

        date = str(monday_of_last_week+datetime.timedelta(days=cnt))
        cnt += 1
        str_dates_all.append(date)
        if(str_dates.count(date)) > 0:
            idx = str_dates.index(date)

            emp_cnt_all.append(emp_count[idx])

        else:
            emp_cnt_all.append(0)

    df = pd.DataFrame()
    df["date"] = str_dates_all
    df["emp_count"] = emp_cnt_all

    sns.lineplot(data=df, x='date', y='emp_count')
    plt.savefig(
        './recognition/static/recognition/img/attendance_graphs/last_week/1.png')
    plt.close()

import time

def days_used(start_time_str):
    start_time = time.mktime(time.strptime(start_time_str, "%Y-%m-%d"))  # Convert start time string to timestamp
    current_time = time.time()
    num_seconds = current_time - start_time
    num_days = num_seconds / (60 * 60 * 24)  # Convert seconds to days
    return int(num_days)


# Creating api here
# To display for Users
def home(request):

    return render(request, 'recognition/home.html')


# For login page
@login_required
def dashboard(request):
    if(request.user.username == 'admin'):
        print("admin")
        total_num_of_emp = total_number_employees()
        emp_present_today = employees_present_today()
        emp_absent_today =  total_num_of_emp - emp_present_today
        present_percent = (  emp_present_today / total_num_of_emp) * 100
        absent_percent = (emp_absent_today / total_num_of_emp) * 100
        
        start_time = '2024-04-01'
        num_days_used = days_used(start_time)

        return render(request, 'recognition/admin_dashboard.html',{'total_num_of_emp': total_num_of_emp, 'emp_present_today': emp_present_today, 'emp_absent_today':emp_absent_today,'emp_present_percent':present_percent,'emp_absent_percent':absent_percent,'days_used':num_days_used})
    else:
        print("not admin")

        return render(request, 'recognition/employee_dashboard.html')


# Inside the login page of Admin
@login_required
def add_photos(request):
    if request.user.username != 'admin':
        return redirect('not-authorised')
    if request.method == 'POST':
        form = usernameForm(request.POST)
        data = request.POST.copy()
        username = data.get('username')
        if username_present(username):
            create_dataset(username)
            messages.success(request, f'Dataset Created Sucessfully')
            return redirect('add-photos')
        else:
            messages.warning(
                request, f'No such username found. Please register employee first.')
            return redirect('dashboard')

    else:

        form = usernameForm()
        return render(request, 'recognition/add_photos.html', {'form': form})



def send_message_in(person_name):
    #Twilio
    SID = 'ACe2993c6d9afd11330e6ba71736696cac'
    AUTH_TOKEN = 'ce0ae7d5f369d1e97d2139561abfe9ce'
    client = Client(SID, AUTH_TOKEN)

    mobile_numbers = {
        'Suman' : '+916383595092',
        'Akii' : '+918125356941',
        'RahulPandey' : '+916206419351',
        'ShivamShivam' : '+918544859338',
        'KunalKishor' : '+919006612353'
    }
    #message_sent = {person_name: False for person_name in mobile_numbers.keys()}

    current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    message = f'Welcome!!! {person_name} your attendance marked at {current_time}'
    client = Client(SID, AUTH_TOKEN)
    client.messages.create(
        body=message,
        from_='+14052618747',
        to=mobile_numbers.get(person_name, "")
    )

def send_message_out(person_name):
    #Twilio
    SID = 'ACe2993c6d9afd11330e6ba71736696cac'
    AUTH_TOKEN = 'ce0ae7d5f369d1e97d2139561abfe9ce'
    client = Client(SID, AUTH_TOKEN)

    mobile_numbers = {
        'Suman' : '+916383595092',
        'Akii' : '+918125356941',
        'RahulPandey' : '+916206419351',
        'ShivamShivam' : '+918544859338',
        'KunalKishor' : '+919006612353'
    }
    #message_sent = {person_name: False for person_name in mobile_numbers.keys()}

    current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    message = f'Thank you {person_name} you went out at {current_time}. See you later'
    client = Client(SID, AUTH_TOKEN)
    client.messages.create(
        body=message,
        from_='+14052618747',
        to=mobile_numbers.get(person_name, "")
    )


'''def mark_your_attendance(request):
    detector = dlib.get_frontal_face_detector()

    # Adding path of the shape predictor to the variable
    predictor = dlib.shape_predictor(
        'face_recognition_data/shape_predictor_68_face_landmarks.dat')
    svc_save_path = "face_recognition_data/svc.sav"

    with open(svc_save_path, 'rb') as f:
        svc = pickle.load(f)
    fa = FaceAligner(predictor, desiredFaceWidth=96)
    encoder = LabelEncoder()
    encoder.classes_ = np.load('face_recognition_data/classes.npy')

    faces_encodings = np.zeros((1, 128))
    no_of_faces = len(svc.predict_proba(faces_encodings)[0])
    count = dict()
    present = dict()
    log_time = dict()
    start = dict()
    for i in range(no_of_faces):
        count[encoder.inverse_transform([i])[0]] = 0
        present[encoder.inverse_transform([i])[0]] = False

    vs = VideoStream(src=0).start()

    sampleNum = 0

    #message_sent = {person_name: False for person_name in mobile_numbers.keys()}

    message_sent = {}

    while(True):

        frame = vs.read()

        frame = imutils.resize(frame, width=800)

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = detector(gray_frame, 0)

        for face in faces:
            print("INFO : inside for loop")
            (x, y, w, h) = face_utils.rect_to_bb(face)

            face_aligned = fa.align(frame, gray_frame, face)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 1)

            (pred, prob) = predict(face_aligned, svc)

            if(pred != [-1]):

                person_name = encoder.inverse_transform(np.ravel([pred]))[0]
                pred = person_name
                if count[pred] == 0:
                    start[pred] = time.time()
                    count[pred] = count.get(pred, 0) + 1

                if count[pred] == 4 and (time.time()-start[pred]) > 1.2:
                    count[pred] = 0
                else:
                    # if count[pred] == 4 and (time.time()-start) <= 1.5:
                    present[pred] = True
                    log_time[pred] = datetime.datetime.now()
                    count[pred] = count.get(pred, 0) + 1
                    print(pred, present[pred], count[pred])
                    #if person_name != "unknown" and person_name not in message_sent:
                        #send_message_in(person_name)
                        #message_sent[person_name] = True

                cv2.putText(frame, str(person_name) + str(prob), (x+6,
                            y+h-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            else:
                person_name = "unknown"
                cv2.putText(frame, str(person_name), (x+6, y+h-6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                # Send message to the default number
        

        # Showing the image in another window
        cv2.imshow("Mark Attendance - In - Press q to exit", frame)
        key = cv2.waitKey(50) & 0xFF
        if(key == ord("q")):
            break

    # Stoping the videostream
    vs.stop()

    # destroying all the windows
    cv2.destroyAllWindows()
    update_attendance_in_db_in(present)
    return redirect('dashboard')'''



'''def mark_your_attendance(request):
    global stop_stream_flag_in, show_stream_flag_in
    #print("Stating the stream")
    detector = dlib.get_frontal_face_detector()

    predictor = dlib.shape_predictor(
        'E:/Suman/Final/face_recognition_data/shape_predictor_68_face_landmarks.dat')
    svc_save_path = "face_recognition_data/svc.sav"

    with open(svc_save_path, 'rb') as f:
        svc = pickle.load(f)
    fa = FaceAligner(predictor, desiredFaceWidth=96)
    encoder = LabelEncoder()
    encoder.classes_ = np.load('face_recognition_data/classes.npy')

    faces_encodings = np.zeros((1, 128))
    no_of_faces = len(svc.predict_proba(faces_encodings)[0])
    count = dict()
    present = dict()
    log_time = dict()
    start = dict()
    for i in range(no_of_faces):
        count[encoder.inverse_transform([i])[0]] = 0
        present[encoder.inverse_transform([i])[0]] = False

    vs = VideoStream(src=0).start()

    sampleNum = 0

    message_sent = {}

    #print("came atlast")
    #print(stop_stream)

    while not stop_stream_flag_in:

        #print("starting frame")

        frame = vs.read()

        frame = imutils.resize(frame, width=800)

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = detector(gray_frame, 0)

        for face in faces:
            print("INFO : inside for loop")
            (x, y, w, h) = face_utils.rect_to_bb(face)

            face_aligned = fa.align(frame, gray_frame, face)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 1)

            (pred, prob) = predict(face_aligned, svc)

            if(pred != [-1]):

                person_name = encoder.inverse_transform(np.ravel([pred]))[0]
                pred = person_name
                if count[pred] == 0:
                    start[pred] = time.time()
                    count[pred] = count.get(pred, 0) + 1

                if count[pred] == 4 and (time.time()-start[pred]) > 1.5:
                    count[pred] = 0
                else:
                    # if count[pred] == 4 and (time.time()-start) <= 1.5:
                    present[pred] = True
                    log_time[pred] = datetime.datetime.now()
                    count[pred] = count.get(pred, 0) + 1
                    print(pred, present[pred], count[pred])
                    #if person_name != "unknown" and person_name not in message_sent:
                        #send_message_out(person_name)
                        #message_sent[person_name] = True 
                cv2.putText(frame, str(person_name) + str(prob), (x+6,
                            y+h-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            else:
                person_name = "unknown"
                cv2.putText(frame, str(person_name), (x+6, y+h-6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        #cv2.imshow("Mark Attendance- Out - Press q to exit", frame)
        #if cv2.waitKey(1) & 0xFF == ord('q'):
            #stop_stream_flag_in = True
        
        if show_stream_flag_in:
            cv2.imshow("Mark Attendance In - Press q to Hide",frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                show_stream_flag_in = False
                cv2.destroyAllWindows()

        update_attendance_in_db_in(present)

    vs.stop()

    cv2.destroyAllWindows()

    return redirect('dashboard')'''

stop_stream_flag_in = False
show_stream_flag_in = False

mtcnn = MTCNN()

# Function to read frames and put them into the queue
def read_frames(vs, frame_queue):
    while not stop_stream_flag_in:
        frame = vs.read()
        frame = imutils.resize(frame, width= 800)
        frame_queue.put(frame)


# Function to detect faces and mark attendance
def mark_your_attendance(request):
    global stop_stream_flag_in, show_stream_flag_in

    # Load face recognition model
    svc_save_path = "face_recognition_data/svc.sav"
    with open(svc_save_path, 'rb') as f:
        svc = pickle.load(f)

    vs = VideoStream(src=0).start()

    frame_queue = queue.Queue(maxsize=1)
    read_thread = threading.Thread(target=read_frames, args=(vs, frame_queue))
    read_thread.start()

    count = dict()
    present = dict()
    log_time = dict()
    start = dict()

    encoder = LabelEncoder()
    encoder.classes_ = np.load('face_recognition_data/classes.npy')


    while not stop_stream_flag_in:
        frame = vs.read()

        # Resize the frame to a smaller size
        resized_frame = cv2.resize(frame, (640, 480))
        rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)

        small_window_width = 200
        small_window_height = 200

        # Detect faces using MTCNN
        boxes, probs, landmarks = mtcnn.detect(rgb_frame, landmarks=True)
        if boxes is not None:
            percentage = 0.8
            for box, landmark in zip(boxes, landmarks):
                x, y, w, h = map(int, box)
                
                # Check if the landmarks array has the expected shape
                if len(landmark) != 5:
                    continue  # Skip this iteration if landmarks are not detected properly

                # Calculate the bounding box coordinates based on the landmarks
                min_x = int(min(landmark[:, 0]))
                max_x = int(max(landmark[:, 0]))
                min_y = int(min(landmark[:, 1]))
                max_y = int(max(landmark[:, 1]))

                # Calculate the width and height of the extended bounding box
                width = max_x - min_x
                height = max_y - min_y
                extend_width = int(percentage * width)
                extend_height = int(percentage * height)

                # Extend the bounding box coordinates
                min_x -= extend_width
                max_x += extend_width
                min_y -= extend_height
                max_y += extend_height

                # Ensure the extended bounding box is within the frame boundaries
                min_x = max(0, min_x)
                max_x = min(frame.shape[1], max_x)
                min_y = max(0, min_y)
                max_y = min(frame.shape[0], max_y)

                # Draw the rectangle
                cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), (0, 255, 0), 2)
                for (x, y) in landmark.astype(int):
                    cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)

                #rgb_frame = frame[min_y:max_y, min_x:max_x]

                
                rgb_frame = frame[min_y:max_y, min_x:max_x]
                zoomed_face = cv2.resize(rgb_frame, (small_window_width, small_window_height))


                # Perform recognition
                (pred, prob) = predict(rgb_frame, svc) # here to change the frame for predictions

                if pred != [-1]:
                    person_name = encoder.inverse_transform(np.ravel([pred]))[0]
                    if person_name not in count:
                        count[person_name] = 0
                        present[person_name] = False

                    if count[person_name] == 0:
                        start[person_name] = time.time()
                    count[person_name] += 1

                    if count[person_name] == 4 and (time.time() - start[person_name]) > 1.5:
                        count[person_name] = 0
                    else:
                        present[person_name] = True
                        log_time[person_name] = datetime.datetime.now()

                    cv2.putText(zoomed_face, f"{person_name} {float(prob):.2f}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)


                else:
                    person_name = "unknown"
                    cv2.putText(frame, person_name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            

        if show_stream_flag_in:
            #cv2.imshow("Mark Attendance In - Press q to Hide",frame)
            cv2.namedWindow("Mark Attendance In - Press q to Hide")
            cv2.moveWindow("Mark Attendance In - Press q to Hide", 800, 0)

            if rgb_frame.shape[0] > 0 and rgb_frame.shape[1] > 0:
                cv2.imshow("Mark Attendance In - Press q to Hide", frame)
                cv2.imshow("Detected Face", zoomed_face)
                cv2.moveWindow("Detected Face", 800, 0)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    show_stream_flag_in = False
                    cv2.destroyAllWindows()

        update_attendance_in_db_in(present)

    vs.stop()

    cv2.destroyAllWindows()

    return redirect('dashboard')


def toggle_stream_in(request):
    global show_stream_flag_in
    show_stream_flag_in = not show_stream_flag_in
    return HttpResponse("Stream toggled")

def stop_stream_in(request):
    global stop_stream_flag_in
    stop_stream_flag_in = True
    
    threading.Timer(1.5, reset_stop_stream_flag_in).start()

    cv2.destroyAllWindows()
    return HttpResponse("Stream stopped")

def reset_stop_stream_flag_in():
    global stop_stream_flag_in
    stop_stream_flag_in = False


'''


def mark_your_attendance_out(request):

    detector = dlib.get_frontal_face_detector()

    # Add path to the shape predictor
    #predictor = dlib.shape_predictor(
    #    'face_recognition_data/shape_predictor_68_face_landmarks.dat')
    predictor = dlib.shape_predictor(
        'E:/Suman/Final/face_recognition_data/shape_predictor_68_face_landmarks.dat')
    svc_save_path = "face_recognition_data/svc.sav"

    with open(svc_save_path, 'rb') as f:
        svc = pickle.load(f)
    fa = FaceAligner(predictor, desiredFaceWidth=96)
    encoder = LabelEncoder()
    encoder.classes_ = np.load('face_recognition_data/classes.npy')

    faces_encodings = np.zeros((1, 128))
    no_of_faces = len(svc.predict_proba(faces_encodings)[0])
    count = dict()
    present = dict()
    log_time = dict()
    start = dict()
    for i in range(no_of_faces):
        count[encoder.inverse_transform([i])[0]] = 0
        present[encoder.inverse_transform([i])[0]] = False

    vs = VideoStream(src=0).start()

    sampleNum = 0

    message_sent = {}

    while(True):

        frame = vs.read()

        frame = imutils.resize(frame, width=800)

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = detector(gray_frame, 0)

        for face in faces:
            print("INFO : inside for loop")
            (x, y, w, h) = face_utils.rect_to_bb(face)

            face_aligned = fa.align(frame, gray_frame, face)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 1)

            (pred, prob) = predict(face_aligned, svc)

            if(pred != [-1]):

                person_name = encoder.inverse_transform(np.ravel([pred]))[0]
                pred = person_name
                if count[pred] == 0:
                    start[pred] = time.time()
                    count[pred] = count.get(pred, 0) + 1

                if count[pred] == 4 and (time.time()-start[pred]) > 1.5:
                    count[pred] = 0
                else:
                    # if count[pred] == 4 and (time.time()-start) <= 1.5:
                    present[pred] = True
                    log_time[pred] = datetime.datetime.now()
                    count[pred] = count.get(pred, 0) + 1
                    print(pred, present[pred], count[pred])
                    #if person_name != "unknown" and person_name not in message_sent:
                        #send_message_out(person_name)
                        #message_sent[person_name] = True 
                cv2.putText(frame, str(person_name) + str(prob), (x+6,
                            y+h-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            else:
                person_name = "unknown"
                cv2.putText(frame, str(person_name), (x+6, y+h-6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_stream = True


        update_attendance_in_db_out(present)
        #return redirect('dashboard')        
        
    vs.stop()
    return redirect('dashboard')        



        # Showing the image in another window
        #cv2.imshow("Mark Attendance- Out - Press q to exit", frame)
        # To get out of the loop
        #key = cv2.waitKey(50) & 0xFF
        #if(key == ord("q")):
           # break

    # Stoping the videostream
    vs.stop()

    # destroying all the windows
    cv2.destroyAllWindows()
    update_attendance_in_db_out(present)
    return redirect('dashboard')
'''



'''def mark_your_attendance_out(request):
    global stop_stream_flag_out, show_stream_flag_out
    #print("Stating the stream")
    detector = dlib.get_frontal_face_detector()

    predictor = dlib.shape_predictor(
        'E:/Suman/Final/face_recognition_data/shape_predictor_68_face_landmarks.dat')
    svc_save_path = "face_recognition_data/svc.sav"

    with open(svc_save_path, 'rb') as f:
        svc = pickle.load(f)
    fa = FaceAligner(predictor, desiredFaceWidth=96)
    encoder = LabelEncoder()
    encoder.classes_ = np.load('face_recognition_data/classes.npy')

    faces_encodings = np.zeros((1, 128))
    no_of_faces = len(svc.predict_proba(faces_encodings)[0])
    count = dict()
    present = dict()
    log_time = dict()
    start = dict()
    for i in range(no_of_faces):
        count[encoder.inverse_transform([i])[0]] = 0
        present[encoder.inverse_transform([i])[0]] = False

    vs = VideoStream(src=0).start()

    sampleNum = 0

    message_sent = {}

    #print("came atlast")
    #print(stop_stream)

    while not stop_stream_flag_out:

        #print("starting frame")

        frame = vs.read()

        frame = imutils.resize(frame, width=800)

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = detector(gray_frame, 0)

        for face in faces:
            print("INFO : inside for loop")
            (x, y, w, h) = face_utils.rect_to_bb(face)

            face_aligned = fa.align(frame, gray_frame, face)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 1)

            (pred, prob) = predict(face_aligned, svc)

            if(pred != [-1]):

                person_name = encoder.inverse_transform(np.ravel([pred]))[0]
                pred = person_name
                if count[pred] == 0:
                    start[pred] = time.time()
                    count[pred] = count.get(pred, 0) + 1

                if count[pred] == 4 and (time.time()-start[pred]) > 1.5:
                    count[pred] = 0
                else:
                    # if count[pred] == 4 and (time.time()-start) <= 1.5:
                    present[pred] = True
                    log_time[pred] = datetime.datetime.now()
                    count[pred] = count.get(pred, 0) + 1
                    print(pred, present[pred], count[pred])
                    #if person_name != "unknown" and person_name not in message_sent:
                        #send_message_out(person_name)
                        #message_sent[person_name] = True 
                cv2.putText(frame, str(person_name) + str(prob), (x+6,
                            y+h-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            else:
                person_name = "unknown"
                cv2.putText(frame, str(person_name), (x+6, y+h-6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        #cv2.imshow("Mark Attendance- Out - Press q to exit", frame)
        #if cv2.waitKey(1) & 0xFF == ord('q'):
            #stop_stream_flag_out = True
        
        if show_stream_flag_out:
            cv2.imshow("Mark Attendance Out - Press q to Hide",frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                show_stream_flag_out = False
                cv2.destroyAllWindows()

        update_attendance_in_db_out(present)

    vs.stop()

    cv2.destroyAllWindows()

    return redirect('dashboard')'''

'''stop_stream_flag_out = False
show_stream_flag_out = False

mtcnn = MTCNN()

def read_frames(vs, frame_queue):
    while not stop_stream_flag_out:
        frame = vs.read()
        frame_queue.put(frame)'''



# Function to detect faces and mark attendance
stop_stream_flag_out = False
show_stream_flag_out = False

mtcnn = MTCNN()

# Function to detect faces and mark attendance
def mark_your_attendance_out(request):
    global stop_stream_flag_out, show_stream_flag_out

    # Load face recognition model
    svc_save_path = "face_recognition_data/svc.sav"
    with open(svc_save_path, 'rb') as f:
        svc = pickle.load(f)

    vs = VideoStream(src=0).start()

    count = dict()
    present = dict()
    log_time = dict()
    start = dict()

    encoder = LabelEncoder()
    encoder.classes_ = np.load('face_recognition_data/classes.npy')

    while not stop_stream_flag_out:
        frame = vs.read()

        # Resize the frame to a smaller size
        resized_frame = cv2.resize(frame, (640, 480))
        rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)

        # Detect faces using MTCNN
        boxes, probs, landmarks = mtcnn.detect(rgb_frame, landmarks=True)
        if boxes is not None:
            percentage = 0.8
            for box, landmark in zip(boxes, landmarks):
                x, y, w, h = map(int, box)
                
                # Check if the landmarks array has the expected shape
                if len(landmark) != 5:
                    continue  # Skip this iteration if landmarks are not detected properly

                # Calculate the bounding box coordinates based on the landmarks
                min_x = int(min(landmark[:, 0]))
                max_x = int(max(landmark[:, 0]))
                min_y = int(min(landmark[:, 1]))
                max_y = int(max(landmark[:, 1]))

                # Calculate the width and height of the extended bounding box
                width = max_x - min_x
                height = max_y - min_y
                extend_width = int(percentage * width)
                extend_height = int(percentage * height)

                # Extend the bounding box coordinates
                min_x -= extend_width
                max_x += extend_width
                min_y -= extend_height
                max_y += extend_height

                # Ensure the extended bounding box is within the frame boundaries
                min_x = max(0, min_x)
                max_x = min(frame.shape[1], max_x)
                min_y = max(0, min_y)
                max_y = min(frame.shape[0], max_y)

                # Draw the rectangle
                cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), (0, 255, 0), 2)
                for (x, y) in landmark.astype(int):
                    cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)

                # Perform recognition
                (pred, prob) = predict(rgb_frame, svc)

                if pred != [-1]:
                    person_name = encoder.inverse_transform(np.ravel([pred]))[0]
                    if person_name not in count:
                        count[person_name] = 0
                        present[person_name] = False

                    if count[person_name] == 0:
                        start[person_name] = time.time()
                    count[person_name] += 1

                    if count[person_name] == 4 and (time.time() - start[person_name]) > 1.5:
                        count[person_name] = 0
                    else:
                        present[person_name] = True
                        log_time[person_name] = datetime.datetime.now()

                    cv2.putText(frame, f"{person_name} {float(prob):.2f}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                else:
                    person_name = "unknown"
                    cv2.putText(frame, person_name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        

        if show_stream_flag_out:
            cv2.imshow("Mark Attendance Out - Press q to Hide",frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                show_stream_flag_out = False
                cv2.destroyAllWindows()

        update_attendance_in_db_out(present)
        

    vs.stop()

    cv2.destroyAllWindows()
    

    return redirect('dashboard')

def toggle_stream_out(request):
    global show_stream_flag_out
    show_stream_flag_out = not show_stream_flag_out
    return HttpResponse("Stream toggled")

def stop_stream_out(request):
    global stop_stream_flag_out
    stop_stream_flag_out = True
    
    threading.Timer(1.5, reset_stop_stream_flag_out).start()
    
    cv2.destroyAllWindows()
    print("Stream stopped")

    return HttpResponse("Stream stopped")

def reset_stop_stream_flag_out():
    global stop_stream_flag_out
    stop_stream_flag_out = False

@login_required
def train(request):
    if request.user.username != 'admin':
        return redirect('not-authorised')

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
    svc_save_path = "face_recognition_data/svc.sav"
    with open(svc_save_path, 'wb') as f:
        pickle.dump(svc, f)

    vizualize_Data(X1, targets)

    messages.success(request, f'Training Completed.')

    return render(request, "recognition/train.html")


@login_required
def not_authorised(request):
    return render(request, 'recognition/not_authorised.html')


@login_required
def view_attendance_home(request):
    total_num_of_emp = total_number_employees()
    emp_present_today = employees_present_today()
    this_week_emp_count_vs_date()
    last_week_emp_count_vs_date()
    return render(request, "recognition/view_attendance_home.html", {'total_num_of_emp': total_num_of_emp, 'emp_present_today': emp_present_today})


@login_required
def view_attendance_date(request):
    if request.user.username != 'admin':
        return redirect('not-authorised')
    qs = None
    time_qs = None
    present_qs = None

    if request.method == 'POST':
        form = DateForm(request.POST)
        if form.is_valid():
            date = form.cleaned_data.get('date')
            print("date:" + str(date))
            time_qs = Time.objects.filter(date=date)
            present_qs = Present.objects.filter(date=date)
            if(len(time_qs) > 0 or len(present_qs) > 0):
                qs = hours_vs_employee_given_date(present_qs, time_qs)

                return render(request, 'recognition/view_attendance_date.html', {'form': form, 'qs': qs})
            else:
                messages.warning(request, f'No records for selected date.')
                return redirect('view-attendance-date')

    else:

        form = DateForm()
        return render(request, 'recognition/view_attendance_date.html', {'form': form, 'qs': qs})


@login_required
def view_attendance_employee(request):
    if request.user.username != 'admin':
        return redirect('not-authorised')
    time_qs = None
    present_qs = None
    qs = None

    if request.method == 'POST':
        form = UsernameAndDateForm(request.POST)
        if form.is_valid():
            username = form.cleaned_data.get('username')
            if username_present(username):

                u = User.objects.get(username=username)

                time_qs = Time.objects.filter(user=u)
                present_qs = Present.objects.filter(user=u)
                date_from = form.cleaned_data.get('date_from')
                date_to = form.cleaned_data.get('date_to')

                if date_to < date_from:
                    messages.warning(request, f'Invalid date selection.')
                    return redirect('view-attendance-employee')
                else:

                    time_qs = time_qs.filter(date__gte=date_from).filter(
                        date__lte=date_to).order_by('-date')
                    present_qs = present_qs.filter(date__gte=date_from).filter(
                        date__lte=date_to).order_by('-date')

                    if (len(time_qs) > 0 or len(present_qs) > 0):
                        qs = hours_vs_date_given_employee(
                            present_qs, time_qs, admin=True)
                        return render(request, 'recognition/view_attendance_employee.html', {'form': form, 'qs': qs})
                    else:
                        #print("inside qs is None")
                        messages.warning(
                            request, f'No records for selected duration.')
                        return redirect('view-attendance-employee')

            else:
                print("invalid username")
                messages.warning(request, f'No such username found.')
                return redirect('view-attendance-employee')

    else:

        form = UsernameAndDateForm()
        return render(request, 'recognition/view_attendance_employee.html', {'form': form, 'qs': qs})


@login_required
def view_my_attendance_employee_login(request):
    if request.user.username == 'admin':
        return redirect('not-authorised')
    qs = None
    time_qs = None
    present_qs = None
    if request.method == 'POST':
        form = DateForm_2(request.POST)
        if form.is_valid():
            u = request.user
            time_qs = Time.objects.filter(user=u)
            present_qs = Present.objects.filter(user=u)
            date_from = form.cleaned_data.get('date_from')
            date_to = form.cleaned_data.get('date_to')
            if date_to < date_from:
                messages.warning(request, f'Invalid date selection.')
                return redirect('view-my-attendance-employee-login')
            else:

                time_qs = time_qs.filter(date__gte=date_from).filter(
                    date__lte=date_to).order_by('-date')
                present_qs = present_qs.filter(date__gte=date_from).filter(
                    date__lte=date_to).order_by('-date')

                if (len(time_qs) > 0 or len(present_qs) > 0):
                    qs = hours_vs_date_given_employee(
                        present_qs, time_qs, admin=False)
                    return render(request, 'recognition/view_my_attendance_employee_login.html', {'form': form, 'qs': qs})
                else:

                    messages.warning(
                        request, f'No records for selected duration.')
                    return redirect('view-my-attendance-employee-login')
    else:

        form = DateForm_2()
        return render(request, 'recognition/view_my_attendance_employee_login.html', {'form': form, 'qs': qs})
