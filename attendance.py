from cv2 import VideoCapture
import face_recognition
import cv2
import numpy as np
import csv
import os
from datetime import datetime

video = cv2.VideoCapture(0)

sherlock_image = face_recognition.load_image_file("photos/sherlock.png")
sherlock_encoding = face_recognition.face_encodings(sherlock_image)[0]

sheldon_image = face_recognition.load_image_file("photos/sheldon.png")
sheldon_encoding = face_recognition.face_encodings(sheldon_image)[0]

thomas_image = face_recognition.load_image_file("photos/thomas.png")
thomas_encoding = face_recognition.face_encodings(thomas_image)[0]

jamie_image = face_recognition.load_image_file("photos/jamie.png")
jamie_encoding = face_recognition.face_encodings(jamie_image)[0]

known_face_encoding = [
    sherlock_encoding,
    sheldon_encoding,
    thomas_encoding,
    jamie_encoding
]

known_faces_names = [
    "Shelock",
    "SHeldon",
    "Thomas",
    "Jamie"
]

students = known_faces_names.copy()
 
face_locations = []
face_encodings = []
face_names = []
s=True
 
 
now = datetime.now()
current_date = now.strftime("%Y-%m-%d")
 
 
 
f = open(current_date+'.csv','w+',newline = '')
lnwriter = csv.writer(f)
 
while True:
    _,frame = video.read()

    small_frame = cv2.resize(frame,(0,0),fx=0.25,fy=0.25)
    rgb_small_frame = small_frame[:,:,::-1]
    if s:
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame,face_locations)
        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encoding,face_encoding)
            name=""
            face_distance = face_recognition.face_distance(known_face_encoding,face_encoding)
            best_match_index = np.argmin(face_distance)
            if matches[best_match_index]:
                name = known_faces_names[best_match_index]
 
            face_names.append(name)
            if name in known_faces_names:
                font = cv2.FONT_HERSHEY_SIMPLEX
                bottomLeftCornerOfText = (10,100)
                fontScale              = 1.5
                fontColor              = (255,0,0)
                thickness              = 3
                lineType               = 2
 
                cv2.putText(frame,name+' Present', 
                    bottomLeftCornerOfText, 
                    font, 
                    fontScale,
                    fontColor,
                    thickness,
                    lineType)
 
                if name in students:
                    students.remove(name)
                    print(students)
                    current_time = now.strftime("%H-%M-%S")
                    lnwriter.writerow([name,current_time])
    cv2.imshow("attendence system",frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
 
video.release()
cv2.destroyAllWindows()
f.close()