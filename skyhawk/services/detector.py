import numpy as np
import cv2
import pickle
import os
from datetime import datetime
import face_recognition


def run():
    know_encodings = []
    student_names = []

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    bin_direc = "../bin"
    bin_dir = os.path.join(BASE_DIR, bin_direc)
    bin_file = os.path.join(bin_dir, 'facedata.bin')

    # recognizer.read("skyhawk/bin/face-trainner.yml")
    with open(bin_file,'rb') as py:
        know_encodedict = pickle.load(py)

    for students,encording in know_encodedict.items():
        know_encodings.append(encording)
        student_names.append(students)

    def markattendance(name):
        with open('skyhawk/bin/Attendance.csv', 'r+') as f:
            dataList = f.readlines()
            nameList = []
            for line in dataList:
                entry = line.split(',')
                nameList.append(entry[0])
            if name not in nameList:
                now = datetime.now()
                date = now.strftime('%b %d %Y')
                day = now.strftime('%a')
                time = now.strftime('%H:%M:%S')
                f.writelines(f'\n{name}, {date}, {time}, {day}')


    capture = cv2.VideoCapture(0)

    while True:
        #capturing face
        result, current_frame = capture.read()
        
        #redusing size
        framesmall = cv2.resize(current_frame,(0,0),None,0.25,0.25)

        #geting encoding of face from camera
        framesmall = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)
        faces_incurrentframe = face_recognition.face_locations(framesmall)
        encodes_ofcurrentframe = np.array(face_recognition.face_encodings(framesmall))
        

        #comparing encodings and finding face
        for encodeface,facelocation in zip(encodes_ofcurrentframe,faces_incurrentframe):
            matches = face_recognition.compare_faces(know_encodings,encodeface)
            face_distance = face_recognition.face_distance(know_encodings,encodeface)
            matchIndex = np.argmin(face_distance)
            
            if matches[matchIndex]:
                name = student_names[matchIndex].lower()
                print(name)
                y1,x2,y2,x1 = facelocation
                #y1,x2,y2,x1 = y1,x2*2,y2*2,x1*2
                cv2.rectangle(current_frame,(x1,y1),(x2,y2),(255,0,0),2)
                cv2.rectangle(current_frame,(x1,y2-35),(x2,y2),(255,0,0),cv2.FILLED)
                cv2.putText(current_frame,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,255),2)

                markattendance(name)
    
        cv2.imshow("camera",current_frame)          
        if cv2.waitKey(20) & 0xFF==ord('q'):
            break

    # When everything done, release the capture
    capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run()