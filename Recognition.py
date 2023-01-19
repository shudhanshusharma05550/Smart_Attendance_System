import cv2
import numpy as np
import datetime
import xlsxwriter

now = datetime.datetime.now()
time = now.strftime("%y-%m-%d %H:%M:%S")
face_cas = cv2.CascadeClassifier('C:/Users/Shudhanshu Sharma/PycharmProjects/Minor_Project/haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('C:/Users/Shudhanshu Sharma/PycharmProjects/Minor_Project/trainer/trainer.yml')
flag = 0
id = 0
workbook = xlsxwriter.Workbook('C:/Users/Shudhanshu Sharma/PycharmProjects/Minor_Project/Attendance_Sheet.xlsx')
worksheet = workbook.add_worksheet()
worksheet.write_row(0, 1, ['TIme_Stamp', 'Class', 'Roll No', 'Name', 'yes/no'])
dict = {
    'item1': 1
}
 #font = cv2.InitFont(cv2.cv.CV_FONT_HERSHEY_SIMPLEX, 5, 1, 0, 1, 1)
font = cv2.FONT_HERSHEY_SIMPLEX

while (True):
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cas.detectMultiScale(gray, 1.3, 7)
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        id, conf = recognizer.predict(roi_gray)
        if (conf < 50):
            if (id == 1):
                id = 'Barun Sobti'
                if ((str(id)) not in dict):
                    worksheet.write_row(1, 1, [time, 'BCA', 1, id, 'yes'])
                    dict[str(id)] = str(id)

            elif (id == 2):
                id = "Monu Sharma"
                if ((str(id)) not in dict):
                    worksheet.write_row(2, 1, [time, 'BCA', 2, id, 'yes'])
                    dict[str(id)] = str(id)
            workbook.close()


        else:
            id = 'Unknown, can not recognize'
            print(id)
            flag = flag + 1
            break

        cv2.putText(img, str(id) + " " + str(conf) + "-" + time, (x, y - 10), font, 1, (220,0,0), 3)

    cv2.imshow('frame', img)
    # cv2.imshow('gray',gray);


    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()