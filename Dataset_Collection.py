import os
import cv2

def path_existence(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)

face_ID = input("Enter face identification number : ")
video_cam = cv2.VideoCapture(0)

face_detector = cv2.CascadeClassifier('C:/Users/Shudhanshu Sharma/PycharmProjects/Minor_Project/haarcascade_frontalface_default.xml')
count = 0

path_existence("C:/Users/Shudhanshu Sharma/PycharmProjects/Minor_Project/dataset")

while (True):
    _, image_frame = video_cam.read()
    gray_image = cv2.cvtColor(image_frame,cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray_image, 1.3, 5)



    #loops
    for (x, y, w, h) in faces:
        cv2.rectangle(image_frame, (x,y), (x+w, y+h), (255, 0, 0), 3)
        count += 1
        cv2.imwrite("dataset/User." + str(face_ID) + '.' + str(count) + ".jpg",gray_image[y:y+h, x: x+w])
        cv2.putText(image_frame, str(count), (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 3)
        cv2.imshow('Frame',image_frame)

    if cv2.waitKey(100) & 0xFF == ord('q'):
        break
    elif count >= 100:
        print("Successfully Captured")
        break

video_cam.release()
cv2.destroyALlWindows()