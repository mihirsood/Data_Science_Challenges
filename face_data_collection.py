# Write a python script that captures images from your webcam video stream

# Extract all Faaces from the image frame (using haarcascades)

# Stores the Face information into numpy arrays

# - Read and show video stream , capture images
# - Detect Faces and show bounding box(haarcascade)
# - Flatten the largest face image(gray scale) and save in a numpy array
# - Repeat the above for multiple people to generate training data
import cv2
import numpy as np

cam = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

face_data = []

dataset_path="./data/"
name=input("Enter your name ")

#only for mac 
# %matplotlib

while True:
    ret, frame = cam.read()
    if ret == False:
        continue
        
    gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) #bgr-> grayscale conversion b/c haarcascade works on gray scale 24x24 window
    faces = face_cascade.detectMultiScale(gray_frame,1.3,5)
    
    for face in faces:
        face_section = None
        x,y,w,h = face
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        
        offset = 10 #increasing 10 pixels on all the sides and cropping the new file
        face_section = frame[y-offset:y+h+offset,x-offset:x+w+offset]
        face_section = cv2.resize(face_section,(100,100))
        
        face_data.append(face_section)
        
        cv2.imshow("cropped face",face_section)
    
    cv2.imshow("window",frame)
    key=cv2.waitKey(1) #1ms 
    # 0 means wait for indefinite time
    if key==ord("q"): # ord returns ascii vale
        break

cam.release()
cv2.destroyAllWindows()


face_data = np.array(face_data)
face_data = face_data.reshape(face_data.shape[0],-1)
print(face_data.shape)

np.save(dataset_path+name+".npy",face_data)
print("data saved at "+ dataset_path+name+".npy")
