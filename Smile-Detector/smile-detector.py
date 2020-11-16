import cv2

#face classifer

face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
smile_detector = cv2.CascadeClassifier('haarcascade_smile.xml')

# WebCam feed
webcam =  cv2.VideoCapture('0')  #0 for webcame and also u can pass mp4 files also

#show the current frame
while True:

    #Read the current frame from the webcam video stream
    successful_frame_Read, frame  = webcam.read()
    
    #if there is an error, abort
    if not successful_frame_Read:
        break
    
    #change to gray scale
    frame_gray =  cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    #detect faces first
    faces = face_detector.detectMultiScale(frame_gray)
    

    # Run the face detection within each of those faces
    for (x, y, w, h) in faces:
        
        # Draw a rectangle around the face
        cv2.rectangle(frame, (x,y), (x+w, y+h), (100,200,50), 4)
        
        # Get the subframe  using numpy N-dimensional array Slicing
        the_face= frame[y:y+h, x:x+w]
        
        #change to gray scale
        face_gray= cv2.cvtColor(the_face, cv2.COLOR_BGR2GRAY)
        
        smiles =  smile_detector.detectMultiScale(face_gray,scaleFactor = 1.7, minNeighbors = 20)
        
        # Label this face as smiling
        if len(smiles)>0:
            cv2.putText(frame,'smiling', (x,y+h+40), fontScale = 3, fontFace = cv2.FONT_HERSHEY_PLAIN, color = (255,255,255))
    #show the current frame
    cv2.imshow('Smile Detector', frame)

    #display 
    cv2.waitKey(1)

#clean up
webcam.release()
cv2.destoryAllWindows()
