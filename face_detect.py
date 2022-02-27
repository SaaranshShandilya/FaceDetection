import cv2 as cv


capture = cv.VideoCapture(0)
while True:
    isTrue, frame = capture.read()
    grey = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    haar_cascade = cv.CascadeClassifier('haar_face_detect.xml')

    faces_rect = haar_cascade.detectMultiScale(grey, scaleFactor=1.1, minNeighbors=1)

    # print(f'Number of faces found = {len(faces_rect)}')

    for (x,y,w,h) in faces_rect:
        cv.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), thickness=2)

    if isTrue:
        cv.imshow('Video', frame)
        if cv.waitKey(1) & 0xFF == ('q'):
            break
    else:
        break

capture.release()
cv.destroyAllWindows()