import cv2

# camera stream
captee = cv2.VideoCapture(0)
# cascade classifier
cascade = cv2.CascadeClassifier("resources/haarcascade_face_front.xml")

while True:
    # capture cam feed
    _, img = captee.read()
    # convert to grayscale
    imggs = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    detectedFaces = cascade.detectMultiScale(imggs)
    for x, y, wdt, hgt in detectedFaces:
        # display rectangle around detected faces
        cv2.rectangle(img, (x, y), (x + wdt, y + hgt), color=(255, 0, 0), thickness=3)
    cv2.imshow("Camera Feed", img)

    # stop application when q has been pressed
    if cv2.waitKey(1) == ord("q"):
        break

# release used resources
captee.release()
cv2.destroyAllWindows()
