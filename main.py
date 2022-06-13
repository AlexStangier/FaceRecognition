import cv2


# check if given points are inside range of previous points
def checkifcoordsareinsample(samplex0, samplex1, sampley0, sampley1, x, y):
    rangex = range(samplex0, samplex1)
    rangey = range(sampley0, sampley1)
    if x in rangex and y in rangey:
        return True


def run():
    # camera stream
    captee = cv2.VideoCapture(0)
    # cascade classifier using horizontal and vertical features
    cascadeFace = cv2.CascadeClassifier("resources/haarcascade_face_front.xml")
    cascadeSmile = cv2.CascadeClassifier("resources/haarcascade_smile.xml")

    while True:
        # capture cam feed
        _, img = captee.read()
        # convert to grayscale
        imggs = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        detectedFaces = cascadeFace.detectMultiScale(imggs)
        detectedSmiles = cascadeSmile.detectMultiScale(imggs)
        for x, y, wdt, hgt in detectedFaces:
            # display rectangle around detected faces
            cv2.rectangle(img, (x, y), (x + wdt, y + hgt), color=(0, 0, 255), thickness=3)
            # loop possibles smiles and mark them green, if face appears to be smiling mark it green aswell
            for x2, y2, wdt2, hgt2 in detectedSmiles:
                if checkifcoordsareinsample(x, x + wdt, y, y + hgt, x2, y2):
                    cv2.rectangle(img, (x2, y2), (x2 + wdt2, y2 + hgt2), color=(0, 255, 0), thickness=2)
                    cv2.rectangle(img, (x, y), (x + wdt, y + hgt), color=(0, 255, 0), thickness=3)

        cv2.imshow("Camera Feed", img)

        # stop application when q has been pressed
        if cv2.waitKey(1) == ord("q"):
            break

    # release used resources
    captee.release()
    cv2.destroyAllWindows()


def main():
    run()


if __name__ == '__main__':
    main()
