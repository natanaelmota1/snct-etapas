from __future__ import division
import cv2
import sys

if __name__ == "__main__":
    source = 0
    if len(sys.argv) > 1:
        source = sys.argv[1]

    cap = cv2.VideoCapture(source)

    while 1:
        hasFrame, frame = cap.read()
        if not hasFrame:
            break

        cv2.imshow("SNCT", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cv2.destroyAllWindows()
