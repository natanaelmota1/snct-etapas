from __future__ import division
import cv2
import sys


def preprocess_frame(frame, in_height=500, in_width=0):
    frame_copy = frame.copy()
    frame_height = frame_copy.shape[0]
    frame_width = frame_copy.shape[1]
    if not in_width:
        in_width = int((frame_width / frame_height) * in_height)

    scale_height = frame_height / in_height
    scale_width = frame_width / in_width
    frame_reescaled = cv2.resize(frame_copy, (in_width, in_height))
    frame_preprocessed = cv2.cvtColor(frame_reescaled, cv2.COLOR_BGR2GRAY)
    return frame_preprocessed, frame_width, frame_height, scale_height, scale_width


def detect(frame):
    frame_processed, frame_width, frame_height, scale_height, scale_width = preprocess_frame(frame)
    faces = faceCascade.detectMultiScale(frame_processed)
    bboxes = []
    for (x, y, w, h) in faces:
        x1 = x
        y1 = y
        x2 = x + w
        y2 = y + h
        cvRect = [int(x1 * scale_width), int(y1 * scale_height),
                  int(x2 * scale_width), int(y2 * scale_height)]
        bboxes.append(cvRect)
        cv2.rectangle(frame_processed, (cvRect[0], cvRect[1]), (cvRect[2], cvRect[3]), (0, 255, 0),
                      int(round(frame_height / 150)), 4)
    return frame_processed, bboxes


if __name__ == "__main__":
    source = 0
    faceCascade = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')

    if len(sys.argv) > 1:
        source = sys.argv[1]

    cap = cv2.VideoCapture(source)

    while 1:
        hasFrame, frame = cap.read()
        if not hasFrame:
            break

        frame_detections, bboxes = detect(frame)
        cv2.imshow("SNCT", frame_detections)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cv2.destroyAllWindows()
