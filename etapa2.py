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
    return frame_preprocessed


if __name__ == "__main__":
    source = 0

    if len(sys.argv) > 1:
        source = sys.argv[1]

    cap = cv2.VideoCapture(source)

    while 1:
        hasFrame, frame = cap.read()
        if not hasFrame:
            break

        frame_processed = preprocess_frame(frame)
        cv2.imshow("SNCT", frame_processed)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cv2.destroyAllWindows()
