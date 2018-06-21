import cv2
from modules import normalization, similarity


def live_capture(model):
    """
    starts the live capture with the webcam
    webcam is needed to run this function
    """
    video_capture = cv2.VideoCapture(0)
    video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
    video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)

    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()

        landmarks, norm = normalization.normalize_face(frame, 200, True)
        if landmarks is not None:
            for landmark in landmarks:
                scalar = (landmark[0], landmark[1])
                cv2.circle(frame, scalar, 2, (0, 0, 255), -1)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect the faces in the scene
        # detected_faces = detection.detect_faces_in_frame(frame, gray)
        # detected_extended = detection.detect_cat_faces_in_frame(frame, gray)
        # detect_eyes = detection.detect_eyes_in_frame(frame, gray)
        # detect_smiles = detection.detect_smiles_in_frame(frame, gray)
        # detected_bodies = detection.detect_full_body_in_frame(frame, gray)

        # measure the frame rate before displaying
        # fps = video_capture.get(cv2.CAP_PROP_FPS)
        # cv2.putText(frame, "{0} FPS".format(fps), (5, 15), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 255), 2)

        # display the new frame
        cv2.imshow('cam', frame)

        pressed_key = cv2.waitKey(1)
        if pressed_key == ord('f'):
            """
            biggest_face_area = 0
            face_to_crop = detected_faces[0]
            for idx, detectedFace in enumerate(detected_faces):
                face_area = detectedFace[2] * detectedFace[3]
                if face_area > biggest_face_area:
                    biggest_face_area = face_area
                    face_to_crop = detectedFace
            similarity.capture_face_features(gray, face_to_crop)
            """
            landmarks, aligned = normalization.normalize_face(gray, size=128, should_show=False)
            if len(aligned) > 0:
                similarity.similarity_on_single_img(aligned, model)
        if pressed_key == ord('q'):
            break

    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()
