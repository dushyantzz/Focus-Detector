import cv2
import numpy as np
import dlib
import time


class GazeMonitor:
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        self.last_focus_time = time.time()
        self.distraction_start = None
        self.focus_threshold = 2  
        self.attention_span = 0 

    def get_gaze_direction(self, eye_points, landmarks):
        try:
            eye_region = np.array([(landmarks.part(point).x, landmarks.part(point).y)
                                   for point in eye_points])
            height = eye_region[:, 1].max() - eye_region[:, 1].min()
            width = eye_region[:, 0].max() - eye_region[:, 0].min()

            eye_center = eye_region.mean(axis=0).astype("int")
            pupil = self._find_pupil(eye_center, height, width)

            return (pupil[0] - eye_center[0], pupil[1] - eye_center[1])
        except:
            return (0, 0)

    def _find_pupil(self, eye_center, height, width):
        try:
            y_start = max(0, eye_center[1] - height // 2)
            y_end = min(self.frame.shape[0], eye_center[1] + height // 2)
            x_start = max(0, eye_center[0] - width // 2)
            x_end = min(self.frame.shape[1], eye_center[0] + width // 2)

            if y_start >= y_end or x_start >= x_end:
                return eye_center

            eye_roi = self.frame[y_start:y_end, x_start:x_end]
            gray_eye = cv2.cvtColor(eye_roi, cv2.COLOR_BGR2GRAY)
            _, threshold = cv2.threshold(gray_eye, 70, 255, cv2.THRESH_BINARY_INV)

            contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                pupil = max(contours, key=cv2.contourArea)
                moments = cv2.moments(pupil)
                if moments["m00"] != 0:
                    cx = int(moments['m10'] / moments['m00'])
                    cy = int(moments['m01'] / moments['m00'])
                    return (x_start + cx, y_start + cy)

            return eye_center
        except:
            return eye_center

    def run(self):
        cap = cv2.VideoCapture(0)
        while True:
            ret, self.frame = cap.read()
            if not ret:
                break

            self.frame = cv2.flip(self.frame, 1)
            gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
            faces = self.detector(gray)

            is_focused = False

            for face in faces:
                try:
                    landmarks = self.predictor(gray, face)

                    left_eye = self.get_gaze_direction([36, 37, 38, 39, 40, 41], landmarks)
                    right_eye = self.get_gaze_direction([42, 43, 44, 45, 46, 47], landmarks)

                    gaze_x = (left_eye[0] + right_eye[0]) / 2
                    gaze_y = (left_eye[1] + right_eye[1]) / 2

                    if abs(gaze_x) < 5 and abs(gaze_y) < 5:  
                        is_focused = True
                        self.distraction_start = None
                        self.attention_span += time.time() - self.last_focus_time
                        self.last_focus_time = time.time()
                    else: 
                        if self.distraction_start is None:
                            self.distraction_start = time.time()

                except Exception as e:
                    print(f"Error processing face: {str(e)}")

            if not is_focused and self.distraction_start is None:
                self.distraction_start = time.time()

            if is_focused:
                status_color = (0, 255, 0)  
                status_text = f"Focused: {int(self.attention_span)}s"
                progress = (time.time() - self.last_focus_time) / self.focus_threshold
            else:
                status_color = (0, 0, 255)
                if self.distraction_start is not None:  
                    distraction_time = time.time() - self.distraction_start
                    status_text = f"Distracted: {int(distraction_time)}s"
                    progress = distraction_time / self.focus_threshold

                    if distraction_time > self.focus_threshold:
                        cv2.putText(self.frame, "ALERT! FOCUS ON SCREEN!",
                                    (50, 100), cv2.FONT_HERSHEY_SIMPLEX,
                                    1.5, (0, 0, 255), thickness=3)
                else:
                    status_text = "Distracted: 0s"
                    progress = 0

            cv2.rectangle(self.frame, (0, 0), (400, 70), (30, 30, 30), -1)
            cv2.putText(self.frame, status_text,
                        (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1.0,
                        color=status_color,
                        thickness=2)

            bar_width = int(self.frame.shape[1] * min(progress, 1))
            cv2.rectangle(self.frame,
                          (0, self.frame.shape[0] - 20),
                          (bar_width, self.frame.shape[0]),
                          status_color,
                          thickness=-1)

            cv2.imshow("Focus Assistant", self.frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    GazeMonitor().run()
