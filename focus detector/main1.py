import dlib

def download_shape_predictor():
    dlib.download_model("shape_predictor_68_face_landmarks.dat.bz2")
    print("Shape predictor downloaded successfully.")

download_shape_predictor()
