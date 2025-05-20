import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from sklearn.ensemble import IsolationForest
import joblib
import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image

cnn_model = load_model("3d_cnn.h5")
resnet_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
iso_forest_model = joblib.load("isolationforest-anomaly_detection_model.h5")
lstm_model = load_model("lstm-anomaly_detection_model.h5")
def preprocess_video_cnn(video_path, target_shape=(64, 64)):
    try:
        video_cap = cv2.VideoCapture(video_path)
        frame_count = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        preprocessed_frames = []

        while True:
            success, frame = video_cap.read()
            if not success:
                break
            resized_frame = cv2.resize(frame, target_shape[::-1])
            blurred_frame = cv2.GaussianBlur(resized_frame, (5, 5), 0)
            normalized_frame = blurred_frame.astype(np.float32) / 255.0
            preprocessed_frames.append(normalized_frame)

        video_cap.release()
        preprocessed_frames = np.expand_dims(preprocessed_frames, axis=-1)
        return np.array(preprocessed_frames), frame_count

    except Exception as e:
        print(f"Error preprocessing video: {e}")
        return None, None

def preprocess_video_resnet(video_path, target_shape=(224, 224)):
    try:
        video_cap = cv2.VideoCapture(video_path)
        frame_count = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        preprocessed_frames = []

        while True:
            success, frame = video_cap.read()
            if not success:
                break
            resized_frame = cv2.resize(frame, target_shape[::-1])
            normalized_frame = resized_frame.astype(np.float32) / 255.0
            preprocessed_frames.append(normalized_frame)

        video_cap.release()
        preprocessed_frames = np.array(preprocessed_frames)
        preprocessed_frames = preprocess_input(preprocessed_frames)
        return preprocessed_frames, frame_count

    except Exception as e:
        print(f"Error preprocessing video: {e}")
        return None, None

def preprocess_video_lstm(video_path, target_shape=(64, 64), num_frames=16):
    try:
        video_cap = cv2.VideoCapture(video_path)
        frame_rate = int(video_cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        target_frames = min(num_frames, frame_count)
        preprocessed_frames = []
        for _ in range(target_frames):
            success, frame = video_cap.read()
            if not success:
                break
            resized_frame = cv2.resize(frame, target_shape[::-1])
            blurred_frame = cv2.GaussianBlur(resized_frame, (5, 5), 0)
            normalized_frame = blurred_frame.astype(np.float32) / 255.0
            preprocessed_frames.append(normalized_frame)

        video_cap.release()
        while len(preprocessed_frames) < num_frames:
            preprocessed_frames.append(np.zeros(target_shape[::-1] + (3,)))

        preprocessed_frames = np.array(preprocessed_frames)
        preprocessed_frames = np.expand_dims(preprocessed_frames, axis=0)
        return preprocessed_frames, frame_count

    except Exception as e:
        print(f"Error preprocessing video: {e}")
        return None, None

def test_video(video_path):
    preprocessed_frames_cnn, num_frames_cnn = preprocess_video_cnn(video_path)
    cnn_predictions = cnn_model.predict(preprocessed_frames_cnn)
    cnn_accident = np.any(cnn_predictions > 0.5)
    if cnn_accident :
        result_label.config(text="Accident Detected", fg="red", font=("Helvetica", 16, "bold"))
    else:

        preprocessed_frames_resnet, num_frames_resnet = preprocess_video_resnet(video_path)
        resnet_features = resnet_model.predict(preprocessed_frames_resnet)
        preprocessed_frames_lstm, num_frames_lstm = preprocess_video_lstm(video_path)
        lstm_prediction = lstm_model.predict(preprocessed_frames_lstm)
        resnet_predictions = iso_forest_model.predict(resnet_features)
        
        lstm_accident = lstm_prediction > 0.5
        resnet_accident = -1 in resnet_predictions
        if (lstm_accident and resnet_accident):
            result_label.config(text="Accident Detected", fg="red", font=("Helvetica", 16, "bold"))
        else:
            result_label.config(text="No Accident Detected", fg="green", font=("Helvetica", 16, "bold"))


def select_file():
    canvas.delete("all")  # Clear the canvas
    label.config(text="Selected File: ")  # Clear the label text
    result_label.config(text="")  # Clear the result label text
    file_path = filedialog.askopenfilename()
    if file_path:
        update_label(file_path)
        test_video(file_path)
        play_video_loop(file_path)



root = tk.Tk()
root.title("Accident Detection")
root.geometry("800x650")
video_frame = tk.Frame(root, width=640, height=480, bg="black")
video_frame.pack(pady=20)
canvas = tk.Canvas(video_frame, width=640, height=480, bg="black")
canvas.pack()
label = tk.Label(root, text="Selected File: ", font=("Helvetica", 14))
label.pack()
result_label = tk.Label(root, text="", font=("Helvetica", 16, "bold"))
result_label.pack(pady=10)
button = tk.Button(root, text="Select Video File", font=("Helvetica", 14), command=select_file)
button.pack()
def update_label(file_path):
    label.config(text="Selected File: " + file_path)
    canvas.delete("all")  # Clear the canvas


def play_video(file_path):
    cap = cv2.VideoCapture(file_path)
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (640, 480))
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            imgtk = ImageTk.PhotoImage(image=img)
            canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
            root.update()
            imgtk.__del__()
            delay = int(1000 / frame_rate)
            root.after(delay)
        else:
            break

    cap.release()
def play_video_loop(file_path):
    while True:
        play_video(file_path)
root.mainloop()
