"""
app.py - Real-Time ASL Gesture Recognition Web Dashboard

Combines OpenCV webcam video capture, MediaPipe Hands skeleton tracking,
PyTorch MLP landmark classification, and an interactive Streamlit UI
with live confidence metrics and a debounce-stabilized sentence builder.
"""

import os
import time
import pickle
from typing import Optional, Tuple, List

import cv2
import numpy as np
import streamlit as st
import torch

from src.utils import (
    extract_landmarks_from_mediapipe,
    draw_landmarks_on_image,
    FEATURE_DIM,
)
from src.model import ASLMLP, load_checkpoint

# ---------------------------------------------------------
# Streamlit Page Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="ASL Translator - Real-Time Gesture Recognition",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E88E5;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #616161;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        border-left: 5px solid #1E88E5;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        margin-bottom: 1rem;
    }
    .sentence-box {
        font-size: 1.6rem;
        font-weight: 600;
        background-color: #f1f8e9;
        border: 2px solid #81c784;
        border-radius: 8px;
        padding: 12px 18px;
        min-height: 60px;
        word-wrap: break-word;
        color: #1b5e20;
    }
    .pred-badge {
        font-size: 2.5rem;
        font-weight: 800;
        color: #0d47a1;
        text-align: center;
        margin: 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------
if "sentence" not in st.session_state:
    st.session_state.sentence = ""
if "last_stable_char" not in st.session_state:
    st.session_state.last_stable_char = None
if "char_hold_count" not in st.session_state:
    st.session_state.char_hold_count = 0
if "candidate_char" not in st.session_state:
    st.session_state.candidate_char = None


# ---------------------------------------------------------
# Model & Encoder Loader (Cached)
# ---------------------------------------------------------
@st.cache_resource
def load_recognition_pipeline(
    weights_path: str = "models/asl_mlp_model.pth",
    encoder_path: str = "models/label_encoder.pkl",
) -> Tuple[Optional[ASLMLP], Optional[List[str]], Optional[str]]:
    """Loads PyTorch model weights and label encoder."""
    if not os.path.exists(weights_path) or not os.path.exists(encoder_path):
        return None, None, f"Model files missing at '{weights_path}' or '{encoder_path}'."

    try:
        with open(encoder_path, "rb") as f:
            label_encoder = pickle.load(f)
        class_names = [str(c) for c in label_encoder.classes_]

        device = torch.device("cpu")
        model, _ = load_checkpoint(weights_path, device=device)
        return model, class_names, None
    except Exception as e:
        return None, None, str(e)


# ---------------------------------------------------------
# Sidebar Controls
# ---------------------------------------------------------
st.sidebar.image("https://img.icons8.com/color/96/sign-language.png", width=80)
st.sidebar.title("App Settings")

app_mode = st.sidebar.radio(
    "Select Mode",
    ["🔴 Live Camera Stream", "📸 Snapshot / Upload Test"],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.subheader("Detection Parameters")

confidence_threshold = st.sidebar.slider(
    "Confidence Threshold (%)",
    min_value=40,
    max_value=99,
    value=75,
    step=5,
) / 100.0

debounce_frames = st.sidebar.slider(
    "Sentence Debounce (Frames held)",
    min_value=5,
    max_value=30,
    value=12,
    help="Number of consecutive frames a gesture must be held to register into the sentence.",
)

show_landmarks = st.sidebar.checkbox("Show Hand Mesh & Landmarks", value=True)
camera_index = st.sidebar.number_input("Camera Device Index", min_value=0, max_value=5, value=0, step=1)

# ---------------------------------------------------------
# Main UI Layout
# ---------------------------------------------------------
st.markdown('<div class="main-header">🤟 Real-Time ASL Translation Engine</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">American Sign Language gesture recognition powered by MediaPipe, PyTorch, and OpenCV.</div>',
    unsafe_allow_html=True,
)

model, class_names, err_msg = load_recognition_pipeline()

if err_msg or model is None:
    st.error(f"⚠️ {err_msg}")
    st.info(
        "💡 **To train the model now**, run in your terminal:\n"
        "```bash\n"
        "python train.py --epochs 35\n"
        "```\n"
        "This will generate starter landmarks and train the weights to `models/asl_mlp_model.pth`."
    )
    if st.button("🚀 Train Model Now with Default Dataset"):
        with st.spinner("Training model in background..."):
            from train import main as run_train
            try:
                run_train()
                st.success("Model trained successfully! Please rerun or refresh the page.")
                st.cache_resource.clear()
            except Exception as ex:
                st.error(f"Training failed: {ex}")
    st.stop()


# Sentence building controls row
col_sent, col_actions = st.columns([3, 1])

with col_sent:
    st.markdown("**Constructed Sentence:**")
    st.markdown(
        f'<div class="sentence-box">{st.session_state.sentence if st.session_state.sentence else "<i>(Hold a sign steady to append characters...)</i>"}</div>',
        unsafe_allow_html=True,
    )

with col_actions:
    st.markdown("**Actions:**")
    btn_c1, btn_c2, btn_c3 = st.columns(3)
    with btn_c1:
        if st.button("␣ Space", use_container_width=True):
            st.session_state.sentence += " "
            st.rerun()
    with btn_c2:
        if st.button("⌫ Del", use_container_width=True):
            st.session_state.sentence = st.session_state.sentence[:-1]
            st.rerun()
    with btn_c3:
        if st.button("🗑 Clear", use_container_width=True):
            st.session_state.sentence = ""
            st.rerun()

st.markdown("---")

# ---------------------------------------------------------
# Video & Inference Stream Area
# ---------------------------------------------------------
col_video, col_metrics = st.columns([2, 1])

with col_metrics:
    st.subheader("Classification Metrics")
    pred_placeholder = st.empty()
    conf_placeholder = st.empty()
    top3_placeholder = st.empty()
    fps_placeholder = st.empty()

with col_video:
    video_placeholder = st.empty()

# ---------------------------------------------------------
# MediaPipe Hands Setup
# ---------------------------------------------------------
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


def update_sentence_buffer(predicted_char: str, conf: float) -> Optional[str]:
    """
    Stabilizes character detections across consecutive frames.
    Returns the character if confirmed, else None.
    """
    if conf < confidence_threshold:
        st.session_state.candidate_char = None
        st.session_state.char_hold_count = 0
        return None

    if predicted_char == st.session_state.candidate_char:
        st.session_state.char_hold_count += 1
    else:
        st.session_state.candidate_char = predicted_char
        st.session_state.char_hold_count = 1

    if st.session_state.char_hold_count == debounce_frames:
        if predicted_char == "SPACE":
            st.session_state.sentence += " "
        else:
            st.session_state.sentence += predicted_char
        st.session_state.last_stable_char = predicted_char
        return predicted_char

    return None


def run_model_inference(landmarks_vec: np.ndarray) -> Tuple[str, float, List[Tuple[str, float]]]:
    """Runs landmark vector through PyTorch model and retrieves predictions."""
    input_tensor = torch.tensor(landmarks_vec, dtype=torch.float32).unsqueeze(0)
    indices, probs = model.predict_top_k(input_tensor, k=3)

    top_idx = indices[0]
    top_char = class_names[top_idx]
    top_conf = probs[0]

    top_3_list = [(class_names[idx], p) for idx, p in zip(indices, probs)]
    return top_char, top_conf, top_3_list


# ---------------------------------------------------------
# Mode 1: Live Video Capture
# ---------------------------------------------------------
if app_mode == "🔴 Live Camera Stream":
    start_cam = st.sidebar.checkbox("Start Live Camera Stream", value=True)

    if start_cam:
        cap = cv2.VideoCapture(int(camera_index))

        if not cap.isOpened():
            st.error(f"Could not open camera device at index {camera_index}. Check camera permissions or switch to Snapshot mode.")
        else:
            hands = mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                min_detection_confidence=0.6,
                min_tracking_confidence=0.5,
            )

            prev_time = time.time()

            try:
                while start_cam:
                    ret, frame = cap.read()
                    if not ret:
                        video_placeholder.warning("Failed to grab camera frame. Reconnecting...")
                        time.sleep(0.1)
                        continue

                    # Mirror for natural user interaction
                    frame = cv2.flip(frame, 1)
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = hands.process(rgb_frame)

                    curr_time = time.time()
                    fps = 1.0 / (curr_time - prev_time + 1e-6)
                    prev_time = curr_time

                    detected_char = None
                    detected_conf = 0.0
                    top3_predictions = []

                    if results.multi_hand_landmarks:
                        for hand_landmarks in results.multi_hand_landmarks:
                            # Draw skeleton
                            if show_landmarks:
                                draw_landmarks_on_image(
                                    rgb_frame, hand_landmarks, mp_hands, mp_drawing, mp_drawing_styles
                                )

                            # Extract & normalize coordinates
                            norm_landmarks = extract_landmarks_from_mediapipe(hand_landmarks)
                            detected_char, detected_conf, top3_predictions = run_model_inference(
                                norm_landmarks
                            )

                            # Sentence builder update
                            new_letter = update_sentence_buffer(detected_char, detected_conf)

                            # Render overlay box on the camera frame
                            h, w, _ = frame.shape
                            x_coords = [int(lm.x * w) for lm in hand_landmarks.landmark]
                            y_coords = [int(lm.y * h) for lm in hand_landmarks.landmark]
                            x_min, x_max = max(0, min(x_coords) - 20), min(w, max(x_coords) + 20)
                            y_min, y_max = max(0, min(y_coords) - 20), min(h, max(y_coords) + 20)

                            box_color = (0, 255, 0) if detected_conf >= confidence_threshold else (255, 165, 0)
                            cv2.rectangle(rgb_frame, (x_min, y_min), (x_max, y_max), box_color, 2)
                            cv2.putText(
                                rgb_frame,
                                f"{detected_char} ({detected_conf*100:.1f}%)",
                                (x_min, max(30, y_min - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.8,
                                box_color,
                                2,
                            )
                    else:
                        st.session_state.char_hold_count = 0
                        st.session_state.candidate_char = None

                    # Update Streamlit UI components
                    video_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)

                    if detected_char and detected_conf >= confidence_threshold:
                        pred_placeholder.markdown(
                            f'<div class="metric-card"><p style="margin:0;font-size:0.9rem;color:#757575;">Current Sign</p>'
                            f'<p class="pred-badge">{detected_char}</p></div>',
                            unsafe_allow_html=True,
                        )
                        conf_placeholder.progress(
                            min(1.0, float(detected_conf)),
                            text=f"Confidence: {detected_conf*100:.1f}%",
                        )
                    else:
                        pred_placeholder.markdown(
                            '<div class="metric-card"><p style="margin:0;font-size:0.9rem;color:#757575;">Current Sign</p>'
                            '<p style="font-size:1.5rem;font-weight:600;color:#9e9e9e;text-align:center;">Waiting for gesture...</p></div>',
                            unsafe_allow_html=True,
                        )
                        conf_placeholder.progress(0.0, text="Confidence: 0.0%")

                    # Render top-3 list
                    if top3_predictions:
                        top3_html = "<b>Top Candidates:</b><br>"
                        for char, prob in top3_predictions:
                            top3_html += f"• <b>{char}</b>: {prob*100:4.1f}%<br>"
                        top3_placeholder.markdown(top3_html, unsafe_allow_html=True)
                    else:
                        top3_placeholder.empty()

                    fps_placeholder.caption(f"⚡ Feed Rate: {fps:.1f} FPS | Hold frames: {st.session_state.char_hold_count}/{debounce_frames}")

            finally:
                cap.release()
                hands.close()
    else:
        video_placeholder.info("Camera is stopped. Check 'Start Live Camera Stream' in the sidebar to activate.")


# ---------------------------------------------------------
# Mode 2: Snapshot / File Upload Test
# ---------------------------------------------------------
else:
    st.subheader("Test Sign Gesture with Image or Snapshot")
    img_file = st.file_uploader("Upload Hand Gesture Image", type=["jpg", "jpeg", "png"])
    cam_snap = st.camera_input("Or take a quick snapshot")

    target_image = cam_snap or img_file

    if target_image is not None:
        file_bytes = np.asarray(bytearray(target_image.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        hands = mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.5,
        )
        results = hands.process(rgb_img)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                if show_landmarks:
                    draw_landmarks_on_image(rgb_img, hand_landmarks, mp_hands, mp_drawing, mp_drawing_styles)

                norm_vec = extract_landmarks_from_mediapipe(hand_landmarks)
                char, conf, top3 = run_model_inference(norm_vec)

                st.success(f"Detected Sign: **{char}** with **{conf*100:.1f}%** confidence")
                for c, p in top3:
                    st.write(f"- {c}: {p*100:.1f}%")

                if st.button("Add to Sentence"):
                    st.session_state.sentence += " " if char == "SPACE" else char
                    st.rerun()
        else:
            st.warning("No hand landmarks detected in this image. Please ensure your hand is visible.")

        video_placeholder.image(rgb_img, channels="RGB", use_container_width=True)
        hands.close()
