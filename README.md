# 🤟 Real-Time ASL Translation App

An end-to-end, modular American Sign Language (ASL) gesture recognition web application built with **MediaPipe Hands**, **PyTorch**, **OpenCV**, and **Streamlit**.

---

## 📌 Architecture Overview

```
asl-translation-app/
├── dataset/
│   └── asl_landmarks.csv         # Kaggle CSV dataset (63 coordinates + label)
├── models/
│   ├── asl_mlp_model.pth         # PyTorch trained model weights & checkpoint
│   └── label_encoder.pkl         # Saved scikit-learn LabelEncoder mapping
├── src/                          # Modular core package
│   ├── __init__.py               # Exports and package markers
│   ├── utils.py                  # Landmark normalization & MediaPipe helpers
│   └── model.py                  # PyTorch MLP Neural Network architecture
├── generate_dataset.py           # Starter landmark generator & augmentor
├── train.py                      # Data pipeline, training loop & validation
├── app.py                        # Streamlit Web UI + Real-time OpenCV/MediaPipe Engine
├── requirements.txt              # Project dependencies
└── README.md                     # Documentation & execution guide
```

---

## 🧠 Landmark Preprocessing & Normalization

Raw hand landmarks captured from MediaPipe Hands or Kaggle CSV datasets vary significantly depending on hand size and distance from the camera. This app applies **Wrist-Relative Scale Normalization** in `src/utils.py`:

1. **Translation**: All 21 3D landmarks are translated relative to the wrist (landmark `0`):
   $$\mathbf{p}_i^{\text{rel}} = \mathbf{p}_i - \mathbf{p}_0 \quad \text{for } i \in [0, 20]$$
   After this step, the wrist is always at $(0, 0, 0)$.

2. **Scale Invariance**: The Euclidean distance from the wrist is calculated for all landmarks, and coordinates are divided by the maximum distance:
   $$d_{\max} = \max_{i} \|\mathbf{p}_i^{\text{rel}}\|_2$$
   $$\mathbf{p}_i^{\text{norm}} = \frac{\mathbf{p}_i^{\text{rel}}}{d_{\max} + \epsilon}$$

3. **Feature Vector**: The 21 normalized points are flattened into a consistent **63-dimensional** 1D vector (`(x0, y0, z0, ..., x20, y20, z20)`).

---

## 🚀 Quickstart Guide

### 1. Installation

Create a virtual environment and install the required dependencies:

```bash
# Optional: create a virtual environment
python -m venv venv
venv\Scripts\activate   # Windows (or: source venv/bin/activate on Linux/macOS)

# Install requirements
pip install -r requirements.txt
```

### 2. Dataset Preparation

You have two options for the dataset:

- **Option A (Instant Starter Dataset)**: Run the included dataset generator:
  ```bash
  python generate_dataset.py
  ```
  This creates `dataset/asl_landmarks.csv` containing realistic 21-point hand landmark coordinates across ASL gestures (`A`, `B`, `C`, `D`, `E`, `L`, `V`, `W`, `Y`, `SPACE`).

- **Option B (Kaggle Dataset)**: Place any MediaPipe landmark CSV into `dataset/asl_landmarks.csv`. The script automatically detects coordinate columns (63 coordinates) and class label columns (`label`, `class`, etc.).

### 3. Model Training

Train the PyTorch MLP classifier using `train.py`:

```bash
python train.py --epochs 35 --batch-size 32 --lr 0.001
```

**Training Highlights:**
- Normalizes features on the fly.
- Encodes labels and saves `models/label_encoder.pkl`.
- Automatically tracks validation accuracy with `ReduceLROnPlateau` scheduler.
- Checkpoints the best model to `models/asl_mlp_model.pth`.
- Prints a complete classification report (Precision, Recall, F1-Score).

### 4. Running the Web Application

Launch the Streamlit interactive dashboard:

```bash
streamlit run app.py
```

Open the local browser URL (typically `http://localhost:8501`).

---

## ✨ Web App Features

- **🔴 Live Video Inference**: Real-time hand landmark extraction using OpenCV and MediaPipe at 30+ FPS.
- **🎯 Visual Feedback**: Real-time bounding box, landmark skeleton mesh overlay, and confidence score.
- **📊 Metric Dashboard**: Top-1 predicted character card with dynamic confidence bar and top-3 ranked candidate probabilities.
- **🔤 Debounced Sentence Builder**:
  - Automatically holds a stable buffer: only appends a sign to the sentence once sustained for $N$ consecutive frames (configurable slider).
  - Handles `SPACE`, `Backspace`, and `Clear`.
- **📸 Snapshot / Image Upload Mode**: Fallback mode for devices without an active webcam or remote environments.

---

## ⚙️ Model Hyperparameters

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `input_dim` | 63 | 21 hand landmarks $\times$ 3 coordinates |
| `hidden_dim1` | 128 | First dense layer with BatchNorm & ReLU |
| `hidden_dim2` | 64 | Second dense layer with BatchNorm & ReLU |
| `dropout` | 0.2 | Dropout probability to prevent overfitting |
| `optimizer` | Adam | Learning rate: 0.001, Weight decay: 1e-4 |
| `scheduler` | ReduceLROnPlateau | Factor: 0.5, Patience: 4 |

---

## 🛠️ Troubleshooting

- **Webcam Access Denied / Black Screen**:
  - Ensure other applications (Zoom, Teams, etc.) are not using the webcam.
  - Adjust the `Camera Device Index` in the Streamlit sidebar (e.g. from `0` to `1`).
  - Or switch to `📸 Snapshot / Upload Test` mode.
- **Weights Not Found Error**:
  - Run `python train.py` once to generate `models/asl_mlp_model.pth` and `models/label_encoder.pkl`.
