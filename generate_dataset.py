"""
generate_dataset.py - Starter Landmark Dataset Generator for ASL Translation

Generates realistic 21-point hand landmark coordinates (63 coordinates) for common
ASL gestures (A, B, C, D, E, L, V, W, Y, SPACE) with natural variations, rotations,
and noise, producing a Kaggle-compatible CSV dataset at dataset/asl_landmarks.csv.
"""

import os
import numpy as np
import pandas as pd
from src.utils import LANDMARK_COUNT, FEATURE_DIM, normalize_landmarks

# Class definitions with hand pose signatures:
# For each finger [Thumb, Index, Middle, Ring, Pinky]:
# Extension state: 0 = folded/curled, 1 = extended, 0.5 = curved
BASE_GESTURES = {
    "A": {"thumb": 0.3, "index": 0.0, "middle": 0.0, "ring": 0.0, "pinky": 0.0},
    "B": {"thumb": 0.0, "index": 1.0, "middle": 1.0, "ring": 1.0, "pinky": 1.0},
    "C": {"thumb": 0.5, "index": 0.5, "middle": 0.5, "ring": 0.5, "pinky": 0.5},
    "D": {"thumb": 0.2, "index": 1.0, "middle": 0.2, "ring": 0.2, "pinky": 0.2},
    "E": {"thumb": 0.1, "index": 0.2, "middle": 0.2, "ring": 0.2, "pinky": 0.2},
    "L": {"thumb": 1.0, "index": 1.0, "middle": 0.0, "ring": 0.0, "pinky": 0.0},
    "V": {"thumb": 0.0, "index": 1.0, "middle": 1.0, "ring": 0.0, "pinky": 0.0},
    "W": {"thumb": 0.0, "index": 1.0, "middle": 1.0, "ring": 1.0, "pinky": 0.0},
    "Y": {"thumb": 1.0, "index": 0.0, "middle": 0.0, "ring": 0.0, "pinky": 1.0},
    "SPACE": {"thumb": 0.8, "index": 0.8, "middle": 0.8, "ring": 0.8, "pinky": 0.8},
}

def synthesize_hand_skeleton(pose_spec: dict, noise_std: float = 0.015) -> np.ndarray:
    """
    Synthesizes a 21x3 hand landmark skeleton matching MediaPipe coordinate scale.
    """
    landmarks = np.zeros((21, 3), dtype=np.float32)

    # Landmark 0: Wrist at base [0.5, 0.8, 0.0]
    wrist = np.array([0.5, 0.8, 0.0], dtype=np.float32)
    landmarks[0] = wrist

    # Base MCP (Metacarpophalangeal) joints x-offsets across palm
    mcp_x = [0.42, 0.46, 0.50, 0.54, 0.58]
    mcp_y = [0.72, 0.60, 0.58, 0.60, 0.63]

    fingers = ["thumb", "index", "middle", "ring", "pinky"]
    finger_indices = [
        [1, 2, 3, 4],       # Thumb: CMC, MCP, IP, TIP
        [5, 6, 7, 8],       # Index: MCP, PIP, DIP, TIP
        [9, 10, 11, 12],    # Middle: MCP, PIP, DIP, TIP
        [13, 14, 15, 16],   # Ring: MCP, PIP, DIP, TIP
        [17, 18, 19, 20],   # Pinky: MCP, PIP, DIP, TIP
    ]

    for f_idx, finger in enumerate(fingers):
        ext = pose_spec[finger]
        joint_ids = finger_indices[f_idx]
        base_x = mcp_x[f_idx]
        base_y = mcp_y[f_idx]

        if finger == "thumb":
            # Thumb moves diagonally outward
            for step, j_id in enumerate(joint_ids):
                factor = (step + 1) * 0.05
                if ext > 0.5:
                    # Extended thumb
                    landmarks[j_id] = [base_x - factor * 1.5, base_y - factor * 0.5, -factor * 0.2]
                else:
                    # Folded thumb
                    landmarks[j_id] = [base_x + factor * 0.4, base_y - factor * 0.3, factor * 0.3]
        else:
            # Four fingers extend upward (negative y) or curl downward/inward
            for step, j_id in enumerate(joint_ids):
                factor = (step + 1) * 0.06
                if ext >= 0.8:
                    # Fully extended
                    landmarks[j_id] = [base_x, base_y - factor * 1.2, -factor * 0.1]
                elif ext >= 0.4:
                    # Curved / C-shape
                    curve = np.sin((step + 1) * np.pi / 4) * 0.04
                    landmarks[j_id] = [base_x + curve, base_y - factor * 0.6, factor * 0.2]
                else:
                    # Curled / fist
                    landmarks[j_id] = [base_x, base_y + factor * 0.4, factor * 0.3]

    # Add subtle random variation & noise
    noise = np.random.normal(0, noise_std, landmarks.shape).astype(np.float32)
    landmarks += noise

    # Random global 3D rotation
    theta_z = np.random.uniform(-0.15, 0.15)
    cos_z, sin_z = np.cos(theta_z), np.sin(theta_z)
    rot_z = np.array([
        [cos_z, -sin_z, 0],
        [sin_z,  cos_z, 0],
        [0,      0,     1]
    ], dtype=np.float32)
    landmarks = np.dot(landmarks - wrist, rot_z) + wrist

    return landmarks


def create_dataset_csv(
    output_path: str = "dataset/asl_landmarks.csv",
    samples_per_class: int = 120,
) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    header = []
    for i in range(LANDMARK_COUNT):
        header.extend([f"x_{i}", f"y_{i}", f"z_{i}"])
    header.append("label")

    rows = []
    for label, pose_spec in BASE_GESTURES.items():
        for _ in range(samples_per_class):
            raw_landmarks = synthesize_hand_skeleton(pose_spec)
            # Use raw coordinates in the CSV to emulate standard Kaggle raw landmarks
            coords_flat = raw_landmarks.flatten().tolist()
            coords_flat.append(label)
            rows.append(coords_flat)

    df = pd.DataFrame(rows, columns=header)
    # Shuffle dataset
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    df.to_csv(output_path, index=False)
    print(f"[Dataset] Generated {len(df)} samples across {len(BASE_GESTURES)} classes at {output_path}")


if __name__ == "__main__":
    create_dataset_csv()
