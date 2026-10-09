"""
train.py - Training Pipeline for Real-Time ASL Gesture Recognition

Loads Kaggle MediaPipe landmark CSV, applies wrist-relative and max-distance
normalization, encodes labels, trains a PyTorch MLP classifier, and saves:
1. models/asl_mlp_model.pth (Best model weights + architecture metadata)
2. models/label_encoder.pkl (Label encoder mapping)
"""

import argparse
import os
import pickle
from typing import Tuple, List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from src.utils import FEATURE_DIM, normalize_landmarks
from src.model import ASLMLP, save_checkpoint


class ASLLandmarkDataset(Dataset):
    """PyTorch Dataset wrapper for normalized hand landmark vectors."""

    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


def load_and_preprocess_data(
    csv_path: str,
) -> Tuple[np.ndarray, np.ndarray, LabelEncoder]:
    """
    Loads landmark CSV, extracts coordinates, applies normalization,
    and encodes string labels into integers.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Dataset file not found at: {csv_path}\n"
            f"Please place your Kaggle landmark CSV or run generate_dataset.py first."
        )

    print(f"[Data] Loading dataset from: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"[Data] Dataset shape: {df.shape}")

    # Identify label column
    label_col = None
    for candidate in ["label", "class", "target", "gesture", "Class"]:
        if candidate in df.columns:
            label_col = candidate
            break
    if label_col is None:
        # Default to last column
        label_col = df.columns[-1]

    # Coordinate columns (all except label column)
    feature_cols = [c for c in df.columns if c != label_col]
    if len(feature_cols) != FEATURE_DIM:
        print(
            f"[Warning] Found {len(feature_cols)} feature columns (expected {FEATURE_DIM}). "
            f"Using the first {FEATURE_DIM} columns."
        )
        feature_cols = feature_cols[:FEATURE_DIM]

    raw_features = df[feature_cols].values.astype(np.float32)
    raw_labels = df[label_col].astype(str).values

    print(f"[Preprocessing] Normalizing {len(raw_features)} landmark samples...")
    normalized_features = np.zeros_like(raw_features)
    for i in range(len(raw_features)):
        normalized_features[i] = normalize_landmarks(raw_features[i])

    # Encode labels
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(raw_labels)

    unique_classes = label_encoder.classes_
    print(f"[Preprocessing] Detected {len(unique_classes)} classes: {list(unique_classes)}")

    return normalized_features, encoded_labels, label_encoder


def train_model(
    model: ASLMLP,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int,
    lr: float,
    device: torch.device,
    save_path: str,
    class_names: List[str],
) -> dict:
    """
    Executes training loop with CrossEntropyLoss and Adam optimizer,
    tracking loss, accuracy, and saving the best checkpoint.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=4
    )

    best_val_acc = 0.0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    print(f"\n[Training] Starting {num_epochs} epochs on device: {device}")
    print("-" * 65)

    for epoch in range(1, num_epochs + 1):
        # --- Training Phase ---
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * batch_x.size(0)
            _, predicted = torch.max(outputs, 1)
            total_train += batch_y.size(0)
            correct_train += (predicted == batch_y).sum().item()

        epoch_train_loss = running_loss / total_train
        epoch_train_acc = correct_train / total_train

        # --- Validation Phase ---
        model.eval()
        val_running_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)

                val_running_loss += loss.item() * batch_x.size(0)
                _, predicted = torch.max(outputs, 1)
                total_val += batch_y.size(0)
                correct_val += (predicted == batch_y).sum().item()

        epoch_val_loss = val_running_loss / total_val
        epoch_val_acc = correct_val / total_val

        scheduler.step(epoch_val_acc)

        history["train_loss"].append(epoch_train_loss)
        history["train_acc"].append(epoch_train_acc)
        history["val_loss"].append(epoch_val_loss)
        history["val_acc"].append(epoch_val_acc)

        # Checkpoint if best accuracy
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            save_checkpoint(
                model,
                save_path,
                class_names=class_names,
                extra_metadata={"best_val_acc": best_val_acc, "epoch": epoch},
            )
            saved_indicator = " * [Saved Best]"
        else:
            saved_indicator = ""

        print(
            f"Epoch [{epoch:02d}/{num_epochs:02d}] | "
            f"Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc*100:5.1f}% | "
            f"Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc*100:5.1f}%"
            f"{saved_indicator}"
        )

    print("-" * 65)
    print(f"[Training Complete] Best Validation Accuracy: {best_val_acc*100:.2f}%")
    return history


def evaluate_final(
    model: ASLMLP,
    val_loader: DataLoader,
    label_encoder: LabelEncoder,
    device: torch.device,
) -> None:
    """Prints classification report on validation split."""
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(device)
            outputs = model(batch_x)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(batch_y.numpy())

    class_names = [str(c) for c in label_encoder.classes_]
    report = classification_report(
        all_targets, all_preds, target_names=class_names, zero_division=0
    )
    print("\n[Final Evaluation Report]")
    print(report)


def main():
    parser = argparse.ArgumentParser(
        description="Train ASL MLP Neural Network on Landmark Coordinates"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="dataset/asl_landmarks.csv",
        help="Path to Kaggle MediaPipe landmark CSV file",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default="models",
        help="Directory to save model weights and label encoder",
    )
    parser.add_argument(
        "--epochs", type=int, default=35, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="DataLoader batch size"
    )
    parser.add_argument(
        "--lr", type=float, default=0.001, help="Initial learning rate"
    )
    parser.add_argument(
        "--hidden1", type=int, default=128, help="First hidden layer size"
    )
    parser.add_argument(
        "--hidden2", type=int, default=64, help="Second hidden layer size"
    )
    parser.add_argument(
        "--dropout", type=float, default=0.2, help="Dropout probability"
    )
    args = parser.parse_args()

    # Create target directories
    os.makedirs(args.models_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.data_path) or ".", exist_ok=True)

    # If dataset missing, automatically generate starter dataset
    if not os.path.exists(args.data_path):
        print(f"[Dataset] No dataset found at {args.data_path}. Generating starter landmarks...")
        from generate_dataset import create_dataset_csv
        create_dataset_csv(output_path=args.data_path)

    # 1. Load and preprocess data
    X, y, label_encoder = load_and_preprocess_data(args.data_path)
    class_names = [str(c) for c in label_encoder.classes_]
    num_classes = len(class_names)

    # 2. Save Label Encoder mapping
    encoder_path = os.path.join(args.models_dir, "label_encoder.pkl")
    with open(encoder_path, "wb") as f:
        pickle.dump(label_encoder, f)
    print(f"[Saved] Label encoder saved to: {encoder_path}")

    # 3. Train/Val Split (80/20 stratified)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    train_dataset = ASLLandmarkDataset(X_train, y_train)
    val_dataset = ASLLandmarkDataset(X_val, y_val)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False
    )

    # 4. Initialize Network
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ASLMLP(
        num_classes=num_classes,
        input_dim=FEATURE_DIM,
        hidden_dim1=args.hidden1,
        hidden_dim2=args.hidden2,
        dropout_rate=args.dropout,
    ).to(device)

    # 5. Train Model
    weights_path = os.path.join(args.models_dir, "asl_mlp_model.pth")
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        lr=args.lr,
        device=device,
        save_path=weights_path,
        class_names=class_names,
    )

    # 6. Evaluate
    evaluate_final(model, val_loader, label_encoder, device)
    print(f"[Success] Model saved at: {weights_path}")


if __name__ == "__main__":
    main()
