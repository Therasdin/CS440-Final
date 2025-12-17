import os
import torch

from ImageLoader import get_dataloader
from ImageProcessor import ImageProcessor
from Model import (
    build_model,
    train,
    evaluate,
    classification_report,
    confusion_matrix,
)

# -----------------------------
# Configuration
# -----------------------------
DATA_DIR = "data"   # Root dataset folder with train/val/test subfolders

CLASS_NAMES = [
    "butterfly", "cat", "cow", "chicken", "dog",
    "elephant", "sheep", "spider", "squirrel", "horse"
]

NUM_CLASSES = len(CLASS_NAMES)

BATCH_SIZE = 32
IMG_SIZE = 224
EPOCHS_HEAD = 8
EPOCHS_FT = 8
LR_HEAD = 1e-3
LR_FT = 1e-5
CKPT_PATH = "animal_classifier.pt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    # -----------------------------
    # Data Loaders
    # -----------------------------
    print("Preparing data loaders...")

    processor = ImageProcessor(img_size=IMG_SIZE)

    train_loader = get_dataloader(
        os.path.join(DATA_DIR, "train"),
        batch_size=BATCH_SIZE,
        shuffle=True,
        transform=processor.get_augment_transform()
    )

    val_loader = get_dataloader(
        os.path.join(DATA_DIR, "val"),
        batch_size=BATCH_SIZE,
        shuffle=False,
        transform=processor.get_base_transform()
    )

    test_loader = get_dataloader(
        os.path.join(DATA_DIR, "test"),
        batch_size=BATCH_SIZE,
        shuffle=False,
        transform=processor.get_base_transform()
    )

    # -----------------------------
    # Model
    # -----------------------------
    print("Building model...")
    model = build_model(
        backbone_name="efficientnet_b0",
        num_classes=NUM_CLASSES,
        pretrained=True
    )

    # -----------------------------
    # Class Weights (optional)
    # -----------------------------
    print("Computing class weights...")
    class_counts = torch.tensor([
        len(os.listdir(os.path.join(DATA_DIR, "train", cls)))
        for cls in CLASS_NAMES
    ], dtype=torch.float32)

    class_weights = (class_counts.sum() / class_counts).clamp(max=5.0)

    # -----------------------------
    # Training
    # -----------------------------
    print("Starting training...")
    model, best_val_acc = train(
        model,
        train_loader,
        val_loader,
        epochs_head=EPOCHS_HEAD,
        epochs_ft=EPOCHS_FT,
        lr_head=LR_HEAD,
        lr_ft=LR_FT,
        optimizer_type="adam",
        class_weights=class_weights,
        ckpt_path=CKPT_PATH
    )

    # -----------------------------
    # Evaluation
    # -----------------------------
    print("Evaluating on test set...")
    test_loss, test_acc = evaluate(model, test_loader)
    print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}")

    print("\nClassification Report:")
    report = classification_report(model, test_loader, CLASS_NAMES)
    for cls, metrics in report.items():
        print(
            f"{cls:10s} | "
            f"Precision: {metrics['precision']:.3f} | "
            f"Recall: {metrics['recall']:.3f} | "
            f"F1: {metrics['f1']:.3f} | "
            f"Support: {metrics['support']}"
        )

    print("\nConfusion Matrix:")
    cm = confusion_matrix(model, test_loader, num_classes=NUM_CLASSES)
    print(cm)

    print("\nTraining complete.")
    print("Best validation accuracy:", best_val_acc)
    print(f"Model saved to {CKPT_PATH}")


# -----------------------------
# Entry Point
# -----------------------------
if __name__ == "__main__":
    main()
