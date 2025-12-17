import os
import random
import shutil
import torch
from PIL import Image

import Train
from ImageProcessor import ImageProcessor
from Model import load_checkpoint, predict


# ======================================================
# DATASET PREPARATION (Animals-10 → train/val/test)
# ======================================================
def prepare_data(
    source_dir,
    target_dir="data",
    split=(0.8, 0.1, 0.1),
    seed=42
):
    """     ````````````````````
    Converts Animals-10 folder structure into train/val/test.
    Runs once and skips if already prepared.
    """
    if os.path.exists(target_dir):
        print("[INFO] 'data' already exists. Skipping dataset split.")
        return

    print("[INFO] Preparing dataset splits...")
    random.seed(seed)

    for split_name in ["train", "val", "test"]:
        os.makedirs(os.path.join(target_dir, split_name), exist_ok=True)

    class_names = sorted(
        d for d in os.listdir(source_dir)
        if os.path.isdir(os.path.join(source_dir, d))
    )

    for cls in class_names:
        cls_path = os.path.join(source_dir, cls)
        images = [
            f for f in os.listdir(cls_path)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        random.shuffle(images)
        n = len(images)
        n_train = int(split[0] * n)
        n_val = int(split[1] * n)

        splits = {
            "train": images[:n_train],
            "val": images[n_train:n_train + n_val],
            "test": images[n_train + n_val:]
        }

        for split_name, imgs in splits.items():
            split_cls_dir = os.path.join(target_dir, split_name, cls)
            os.makedirs(split_cls_dir, exist_ok=True)
            for img in imgs:
                shutil.copy2(
                    os.path.join(cls_path, img),
                    os.path.join(split_cls_dir, img)
                )

    print("[INFO] Dataset preparation complete.")


# ======================================================
# PET vs PEST CAMERA SIMULATION
# ======================================================
@torch.no_grad()
def run_pet_vs_pest_camera(
    model,
    test_dir,
    class_names,
    pet_classes,
    processor,
    num_samples=6
):
    model.eval()
    device = next(model.parameters()).device

    all_images = []
    for cls in class_names:
        cls_dir = os.path.join(test_dir, cls)
        if not os.path.exists(cls_dir):
            continue
        for fname in os.listdir(cls_dir):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                all_images.append((cls, os.path.join(cls_dir, fname)))

    sampled = random.sample(all_images, min(num_samples, len(all_images)))

    print("\n================ CAMERA: PET vs PEST =================\n")

    for true_cls, img_path in sampled:
        img = Image.open(img_path).convert("RGB")
        tensor = processor.get_base_transform()(img).unsqueeze(0).to(device)

        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]
        pred_idx = probs.argmax().item()
        pred_cls = class_names[pred_idx]
        confidence = probs[pred_idx].item() * 100

        decision = "PET" if pred_cls in pet_classes else "PEST"
        action = "IGNORE" if decision == "PET" else "ALERT"

        print(f"[CAMERA] Image: {os.path.basename(img_path)}")
        print(f"  True animal:      {true_cls}")
        print(f"  Predicted animal: {pred_cls} ({confidence:.1f}%)")
        print(f"  Decision:         {decision} → {action}")
        print("-" * 60)


# ======================================================
# MAIN PIPELINE
# ======================================================
def main():
    ANIMALS10_DIR = "Animals-10"
    DATA_DIR = "data"
    CKPT_PATH = "animal_classifier.pt"

    CLASS_NAMES = [
        "butterfly", "cat", "cow", "chicken", "dog",
        "elephant", "sheep", "spider", "squirrel", "horse"
    ]

    PET_CLASSES = {"dog", "cat", "cow", "chicken", "sheep", "horse"}

    # -----------------------------
    # Prepare dataset
    # -----------------------------
    prepare_data(ANIMALS10_DIR, DATA_DIR)

    # -----------------------------
    # Train model (delegated to Train.py)
    # -----------------------------
    print("\n[INFO] Running training pipeline...")
    Train.main()

    # -----------------------------
    # Load trained model
    # -----------------------------
    print("\n[INFO] Loading trained model...")
    model = load_checkpoint(
        model_path=CKPT_PATH,
        backbone_name="efficientnet_b0",
        num_classes=len(CLASS_NAMES)
    )

    # -----------------------------
    # Camera simulation
    # -----------------------------
    processor = ImageProcessor(img_size=224)

    run_pet_vs_pest_camera(
        model=model,
        test_dir=os.path.join(DATA_DIR, "test"),
        class_names=CLASS_NAMES,
        pet_classes=PET_CLASSES,
        processor=processor,
        num_samples=6
    )

    print("\n[INFO] Pipeline complete.")

def predict_single_image(
    image_path,
    model_path="animal_classifier.pt",
    class_names=None
):
    processor = ImageProcessor(img_size=224)

    # Load model
    model = load_checkpoint(
        model_path=model_path,
        backbone_name="efficientnet_b0",
        num_classes=len(class_names)
    )

    # Load + preprocess image
    img = Image.open(image_path).convert("RGB")
    tensor = processor.get_base_transform()(img).unsqueeze(0)

    # Predict
    result = predict(model, tensor, class_names=class_names)

    pred_idx = result["preds"].item()
    pred_label = result["labels"][0]
    confidence = result["probs"][0][pred_idx].item() * 100

    print("\n=== SINGLE IMAGE PREDICTION ===")
    print(f"Image: {image_path}")
    print(f"Prediction: {pred_label}")
    print(f"Confidence: {confidence:.2f}%")


# ======================================================
# ENTRY POINT
# ======================================================
if __name__ == "__main__":
    CLASS_NAMES = [
        "butterfly", "cat", "cow", "chicken", "dog",
        "elephant", "sheep", "spider", "squirrel", "horse"
    ]

    print("\nChoose an option:")
    print("1 - Train / Retrain model")
    print("2 - Classify a single image")
    choice = input("Enter 1 or 2: ").strip()

    if choice == "1":
        print("\n[INFO] Training model...")
        main()

    elif choice == "2":
        image_path = input("\nEnter path to image: ").strip()

        if not os.path.exists(image_path):
            print("[ERROR] Image file not found.")
        else:
            predict_single_image(
                image_path=image_path,
                class_names=CLASS_NAMES
            )

    else:
        print("[ERROR] Invalid choice. Please run again and enter 1 or 2.")
