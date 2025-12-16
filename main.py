import os
import random
import shutil
import torch
from PIL import Image

# reuse existing modules
from ImageProcessor import ImageProcessor
from Model import load_checkpoint
import train


# ======================================================
# DATASET PREPARATION (Animals-10 → train/val/test)
# ======================================================
def prepare_data(
    source_dir,
    target_dir="data",
    split=(0.8, 0.1, 0.1),
    seed=42
):
    if os.path.exists(target_dir):
        print(f"[INFO] '{target_dir}' already exists. Skipping dataset split.")
        return

    print("[INFO] Preparing dataset splits...")
    random.seed(seed)

    for split_name in ["train", "val", "test"]:
        os.makedirs(os.path.join(target_dir, split_name), exist_ok=True)

    class_names = sorted([
        d for d in os.listdir(source_dir)
        if os.path.isdir(os.path.join(source_dir, d))
    ])

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
# MAIN ORCHESTRATOR
# ======================================================
def main():
    ANIMALS10_DIR = "Animals-10"
    DATA_DIR = "data"

    CLASS_NAMES = [
        "butterfly", "cat", "cow", "chicken", "dog",
        "elephant", "sheep", "spider", "squirrel", "horse"
    ]

    PET_CLASSES = {"dog", "cat", "cow", "chicken", "sheep", "horse"}
    CKPT_PATH = "animal_classifier.pt"

    # -----------------------------
    # Prepare dataset
    # -----------------------------
    prepare_data(
        source_dir=ANIMALS10_DIR,
        target_dir=DATA_DIR
    )

    # -----------------------------
    # Train + evaluate (reused)
    # -----------------------------
    print("\n[INFO] Running training pipeline...")
    # This executes train.py top-to-bottom
    # and saves animal_classifier.pt
    import importlib
    importlib.reload(train)

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
    # Camera demo
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

    print("\n[INFO] Project pipeline complete.")


# ======================================================
# ENTRY POINT
# ======================================================
if __name__ == "__main__":
    main()
