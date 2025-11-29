import os
import time
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import models

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AnimalClassifier(nn.Module):
    def __init__(self, backbone_name="efficientnet_b0", num_classes=10, pretrained=True):
        """
        Supported backbones: 'efficientnet_b0', 'mobilenet_v2', 'vgg16'
        """
        super().__init__()
        self.backbone_name = backbone_name
        self.num_classes = num_classes

        if backbone_name == "efficientnet_b0":
            base = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
            self.feature_extractor = base.features
            in_feats = base.classifier[1].in_features
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
        elif backbone_name == "mobilenet_v2":
            base = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None)
            self.feature_extractor = base.features
            in_feats = 1280
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
        elif backbone_name == "vgg16":
            base = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1 if pretrained else None)
            self.feature_extractor = base.features
            in_feats = 512
            self.pool = nn.AdaptiveAvgPool2d((7, 7))
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        # Classification head (light MLP)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_feats, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_classes),
        )

        # Freeze backbone by default; training loop can unfreeze selectively
        for p in self.feature_extractor.parameters():
            p.requires_grad = False

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        return logits


def build_model(backbone_name="efficientnet_b0", num_classes=10, pretrained=True):
    model = AnimalClassifier(backbone_name=backbone_name, num_classes=num_classes, pretrained=pretrained)
    return model.to(DEVICE)


def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def set_backbone_trainable(model, train_ratio=0.3):
    """
    Unfreeze the last `train_ratio` portion of backbone layers for fine-tuning.
    """
    layers = list(model.feature_extractor.children())
    cutoff = int(len(layers) * (1 - train_ratio))
    for i, layer in enumerate(layers):
        for p in layer.parameters():
            p.requires_grad = (i >= cutoff)


def train(
    model,
    train_loader,
    val_loader,
    epochs_head=8,
    epochs_ft=8,
    lr_head=1e-3,
    lr_ft=1e-5,
    optimizer_type="adam",
    class_weights=None,
    log_interval=50,
    ckpt_path="animal_classifier.pt",
):
    """
    Two-stage training:
      1) Train head with frozen backbone
      2) Fine-tune last portion of backbone
    """
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE) if class_weights is not None else None)

    # Optimizer setup
    def make_optimizer(lr):
        if optimizer_type == "adam":
            return Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
        elif optimizer_type == "sgd":
            return SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, momentum=0.9, nesterov=True)
        else:
            raise ValueError("optimizer_type must be 'adam' or 'sgd'")

    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_acc = 0.0

    # Stage 1: Train head
    for p in model.feature_extractor.parameters():
        p.requires_grad = False
    optimizer = make_optimizer(lr_head)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    print(f"[Stage 1] Trainable params: {count_trainable_params(model)}")
    best_val_acc = _train_loop(model, train_loader, val_loader, epochs_head, criterion, optimizer, scheduler, log_interval)
    best_model_wts = copy.deepcopy(model.state_dict())

    # Stage 2: Fine-tune backbone tail
    set_backbone_trainable(model, train_ratio=0.3)
    optimizer = make_optimizer(lr_ft)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    print(f"[Stage 2] Trainable params: {count_trainable_params(model)}")

    val_acc = _train_loop(model, train_loader, val_loader, epochs_ft, criterion, optimizer, scheduler, log_interval)
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model_wts = copy.deepcopy(model.state_dict())

    # Save best
    torch.save(best_model_wts, ckpt_path)
    model.load_state_dict(best_model_wts)
    print(f"Saved best model to {ckpt_path} (val_acc={best_val_acc:.4f})")
    return model, best_val_acc


def _train_loop(model, train_loader, val_loader, epochs, criterion, optimizer, scheduler, log_interval):
    model = model.to(DEVICE)
    best_val_acc = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        start_t = time.time()

        for i, (images, labels) in enumerate(train_loader, 1):
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            if i % log_interval == 0:
                print(f"Epoch {epoch}/{epochs} | Batch {i} | Loss {loss.item():.4f}")

        train_loss = running_loss / total
        train_acc = correct / total

        val_loss, val_acc = evaluate(model, val_loader, criterion=criterion)
        scheduler.step(val_loss)

        dur = time.time() - start_t
        print(f"Epoch {epoch}/{epochs} | TrainLoss {train_loss:.4f} Acc {train_acc:.4f} | "
              f"ValLoss {val_loss:.4f} Acc {val_acc:.4f} | {dur:.1f}s")

        best_val_acc = max(best_val_acc, val_acc)

    return best_val_acc


@torch.no_grad()
def evaluate(model, data_loader, criterion=None):
    model.eval()
    total = 0
    correct = 0
    running_loss = 0.0

    for images, labels in data_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        logits = model(images)
        if criterion is not None:
            loss = criterion(logits, labels)
            running_loss += loss.item() * images.size(0)

        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    loss_avg = running_loss / total if criterion is not None else 0.0
    acc = correct / total
    return loss_avg, acc


@torch.no_grad()
def confusion_matrix(model, data_loader, num_classes):
    model.eval()
    cm = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    for images, labels in data_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        logits = model(images)
        preds = logits.argmax(dim=1)
        for t, p in zip(labels.view(-1), preds.view(-1)):
            cm[t.long(), p.long()] += 1
    return cm.cpu()


@torch.no_grad()
def classification_report(model, data_loader, class_names):
    """
    Returns per-class precision, recall, F1 dictionary.
    """
    num_classes = len(class_names)
    cm = confusion_matrix(model, data_loader, num_classes=num_classes).float()
    tp = torch.diag(cm)
    precision = tp / (cm.sum(0) + 1e-9)
    recall = tp / (cm.sum(1) + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)

    report = {}
    for i, name in enumerate(class_names):
        report[name] = {
            "precision": precision[i].item(),
            "recall": recall[i].item(),
            "f1": f1[i].item(),
            "support": cm.sum(1)[i].item(),
        }
    return report


def load_checkpoint(model_path, backbone_name="efficientnet_b0", num_classes=10):
    model = build_model(backbone_name=backbone_name, num_classes=num_classes, pretrained=False)
    state = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    return model


@torch.no_grad()
def predict(model, images, class_names=None, temperature=1.0):
    """
    images: Tensor [B,3,H,W]
    Returns: dict with 'probs', 'preds', and optional 'labels'
    """
    model.eval()
    images = images.to(DEVICE)
    logits = model(images) / max(temperature, 1e-6)
    probs = F.softmax(logits, dim=1)
    preds = probs.argmax(dim=1)
    result = {
        "probs": probs.cpu(),
        "preds": preds.cpu(),
        "labels": [class_names[i] for i in preds.cpu().tolist()] if class_names is not None else None,
    }
    return result
