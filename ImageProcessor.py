from torchvision import transforms

class ImageProcessor:
    def __init__(self, img_size=224):
        self.img_size = img_size

        # Basic preprocessing (resize, normalize)
        self.base_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])  # ImageNet stats
        ])

        # Augmentation pipeline
        self.augment_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def get_base_transform(self):
        return self.base_transform

    def get_augment_transform(self):
        return self.augment_transform
