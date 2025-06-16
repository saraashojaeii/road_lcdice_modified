from torch.utils.data import Dataset
from torchvision import transforms as T
from PIL import Image

class DataPrep(Dataset):
    def __init__(self, image_paths, label_paths, transform=None):
        self.image_paths = image_paths
        self.label1_paths = label_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load and transform image
        image = Image.open(self.image_paths[idx]).convert('RGB')
        mask = Image.open(self.label1_paths[idx]).convert('L')
        image = image.resize((512, 512))
        mask = mask.resize((512, 512))

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask

# Define a simple transform
transform = T.Compose([
    T.ToTensor(),  # Automatically converts PIL images to tensors and scales to [0, 1]
])

