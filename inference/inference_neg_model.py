import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18
from torchvision.datasets import ImageFolder
from torchvision.transforms import Resize, ToTensor
from torch.utils.data import DataLoader, Dataset
from glob import glob
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from PIL import Image

class NegativeDataset(Dataset):
    def __init__(self, image_ids, img_dir, labels=None):
        self.image_ids = image_ids
        self.labels = labels
        self.img_dir = img_dir
        self.transforms = Resize((224, 224)), ToTensor()

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_path = f"{self.img_dir}/{self.image_ids[idx]}"
        image = Image.open(img_path)
        for transform in self.transforms:
            image = transform(image)
        if self.labels is not None:
            return image, self.labels[idx]
        else:
            return image, self.image_ids[idx]

def inference(model, img_dir, test_csv):
    # Load data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_df = pd.read_csv(test_csv)

    test_ids = test_df['Image_ID'].unique().tolist()

    test_dataset = NegativeDataset(test_ids, img_dir)

    test_loader = DataLoader(test_dataset, batch_size=32)

    # Inference on test set
    print("Evaluating on test set")
    model.eval()
    test_results = {}
    with torch.no_grad():
        for images, ids in tqdm(test_loader, desc="Testing"):
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(1).tolist()
            for i, pred in enumerate(preds):
                test_results[ids[i]] = 'NEG' if pred else 'POS'

    return test_results

if __name__ == "__main__":
    from train_neg_model import train_model
    model = train_model('data/img', 'data/csv_files/Train.csv', 1)
    result = inference(model, 'data/img', 'data/csv_files/Test.csv')

    neg_resolutions = {}
    pos_resolutions = {}
    for id, pred in result.items():
        h, w = Image.open(f'data/img/{id}').size
        if pred == 'NEG':
            if (h, w) not in neg_resolutions:
                neg_resolutions[(h, w)] = 0
            neg_resolutions[(h, w)] += 1
        else:
            if (h, w) not in pos_resolutions:
                pos_resolutions[(h, w)] = 0
            pos_resolutions[(h, w)] += 1

    print(neg_resolutions)
    print(pos_resolutions)
