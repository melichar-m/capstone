import numpy as np
import torch
import pandas as pd
import string
import os

from PIL import Image
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset, default_collate
from torchvision import transforms
from torch.nn.utils.rnn import pad_sequence

class CustomImageDataset(Dataset):
    # File locations
    csv_file = "/content/project-1-at-2023-11-03-00-00-1b95ace8.csv"
    root_dir = "/content/training"

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Create a character set including uppercase, lowercase, and digits
        all_chars = string.ascii_uppercase + string.ascii_lowercase + string.digits + " .,;'-!@*&%$()+=_-\"[]{}\|/?<>:\\\#$" # Total: 109 characters

        # Create the char_to_index dictionary with an additional index for the CTC blank token
        char_to_index = {char: idx + 1 for idx, char in enumerate(all_chars)}
        char_to_index[''] = 0  # 0 is the standard for CTC blank token

        img_name = os.path.join(self.root_dir, str(self.data_frame.iloc[idx, 0]) + ".png")
        image = Image.open(img_name)

        transcription = self.data_frame.iloc[idx, 1]
        encoded_transcription = [char_to_index[char] for char in transcription]

        if self.transform:
            image = self.transform(image)

        return image, encoded_transcription

def custom_collate_fn(batch):
    MAX_TRANSCRIPTION_LENGTH = 110
    images = [item[0] for item in batch]
    transcriptions = [item[1] for item in batch]
    padded_transcriptions = pad_sequence([torch.tensor(transcription) for transcription in transcriptions],
                                         batch_first=True,
                                         padding_value=0)
    if padded_transcriptions.size(1) < MAX_TRANSCRIPTION_LENGTH:
        padded_transcriptions = torch.cat([
            padded_transcriptions,
            torch.zeros(padded_transcriptions.size(0), MAX_TRANSCRIPTION_LENGTH - padded_transcriptions.size(1))
        ], dim=1)

    images = torch.stack(images, 0)  # Stack images along a new dimension
    return images, padded_transcriptions

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[1], std=[1]),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
])

training_dataset = CustomImageDataset(
    csv_file="/content/capstone-project.csv",
    root_dir="/content/training",
    transform=transform
)

validation_dataset = CustomImageDataset(
    csv_file="/content/capstone-project.csv",
    root_dir="/content/validation",
    transform=transform
)
def find_max_transcription_length(dataset):
    max_length = 0
    for i in range(len(dataset)):
        _, transcription = dataset[i]
        max_length = max(max_length, len(transcription))
    return max_length

# Use the function to find the max transcription length in the training dataset
# This is only necessary if your dataset varies in length. 
# This example uses all 109 upper, lower, and special characters in every image.
max_length_training = find_max_transcription_length(training_dataset)
print(f'Max transcription length in training dataset: {max_length_training}')

# Then you can create a DataLoader to batch and shuffle your data
data_loader = torch.utils.data.DataLoader(training_dataset, batch_size=16, shuffle=True, collate_fn=custom_collate_fn)
