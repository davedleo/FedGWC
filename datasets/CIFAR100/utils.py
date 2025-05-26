import json
import random
import torch
import torch.nn as nn
import torchvision.transforms as transforms 
from torch.utils.data import Dataset 
from PIL import Image, ImageFilter





# GLOBAL VARIABLES 
DIR_PATH = "./datasets/CIFAR100/data/"





# DATASET 
class ImagesDataset(Dataset):



    def __init__(self, data: dict, train: bool = True, anomaly_type: str = None, anomaly_level: int = -1):
        super(ImagesDataset, self).__init__()

        # Store arguments and dataset metadata
        self._data = data  # Store the data dict internally
        self._train = train
        self._anomaly_type = anomaly_type
        self._anomaly_level = anomaly_level

        # Attributes to be filled when load_data is called
        self._images = None
        self._labels = data["y"]
        self._transforms = None

        # Noise setup
        self._add_noise = anomaly_type == "noise"
        self._noise_sigma = anomaly_level if self._add_noise else None

        # Number of images initially set to 0
        self._num_images = len(self._labels)



    def load_data(self):
        """Load images and set up transformations."""
        path = DIR_PATH + "raw/train/" if self._train else DIR_PATH + "raw/test/"
        self._images = []
        self._labels = []

        # Load images into memory
        for image_name, label in zip(self._data["x"], self._data["y"]):
            image = Image.open(path + image_name).convert("RGB")
            image.load()

            # Apply anomaly perturbations
            if self._anomaly_type:
                if self._anomaly_type == "blur":
                    image = image.filter(ImageFilter.GaussianBlur(radius=self._anomaly_level))

            self._images.append(image)
            self._labels.append(label)

        # Define transformations
        if self._train:
            self._transforms = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
            ])
        else:
            self._transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
            ])
        
        # Cache the number of images
        self._num_images = len(self._labels)



    def clear(self):
        """Clear cached images to free memory."""
        self._images = None
        self._labels = None



    def __len__(self):
        return self._num_images 



    def __getitem__(self, index: int):
        if self._images is None or self._labels is None:
            raise RuntimeError("Data has not been loaded. Call 'load_data' before accessing elements.")
        
        if self._add_noise:
            gaussian_noise = torch.randn(3, 32, 32) * self._noise_sigma
            return self._transforms(self._images[index]) + gaussian_noise, self._labels[index]
        else:
            return self._transforms(self._images[index]), self._labels[index]




# JSON LOADER 
def load_split_from_json(filename: str):

    # Open file 
    with open(DIR_PATH + "partitions/" + filename, "r") as f:
        d = json.load(f)

    # Extract the distillated dataset
    server_imgs = d.pop("server_imgs")

    # Extract and shuffle shards of the partition
    d_list = [it for it in d.items()]
    random.shuffle(d_list)

    # Return data
    return d_list, server_imgs





# FEDERATED SETUP 
def load_federated_dataset(
        alpha_dirichlet: str = "1000.00",
        num_noise_clients: int = 10,
        num_blur_clients: int = 10,
        noise_sigma: float = -1.,
        blur_radius: int = -1
): 
    
    # Load partition
    partition, server_imgs = load_split_from_json("dataset_alpha_" + alpha_dirichlet + ".json")

    # Anomalies setup
    noise_flag = num_noise_clients 
    blur_flag = noise_flag + num_blur_clients 

    # Datasets setup 
    datasets = dict()

    # Iterate shards
    # - Noise images 
    for shard in partition[:noise_flag]: 

        # Extract key and value 
        shard_id, data = shard 

        # Store data in the dict as a dataset 
        datasets[shard_id] = {
            "augmentation": "noise",
            "train": ImagesDataset(data["train"], train = True, anomaly_type = "noise", anomaly_level = noise_sigma),
            "test": ImagesDataset(data["test"], train = False, anomaly_type = "noise", anomaly_level = noise_sigma)
        }

    # - Blur images
    for shard in partition[noise_flag: blur_flag]: 

        # Extract key and value 
        shard_id, data = shard 

        # Store data in the dict as a dataset 
        datasets[shard_id] = {
            "augmentation": "blur",
            "train": ImagesDataset(data["train"], train = True, anomaly_type = "blur", anomaly_level = blur_radius),
            "test": ImagesDataset(data["test"], train = False, anomaly_type = "blur", anomaly_level = blur_radius)
        }

    # - Clean images
    for shard in partition[blur_flag:]:

        # Extract key and value 
        shard_id, data = shard 

        # Store data in the dict as a dataset 
        datasets[shard_id] = {
            "augmentation": "clean",
            "train": ImagesDataset(data["train"], train = True),
            "test": ImagesDataset(data["test"], train = False)
        }

    return {"clients": datasets, "server": ImagesDataset(server_imgs, train = False)}





# MODEL CLASS 
class CNN(nn.Module): 



    def __init__(self): 
        super(CNN, self).__init__()

        # Feature Map
        # - block 1
        self._fmap1 = nn.Sequential(
            nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size = 5, stride = 1),
            nn.MaxPool2d(kernel_size = 2, stride = 2), 
            nn.ReLU()
        )
        # - block 2
        self._fmap2 = nn.Sequential(
            nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 5, stride = 1),
            nn.MaxPool2d(kernel_size = 2, stride = 2), 
            nn.ReLU()
        )

        # Classifier
        self._mlp = nn.Sequential(
            nn.Linear(1600, 384), 
            nn.ReLU(),
            nn.Linear(384, 192),
            nn.ReLU(),
            nn.Linear(192, 100)
        )



    def forward(self, X): 
        
        # Feature map 
        out = self._fmap1(X)
        out = self._fmap2(out)

        # Classifer
        return self._mlp(out.view(out.size(0), -1))






# MODEL LOADER  
def load_model(device: str = "cpu"): 
    # return CNN().to(device)
    return CNN().to(device)