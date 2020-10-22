import os
from tqdm import tqdm
import pickle

import numpy as np
import torch
from torchvision.datasets import CIFAR10, MNIST
import torchvision.transforms as T

from config import ex

from data.corrupted_cifar10_protocol import CORRUPTED_CIFAR10_PROTOCOL
from data.colored_mnist_protocol import COLORED_MNIST_PROTOCOL

import cv2

def make_attr_labels(target_labels, bias_aligned_ratio):
    num_classes = target_labels.max().item() + 1
    num_samples_per_class = np.array(
        [
            torch.sum(target_labels == label).item()
            for label in range(num_classes)
        ]
    )
    ratios_per_class = bias_aligned_ratio * np.eye(num_classes) + (
        1 - bias_aligned_ratio
    ) / (num_classes - 1) * (1 - np.eye(num_classes))

    corruption_milestones_per_class = (
        num_samples_per_class[:, np.newaxis]
        * np.cumsum(ratios_per_class, axis=1)
    ).round()
    num_corruptions_per_class = np.concatenate(
        [
            corruption_milestones_per_class[:, 0, np.newaxis],
            np.diff(corruption_milestones_per_class, axis=1),
        ],
        axis=1,
    )

    attr_labels = torch.zeros_like(target_labels)
    for label in range(10):
        indices = (target_labels == label).nonzero().squeeze()
        corruption_milestones = corruption_milestones_per_class[label]
        for corruption_idx, idx in enumerate(indices):
            attr_labels[idx] = np.min(
                np.nonzero(corruption_milestones > corruption_idx)[0]
            ).item()

    return attr_labels


@ex.capture
def make_corrupted_cifar10(
    data_dir, skewed_ratio, corruption_names, severity, postfix="0"
    ):
    cifar10_dir = os.path.join(data_dir, "CIFAR10")
    corrupted_cifar10_dir = os.path.join(
        data_dir, f"CorruptedCIFAR10-Type{postfix}-Skewed{skewed_ratio}-Severity{severity}"
    )
    os.makedirs(corrupted_cifar10_dir, exist_ok=True)
    print(corrupted_cifar10_dir)
    protocol = CORRUPTED_CIFAR10_PROTOCOL
    convert_img = T.Compose([T.ToTensor(), T.ToPILImage()])

    attr_names = ["object", "corruption"]
    attr_names_path = os.path.join(corrupted_cifar10_dir, "attr_names.pkl")
    with open(attr_names_path, "wb") as f:
        pickle.dump(attr_names, f)

    for split in ["train", "valid"]:
        dataset = CIFAR10(cifar10_dir, train=(split == "train"), download=True)
        os.makedirs(os.path.join(corrupted_cifar10_dir, split), exist_ok=True)

        if split == "train":
            bias_aligned_ratio = 1-skewed_ratio
        else:
            bias_aligned_ratio = 0.1

        corruption_labels = make_attr_labels(
            torch.LongTensor(dataset.targets), bias_aligned_ratio
        )

        images, attrs = [], []
        for img, target_label, corruption_label in tqdm(
            zip(dataset.data, dataset.targets, corruption_labels),
            total=len(corruption_labels),
        ):
            
            method_name = corruption_names[corruption_label]
            corrupted_img = protocol[method_name](convert_img(img), severity+1)
            images.append(np.array(corrupted_img).astype(np.uint8))
            attrs.append([target_label, corruption_label])
                    
        image_path = os.path.join(corrupted_cifar10_dir, split, "images.npy")
        np.save(image_path, np.array(images).astype(np.uint8))
        attr_path = os.path.join(corrupted_cifar10_dir, split, "attrs.npy")
        np.save(attr_path, np.array(attrs).astype(np.uint8))


@ex.capture
def make_colored_mnist(data_dir, skewed_ratio, severity):
    mnist_dir = os.path.join(data_dir, "MNIST")
    colored_mnist_dir = os.path.join(
        data_dir, f"ColoredMNIST-Skewed{skewed_ratio}-Severity{severity}"
    )
    os.makedirs(colored_mnist_dir, exist_ok=True)
    print(colored_mnist_dir)
    protocol = COLORED_MNIST_PROTOCOL
    attr_names = ["digit", "color"]
    attr_names_path = os.path.join(colored_mnist_dir, "attr_names.pkl")
    with open(attr_names_path, "wb") as f:
        pickle.dump(attr_names, f)

    for split in ["train", "valid"]:
        dataset = MNIST(mnist_dir, train=(split == "train"), download=True)
        os.makedirs(os.path.join(colored_mnist_dir, split), exist_ok=True)

        if split == "train":
            bias_aligned_ratio = 1. - skewed_ratio
        else:
            bias_aligned_ratio = 0.1

        color_labels = make_attr_labels(
            torch.LongTensor(dataset.targets), bias_aligned_ratio
        )

        images, attrs = [], []
        for img, target_label, color_label in tqdm(
            zip(dataset.data, dataset.targets, color_labels),
            total=len(color_labels),
        ):
            colored_img = protocol[color_label.item()](img, severity)
            colored_img = np.moveaxis(np.uint8(colored_img), 0, 2)

            images.append(colored_img)
            attrs.append([target_label, color_label])
        
        colors_path = os.path.join("./data", "resource", "colors.th")
        mean_color = torch.load(colors_path)
        
        image_path = os.path.join(colored_mnist_dir, split, "images.npy")
        np.save(image_path, np.array(images).astype(np.uint8))
        attr_path = os.path.join(colored_mnist_dir, split, "attrs.npy")
        np.save(attr_path, np.array(attrs).astype(np.uint8))

    
@ex.automain
def make(make_target):

    for skewed_ratio in [5e-2, 2e-2, 1e-2, 5e-3]:

        for severity in [1, 2, 3, 4]:
            if make_target == "colored_mnist":
                make_colored_mnist(skewed_ratio=skewed_ratio, severity=severity)
    
            if make_target == "cifar10_type0":
                make_corrupted_cifar10(
                    corruption_names=[
                        "Snow",
                        "Frost",
                        "Fog",
                        "Brightness",
                        "Contrast",
                        "Spatter",
                        "Elastic",
                        "JPEG",
                        "Pixelate", 
                        "Saturate",
                    ],
                    skewed_ratio=skewed_ratio,
                    severity=severity,
                    postfix="0",
                )
            
            if make_target == "cifar10_type1":            
                make_corrupted_cifar10(
                    corruption_names=[
                        "Gaussian Noise",
                        "Shot Noise",
                        "Impulse Noise",
                        "Speckle Noise",
                        "Gaussian Blur",
                        "Defocus Blur",
                        "Glass Blur",
                        "Motion Blur",
                        "Zoom Blur",
                        "Original",
                    ],
                    skewed_ratio=skewed_ratio,
                    severity=severity,
                    postfix="1",
                )