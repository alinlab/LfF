import os
import pickle
import torch
import numpy as np
from torch.utils.data import Dataset

class AttributeDataset(Dataset):
    def __init__(self, root, split, query_attr_idx=None, transform=None):
        super(AttributeDataset, self).__init__()
        data_path = os.path.join(root, split, "images.npy")
        self.data = np.load(data_path)
        
        attr_path = os.path.join(root, split, "attrs.npy")
        self.attr = torch.LongTensor(np.load(attr_path))

        colors_path = os.path.join("./data", "resource", "colors.th")
        mean_color = torch.load(colors_path)
        attr_names_path = os.path.join(root, "attr_names.pkl")
        with open(attr_names_path, "rb") as f:
            self.attr_names = pickle.load(f)
        
        self.num_attrs =  self.attr.size(1)
        self.set_query_attr_idx(query_attr_idx)
        self.transform = transform
    
    def set_query_attr_idx(self, query_attr_idx):
        if query_attr_idx is None:
            query_attr_idx = torch.arange(self.num_attrs)
        
        self.query_attr = self.attr[:, query_attr_idx]
        
    def __len__(self):
        return self.attr.size(0)

    def __getitem__(self, index):
        image, attr = self.data[index], self.query_attr[index]
        if self.transform is not None:
            image = self.transform(image)

        return image, attr