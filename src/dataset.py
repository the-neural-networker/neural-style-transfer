import os 

import torch 
from torch import nn 
from torch.utils.data import Dataset 

from torchvision import transforms 
from PIL import Image

from typing import Tuple, Union

class StyleTransferDataset(Dataset):
    def __init__(self, image_dir: str="../images/", 
                transform: transforms.Compose=transforms.Compose([ 
                           transforms.Resize((512, 512)),
                           transforms.ToTensor()
                           ])) -> None: 
        self.image_dir = image_dir
        self.transform = transform
        self.content_dir = os.path.join(self.image_dir, "content")
        self.style_dir = os.path.join(self.image_dir, "style")
        self.content_list = [os.path.join(self.content_dir, d) for d in sorted(os.listdir(self.content_dir))]
        self.style_list = [os.path.join(self.style_dir, d) for d in sorted(os.listdir(self.style_dir))]

        # one content and one style image only
        assert len(self.content_list) == 1, f"Should be 1 content image only, got {len(self.content_list)}"
        assert len(self.style_list) == 1, f"Should be 1 style image only, got {len(self.style_list)}"

    def __len__(self) -> str:
        return len(self.content_list)

    def __getitem__(self, idx: Union[int, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        if torch.is_tensor(idx):
            idx = idx.tolist()

        self.content = Image.open(self.content_list[idx]).convert("RGB")
        self.style = Image.open(self.style_list[idx]).convert("RGB")

        if self.transform:
            self.content = self.transform(self.content)
            self.style = self.transform(self.style)

        return self.content, self.style



        

