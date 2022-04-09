from PIL import Image
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import faster_rcnn.transforms as T
import os


class PennFudanDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.transforms = T.Compose([T.ToTensor()])
        self.imgs = list(sorted(os.listdir(os.path.join(root_dir, "PNGImages"))))
        self.masks = list(sorted(os.listdir(os.path.join(root_dir, "PedMasks"))))

    def __len__(self):
        return len(self.imgs)

    def num_of_samples(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        # load images and bbox
        img_path = os.path.join(self.root_dir, "PNGImages", self.imgs[idx])
        mask_path = os.path.join(self.root_dir, "PedMasks", self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)
        # convert the PIL Image into a numpy array
        mask = np.array(mask)
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None]

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target


if __name__ == "__main__":
    ds = PennFudanDataset("D:/pytorch/PennFudanPed")
    for i in range(len(ds)):
        img, target = ds[i]
        print(i, img.size(), target)
        device = torch.device('cuda:0')
        boxes = target["boxes"]
        xmin, ymin, xmax, ymax = boxes.unbind(1)

        targets = [{k: v.to(device) for k, v in t.items()} for t in [target]]
        if i == 3:
            break