import os
import json
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

class LettuceDetectionDataset(Dataset):
    def __init__(self, root_dir, annotation_file):
        self.root_dir = root_dir
        with open(annotation_file, 'r') as f:
            self.coco_data = json.load(f)
            
        self.images = self.coco_data['images']
        self.annotations = self.coco_data['annotations']
        
        # 448x448 prevents Out of Memory errors on the 16GB Colab T4 GPU
        self.img_size = 448 
        self.transforms = T.Compose([
            T.Resize((self.img_size, self.img_size)),
            T.ToTensor(),
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_info = self.images[idx]
        img_id = img_info['id']
        img_path = os.path.join(self.root_dir, img_info['file_name'])
        
        image = Image.open(img_path).convert("RGB")
        orig_w, orig_h = image.size
        
        img_anns = [ann for ann in self.annotations if ann['image_id'] == img_id]
        
        boxes = []
        labels = []
        
        for ann in img_anns:
            x, y, w, h = ann['bbox']
            
            xmin = (x / orig_w) * self.img_size
            ymin = (y / orig_h) * self.img_size
            xmax = ((x + w) / orig_w) * self.img_size
            ymax = ((y + h) / orig_h) * self.img_size
            
            if xmax > xmin and ymax > ymin:
                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(ann['category_id'])
        
        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            
        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([img_id])
        }
        
        image = self.transforms(image)
        return image, target

def collate_fn(batch):
    return tuple(zip(*batch))