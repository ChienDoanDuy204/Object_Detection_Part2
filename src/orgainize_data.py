from torch.utils.data import Dataset
from torchvision.datasets import VOCDetection
import torch
from torchvision.transforms import transforms
import torch.nn.functional as F
import xml.etree.ElementTree as ET
import random
import os

class VOCObjectDetectionDataset(Dataset):
    def __init__(
        self,
        split: str = 'train',
        label2idx: dict = {},
        data_dir: str = None,
        list_transforms: list = [],
        image_pyramid: bool = True,
        pyramid_type: str = 'both',          # 'up' (upscale), 'down' (downscale), or 'both'
        min_size: tuple = (224, 224),      # Minimum (H, W) limit for downscale
        max_size: tuple = (1000, 1000),    # Maximum (H, W) limit for upscale
        scale: float = 1.5,                # Scaling ratio factor
        max_scale: float = 2.5,            # Maximum scale multiplier for up-pyramid
        pyramid_mode: str = 'all',         # 'all' (multi-scale expansion) or 'random' (random scale per epoch)
        RPN : bool = False
    ) -> None:
        self.list_transforms = [transforms.ToTensor()]
        if list_transforms:
            self.list_transforms.extend(list_transforms)
            
        need_download = not (data_dir and os.path.exists(os.path.join(data_dir, 'VOCdevkit', 'VOC2007')))
        self.dataset = VOCDetection(
            root=data_dir,
            year='2007',
            image_set=split,
            download=need_download,
            transform=transforms.Compose(self.list_transforms),
        )
        self.label2idx = label2idx
        self.image_pyramid = image_pyramid
        self.pyramid_type = pyramid_type
        self.min_size = min_size
        self.max_size = max_size
        self.scale = scale
        self.max_scale = max_scale
        self.pyramid_mode = pyramid_mode
        self.samples = []
        self.RPN = RPN

        if self.image_pyramid and self.pyramid_mode == 'all':
            # Fast index creation from XML annotations without loading images into RAM
            for idx, xml_path in enumerate(self.dataset.annotations):
                try:
                    tree = ET.parse(xml_path)
                    size_elem = tree.find('size')
                    W = int(size_elem.find('width').text)
                    H = int(size_elem.find('height').text)
                except Exception:
                    W, H = 500, 375

                factors = self._get_scale_factors(H, W)
                for f in factors:
                    self.samples.append((idx, f))
        else:
            self.samples = [(i, 1.0) for i in range(len(self.dataset))]

    def _get_scale_factors(self, H: int, W: int) -> list:
        """
        Calculates valid pyramid scale factors according to pyramid_type ('up', 'down', 'both').
        """
        factors = [1.0]

        # 1. Up Pyramid (Upscale for detecting smaller objects)
        if self.pyramid_type in ('up', 'both'):
            curr_factor = self.scale
            while True:
                h, w = int(H * curr_factor), int(W * curr_factor)
                if curr_factor > self.max_scale or h > self.max_size[0] or w > self.max_size[1]:
                    break
                factors.append(round(curr_factor, 4))
                curr_factor *= self.scale

        # 2. Down Pyramid (Downscale)
        if self.pyramid_type in ('down', 'both'):
            curr_factor = 1.0 / self.scale
            while True:
                h, w = int(H * curr_factor), int(W * curr_factor)
                if h < self.min_size[0] or w < self.min_size[1]:
                    break
                factors.append(round(curr_factor, 4))
                curr_factor /= self.scale

        return sorted(list(set(factors)))

    def __len__(self) -> int:
        if self.image_pyramid and self.pyramid_mode == 'random':
            return len(self.dataset)
        return len(self.samples)

    def __getitem__(self, index):
        if self.image_pyramid and self.pyramid_mode == 'random':
            item_idx = index
            img, tgt = self.dataset[item_idx]
            _, H, W = img.shape
            factors = self._get_scale_factors(H, W)
            factor = random.choice(factors)
        else:
            item_idx, factor = self.samples[index]
            img, tgt = self.dataset[item_idx]
            _, H, W = img.shape

        # Scale image using factor (factor > 1.0: upscale, factor < 1.0: downscale)
        if factor != 1.0:
            h, w = int(H * factor), int(W * factor)
            images_resized = F.interpolate(img.unsqueeze(0), size=(h, w), mode='bilinear', align_corners=False).squeeze(0)
        else:
            h, w = H, W
            images_resized = img

        boxes, labels = [], []
        objects = tgt['annotation']['object']
        if isinstance(objects, dict):
            objects = [objects]

        for obj in objects:
            if not self.RPN and len(self.label2idx) > 0:
                labels.append(self.label2idx[obj['name']])
            else: 
                labels.append(self.label2idx.get(obj['name'],1))
            bx1 = float(obj['bndbox']['xmin']) * (w / W)
            by1 = float(obj['bndbox']['ymin']) * (h / H)
            bx2 = float(obj['bndbox']['xmax']) * (w / W)
            by2 = float(obj['bndbox']['ymax']) * (h / H)

            # Clamp coordinates within scaled image boundaries
            bx1 = max(0.0, min(float(w), bx1))
            by1 = max(0.0, min(float(h), by1))
            bx2 = max(0.0, min(float(w), bx2))
            by2 = max(0.0, min(float(h), by2))
            boxes.append([bx1, by1, bx2, by2])

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.long)

        target = {"boxes": boxes, "labels": labels}
        return images_resized, target


# Padding các ảnh trong batch theo kích thước của ảnh lớn nhất trong Batch, padding các GTboxes và GTlabels theo số lượng boxes lớn nhất trong Batch
def collate_fn(batch):
    imgs, targets = zip(*batch)
    Hmax = max(img.shape[1] for img in imgs)
    Wmax = max(img.shape[2] for img in imgs)
    num_boxes_max = max(len(target['boxes']) for target in targets)
    GT_boxes = torch.zeros(len(imgs), num_boxes_max, 4)
    GT_labels = torch.zeros(len(imgs), num_boxes_max, dtype=torch.long)
    mask_GT_boxes = torch.zeros(len(imgs), num_boxes_max, dtype=torch.bool)
    
    for i, target in enumerate(targets):
        num_boxes = len(target['boxes'])
        GT_boxes[i, :num_boxes, :] = target['boxes']
        GT_labels[i, :num_boxes] = target['labels']
        mask_GT_boxes[i, :num_boxes] = True

    img_padded = torch.zeros(len(imgs), 3, Hmax, Wmax)
    size_GT = [] # kích thước của ảnh thật
    for k, img in enumerate(imgs):
        img_padded[k, :, :img.shape[1], :img.shape[2]] = img
        size_GT.append([img.shape[1], img.shape[2]])
    return img_padded, GT_boxes, GT_labels, size_GT, mask_GT_boxes
