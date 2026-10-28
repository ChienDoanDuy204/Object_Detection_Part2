from pathlib import Path
from torchvision.datasets import VOCDetection
import os

if __name__ == '__main__':
    ROOT_DIR = Path.cwd().parent
    DATA_DIR = str(ROOT_DIR/'data')
    os.makedirs(DATA_DIR, exist_ok=True)
    train = VOCDetection(
    root = DATA_DIR,
    year = '2007',
    image_set = 'trainval',
    download = True
)