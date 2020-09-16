from torch.utils.data import Dataset
import os
import cv2

class ValidationDataset(Dataset):
    def __init__(self, main_folder='tum_dataset'):
        self.main_folder = main_folder
        self.id = 0
        with open(os.path.join(os.curdir, main_folder, "rgb.txt")) as f:
            self.rgb = f.read().splitlines()
            self.rgb = [el.split(" ")[1] for el in self.rgb]
        with open(os.path.join(os.curdir, main_folder, "depth.txt")) as f:
            self.depth = f.read().splitlines()
            self.depth = [el.split(" ")[1] for el in self.depth]
        self.length = len(self.depth)
    
    def __getitem__(self, id):
        path4depth = os.path.join(os.curdir, self.main_folder, self.depth[id])
        path4rgb = os.path.join(os.curdir, self.main_folder, self.rgb[id])
        img = cv2.imread(path4rgb)
        print(img.shape)
        groundtruth_dict = {'image_transformed': self.transform(img), 'image' : img, 'tensor': torch.tensor(self.transform(img))[None].permute(0,3,1,2), 'groundtruth_depth': cv2.imread(path4depth, 0)}
        return groundtruth_dict
    
    def transform(self, img):
        img = cv2.resize(img, (384, 128))
        img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        
        # albumentations.Normalize()
        return img
    
    def __len__(self):
        return self.length