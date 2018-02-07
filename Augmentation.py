''' Module for data augmentation. Two strategies have been demonstrated below. 
You can check for more strategies at 
http://pytorch.org/docs/master/torchvision/transforms.html '''
import torch
from torchvision import transforms
import random

class Augmentation:
    def __init__(self,strategy):
        print ("Data Augmentation Initialized with strategy %s"%(strategy));
        self.strategy = strategy;


    def applyTransforms(self):
        if self.strategy == "H_FLIP": # horizontal flip with a probability of 0.5
            data_transforms = {
            'train': transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }

        elif self.strategy == "SCALE_H_FLIP": # resize to 224*224 and then do a random horizontal flip.
            data_transforms = {
            'train': transforms.Compose([
                transforms.Scale([224,224]),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Scale([224,224]),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }

        elif self.strategy == 'FIVE_CROP_FLIP': # Crop the given PIL Image into four corners and the central crop
            h = random.randint(5, 64)
            w = random.randint(5, 64)
            data_transforms = {
                'train': transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.FiveCrop([h, w]),
                    transforms.ToTensor,
                    transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops]))]),
                'val': transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.FiveCrop(h, w),
                    transforms.ToTensor,
                    transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops]))]),
                # returns a 4D tensor
            }

        elif self.strategy == 'RANDOM_FLIP_CROP': # crop the pil image randomly and flip horizontally
            data_transforms = {
                'train': transforms.Compose([
                    transforms.RandomCrop(56),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
                'val': transforms.Compose([
                    transforms.RandomCrop(56),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

            }

        else:
            print ("Please specify correct augmentation strategy : %s not defined" % self.strategy);
            exit();

        return data_transforms;

