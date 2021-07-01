import os
import PIL

import torch
import torchvision

class AnnotatedIHC(torchvision.datasets.VisionDataset):
    """Annotated IHC dataset for classification.

    Parameters
    ----------
    root : str
        directory containing images and annotations
    stain : str
        name of the stain to use
    split : str
        one of {'train', 'valid', 'test'}
    transform : callable or None
        a function/transform that  takes in an PIL image and returns a
        transformed version. E.g, ``torchvision.transforms.ToTensor``
    target_transform : callable or None
        a function/transform that takes in the target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
        and returns a transformed version.
    transforms : TODO
        TODO

    Returns
    -------
    TODO
    """

    def __init__(self, root, stain, n_targets, split="train", transform=None,
                 target_transform=None, transforms=None):
        super(AnnotatedIHC, self).__init__(root, transforms, transform,
                                      target_transform)

        self.root = root
        self.stain = stain
        self.n_targets = n_targets
        self.split = split

        self.images = sorted([os.path.splitext(f)[0] for f in os.listdir(os.path.join(root, "labels", stain))])
        patients = sorted(set(map(lambda x: x.split("-")[0], self.images)))
        s1 = round(0.70 * len(patients))
        s2 = round(0.85 * len(patients))
        if split == "train":
            patients = patients[:s1]
        elif split == "valid":
            patients = patients[s1:s2]
        elif split == "test":
            patients = patients[s2:]
        else:
            raise ValueError("Unrecognized split \"{}\"".format(split))

        self.images = [f for f in self.images if f.split("-")[0] in patients]
        self.cache = [None for _ in self.images]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img = PIL.Image.open(os.path.join(self.root, "images", self.stain, self.images[index] + ".jpg"))
        label = PIL.Image.open(os.path.join(self.root, "labels", self.stain, self.images[index] + ".png"))

        if self.transforms is not None:
            img, label = self.transforms(img, label)

        target = torch.zeros(self.n_targets, dtype=torch.bool)
        for i in range(self.n_targets):
            if (label == (i + 1)).any():
                target[i] = True

        return img, target, self.images[index]
