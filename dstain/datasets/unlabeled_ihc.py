import PIL
import glob
import os
import zipfile

import torchvision

class UnlabeledIHC(torchvision.datasets.VisionDataset):
    """TODO
    """
    def __init__(self, root, stain, window, downsample, transform=None, use_zip=True, skip_completed=True, model_timestamp=None):
        super(UnlabeledIHC, self).__init__(root, transform=transform)

        self.use_zip = use_zip

        self.images = []
        if use_zip:
            for filename in sorted(glob.glob(os.path.join(root, "*", "4_patches", "{}_{}.zip".format(window, downsample)))):
                with zipfile.ZipFile(filename, "r") as f:  # TODO: multithread
                    for img in f.namelist():
                        if os.path.dirname(img).split("_")[-1] == stain:
                            des = os.path.join(os.path.splitext(filename.replace("4_patches", "5_annotations"))[0], os.path.splitext(img)[0] + ".pkl")
                            if not skip_completed or not os.path.isfile(des) or os.path.getmtime(des) < model_timestamp or os.path.getmtime(des) < os.path.getmtime(filename):
                                self.images.append(((filename, img), des))
        else:
            self.images = sorted(glob.glob(os.path.join(root, "*", "4_patches", "{}_{}".format(window, downsample), "*_" + stain, "*")))
            if skip_completed:  # TODO: Could also check timestamp TODO TODO
                asdknsadjka
            #     self.images = [f for f in self.images if not os.path.isfile(os.path.splitext(f.replace("4_patches", "5_annotations"))[0] + ".txt")]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        if self.use_zip:
            ((z, i), des) = self.images[index]
            with zipfile.ZipFile(z, "r") as f:
                with f.open(i) as g:
                    img = PIL.Image.open(g)
                    img.load()
        else:
            img = PIL.Image.open(self.images[index])
            des = os.path.splitext(self.images[index])[0].replace("4_patches", "5_annotations") + ".pkl"

        if self.transform is not None:
            img = self.transform(img)

        return img, des
