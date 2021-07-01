import torchvision
import dstain
import openslide
import os
import math
import numpy as np
import PIL


# TODO: better class name
class WSI(torchvision.datasets.vision.VisionDataset):
    def __init__(self, root, files, patch, warp, window, downsample, transform=None, target_transform=None, ref_stain="H", stains=dstain.constant.stains):
        super(WSI, self).__init__(root, transform=transform,
                                      target_transform=target_transform)
        self.files = files
        self.slide = {}
        for (i, (path, basename, stain)) in enumerate(self.files):
            if stain in self.slide:
                raise ValueError("Duplicate {} for {}".format(stain, root))
            self.slide[stain] = openslide.open_slide(os.path.join(self.root, path))

        self.patch = patch
        self.warp = warp
        self.window = window
        self.downsample = downsample
        self.ref_stain = ref_stain
        self.stains = stains

    def __len__(self):
        return len(self.patch)

    def __getitem__(self, index):
        h = dstain.utils.openslide.read_region_at_mag(self.slide[self.ref_stain], self.patch[index], 40, (self.window, self.window), downsample=self.downsample)
        h_img = h
        if self.transform is not None:
            h = self.transform(h)
        ihc = []
        # ihc_img = []
        for s in self.stains:
            if s in self.slide:
                img = dstain.utils.register.load_warped(self.slide[s], self.patch[index], (self.window, self.window), self.warp[s], self.downsample)
                if self.target_transform is not None and s in self.target_transform and self.target_transform[s] is not None:
                    img = self.target_transform[s](img)
                ihc.append(img)
            else:
                # TODO: should make target_transform handle this correctly
                if self.target_transform is not None and s in self.target_transform and self.target_transform[s] is not None:
                    # TODO: implemented assuming totensor
                    ihc.append(h * math.nan)
                else:
                    ihc.append(PIL.Image.new("RGB", (0, 0)))
                # ihc_img.append(PIL.Image.new("RGB", (0, 0)))

        return h, ihc, index, np.array(self.patch[index])
