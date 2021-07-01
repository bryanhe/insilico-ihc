import math
import torchvision
import PIL
import collections
import os
import numpy as np
import zipfile
import pickle

import concurrent.futures
import torch.utils.data
import tqdm


# TODO: pick a better name
class Stain(torchvision.datasets.VisionDataset):
    def __init__(self, root, samples, transform=None, target_transform=None, transforms=None, window=8192, downsample=16, ref_stain="H",
                 # stains=dstain.constant.stains,
                 stains=["aB", "T"],
                 skip_missing=True,
                 use_zip=True,
                 num_workers=0,
        ):
        super(Stain, self).__init__(root, transforms, transform, target_transform)

        self.window = window
        self.downsample = downsample
        self.ref_stain = ref_stain
        self.stains = stains
        self.skip_missing = skip_missing
        self.use_zip = use_zip

        self.sample = []
        self.patch = []
        self.stains_available = collections.defaultdict(dict)
        self.zipfile = {}

        def get_patches_in_sample(sample):
            stains_available = {}
            for (_, basename, stain) in samples[sample]:
                if stain in stains_available:
                    raise ValueError("{} has multiple slides with {}.".format(sample, stain))
                stains_available[stain] = basename
            if self.ref_stain not in stains_available:
                raise ValueError("{} missing reference stain ({}).".format(sample, self.ref_stain))
            if skip_missing and set(self.stains).intersection(set(stains_available)) == set():
                print("{} has no relevant IHC.".format(sample))
                return sample, stains_available, []

            if use_zip:
                zf = os.path.join(root, sample, "4_patches", "{}_{}.zip".format(window, downsample))
                if not os.path.isfile(zf):
                    raise ValueError("{} missing patches".format(sample))

                #  self.zipfile[sample] = zipfile.ZipFile(zf, "r")
                #  p = [os.path.basename(i) for i in self.zipfile[sample].namelist() if os.path.dirname(i) == self.stains_available[sample][self.ref_stain]]
                with zipfile.ZipFile(zf, "r") as f:
                    p = [os.path.basename(i) for i in f.namelist() if os.path.dirname(i) == stains_available[self.ref_stain]]
            else:
                if not os.path.exists(os.path.join(root, sample, "4_patches", "{}_{}".format(window, downsample))):
                    raise ValueError("{} missing patches".format(sample))

                p = os.listdir(os.path.join(root, sample, "4_patches", "{}_{}".format(window, downsample), stains_available[self.ref_stain]))
                if not all(os.path.splitext(i)[-1] == ".jpg" for i in p):
                    raise ValueError("{} contains non-jpg files".format(sample))
                p = [i for i in p if i[-4:] == ".jpg"]  # TODO: could filter .jpg suffix

            return sample, stains_available, p

        with tqdm.tqdm(total=len(samples), desc="Reading patches for dataset") as pbar:
            if num_workers != 0:  # TODO: not actually faster
                with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
                    for (sample, stains_available, p) in executor.map(get_patches_in_sample, samples):
                        self.sample.extend(len(p) * [sample])
                        self.stains_available[sample] = stains_available
                        self.patch.extend(p)
                        pbar.update()
            else:
                for (sample, stains_available, p) in map(get_patches_in_sample, samples):
                    self.sample.extend(len(p) * [sample])
                    self.stains_available[sample] = stains_available
                    self.patch.extend(p)
                    pbar.update()


    # TODO: define function to close zipfiles
    def __len__(self):
        return len(self.sample)

    def __getitem__(self, index):
        ihc = []
        label = []

        # Load reference stain (typically H&E)
        if self.use_zip:
            path = os.path.join(self.stains_available[self.sample[index]][self.ref_stain], self.patch[index])
            # with self.zipfile[self.sample[index]].open(path) as g:
            #     # img = PIL.Image.open(f.extractfile(path))
            #     ref = PIL.Image.open(g)
            #     ref.load()
            zf = os.path.join(self.root, self.sample[index], "4_patches", "{}_{}.zip".format(self.window, self.downsample))
            with zipfile.ZipFile(zf, "r") as f:
                with f.open(path) as g:
                    ref = PIL.Image.open(g)
                    ref.load()
        else:
            path = os.path.join(self.root, self.sample[index], "4_patches", "{}_{}".format(self.window, self.downsample), self.stains_available[self.sample[index]][self.ref_stain], self.patch[index])
            ref = PIL.Image.open(path)

        for s in self.stains:
            if s in self.stains_available[self.sample[index]] and os.path.isfile(os.path.join(self.root, self.sample[index], "4_patches", "{}_{}".format(self.window, self.downsample), self.stains_available[self.sample[index]][s], self.patch[index])):
                # path = os.path.join(self.root, self.sample[index], "4_patches", "{}_{}".format(self.window, self.downsample), self.stains_available[self.sample[index]][s], self.patch[index])
                # img = PIL.Image.open(path)

                # with self.zipfile[self.sample[index]] as f:
                # path = os.path.join(self.sample[index], "4_patches", "{}_{}".format(self.window, self.downsample), self.stains_available[self.sample[index]][s], self.patch[index])
                # with self.zipfile[self.sample[index]].open(path) as g:
                #     # img = PIL.Image.open(f.extractfile(path))
                #     img = PIL.Image.open(g)
                #     img.load()

                # ihc.append(img)
                ihc.append(0)
            else:
                ihc.append(0)

        values = {}
        for s in set(self.stains): # TODO: should use stain/target as arg instead; might need to make ihc put different attrs in different files
            if s in self.stains_available[self.sample[index]]:
                path = os.path.join(self.root, self.sample[index], "5_annotations", "{}_{}".format(self.window, self.downsample), self.stains_available[self.sample[index]][s], os.path.splitext(self.patch[index])[0] + ".pkl")
                try:
                    with open(path, "rb") as f:
                        values[s] = pickle.load(f)
                except FileNotFoundError:
                    pass

        count = collections.defaultdict(int)
        for s in self.stains:
            # TODO: should use target as additional arg; might need to make ihc put different attrs in different files
            # Right now, just assumes that the order saved by dstain.commands.ihc is the order desired
            if s in values:
                label.append(values[s][count[s]])
                count[s] += 1
            else:
                label.append(math.nan)

        for s in set(self.stains):
            # check that the number of times a stain appears matches the saved file
            assert s not in values or values[s].shape == (count[s],)

        if self.transforms is not None:
            # ((ref, *ihc), seg) = self.transforms([ref] + ihc, seg)
            ref = self.transforms(ref)

        return ref, ihc, torch.as_tensor(label), self.sample[index], np.array(list(map(int, os.path.splitext(self.patch[index])[0].split("_"))))
