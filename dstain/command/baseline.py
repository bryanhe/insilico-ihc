import glob
import math
import os
import pickle
import sys

import click
import numpy as np
import torch
import torchvision
import tqdm
import sklearn

import dstain


@click.command()
@click.argument("sample_file",
                type=click.File("r"))
@click.option("--src", "-s", type=click.Path(exists=True), default="data/registration")
@click.option("--output", "-o", type=click.Path(), default="output/baseline")
def baseline(
        sample_file,
        src,
        output,
        stains=["aB", "T", "T"],  # TODO flag to select (or just create another sample file?)
        targets=["Plaque", "Tangle", "Neuritic Plaque"],
):
    import histomicstk as htk


    # Based on https://digitalslidearchive.github.io/HistomicsTK/examples/nuclei-segmentation.html#Load-input-image
    # Strain matrix from https://digitalslidearchive.github.io/HistomicsTK/histomicstk.preprocessing.color_deconvolution.html
    W = np.array([[0.650, 0.072, 0], [0.704, 0.990, 0], [0.286, 0.105, 0]])

    # W = (W - mean) / std


    os.makedirs(output, exist_ok=True)
    dstain.utils.latexify()
    device = torch.device("cuda")
    sample = dstain.utils.read_sample_file(sample_file)

    dataset = {s: dstain.datasets.Stain(src, sample[s], window=window, downsample=downsample, transforms=transforms, stains=stains) for s in sample}

    split = "train"

    ds = dataset[split]

    n_samples = 1000 * batch_size
    if n_samples < len(ds):
        indices = np.random.choice(len(ds), n_samples, replace=False)
        ds = torch.utils.data.dataset.Subset(dataset[s], indices)

    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=(device.type == "cuda"), drop_last=True)

    for (X, _, y, block, pixel) in label:

        im_stains = htk.preprocessing.color_deconvolution.color_deconvolution(im_input[i, :, :, :], W).Stains

        # # Display results
        # plt.figure(figsize=(20, 10))
        #
        # plt.subplot(1, 2, 1)
        # plt.imshow(im_stains[:, :, 0])
        # plt.title("Hematoxylin")
        #
        # plt.subplot(1, 2, 2)
        # plt.imshow(im_stains[:, :, 1])
        # _ = plt.title("Eosin")
        # plt.savefig("temp/{}_sep.jpg".format(i))

        # get nuclei/hematoxylin channel
        im_nuclei_stain = im_stains[:, :, 0]

        # segment foreground
        foreground_threshold = 60

        im_fgnd_mask = scipy.ndimage.morphology.binary_fill_holes(
            im_nuclei_stain < foreground_threshold)

        # run adaptive multi-scale LoG filter
        min_radius = 10
        max_radius = 15

        im_log_max, im_sigma_max = htk.filters.shape.cdog(
            im_nuclei_stain, im_fgnd_mask,
            sigma_min=min_radius * np.sqrt(2),
            sigma_max=max_radius * np.sqrt(2)
        )

        # detect and segment nuclei using local maximum clustering
        local_max_search_radius = 10

        im_nuclei_seg_mask, seeds, maxima = htk.segmentation.nuclear.max_clustering(
            im_log_max, im_fgnd_mask, local_max_search_radius)

        # filter out small objects
        min_nucleus_area = 80

        im_nuclei_seg_mask = htk.segmentation.label.area_open(
            im_nuclei_seg_mask, min_nucleus_area).astype(np.int)

        # compute nuclei properties
        objProps = skimage.measure.regionprops(im_nuclei_seg_mask)

        count[i] = len(objProps)
        area[i] = sum(map(lambda x: x.area, objProps)) / h / w

