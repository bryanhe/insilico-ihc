"""Registering H&E with IHC slides.

This file only provides a skeleton for the command line interface. Most helper
functions are in dstain.utils.register.
"""

import collections
import os

import click

import dstain


@click.command()
@click.argument("sample_file",
                type=click.File("r"))
@click.argument("select",
                nargs=-1,
                type=str)
@click.option("--src", "-s", type=click.Path(exists=True), default=dstain.config.DATA_RAW)
@click.option("--output", "-o", type=click.Path(), default=dstain.config.REGISTRATION)
@click.option("--window", "-w", type=int, default=512)
@click.option("--downsample", "-d", type=int, default=1)
@click.option("--patches", type=int, default=0)
@click.option("--zip_patches/--no_zip", default=False)
@click.option("--verbose/--quiet", default=True)
@click.option("--thumbnail_downsample", type=int, default=128)
@click.option("--nfeatures", type=int, default=5000)
@click.option("--ransacReprojThreshold", type=int, default=25)
@click.option("--affine/--proj", default=True)
def register(
    sample_file,
    select,
    src=dstain.config.DATA_RAW,
    output=dstain.config.REGISTRATION,
    window=512,
    downsample=1,
    patches=0,
    zip_patches=False,
    verbose=True,
    thumbnail_downsample=128,
    nfeatures=5000,
    ransacreprojthreshold=25,
    affine=True):
    """Registering H&E with IHC slides.

    \b
    Parameters
    ----------
    sample_file : TODO
        TODO
    src : TODO
        TODO
    output : TODO
        TODO
    select : TODO
        TODO
    window : TODO
        TODO
    patches : TODO
        TODO
    downsample : TODO
        TODO
    thumbnail_downsample : TODO
        TODO
    verbose : TODO
        TODO
    """

    # Reading list of files in each sample
    # samples[key] = value
    #     key is the name of the sample
    #     value is a list of filenames
    samples = dstain.utils.read_sample_file(sample_file, use_split=False)

    # Allow subselection of samples to register
    if select == ():
        select = sorted(samples.keys())

    # Loop through samples
    for (i, sample) in enumerate(select):
        print("Sample #{} / {}: {}".format(i + 1, len(select), sample), flush=True)
        dstain.utils.register.register_sample(
            src, os.path.join(output, sample), samples[sample], window, downsample,
            patches, zip_patches, verbose, thumbnail_downsample, nfeatures, ransacreprojthreshold, affine)

