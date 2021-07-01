#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import os
import click
import collections
import sys
import pickle

@click.command()
@click.argument("root", default="data/registration", type=str)
@click.argument("output", default=".", type=str)
def main(root, output):
    # Based on https://nipunbatra.github.io/blog/2014/latexify.html
    def latexify():
        import matplotlib
        params = {'backend': 'pdf',
              'axes.titlesize':  8,
              'axes.labelsize':  8,
              'font.size':       8,
              'legend.fontsize': 8,
              'xtick.labelsize': 8,
              'ytick.labelsize': 8,
              # 'text.usetex': True,
              'font.family': 'DejaVu Serif',
              'font.serif': 'Computer Modern',
        }
        matplotlib.rcParams.update(params)
    latexify()

    patches = collections.defaultdict(list)
    blocks = sorted(os.listdir(root))
    for block in blocks:
        print(block)
        try:
            with open(os.path.join(root, block, "2a_initialization", "patch.pkl"), "rb") as f:
                patch = pickle.load(f)
        except FileNotFoundError:
            continue
        patches[block.split("_")[-1]].append(len(patch))

    # fig = plt.figure(figsize=(3, 3))
    # for region in sorted(patches):
    #     plt.hist(patches[region])
    # plt.savefig(os.path.join(output, "patches.pdf"))
    # plt.close(fig)
    fig = plt.figure(figsize=(3, 3))
    plt.hist(sum(patches.values(), []), bins=10)
    plt.xlim([0, plt.xlim()[1]])
    plt.xlabel("Patches")
    plt.ylabel("Sections")
    plt.tight_layout()
    plt.savefig(os.path.join(output, "patches.pdf"))
    plt.close(fig)


if __name__ == "__main__":
    main()
