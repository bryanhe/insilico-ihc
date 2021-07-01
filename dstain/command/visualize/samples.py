#!/usr/bin/env python3

import collections
import click
import os

import dstain


@click.command()

@click.argument("sample_file", type=click.File("r"))
@click.argument("root", type=str, default=os.path.abspath(dstain.config.REGISTRATION))  # TODO: could do rel path with hard coded ..'s?
@click.option("--output", type=click.Path(), default=os.path.join(dstain.config.OUTPUT, "visualize", "samples"))
def samples(sample_file, root, output):
    samples = dstain.utils.read_sample_file(sample_file, use_split=False)
    stains = ["H", "aB", "T"]
    regions = ["AMY", "cHIP", "HIP", "MID", "MF"]

    patients = collections.defaultdict(lambda: [{"name": r, "slides": ["" for _ in stains]} for r in regions])
    for s in samples:
        *patient, r = s.split("_")
        patient = "_".join(patient)
        for (path, basename, stain) in samples[s]:
            patients[patient][regions.index(r)]["slides"][stains.index(stain)] = (s, basename)

    patients = [{"name": p, "slides": patients[p]} for p in patients]

    os.makedirs(output, exist_ok=True)

    tex = dstain.utils.render_latex(
        os.path.join(dstain.config.ROOT, "latex", "samples.tex"),
        root=root,
        stains=stains,
        regions=regions,
        patients=patients,
    )
    with open(os.path.join(output, "samples.tex"), "w") as tex_file:
        tex_file.write(tex)
