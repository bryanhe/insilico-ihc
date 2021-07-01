#!/usr/bin/env python3

import click
import os

import dstain


@click.command()

@click.argument("sample_file", type=click.File("r"))
@click.argument("root", type=str, default=os.path.abspath(dstain.config.REGISTRATION))  # TODO: could do rel path with hard coded ..'s?
@click.option("--output", type=click.Path(), default=os.path.join(dstain.config.OUTPUT, "visualize", "registration"))
def register(sample_file, root, output):
    samples = dstain.utils.read_sample_file(sample_file, use_split=False)
    samples = [{"name": s, "slides": list(zip(*samples[s]))[1]} for s in sorted(samples)]

    os.makedirs(output, exist_ok=True)

    for template in ["mask.tex", "registration.tex", "hist.tex"]:
        tex = dstain.utils.render_latex(
            os.path.join(dstain.config.ROOT, "latex", template),
            root=root,
            samples=samples
        )
        with open(os.path.join(output, template), "w") as tex_file:
            tex_file.write(tex)
