"""Generates train/valid/test splits and statistics from raw brain data."""

import collections
import os

import click
import numpy as np

import dstain


@click.command()
@click.argument("root", type=click.Path(exists=True),
                default=dstain.config.DATA_RAW)
@click.argument("output", type=click.Path(),
                default=os.path.join(dstain.config.OUTPUT, "brain"))
@click.option("--regions", "-r",
              multiple=True,
              type=click.Choice(dstain.constant.regions),
              default=dstain.constant.regions)
@click.option("--stains", "-s",
              multiple=True,
              type=click.Choice(("H",) + dstain.constant.stains),
              default=("H",) + dstain.constant.stains)
@click.option("--check/--no-check", default=False)
def process(root=dstain.config.DATA_RAW,
            output=os.path.join(dstain.config.OUTPUT, "brain"),
            regions=dstain.constant.regions,
            stains=dstain.constant.stains,
            check=False):
    """Generates train/valid/test splits and statistics from raw brain data.

    \b
    Parameters
    ----------
    root : Path
        directory containing raw data
    output : Path
        directory to write output files to
    check : bool
        flag to check if WSI files can be opened
    """

    os.makedirs(output, exist_ok=True)

    listed_regions = set()               # regions appearing in dataset
    listed_stains = set()                # stains appearing in dataset

    # slides[patient][region] is list of tuples describing each slide
    # available for the (patient, region)
    # Each tuple contains (stain, basename, full path)
    slides = collections.defaultdict(lambda: collections.defaultdict(list))

    # count[region][stain] is the number of slides from this region/stain
    count = np.zeros(
        (len(regions), len(stains)), dtype=np.int)

    with open(os.path.join(output, "files_with_read_issue.txt") if check
              else os.devnull, "w") as check_file:
        for patient in sorted(os.listdir(root)):
            for region in os.listdir(os.path.join(root, patient)):
                listed_regions.add(region)
                if region not in regions:
                    continue

                # filenames have a short string associated with a section
                # used to check that the names match
                sections = set()
                for slide in os.listdir(os.path.join(root, patient, region)):
                    if slide[-4:] != ".svs":
                        raise ValueError("\"{}\" has unknown suffix.".format(slide))

                    if check:
                        error = _check_read(os.path.join(root, patient, region, slide))
                        if error != "":
                            check_file.write(error)
                            check_file.flush()
                            continue

                    info = os.path.splitext(slide)[0].split("_")
                    if len(info) == 3:
                        _, section, stain = info
                    elif len(info) == 2:
                        _, stain = info
                        section = "UNNAMED"
                    else:
                        raise ValueError("Filename \"{}\" could not be interpreted".format(slide))
                    sections.add(section)

                    listed_stains.add(stain)

                    if stain in stains:
                        slides[patient][region].append((
                            stain,
                            os.path.splitext(slide)[0],
                            os.path.join(patient, region, slide)
                        ))

                # Sort the slides to match order of stains in argument
                slides[patient][region] = \
                    sorted(slides[patient][region],
                           key=lambda x: stains.index(x[0]))

                errors = _check_section(sections, slides[patient][region])
                if errors != "":
                    print(os.path.join(patient, region))
                    print(errors)
                stains_in_section = list(zip(*slides[patient][region]))[0]
                if len(stains_in_section) == 1:
                    del slides[patient][region]
                    continue
                for (stain, _, _) in slides[patient][region]:
                    if stain in stains:
                        count[regions.index(region), stains.index(stain)] += 1

    print("Regions:", " ".join(sorted(listed_regions)))
    print("Stains:", " ".join(sorted(listed_stains)))

    # count[region][stain] is the number of slides from this region/stain
    count = np.zeros(
        (len(regions), len(stains)), dtype=np.int)
    for patient in slides:
        if patient[0] != "e":
            for region in slides[patient]:
                for (stain, _, _) in slides[patient][region]:
                    if stain in stains:
                        count[regions.index(region), stains.index(stain)] += 1
    _write_stain_table(os.path.join(output, "stains.tex"), stains, regions, count)
    
    # count[region][stain] is the number of slides from this region/stain
    count = np.zeros(
        (len(regions), len(stains)), dtype=np.int)
    for patient in slides:
        if patient[0] == "e":
            for region in slides[patient]:
                for (stain, _, _) in slides[patient][region]:
                    if stain in stains:
                        count[regions.index(region), stains.index(stain)] += 1
    _write_stain_table(os.path.join(output, "stains_eval.tex"), stains, regions, count)

    _write_sample_list(os.path.join(output, "samples.tsv"), slides)


def _check_read(filename):
    """Check if a WSI can be read properly.

    Attempts to read a small part of the image at all levels, and verifies
    that all are read properly.

    Parameters
    ----------
    filename : Path
        path to slide to try to read

    Returns
    -------
    error : Str
        filename and level with error; "" if no error
    """
    import openslide  # pylint: disable=import-outside-toplevel
    slide = openslide.open_slide(filename)
    try:
        for level in range(slide.level_count):
            slide.read_region((0, 0), level, (100, 100))
    except openslide.lowlevel.OpenSlideError as error:
        print(error)
        return "{}\t{}\n".format(filename, level)
    return ""


def _check_section(sections, slides):
    """Check that the stains available for this section make sense.

    Parameters
    ----------
    sections : Set of Str
        Section identifiers that have appeared in the filenames in the section
    slides : List of Tuples of Str
        information about the slides in this section
        each tuple contains (stain, basename, full path)

    Returns
    -------
    error : Str
        Information about the error(s) encountered; "" if no error
    """

    errors = ""

    # Checking for exactly one section name
    if len(sections) == 0:
        errors += "(no slides)\n"
    if len(sections) > 1:
        errors += "(multiple sections: {})\n".format(", ".join(sections))

    # Checking for H and at least one other stain
    stains_in_section = list(zip(*slides))[0]
    if "H" not in stains_in_section:
        errors += "(no H)\n"
    elif len(stains_in_section) == 1:
        errors += "(only {})\n".format(stains_in_section[0])

    # Check that this section only has one of each stain
    if len(stains_in_section) != len(set(stains_in_section)):
        errors += "(duplicated stain)\n"

    return errors

def _write_stain_table(filename, stains, regions, count):
    """Write latex table of stains available by region.

    Parameters
    ----------
    filename : Path
        output location
    stains : List of Str
        list of the stains processed
    regions : List of Str
        list of the regions processed
    count : np.array((len(regions), len(stains)), dtype=int)
        number of slides from the (region, stain) tuple
    """
    # Remove empty regions
    mask = (count != 0).any(axis=1)
    regions = tuple(r for (r, m) in zip(regions, mask) if m)
    count = count[mask, :]

    # Writing table of stain counts
    length = list(map(lambda x: max(map(lambda y: len(str(y)), x)), zip(stains, count.sum(0))))
    region_count = [{
        # 6 is from the "Region" in the latex template
        "name": "{s:{w}s}".format(w=6, s=r),
        "count": ["{i:{w}d}".format(w=l, i=i) for (i, l) in zip(c, length)]
    } for (r, c) in zip(regions, count)]
    table = dstain.utils.render_latex(
        os.path.join(dstain.config.ROOT, "latex", "stains.tex"),
        stains=["{s:>{w}s}".format(w=l, s=s) for (s, l) in zip(stains, length)],
        regions=region_count,
        total=count.sum(0)
    )

    with open(filename, "w") as stain_file:
        stain_file.write(table)

def _write_sample_list(filename, slides):
    """Write tsv of samples in the dataset.

    Each row describes one slide:
        1) Name for the section (patient and region)
        2) Path to the WSI of this slide
        3) Basename (short identifier for this slide)
        4) Stain for this slide
        5) Split ("train"/"valid"/"test")

    Parameters
    ----------
    filename : Path
        output location
    slides : dict of dict of tuples
        slides[patient][region] is a tuple of (stain, basename, path)
    """
    def get_split(index, total):
        if index < 0.70 * total:
            return "train"
        if index < 0.85 * total:
            return "valid"
        return "test"

    patients = sorted(slides)
    split = {p: get_split(i, len(list(p for p in patients if p[0] != "e"))) for (i, p) in enumerate(p for p in patients if p[0] != "e")}
    for p in patients:
        if p[0] == "e":
            split[p] = "eval"
    with open(filename, "w") as sample_file:
        for patient in sorted(slides):
            for region in slides[patient]:
                for (stain, basename, path) in slides[patient][region]:
                    sample_file.write("{}_{}\t{}\t{}\t{}\t{}\n".format(
                        patient, region,
                        path,
                        basename,
                        stain,
                        split[patient],
                    ))
