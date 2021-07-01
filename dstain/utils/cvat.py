import concurrent.futures
import os
import xml.dom.minidom
import zipfile

import numpy as np
import PIL
import skimage
import tqdm


def save_annotations(cvat, output, index, grid=None, num_workers=4, downsample=1):
    """Save the annotations from CVAT.

    Parameters
    ----------
    cvat : str
        Path to CVAT annotations (as zip file)
    output : str
        Directory to write the annotations to
    index : dict from str to int
        Mapping from annotations name to class index
    grid : int or None
        If None, save the patch as one image; if int, cut the patch into
        smaller subpatches
    num_workers : int
        Number of subprocesses to use for data loading. If 0, data is loaded
        in the main process.
    downsample : int
        Downsampling factor

    Notes
    -----
    grid refers to the size of the subpatches at full resolution (see
    dstain.command.ihc for details).
    """
    # Loading annotations
    with zipfile.ZipFile(cvat) as zf:
        with zf.open("annotations.xml") as f:
            data = f.read()

    doc = xml.dom.minidom.parseString(data)

    # Read the labels in the annotation and the color used in CVAT
    color = {l.childNodes[1].firstChild.data: l.childNodes[3].firstChild.data for l in doc.getElementsByTagName("label")}
    # Convert the color from hex to RGB tuple
    color = {k: tuple(int(color[k].lstrip("#")[i:(i + 2)], 16) for i in (0, 2, 4)) for k in color}

    # Get the annotated images
    images = doc.getElementsByTagName("image")

    def save_annotation(img):
        # Save a single annotation
        # This function is to allow multiprocessing with concurrent.futures

        # Filename of image corresponding to this annotation
        filename = img.attributes["name"].value

        # Sort the annotations by z-order
        # (allows the annotations to be covered in the right order)
        # Note: the i from enumerate is needed to break ties; the z-orders are
        # not unique, and the a's themselves cannot be compared (no < operator
        # defined)
        annotation = sorted((int(a.attributes["z_order"].value), i, a) for (i, a) in enumerate(img.childNodes) if isinstance(a, xml.dom.minidom.Element))
        w = int(img.attributes["width"].value)
        h = int(img.attributes["height"].value)
        if annotation == []:
            return None

        annotation = list(zip(*annotation))[2]

        if (((grid is None) and not os.path.isfile(os.path.join(output, "annotations", os.path.splitext(filename)[0] + ".png"))) or
            ((grid is not None) and not all(os.path.isfile(os.path.join(output, "annotations", os.path.splitext(filename)[0] + "_{}_{}.png".format(p0 * downsample, p1 * downsample))) for p0 in range(0, w, grid // downsample) for p1 in range(0, h, grid // downsample)))):
            # index of the class (used for training model; saved in the labels
            # directory)
            label_index = np.full((h, w), 0, dtype=np.uint8)

            # colored image (used for visual inspection; saved in the
            # annotations directory)
            label_color = np.zeros((h, w, 3), dtype=np.uint8)

            # number of the annotation (used to check that all objects are
            # visible; not saved)
            label_check = np.full((h, w), -1)

            # if the pixel in marked by a point (used to check that no
            # individual points are covered; not saved)
            label_point = np.zeros((h, w), dtype=np.bool)

            for (i, a) in enumerate(annotation):
                label = a.attributes["label"].value
                if a.tagName == "box":
                    xtl = round(float(a.attributes["xtl"].value))
                    ytl = round(float(a.attributes["ytl"].value))
                    xbr = round(float(a.attributes["xbr"].value))
                    ybr = round(float(a.attributes["ybr"].value))
                    label_index[ytl:ybr, xtl:xbr] = index[label]
                    label_color[ytl:ybr, xtl:xbr, :] = color[label]
                    label_check[ytl:ybr, xtl:xbr] = i
                    if label_point[ytl:ybr, xtl:xbr].any():
                        print("{} has a point hidden".format(filename))
                elif a.tagName == "polygon":
                    points = list(map(lambda x: list(map(float, x.split(","))),
                                      a.attributes["points"].value.split(";")))
                    rr, cc = skimage.draw.polygon(*zip(*points), shape=(w, h))
                    label_index[cc, rr] = index[label]
                    label_color[cc, rr, :] = color[label]
                    label_check[cc, rr] = i
                    if label_point[cc, rr].any():
                        print("{} has a point hidden".format(filename))
                elif a.tagName == "points":
                    points = list(map(lambda x: list(map(float, x.split(","))),
                                      a.attributes["points"].value.split(";")))
                    for (r, c) in points:
                        rr, cc = skimage.draw.circle(r, c, radius=1, shape=(w, h))
                        label_index[cc, rr] = index[label]
                        label_color[cc, rr, :] = color[label]
                        label_check[cc, rr] = i
                        label_point[cc, rr] = True
                else:
                    raise NotImplementedError("Annotation of type {} no implemented yet".format(a.tagName))

            visible = set(np.unique(label_check))
            for i in range(len(annotation)):
                if i not in visible:
                    print("{} has a {} fully hidden".format(filename, annotation[i].attributes["label"].value))
                    break

            os.makedirs(os.path.join(output, "labels", os.path.dirname(filename)), exist_ok=True)
            os.makedirs(os.path.join(output, "annotations", os.path.dirname(filename)), exist_ok=True)
            label_index = PIL.Image.fromarray(label_index)
            label_color = PIL.Image.fromarray(label_color)

            if grid is None:
                label_index.save(os.path.join(output, "labels", os.path.splitext(filename)[0] + ".png"))
                label_color.save(os.path.join(output, "annotations", os.path.splitext(filename)[0] + ".png"))
            else:
                for p0 in range(0, w, grid // downsample):
                    for p1 in range(0, h, grid // downsample):
                        label_index.crop((p0, p1, p0 + grid // downsample, p1 + grid // downsample)).save(os.path.join(output, "labels",      os.path.splitext(filename)[0] + "_{}_{}.png".format(p0 * downsample, p1 * downsample)))
                        label_color.crop((p0, p1, p0 + grid // downsample, p1 + grid // downsample)).save(os.path.join(output, "annotations", os.path.splitext(filename)[0] + "_{}_{}.png".format(p0 * downsample, p1 * downsample)))

        return filename.split("-")[0], filename.split("-")[1]

    patient = set()
    sample = set()
    count = 0
    with tqdm.tqdm(total=len(images), desc="Saving annotations") as pbar:
        if num_workers != 0:
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
                for ps in executor.map(save_annotation, images):
                    if ps is not None:
                        patient.add(ps[0])
                        sample.add(ps)
                        count += 1
                    pbar.update()
        else:
            for ps in map(save_annotation, images):
                if ps is not None:
                    patient.add(ps[0])
                    sample.add(ps)
                    count += 1
                pbar.update()
    print("{} slides, {} patches".format(len(sample), count))
