"""Registering H&E with IHC slides.

TODO: write whole contents"""

import concurrent.futures
import itertools
import math
import os
import pickle
import random
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import PIL
import scipy
import skimage.feature
import torch
import tqdm
import zipfile

import dstain
import kornia
import openslide


def register_sample(
        src,
        output,
        filenames,
        window,
        downsample,
        patches,
        zip_patches=False,
        verbose=True,
        thumbnail_downsample=128,
        nfeatures=5000,
        ransacReprojThreshold=25,
        affine=True,
    ):
    """Registering H&E with IHC slides for a single block.

    Parameters
    ----------
    src : TODO
        TODO
    output : TODO
        TODO
    filenames : TODO
        TODO
    window : TODO
        TODO
    downsample : TODO
        TODO
    patches : TODO
        TODO
    thumbnail_downsample : TODO
        TODO
    verbose : TODO
        TODO
    """

    # TODO: cache return values
    if output is not None:
        os.makedirs(output, exist_ok=True)

    with open(os.devnull if output is None else os.path.join(output, "log.txt"), "w") as f:
        f.write("Begin\n")
        f.flush()

        # ---------------------------------------------------------------------
        # Load downsampled images and compute binary mask (tissue vs background)
        #   These images are cached after the first run since loading from the
        #   WSI is significantly slower.
        # ---------------------------------------------------------------------
        image, tissue_mask, otsu = load_block(
            src, filenames, thumbnail_downsample,
            None if output is None else os.path.join(output, "1_thumbnail"),
            verbose)

        # ----------------------------------------------------------------------
        # Coarse alignment
        #
        # Based on
        #   https://www.geeksforgeeks.org/image-registration-using-opencv-python/
        # ----------------------------------------------------------------------
        homography_matrix, transformed_image, transformed_mask, keypoint, descriptor, use_keypoint = \
            match_keypoints(
                filenames, tissue_mask, image, thumbnail_downsample,
                None if output is None else os.path.join(output, "2_register"),
                f,
                verbose, nfeatures, ransacReprojThreshold, affine)

        # TODO: transformed_image[0] should just be image 0?
        # TODO: same for transformed_mask?
        # TODO: get_patches should use transformed_mask instead of tissue_mask?
        patch = get_patches(
            src, filenames, transformed_mask, transformed_image, window, thumbnail_downsample,
            None if output is None else os.path.join(output, "3_selection"),
            verbose)

        # transformed_image, new_homography_matrix = adjust_coarse(block, tissue_mask, image, homography_matrix)
        # os.makedirs(os.path.join(output, block, "2b_adjust"), exist_ok=True)
        # for (i, (filename, mask)) in enumerate(zip(tqdm.tqdm(filenames), tissue_mask)):
        #     path_box = os.path.join(output, block, "2b_adjust", os.path.basename(filename) + "_box.pdf")
        #     fig = _drawBoxes(np.array(transformed_image[i]), patch, window, thumbnail_downsample, dpi=300)
        #     plt.savefig(path_box)
        #     plt.close(fig)

        # Set up slides and transforms
        # TODO: does it make sense to move the verification into either match_keypoints or save_patches,
        #       and then this initialization can just go in save_patches?
        slide = []
        transform = []
        for (homography, (path, basename, stain)) in zip(homography_matrix, filenames):
            transform.append(skimage.transform.ProjectiveTransform(homography))
            slide.append(openslide.open_slide(os.path.join(src, path)))

        # adjust_keypoints(slide, filenames, transform, otsu, keypoint, thumbnail_downsample, image, transformed_image, tissue_mask, transformed_mask, os.path.join(output, block, "3_refine"))

        # Rough check for reasonable transform
        has_error = set()
        for (x, y) in tqdm.tqdm(patch[::10], desc="Verifying", disable=not verbose):
            for (i, s) in enumerate(filenames):
                if i == 0:
                    continue
                base = np.array([[x, y],
                                 [x, y + window],
                                 [x + window, y + window],
                                 [x + window, y]])
                coord = transform[i].inverse(base)
                dist = np.linalg.norm(coord - np.vstack((coord[-1, :], coord[:-1, :])), axis=1)  # Lengths of the sides
                v = coord - np.vstack((coord[-1, :], coord[:-1, :]))
                v /= np.linalg.norm(v, axis=1, keepdims=True)
                angle = np.arccos((v * np.vstack((v[-1, :], v[:-1, :]))).sum(1))  # Angles between sides (in radians)
                if s not in has_error and \
                    not (np.logical_and(0.95 * window < dist, dist < 1.05 * window).all() and (np.abs(angle - math.pi / 2) < 0.05).all()):
                    f.write("Error {} (dist = {}) (angle = {})\n".format(s, dist, angle))
                    has_error.add(s)

        # TODO: if patches == 0, can this just be skipped?
        save_patches(slide, transform, filenames, patch, os.path.join(output, "4_patches", "{}_{}".format(window, downsample)), window, downsample, patches, zip_patches)

        f.write("Complete\n")
        f.flush()

    return patch, transform


def load_block(src, filenames, downsample, output, verbose=True):
    """Loads images, and computes masks and thresholds.

    Parameters
    ----------
    src : Path
        directory containing raw data
    filenames : List[str]  TODO: Fix
        list of filenames in this block
    downsample : int
        downsampling rate for image
    output : Path or None
        directory to output cached results
    verbose : bool
        print progress bar

    Returns
    -------
    image : List[PIL.Image.Image]
        downsampled raw images
    mask : List[np.array[bool]]
        binary masks for tissue
    otsu : List[int]
        otsu threshold for tissue (sum of RGB)
    """
    image = []
    mask = []
    otsu = []
    if output is not None:
        os.makedirs(output, exist_ok=True)
    for (path, basename, stain) in tqdm.tqdm(filenames, desc="Loading", disable=not verbose):
        try:
            path_hist = None

            if output is None:
                raise FileNotFoundError()

            # Set paths for caching images and intermediate computations
            path_tiff = os.path.join(output, basename + ".tif")
            path_jpeg = os.path.join(output, basename + ".jpg")
            path_mask = os.path.join(output, basename + "_mask.png")
            path_otsu = os.path.join(output, basename + "_otsu.txt")
            path_hist = os.path.join(output, basename + "_hist.pdf")

            # Attempt to load cached results
            img = PIL.Image.open(path_tiff)
            m = np.array(PIL.Image.open(path_mask))
            with open(path_otsu, "r") as f:
                t = int(f.read())
        except FileNotFoundError:
            # Load downsampled images and save
            slide = openslide.open_slide(os.path.join(src, path))
            img = dstain.utils.openslide.read_region_at_mag(
                slide, (0, 0), 40, slide.dimensions, downsample=downsample)

            # Compute tissue mask
            ng = np.logical_not(dstain.utils.histo.grey(img))
            m, t = dstain.utils.histo.otsu(img, ng, path_hist, lower=700)
            m = np.logical_and(m, ng)

            # Save cache if requested
            if output is not None:
                img.save(path_tiff)
                img.save(path_jpeg)
                PIL.Image.fromarray(m).save(path_mask)
                with open(path_otsu, "w") as f:
                    f.write(str(t))

        image.append(img)
        mask.append(m)
        otsu.append(t)

    return image, mask, otsu


def match_keypoints(filenames, tissue_mask, image, thumbnail_downsample, output, log=None, verbose=True,
        nfeatures=5000, ransacReprojThreshold=25, affine_filter=True):
    """Find keypoints from tissue masks and match keypoints.

    Parameters
    ----------
    filenames : List[str]  TODO: Fix
        list of filenames in this block
    tissue_mask : List[np.array((h, w), dtype=TODO)]
        list of tissue masks
    image : List[np.array((h, w, 3?), dtype=TODO)]
        list of images
    thumbnail_downsample : TODO
        TODO
    output : Path or None
        directory to output images
    verbose : bool
        print progress bar

    Returns
    -------
    keypoint : TODO
        TODO
    descriptor : TODO
        TODO
    transformed_image : TODO
        TODO
    transformed_mask : TODO
        TODO
    homography_matrix : TODO
        TODO
    """
    orb = cv2.ORB_create(nfeatures=nfeatures)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    homography_matrix = []
    transformed_image = []
    transformed_mask = []
    keypoint = []
    descriptor = []
    use_keypoint = []
    n0, m0 = tissue_mask[0].shape

    if output is not None:
        os.makedirs(output, exist_ok=True)

    for (i, (path, basename, stain)) in enumerate(tqdm.tqdm(filenames, desc="Keypoint", disable=not verbose)):
        cache_filename = os.path.join(output, "{}_cache.pkl".format(basename))
        try:
            with open(cache_filename, "rb") as f:
                (homography, trans_image, trans_mask, kp, des, use_kp) = pickle.load(f)
            keypoint.append(kp)
            descriptor.append(des)
            transformed_image.append(trans_image)
            transformed_mask.append(trans_mask)
            homography_matrix.append(homography)
            use_keypoint.append(use_kp)
        except FileNotFoundError:

            kp, des = orb.detectAndCompute((tissue_mask[i] * 255).astype(np.uint8), None)
            keypoint.append(kp)
            descriptor.append(des)

            # Match descriptors.
            matches = bf.match(descriptor[0], des)

            # Sort them in the order of their distance.
            matches = sorted(matches, key=lambda x: x.distance)

            # Put matching points into matrices
            p1 = np.array([keypoint[0][m.queryIdx].pt for m in matches])
            p2 = np.array([keypoint[i][m.trainIdx].pt for m in matches])

            # Find the homography matrix.
            if affine_filter:
                partial_affine, use_kp = cv2.estimateAffinePartial2D(p2, p1, None, cv2.RANSAC, ransacReprojThreshold)
                kp_filter = use_kp.astype(np.bool).squeeze()
                homography, _ = cv2.findHomography(p2[kp_filter, :], p1[kp_filter, :], 0)
            else:
                homography, use_kp = cv2.findHomography(p2, p1, cv2.RANSAC, ransacReprojThreshold)

            use_keypoint.append(use_kp)
            if log is not None:
                log.write("{}: {} keypoints, {} matches, {} ransac\n".format(path, len(kp), len(matches), use_kp.sum()))
                log.flush()

            trans_image = cv2.warpPerspective(np.array(image[i]),
                                              homography, (m0, n0), borderValue=[255, 255, 255])
            transformed_image.append(trans_image)

            trans_mask = cv2.warpPerspective(255 * tissue_mask[i].astype(np.uint8), homography, (m0, n0))
            transformed_mask.append(trans_mask)

            if output is not None:
                for (img0, imgi, suffix) in [(np.array(image[0]), np.array(image[i]), "img"),
                                             (tissue_mask[0],     tissue_mask[i],     "mask")]:
                    fig = _drawKeypoints(imgi, keypoint[i])
                    plt.savefig(os.path.join(output, "{}_{}_keypoints.pdf".format(basename, suffix)))
                    plt.close(fig)

                    fig = _drawMatches(img0, imgi, p1, p2, matches, use_kp)
                    plt.savefig(os.path.join(output, "{}_{}_match.pdf".format(basename, suffix)))
                    plt.close(fig)

                cv2.imwrite(os.path.join(output, "{}_img_warp.jpg".format(basename)), cv2.cvtColor(trans_image, cv2.COLOR_BGR2RGB))
                cv2.imwrite(os.path.join(output, "{}_mask_warp.png".format(basename)), trans_mask)

            # Convert homography to full res
            homography[0:2, 2] *= thumbnail_downsample
            homography[2, 0:2] /= thumbnail_downsample
            homography_matrix.append(homography)

            with open(cache_filename, "wb") as f:
                # pickle has an issue with cv2.Keypoint
                # fix from https://stackoverflow.com/questions/10045363/pickling-cv2-keypoint-causes-picklingerror
                import copyreg

                def _pickle_keypoints(point):
                    return cv2.KeyPoint, (*point.pt, point.size, point.angle,
                                          point.response, point.octave, point.class_id)
                copyreg.pickle(cv2.KeyPoint().__class__, _pickle_keypoints)

                pickle.dump((homography, trans_image, trans_mask, kp, des, use_kp), f)

    return homography_matrix, transformed_image, transformed_mask, keypoint, descriptor, use_keypoint


def get_patches(src, filenames, tissue_mask, transformed_image, window, downsample, output, verbose=True):
    # TODO: lots of arguments are not actually needed
    os.makedirs(output, exist_ok=True)
    cache_name = os.path.join(output, "patch_{}_{}.pkl".format(window, downsample))
    try:
        with open(cache_name, "rb") as f:
            patch = pickle.load(f)
    except FileNotFoundError:
        size = np.array(openslide.open_slide(os.path.join(src, filenames[0][0])).dimensions)
        offset = size % window // 2
        patch = []

        for (x, y) in tqdm.tqdm(itertools.product(range(offset[0], size[0] - window, window), range(offset[1], size[1] - window, window)), desc="Patch selection", disable=not verbose):
            if all([mask[(y // downsample):((y + window) // downsample), (x // downsample):((x + window) // downsample)].mean() > 0.25 for mask in tissue_mask]):
                patch.append((x, y))
        with open(cache_name, "wb") as f:
            pickle.dump(patch, f)

        for (i, (path, basename, stain)) in enumerate(tqdm.tqdm(filenames, desc="Visualization", disable=not verbose)):
            path_box = os.path.join(output, "{}_{}_box".format(basename, window))
            fig, ax = _drawBoxes(np.array(transformed_image[i]), patch, window, downsample, dpi=300)
            plt.savefig(path_box + ".pdf")
            plt.savefig(path_box + ".jpg")
            plt.close(fig)

    return patch


def save_patches(slide, transform, filenames, patch, root, window=512, downsample=1, patches=500, zip_patches=False):
    os.makedirs(root, exist_ok=True)
    for (path, basename, stain) in filenames:
        os.makedirs(os.path.join(root, basename), exist_ok=True)

    def f(arg):
        (j, (x, y)) = arg
        # Load patch from base
        # TODO how to handle if image is corrupted
        if os.path.isfile(os.path.join(root, filenames[-1][1], "{:06d}_{:06d}.jpg".format(x, y))):
            return
        ref = dstain.utils.openslide.read_region_at_mag(slide[0], (x, y), 40, (window, window), downsample=downsample)
        ref.save(os.path.join(root, filenames[0][1], "{:06d}_{:06d}.jpg".format(x, y)))
        for (i, s) in enumerate(slide):
            if i == 0:
                continue
            w = load_warped(slide[i], (x, y), (window, window), transform[i], downsample)
            w.save(os.path.join(root, filenames[i][1], "{:06d}_{:06d}.jpg".format(x, y)))

    patch = sorted(patch)
    random.seed(0)
    random.shuffle(patch)
    if patches is not None and patches >= 0:
        patch = patch[:patches]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        for _ in tqdm.tqdm(executor.map(f, enumerate(patch)), total=len(patch), desc="Saving patches"):
            pass

    if zip_patches:
        # TODO: save in alternate location and move
        # TODO only run if not done
        with zipfile.ZipFile(root + ".zip", "w") as f:
            for (x, y) in tqdm.tqdm(patch, desc="Creating zip"):
                for (_, basename, _) in filenames:
                    f.write(os.path.join(root, basename, "{:06d}_{:06d}.jpg".format(x, y)), os.path.join(basename, "{:06d}_{:06d}.jpg".format(x, y)))


def adjust_keypoints(slide, filenames, transform, otsu, keypoint, downsample, image, transformed_image, tissue_mask, transformed_mask, root, labels=3, adjust_window=8192, adjust_context=1024, adjust_downsample=16):
    os.makedirs(root, exist_ok=True)

    adjust = []
    for kp in keypoint[0][::-1]:
        x = round(kp.pt[0])
        y = round(kp.pt[1])
        s = adjust_window // downsample // 2
        frac = ([tissue_mask[0][max(0, y - s):(y + s), max(0, x - s):(x + s)].mean()] +
                [tm[max(0, y - s):(y + s), max(0, x - s):(x + s)].mean() / 255 for tm in transformed_mask[1:]])
        if all([(0.1 <= f <= 0.9) for f in frac]):
            # TODO: quadratic time
            if all(abs(x[0] - kp.pt[0] + s) + abs(x[1] - kp.pt[1] + s) > adjust_window // downsample for x in adjust):
                adjust.append([round(x - s) for x in kp.pt])

    for i in range(len(image)):
        fig, ax = _drawBoxes(np.array(transformed_image[i]), adjust, adjust_window // downsample, downsample=1, dpi=300)
        for j in range(labels):
            ax.text(*np.array(adjust[j]) + adjust_window // downsample // 2, str(j + 1), horizontalalignment="center", verticalalignment="center", fontfamily="DejaVu Serif", fontsize=6)

            target = "{:06d}_{:06d}_{}_orig.jpg".format(*np.array(adjust[j]) * downsample, i)
            link_name = os.path.join(root, "example_{}_{}.jpg".format(i, j))
            try:
                os.symlink(target, link_name)
            except FileExistsError:
                os.remove(link_name)
                os.symlink(target, link_name)
        plt.savefig(os.path.join(root, "kp_{}.pdf".format(i)))
        plt.close(fig)
        # fig = _drawKeypoints(np.array(image[i]), [kp for kp in keypoint[i] if kp.size > 64])
        # plt.savefig("kp_size_{}.pdf".format(i))
        # plt.close(fig)
    adjust = np.array(sorted(adjust)) * downsample

    def f(arg):
        (j, (x, y)) = arg
        # Load patch from base
        ref = dstain.utils.openslide.read_region_at_mag(slide[0], (x - adjust_context, y - adjust_context), 40, (adjust_window + 2 * adjust_context, adjust_window + 2 * adjust_context), downsample=adjust_downsample)
        ref.save(os.path.join(root, "{:06d}_{:06d}_{}_orig.jpg".format(x, y, 0)))
        for (i, filename) in enumerate(filenames):
            if i == 0:
                continue
            w = load_warped(slide[i], (x - adjust_context, y - adjust_context), (adjust_window + 2 * adjust_context, adjust_window + 2 * adjust_context), transform[i], adjust_downsample)
            w.save(os.path.join(root, "{:06d}_{:06d}_{}_orig.jpg".format(x, y, i)))

            f0 = 255 * (np.sum(ref, 2) < otsu[0]).astype(np.uint8)
            fi = 255 * (np.sum(w, 2) < otsu[i]).astype(np.uint8)
            PIL.Image.fromarray(f0[(adjust_context // adjust_downsample):-(adjust_context // adjust_downsample), (adjust_context // adjust_downsample):-(adjust_context // adjust_downsample)]).save(os.path.join(root, "binary_{:06d}_{:06d}_0.png".format(x, y)))
            PIL.Image.fromarray(fi[(adjust_context // adjust_downsample):-(adjust_context // adjust_downsample), (adjust_context // adjust_downsample):-(adjust_context // adjust_downsample)]).save(os.path.join(root, "binary_{:06d}_{:06d}_{}.png".format(x, y, i)))

            f0 = 255 * scipy.ndimage.binary_closing(f0, iterations=3).astype(np.uint8)
            fi = 255 * scipy.ndimage.binary_closing(fi, iterations=3).astype(np.uint8)
            PIL.Image.fromarray(f0[(adjust_context // adjust_downsample):-(adjust_context // adjust_downsample), (adjust_context // adjust_downsample):-(adjust_context // adjust_downsample)]).save(os.path.join(root, "closed_{:06d}_{:06d}_0.png".format(x, y)))
            PIL.Image.fromarray(fi[(adjust_context // adjust_downsample):-(adjust_context // adjust_downsample), (adjust_context // adjust_downsample):-(adjust_context // adjust_downsample)]).save(os.path.join(root, "closed_{:06d}_{:06d}_{}.png".format(x, y, i)))
            continue

            warper = kornia.HomographyWarper((adjust_window + 2 * adjust_context) // adjust_downsample, (adjust_window + 2 * adjust_context) // adjust_downsample, normalized_coordinates=True, padding_mode="reflection")
            homography = torch.nn.Parameter(torch.eye(3).view(1, 3, 3))
            padded_input = torch.as_tensor(fi.reshape(1, 1, (adjust_window + 2 * adjust_context) // adjust_downsample, (adjust_window + 2 * adjust_context) // adjust_downsample), dtype=torch.float32)
            padded_output = torch.as_tensor(f0.reshape(1, 1, (adjust_window + 2 * adjust_context) // adjust_downsample, (adjust_window + 2 * adjust_context) // adjust_downsample), dtype=torch.float32)
            optim = torch.optim.SGD([homography], lr=1e-7)
            # kernel = 5
            # blur = torch.nn.Conv2d(1, 1, (kernel, kernel), padding=(kernel // 2), bias=False, padding_mode="reflection")
            # blur.weight.data[:] = 1 / (kernel ** 2)
            # padded_input = blur(blur(padded_input))
            # padded_output = blur(blur(padded_output))
            # PIL.Image.fromarray(padded_output.detach().numpy()[0, 0, :, :].astype(np.uint8)).save(os.path.join(root, "padded_output.png"))
            # PIL.Image.fromarray(padded_input.detach().numpy()[0, 0, :, :].astype(np.uint8)).save( os.path.join(root, "padded_input.png"))

            for j in range(250):
                asd = time.time()
                warp = warper(padded_input, homography)
                PIL.Image.fromarray(warp.detach().numpy()[0, 0, (adjust_context // adjust_downsample):-(adjust_context // adjust_downsample), (adjust_context // adjust_downsample):-(adjust_context // adjust_downsample)].astype(np.uint8)).resize((512, 512)).save(os.path.join(root, "warp_{:04d}.png".format(j)))
                loss = ((warp - padded_output)[:, :, (adjust_context // adjust_downsample):-(adjust_context // adjust_downsample), (adjust_context // adjust_downsample):-(adjust_context // adjust_downsample)] ** 2).mean()
                optim.zero_grad()
                loss.backward(retain_graph=True)
                print("{: 3d}: {}".format(j, loss.item()))
                optim.step()
                homography.data[:, :, 0:2] = torch.eye(3, 2)
                homography.data[:, 2, 2] = 1
                # print(homography)
                # print(time.time() - asd)

            # base = np.array(list(itertools.product(np.linspace(x - 5, x + adjust_window + 5, 5), np.linspace(y - 5, y + adjust_window + 5, 5))))
            # coord = transform[i].inverse(base) + homography[0, 0:2, 2].detach().numpy() * adjust_window / 2
            # loc = coord.min(0).astype(np.int)
            # size = (coord.max(0) - coord.min(0)).astype(np.int)
            # # print("B: ", time.time() - start); start = time.time()
            # img = dstain.utils.openslide.read_region_at_mag(slide[i], loc, 40, size)
            # transform_local = skimage.transform.ProjectiveTransform()
            # transform_local.estimate(coord - loc, base - base[0, :])  # TODO: would be nice to compute analytically
            # w = skimage.transform.warp(np.array(img), transform_local.inverse, output_shape=(adjust_window, adjust_window), cval=1)
            # w = PIL.Image.fromarray((w * 255).astype(np.uint8))
            # w.save(os.path.join(root, "{:06d}_{:06d}_{}_refine.jpg".format(x, y, i)))
            h = transform[i].params.copy()
            h[0:2, 2] -= homography[0, 0:2, 2].detach().numpy() * (adjust_window + 2 * adjust_context) / 2
            t = skimage.transform.ProjectiveTransform(h)
            w = load_warped(slide[i], (x - adjust_context, y - adjust_context), (adjust_window + 2 * adjust_context, adjust_window + 2 * adjust_context), t, adjust_downsample)
            w.save(os.path.join(root, "{:06d}_{:06d}_{}_refine.jpg".format(x, y, i)))

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        for _ in tqdm.tqdm(map(f, enumerate(adjust)), total=len(adjust)):
            pass


def adjust_coarse(block, tissue_mask, image, homography_matrix):
    n0, m0 = tissue_mask[0].shape
    transformed_image = []
    new_homography_matrix = []
    for (i, (filename, mask)) in enumerate(zip(tqdm.tqdm(filenames), tissue_mask)):

        n = max(tissue_mask[0].shape[0], tissue_mask[i].shape[0])
        m = max(tissue_mask[0].shape[1], tissue_mask[i].shape[1])
        # print("n, m", n, m)
        warper = kornia.HomographyWarper(n, m, normalized_coordinates=True)
        # print("tissue_mask 0", tissue_mask[0].shape)
        # print("tissue_mask i", tissue_mask[i].shape)
        transform = skimage.transform.ProjectiveTransform(homography_matrix[i])
        base = np.array(list(itertools.product(np.linspace(0, n, 10), np.linspace(0, m, 10))))
        coord = transform(base)
        coord = 2 * coord / np.array([m, n]) - 1
        base = 2 * base / np.array([m, n]) - 1
        transform.estimate(coord, base) # TODO: would be nice to compute analytically
        homography = torch.nn.Parameter(torch.as_tensor(transform.params, dtype=torch.float).view(1, 3, 3))
        # homography = torch.nn.Parameter(torch.as_tensor(homography, dtype=torch.float).view(1, 3, 3))
        # homography = torch.nn.Parameter(torch.eye(3).view(1, 3, 3))
        padded_input = torch.zeros((1, 1, n, m))
        padded_input[0, 0, :tissue_mask[i].shape[0], :tissue_mask[i].shape[1]] = torch.as_tensor(tissue_mask[i], dtype=torch.float32)
        padded_output = torch.zeros((1, 1, n, m))
        padded_output[0, 0, :tissue_mask[0].shape[0], :tissue_mask[0].shape[1]] = torch.as_tensor(tissue_mask[0], dtype=torch.float32)
        optim = torch.optim.SGD([homography], lr=1e-8, momentum=0.9)
        PIL.Image.fromarray(padded_output.detach().numpy()[0, 0, :, :].astype(np.uint8)).save("padded_output.png")
        PIL.Image.fromarray(padded_input.detach().numpy()[0, 0, :, :].astype(np.uint8)).save("padded_input.png")
        # M = torch.nn.Parameter(torch.zeros((1, 2, 16)))
        # degree = (M.shape[-1] // 2) - 1
        # X = np.concatenate([coord ** i for i in range(degree + 1)], axis=1)
        # M.data = torch.as_tensor(np.expand_dims(np.matmul(np.linalg.pinv(X), base).transpose(), 0), dtype=torch.float32)
        # M.data[0, 0, 2] = 1.00
        # M.data[0, 1, 3] = 1
        # optim = torch.optim.SGD([M], lr=1e-8, momentum=0.9)

        for j in range(10):
            # warp = dstain.utils.warp.polynomial_warp(padded_input, M, (n, m))
            warp = warper(padded_input, homography)
            PIL.Image.fromarray(warp.detach().numpy()[0, 0, :, :].astype(np.uint8)).save("warp_{:04d}.png".format(j))
            loss = ((warp - padded_output) ** 2).mean()
            optim.zero_grad()
            loss.backward()
            print("{: 3d}: {}".format(j, loss.item()))
            optim.step()
        # print("warp", asdf.shape)
        homography = homography.detach().numpy().reshape(3, 3)
        transform = skimage.transform.ProjectiveTransform(homography)
        base = np.array(list(itertools.product(np.linspace(0, n, 10), np.linspace(0, m, 10))))
        base = 2 * base / np.array([m, n]) - 1
        coord = transform(base)
        base = (base + 1) / 2 * np.array([m, n])
        coord = (coord + 1) / 2 * np.array([m, n])
        transform.estimate(coord, base) # TODO: would be nice to compute analytically
        homography = transform.params
        new_homography_matrix.append(homography)

        transformed = cv2.warpPerspective(np.array(image[i]),
                                          homography, (m0, n0), borderValue=[255, 255, 255])
        transformed_image.append(transformed)
    return transformed_image, new_homography_matrix


def load_warped(slide, loc, size, transform, downsample=1):
    (x, y) = loc

    base = np.array(list(itertools.product(np.linspace(x - 5, x + size[1] + 5, 5), np.linspace(y - 5, y + size[0] + 5, 5))))
    coord = transform.inverse(base)
    orig_loc = coord.min(0).astype(np.int)
    orig_size = (coord.max(0) - coord.min(0)).astype(np.int)
    img = dstain.utils.openslide.read_region_at_mag(slide, orig_loc, 40, orig_size, downsample=downsample)
    # h = homography.copy()
    # h[0, 2] =  h[0, 2] + h[0, 0] * loc[0] + h[0, 1] * loc[1] - x
    # h[1, 2] =  h[1, 2] + h[1, 0] * loc[0] + h[1, 1] * loc[1] - y
    # h[2, 2] =  h[2, 2] + h[2, 0] * loc[0] + h[2, 1] * loc[1]
    # transform_local = skimage.transform.ProjectiveTransform(h)
    transform_local = skimage.transform.ProjectiveTransform()
    transform_local.estimate((coord - orig_loc) / downsample, (base - base[0, :]) / downsample)  # TODO: would be nice to compute analytically

    w = skimage.transform.warp(np.array(img), transform_local.inverse, output_shape=list(map(lambda x: x / downsample, size)), cval=1)
    w = PIL.Image.fromarray((w * 255).astype(np.uint8))

    return w

def _drawMatches(img0, img1, p0, p1, matches, use_keypoint, show_rejected=False, dpi=300, gap=10):
    """Cleaner version of cv2.drawMatches"""
    use_keypoint = use_keypoint.astype(np.bool).reshape(-1)
    size = (max(img0.shape[0], img1.shape[0]), (img0.shape[1] + img1.shape[1] + gap))
    fig = plt.figure(figsize=(size[1] / dpi, size[0] / dpi), frameon=False, dpi=dpi)
    img = np.full(size + (() if len(img0.shape) == 2 else (3,)), 255, dtype=img0.dtype)
    offset0 = max(0, (img1.shape[0] - img0.shape[0]) // 2)
    offset1 = max(0, (img0.shape[0] - img1.shape[0]) // 2)
    img[offset0:(img0.shape[0] + offset0), 0:img0.shape[1]] = img0
    img[offset1:(img1.shape[0] + offset1), (gap + img0.shape[1]):] = img1
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")

    plt.figimage(img, cmap="gray", zorder=-1)
    # import code; code.interact(local=dict(globals(), **locals()))
    subsample = True
    if subsample:
        idx = np.random.choice(np.where(use_keypoint)[0], 100)
        mask = np.zeros_like(use_keypoint)
        mask[idx] = True
        ax.plot([i for (a, b) in zip(p0[np.logical_and(use_keypoint, mask), 0], p1[np.logical_and(use_keypoint, mask), 0]) for i in [a, b + img0.shape[1] + gap, math.nan]],
                [i for (a, b) in zip(p0[np.logical_and(use_keypoint, mask), 1], p1[np.logical_and(use_keypoint, mask), 1]) for i in [a + offset0, b + offset1, math.nan]],
                color="black",
                linestyle="-",
                linewidth=0.1,
                zorder=2)
    else:
        ax.plot([i for (a, b) in zip(p0[use_keypoint, 0], p1[use_keypoint, 0]) for i in [a, b + img0.shape[1] + gap, math.nan]],
                [i for (a, b) in zip(p0[use_keypoint, 1], p1[use_keypoint, 1]) for i in [a + offset0, b + offset1, math.nan]],
                color="black",
                linestyle="-",
                linewidth=0.1,
                zorder=2)

    if show_rejected:
        ax.plot([i for (a, b) in zip(p0[~use_keypoint, 0], p1[~use_keypoint, 0]) for i in [a, b + img0.shape[1] + gap, math.nan]],
                [i for (a, b) in zip(p0[~use_keypoint, 1], p1[~use_keypoint, 1]) for i in [a + offset0, b + offset1, math.nan]],
                color="red",
                linestyle=(0, (1, 5)),
                linewidth=0.1,
                zorder=1)
    # for i in range(len(matches)):
    #     ax.plot([p0[i, 0], p1[i, 0] + img0.shape[1] + gap], [p0[i, 1] + offset0, p1[i, 1] + offset1], linewidth=0.1,
    #              linestyle=("-" if use_keypoint[i] else (0, (1, 5))),
    #              # transform=fig.transFigure,
    #              clip_on=False,
    #              zorder=(2 if use_keypoint[i] else 1)
    #              )
    ax.axis([0, size[1], size[0], 0])

    return fig


def _drawBoxes(img, patch, w, downsample=1, dpi=300):
    fig = plt.figure(figsize=(img.shape[1] / dpi, img.shape[0] / dpi), frameon=False, dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")

    plt.figimage(img, cmap="gray", zorder=-1)
    ax.plot([i / downsample for (x, y) in patch for i in [x, x, x + w, x + w, x, math.nan]],
            [j / downsample for (x, y) in patch for j in [y, y + w, y + w, y, y, math.nan]],
            linewidth=0.1,
            color="k",
            clip_on=False,
            zorder=2
            )
    ax.axis([0, img.shape[1], img.shape[0], 0])

    if False:
        # bryanhe (3/28/2020): old code for rendering the boxes on original space
        buf = 2
        orig = np.array(image[i])
        for (j, (x, y)) in enumerate(patch):
            rr, cc = skimage.draw.rectangle_perimeter(np.array((y, x), dtype=np.int) / downsample + buf, np.array((y + window, x + window), dtype=np.int) / downsample - buf)
            # transformed_image[i][rr, cc, :] = 0

            base = np.array([[x, y],
                             [x, y + window],
                             [x + window, y],
                             [x + window, y + window]])
            coord = transform.inverse(base)
            loc = coord.min(0).astype(np.int)
            size = (coord.max(0) - coord.min(0)).astype(np.int)
            cc, rr = transform.inverse(np.vstack((cc, rr)).transpose()).transpose()  # TODO: This needs do be done without scaling
            rr = rr.astype(np.int)
            cc = cc.astype(np.int)
            valid = np.logical_and.reduce((0 <= rr, rr < orig.shape[0], 0 <= cc, cc < orig.shape[1]))
            orig[rr[valid], cc[valid], :] = 0

        PIL.Image.fromarray(orig).save(path_orig_box)

    return fig, ax


def _drawKeypoints(img, keypoint, dpi=300):
    """Cleaner version of cv2.drawMatches"""
    size = img.shape
    fig = plt.figure(figsize=(size[1] / dpi, size[0] / dpi), frameon=False, dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")

    plt.figimage(img, cmap="gray", zorder=-1)
    if isinstance(keypoint[0], cv2.KeyPoint):
        keypoint = [kp.pt for kp in keypoint]
    ax.scatter(*zip(*keypoint),
               zorder=2,
               s=1, edgecolor="none")
    ax.axis([0, size[1], size[0], 0])

    return fig

