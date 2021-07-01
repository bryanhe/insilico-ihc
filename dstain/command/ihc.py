import collections
import concurrent.futures
import glob
import math
import os
import pickle
import sys

import click
import openslide
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import torch
import torchvision
import tqdm

import dstain


@click.command()
@click.option("--wsi", type=click.Path(exists=True),
              default=dstain.config.DATA_RAW)
@click.option("--annotation", type=click.Path(),
              default=os.path.join(dstain.config.ANNOTATION))
@click.option("--output", type=click.Path(),
              default=os.path.join(dstain.config.OUTPUT, "ihc"))
@click.option("--patches", type=click.Path(exists=True), default=None)
@click.option("--patch_window", type=int, default=2048)
@click.option("--patch_downsample", type=int, default=4)
@click.option("--model_name", type=str, default="densenet121")
@click.option("--lr", type=float, default=1e-5)
@click.option("--momentum", type=float, default=0.9)
@click.option("--weight_decay", type=float, default=0)
@click.option("--lr_step_period", type=int, default=50)
@click.option("--epochs", type=int, default=100)
@click.option("--batch_size", type=int, default=32)
@click.option("--num_workers", type=int, default=4)
@click.option("--window_full", type=int, default=4096)
@click.option("--window", type=int, default=2048)
@click.option("--downsample", type=int, default=4)
@click.option("--resume/--scratch", default=True)
@click.option("--cvat/--no_cvat", default=False)
@click.option("--keep_checkpoints/--remove_checkpoints", default=True)
def ihc(wsi, annotation, output, patches, patch_window, patch_downsample, model_name, lr, momentum, weight_decay, lr_step_period, epochs, batch_size, num_workers, window_full, window, downsample, resume, cvat, keep_checkpoints):
    """Train models to classify IHC patches.

    \b
    Parameters
    ----------
    wsi : str
        Directory containing whole-slide images.
    annotations : str
        Directory containing selected patches (as pickle files) and CVAT
        annotations (as zip files).
    output : str
        Directory to write patches, annotations, checkpoints, and logs.
    patches : str or None
        Directory containing registered slides to label.
    patch_window : TODO
        TODO
    patch_downsample : TODO
        TODO
    model_name : str
        Name of model to train on (in torchvision.models)
    lr : float
        Learning rate
    momentum : float
        Momentum factor
    weight_decay : float
        Weight decay (L2 penalty)
    lr_step_period : int
        Period of learning rate
    epochs : int
        Number of epochs to train
    batch_size : int
        Number of samples to load per batch.
    num_workers : int
        Number of subprocesses to use for data loading. If 0, data is loaded
        in the main process.
    window_full : int
        Size of the patches to save (in full resolution)
    window: int
        size of the patches to give to model after cropping from window_full
        (in full resolution)
    downsample : int
        Downsampling factor for images
    resume : bool
        If True, continue from last checkpoint
    cvat : bool
        If True, save images for CVAT
    keep_checkpoints : bool
        If True, keep all intermediate checkpoints; if False, keep only best
        and final checkpoint

    \b
    Notes
    -----
    window_full and window refer to the size of the patch at full resolution,
    not the actual size of the image saved (e.g. if w=16384, h=16384,
    downsample=4; then the actual saved image size will be 16384 / 4 = 4096).
    """
    #########################
    # Save patches for CVAT #
    #########################

    # Parameters for the size of the patches to save.
    # This can be changed, but the existing annotations from CVAT will break.
    WIDTH = 16384
    HEIGHT = 16384
    DOWNSAMPLE = 4

    # Patches in os.path.join(annotation, "select") are obtained from the
    # dstain.commands.annotate interface.
    if cvat:
        print("Saving patches for CVAT")
        save_patches(wsi,
                     os.path.join(annotation, "select"),
                     os.path.join(annotation, "cvat"),
                     w=WIDTH, h=HEIGHT, downsample=DOWNSAMPLE, num_workers=num_workers)

    #############################
    # Save dataset for training #
    #############################
    save_patches(wsi,
                 os.path.join(annotation, "select"),
                 os.path.join(output, "dataset", "images"),
                 w=WIDTH, h=HEIGHT, downsample=downsample, grid=window_full, num_workers=num_workers)

    for (stain, index) in [("aB", {"Unstained": 0, "Artifact": 0, "Plaque": 1}),
                           ("T", {"Unstained": 0, "Artifact": 0, "Tangle": 1, "Neuritic Plaque": 2})]:
        dstain.utils.cvat.save_annotations(os.path.join(annotation, stain + ".zip"), os.path.join(output, "dataset"), index=index, downsample=downsample, grid=window_full, num_workers=num_workers)

    device = torch.device("cuda")

    for (stain, target) in [
        ("T", ["Tangle", "Neuritic Plaque"]),
        ("aB", ["Plaque"]),
    ]:
        print(80 * "=")
        print("Running on {}".format(stain))

        ############
        # Training #
        ############
        dataset_dir = os.path.join(output, "dataset")

        # Compute mean, std, and number times each class appears
        try:
            with open(os.path.join(dataset_dir, "labels", stain + "_statistics.pkl"), "rb") as f:
                (mean, std, count) = pickle.load(f)
        except FileNotFoundError:
            mean, std, count = dstain.utils.get_mean_and_std(
                dstain.datasets.AnnotatedIHC(
                    dataset_dir, stain, len(target), "train",
                    transforms=dstain.transforms.Compose([
                        dstain.transforms.RandomCrop(window // downsample),
                        dstain.transforms.ToTensor(),
                    ])),
                class_index=1, samples=None, batch_size=batch_size, num_workers=num_workers)
            with open(os.path.join(dataset_dir, "labels", stain + "_statistics.pkl"), "wb") as f:
                pickle.dump((mean, std, count), f)
        count = np.array([c[True] / (c[True] + c[False]) for c in count])

        # Set up dataset
        transforms = [
            dstain.transforms.RandomCrop(window // downsample),
            dstain.transforms.RandomHorizontalFlip(0.5),
            dstain.transforms.RandomVerticalFlip(0.5),
            dstain.transforms.RandomApply([dstain.transforms.RandomRotation((90, 90))]),
            dstain.transforms.ToTensor(),
            dstain.transforms.Normalize(mean=mean, std=std)
        ]
        transforms = dstain.transforms.Compose(transforms)
        dataset = {split: dstain.datasets.AnnotatedIHC(dataset_dir, stain, len(target), split, transforms=transforms) for split in ["train", "valid", "test"]}

        # Set up model
        model = torchvision.models.__dict__[model_name](pretrained=True)
        model.classifier = torch.nn.Linear(1024, len(target))
        model.classifier.weight.data[:] = 0
        model.classifier.bias.data[:] = torch.as_tensor(np.log(count))  # TODO: dtypes may not match?
        if device.type == "cuda":
            model = torch.nn.DataParallel(model)
        model.to(device)

        # Set up optimizer
        optim = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        sched = torch.optim.lr_scheduler.StepLR(optim, lr_step_period)

        os.makedirs(os.path.join(output, "log"), exist_ok=True)
        skip_header = (resume and os.path.isfile(os.path.join(output, "log", stain + ".csv")))
        with open(os.path.join(output, "log", stain + ".csv"), "a") as log:
            if not skip_header:
                log.write("Epoch,Split,{}\n".format(",".join("{} {}".format(t, m) for m in ["CE", "AUC"] for t in target)))
                log.write("Baseline,train,{}\n".format(",".join([str(-p * math.log(p) - (1 - p) * math.log(1 - p)) for p in count] + ["0.5"] * len(target))))
                log.write("Baseline,valid,{}\n".format(",".join([str(-p * math.log(p) - (1 - p) * math.log(1 - p)) for p in count] + ["0.5"] * len(target))))
                log.flush()

            epoch = 0
            best_epoch = None
            best_auc = -math.inf
            if resume:
                # Attempt to load from checkpoint
                filenames = glob.glob(os.path.join(output, "checkpoints", stain, "epoch_*.pt"))
                if filenames != []:
                    checkpoint = sorted(filenames, key=dstain.utils.alphanum_key)[-1]
                    print("Loading from {}".format(checkpoint))
                    checkpoint = torch.load(checkpoint)
                    model.load_state_dict(checkpoint["model_dict"])
                    optim.load_state_dict(checkpoint["optim_dict"])
                    sched.load_state_dict(checkpoint["sched_dict"])
                    epoch = checkpoint["epoch"] + 1
                    best_epoch = checkpoint["best_epoch"]
                    best_auc = checkpoint["best_auc"]

            for epoch in range(epoch, epochs):
                for s in ["train", "valid"]:
                    print("Epoch #{} {}:".format(epoch, s))
                    ds = dataset[s]
                    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=(device.type == "cuda"), drop_last=True)

                    interrupted, loss, auc, y_true, y_score, filename = run_epoch(model, optim, loader, device, s == "train", count)

                    if interrupted:
                        sys.exit(1)
                    else:
                        log.write((",".join("{}" for _ in range(2 + 2 * loss.shape[0]))).format(epoch, s, *loss, *auc) + "\n")
                        log.flush()

                        if s == "valid":
                            if auc.mean() > best_auc:
                                best_epoch = epoch
                                best_auc = auc.mean()

                            sched.step()

                            # fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_true, [s - bg for (bg, s) in y_score])
                            # thresh = thresholds[(tpr - fpr).argmax()]

                            # TODO: checkpoint without dataparallel (and on cpu?)
                            save = {
                                "model_name": model_name,
                                "stain": stain,
                                "targets": target,
                                "model_dict": model.state_dict(),
                                "optim_dict": optim.state_dict(),
                                "sched_dict": sched.state_dict(),
                                "epoch": epoch,
                                "best_epoch": best_epoch,
                                "best_auc": best_auc,
                                "loss": loss,
                                "auc": auc,
                                "window": window,
                                "downsample": downsample,
                                "mean": mean,
                                "std": std,
                                # "thresh": thresh,
                            }
                            os.makedirs(os.path.join(output, "checkpoints", stain), exist_ok=True)
                            torch.save(save, os.path.join(output, "checkpoints", stain, "epoch_{}.pt".format(epoch)))
                            if not keep_checkpoints:
                                try:
                                    os.remove(os.path.join(output, "checkpoints", stain, "epoch_{}.pt".format(epoch - 1)))
                                except FileNotFoundError:
                                    pass
                            if best_epoch == epoch:
                                torch.save(save, os.path.join(output, "checkpoints", "{}.pt".format(stain)))

        ###################
        # Load best model #
        ###################
        checkpoint = os.path.join(output, "checkpoints", "{}.pt".format(stain))
        checkpoint = torch.load(checkpoint)
        model.load_state_dict(checkpoint["model_dict"])
        # thresh = checkpoint["thresh"]
        print("Best model from epoch #{}".format(checkpoint["best_epoch"]))

        ####################
        # Plotting Results #
        ####################

        # Training Curve
        with open(os.path.join(output, "log", stain + ".csv"), "r") as log:
            res = collections.defaultdict(list)
            for line in log:
                if line.startswith("Epoch,Split"):
                    continue
                epoch, split, *r = line.split(",")
                if epoch == "Baseline":
                    epoch = 0
                else:
                    epoch = int(epoch) + 1
                assert len(res[split]) == epoch
                r = list(map(float, r))
                res[split].append(r)

        ce = {}
        auc = {}
        for split in res:
            ce[split] = np.array(res[split])[:,:len(target)]
            auc[split] = np.array(res[split])[:,len(target):]

        dstain.utils.latexify()
        os.makedirs(os.path.join(output, "fig"), exist_ok=True)
        for (i, t) in enumerate(target):
            fig = plt.figure(figsize=(2.0, 2.0))
            plt.plot(np.arange(ce["train"].shape[0]), ce["train"][:, i], "--", color="k", linewidth=1, label="Training")
            plt.plot(np.arange(ce["valid"].shape[0]), ce["valid"][:, i], "-", color="k", linewidth=1, label="Validation")
            plt.legend(loc="best")
            plt.title("{}: {}".format(stain, t))
            plt.xlabel("Epochs")
            plt.ylabel("Cross-Entropy")
            # TODO y axis start from 0
            plt.tight_layout()
            plt.savefig(os.path.join(output, "fig", "ce_{}_{}.pdf".format(stain, t.lower().replace(" ", "-"))))
            plt.savefig(os.path.join(output, "fig", "ce_{}_{}.png".format(stain, t.lower().replace(" ", "-"))))
            plt.close(fig)

            fig = plt.figure(figsize=(2.0, 2.0))
            plt.plot(np.arange(auc["train"].shape[0]), auc["train"][:, i], "--", color="k", linewidth=1, label="Training")
            plt.plot(np.arange(auc["valid"].shape[0]), auc["valid"][:, i], "-", color="k", linewidth=1, label="Validation")
            plt.legend(loc="best")
            plt.title("{}: {}".format(stain, t))
            plt.xlabel("Epochs")
            plt.ylabel("AUC")
            # TODO y axis start from 0.5
            plt.tight_layout()
            plt.savefig(os.path.join(output, "fig", "auc_{}_{}.pdf".format(stain, t.lower().replace(" ", "-"))))
            plt.savefig(os.path.join(output, "fig", "auc_{}_{}.png".format(stain, t.lower().replace(" ", "-"))))
            plt.close(fig)

        # Evaluate model on test
        print("Evaluating best model")
        ds = dataset["test"]
        loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=(device.type == "cuda"), drop_last=True)

        interrupted, loss, auc, y_true, y_score, filename = run_epoch(model, None, loader, device, False, count)

        if interrupted:
            sys.exit(1)

        # Plot ROC curve
        for (i, t) in enumerate(target):
            fpr, tpr, thresholds = sklearn.metrics.roc_curve(np.array(y_true)[:, i], np.array(y_score)[:, i])
            # TODO: put bootstrap on ROC

            fig = plt.figure(figsize=(2.0, 2.0))
            plt.plot(fpr, tpr, "-", color="k", linewidth=1)
            # index = (thresholds > 0).argmin()
            # plt.scatter([fpr[index]], [tpr[index]], color="k", s=9)
            plt.plot([0, 1], [0, 1], color='k', linewidth=1, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.0])
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("{}: {} (AUROC: {:.2f})".format(stain, t, sklearn.metrics.roc_auc_score(np.array(y_true)[:, i], np.array(y_score)[:, i])))
            # plt.title("{}: {}".format(stain, t, sklearn.metrics.roc_auc_score(np.array(y_true)[:, i], np.array(y_score)[:, i])))
            plt.tight_layout()
            plt.savefig(os.path.join(output, "fig", "roc_{}_{}.pdf".format(stain, t.lower().replace(" ", "-"))))
            plt.savefig(os.path.join(output, "fig", "roc_{}_{}.png".format(stain, t.lower().replace(" ", "-"))))
            plt.close(fig)
            print("{}: {}".format(stain, t))
            print("AUC: ", sklearn.metrics.roc_auc_score(np.array(y_true)[:, i], np.array(y_score)[:, i]))
            # print("Precision: ", sklearn.metrics.precision_score(y_true, y_score > thresh))
            # print("Recall: ", sklearn.metrics.recall_score(y_true, y_score > thresh))

            # Save latex file with images sorted by prediction
            # TODO: the random cropping makes the list a bit weird
            os.makedirs(os.path.join(output, "latex"), exist_ok=True)
            tex = dstain.utils.render_latex(
                os.path.join(dstain.config.ROOT, "latex", "ihc.tex"),
                root=os.path.join("..", "dataset", "images", stain),
                files=sorted(zip([y[i] for y in y_score], filename))
            )
            with open(os.path.join(output, "latex", "{}_{}.tex".format(stain, t.lower().replace(" ", "-"))), "w") as tex_file:
                tex_file.write(tex)

        ##########################################
        # Apply best model to H&E => IHC dataset #
        ##########################################
        if patches is not None:
            if patch_downsample < downsample:
                print("WARNING")
            transform = [
                torchvision.transforms.Resize(patch_window // downsample),
                torchvision.transforms.CenterCrop(window // downsample),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=mean, std=std)
            ]
            transform = torchvision.transforms.Compose(transform)
            ds = dstain.datasets.UnlabeledIHC(
                patches,
                stain,
                window=patch_window,
                downsample=patch_downsample,
                transform=transform,
                model_timestamp=os.path.getmtime(os.path.join(output, "checkpoints", "{}.pt".format(stain)))
            )
            loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, num_workers=num_workers, shuffle=False, pin_memory=(device.type == "cuda"))

            soft = 0.
            positive = 0
            total = 0
            with torch.set_grad_enabled(False), tqdm.tqdm(total=len(loader), desc="Generating") as pbar, dstain.utils.GracefulInterruptHandler() as interrupt_handler:
                model.train(False)
                for (X, des) in loader:
                    X = X.to(device)
                    yhat = model(X)

                    yhat = yhat.cpu().numpy()

                    positive += (yhat > 0).sum(0)
                    soft += (1 / (1 + np.exp(-yhat))).sum(0)
                    total += yhat.shape[0]

                    for (i, (filename, p)) in enumerate(zip(des, yhat)):
                        os.makedirs(os.path.dirname(filename), exist_ok=True)
                        with open(filename, "wb") as f:
                            pickle.dump(p, f)

                    pbar.set_postfix_str("; ".join("{}: {:.3f} positive ({:.3f} soft label)".format(t, p / total, s / total) for (t, p, s) in zip(target, positive, soft)))
                    pbar.update()

                    if interrupt_handler.interrupted:
                        sys.exit(1)


def save_patches(wsi, selections, output, w=16384, h=16384, downsample=4, grid=None, num_workers=4):
    """Save images centered on selected points.

    The points are selected from the dstain.commands.annotate interface.

    Parameters
    ----------
    wsi : str
        directory containing whole-slide images
    selections : str
        directory containing the selected points
    output : str
        directory to write the patches to
    w : int
        width of the patch (in full resolution)
    h : int
        height of the patch (in full resolution)
    downsample : int
        downsampling factor
    grid : int or None
        if None, save the patch as one image; if int, cut the patch into
        smaller subpatches
    num_workers : int
        Number of subprocesses to use for data loading. If 0, data is loaded
        in the main process.

    Notes
    -----
    w, h, and grid refer to the size of the patch at full resolution (see
    dstain.command.ihc for details).

    If grid is None, the patches are saved to
        {path}--{x}-{y}.jpg
    If grid is an int, the patches are saved to
        {path}--{x}-{y}_{x0}_{y0}.jpg
    where
       - path is the name of the whole-slide image (directory separators
         replaced with '-')
       - x, y are the coords on the center of the patch in raw pixel space
       - x0, y0 are the offsets of the grid in raw pixel space (starts at 0, 0
         and increments in steps of grid)
    """
    # The selections directory contains a directory for each stain with
    # selected patches. Loop through the different stains.
    for stain in os.listdir(selections):
        os.makedirs(os.path.join(output, stain), exist_ok=True)

        # Get the pickles containing the patches
        # Each root (directory) in the os.walk represents one whole-slide
        # image, which contains (potentially multiple) pickle files, labeled
        # by date/time it was created. If multiple, then last one is taken.
        pickle_names = sorted(os.path.join(root, sorted(files)[-1]) for root, _, files in os.walk(os.path.join(selections, stain)) if files != [])

        # Read the WSI paths and pixels from the pickles
        n_patches = 0
        path_pixel = []
        for pkl in pickle_names:
            with open(pkl, "rb") as f:
                data = pickle.load(f)
                path = data["path"]
                pixel = data["raw"]
            path_pixel.append((path, pixel))
            n_patches += len(pixel)
        print("{}: {} slides, {} patches".format(stain, len(pickle_names), n_patches))

        def save_patch(path_pixel):
            # Save a single patch
            # This function is to allow multiprocessing with concurrent.futures

            slide = None
            path, pixel = path_pixel

            for p in pixel:
                base = os.path.join(output, stain, "{}--{}_{}".format(path.replace("/", "-"), *p))
                p[0] -= w // 2
                p[1] -= h // 2

                if grid is None:
                    # Save whole patch as one image
                    if not os.path.isfile(base + ".jpg"):
                        if slide is None:
                            # Open WSI only if patch is missing
                            slide = openslide.open_slide(os.path.join(wsi, path))
                        img = dstain.utils.openslide.read_region_at_mag(
                            slide, p, 40, (w, h), downsample)
                        img.save(base + ".jpg")
                else:
                    # Split patch into grid
                    for p0 in range(0, w, grid):
                        for p1 in range(0, h, grid):
                            img_name = (base + "_{}_{}.jpg".format(p0, p1))
                            if not os.path.isfile(img_name):
                                if slide is None:
                                    # Open WSI only if patch is missing
                                    slide = openslide.open_slide(os.path.join(wsi, path))
                                img = dstain.utils.openslide.read_region_at_mag(
                                    slide, (p[0] + p0, p[1] + p1), 40, (grid, grid), downsample)
                                img.save(img_name)

            return len(pixel)

        n_patches = 0
        with tqdm.tqdm(total=len(path_pixel), desc="Saving patches for {}".format(stain)) as pbar:
            if num_workers != 0:
                with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
                    for c in executor.map(save_patch, path_pixel):
                        n_patches += c
                        pbar.set_postfix_str("{} patches".format(n_patches))
                        pbar.update()
            else:
                for c in map(save_patch, path_pixel):
                    n_patches += c
                    pbar.set_postfix_str("{} patches".format(n_patches))
                    pbar.update()


def run_epoch(model, optim, loader, device, train, count, weight=None):
    total = 0.
    n = 0

    y_true = []
    y_score = []
    filename = []

    with torch.set_grad_enabled(train), dstain.utils.GracefulInterruptHandler() as interrupt_handler, tqdm.tqdm(total=len(loader)) as pbar:
        model.train(train)
        for (X, y, fn) in loader:
            y_true.extend(y.numpy().tolist())

            X = X.to(device)
            y = y.to(device)
            yhat = model(X)

            loss = torch.nn.functional.binary_cross_entropy_with_logits(yhat, y.type(torch.float), reduction="none").sum(0)
            n += y.shape[0]
            total += loss.detach().cpu().numpy()
            y_score.extend(yhat.detach().cpu().numpy().tolist())
            filename.extend(fn)

            pbar.set_postfix_str(
                " ".join("CE: {:.3f} ({:.3f}) / {:.3f}, Acc: {:.3f} ({:.3f}) / {:.3f}, AUC: {:.3f}".format(
                total[i] / n,
                loss[i].item() / y.shape[0],
                -count[i] * math.log(count[i]) - (1 - count[i]) * math.log(1 - count[i]),
                ((np.array(y_score)[:, i] > 0) == np.array(y_true)[:, i]).mean(),
                ((yhat[:, i] > 0) == y[:, i]).type(torch.float).mean().item(),
                max(count[i], 1 - count[i]),
                safe_roc_auc_score(np.array(y_true)[:, i], np.array(y_score)[:, i]))
                for i in range(total.shape[0])
                )
            )
            pbar.update()

            if train:
                optim.zero_grad()
                loss = loss.mean()
                loss.backward()
                optim.step()

            if interrupt_handler.interrupted:
                print("Program interrupted")
                break

    return interrupt_handler.interrupted, total / n, np.array([safe_roc_auc_score(np.array(y_true)[:, i], np.array(y_score)[:, i]) for i in range(total.shape[0])]), y_true, y_score, filename


def safe_roc_auc_score(a, b):
    # TODO: make into util function
    try:
        return sklearn.metrics.roc_auc_score(a, b)
    except ValueError:
        return math.nan
