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
@click.option("--output", "-o", type=click.Path(), default="output/training")
@click.option("--model_name", type=str, default="densenet121")
@click.option("--lr", type=float, default=1e-4)
@click.option("--momentum", type=float, default=0.9)
@click.option("--weight_decay", type=float, default=0)
@click.option("--lr_step_period", type=int, default=50)
@click.option("--epochs", type=int, default=150)
@click.option("--batch_size", type=int, default=32)
@click.option("--num_workers", type=int, default=4)
@click.option("--window", type=int, default=2048)
@click.option("--downsample", type=int, default=4)
@click.option("--resume/--scratch", default=True)
@click.option("--keep_checkpoints/--remove_checkpoints", default=True)
def train(
        sample_file,
        src,
        output,
        model_name,
        lr,
        momentum,
        weight_decay,
        lr_step_period,
        epochs,
        batch_size,
        num_workers,
        window,
        downsample,
        resume,
        keep_checkpoints,
        stains=["aB", "T", "T"],  # TODO flag to select (or just create another sample file?)
        targets=["Plaque", "Tangle", "Neuritic Plaque"],
):

    os.makedirs(output, exist_ok=True)
    dstain.utils.latexify()
    device = torch.device("cuda")
    sample = dstain.utils.read_sample_file(sample_file)

    try:
        with open(os.path.join(output, "mean_std.pkl"), "rb") as f:
            (mean, std) = pickle.load(f)
    except FileNotFoundError:
        # TODO: compute s1 and s2 in dataloader to multiprocess?
        transforms = [
            # torchvision.transforms.Resize(512),  # TODO: use downsample  # TODO: give option to downsample from saved image size
            torchvision.transforms.ToTensor(),
        ]
        transforms = torchvision.transforms.Compose(transforms)
        mean, std = dstain.utils.get_mean_and_std(dstain.datasets.Stain(src, sample["train"], window=window, downsample=downsample, transforms=transforms, stains=stains), samples=8192, batch_size=batch_size, num_workers=num_workers)
        with open(os.path.join(output, "mean_std.pkl"), "wb") as f:
            pickle.dump((mean, std), f)

    transforms = [
        # torchvision.transforms.Resize(512),  # TODO: use downsample  # TODO: give option to downsample from saved image size
        # dstain.transforms.RandomCrop(window),  # TODO: change random crop to allow non-square # TODO: allow crop from saved image size
        torchvision.transforms.RandomHorizontalFlip(0.5),
        torchvision.transforms.RandomVerticalFlip(0.5),
        torchvision.transforms.RandomApply([torchvision.transforms.RandomRotation((90, 90))]),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=mean, std=std)
    ]
    transforms = torchvision.transforms.Compose(transforms)

    # TODO not following proper convention with transforms vs transform
    dataset = {s: dstain.datasets.Stain(src, sample[s], window=window, downsample=downsample, transforms=transforms, stains=stains) for s in sample}

    # TODO should merge this into get_mean_and_std
    try:
        p = np.load(os.path.join(output, "p.npy"))
    except:
        n_samples = 8192
        ds = dataset["train"]
        if n_samples < len(ds):
            indices = np.random.choice(len(ds), n_samples, replace=False)
            ds = torch.utils.data.dataset.Subset(ds, indices)
        loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=(device.type == "cuda"), drop_last=True)
        num = np.zeros(3)  # TODO: should be int? # TODO TODO TODO hard coded 3
        den = np.zeros(3)
        # num = np.zeros(len(stains))
        # den = np.zeros(len(stains))
        for (X, _, y, block, pixel) in tqdm.tqdm(loader, desc="Counting classes"):  # TODO: ihc image not used, try not to load?
            valid = ~torch.isnan(y)
            y = (y > 0)
            # y = torch.sigmoid(y)
            # y[~valid] = 0

            num += y.sum(0).cpu().numpy()
            den += valid.sum(0).cpu().numpy()

        p = num / den
        np.save(os.path.join(output, "p.npy"), p)
    print(p)

    model = torchvision.models.__dict__[model_name](pretrained=True)
    model.classifier = torch.nn.Linear(1024, len(stains), bias=True)
    # model.classifier = torch.nn.Linear(1024, len(stains), bias=True)
    model.classifier.weight.data[:] = 0
    model.classifier.bias.data[:] = torch.as_tensor(np.log(p))  # TODO: check dtype
    if device.type == "cuda":
        model = torch.nn.DataParallel(model)
    model.to(device)

    # Set up optimizer
    optim = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.StepLR(optim, lr_step_period)

    skip_header = (resume and os.path.isfile(os.path.join(output, "log.csv")))
    with open(os.path.join(output, "log.csv"), "a") as log:
        if not skip_header:
            # log.write("Epoch,Split,{}\n".format(",".join("{} {}".format(s, m) for m in ["CE", "AUC", "AUC (hl)"] for s in stains)))
            log.write("Epoch,Split,{}\n".format(",".join("{} {}".format(t, m) for m in ["CE", "AUC", "AUC (hl)"] for t in targets)))  # TODO Hard coded
            log.write("Baseline,train,{}\n".format(",".join(map(str, (-(p * np.log(p) + (1 - p) * np.log(1 - p))).tolist() + (2 * len(stains)) * [0.5]))))
            log.write("Baseline,valid,{}\n".format(",".join(map(str, (-(p * np.log(p) + (1 - p) * np.log(1 - p))).tolist() + (2 * len(stains)) * [0.5]))))
            log.flush()

        epoch = 0
        best_epoch = None
        best_auc = -math.inf
        if resume:
            # Attempt to load from checkpoint
            filenames = glob.glob(os.path.join(output, "checkpoints", "epoch_*.pt"))
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

                n_samples = 1000 * batch_size
                if n_samples < len(ds):
                    indices = np.random.choice(len(ds), n_samples, replace=False)
                    ds = torch.utils.data.dataset.Subset(dataset[s], indices)

                loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=(device.type == "cuda"), drop_last=True)

                # if epoch == 0:
                #     # Run with high lr only on random weights
                #     optim_ft = torch.optim.SGD(model.module.classifier.parameters(), lr=1e-4, momentum=0.9, weight_decay=0e-1)
                #     block_list, pixel_list, y_true, y_pred, total, n, interrupted = run_epoch(model, optim_ft, loader, device, s == "train", stains)
                # else:
                block_list, pixel_list, y_true, y_pred, total, n, interrupted = run_epoch(model, optim, loader, device, s == "train", stains)
                if s == "train":
                    sched.step()

                if interrupted:
                    sys.exit(0)

                # TODO: can remove _interrupted flag
                os.makedirs(os.path.join(output, "predictions"), exist_ok=True)
                np.savez_compressed(
                    os.path.join(output, "predictions", "{}_{}{}".format(epoch, s, "_interruped" if interrupted else "")),
                    y_true=y_true,
                    y_pred=y_pred,
                    block_list=block_list,
                    pixel_list=pixel_list,
                    window=window,
                    downsample=downsample,
                    stain=np.array(stains),
                )
                mask = [np.logical_or(np.array(y_true[j]) < -3, np.array(y_true[j]) > 3) for j in range(len(y_true))]  # TODO hardcoded 3

                log.write(
                    "{},{},{},{},{},{},{},{},{},{},{}\n".format( # TODO: hardcoded number of {}
                        epoch, s,
                        *(total / n),
                        *[sklearn.metrics.roc_auc_score(np.array(y_true[j]) > 0, y_pred[j]) for j in range(3)],  # TODO hardcoded 3
                        *[sklearn.metrics.roc_auc_score(np.array(y_true[j])[mask[j]] > 0, np.array(y_pred[j])[mask[j]]) for j in range(3)],  # TODO hardcoded 3
                ))
                log.flush()

                # TODO: checkpoint without dataparallel (and on cpu?)
                os.makedirs(os.path.join(output, "checkpoints"), exist_ok=True)
                if s == "valid":
                    loss = (total / n).mean()
                    auc = sum(sklearn.metrics.roc_auc_score(np.array(y_true[j]) > 0, y_pred[j]) for j in range(3)) / 3  # TODO hardcoded 3
                    auc_hl = sum(sklearn.metrics.roc_auc_score(np.array(y_true[j])[mask[j]] > 0, np.array(y_pred[j])[mask[j]]) for j in range(3)) / 3  # TODO hardcoded 3
                    if auc_hl > best_auc:
                        best_epoch = epoch
                        best_auc = auc_hl

                    save = {
                        "model_name": model_name,
                        "model_dict": model.state_dict(),
                        "optim_dict": optim.state_dict(),
                        "sched_dict": sched.state_dict(),
                        "epoch": epoch,
                        "best_epoch": best_epoch,
                        "best_auc": best_auc,
                        "loss": loss,
                        "auc": auc,
                        "auc_hl": auc_hl,
                        "window": window,
                        "downsample": downsample,
                        "stains": stains,
                        "targets": targets,
                        "mean": mean,
                        "std": std,
                        "y_true": y_true,
                        "y_score": y_pred,
                    }
                    torch.save(save, os.path.join(output, "checkpoints", "epoch_{}.pt".format(epoch)))
                    if not keep_checkpoints:
                        try:
                            os.remove(os.path.join(output, "checkpoints", "epoch_{}.pt".format(epoch - 1)))
                        except FileNotFoundError:
                            pass
                    if best_epoch == epoch:
                        torch.save(save, os.path.join(output, "checkpoints", "best.pt"))


def run_epoch(model, optim, loader, device, train, stains, regression=False):
    # TODO: drop last for only train, but need to deal with div by zero

    total = 0.
    s1 = 0.
    s2 = 0.
    n = 0
    # num = np.zeros(len(stains))  # TODO: should be int?
    # den = np.zeros(len(stains))
    num = np.zeros(3)  # TODO: should be int?  # TODO hardcoded 3
    den = np.zeros(3)

    # block_list = [[] for _ in stains]
    # pixel_list = [[] for _ in stains]
    # y_true = [[] for _ in stains]
    # y_pred = [[] for _ in stains]
    block_list = [[] for _ in range(3)]  # TODO hardcoded 3
    pixel_list = [[] for _ in range(3)]
    y_true = [[] for _ in range(3)]
    y_pred = [[] for _ in range(3)]
    with torch.set_grad_enabled(train), dstain.utils.GracefulInterruptHandler() as interrupt_handler, tqdm.tqdm(total=len(loader)) as pbar:
        model.train(train)
        for (X, _, ylogit, block, pixel) in loader:  # TODO: ihc image not used, try not to load?
            X = X.to(device)
            y = ylogit.to(device)
            yhat = model(X)
            valid = ~torch.isnan(y)
            y = (y > 0)
            # y = torch.sigmoid(y).clip(0.05, 0.95)
            # y[~valid] = 0

            loss = (valid * torch.nn.functional.binary_cross_entropy_with_logits(yhat, y.type(torch.float), reduction="none")).sum(0)
            num += (y * valid).sum(0).cpu().numpy()
            den += valid.sum(0).cpu().numpy()

            total += loss.detach().cpu().numpy()
            n += valid.sum(0).cpu().numpy()
            s1 += (valid * y).sum(0).cpu().numpy()
            s2 += (valid * (y ** 2)).sum(0).cpu().numpy()

            yhat = yhat.detach().cpu().numpy()
            valid = valid.cpu().numpy()
            block = np.array(block)
            pixel = pixel.numpy()
            # for i in range(len(stains)):  # TODO: could iterate through valid transpose
            for i in range(3):  # TODO hard coded 3
                for (des, src) in [
                    (y_true, ylogit[:, i]),
                    (y_pred, yhat[:, i]),
                    (block_list, block),
                    (pixel_list, pixel),
                ]:
                    des[i].extend(src[valid[:, i]].tolist())

            p = num / den
            p = p.clip(1e-10, 1 - 1e-10)
            try:  # TODO: is the protection needed
                auc_hl = []
                for i in range(len(total)):
                    mask = np.logical_or(np.array(y_true[i]) < -3, np.array(y_true[i]) > 3)
                    auc_hl.append(safe_roc_auc_score(np.array(y_true[i])[mask] > 0, np.array(y_pred[i])[mask]))

                pbar.set_postfix_str(
                    "CE: " + ", ".join("{:.5f} ({:.5f}) / {:.5f}".format(total[i] / n[i], loss[i].item() / max(valid.sum(0)[i].item(), 1e-10), (p[i] - 1) * math.log(1 - p[i]) - p[i] * math.log(p[i])) for i in range(len(total)))
                    +
                    " AUC: " + ", ".join("{:.3f}".format(safe_roc_auc_score(np.array(y_true[i]) > 0, y_pred[i])) for i in range(len(total)))
                    +
                    " AUC (conf): " + ", ".join("{:.3f}".format(i) for i in auc_hl)
                )
            except Exception as e:
                pass

            if train:
                optim.zero_grad()
                loss = loss.sum()
                loss.backward()
                optim.step()

            pbar.update()

            if interrupt_handler.interrupted:
                print("Program interrupted")
                break

    return block_list, pixel_list, y_true, y_pred, total, n, interrupt_handler.interrupted


def safe_roc_auc_score(a, b):
    # TODO: make into util function
    try:
        return sklearn.metrics.roc_auc_score(a, b)
    except ValueError:
        return math.nan
