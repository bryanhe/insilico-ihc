#!/usr/bin/env python3

import tqdm
import pickle
import torch
import torchvision
import collections
import os
import click
import dstain

@click.command()
@click.argument("sample_file",
                type=click.File("r"))
@click.argument("aB_checkpoint",
                type=click.File("rb"))
@click.argument("T_checkpoint",
                type=click.File("rb"))
@click.option("--src", "-s", type=click.Path(exists=True), default=dstain.config.DATA_RAW)  # TODO: rename to wsi
@click.option("--output", "-o", type=click.Path(), default="cache")
def main(sample_file, ab_checkpoint, t_checkpoint, src, output):
    os.makedirs(output, exist_ok=True)
    device = torch.device("cuda")

    sample = dstain.utils.read_sample_file(sample_file, use_split=True)
    sample["main"] = {}
    sample["main"].update(sample["train"])
    sample["main"].update(sample["valid"])
    sample["main"].update(sample["test"])
    del sample["train"]
    del sample["valid"]
    del sample["test"]

    for split in sample:
        print(split)
        print("=" * len(split))

        patients = set()
        count = collections.Counter()
        for s in sample[split]:
            *patient, region = s.split("_")
            patient = "_".join(patient)
            patients.add(patient)
            stains_available = set(stain for (_, _, stain) in sample[split][s])
            assert "H" in stains_available
            assert len(stains_available) > 1
            for stain in stains_available:
                count[region, stain] += 1

        print("Patients:", len(patients))
        print("Samples:", len(sample[split]))
        regions = sorted(set(region for (region, _) in count.keys()))
        # stains = sorted(set(stains for (_, stains) in count.keys()))
        stains = ("H", "aB", "T")
        print("\t".join(("",) + stains))
        for r in regions:
            print(r, end="\t")
            for s in stains:
                print(count[r, s], end="\t")
            print()

    ihc_model = {}
    ihc_targets = {}
    mean = {}
    std = {}
    window = {}
    downsample = {}

    for checkpoint in [ab_checkpoint, t_checkpoint]:
        checkpoint = torch.load(checkpoint)
        stain = checkpoint["stain"]

        ihc_targets[stain] = checkpoint["targets"]

        ihc_model[stain] = torchvision.models.densenet121(pretrained=True)
        ihc_model[stain].classifier = torch.nn.Linear(1024, len(ihc_targets[stain]), bias=True)
        if device.type == "cuda":
            ihc_model[stain] = torch.nn.DataParallel(ihc_model[stain])
        ihc_model[stain].to(device)
        ihc_model[stain].load_state_dict(checkpoint["model_dict"])
        ihc_model[stain].eval()
        mean[stain] = checkpoint["mean"]
        std[stain] = checkpoint["std"]
        window[stain] = checkpoint["window"]
        downsample[stain] = checkpoint["downsample"]

    max_window = max(window.values())
    min_downsample = min(downsample.values())

    for split in sample:
        total = len(sample[split])
        for (i, s) in enumerate(sample[split]):
            print("# {} / {}".format(i + 1, total))
            for slide in sample[split][s]:
                (path, _, stain) = slide
                if stain == "H":
                    continue

                try:
                    with open(os.path.join(output, os.path.splitext(path)[0] + ".pkl"), "rb") as f:
                        data = pickle.load(f)
                except FileNotFoundError:
                    patch, warp = dstain.utils.register.register_sample(
                        src=src,
                        output=output,
                        filenames=[slide],
                        window=max_window,
                        downsample=min_downsample,
                        patches=0,
                    )

                    transform = [
                        torchvision.transforms.Resize(max_window // downsample[stain]),
                        torchvision.transforms.CenterCrop(window[stain] // downsample[stain]),
                        torchvision.transforms.ToTensor(),
                        torchvision.transforms.Normalize(mean=mean[stain], std=std[stain])
                    ]
                    transform = torchvision.transforms.Compose(transform)

                    dataset = dstain.datasets.WSI(
                        root=src,
                        files=[slide],
                        patch=patch,
                        warp=None,
                        window=max_window,
                        downsample=min_downsample,
                        transform=transform,
                        target_transform=None,
                        ref_stain=stain,
                        stains=[]
                    )

                    n_patches = 100
                    if len(dataset) < n_patches:
                        indices = np.random.choice(len(dataset), n_patches, replace=False)
                        dataset = torch.utils.data.Subset(dataset, indices)

                    loader = torch.utils.data.DataLoader(dataset, batch_size=16, num_workers=8, shuffle=False, pin_memory=False, drop_last=False)

                    data = []
                    for (X, *_) in tqdm.tqdm(loader):
                        data.extend(ihc_model[stain](X).detach().cpu().tolist())

                    os.makedirs(os.path.join(output, os.path.dirname(path)), exist_ok=True)
                    with open(os.path.join(output, os.path.splitext(path)[0] + ".pkl"), "wb") as f:
                        pickle.dump(data, f)

                    print(data)


if __name__ == "__main__":
    main()
