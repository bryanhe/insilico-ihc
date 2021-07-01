import collections
import math
import os
import matplotlib.pyplot as plt

import captum.attr
import click
import numpy as np
import shutil
import openslide
import PIL
import PIL.ImageDraw
import scipy
import torch
import torchvision
import tqdm
import random
import sklearn
# import umap

import dstain


@click.command()
@click.argument("sample_file",
                type=click.File("r"))
@click.argument("checkpoint",
                type=click.File("rb"))
@click.argument("aB_checkpoint",
                type=click.File("rb"))
@click.argument("T_checkpoint",
                type=click.File("rb"))
@click.option("--src", "-s", type=click.Path(exists=True), default=dstain.config.DATA_RAW)  # TODO: rename to wsi
@click.option("--output", "-o", type=click.Path(), default="output/evaluate")
@click.option("--batch_size", type=int, default=32)
@click.option("--num_workers", type=int, default=4)
def evaluate(
        sample_file,
        checkpoint,
        ab_checkpoint,
        t_checkpoint,
        src,
        output,
        batch_size,
        num_workers):
    dstain.utils.latexify()

    device = torch.device("cuda")
    samples = dstain.utils.read_sample_file(sample_file)

    mean = {}
    std = {}
    window = {}
    downsample = {}

    checkpoint = torch.load(checkpoint, map_location=device)
    print("H&E Model Epoch #{}".format(checkpoint["best_epoch"]))
    stains = checkpoint["stains"]
    targets = checkpoint["targets"]

    model = torchvision.models.densenet121(pretrained=True)
    model.classifier = torch.nn.Linear(1024, len(stains), bias=True)
    if device.type == "cuda":
        model = torch.nn.DataParallel(model)
    model.to(device)
    model.load_state_dict(checkpoint["model_dict"])
    model.eval()
    dstain.utils.nn.extract_features(model)
    mean["H"] = checkpoint["mean"]
    std["H"] = checkpoint["std"]
    window["H"] = checkpoint["window"]
    downsample["H"] = checkpoint["downsample"]

    ihc_model = {}
    ihc_targets = {}
    offset = 0

    for checkpoint in [ab_checkpoint, t_checkpoint]:
        checkpoint = torch.load(checkpoint)
        stain = checkpoint["stain"]
        print("{} IHC Model Epoch #{}".format(stain, checkpoint["best_epoch"]))

        ihc_targets[stain] = checkpoint["targets"]
        offset += len(ihc_targets[stain])

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
    stain_target = [(stain, target) for stain in sorted(set(stains)) for target in ihc_targets[stain]]  # IHC output after zipping
    ihc_index = [stain_target.index(st) for st in zip(stains, targets)] # ihc_index[i] gives the  index of target after zipping together corresponding to the i-th H&E output

    # TODO generate colorbar
    # if True:
    #     fig = plt.figure(figsize=(2.25, 0.75))
    #     norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
    #     cb = matplotlib.colorbar.ColorbarBase(plt.gca(), cmap=ihc_cmap,
    #                                           norm=norm,
    #                                           orientation='horizontal')
    #     cb.set_ticks([0, 0.5, 1])
    #     cb.set_label("Probability")
    #     plt.tight_layout()
    #     fig.savefig(output, "heatmap", "colorbar.pdf")
    #     plt.close(fig)

    for split in ["test"]:
    # for split in ["eval"]:
        pixel_list = []
        pred_list = []
        feature_list = []
        value_list = []
        lfb_list = []
        sample_list = []
        latex_info = collections.defaultdict(list)
        for (i, sample) in enumerate(samples[split]):
            print("Sample #{} / {}: {}".format(i + 1, len(samples[split]), sample))

            try:
                pixel, feature, pred, value, lfb, size, index = get_predictions(src, output, sample, samples[split][sample], model, ihc_model, mean, std, ihc_index, device, batch_size, num_workers, stains, window, downsample)
            except BadSlideError as error:
                print(error)
                continue
            except MemoryError as error:
                print("Memory error (likely failed to register)")
                continue

            mask = [s in set(map(lambda x : x[2], samples[split][sample])) for s in stains]
            value[:, ~np.array(mask)] = math.nan

            pixel_list.append(pixel)
            pred_list.append(pred)
            feature_list.append(feature)
            value_list.append(value)
            lfb_list.append(lfb)
            sample_list.append(sample)

            # Saving predictions on WSI
            os.makedirs(os.path.join(output, "heatmap", sample), exist_ok=True)

            path = os.path.join(output, "heatmap", sample, "lfb.pdf")
            if not os.path.isfile(path):
                fig, ax = plot_grid(pixel, (lfb < 0.25).astype(float), size, cmap=lfb_cmap)
                fig.savefig(path)
                plt.close(fig)

            for (j, (stain, target)) in enumerate(zip(stains, targets)):
                if not np.isnan(value[:, j]).all():
                    print("{} ({}): {:.3f}".format(stain, target, safe_roc_auc_score(value[:, j] > 0, pred[:, j])))

                for (v, name) in [(value[:, j] > 0, "value"), (pred[:, j], "pred")]:  # TODO: should skip value if nan
                    # TODO: use tissue mask to look more real
                    path = os.path.join(output, "heatmap", sample, "{}_{}_{}.pdf".format(stain, target.lower().replace(" ", "-"), name))
                    # if not os.path.isfile(path):
                    if False:
                        offset = None
                        if split == "test" or True:
                            if target == "Plaque":
                                offset = -0.3235231637954712
                            elif target == "Tangle":
                                offset = -0.7584477663040161
                            elif target == "Neuritic Plaque":
                                offset = -1.2803025245666504


                        if name == "pred":
                            for (suffix, m) in [("very_low", -2), ("low", -1), ("high", 1), ("med", 0)]:
                                u = (1 / (1 + np.exp(-v + offset - m)))
                                fig, ax = plot_grid(pixel, u, size, cmap=ihc_cmap)
                                path = os.path.join(output, "heatmap", sample, "{}_{}_{}_{}.pdf".format(stain, target.lower().replace(" ", "-"), name, suffix))
                                print(suffix, m, u.min(), u.max(), path)
                                fig.savefig(path)
                                plt.close(fig)
                            v = (1 / (1 + np.exp(-v + offset)))

                        v = v.clip(0, 1)
                        fig, ax = plot_grid(pixel, v, size, cmap=ihc_cmap)
                        fig.savefig(path)
                        # path = os.path.join(output, "heatmap", sample, "{}_{}_{}.png".format(stain, target.lower().replace(" ", "-"), name))
                        # fig.savefig(path)
                        plt.close(fig)

            if sample == "2018_18_cHIP":
                zoom = [(24860, 45340, 33727, 56255)]
                zoom = [(24860, 45340, 33727, 56255)]
            elif sample == "2018_34_AMY":
                zoom = [(24860, 45340, 33727, 56255)]
                zoom = []
                print(np.unique(pixel[:, 0]))
                print(np.unique(pixel[:, 1]))
            elif sample == "2018_34_MID":
                zoom = [(25140, 45620, 55659, 78187)]
                print(np.unique(pixel[:, 0]))
                print(np.unique(pixel[:, 1]))
            elif sample == "2018_36_MF":
                zoom = [(27132, 43516, 66152, 80488)]
                print(np.unique(pixel[:, 0]))
                print(np.unique(pixel[:, 1]))
            else:
                zoom = []

            # if zoom != []:
            # if False:
            #     scale = 10
            #     os.makedirs(os.path.join(output, "postprocessing", sample), exist_ok=True)
            #     for (_, basename, stain) in samples[split][sample]:
            #         img = PIL.Image.open(os.path.join(output, "registration", sample, "2_register", basename + "_img_warp.jpg"))
            #         draw = PIL.ImageDraw.Draw(img)
            #         for (x0, x1, y0, y1) in zoom:
            #             draw.rectangle((np.array([x0, y0, x1, y1]) / 128).astype(np.int).tolist(), outline=0)

            #         path = os.path.join(output, "postprocessing", sample, stain + ".jpg")
            #         img.save(path)

            #     slides = samples[split][sample]
            #     files = [(path, basename, stain) for (path, basename, stain) in slides if stain in set(["H"] + stains)]

            #     max_window = max(window.values())
            #     min_downsample = min(downsample.values())
            #     patch, warp = \
            #         dstain.utils.register.register_sample(
            #             src=src,
            #             output=output,
            #             filenames=files,
            #             window=max_window,
            #             downsample=min_downsample,
            #             patches=0,
            #         )
            #     # TODO: raise error if fail to register
            #     for (j, (path, basename, stain)) in enumerate(files):
            #         slide = openslide.open_slide(os.path.join(src, path))
            #         for (k, (x0, x1, y0, y1)) in enumerate(zoom):
            #             img = dstain.utils.register.load_warped(slide, (x0, y0), (y1 - y0, x1 - x0), warp[j], downsample=32)  # TODO load warped has size coords transposed
            #             path = os.path.join(output, "postprocessing", sample, "{}_zoom_{}.jpg".format(stain, k))
            #             img.save(path)

            #     for (j, (stain, target)) in enumerate(zip(stains, targets)):
            #         for (v, name) in [(value[:, j] > 0, "value"), ((1 / (1 + np.exp(-pred[:, j]))), "pred")]:  # TODO: should skip value if nan
            #             fig, ax = plot_grid(pixel, v, size, cmap=ihc_cmap)
            #             for (x0, x1, y0, y1) in zoom:
            #                 plt.plot(np.array([x0, x0, x1, x1, x0]) / scale, np.array([y0, y1, y1, y0, y0]) / scale, color="k", linewidth=1)
            #             path = os.path.join(output, "postprocessing", sample, "{}_{}_{}.pdf".format(stain, target.lower().replace(" ", "-"), name))
            #             fig.savefig(path)
            #             plt.close(fig)
            #             for (k, (x0, x1, y0, y1)) in enumerate(zoom):
            #                 fig, ax = plot_grid(pixel - np.array([x0, y0]), v, (x1 - x0, y1 - y0), cmap=ihc_cmap)
            #                 path = os.path.join(output, "postprocessing", sample, "{}_{}_{}_zoom_{}.pdf".format(stain, target.lower().replace(" ", "-"), name, k))
            #                 fig.savefig(path)


            ihc_values = save_ihc_patches(src, output, value, index, sample, samples[split][sample], stains, targets, window, downsample)

            # # value[~np.isnan(value)] = (value[~np.isnan(value)] > 0)

            save_attributions(src, output, pred, index, sample, samples[split][sample], model, device, mean, std, stains, targets, window, downsample)

            basename = {stain: basename for (path, basename, stain) in samples[split][sample]}

            for (j, (stain, target)) in enumerate(zip(stains, targets)):
                offset = None
                if split == "test" or True:
                    if target == "Plaque":
                        offset = -0.3235231637954712
                    elif target == "Tangle":
                        offset = -0.7584477663040161
                    elif target == "Neuritic Plaque":
                        offset = -1.2803025245666504
                latex_info[(stain, target)].append({
                    "name": sample,
                    "he": basename["H"],
                    "ihc": basename[stain] if stain in basename else "",
                    "real": math.nan if np.isnan(value[:, j]).all() else (value[:, j] > 0).mean(),  # TODO: should be nanmean
                    # "real": (1 / (1 + np.exp(-value[:, j]))).mean(),
                    "pred": (1 / (1 + np.exp(-pred[:, j] + offset)) > 0.5).mean(),
                    "ihc_values": (1 / (1 + np.exp(-np.array(ihc_values[(stain, target)])))).tolist(),
                })

            os.makedirs(os.path.join(output, "latex"), exist_ok=True)

            for (j, (stain, target)) in enumerate(zip(stains, targets)):
                tex = dstain.utils.render_latex(
                    os.path.join(dstain.config.ROOT, "latex", "threshold.tex"),
                    stain=stain,
                    target=target,
                    samples=latex_info[(stain, target)],
                )
                with open(os.path.join(output, "latex", "threshold_{}_{}.tex".format(stain, target.lower().replace(" ", "-"))), "w") as tex_file:
                    tex_file.write(tex)

            for (j, (stain, target)) in enumerate(zip(stains, targets)):
                tex = dstain.utils.render_latex(
                    os.path.join(dstain.config.ROOT, "latex", "evaluate.tex"),
                    stain=stain,
                    target=target,
                    samples=latex_info[(stain, target)],
                )
                with open(os.path.join(output, "latex", "{}_{}.tex".format(stain, target.lower().replace(" ", "-"))), "w") as tex_file:
                    tex_file.write(tex)

                os.makedirs(os.path.join(output, "latex"), exist_ok=True)
                tex = dstain.utils.render_latex(
                    os.path.join(dstain.config.ROOT, "latex", "ihc_wsi.tex"),
                    stain=stain,
                    target=target,
                    samples=latex_info[(stain, target)],
                )
                with open(os.path.join(output, "latex", "{}_{}_ihc.tex".format(stain, target.lower().replace(" ", "-"))), "w") as tex_file:
                    tex_file.write(tex)

            tex = dstain.utils.render_latex(
                os.path.join(dstain.config.ROOT, "latex", "lfb.tex"),
                stain=stain,
                samples=latex_info[(stains[0], targets[0])],  # LFB doesn't care about IHC
            )
            with open(os.path.join(output, "latex", "lfb.tex"), "w") as tex_file:
                tex_file.write(tex)

            # for (path, basename, stain) in samples[split][sample]:
            #     if stain in ["H", "aB", "T"]:
            #         shutil.copyfile(os.path.join("garbage", block, "2a_initialization", os.path.basename(filename) + ".jpg"),
            #                         os.path.join(output, block, "{}.jpg".format(stain)))
            #         img = PIL.Image.open(os.path.join(output, block, "{}.jpg".format(stain)))
            #         draw = PIL.ImageDraw.Draw(img)
            #         for (color, ra) in [((0, 0, 255), rank[:5]), ((0, 255, 0), rank[-1:-6:-1])]:
            #             for (k, r) in enumerate(ra):
            #                 draw.text((pixel[r, :] / (size / np.array(img.size))).tolist(), str(k + 1), color)
            #         img.save(os.path.join(output, block, "{}_label.jpg".format(stain)))


    os.makedirs(os.path.join(output, "fig"), exist_ok=True)
    for (i, (stain, target)) in enumerate(zip(stains, targets)):
        print(80 * "=")
        print(stain, target)
        print(80 * "=")

        v = np.concatenate(value_list)[:, i]
        print("Confident neg: ", (v[~np.isnan(v)] < -3).mean())
        print("Confident pos: ", (v[~np.isnan(v)] > 3).mean())

        value = np.concatenate(([v[:, i] for (v, p, s) in zip(value_list, pred_list, sample_list)]))
        pred = np.concatenate(([p[:, i] for (v, p, s) in zip(value_list, pred_list, sample_list)]))
        pred = pred[~np.isnan(value)]
        value = value[~np.isnan(value)]
        print("offset: ", np.sort(pred)[int((value < 0).mean() * value.size)])
        pred = pred[np.logical_or(value < -3, value > 3)]
        value = value[np.logical_or(value < -3, value > 3)]
        print("offset: ", np.sort(pred)[int((value < 0).mean() * value.size)])

        pred = np.concatenate(pred_list)[:, i]
        value = np.concatenate(value_list)[:, i]
        pv = [(p[np.logical_or(v[:, i] < -3, 3 < v[:, i]), i], v[np.logical_or(v[:, i] < -3, 3 < v[:, i]), i]) for (p, v) in zip(pred_list, value_list) if not np.isnan(v[:, i]).any()]
        p, v = zip(*pv)
        def bootstrap(p, v, samples=1000):
            bootstraps = []
            for _ in range(samples):
                ind = np.random.choice(len(p), len(p))
                bootstraps.append(sklearn.metrics.roc_auc_score(np.concatenate([v[i] for i in ind]) > 0, np.concatenate([p[i] for i in ind])))
            bootstraps = sorted(bootstraps)

            return sklearn.metrics.roc_auc_score(np.concatenate(v) > 0, np.concatenate(p)), bootstraps[round(0.025 * len(bootstraps))], bootstraps[round(0.975 * len(bootstraps))]

        print("AUROC: {:.2f} ({:.2f} - {:.2f})".format(*bootstrap(p, v)))

        fig = plt.figure(figsize=(2.00, 1.75))
        tpr = []
        # https://stats.stackexchange.com/questions/186337/average-roc-for-repeated-10-fold-cross-validation-with-probability-estimates
        base_fpr = np.linspace(0, 1, 1001)
        for _ in range(1000):
            ind = np.random.choice(len(p), len(p))
            y_true = np.concatenate([v[i] for i in ind]) > 0
            y_score = np.concatenate([p[i] for i in ind])
            fpr_, tpr_, thresholds = sklearn.metrics.roc_curve(y_true, y_score)
            tpr.append(np.interp(base_fpr, fpr_, tpr_))
        tpr = list(map(sorted, zip(*tpr)))
        lower = [x[int(len(x) * 0.025)] for x in tpr]
        upper = [x[int(len(x) * 0.975)] for x in tpr]
        plt.fill_between(base_fpr, lower, upper, color="gray", alpha=0.3, linewidth=0)

        y_true = np.concatenate(v) > 0
        y_score = np.concatenate(p)
        fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_true, y_score)
        plt.plot(fpr, tpr, "-",  color="k", linewidth=1)

        plt.plot([0, 1], [0, 1], color='k', linewidth=1, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("{} (AUC: {:.2f})".format(target, sklearn.metrics.roc_auc_score(y_true, y_score)))
        # plt.title("{}: {}".format(stain, target))
        plt.tight_layout()
        plt.savefig(os.path.join(output, "fig", "roc_{}_{}.pdf".format(stain, target.lower().replace(" ", "-"))))
        plt.close(fig)

        bins = np.linspace(0, 1, 11)

        fig = plt.figure(figsize=(2, 2))
        weights, _ = np.histogram(1 / (1 + np.exp(-np.concatenate(p))), bins=bins)
        weights = weights / weights.sum()
        plt.hist(bins[:-1], bins=bins, weights=weights)
        plt.xlabel("Predicted Probability")
        plt.ylabel("Fraction of Patches")
        plt.title(target)
        plt.tight_layout()
        plt.savefig(os.path.join(output, "fig", "prob_hist_{}_{}.pdf".format(stain, target.lower().replace(" ", "-"))))

        def calibration(p, v, bins):
            ans = []
            for (i, (lower, upper)) in enumerate(zip(bins, bins[1:])):
                ans.append(v[np.logical_and(lower <= p, p < upper)].mean())
            return ans

        c = calibration(1 / (1 + np.exp(-np.concatenate(p))), np.concatenate(v) > 0, bins)  # TODO bootstrap
        fig = plt.figure(figsize=(2, 2))
        plt.plot((bins[:-1] + bins[1:]) / 2, c, marker=".", linewidth=1, color="k")
        plt.plot([0, 1], [0, 1], "--", linewidth=1, color="k")
        plt.axis([0, 1, 0, 1])
        plt.xlabel("Predicted Probability")
        plt.ylabel("Fraction of True Positives in Bin")
        plt.title(target)
        plt.tight_layout()
        plt.savefig(os.path.join(output, "fig", "calib_{}_{}.pdf".format(stain, target.lower().replace(" ", "-"))))

        if target == "Plaque":
            offset = -0.3235231637954712
        elif target == "Tangle":
            offset = -0.7584477663040161
        elif target == "Neuritic Plaque":
            offset = -1.2803025245666504
        else:
            raise ValueError()

        fig = plt.figure(figsize=(3, 3))
        plt.hist(1 / (1 + np.exp(-np.concatenate(p) + offset)), bins=bins)
        plt.savefig(os.path.join(output, "fig", "prob_hist_fix_{}_{}.pdf".format(stain, target.lower().replace(" ", "-"))))

        c = calibration(1 / (1 + np.exp(-np.concatenate(p) + offset)), np.concatenate(v) > 0, bins)  # TODO bootstrap
        fig = plt.figure(figsize=(3, 3))
        plt.scatter((bins[:-1] + bins[1:]) / 2, c)
        plt.axis([0, 1, 0, 1])
        plt.tight_layout()
        plt.savefig(os.path.join(output, "fig", "calib_fix_{}_{}.pdf".format(stain, target.lower().replace(" ", "-"))))


        fig = plt.figure(figsize=(1.75, 1.75))
        value, pred, sample = zip(*[((1 / (1 + np.exp(-v[:, i]))).mean(), (1 / (1 + np.exp(-p[:, i]))).mean(), s) for (v, p, s) in zip(value_list, pred_list, sample_list) if not math.isnan(v[:, i].mean())])
        print("Spearman", scipy.stats.spearmanr(value, pred).correlation)
        print("(gap, value, pred, sample)", sorted(zip(np.array(pred) - np.array(value), value, pred, sample)))
        plt.scatter(value, pred, s=4)
        print("scaling: ", sum(value) / sum(pred))
        lower = -0.01
        upper = max(value + pred) + 0.01
        # plt.plot([lower, upper], [lower, upper], "--", color="k", linewidth=1)
        plt.title("Region-level Predictions")
        # plt.title("Patient-level (Spearman={:.2f})".format(scipy.stats.spearmanr(value, pred).correlation))
        plt.xlabel("Real Staining")
        plt.ylabel("Predicted Staining")
        plt.axis([lower, max(value) + 0.01, lower, max(pred) + 0.01])
        plt.tight_layout()
        # plt.axis("square")
        plt.savefig(os.path.join(output, "fig", "scatter_{}_{}.pdf".format(stain, target.lower().replace(" ", "-"))))
        plt.close(fig)


        continue

        yt = []
        auc = []
        sample = []
        for (v, p, s) in zip(value_list, pred_list, sample_list):
            if np.isnan(v[:, i]).all():
                continue
            value = v[:, i]
            pred = p[:, i]
            mask = ~np.isnan(value)
            mask = np.logical_and(mask, np.logical_or(value < -3, value > 3))
            y_true = (value[mask] > 0)
            print(y_true.mean())
            if (target != "Neuritic Plaque" and y_true.mean() < 0.1) or (target == "Neuritic Plaque" and y_true.mean() <= 0.001):
                continue
            yt.append(y_true.mean())
            y_score = pred[mask]
            auc.append(safe_roc_auc_score(y_true, y_score))
            sample.append(s)

        fig = plt.figure(figsize=(1.75, 1.75))
        plt.hist(auc, bins=np.arange(0, 1.01, 0.05))
        # breakpoint()
        # assert all(i > 0.5 for i in auc)
        plt.xlim([0.5, 1])
        plt.xlim([0, 1])
        # plt.title("{}: {}".format(stain, target))
        plt.title(target)
        plt.xlabel("Slide-level AUROC")
        plt.ylabel("# Slides")
        plt.tight_layout()
        plt.savefig(os.path.join(output, "fig", "hist_{}_{}.pdf".format(stain, target.lower().replace(" ", "-"))))
        plt.close(fig)

        fig = plt.figure(figsize=(1.75, 1.75))
        plt.scatter(yt, auc)
        plt.savefig(os.path.join(output, "fig", "test_{}_{}.pdf".format(stain, target.lower().replace(" ", "-"))))
        plt.close(fig)




        # fig = plt.figure(figsize=(3, 3))
        # value, pred = zip(*[(v[lfb < 0.25, i].mean(), (1 / (1 + np.exp(-p[lfb < 0.25, i]))).mean()) for (p, v, lfb) in zip(pred_list, value_list, lfb_list)])
        # print(scipy.stats.spearmanr(value, pred).correlation)
        # print(sorted(zip(value, sample_list)))
        # plt.scatter(value, pred, s=4)
        # plt.title("Patient-level Grey Matter Predictions")
        # # plt.title("Patient-level (Spearman={:.2f})".format(scipy.stats.spearmanr(value, pred).correlation))
        # plt.xlabel("Real Staining")
        # plt.ylabel("Predicted Staining")
        # plt.axis([lower, max(value) + 0.01, lower, max(pred) + 0.01])
        # plt.tight_layout()
        # plt.savefig(os.path.join(output, "fig", "scatter_gm_{}_{}_{}.pdf".format(stain, annotation, thresh)))
        # plt.close(fig)

        # fig = plt.figure(figsize=(3, 3))
        # value, pred = zip(*[(v[:, i].mean(), (1 / (1 + np.exp(-p[:, i]))).mean()) for (p, v) in zip(pred_list, value_list)])
        # # print(sorted(zip(value, sample_list)))
        # plt.scatter(np.argsort(np.argsort(value)), np.argsort(np.argsort(pred)), s=4)
        # plt.title("Patient-level (Spearman={:.2f})".format(scipy.stats.spearmanr(value, pred).correlation))
        # plt.xlabel("Real Staining")
        # plt.ylabel("Predicted Staining")
        # plt.tight_layout()
        # plt.savefig(os.path.join(output, "fig", "scatter_rank_{}_{}_{}.pdf".format(stain, annotation, thresh)))
        # plt.close(fig)

        # grey_matter = (np.concatenate(lfb_list) < 0.25)
        # print(safe_roc_auc_score(np.concatenate(value_list)[:, i], np.concatenate(pred_list)[:, i]))
        # print(safe_roc_auc_score(np.concatenate(value_list)[grey_matter, i], np.concatenate(pred_list)[grey_matter, i]))

        # fig = plt.figure(figsize=(3, 3))
        # precision, recall, threshold = sklearn.metrics.precision_recall_curve(np.concatenate(value_list)[:, i], np.concatenate(pred_list)[:, i])
        # plt.plot(recall, precision, linewidth=1)
        # pos = np.concatenate(value_list)[:, i].mean()
        # plt.plot([0, 1], [pos, pos], "--", linewidth=1)
        # plt.xlabel("Recall")
        # plt.ylabel("Precision")
        # plt.xlim([0, 1])
        # plt.tight_layout()
        # plt.savefig(os.path.join(output, "fig", "pr_{}_{}_{}.pdf".format(stain, annotation, thresh)))

        # fig = plt.figure(figsize=(3, 3))
        # precision, recall, threshold = sklearn.metrics.precision_recall_curve(np.concatenate(value_list)[grey_matter, i], np.concatenate(pred_list)[grey_matter, i])
        # plt.plot(recall, precision, linewidth=1)
        # pos = np.concatenate(value_list)[grey_matter, i].mean()
        # plt.plot([0, 1], [pos, pos], "--", linewidth=1)
        # plt.xlabel("Recall")
        # plt.ylabel("Precision")
        # plt.xlim([0, 1])
        # plt.tight_layout()
        # plt.savefig(os.path.join(output, "fig", "pr_gm_{}_{}_{}.pdf".format(stain, annotation, thresh)))

        # tex[(stain, annotation, thresh)].write(
        #     "\\begin{{frame}}\n"
        #     "  \\begin{{itemize}}\n"
        #     "    \\item Overall AUC: {}\n"
        #     "    \\item Grey matter AUC: {}\n"
        #     "  \\end{{itemize}}\n"
        #     "  \\frametitle{{Overall Performance}}\n"
        #     "  \\begin{{center}}\n"
        #     "    \\includegraphics[scale=0.6]{{{}}}\n"
        #     "    \\includegraphics[scale=0.6]{{{}}}\n"
        #     "    \\includegraphics[scale=0.6]{{{}}}\n"
        #     "    \\includegraphics[scale=0.6]{{{}}}\n"
        #     "  \\end{{center}}\n"
        #     "\\end{{frame}}\n".format(
        #         "{:.2f}".format(safe_roc_auc_score(np.concatenate(value_list)[:, i], np.concatenate(pred_list)[:, i])),
        #         "{:.2f}".format(safe_roc_auc_score(np.concatenate(value_list)[grey_matter, i], np.concatenate(pred_list)[grey_matter, i])),
        #         os.path.join("..", "fig", "{{\"hist_{}_{}_{}\"}}.pdf".format(stain, annotation, thresh)),
        #         os.path.join("..", "fig", "{{\"hist_high_{}_{}_{}\"}}.pdf".format(stain, annotation, thresh)),
        #         os.path.join("..", "fig", "{{\"scatter_{}_{}_{}\"}}.pdf".format(stain, annotation, thresh)),
        #         os.path.join("..", "fig", "{{\"scatter_gm_{}_{}_{}\"}}.pdf".format(stain, annotation, thresh)),
        # ))


    for r in ["AMY", "cHIP", "HIP", "MID", "MF"]:
        mask = [r in s for s in sample_list]
        print(np.array([(v > 0).mean(0) > 0.05 for v in value_list])[mask, :].mean(0))
    
        breakpoint()
        exit()
        f = []
        s = []
        p = []
        he_image = []
        ihc_image = []
        lfb = []
        n = 50
        
        for (feature, l, pixel, sample) in zip(tqdm.tqdm(feature_list), lfb_list, pixel_list, sample_list):
            index = np.random.choice(feature.shape[0], n)  # TODO check no replacement
            f.append(feature[index, :])
            lfb.extend(l[index])
            p.append(pixel[index, :])
            s.extend([sample] * n) # TODO: check if samples > real number of patches
        
        mask = (np.array(lfb) < 0.1)
        f = np.concatenate(f)
        f = f[mask, :]
        s = [i for (i, m) in zip(s, mask) if m]
        p = np.concatenate(p)[mask, :]
        
        # he_image = [i for (i, m) in zip(he_image, mask) if m]
        # ihc_image = [i for (i, m) in zip(ihc_image, mask) if m]
        emb = umap.UMAP().fit_transform(f)
        s.append(None)
        
        fig = plt.figure(figsize=(3, 3))
        start = 0
        for i in range(1, len(s)):
            if s[i] != s[i - 1]:
                plt.scatter(emb[start:i, 0], emb[start:i, 1], s=1)
                start = i
        
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
        plt.savefig(os.path.join(output, "fig", "umap.pdf"))
        plt.close(fig)
        
        size = 20
        emb -= emb.min(0)
        emb /= emb.max(0)
        emb *= (size - 1e-5)
        emb = emb.astype(np.int)
        index = [[[] for _ in range(size)] for _ in range(size)]
        for (i, e) in enumerate(emb):
            print(i, e)
            index[e[0]][e[1]].append(i)
            
        full = {stain: PIL.Image.new("RGB", (size * window[stain] // downsample[stain], size * window[stain] // downsample[stain]), (255, 255, 255)) for stain in set(["H"] + stains)}
        for i in range(size):
            for j in range(size):
                if index[i][j] != []:
                    print(i, j, flush=True)
                    seen = [False for _ in set(stains)]
                    count = 0
                    while not all(seen) and count < 10:
                        ind = random.choice(index[i][j])
                        sample = s[ind]
                        slides = samples[split][sample]
                        # TODO: either make registration cache more directly, or reuse dataset between iters (if same sample used for two umap examples, dataset loaded twice)
                        dataset = _get_dataset(src, os.path.join(output, "registration", sample), slides, stains, window, downsample, transform=None, target_transform=None)
                        dataset.patch = [p[ind, :]]
                        h, ihc, *_ = dataset[0]
                        full["H"].paste(h, (i * window["H"] // downsample["H"], (size - j - 1) * window["H"] // downsample["H"]))
                        for (k, stain) in enumerate(sorted(set(stains))):
                            if ihc[k].size != (0, 0):
                                seen[k] = True
                            full[stain].paste(ihc[k], (i * window[stain] // downsample[stain], (size - j - 1) * window[stain] // downsample[stain]))
                        count += 1
        
        for stain in set(["H"] + stains):
            full[stain].save(os.path.join(output, "fig", "umap_{}.jpg".format(stain)))
    
        WINDOW = [2048, 4096, 8192, 16384]
        for (i, stain) in enumerate(stains):
            print(stain)
            value = np.concatenate(value_list)[:, i]
            pred = np.concatenate(pred_list)[:, i]
            mask = ~np.isnan(value)
            auc = [safe_roc_auc_score(value[mask], pred[mask])]
            patches = []
            for window in WINDOW[1:]:
                a = []
                b = []
                patches.append([])
                for (pixel, pred, value) in zip(pixel_list, pred_list, value_list):
                    pred = pred[:, i]
                    value = value[:, i]
                    if np.isnan(value).any():
                        continue
                    patches[-1].append(0)
                    lower = pixel.min(0)
                    upper = pixel.max(0)
                    for x in range(lower[0], upper[0], window):
                        for y in range(lower[1], upper[1], window):
                            ul = np.array([x, y])
                            lr = ul + window
                            mask = np.logical_and(ul <= pixel, pixel < lr).all(1)
                            if mask.sum() >= ((window // 2048) ** 2) * 0.9:
                                a.append((1 / (1 + np.exp(-pred[mask]))).mean())  # TODO logsumexp trick, and need to handle multiple dimensions
                                b.append(value[mask].mean())
                                patches[-1][-1] += 1
                print((np.array(b) > 0.30).mean())
                auc.append(safe_roc_auc_score(np.array(b) > 0.05, a))
            print(auc)
            fig = plt.figure(figsize=(3, 3))
            plt.plot(WINDOW, auc, color="k")
            plt.xlabel("Window Size")
            plt.ylabel("ROAUC")
            plt.tight_layout()
            fig.savefig(os.path.join(output, "fig", "average.pdf"))
            plt.close(fig)


class BadSlideError(Exception):
    pass


def _get_dataset(src, output, slides, stains, window, downsample, transform, target_transform):
    files = [(path, basename, stain) for (path, basename, stain) in slides if stain in set(["H"] + stains)]

    max_window = max(window.values())
    min_downsample = min(downsample.values())
    patch, warp = \
        dstain.utils.register.register_sample(
            src=src,
            output=output,
            filenames=files,
            window=max_window,
            downsample=min_downsample,
            patches=0,
        )
    # TODO: raise error if fail to register

    dataset = dstain.datasets.WSI(
        root=src,
        files=slides,
        patch=patch,
        warp={stain: w for ((_, _, stain), w) in zip(files, warp)},
        window=max_window,
        downsample=min_downsample,
        transform=transform,
        target_transform=target_transform,
        ref_stain="H",
        stains=sorted(set(stains)),
    )

    return dataset

def get_predictions(src, output, sample, slides, model, ihc_model, mean, std, ihc_index, device, batch_size, num_workers, stains, window, downsample):
    color = {
        "background": [0.949, 0.957, 0.953],
        "eosin": [0.844, 0.724, 0.830],
        "hema": [0.305, 0.193, 0.467],
        "lfb": [0.183, 0.338, 0.631],
    }
    color = {k: (np.array(color[k]) - mean["H"]) / std["H"] for k in color}

    cache_name = os.path.join(output, "cache", "{}.npz".format(sample))
    try:
        data = np.load(cache_name)
        pixel = data["pixel"]
        pred = data["pred"]
        feature = data["feature"]
        value = data["value"]
        index = data["index"]
        size = data["size"]
        lfb = data["lfb"]
        print("Loaded")
    except FileNotFoundError:
        if len(set(stain for (_, _, stain) in slides).intersection(stains)) == 0:
            raise BadSlideError("No relevant IHC")

        pred = []
        feature = []
        value = []
        pixel = []
        index = []
        lfb = []

        max_window = max(window.values())
        min_downsample = min(downsample.values())

        transform = [
            torchvision.transforms.Resize(max_window // downsample["H"]),
            torchvision.transforms.CenterCrop(window["H"] // downsample["H"]),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=mean["H"], std=std["H"])
        ]
        transform = torchvision.transforms.Compose(transform)

        target_transform = {
            stain: [
                torchvision.transforms.Resize(max_window // downsample[stain]),
                torchvision.transforms.CenterCrop(window[stain] // downsample[stain]),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=mean[stain], std=std[stain])
            ]
            for stain in set(stains)
        }
        target_transform = {stain: torchvision.transforms.Compose(target_transform[stain]) for stain in target_transform}

        dataset = _get_dataset(src, os.path.join(output, "registration", sample), slides, stains, window, downsample, transform, target_transform)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, pin_memory=False, drop_last=False)

        with torch.set_grad_enabled(False):
            for (h, ihc, i, p) in tqdm.tqdm(loader, desc="Generating predictions"):
                h = h.to(device)
                he_label = torch.stack([torch.norm(h - torch.as_tensor(color[k], device=device).reshape(1, 3, 1, 1), dim=1) for k in sorted(color)]).argmin(0).cpu().numpy()

                y, f = model(h)
                pred.extend(y.cpu().numpy().tolist()) # TODO: these could be written to numpy directly
                feature.extend(f.cpu().numpy().tolist())
                pixel.extend(p.numpy().tolist())
                index.extend(i)
                lfb.extend((he_label == sorted(color).index("lfb")).astype(np.float).mean((1, 2)))

                label = []
                for (j, stain) in enumerate(sorted(set(stains))):
                    if False: # torch.isnan(ihc[j]).all():  # TODO shouldn't bother running on nan; need to extract shape somehow
                        label.append(ihc[j].shape[0] * [math.nan])
                    else:
                        # TODO: check why not nan
                        x = ihc_model[stain](ihc[j])
                        label.append(x.type(torch.float).cpu().numpy().tolist())
                value.extend([[k for j in i for k in j] for i in zip(*label)])

        pixel = np.array(pixel)
        pred = np.array(pred)
        feature = np.array(feature)
        value = np.array(value)  # TODO: this is a float?
        index = np.array(index)
        lfb = np.array(lfb)
        size = dataset.slide["H"].dimensions

        os.makedirs(os.path.dirname(cache_name), exist_ok=True)
        np.savez_compressed(
            cache_name,
            pixel=pixel,
            feature=feature,
            pred=pred,
            value=value,
            size=size,
            index=index,
            lfb=lfb,
        )

    return pixel, feature, pred, value[:, ihc_index], lfb, size, index


def save_ihc_patches(src, output, value, index, sample, slides, stains, targets, window, downsample, num_workers=4, n_examples=20):
    dataset = None
    ihc_values = {}
    for (j, (stain, target)) in enumerate(zip(stains, targets)):
        if not np.isnan(value[:, j]).any():
            os.makedirs(os.path.join(output, "ihc", sample, stain, target.lower().replace(" ", "-")), exist_ok=True)
            v_i = sorted(zip(value[:, j], index))
            ind = [round(i / (n_examples - 1) * len(v_i)) for i in range(n_examples - 1)] + [len(v_i) - 1]
            v_i = [v_i[i] for i in ind]
            ihc_values[(stain, target)] = list(zip(*v_i))[0]

            # # TODO: multithread
            # for (k, (v, ind)) in enumerate(tqdm.tqdm(v_i, desc="Saving IHC patches for {} {} ".format(stain, target))):
            #     if not os.path.isfile(os.path.join(output, "ihc", sample, stain, target.lower().replace(" ", "-"), "ihc_{}.jpg".format(k))):
            #         if dataset is None:
            #             dataset = _get_dataset(src, os.path.join(output, "registration", sample), slides, stains, window, downsample, transform=None, target_transform=None)
            #         (h, ihc, _, p) = dataset[ind]
            #         h.save(os.path.join(output, "ihc", sample, stain, target.lower().replace(" ", "-"), "h_{}.jpg".format(k)))
            #         ihc[sorted(set(stains)).index(stain)].save(os.path.join(output, "ihc", sample, stain, target.lower().replace(" ", "-"), "ihc_{}.jpg".format(k)))
        else:
            ihc_values[(stain, target)] = []
    return ihc_values


def save_attributions(src, output, pred, index, sample, slides, model, device, mean, std, stains, targets, window, downsample):
    dataset = None

    for (j, (stain, target)) in enumerate(zip(stains, targets)):
        rank = list(zip(*sorted(zip(pred[:, j], index))))[1]
        for (hl, r) in [("high", rank[-1:-6:-1]), ("low", rank[:5])]:
            root = os.path.join(output, "patches", sample, stain, target.lower().replace(" ", "-"), hl)
            os.makedirs(root, exist_ok=True)
            for (k, ind) in enumerate(r):  # TODO: should use dataloader to parallel
                if not os.path.isfile(os.path.join(root, "{}_attr.png".format(k))):
                    if dataset is None:
                        dataset = _get_dataset(src, os.path.join(output, "registration", sample), slides, stains, window, downsample, transform=None, target_transform=None)

                    # h, ihc, _, _, label, p = dataset[ind]
                    h, ihc, label, p = dataset[ind]

                    h.resize((512, 512)).save(os.path.join(root, "{}_h.jpg".format(k)))  # TODO hardcoded resize
                    ihc_image = ihc[sorted(set(stains)).index(stain)]
                    if ihc_image.size != (0, 0):
                        ihc_image.resize((512, 512)).save(os.path.join(root, "{}_ihc.jpg".format(k)))

                    context = 2048
                    if stain in dataset.slide:
                        img = dstain.utils.register.load_warped(dataset.slide[stain], (np.array(dataset.patch[ind]) - context // 2).tolist(), (dataset.window + context, dataset.window + context), dataset.warp[stain], dataset.downsample)
                        img.save(os.path.join(root, "{}_ihc_full.jpg".format(k)))

                    with dstain.utils.nn.set_extract_enabled(False):
                        ig = captum.attr.IntegratedGradients(model)
                        transform = [
                            torchvision.transforms.Resize(512),  # TODO: use downsample
                            torchvision.transforms.ToTensor(),
                            torchvision.transforms.Normalize(mean=mean["H"], std=std["H"])
                        ]
                        transform = torchvision.transforms.Compose(transform)

                        attribution, delta = ig.attribute(transform(h).unsqueeze(0).to(device), target=j, return_convergence_delta=True)  # TODO: check if target makes sense
                        attribution = attribution.squeeze(0)
                        # attribution = attribution.mean(0)
                        attribution = attribution.cpu().numpy()
                        assert(attribution.shape[1] == attribution.shape[2])

                    fig, ax = plt.subplots(figsize=(1, 1), dpi=attribution.shape[1], frameon="off")
                    fig.subplots_adjust(0, 0, 1, 1)
                    fig, ax = captum.attr.visualization.visualize_image_attr(
                            np.transpose(attribution, (1, 2, 0)),
                            original_image=np.array(h),
                            method="blended_heat_map",
                            alpha_overlay=0.75,
                            plt_fig_axis=(fig, ax),
                            sign="positive",
                            outlier_perc=10
                            )
                    # fig.canvas.draw()
                    plt.axis('off')
                    fig.savefig(os.path.join(root, "{}_attr.png".format(k)), bbox_inches='tight', pad_inches=0)
                    fig.savefig(os.path.join(root, "{}_attr.jpg".format(k)), bbox_inches='tight', pad_inches=0)
                    # fig.savefig('test.png')
                    # PIL.Image.frombytes('RGB', fig.canvas.get_width_height(),fig.canvas.tostring_rgb()).save(os.path.join(root, "{}_attr.png".format(j)))
                    # attribution = (attribution - attribution.min()) / (attribution.max() - attribution.min())
                    # PIL.Image.fromarray((255 * attribution).astype(np.uint8)).save(os.path.join(root, "{}_attr.png".format(j)))
                    # PIL.Image.fromarray(255 * label[i].numpy().astype(np.uint8)).save(os.path.join(output, "patches", sample, stain, hl, "{}_label.png".format(j)))
                    plt.close(fig)

                    if target == "Tangle":
                        min_size = 500
                        smooth_size = 5
                    else:
                        min_size = 2000
                        smooth_size = 10
                    
                    h, ihc, label, p = dataset[ind]
                    attr = attribution.sum(0)
                    attr[attr < 0] = 0
                    attr = scipy.ndimage.filters.gaussian_filter(attr, sigma=5)
                    attr /= attr.max()
                    thresh = np.sort(attr.reshape(-1))[round(attr.size * 0.9)]
                    attr = (attr > thresh)
                    structure = np.ones((3, 3), dtype=np.int)
                    labeled, ncomponents = scipy.ndimage.measurements.label(attr, structure)
                    np.unique(labeled, return_counts=True)
                    idx, count = np.unique(labeled, return_counts=True)
                    remove = set(idx[count < min_size])
                    for r in remove:  # TODO: slow...
                        attr[labeled == r] = False
                        labeled[labeled == r] = 0
                    
                    
                    def get_circle(n):
                        structure = np.zeros((2 * n + 1, 2 * n + 1), dtype=np.bool)
                        for i in range(n + 1):
                            for j in range(n + 1):
                                if i ** 2 + j ** 2 <= n ** 2:
                                    structure[n - i, n - j] = True
                                    structure[n - i, n + j] = True
                                    structure[n + i, n - j] = True
                                    structure[n + i, n + j] = True
                        return structure
                    
                    
                    attr = scipy.ndimage.binary_fill_holes(attr)
                    structure = get_circle(smooth_size)
                    attr = scipy.ndimage.binary_erosion(attr, structure=structure)
                    attr = scipy.ndimage.binary_dilation(attr, structure=structure)
                    
                    structure = np.ones((3, 3), dtype=np.int)
                    labeled, ncomponents = scipy.ndimage.measurements.label(attr, structure)
                    np.unique(labeled, return_counts=True)
                    idx, count = np.unique(labeled, return_counts=True)
                    remove = set(idx[count < min_size])
                    for r in remove:  # TODO: slow...
                        labeled[labeled == r] = 0
                    
                    labeled = (labeled != 0)
                    r = (labeled != np.vstack((labeled[1:, :], np.zeros((1, labeled.shape[1]), dtype=labeled.dtype))))
                    c = (labeled != np.hstack((labeled[:, 1:], np.zeros((labeled.shape[0], 1), dtype=labeled.dtype))))
                    edge = np.logical_or(r, c)
                    edge = scipy.ndimage.filters.maximum_filter(edge, size=2)
                    h = np.array(h)
                    h[edge, :] = 0
                    PIL.Image.fromarray(attr).save(os.path.join(root, "test.png"))
                    PIL.Image.fromarray(h).save(os.path.join(root, "{}_traced.png".format(k)))
                    PIL.Image.fromarray(h).save(os.path.join(root, "{}_traced.jpg".format(k)))


def ihc_cmap(x):
    if math.isnan(x):
        return (0., 0., 0., 0.)
    x = (2 * (x - 0.5))
    if x < 0:
        x = 0
    return tuple(((1 - x) * np.array([217, 211, 208]) + x * np.array([187, 154, 126])) / 255) + (1.,)


def lfb_cmap(x):
    if math.isnan(x):
        return (0., 0., 0., 0.)
    if x < 0.5:
        return tuple(np.array([118, 122, 184]) / 255) + (1.,)
    return tuple(np.array([203, 160, 201]) / 255) + (1.,)

def plot_grid(pixel, value, size, spot_size=512, scale=300, cmap=ihc_cmap, dpi=300):
    # print(pixel.min(0))
    # print(pixel.max(0))
    # print(size)
    # x = np.unique(pixel[:, 0])
    # x.sort()
    # print(x[1:] - x[:-1])
    # x = np.unique(pixel[:, 1])
    # x.sort()
    # print(x[1:] - x[:-1])
    fig = plt.figure(figsize=(size[0] / dpi / scale, size[1] / dpi / scale), frameon=False, dpi=dpi)  # TODO fig needs to be marginally bigger? (spots cut off)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")

    ax.scatter(pixel[:, 0] / scale, pixel[:, 1] / scale, color=list(map(cmap, value)), s=(spot_size / scale) ** 2, linewidth=0, edgecolors="none")

    ax.axis([0, size[0] / scale, size[1] / scale, 0])

    return fig, ax


def safe_roc_auc_score(a, b):
    try:
        return sklearn.metrics.roc_auc_score(a, b)
    except ValueError:
        return math.nan

