#!/usr/bin/env python3

import os
import sklearn.metrics
import pickle
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import dstain
import click

import scipy
@click.command()
@click.argument("prediction", type=click.File("rb"))
@click.option("--output", type=click.Path(), default="output/analyze")
def analyze(prediction, output):
    os.makedirs(output, exist_ok=True)
    dstain.utils.latexify()

    # Make colorbar
    fig = plt.figure(figsize=(2.25, 0.75))
    norm = matplotlib.colors.Normalize(vmin=-3, vmax=3)
    cmap = plt.get_cmap("viridis")
    cb = matplotlib.colorbar.ColorbarBase(plt.gca(), cmap=cmap,
                                          norm=norm,
                                          orientation='horizontal')
    cb.set_ticks(range(-3, 4))
    cb.set_label("z-score")
    plt.tight_layout()
    fig.savefig(os.path.join(output, "colorbar.pdf"))
    plt.close(fig)

    data = np.load(prediction, allow_pickle=True)
    value = [np.array(_) for _ in data["y_true"]]
    binary = [np.array(_) for _ in data["y_true_bin"]]
    pred = [np.array(_) for _ in data["y_pred"]]
    block = data["block_list"]
    pixel = [np.array(_) for _ in data["pixel_list"]]

    for (i, stain) in enumerate(["aBeta", "pTau"]):
        thresh = {"aBeta": 0.004, "pTau": 0.006}[stain]
        print(stain)
        # print(stain, sklearn.metrics.r2_score(value[i], pred[i]))
        # print("AUC (median): ", sklearn.metrics.roc_auc_score(value[i][value_valid[i]] > np.median(value[i][value_valid[i]]), pred[i][value_valid[i]]))
        # p = 0.001
        # print("Percent with at least {} staining: {}".format(p, (np.array(value[i][value_valid[i]]) > p).mean()))
        # print("Median", np.median(value[i][value_valid[i]]))
        # print("Mean", np.mean(value[i][value_valid[i]]))
        # print("AUC ({}): {}".format(p, sklearn.metrics.roc_auc_score(np.array(value[i][value_valid[i]]) > p, pred[i][value_valid[i]])))
        print("AUC: {}".format(sklearn.metrics.roc_auc_score(binary[i], pred[i])))
        fig = plt.figure(figsize=(2.0, 2.0))
        plt.scatter(value[i], pred[i], s=1, edgecolor="none")
        plt.xlabel("True Staining")
        plt.ylabel("Predicted Staining")
        plt.title(stain)
        plt.tight_layout()
        plt.savefig(os.path.join(output, "{}.pdf".format(stain)))

        real = []
        prediction = []
        hist = []
        for b in set(block[i]):
            mask = (np.array(block[i]) == b)
            real.append((value[i][mask] > thresh).mean())
            prediction.append((pred[i][mask] > 0).mean())
            try:
                print("{}: {}".format(b, sklearn.metrics.roc_auc_score(binary[i][mask], pred[i][mask])))
                hist.append(sklearn.metrics.roc_auc_score(binary[i][mask], pred[i][mask]))
            except:
                pass
            p = pixel[i][mask, :]
            xmin = p[:, 0].min()
            xmax = p[:, 0].max()
            ymin = p[:, 1].min()
            ymax = p[:, 1].max()

            for (v, name) in [(value[i][mask], "value"), (pred[i][mask], "pred")]:

                margin = 10000
                xsize = xmax - xmin + 2 * margin
                ysize = ymax - ymin + 2 * margin
                figsize = (0.00017 * xsize, 0.00017 * ysize)

                tol = 1e-5
                v = ((v - v.mean(0)) / (3 * v.std(0) + tol) + 0.5).clip(tol, 1 - tol)
                fig = plt.figure(figsize=figsize)
                plt.scatter(p[:, 0], p[:, 1], color=list(map(cmap, v)), s=1000, linewidth=0, edgecolors="none")
                plt.gca().axis("off")
                plt.gca().set_aspect("equal")
                plt.xticks([])
                plt.yticks([])
                plt.axis([xmin - margin, xmax + margin, ymin - margin, ymax + margin])
                plt.gca().invert_yaxis()
                plt.tight_layout()
                os.makedirs(os.path.join(output, "images", b), exist_ok=True)
                fig.savefig(os.path.join(output, "images", b, "{}_{}.pdf".format(stain, name)))
                plt.close(fig)

        fig = plt.figure(figsize=(2.0, 2.0))
        plt.hist(hist)
        plt.xlim([0, 1])
        plt.xlabel("AUC")
        plt.ylabel("Number of Blocks")
        plt.title(stain)
        plt.tight_layout()
        plt.savefig(os.path.join(output, "hist_{}.pdf".format(stain)))

        fig = plt.figure(figsize=(2.0, 2.0))
        plt.scatter(real, prediction, s=9)
        #plt.xlim([0, 1])
        plt.axis([-0.1, 1, -0.1, 1])
        plt.xlabel("Real Staining")
        plt.ylabel("Predicted Staining")
        plt.title(stain)
        plt.tight_layout()
        plt.savefig(os.path.join(output, "scatter_{}.pdf".format(stain)))

        print("AUC (patient): ", sklearn.metrics.roc_auc_score(real > np.median(real), prediction))
        print("R2 (patient): ", sklearn.metrics.r2_score(real, prediction))
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(real, prediction)
        print( scipy.stats.linregress(real, prediction))

    plt.close(fig)

if __name__ == "__main__":
        main()
