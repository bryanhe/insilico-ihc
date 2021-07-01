#!/usr/bin/env python3
import collections
import os
import numpy as np
import matplotlib.pyplot as plt

output = "."
# with open(os.path.join("output", "training", "log.csv"), "r") as log:
with open(os.path.join("output", "training_lr_1e-4", "log.csv"), "r") as log:
# with open(os.path.join("output", "training_soft", "log.csv"), "r") as log:
# with open(os.path.join("old_output", "old_training", "log.csv"), "r") as log:
    res = collections.defaultdict(list)
    for line in log:
        if line == "Epoch,Split,aB CE,T CE,aB AUC,T AUC\n":  # TODO: hardcoded
            continue
        epoch, split, *r = line.split(",")
        if epoch == "Baseline":
            epoch = 0
        else:
            try:
                epoch = int(epoch) + 1
            except:
                continue
        # assert len(res[split]) == epoch
        if len(res[split]) == epoch:
            res[split].append(None)
        res[split][epoch] = list(map(float, r))

res = {s: np.array(res[s]) for s in res}

fig = plt.figure(figsize=(3, 3))
plt.plot(1000 * np.arange(1, 1 + res["train"].shape[0]), res["train"][:, 0:2].mean(1), label="Train")
plt.plot(1000 * np.arange(1, 1 + res["valid"].shape[0]), res["valid"][:, 0:2].mean(1), label="Validation")
plt.legend(loc="best")
plt.xlabel("Iterations")
plt.ylabel("Cross-Entropy")
# TODO y axis start from 0
plt.tight_layout()
plt.savefig("ce.pdf")
plt.close(fig)

fig = plt.figure(figsize=(3, 3))
plt.plot(1000 * np.arange(1, 1 + res["train"].shape[0]), res["train"][:, 2:4].mean(1), label="Train")
plt.plot(1000 * np.arange(1, 1 + res["valid"].shape[0]), res["valid"][:, 2:4].mean(1), label="Validation")
# plt.plot(500 * np.arange(1, 1 + len(x)), [(x[i][1][1][0] + x[i][1][1][1] + x[i][2][1][0] + x[i][2][1][1]) / 4 for i in range(len(x))], "--", label="Test")
plt.legend(loc="best")
plt.xlabel("Iterations")
plt.ylabel("AUC")
# TODO y axis start from 0
plt.tight_layout()
plt.savefig("auc.pdf")
plt.close(fig)

fig = plt.figure(figsize=(3, 3))
plt.plot(1000 * np.arange(1, 1 + res["train"].shape[0]), res["train"][:, 4:6].mean(1), label="Train")
plt.plot(1000 * np.arange(1, 1 + res["valid"].shape[0]), res["valid"][:, 4:6].mean(1), label="Validation")
# plt.plot(500 * np.arange(1, 1 + len(x)), [(x[i][1][1][0] + x[i][1][1][1] + x[i][2][1][0] + x[i][2][1][1]) / 4 for i in range(len(x))], "--", label="Test")
plt.legend(loc="best")
plt.xlabel("Iterations")
plt.ylabel("AUC (high/low)")
# TODO y axis start from 0
plt.tight_layout()
plt.savefig("auc_hl.pdf")
plt.close(fig)
