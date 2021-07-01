import numpy as np
import PIL


def grey(img):
    if isinstance(img, PIL.Image.Image):
        img = np.array(img)
    return np.logical_and(np.all(img < 150, 2), (img.max(2) - img.min(2)) <= 10)
    # return np.logical_or(np.all(img == 255, 2), np.logical_and(np.all(img < 150, 2), (img.max(2) - img.min(2)) <= 10))


def otsu(img, mask=None, histogram=None, lower=None, upper=None):
    if isinstance(img, PIL.Image.Image):
        img = np.array(img)
    assert len(img.shape) in [2, 3]
    if len(img.shape) == 3:
        img = img.astype(np.int).sum(2)

    if mask is not None:
        x = img[mask]
    else:
        x = img.reshape(-1)

    if lower is not None or upper is not None:
        x = x.clip(lower, upper)
    n = x.size
    value, count = np.unique(x, return_counts=True)

    sigma2 = []
    for v in value[1:]:
        mask0 = (value < v)  # Mask for below threshld v (low)
        mask1 = (value >= v)  # Mask for at least threshld v (high)

        # Fraction low/high
        w0 = (count[mask0].sum() / n)
        w1 = (count[mask1].sum() / n)

        # Mean of low/high values
        u0 = (value[mask0] * count[mask0]).sum() / count[mask0].sum()
        u1 = (value[mask1] * count[mask1]).sum() / count[mask1].sum()

        sigma2.append(w0 * w1 * (u0 - u1) ** 2)
    sigma2 = np.array(sigma2)
    thresh = value[sigma2.argmax() + 1]

    if histogram is not None:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2, 1, sharex=True)

        ax[0].hist(x, range(3 * 255))
        axis = ax[0].axis()
        ax[0].plot([thresh, thresh], ax[0].get_ylim(), color="k", linewidth=1)
        ax[0].axis(axis)

        ax[1].plot(value[1:], sigma2)
        axis = ax[1].axis()
        ax[1].plot([thresh, thresh], ax[0].get_ylim(), color="k", linewidth=1)
        ax[1].axis(axis)

        plt.tight_layout()
        plt.savefig(histogram)
        plt.close(fig)

    return img < thresh, thresh

def ght(img, mask=None, histogram=None, nu=(2 ** 29.5), tau=(2 ** 3.125), kappa=(2 ** 22.25), omega=(2 ** -3.25), lower=None, upper=None):
    if isinstance(img, PIL.Image.Image):
        img = np.array(img)
    assert len(img.shape) in [2, 3]
    if len(img.shape) == 3:
        img = img.astype(np.int).sum(2)
    if mask is not None:
        vals = img[mask]
    else:
        vals = img.reshape(-1)

    # if lower is not None or upper is not None:
    #     vals = vals.clip(lower, upper)

    # nsum = vals.size
    # x, n = np.unique(vals, return_counts=True)
    # w0 = n.cumsum()
    # w1 = n[::-1].cumsum()[::-1]
    # pi0 = w0 / nsum
    # pi1 = w1 / nsum
    # mu0 = (n * x).cumsum() / w0
    # mu1 = (n * x)[::-1].cumsum()[::-1] / w1
    # d0 = (n * x ** 2).cumsum() - w0 * mu0 ** 2
    # d1 = (n * x ** 2)[::-1].cumsum()[::-1] - w1 * mu1 ** 2

    # v0 = (pi0 * nu * tau ** 2 + d0) / (pi0 * nu + w0)
    # v1 = (pi1 * nu * tau ** 2 + d1) / (pi1 * nu + w1)

    # f0 = -d0 / v0 - w0 * np.log(v0) + 2 * (w0 + kappa * omega) * np.log(w0)
    # f1 = -d1 / v1 - w1 * np.log(v1) + 2 * (w1 + kappa * (1 - omega)) * np.log(w1)

    csum = lambda z: np.cumsum(z)[:-1]
    dsum = lambda z: np.cumsum(z[::-1])[-2::-1]
    argmax = lambda x, f: np.mean(x[:-1][f == np.max(f)])  # Use the mean for ties.
    clip = lambda z: np.maximum(1e-30, z)

    assert nu >= 0
    assert tau >= 0
    assert kappa >= 0
    assert omega >= 0 and omega <= 1

    x, n = np.unique(vals, return_counts=True)
    assert np.all(n >= 0)
    assert np.all(x[1:] >= x[:-1])
    w0 = clip(csum(n))
    w1 = clip(dsum(n))
    p0 = w0 / (w0 + w1)
    p1 = w1 / (w0 + w1)
    mu0 = csum(n * x) / w0
    mu1 = dsum(n * x) / w1
    d0 = csum(n * x**2) - w0 * mu0**2
    d1 = dsum(n * x**2) - w1 * mu1**2

    v0 = clip((p0 * nu * tau**2 + d0) / (p0 * nu + w0))
    v1 = clip((p1 * nu * tau**2 + d1) / (p1 * nu + w1))
    f0 = -d0 / v0 - w0 * np.log(v0) + 2 * (w0 + kappa *      omega)  * np.log(w0)
    f1 = -d1 / v1 - w1 * np.log(v1) + 2 * (w1 + kappa * (1 - omega)) * np.log(w1)

    thresh = x[(f0 + f1).argmax()]

    if histogram is not None:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2, 1, sharex=True)

        ax[0].hist(vals, range(3 * 255))
        axis = ax[0].axis()
        ax[0].plot([thresh, thresh], ax[0].get_ylim(), color="k", linewidth=1)
        ax[0].axis(axis)

        ax[1].plot(x[:-1], f0 + f1)
        axis = ax[1].axis()
        ax[1].plot([thresh, thresh], ax[0].get_ylim(), color="k", linewidth=1)
        ax[1].axis(axis)

        plt.tight_layout()
        plt.savefig(histogram)
        plt.close(fig)

    return img < thresh, thresh

