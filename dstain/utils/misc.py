import signal
import re
import torch
import numpy as np
import tqdm
import collections
import tempfile
import os
import jinja2


class GracefulInterruptHandler(object):
    # https://stackoverflow.com/questions/1112343/how-do-i-capture-sigint-in-python
    # https://gist.github.com/nonZero/2907502

    def __init__(self, sig=signal.SIGINT):
        self.sig = sig

    def __enter__(self):

        self.interrupted = False
        self.released = False

        self.original_handler = signal.getsignal(self.sig)

        def handler(signum, frame):
            self.release()
            self.interrupted = True

        signal.signal(self.sig, handler)

        return self

    def __exit__(self, type, value, tb):
        self.release()

    def release(self):

        if self.released:
            return False

        signal.signal(self.sig, self.original_handler)

        self.released = True

        return True


def symlink(target, link_name, overwrite=False):
    # From https://stackoverflow.com/questions/8299386/modifying-a-symlink-in-python/55742015#55742015
    '''
    Create a symbolic link named link_name pointing to target.
    If link_name exists then FileExistsError is raised, unless overwrite=True.
    When trying to overwrite a directory, IsADirectoryError is raised.
    '''

    if not overwrite:
        os.symlink(target, link_name)
        return

    # os.replace() may fail if files are on different filesystems
    link_dir = os.path.dirname(link_name)

    # Create link to target with temporary filename
    while True:
        temp_link_name = tempfile.mktemp(dir=link_dir)

        # os.* functions mimic as closely as possible system functions
        # The POSIX symlink() returns EEXIST if link_name already exists
        # https://pubs.opengroup.org/onlinepubs/9699919799/functions/symlink.html
        try:
            os.symlink(target, temp_link_name)
            break
        except FileExistsError:
            pass

    # Replace link_name with temp_link_name
    try:
        # Pre-empt os.replace on a directory with a nicer message
        if os.path.isdir(link_name):
            raise IsADirectoryError(f"Cannot symlink over existing directory: '{link_name}'")
        os.replace(temp_link_name, link_name)
    except:
        if os.path.islink(temp_link_name):
            os.remove(temp_link_name)
        raise


def latexify():
    import matplotlib
    params = {
        'backend': 'pdf',
        'axes.titlesize':  8,
        'axes.labelsize':  8,
        'font.size':       8,
        'legend.fontsize': 8,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        # 'text.usetex': True,
        'font.family': 'DejaVu Serif',
        'font.serif': 'Computer Modern',
    }
    matplotlib.rcParams.update(params)


def get_mean_and_std(dataset: torch.utils.data.Dataset,
                     samples: int = 128,
                     batch_size: int = 8,
                     num_workers: int = 4,
                     index: int = 0,
                     class_index: int = None,
                     class_by_col: bool = True
                     ):
    """Computes mean and std from samples from a Pytorch dataset.

    Args:
        dataset (torch.utils.data.Dataset): A Pytorch dataset.
            ``dataset[i][0]'' is expected to be the i-th video in the dataset, which
            should be a ``torch.Tensor'' of dimensions (channels=3, frames, height, width)
        samples (int or None, optional): Number of samples to take from dataset. If ``None'', mean and
            standard deviation are computed over all elements.
            Defaults to 128.
        batch_size (int, optional): how many samples per batch to load
            Defaults to 8.
        num_workers (int, optional): how many subprocesses to use for data
            loading. If 0, the data will be loaded in the main process.
            Defaults to 4.
        index (int or List[int]): TODO documentation
        class_index (int, optional): TODO documentation
        class_by_col (bool, optional): TODO documentation

    Returns:
       A tuple of the mean and standard deviation. Both are represented as np.array's of dimension (channels,).
    """

    if samples is not None and len(dataset) > samples:
        indices = np.random.choice(len(dataset), samples, replace=False)
        dataset = torch.utils.data.Subset(dataset, indices)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)

    collapse = isinstance(index, int)
    if collapse:
        index = [index]

    n = [0]  # number of elements taken (should be equal to samples by end of for loop)
    s1 = [0.]  # sum of elements along channels (ends up as np.array of dimension (channels,))
    s2 = [0.]  # sum of squares of elements along channels (ends up as np.array of dimension (channels,))
    count = None
    for data in tqdm.tqdm(dataloader, desc="Mean/Std"):
        for i in range(len(index)):
            x = data[index[i]].type(torch.float64)
            x = x.transpose(0, 1).contiguous().view(3, -1)
            n[i] += x.shape[1]
            s1[i] += torch.sum(x, dim=1).numpy()
            s2[i] += torch.sum(x ** 2, dim=1).numpy()
        if class_index is not None:
            if class_by_col:
                if count is None:
                    count = [collections.defaultdict(int) for _ in range(data[class_index].shape[1])]
                for (i, d) in enumerate(data[class_index].t()):
                    for (j, c) in list(zip(*map(lambda x: x.numpy(), d.unique(return_counts=True)))):
                        count[i][j] += c
            else:
                if count is None:
                    count = collections.defaultdict(int)
                for (i, c) in list(zip(*map(lambda x: x.numpy(), data[class_index].unique(return_counts=True)))):
                    count[i] += c
    mean = [s1[i] / n[i] for i in range(len(index))]  # type: np.ndarray
    std = [np.sqrt(s2[i] / n[i] - mean[i] ** 2) for i in range(len(index))]  # type: np.ndarray

    mean = [m.astype(np.float32) for m in mean]
    std = [s.astype(np.float32) for s in std]

    if collapse:
        mean = mean[0]
        std = std[0]

    if class_index is None:
        return mean, std
    return mean, std, count


def read_sample_file(sample_file, use_split=True):
    # TODO: allow sample_file to be File or str
    samples = collections.defaultdict(lambda: collections.defaultdict(list))
    for line in sample_file:
        sample, path, basename, stain, split = line.strip().split("\t")
        if not use_split:
            split = ""
        samples[split][sample].append((path, basename, stain))

    if not use_split:
        samples = samples[""]

    return samples


def render_latex(filename, **kwargs):
    # TODO: could also write output?

    # http://eosrei.net/articles/2015/11/latex-templates-python-and-jinja2-generate-pdfs
    # https://tug.org/tug2019/slides/slides-ziegenhagen-python.pdf
    latex_jinja_env = jinja2.Environment(
        block_start_string='\\BLOCK{',
        block_end_string='}',
        variable_start_string='\\VAR{',
        variable_end_string='}',
        comment_start_string='\\#{',
        comment_end_string='}',
        line_statement_prefix='%%',
        line_comment_prefix='%#',
        trim_blocks=True,
        lstrip_blocks=True,
        autoescape=False,
        loader=jinja2.FileSystemLoader(os.path.abspath(os.path.dirname(filename)))
    )
    template = latex_jinja_env.get_template(os.path.basename(filename))
    return template.render(**kwargs)


# https://stackoverflow.com/questions/4623446/how-do-you-sort-files-numerically
def tryint(s):
    try:
        return int(s)
    except:
        return s


def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [ tryint(c) for c in re.split('([0-9]+)', s) ]

