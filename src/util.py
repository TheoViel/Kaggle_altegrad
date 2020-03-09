from params import *

import os
import torch
import random
import sklearn
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from itertools import product


def seed_everything(seed):
    """
    Seeds basic parameters for reproductibility of results
    
    Arguments:
        seed {int} -- Number of the seed
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_model_weights(model, filename, verbose=1, cp_folder=CP_PATH):
    """
    Saves the weights of a PyTorch model
    
    Arguments:
        model {torch module} -- Model to save the weights of
        filename {str} -- Name of the checkpoint
    
    Keyword Arguments:
        verbose {int} -- Whether to display infos (default: {1})
        cp_folder {str} -- Folder to save to (default: {CP_PATH})
    """
    if verbose:
        print(f"\n -> Saving weights to {os.path.join(cp_folder, filename)}\n")
    torch.save(model.state_dict(), os.path.join(cp_folder, filename))


def load_model_weights(model, filename, verbose=1, cp_folder=CP_PATH):
    """
    Loads the weights of a PyTorch model. The exception handles cpu/gpu incompatibilities
    
    Arguments:
        model {torch module} -- Model to load the weights to
        filename {str} -- Name of the checkpoint
    
    Keyword Arguments:
        verbose {int} -- Whether to display infos (default: {1})
        cp_folder {str} -- Folder to load from (default: {CP_PATH})
    
    Returns:
        torch module -- Model with loaded weights
    """
    if verbose:
        print(f"\n -> Loading weights from {os.path.join(cp_folder,filename)}\n")
    try:
        model.load_state_dict(os.path.join(cp_folder, filename), strict=strict)
    except BaseException:
        model.load_state_dict(
            torch.load(os.path.join(cp_folder, filename), map_location="cpu"),
            strict=True,
        )
    return model


def count_parameters(model, all=False):
    """
    Count the parameters of a model
    
    Arguments:
        model {torch module} -- Model to count the parameters of
    
    Keyword Arguments:
        all {bool} -- Whether to include not trainable parameters in the sum (default: {False})
    
    Returns:
        int -- Number of parameters
    """
    if all:
        return sum(p.numel() for p in model.parameters())
    else:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


class ConfusionMatrixDisplay:
    """
    From sklearn : https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ConfusionMatrixDisplay.html
    """

    def __init__(self, confusion_matrix, display_labels):
        self.confusion_matrix = confusion_matrix
        self.display_labels = display_labels

    def plot(
        self,
        include_values=True,
        cmap="viridis",
        xticks_rotation="horizontal",
        values_format=None,
        ax=None,
    ):
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))
        else:
            fig = ax.figure

        cm = self.confusion_matrix
        n_classes = cm.shape[0]
        self.im_ = ax.imshow(cm, interpolation="nearest", cmap=cmap)
        self.text_ = None

        cmap_min, cmap_max = self.im_.cmap(0), self.im_.cmap(256)

        if include_values:
            self.text_ = np.empty_like(cm, dtype=object)
            if values_format is None:
                values_format = ".0f"

            # print text with appropriate color depending on background
            thresh = (cm.max() + cm.min()) / 2.0
            for i, j in product(range(n_classes), range(n_classes)):
                color = cmap_max if cm[i, j] < thresh else cmap_min
                self.text_[i, j] = ax.text(
                    j,
                    i,
                    format(cm[i, j], values_format),
                    ha="center",
                    va="center",
                    color=color,
                )

        ax.set(
            xticks=np.arange(n_classes),
            yticks=np.arange(n_classes),
            xticklabels=self.display_labels,
            yticklabels=self.display_labels,
        )

        ax.set_ylabel("True label", fontsize=12)
        ax.set_xlabel("Predicted label", fontsize=12)

        ax.tick_params(axis="both", which="major", labelsize=11)
        ax.tick_params(axis="both", which="minor", labelsize=11)

        ax.set_ylim((n_classes - 0.5, -0.5))
        plt.setp(ax.get_xticklabels(), rotation=40)

        self.figure_ = fig
        self.ax_ = ax
        return self


def plot_confusion_matrix(
    y_pred,
    y_true,
    labels=None,
    sample_weight=None,
    normalize=None,
    display_labels=None,
    include_values=True,
    xticks_rotation="horizontal",
    values_format=None,
    cmap="viridis",
    ax=None,
):
    """
    From sklearn : https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ConfusionMatrixDisplay.html
    """
    cm = sklearn.metrics.confusion_matrix(
        y_true, y_pred, sample_weight=sample_weight, labels=labels
    )

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASSES)
    return disp.plot(
        include_values=include_values,
        cmap=cmap,
        ax=ax,
        xticks_rotation=xticks_rotation,
        values_format=values_format,
    )


def cross_entropy(pred, truth):
    """
    Computes the cross entropy metric
    
    Arguments:
        pred {array} -- predictions
        truth {array} -- ground truth
    
    Returns:
        float -- The cross entropy value
    """
    return -np.sum([np.log(pred[i, int(truth[i])]) for i in range(len(truth))]) / len(
        truth
    )


def plot_categorical(data, orient="v", title="", ticks=[], ylim=None, xlim=None, figsize=(10, 6), order=None):
    """
    Helper to plot categorical variables
    
    Arguments:
        data {list or array} -- Data to plot
    
    Keyword Arguments:
        orient {str} -- Orientation of the plot (default: {"v"})
        title {str} -- Title of the plot (default: {""})
        ticks {list} -- Plot ticks i.e. categories (default: {[]})
        ylim {2 elements tuple} -- limits of the y axis  (default: {None})
        xlim {2 elements tuple} -- limits of the x axis (default: {None})
        figsize {2 elements tuple} -- Size of the figure (default: {(10, 6)})
        order {list} -- Order in which to display the categories (default: {None})
    """
    plt.figure(figsize=figsize)

    if orient == "v":
        splot = sns.countplot(data, order=order)
    else:
        splot = sns.countplot(y=data, order=order)

    for p in splot.patches:
        if orient == "v":
            splot.annotate(
                format(p.get_height() / len(data) * 100, ".1f") + "%",
                (p.get_x() + p.get_width() / 2.0, p.get_height()),
                ha="center",
                va="center",
                xytext=(0, 10),
                textcoords="offset points",
            )
        else:
            splot.annotate(
                format(p.get_width() / len(data) * 100, ".1f") + "%",
                (p.get_width() * 1.01, p.get_y() + p.get_height() / 2),
                ha="center",
                va="center",
                xytext=(15, 0),
                textcoords="offset points",
            )

    plt.title(title, size=15)

    if ylim is not None:
        plt.ylim(ylim)
    if xlim is not None:
        plt.xlim(xlim)

    if order is not None:
        ticks = order

    if ticks is not None:
        if orient == "v":
            plt.xticks(range(len(ticks)), ticks, fontsize=14)
        else:
            plt.yticks(range(len(ticks)), ticks, fontsize=14)
    plt.show()
