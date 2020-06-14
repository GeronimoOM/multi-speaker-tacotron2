import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pylab as plt
import seaborn as sns


def save_figure_to_numpy(fig):
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return data


def plot_alignment_to_numpy(alignment, info=None):
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(alignment, aspect='auto', origin='lower',
                   interpolation='none')
    fig.colorbar(im, ax=ax)
    xlabel = 'Decoder timestep'
    if info is not None:
        xlabel += '\n\n' + info
    plt.xlabel(xlabel)
    plt.ylabel('Encoder timestep')
    plt.tight_layout()

    fig.canvas.draw()
    data = save_figure_to_numpy(fig)
    plt.close()
    return data


def plot_spectrogram_to_numpy(spectrogram):
    fig, ax = plt.subplots(figsize=(12, 3))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower",
                   interpolation='none')
    plt.colorbar(im, ax=ax)
    plt.xlabel("Frames")
    plt.ylabel("Channels")
    plt.tight_layout()

    fig.canvas.draw()
    data = save_figure_to_numpy(fig)
    plt.close()
    return data


def plot_gate_outputs_to_numpy(gate_targets, gate_outputs):
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.scatter(range(len(gate_targets)), gate_targets, alpha=0.5,
               color='green', marker='+', s=1, label='target')
    ax.scatter(range(len(gate_outputs)), gate_outputs, alpha=0.5,
               color='red', marker='.', s=1, label='predicted')

    plt.xlabel("Frames (Green target, Red predicted)")
    plt.ylabel("Gate State")
    plt.tight_layout()

    fig.canvas.draw()
    data = save_figure_to_numpy(fig)
    plt.close()
    return data


def plot_speaker_embeddings_to_numpy(embeddings_pca, labels):
    fig, ax = plt.subplots(figsize=(6, 4))
    for label in np.unique(labels):
        label_index = labels == label
        plt.scatter(embeddings_pca[label_index, 0], embeddings_pca[label_index, 1])

    plt.tight_layout()
    fig.canvas.draw()
    data = save_figure_to_numpy(fig)
    plt.close()
    return data


def plot_speaker_embeddings2_to_numpy(cosine_mat, labels):
    fig, ax = plt.subplots(figsize=(15, 15))
    N = len(np.unique(labels))
    M = cosine_mat.shape[0] // N
    mask = np.triu(np.ones_like(cosine_mat, dtype=np.bool), k=1)
    sns.heatmap(cosine_mat, mask=mask, cmap=sns.color_palette("Purples"), cbar=False, vmin=-1, vmax=1,
                xticklabels=False, yticklabels=False, square=True)
    ax.hlines([M * i for i in range(N)], xmin=0, xmax=[M * i for i in range(N)], colors='w')
    ax.vlines([M * i for i in range(N)], ymin=[M * i for i in range(N)], ymax=ax.get_ylim()[0], colors='w')

    plt.tight_layout()
    fig.canvas.draw()
    data = save_figure_to_numpy(fig)
    plt.close()
    return data
