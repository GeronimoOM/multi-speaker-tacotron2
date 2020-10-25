import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pylab as plt
import seaborn as sns


genders = {
    'Kristo': 1,
    'Robert': 1,
    'Katarina': 0,
    'Stella': 0,
    'Birgit_Itse': 0,
    '01712caf3ec1a54c0a84c76e684055ffe51c42afed8ba7bf11353a1e8f5271a998cd7068a14a913bdb1d156c9daaf6eac321b16e449d5f7afefefa9045ea148e': 0,
    '25ed618a723bace6bcd39a27daaafce7a0ed93731c62ffff9ad287e167dd7692845c427db60976a227560627936194554cdfd6bfba73a6a096c074a7e36ce29b': 1,
    '30dcad1da7c1a203fd74d11eee0b2ccd8982a2ec2028e0d05e0d7afe13760a1f7da0fee2860aa9dd61bea8dde7df89090105a869353cf047a6367478f76638a5': 1,
    '3a27e89f0e71bebb6f635ba79139f4dd76233a7b959a1826d9b12f0017cd0ffced99b98eff82cf5447e3d53cd77115c34df7a66667737eb9f9877fb3dcd3bdfb': 1,
    '5f856928bd63217b78c2bb23e884f3f99053891a614588b705fe22e89efd119d9a978e6aa5a16b621a9b19717e93e617e92ba3cabb44dab19c6d1898cfede58b': 0,
    '77126e7e57223ba4e9481008de80b9d821ceb4d743d9d702ab44079f47d20bf609c8953a4c3d402fad7db080100aeae05bea34a4737982814bee3a8fcf65ff1c': 1,
    '8358af0a31f656f9270a9640c63752897ef742c5134b3ab324a1ff4b7500ce6fbb2f4971e4a622b0fa1fe1528061abf32befae3d6236dbd74d426810bf0b7467': 1,
    '9cb461c87172698a4d32957fdf32a6964670bdeebc5a7e671d7e97191fc6ed5eeb55e3c27c06b6813bb1241336c83b2f690701cfbe8ac597ab258e175612f867': 1,
    'e2fa85a7d0b8e49d676cfebaf2a999afa6d5f0958df2ee266c055b1c7e99b390eb26e706d58f7e95873ca33db0c315c907077c68380bba82a4edf9939be54da8': 0,
    'e4fcec16b2c8b4740377b19f37e90738f8415cafc1fb58fe37731a07f5450fdfae3fe17bdf28f19092d2f8c06003635e14f6c4d1c16f555e306cf67c95baaa63': 0,
}

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


def plot_speaker_embeddings_to_numpy(embeddings_pca, speakers):
    fig, ax = plt.subplots(figsize=(15, 15))
    for speaker in speakers:
        speaker_index = speakers == speaker
        plt.scatter(embeddings_pca[speaker_index, 0], embeddings_pca[speaker_index, 1],
                    marker=('x' if genders[speaker] == 1 else '.'))

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
