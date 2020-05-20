import tensorflow as tf
from text import symbols

tacotron_hparameters = tf.contrib.training.HParams(
    ################################
    # Experiment Parameters        #
    ################################
    epochs=500,
    iters_per_checkpoint=1000,
    seed=1234,
    fp16_run=False,
    use_cuda=False,
    ignore_layers=['embedding.weight'],

    ################################
    # Data Parameters             #
    ################################
    load_mel_from_disk=True,
    training_files='/Users/olehmatsuk/Thesis/data/mels/konekorpus/data.csv',
    validation_files='/Users/olehmatsuk/Thesis/data/mels/konekorpus/data.csv',
    text_cleaners=['english_cleaners'],

    ################################
    # Audio Parameters             #
    ################################
    sampling_rate=22050,
    filter_length=1024,
    hop_length=256,
    win_length=1024,
    n_mel_channels=80,
    mel_fmin=0.0,
    mel_fmax=8000.0,

    ################################
    # Model Parameters             #
    ################################
    n_symbols=len(symbols),
    symbols_embedding_dim=512,

    # Encoder parameters
    encoder_kernel_size=5,
    encoder_n_convolutions=3,
    encoder_embedding_dim=512,

    # Decoder parameters
    n_frames_per_step=1,  # currently only 1 is supported
    decoder_rnn_dim=1024,
    prenet_dim=256,
    max_decoder_steps=1000,
    gate_threshold=0.5,
    p_attention_dropout=0.1,
    p_decoder_dropout=0.1,

    # Attention parameters
    attention_rnn_dim=1024,
    attention_dim=128,

    # Location Layer parameters
    attention_location_n_filters=32,
    attention_location_kernel_size=31,

    # Mel-post processing network parameters
    postnet_embedding_dim=512,
    postnet_kernel_size=5,
    postnet_n_convolutions=5,

    ################################
    # Optimization Hyperparameters #
    ################################
    learning_rate=1e-3,
    weight_decay=1e-6,
    grad_clip_thresh=1.0,
    batch_size=64,
    mask_padding=True  # set model's padded outputs to padded values
)

speaker_encoder_hparameters = tf.contrib.training.HParams(
    ################################
    # Experiment Parameters        #
    ################################
    epochs=100,
    iters_per_checkpoint=1000,
    seed=1234,
    fp16_run=False,
    use_cuda=False,

    ################################
    # Data Parameters             #
    ################################
    training_files='/Users/olehmatsuk/Thesis/data/mels/konekorpus/train.csv',
    validation_files='/Users/olehmatsuk/Thesis/data/mels/konekorpus/val.csv',

    ################################
    # Model Parameters             #
    ################################
    n_mel_channels=80,
    n_fragment_mel_windows=70,
    speaker_encoder_n_layers=3,
    speaker_encoder_rnn_dim=756,
    speaker_encoder_dim=256,

    ################################
    # Optimization Hyperparameters #
    ################################
    learning_rate=1e-2,
    grad_clip_thresh=3.0,
    batch_size_speakers=11,
    batch_size_speaker_samples=30,
)

model_hparams = {
    'tacotron': tacotron_hparameters,
    'speaker_encoder': speaker_encoder_hparameters
}


def create_hparams(model='tacotron', hparams_string=None):
    hparams = model_hparams[model]

    if hparams_string:
        hparams.parse(hparams_string)

    return hparams
