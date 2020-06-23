from hparams import create_hparams
import argparse
import os
from audio import init_stft
from preprocessors import common_voice_preprocessor, konekorpus_preprocessor, vctk_preprocessor

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--preprocessor', type=str)
    parser.add_argument('-i', '--input_directory', type=str)
    parser.add_argument('-o', '--output_directory', type=str)

    args = parser.parse_args()

    preprocessor = None
    if args.preprocessor == 'common_voice':
        preprocessor = common_voice_preprocessor
    elif args.preprocessor == 'konekorpus':
        preprocessor = konekorpus_preprocessor
    elif args.preprocessor == 'vctk':
        preprocessor = vctk_preprocessor

    os.makedirs(args.output_directory, exist_ok=True)

    stft = init_stft(create_hparams())
    preprocessor.preprocess(args.input_directory, args.output_directory, stft)
