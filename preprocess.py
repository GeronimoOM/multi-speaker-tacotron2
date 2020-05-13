from hparams import create_hparams
import argparse
import importlib.util
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--preprocessor', type=str, default=None,
                        help='preprocessor name')
    parser.add_argument('-i', '--input_directory', type=str,
                        help='directory to save mels to')
    parser.add_argument('-o', '--output_directory', type=str,
                        help='directory to read audio from')

    args = parser.parse_args()

    spec = importlib.util.spec_from_file_location(args.preprocessor,
                                                  os.path.join('preprocessors', f'{args.preprocessor}_preprocessor.py'))
    preprocessor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(preprocessor)

    hparameters = create_hparams()

    os.makedirs(args.output_directory, exist_ok=True)

    preprocessor.preprocess(args.input_directory, args.output_directory, hparameters)
