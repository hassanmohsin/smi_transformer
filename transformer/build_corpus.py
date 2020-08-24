import argparse

import pandas as pd
from tqdm import tqdm

from .utils import split


def main():
    parser = argparse.ArgumentParser(description='Build a corpus file')
    parser.add_argument('--in_path',
                        '-i',
                        type=str,
                        required=True,
                        help='input file')
    parser.add_argument('--smiles_column',
                        '-s',
                        type=str,
                        required=True,
                        help="SMILES column")
    parser.add_argument('--sep',
                        type=str,
                        default=',',
                        help='Column separator for the csv file')
    parser.add_argument('--out_path',
                        '-o',
                        type=str,
                        required=True,
                        help='output file')
    args = parser.parse_args()

    # smiles = pd.read_csv(args.in_path, sep=args.sep)[args.smiles_column].values
    tqdm.pandas()
    df = pd.read_csv(args.in_path, sep=args.sep)
    print(f"INFO: {len(df)} samples in the dataset.")
    print(f"INFO: Processing the SMILES...")
    df['processed_smiles'] = df[args.smiles_column].progress_apply(
        lambda x: split(x))
    df.to_csv(args.out_path, index=None)
    print(f'INFO: Built a corpus file! Saved to {args.out_path}')


if __name__ == '__main__':
    main()
