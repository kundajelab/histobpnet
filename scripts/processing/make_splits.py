import argparse
import json
import polars as pl

from histobpnet.utils.general_utils import set_random_seed

# adapted from https://github.com/kundajelab/chrombpnet/blob/master/chrombpnet/helpers/make_chr_splits/splits.py

def get_parsers():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_prefix', type=str, required=True,
                    help="Path prefix to store the fold information (appended with .json)")
    parser.add_argument('--chrom_sizes', type=str, required=True,
                    help="TSV file with chromosome sizes. All chromosomes from the first column of chrom sizes file are used")
    parser.add_argument('--test_chroms', nargs="*", type=str, required=True,
                    help="List of chromosomes to use for the test set")
    parser.add_argument('--valid_chroms', nargs="*", type=str, required=True,
                    help="List of chromosomes to use for the validation set")
    return parser

def validate_args(args, chroms):
    assert set(args.test_chroms) <= set(chroms), "Test chromosomes are not in the args.chrom_sizes file!"
    assert set(args.valid_chroms) <= set(chroms), "Valid chromosomes are not in the args.chrom_sizes file!"
    assert not (set(args.valid_chroms) & set(args.test_chroms)), "Test and Valid chromosomes should not share any chromosomes!"

def main(): 
    parser = get_parsers()
    args = parser.parse_args() 

    chrom_sizes = pl.read_csv(args.chrom_sizes, separator="\t", has_header=False)
    chroms = chrom_sizes[:,0].to_list()

    validate_args(args, chroms)

    splits = {'test': args.test_chroms, 'valid': args.valid_chroms}
    splits['train'] = [chrom for chrom in chroms if chrom not in splits['test'] + splits['valid']]
    with open(args.output_prefix + ".json", "w") as f:
        json.dump(splits, f, indent=4)
    print(f"Saved splits to {args.output_prefix}.json")

if __name__ == '__main__':
    set_random_seed(seed = 42)
    main()

# Example usage:
# python make_splits.py \
# --output_prefix /large_storage/goodarzilab/valehvpa/projects/scCisTrans/for_chrombpnet_tuto/splits/fold_0 \
# --chrom_sizes ./hg38.chrom.subset.sizes \
# --test_chroms chr1 chr3 chr6 \
# --valid_chroms chr8 chr20
