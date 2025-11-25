import argparse
import json
import polars as pl
import os
from toolbox.utils import get_instance_id, set_random_seed
from toolbox.logger import SimpleLogger

# adapted from https://github.com/kundajelab/chrombpnet/blob/master/chrombpnet/helpers/make_chr_splits/splits.py

# Example usage:
# python make_splits.py \
# --output_prefix /large_storage/goodarzilab/valehvpa/projects/scCisTrans/for_chrombpnet_tuto/splits \
# --fold_name fold_0 \
# --chrom_sizes /large_storage/goodarzilab/valehvpa/projects/scCisTrans/for_chrombpnet_tuto/hg38.chrom.subset.sizes \
# --test_chroms chr1 chr3 chr6 \
# --valid_chroms chr8 chr20

def get_parsers():
    parser = argparse.ArgumentParser()

    parser.add_argument('--output_prefix', type=str, required=True,
                    help="Output directory to store the results. Will create a subdirectory with instance_id.")
    parser.add_argument('--fold_name', type=str, required=True,
                    help="Name of the fold to be used in the output file.")
    parser.add_argument('--chrom_sizes', type=str, required=True,
                    help="TSV file with chromosome sizes. All chromosomes from the first column of chrom sizes file are used")
    parser.add_argument('--test_chroms', nargs="*", type=str, required=True,
                    help="List of chromosomes to use for the test set")
    parser.add_argument('--valid_chroms', nargs="*", type=str, required=True,
                    help="List of chromosomes to use for the validation set")

    return parser

def validate_args(args_d, chroms):
    assert set(args_d["test_chroms"]) <= set(chroms), "Test chromosomes are not in the args.chrom_sizes file!"
    assert set(args_d["valid_chroms"]) <= set(chroms), "Valid chromosomes are not in the args.chrom_sizes file!"
    assert not (set(args_d["valid_chroms"]) & set(args_d["test_chroms"])), "Test and Valid chromosomes should not share any chromosomes!"

def setup(instance_id: str, skip_validate: bool = False):
    parser = get_parsers()
    args = parser.parse_args() 

    # convert Namespace to dict
    args_d = vars(args)

    if not skip_validate:
        validate_args(args_d)

    # set up instance_id and output directory
    output_dir = os.path.join(args_d["output_prefix"], instance_id)
    os.makedirs(output_dir, exist_ok=False)

    # set up logger
    script_name = os.path.basename(__file__).replace(".py", "")
    logger = SimpleLogger(os.path.join(output_dir, f"{script_name}.log"))

    # save configs to disk in output dir
    with open(os.path.join(output_dir, 'args.json'), 'w') as f:
        json.dump(args_d, f, indent=4)

    return args_d, output_dir, logger

def main(instance_id: str):
    args_d, output_dir, logger = setup(instance_id, skip_validate=True)

    chrom_sizes = pl.read_csv(args_d["chrom_sizes"], separator="\t", has_header=False)
    chroms = chrom_sizes[:,0].to_list()

    validate_args(args_d, chroms)

    splits = {'test': args_d["test_chroms"], 'valid': args_d["valid_chroms"]}
    splits['train'] = [chrom for chrom in chroms if chrom not in splits['test'] + splits['valid']]
    fp = os.path.join(output_dir, f"{args_d['fold_name']}.json")
    with open(fp, "w") as f:
        json.dump(splits, f, indent=4)
    logger.add_to_log(f"Saved splits to {fp}")

    logger.add_to_log("All done!")

if __name__ == '__main__':
    # get the instance id first so we can print it fast, then continue with the rest
    instance_id = get_instance_id()
    print(f"*** Using instance_id: {instance_id}")

    set_random_seed(seed=42)
    main(instance_id)
