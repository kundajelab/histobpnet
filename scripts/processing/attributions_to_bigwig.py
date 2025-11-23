# COPIED/ADAPTED FROM https://github.com/kundajelab/chrombpnet/tree/master/chrombpnet/evaluation/make_bigwigs

import argparse
from toolbox.utils import get_instance_id, set_random_seed
from toolbox.logger import SimpleLogger
import os
import json
import time
from histobpnet.utils.data_utils import read_chrom_sizes, hdf5_to_bigwig

# Example usage:
# python /home/valehvpa/projects/scCisTrans/histobpnet/scripts/processing/attributions_to_bigwig.py \
#     -h5 /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_borzoi/histone/input_x_grad/gm12878_h3k27ac/instance-20251119_125710/scores_f3c0.h5 \
#     -r /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_borzoi/histone/input_x_grad/gm12878_h3k27ac/instance-20251119_125710/regions.bed \
#     --regions_format bed \
#     -c /large_storage/goodarzilab/valehvpa/refs/hg38/hg38.chrom.sizes \
#     -op /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_borzoi/histone/input_x_grad/to_shap \
#     --output_prefix_stats attributions_stats.txt \
#     --h5_read_tool h5py \
#     --hdf5_key grads

def get_parsers():
    parser = argparse.ArgumentParser(description="Convert importance scores in hdf5 format to bigwig. The output can be visualized using WashU Epigenome Browser as a dynseq track. Please read all parameter argument requirements. PROVIDE ABSOLUTE PATHS!")
    
    parser.add_argument("-h5", "--hdf5", type=str, required=True,
                        help="HDF5 file f such that f[hdf5_key] has (N x 4 x seqlen) shape with importance score * sequence so that each f[hdf5_key][i, :, j] has 3 zeros and 1 non-zero value")
    parser.add_argument("-r", "--regions", type=str, required=True,
                        help="BED file of length = N which matches f[hdf5_key].shape[0]. The ith region in the BED file corresponds to ith entry in importance matrix.")
    parser.add_argument("--regions_format", type=str, required=False, choices=['bed10', 'bed'], default='bed10',
                        help="Format of the regions bed file. If 'bed10' then the importance scores are assumed to be for [start+summit-(seqlen/2):start+summit+(seqlen/2)]. If 'bed' then the importance scores are assumed to be for [start:end] where start and end are 2nd and 3rd columns of the bed file respectively.")
    parser.add_argument("-c", "--chrom_sizes", type=str, required=True,
                        help="Chromosome sizes 2 column tab-separated file")
    parser.add_argument("-op", "--output_prefix", type=str, required=True,
                        help="Output prefix directory")
    parser.add_argument("-os", "--output_prefix_stats", type=str, required=False,
                        help="Output file with stats of low and high quantiles")
    parser.add_argument('--tqdm', required=False, action='store_true',
                        help="Use tqdm. If yes then you need to have it installed.")
    parser.add_argument("-d", "--debug_chr", type=str, default=None,
                        help="Run for one chromosome only (e.g. chr12) for debugging")
    parser.add_argument('--h5_read_tool', type=str, required=False, default='deepdish', choices=['deepdish', 'h5py'],
                        help="Tool to read hdf5 file. If deepdish then you need to have it installed. If h5py then you need to have it installed and the hdf5 file should be in a format that h5py can read (i.e. not compressed with gzip or other compression that h5py cannot read).")
    parser.add_argument('--hdf5_key', type=str, required=False, default='/projected_shap/seq',
                        help="Key in the hdf5 file to read the importance scores from. Default is '/projected_shap/seq'.")
    
    return parser

def validate_args(args_d):
    pass

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

    return args, output_dir, logger

def main(args):
    args, output_dir, logger = setup(instance_id)

    hdf5_to_bigwig(
        args.hdf5,
        args.regions,
        read_chrom_sizes(args.chrom_sizes),
        os.path.join(output_dir, "attributions"),
        output_prefix_stats = os.path.join(output_dir, args.output_prefix_stats),
        debug_chr = args.debug_chr,
        tqdm = args.tqdm,
        h5_read_tool = args.h5_read_tool,
        hdf5_key = args.hdf5_key,
        regions_format = args.regions_format,
    )

    return logger
    
if __name__=="__main__":
    # get the instance id first so we can print it fast, then continue with the rest
    instance_id = get_instance_id()
    print(f"*** Using instance_id: {instance_id}")

    set_random_seed(seed=42)

    t0 = time.time()
    logger = main(instance_id)
    tt = time.time() - t0
    logger.add_to_log(f"All done! Time taken: {tt//3600:.0f} hrs, {(tt%3600)//60:.0f} mins, {tt%60:.1f} secs")