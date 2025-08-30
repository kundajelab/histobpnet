import argparse
import numpy as np
import polars as pl
import pyfaidx
from tqdm import tqdm
import os
from toolbox.utils import get_instance_id, set_random_seed
from toolbox.logger import SimpleLogger
import json
import random
import matplotlib.pyplot as plt

# adapted from https://github.com/kundajelab/chrombpnet/tree/master/chrombpnet/helpers/make_gc_matched_negatives

# Example usage:
# python make_negatives.py \
# --output_dir /large_storage/goodarzilab/valehvpa/projects/scCisTrans/for_chrombpnet_tuto/negatives \
# --peaks /large_storage/goodarzilab/valehvpa/projects/scCisTrans/for_chrombpnet_tuto/peaks_no_blacklist.bed \
# --chrom_sizes /large_storage/goodarzilab/valehvpa/refs/hg38/hg38.chrom.sizes \
# --genomewide_gc /large_storage/goodarzilab/valehvpa/refs/hg38/genomewide_gc_hg38_stride_1000_inputlen_2114.bed \
# --genome /large_storage/goodarzilab/valehvpa/refs/hg38/GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta \
# --blacklist_regions /large_storage/goodarzilab/valehvpa/refs/hg38/hg38.blacklist.bed.gz \
# --chr_fold_path /large_storage/goodarzilab/valehvpa/projects/scCisTrans/for_chrombpnet_tuto/splits/instance-20250727_104312/fold_0.json

def get_parsers():
    parser = argparse.ArgumentParser()

    parser.add_argument('--output_dir', type=str, required=True,
                        help="Output directory to store the results. Will create a subdirectory with instance_id.")
    parser.add_argument('--peaks', type=str, required=True,
                        help="Peaks bed file in narrowPeak format to calculate gc content from. The gc content will be calculated for the region centered on the peak summit and extending inputlen/2 on either side.")
    parser.add_argument('--chrom_sizes', type=str, required=True,
                        help="TSV file with chromosome sizes. All chromosomes from the first column of chrom sizes file are used")
    # see here https://github.com/kundajelab/chrombpnet/tree/master/chrombpnet/helpers/make_gc_matched_negatives/get_genomewide_gc_buckets
    # if you need to generate a new genomewide gc bed file
    parser.add_argument('--genomewide_gc', type=str, required=True,
                        help="Genomewide gc bed file with regions and their gc content in the 4th column rounded to 2 decimals")
    parser.add_argument('--genome', type=str, required=True,
                        help="Reference genome fasta file to use for gc content calculation")
    parser.add_argument('--neg_to_pos_ratio_train', type=int, default=2,
                        help="Ratio of negatives to positives to sample for training. For example, if set to 1, it will sample 1 negative for each positive. If set to 2, it will sample 2 negatives for each positive. Default is 2.")
    parser.add_argument('--blacklist_regions', type=str,
                        help="TSV file with 3 columns - chr, start, end. Backlisted regions not to be used for training.")
    parser.add_argument('--chr_fold_path', type=str, required=True,
                        help="Path to a JSON file with fold information - dictionary with test,valid and train keys and values with corresponding chromosomes. This is used to remap the chromosomes in the candidate negatives to the train, valid, and test sets.")
    parser.add_argument('--inputlen', type=int, default=2114,
                        help="Input sequence length. Default is 2114. Must be even number.")

    return parser

def validate_args(args_d: dict):
    assert(args_d["inputlen"] % 2 == 0)

def setup(instance_id: str):
    parser = get_parsers()
    args = parser.parse_args() 

    # convert Namespace to dict
    args_d = vars(args)

    validate_args(args_d)

    # set up instance_id and output directory
    output_dir = os.path.join(args_d["output_dir"], instance_id)
    os.makedirs(output_dir, exist_ok=False)

    # set up logger
    script_name = os.path.basename(__file__).replace(".py", "")
    logger = SimpleLogger(os.path.join(output_dir, f"{script_name}.log"))

    # save configs to disk in output dir
    with open(os.path.join(output_dir, 'args.json'), 'w') as f:
        json.dump(args_d, f, indent=4)

    return args_d, output_dir, logger

def remap_chrom(chrom, splits_dict):
    if chrom in splits_dict["train"]:
        chrom_mod = "chrom_train"
    elif chrom in splits_dict["valid"]:
        chrom_mod = "chrom_valid"
    elif chrom in splits_dict["test"]:
        chrom_mod = "chrom_test"
    else:
        chrom_mod = "ignore"
    return chrom_mod

def scale_gc(cur_gc):
    """
    Randomly increase/decrease the gc-fraction value by 0.01
    """
    if random.random()>0.5:
        cur_gc+=0.01
    else:
        cur_gc-=0.01
    cur_gc=round(cur_gc,2)
    if cur_gc<=0:
        cur_gc+=0.01
    if cur_gc>=1:
        cur_gc-=0.01
    assert cur_gc >=0
    assert cur_gc <=1
    return cur_gc 
    
def adjust_gc(chrom_mod, cur_gc, negatives, used_negatives):
    if chrom_mod not in used_negatives:
        used_negatives[chrom_mod] = {}

    if cur_gc not in used_negatives[chrom_mod]:
        used_negatives[chrom_mod][cur_gc]=[]

    # check if (1) the given gc fraction value is available
    # in the negative candidates and (2) the given gc fraction value has
    # candidates not already sampled. If either of the condition fails we
    # update the gc fraction value randomly and try again.
    while (cur_gc not in negatives[chrom_mod]) or (len(used_negatives[chrom_mod][cur_gc])>=len(negatives[chrom_mod][cur_gc])):
        cur_gc=scale_gc(cur_gc)
        if cur_gc not in used_negatives[chrom_mod]:
            used_negatives[chrom_mod][cur_gc]=[]
    return cur_gc, used_negatives 
    
def make_gc_dict(candidate_negatives, splits_dict, logger):
    """
    Imports the candidate negatives into a dictionary structure.
    The dictionary structure is as follows:
    {
        chrom_train: {
            gc_fraction1: [(chrom_train, start1, end1, chrom1), (chrom_train, start2, end2, chrom2), ...],
            ...
        },
        chrom_valid: {
            gc_fraction2: [(chrom_valid, start3, end3, chrom3), (chrom_valid, start4, end4, chrom4), ...],
            ...
        },
        chrom_test: {
            gc_fraction3: [(chrom_test, start5, end5, chrom5), (chrom_test, start6, end6, chrom6), ...],
            ...
        }
    }
    In each of the above dictionaries, the keys are gc content fractions rounded to 2 decimals,
    and the values are lists of tuples containing the (chrom,start,end) of a region with the corresponding 
    gc content fraction.
    """
    with open(candidate_negatives, 'r') as f:
        data = f.readlines()
    gc_dict={}
    ignored_chroms = []
    for line in tqdm(list(data)):
        # for example:
        # chr1    0       2114    0.0
        tokens = line.strip('\n').split('\t')
        chrom, gc, start, end = tokens[0], float(tokens[3]), tokens[1], tokens[2]
        chrom_mod = remap_chrom(chrom, splits_dict)
        if chrom_mod == "ignore":
            ignored_chroms.append(chrom)
            continue
        if chrom_mod not in gc_dict:
            gc_dict[chrom_mod]={}
        if gc not in gc_dict[chrom_mod]:
            gc_dict[chrom_mod][gc]=[(chrom_mod,start,end,chrom)]
        else:
            gc_dict[chrom_mod][gc].append((chrom_mod,start,end,chrom))

    logger.add_to_log("Following background chromosomes {}  were ignored since they are not present in the given fold".format(
        ",".join(list(set(ignored_chroms))))
    )
    return gc_dict

def get_gc_matched_negatives(
    candidate_negatives: str,
    foreground_gc_bed: str,
    output_prefix: str,
    chr_fold_path: str,
    neg_to_pos_ratio_train: int,
    logger,
):
    """
    generate a bed file of non-peak regions that are gc-matched with foreground

    Args:
        candidate_negatives: candidate negatives bed file with gc content in 4th column rounded to 2 decimals
        foreground_gc_bed: regions with their corresponding gc fractions for matching, 4th column has gc content value rounded to 2 decimals
        output_prefix: output prefix for the generated bed file of gc-matched negatives
        chr_fold_path: path to a JSON file with fold information - dictionary with test,valid and train keys and values with corresponding chromosomes
        neg_to_pos_ratio_train: ratio of negatives to positives to sample for training
    """
    peaks = pl.read_csv(foreground_gc_bed, has_header=False, separator='\t')
    logger.add_to_log("Number of foreground peaks: {}".format(len(peaks)))

    splits_dict = json.load(open(chr_fold_path))
    negatives = make_gc_dict(candidate_negatives, splits_dict, logger)

    used_negatives = {}
    neg_tuples = []
    foreground_gc_vals = []
    output_gc_vals = []
    ignored_chroms = []
    for row in tqdm(peaks.iter_rows(named=False), total=peaks.height):
        chrom, gc_value = row[0], row[3]

        chrom_mod = remap_chrom(chrom, splits_dict)
        if chrom_mod == "ignore":
            ignored_chroms.append(chrom)
            continue

        if chrom_mod=="chrom_train" or chrom_mod=="chrom_valid":
            neg_to_pos_ratio = neg_to_pos_ratio_train
        else:
            # we keep the ratio of positives to negatives in the test set same
            neg_to_pos_ratio = 1
        assert neg_to_pos_ratio >= 1, "neg_to_pos_ratio must be >= 1"
    
        # for every gc value in positive how many negatives to find
        for _ in range(neg_to_pos_ratio):
            cur_gc, used_negatives = adjust_gc(chrom_mod, gc_value, negatives, used_negatives)
            # num_candidates = len(negatives[chrom_mod][cur_gc])
            # rand_neg_index = random.randint(0, num_candidates - 1)
            # while rand_neg_index in used_negatives[chrom_mod][cur_gc]:
            #     cur_gc, used_negatives = adjust_gc(chrom_mod, cur_gc, negatives, used_negatives)
            #     num_candidates = len(negatives[chrom_mod][cur_gc])
            #     rand_neg_index=random.randint(0,num_candidates-1)
            num_candidates = len(negatives[chrom_mod][cur_gc])
            used_negatives_set = set(used_negatives[chrom_mod][cur_gc]) # set for faster lookup
            unused_negatives = [i for i in range(num_candidates) if i not in used_negatives_set]
            # there must be at least one unused negative, select one randomly
            assert unused_negatives
            rand_neg_index = random.choice(unused_negatives)

            used_negatives[chrom_mod][cur_gc].append(rand_neg_index)
            n = negatives[chrom_mod][cur_gc][rand_neg_index]
            n_start, n_end, n_chrom = n[1], n[2], n[3]
            neg_tuples.append([n_chrom, int(n_start), int(n_end), cur_gc]) 
            output_gc_vals.append(cur_gc)
            foreground_gc_vals.append(gc_value)       
  
    # 44877000
    logger.add_to_log("Following foreground chromosomes {} were ignored since they are not present in the given fold".format(",".join(list(set(ignored_chroms)))))     
    neg_tuples = pl.DataFrame(
        neg_tuples,
        schema={"chrom": pl.Utf8, "start": pl.Utf8, "end": pl.Utf8}
    )
    # neg_tuples.to_csv(output_prefix+".bed", sep='\t', index=False, header=False, quoting=csv.QUOTE_NONE)
    n_path = os.path.join(output_prefix, "_negatives.bed")
    neg_tuples.write_csv(
        n_path,
        separator="\t",
        include_header=False,
        quote_style="never",
    )

    # produce the below plot that helps visualize how far the true distribution of foreground gc content
    # is compared to the gc content of the negatives that were sampled
    plt.hist(
        [output_gc_vals, foreground_gc_vals],
        np.linspace(0, 1, 100),
        density=True,
        label=['negatives gc distribution', 'foreground gc distribution']
    )
    plt.xlabel("GC content")
    plt.ylabel("Density")
    plt.legend(loc='upper right')
    plt.savefig(os.path.join(output_prefix, "negatives_compared_with_foreground.png"))

    return n_path

def get_gc_content(genome, peaks_bed: str, chrom_sizes, inputlen: int, output_prefix: str, logger):
    """
    Get GC content from a peaks (aka foreground) bed file (narrowPeak format)

    Args:
        genome: reference genome fasta
        peaks_bed: peaks bed file in narrowPeak format - we will find gc content of these regions centered on the peak summit
        chrom_sizes: TSV file with chromosome name in first column and size in the second column
        inputlen: length of the region to use for gc content calculation (will be centered on the peak summit)
        output_prefix: output bed file prefix to store the gc content of peaks. suffix .bed will be appended by the code.
    
    Returns:
        None - writes a bed file with gc content as the 4th column
    """

    ref = pyfaidx.Fasta(genome)
    # this dict will look like {'chr1': 248956422, 'chr2': 242193529, ...}
    chrom_sizes_dict = {line.strip().split("\t")[0]:int(line.strip().split("\t")[1]) for line in open(chrom_sizes).readlines()}

    data = pl.read_csv(peaks_bed, has_header=False, separator='\t')
    logger.add_to_log("Number of peaks in the input narrowPeak file: " + str(data.shape[0]))

    filtered_peaks = 0
    with open(output_prefix + ".bed", 'w') as f:
        for row in tqdm(data.iter_rows(named=False), total=data.height):
            # narrowPeak format has chrom, start, end in the first 3 columns (0-based)
            chrom=row[0]
            start=row[1]
            end=row[2]

            # we will center the region on the summit (start + 10th column value)
            # and then extend inputlen/2 on either side to get the final region for gc content calculation
            summit=start+row[9]
            start=summit-inputlen//2
            end=summit+inputlen//2

            # if the region goes beyond chrom boundaries - we skip that region
            if start < 0:
                filtered_peaks += 1
                continue
            if end > chrom_sizes_dict[chrom]:
                filtered_peaks += 1
                continue

            # calculate gc content for the region centered at the summit of the peak
            seq=str(ref[chrom][start:end]).upper()
            gc=seq.count('G') + seq.count('C')
            gc_frac=round(gc/len(seq), 2)    

            # write the gc content in the 4th column of the output bed file
            f.write(chrom+'\t'+str(start)+'\t'+str(end)+'\t'+str(gc_frac)+"\n")

    logger.add_to_log("Number of regions filtered: " + str(filtered_peaks))
    pct_filtered = round(filtered_peaks*100.0/data.shape[0], 3)
    logger.add_to_log(f"Percentage of regions filtered: {str(pct_filtered)}%" )
    if pct_filtered > 25:
        logger.add_to_log("WARNING: Percentage of regions filtered is high (>25%), your genome is likely very small - consider using a smaller inputlen")

def main(instance_id: str):
    args_d, output_dir, logger = setup(instance_id)

    aux_dir = os.path.join(output_dir, "_auxiliary")
    os.makedirs(aux_dir, exist_ok=False)

    # get gc content in peaks
    get_gc_content(
        args_d["genome"],
        args_d["peaks"],
        args_d["chrom_sizes"],
        args_d["inputlen"],
        os.path.join(aux_dir, "foreground.gc"),
        logger
    )

    # prepare candidate negatives
    def get_excluded_regions(region_type: str):
        assert region_type in ["peaks", "blacklist"], "region_type must be either 'peaks' or 'blacklist'"
        # run bedtools slop to expand/contract each genomic interval in peaks by `flank_size`
        # while respecting chromosome boundaries defined in `chrom_sizes`, and save result to `output`
        os.system("bedtools slop -i {peaks} -g {chrom_sizes} -b {flank_size} > {output}".format(
            peaks=args_d["peaks"] if region_type == "peaks" else args_d["blacklist_regions"],
            chrom_sizes=args_d["chrom_sizes"],
            flank_size=args_d["inputlen"]//2,
            output=os.path.join(aux_dir, f"{region_type}_slop.bed")
        ))
        exclude_bed = pl.read_csv(os.path.join(aux_dir, f"{region_type}_slop.bed"), separator="\t", has_header=False, columns=[0,1,2])
        return exclude_bed

    exclude_bed = get_excluded_regions("peaks")
    if args_d["blacklist_regions"]:
        exclude_blacklist = get_excluded_regions("blacklist")
        exclude_bed = pl.concat([exclude_bed,exclude_blacklist])
    exclude_bed.write_csv(os.path.join(aux_dir, "exclude_unmerged.bed"), separator="\t", include_header=False)

    # merge the exclude_bed regions
    os.system("bedtools sort -i {inputb} | bedtools merge -i stdin > {output}".format(
        inputb=os.path.join(aux_dir, "exclude_unmerged.bed"),
        output=os.path.join(aux_dir, "exclude.bed"))
    )

    # the following command will create a bed file with candidate regions (and their gc content)
    # that are not overlapping with the exclude_bed regions
    # it will be used to get gc content and then filter out the ones that are not gc matched
    # to the foreground gc content
    bedtools_command = "bedtools intersect -v -a {genomewide_gc} -b {exclude_bed}  > {candidate_bed}".format(
        genomewide_gc=args_d["genomewide_gc"],
        exclude_bed=os.path.join(aux_dir, "exclude.bed"),
        candidate_bed=os.path.join(aux_dir, "candidates.bed")
    )
    os.system(bedtools_command)

    # sample negatives that are gc-matched with foreground peaks
    n_path = get_gc_matched_negatives(
        os.path.join(aux_dir, "candidates.bed"),
        foreground_gc_bed = os.path.join(aux_dir, "foreground.gc.bed"),
        output_prefix = aux_dir,
        chr_fold_path = args_d["chr_fold_path"],
        neg_to_pos_ratio_train = args_d["neg_to_pos_ratio_train"],
        logger = logger
    )

    # save the negatives in the final format
    negatives = pl.read_csv(n_path, separator="\t", has_header=False)
    negatives[3]="."
    negatives[4]="."
    negatives[5]="."
    negatives[6]="."
    negatives[7]="."
    negatives[8]="."
    negatives[9]=args_d["inputlen"]//2
    negatives.write_csv(
        os.path.join(output_dir, "negatives.bed"),
        separator="\t",
        include_header=False,
        include_index=False
    )

    logger.add_to_log("All done!")

if __name__ == '__main__':
    # # TODO
    # raise NotImplementedError("This script is not ready for execution.")

    # get the instance id first so we can print it fast, then continue with the rest
    instance_id = get_instance_id()
    print(f"*** Using instance_id: {instance_id}")

    set_random_seed(seed=42)
    main(instance_id)