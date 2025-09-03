import argparse
import warnings
import shutil
import pyfaidx
from typing import Optional
import os
from toolbox.utils import get_instance_id, set_random_seed
from toolbox.logger import SimpleLogger
import json
import subprocess
import tempfile
from .auto_shift_detect import compute_shift
from histobpnet.data import get_data_path, DataFile

# adapted from https://github.com/kundajelab/chrombpnet/blob/master/chrombpnet/helpers/preprocessing/reads_to_bigwig.py

# Example usage:
# TODO

def get_parsers():
    parser = argparse.ArgumentParser(description= "Convert input BAM/fragment/tagAlign file to appropriately shifted unstranded Bigwig")

    parser.add_argument('--output_dir', type=str, required=True,
                        help="Output directory to store the results. Will create a subdirectory with instance_id.")
    parser.add_argument('--genome', type=str, required=True,
                        help="Reference genome fasta file to use")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-ibam', '--input-bam-file', type=str,
                       help="Input BAM file")
    group.add_argument('-ifrag', '--input-fragment-file', type=str,
                       help="Input fragment file")
    group.add_argument('-itag', '--input-tagalign-file', type=str,
                       help="Input tagAlign file")
    parser.add_argument('-c', '--chrom-sizes', type=str, required=True,
                        help="Chrom sizes file")
    parser.add_argument('-d', '--data-type', required=True, type=str, choices=['ATAC', 'DNASE', 'HistChIP'],
                        help="assay type")
    parser.add_argument('--bedsort', required=False, action='store_true',
                        help="use bedtools sort (default is unix sort)")
    parser.add_argument('--filter_chroms', required=False,  action='store_false',
                        help="Filter chromosomes not in reference genome")
    parser.add_argument('-ps', '--plus-shift', type=int, default=None,
                        help="Plus strand shift applied to reads (eg 2). Estimated if not specified")
    parser.add_argument('-ms', '--minus-shift', type=int, default=None,
                        help="Minus strand shift applied to reads (eg -2). Estimated if not specified")
    parser.add_argument('--ATAC-ref-path', type=str, default=None,
                        help="Path to ATAC reference motifs (histobpnet/data/ATAC.ref.motifs.txt used by default)")
    parser.add_argument('--DNASE-ref-path', type=str, default=None,
                        help="Path to DNASE reference motifs (histobpnet/data/DNASE.ref.motifs.txt used by default)")
    parser.add_argument('--num-samples', type=int, default=10_000,
                        help="Number of reads to sample from BAM/fragment/tagAlign file for shift estimation")

    return parser

def validate_args(args_d):
    assert (args_d["ibam"] is None) + (args_d["ifrag"] is None) + (args_d["itag"] is None) == 2, "Only one input file must be specified."

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

def is_gz_file(filepath):
    # https://stackoverflow.com/questions/3703276/how-to-tell-if-a-file-is-gzip-compressed
    with open(filepath, 'rb') as test_f:
        return test_f.read(2) == b'\x1f\x8b'
    
def bam_to_tagalign_stream(bam_path):
    """
    Input: BAM file (binary alignments).
    Command: bedtools bamtobed -i <bam>
    Output: BED6 format (chrom, start, end, name, score, strand) — one record per aligned read.
    Use case: First step when you only have BAM alignments and want BED intervals.
    """
    p = subprocess.Popen(["bedtools", "bamtobed", "-i", bam_path], stdout=subprocess.PIPE)
    return p

def fragment_to_tagalign_stream(fragment_file_path):
    """
    Input: Fragment file (BED-like: chrom, start, end, …).
    Command: cat/zcat <fragments> | awk …
    Logic: For each fragment, prints two lines: one with + strand and one with – strand.
    Example input (fragment):
    chr1   100   200
    Output (tagAlign pseudo-reads):
    chr1   100   200   1000   0   +
    chr1   100   200   1000   0   -
    Use case: Converts paired-end fragment representation into tagAlign (BED6 with dummy score/MAPQ),
    for pipelines that expect tagAlign rather than fragment files.
    """
    read_method = "zcat " if is_gz_file(fragment_file_path) else "cat "
    cmd = read_method + fragment_file_path + """ | awk -v OFS="\\t" '{print $1,$2,$3,1000,0,"+"; print $1,$2,$3,1000,0,"-"}'"""
    p = subprocess.Popen([cmd], stdout=subprocess.PIPE, shell=True)
    return p

def tagalign_stream(tagalign_file_path):
    """
    Input: A pre-existing tagAlign file (already in BED6 format).
    Command: just cat/zcat <tagAlign>
    Output: A stream of tagAlign lines (chrom, start, end, dummy, dummy, strand).
    Use case: No transformation, just a reader.
    """
    read_method = "zcat " if is_gz_file(tagalign_file_path) else "cat "
    p = subprocess.Popen([read_method, tagalign_file_path], stdout=subprocess.PIPE)
    return p

def stream_filtered_tagaligns(src_tagaligns_stream, genome_file, out_stream, do_warn: bool = True):
    """
    Given a tagalign subprocess stream and reference genome file, filters
    out any reads in chromosomes not included in the reference. Reads in the
    reference chromosomes are sent to the specified output stream.

    Returns:
        Boolean. Indicates whether any reads not in the reference fasta were 
        detected.
    """
    has_unknown_chroms = False
    with pyfaidx.Fasta(genome_file) as g:
        for line in iter(src_tagaligns_stream.stdout.readline, b''):
            tagalign_chrom = line.decode('utf-8').strip().split('\t')[0]
            if tagalign_chrom in g.keys():
                out_stream.write(line)
            else:
                has_unknown_chroms = True

    if has_unknown_chroms and do_warn:
        msg = "!!! WARNING: Input reads contain chromosomes not in the reference" \
            " genome fasta provided. Please ensure you are using the correct" \
            " reference genome. If you are confident you are using the correct reference" \
            " genome, you can safely ignore this message."
        warnings.warn(msg)

def generate_bigwig(
    output_dir: str,
    input_bam_file: Optional[str],
    input_fragment_file: Optional[str],
    input_tagalign_file: Optional[str],
    genome_fasta_file: str,
    bedsort: bool,
    filter_chroms: bool,
    chrom_sizes_file: str,
    plus_shift_delta: int,
    minus_shift_delta: int,
    logger,
):
    if input_bam_file is not None:
        p1 = bam_to_tagalign_stream(input_bam_file)
    elif input_fragment_file is not None:
        # valeh: am a bit confused how this works for frag files b/c in this case
        # we will have two entries per fragment and for one entry we will shift the
        # start by plus_shift_delta and for the other entry we will shift the end by
        # minus_shift_delta. Do we coalesce these two somehow later? Idk...
        # I confirmed with chatgpt too that this seems incorrect as is.
        warnings.warn("The AWK shift code is only correct on true read-level tagAligns, not" \
            "on fragment-to-tagAlign pseudo-reads. Using it on the latter will distort your coverage.")
        p1 = fragment_to_tagalign_stream(input_fragment_file)
    elif input_tagalign_file is not None:
        p1 = tagalign_stream(input_tagalign_file)

    tmp_dir = os.path.join(output_dir, "temp_sort")
    logger.add_to_log(f"Creating temporary directory at path: {tmp_dir}")
    os.makedirs(output_dir, exist_ok=False)

    sort_cmd = "bedtools sort -i stdin" if bedsort else f'LC_COLLATE="C" sort -T {tmp_dir} -k1,1 -k2,2n'
    # $6 is the strand field.
    # If the strand is +, it shifts the start coordinate by plus_shift_delta.
    # If the strand is -, it shifts the end coordinate by minus_shift_delta.
    # {0:+} and {1:+} are Python .format() placeholders; the + sign specifier makes the string have
    # +N or -N so AWK sees it as an arithmetic offset.
    cmd = """
        awk -v OFS="\\t" '{{if ($6=="+"){{print $1,$2{0:+},$3,$4,$5,$6}} else if ($6=="-") {{print $1,$2,$3{1:+},$4,$5,$6}}}}' |
        sort -T {3} -k1,1 |
        bedtools genomecov -bg -5 -i stdin -g {2} |
        {4}
    """.format(plus_shift_delta, minus_shift_delta, chrom_sizes_file, tmp_dir, sort_cmd)

    logger.add_to_log(f"Command to run:\n{cmd}")

    logger.add_to_log("Making BedGraph (Do not filter chromosomes not in reference fasta)")
    tmp_bedgraph = tempfile.NamedTemporaryFile(dir=output_dir)
    with open(tmp_bedgraph.name, 'w') as f:
        p2 = subprocess.Popen([cmd], stdin=subprocess.PIPE if filter_chroms else p1.stdout, stdout=f, shell=True)
        if filter_chroms:
            stream_filtered_tagaligns(p1, genome_fasta_file, p2.stdin)
        p1.stdout.close()
        p2.communicate()

    logger.add_to_log("Making Bigwig")
    subprocess.run(["bedGraphToBigWig", tmp_bedgraph.name, chrom_sizes_file, os.path.join(output_dir, "unstranded.bw")])
    tmp_bedgraph.close()

    logger.add_to_log(f"Removing temporary directory at path: {tmp_dir}")
    shutil.rmtree(tmp_dir)

def main(instance_id: str):
    args_d, output_dir, logger = setup(instance_id)

    plus_shift, minus_shift = args_d["plus_shift"], args_d["minus_shift"]

    if (plus_shift is None) or (minus_shift is None):
        if args_d["data_type"]=="ATAC":
            ref_motifs_file = args_d["ATAC_ref_path"] if args_d["ATAC_ref_path"] is not None else get_data_path(DataFile.atac_ref_motifs)
        elif args_d["data_type"]=="DNASE":
            ref_motifs_file = args_d["DNASE_ref_path"] if args_d["DNASE_ref_path"] is not None else get_data_path(DataFile.dnase_ref_motifs)
        
        logger.add_to_log("Estimating enzyme shift in input file...")
        plus_shift, minus_shift = compute_shift(
            args_d["input_bam_file"],
            args_d["input_fragment_file"],
            args_d["input_tagalign_file"],
            args_d["num_samples"],
            args_d["genome"],
            args_d["data_type"],
            ref_motifs_file,
        )
    
        # {:+} means: format the number with its sign always shown
        logger.add_to_log("Current estimated shift: {:+}/{:+}".format(plus_shift, minus_shift))
    else:
        logger.add_to_log("The specified shift is: {:+}/{:+}".format(plus_shift, minus_shift))

    # computing additional shifting to apply
    if args_d["data_type"]=="ATAC":
        plus_shift_delta, minus_shift_delta = 4-plus_shift, -4-minus_shift
    elif args_d["data_type"]=="DNASE":
        plus_shift_delta, minus_shift_delta = -plus_shift, 1-minus_shift

    generate_bigwig(
        output_dir,
        args_d["input_bam_file"],
        args_d["input_fragment_file"],
        args_d["input_tagalign_file"],
        args_d["genome"],
        args_d["bedsort"],
        args_d["filter_chroms"],
        args_d["chrom_sizes"],
        plus_shift_delta,
        minus_shift_delta,
        logger,
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
