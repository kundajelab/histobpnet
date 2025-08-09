import argparse
import pyBigWig
import pyfaidx
import subprocess
import tempfile
import os
import numpy as np
import chrombpnet.helpers.preprocessing.auto_shift_detect as auto_shift_detect
from chrombpnet.data import DefaultDataFile, get_default_data_path

# adapted from https://github.com/kundajelab/chrombpnet/blob/master/chrombpnet/helpers/preprocessing/reads_to_bigwig.py

# Example usage:
# TODO

def get_parsers():
    parser = argparse.ArgumentParser(description= "Convert input BAM/fragment/tagAlign file to appropriately shifted unstranded Bigwig")

    # parser.add_argument('--genome', type=str, required=True,
    #                 help="Output directory to store the results. Will create a subdirectory with instance_id.")
    # parser.add_argument('--fold_name', type=str, required=True,
    #                 help="Name of the fold to be used in the output file.")
    # parser.add_argument('--chrom_sizes', type=str, required=True,
    #                 help="TSV file with chromosome sizes. All chromosomes from the first column of chrom sizes file are used")
    # parser.add_argument('--test_chroms', nargs="*", type=str, required=True,
    #                 help="List of chromosomes to use for the test set")
    # parser.add_argument('--valid_chroms', nargs="*", type=str, required=True,
    #                 help="List of chromosomes to use for the validation set")

    # STOPPED HERE
    # parser.add_argument('-g','--genome', required=True, type=str, help="reference genome fasta file")
    # group = parser.add_mutually_exclusive_group(required=True)
    # group.add_argument('-ibam', '--input-bam-file', type=str, help="Input BAM file")
    # group.add_argument('-ifrag', '--input-fragment-file', type=str, help="Input fragment file")
    # group.add_argument('-itag', '--input-tagalign-file', type=str, help="Input tagAlign file")
    # parser.add_argument('-c', '--chrom-sizes', type=str, required=True, help="Chrom sizes file")
    # parser.add_argument('-op', '--output-prefix', type=str, required=True, help="Output prefix (path/to/prefix)")
    # parser.add_argument('-d', '--data-type', required=True, type=str, choices=['ATAC', 'DNASE'], help="assay type")
    # parser.add_argument('--bsort', required=False, default=False, action='store_true', help="use bedtools sort (default is unix sort)")
    # parser.add_argument('--no-st', required=False,  default=False, action='store_true', help="No streaming in preprocessing")
    # parser.add_argument('--tmpdir', required=False,  type=str, default=None, help="tmp dir path for unix sort command")
    # parser.add_argument('-ps', '--plus-shift', type=int, default=None, help="Plus strand shift applied to reads. Estimated if not specified")
    # parser.add_argument('-ms', '--minus-shift', type=int, default=None, help="Minus strand shift applied to reads. Estimated if not specified")
    # parser.add_argument('--ATAC-ref-path', type=str, default=None, help="Path to ATAC reference motifs (chrombpnet/data/ATAC.ref.motifs.txt used by default)")
    # parser.add_argument('--DNASE-ref-path', type=str, default=None, help="Path to DNASE reference motifs (chrombpnet/data/DNASE.ref.motifs.txt used by default)")
    # parser.add_argument('--num-samples', type=int, default=10000, help="Number of reads to sample from BAM/fragment/tagAlign file for shift estimation")

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

def generate_bigwig(input_bam_file, input_fragment_file, input_tagalign_file, output_prefix, genome_fasta_file, bsort, tmpdir, no_st, chrom_sizes_file, plus_shift_delta, minus_shift_delta):
    assert (input_bam_file is None) + (input_fragment_file is None) + (input_tagalign_file is None) == 2, "Only one input file!"

    if input_bam_file:
        p1 = auto_shift_detect.bam_to_tagalign_stream(input_bam_file)
    elif input_fragment_file:
        p1 = auto_shift_detect.fragment_to_tagalign_stream(input_fragment_file)
    elif input_tagalign_file:
        p1 = auto_shift_detect.tagalign_stream(input_tagalign_file)

    if tmpdir is None:
        if bsort:
            cmd = """awk -v OFS="\\t" '{{if ($6=="+"){{print $1,$2{0:+},$3,$4,$5,$6}} else if ($6=="-") {{print $1,$2,$3{1:+},$4,$5,$6}}}}' | sort -k1,1 | bedtools genomecov -bg -5 -i stdin -g {2} | bedtools sort -i stdin """.format(plus_shift_delta, minus_shift_delta, chrom_sizes_file)
        else:
            cmd = """awk -v OFS="\\t" '{{if ($6=="+"){{print $1,$2{0:+},$3,$4,$5,$6}} else if ($6=="-") {{print $1,$2,$3{1:+},$4,$5,$6}}}}' | sort -k1,1 | bedtools genomecov -bg -5 -i stdin -g {2} | LC_COLLATE="C" sort -k1,1 -k2,2n """.format(plus_shift_delta, minus_shift_delta, chrom_sizes_file)
    else:
        assert(os.path.isdir(tmpdir)) # tmp dir path does not exsist
        if bsort:
            cmd = """awk -v OFS="\\t" '{{if ($6=="+"){{print $1,$2{0:+},$3,$4,$5,$6}} else if ($6=="-") {{print $1,$2,$3{1:+},$4,$5,$6}}}}' | sort -T {3} -k1,1 | bedtools genomecov -bg -5 -i stdin -g {2} | bedtools sort -i stdin """.format(plus_shift_delta, minus_shift_delta, chrom_sizes_file, tmpdir)
        else:
            cmd = """awk -v OFS="\\t" '{{if ($6=="+"){{print $1,$2{0:+},$3,$4,$5,$6}} else if ($6=="-") {{print $1,$2,$3{1:+},$4,$5,$6}}}}' | sort -T {3} -k1,1 | bedtools genomecov -bg -5 -i stdin -g {2} | LC_COLLATE="C" sort -T {3} -k1,1 -k2,2n """.format(plus_shift_delta, minus_shift_delta, chrom_sizes_file, tmpdir)

    print(cmd)


    tmp_bedgraph = tempfile.NamedTemporaryFile()
    if no_st:
        print("Making BedGraph (Do not filter chromosomes not in reference fasta)")

        with open(tmp_bedgraph.name, 'w') as f:
            p2 = subprocess.Popen([cmd], stdin=p1.stdout, stdout=f, shell=True)
            p1.stdout.close()
            p2.communicate()
    else:
        print("Making BedGraph (Filter chromosomes not in reference fasta)")

        with open(tmp_bedgraph.name, 'w') as f:
            p2 = subprocess.Popen([cmd], stdin=subprocess.PIPE, stdout=f, shell=True)
            auto_shift_detect.stream_filtered_tagaligns(p1, genome_fasta_file, p2)
            p2.communicate()

    print("Making Bigwig")
    subprocess.run(["bedGraphToBigWig", tmp_bedgraph.name, chrom_sizes_file, output_prefix + "_unstranded.bw"])

    tmp_bedgraph.close()

def main(instance_id: str):
    args_d, output_dir, logger = setup(instance_id, skip_validate=True)

    chrom_sizes = pl.read_csv(args_d["chrom_sizes"], separator="\t", has_header=False)
    chroms = chrom_sizes[:,0].to_list()

    validate_args(args_d, chroms)

    plus_shift, minus_shift = args.plus_shift, args.minus_shift

    if (plus_shift is None) or (minus_shift is None):
        # TODO: validate inputs, check bedtools and ucsc tools
        if args.data_type=="ATAC":
            ref_motifs_file = args.ATAC_ref_path
            if ref_motifs_file is None:
                ref_motifs_file=get_default_data_path(DefaultDataFile.atac_ref_motifs)
        elif args.data_type=="DNASE":
            ref_motifs_file = args.DNASE_ref_path
            if ref_motifs_file is None:
                ref_motifs_file =  get_default_data_path(DefaultDataFile.dnase_ref_motifs)
    
        print("Estimating enzyme shift in input file")
        plus_shift, minus_shift = auto_shift_detect.compute_shift(args.input_bam_file,
                args.input_fragment_file,
                args.input_tagalign_file,
                args.num_samples,
                args.genome,
                args.data_type,
                ref_motifs_file)
    
        print("Current estimated shift: {:+}/{:+}".format(plus_shift, minus_shift))

    else:
        print("The specified shift is: {:+}/{:+}".format(plus_shift, minus_shift))

    # computing additional shifting to apply
    if args.data_type=="ATAC":
        plus_shift_delta, minus_shift_delta = 4-plus_shift, -4-minus_shift
    elif args.data_type=="DNASE":
        plus_shift_delta, minus_shift_delta = -plus_shift, 1-minus_shift

    generate_bigwig(
        args.input_bam_file,
        args.input_fragment_file,
        args.input_tagalign_file,
        args.output_prefix,
        args.genome,
        args.bsort,
        args.tmpdir,
        args.no_st,
        args.chrom_sizes,
        plus_shift_delta,
        minus_shift_delta
    )

    logger.add_to_log("All done!")

if __name__ == '__main__':
    # get the instance id first so we can print it fast, then continue with the rest
    instance_id = get_instance_id()
    print(f"*** Using instance_id: {instance_id}")

    set_random_seed(seed=42)
    main(instance_id)
