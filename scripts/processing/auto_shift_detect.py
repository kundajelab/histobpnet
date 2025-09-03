import subprocess
import pandas as pd
import numpy as np
import itertools
import warnings
from toolbox.logger import SimpleLogger
import polars as pl
from typing import Optional
from modisco.visualization import viz_sequence
from histobpnet.utils.general_utils import get_pwms
from .reads_to_bigwig import (
    bam_to_tagalign_stream,
    fragment_to_tagalign_stream,
    tagalign_stream,
    stream_filtered_tagaligns,
)

# adapted from https://github.com/kundajelab/chrombpnet/blob/master/chrombpnet/helpers/preprocessing/auto_shift_detect.py

def sample_reads(
    input_bam_file: Optional[str],
    input_fragment_file: Optional[str],
    input_tagalign_file: Optional[str],
    num_samples: int,
    genome_fasta_path: str,
    logger,
):
    if input_bam_file is not None:
        p1 = bam_to_tagalign_stream(input_bam_file)
    elif input_fragment_file is not None:
        p1 = fragment_to_tagalign_stream(input_fragment_file)
    elif input_tagalign_file is not None:
        p1 = tagalign_stream(input_tagalign_file)

    logger.add_to_log(f"Sampling reads from input...")
    # num_samples is per strand, so multiply by 2
    p2 = subprocess.Popen(["shuf", "-n", str(2*num_samples)], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    stream_filtered_tagaligns(p1, genome_fasta_path, p2.stdin)
    output = p2.communicate()[0]

    logger.add_to_log(f"Saving as strand-specific dataframes....")
    # "if x" -> If the trailing line is empty (""), it’s dropped.
    rows = [x.split("\t") for x in output.decode("utf-8").split("\n") if x]
    reads = pl.DataFrame(
        rows,
        schema=["chr", "start", "end", "x1", "x2", "strand"]
    )

    # strand-specific subsets
    plus_reads = reads.filter(pl.col("strand") == "+").select(["chr", "start", "end"])
    minus_reads = reads.filter(pl.col("strand") == "-").select(["chr", "start", "end"])

    return plus_reads, minus_reads

def get_ref_pwms(ref_motifs_path):
    """
    Expected format of file: 
    Contains motifs one position per line, 4 columns per base tab-separated. First 
    line of each motif starts with ">" followed by name. "_" is used to name motifs 
    and name should end with "_plus" or "_minus".
    """
    pwms = {'+':{}, '-':{}}
    cur_orient = None
    cur_motif = None
    with open(ref_motifs_path) as f:
        for x in f:
            x = x.strip()
            if x.startswith(">"):
                # format current as numpy array before starting new
                if cur_motif is not None:
                    pwms[cur_orient][cur_motif] = np.array(pwms[cur_orient][cur_motif])

                if x.endswith("_plus"):
                    cur_orient = "+"
                elif x.endswith("_minus"):
                    cur_orient = "-"
                else:
                    raise ValueError("Invalid reference motif file")
                cur_motif = x[1:]
                pwms[cur_orient][cur_motif] = []
            else:
                pwms[cur_orient][cur_motif].append([float(y) for y in x.split('\t')])
    pwms[cur_orient][cur_motif] = np.array(pwms[cur_orient][cur_motif])

    return pwms['+'], pwms['-']

# from https://github.com/kundajelab/tfmodisco/blob/master/modisco/util.py#L492
def compute_per_position_ic(ppm, background, pseudocount):
    """Compute information content at each position of ppm.

    Arguments:
        ppm: should have dimensions of length x alphabet. Entries along the
            alphabet axis should sum to 1.
        background: the background base frequencies
        pseudocount: pseudocount to be added to the probabilities of the ppm
            to prevent overflow/underflow.

    Returns:
        total information content at each positon of the ppm.
    """
    assert len(ppm.shape)==2
    assert ppm.shape[1]==len(background), "Make sure the letter axis is the second axis"
    if (not np.allclose(np.sum(ppm, axis=1), 1.0, atol=1.0e-5)):
        print("WARNING: Probabilities don't sum to 1 in all the rows; this can"
              +" be caused by zero-padding. Will renormalize. PPM:\n"
              +str(ppm)
              +"\nProbability sums:\n"
              +str(np.sum(ppm, axis=1)))
        ppm = ppm/np.sum(ppm, axis=1)[:,None]

    alphabet_len = len(background)
    # add a pseudocount before logging for numerical stability, and re-normalize
    ppm_ps = (ppm + pseudocount)/(1 + pseudocount*alphabet_len)
    # note that the ic formula below is not quite computing the per-position KL divergence: sum_{b}(p*logp - p*logq)
    # instead it is computing sum_{b}(p*logp - q*logq) but they turn out to be the same if q is the uniform distribution
    # (valeh: Im not sure why it was done this way but Im not changing it for now, only throw if q is not uniform)
    uniform = np.ones(len(background)) / len(background)
    if not np.allclose(background, uniform):
        warnings.warn("Background distribution is not uniform, IC calculation might be incorrect")
    ic = np.sum(np.log2(ppm_ps)*ppm - (np.log2(background)*background)[None,:], axis=1)
    return ic

# from https://github.com/kundajelab/tfmodisco/blob/master/modisco/visualization/viz_sequence.py
def ic_scale(pwm):
    # renormalize just in case
    pwm = pwm/np.sum(pwm, axis=-1, keepdims=True)
    per_position_ic = compute_per_position_ic(ppm=pwm, background=[.25]*4, pseudocount=0.001)
    return pwm*(per_position_ic[:,None])

def convolve(to_scan, longer_seq):
    # Convolve to_scan matrix against longer_seq matrix
    vals = []
    for i in range(len(longer_seq) - len(to_scan) + 1):
        vals.append(np.sum(to_scan*longer_seq[i:i+len(to_scan)]))
    return vals

def compute_shift_ATAC(ref_plus_pwms, ref_minus_pwms, plus_pwm, minus_pwm):
    plus_shifts = set()
    minus_shifts = set()

    for x in ref_plus_pwms:
        # 14 is the value when comparing unshifted BAM pwm
        shift = 14 - np.argmax(convolve(ic_scale(ref_plus_pwms[x]), ic_scale(plus_pwm)))
        plus_shifts.add(shift)
    for x in ref_minus_pwms:
        shift = 5 - np.argmax(convolve(ic_scale(ref_minus_pwms[x]), ic_scale(minus_pwm)))
        minus_shifts.add(shift)

    if len(plus_shifts) != 1 or len(minus_shifts) != 1:
        raise ValueError("Input file shifts inconsistent. Please post an issue")
    
    plus_shift = list(plus_shifts)[0]
    minus_shift = list(minus_shifts)[0]

    if (plus_shift,minus_shift) not in [(0,0)]+ list(itertools.product([3,4,5],[-4,-5,-6])):
        raise ValueError("Input shift is non-standard ({:+}/{:+}). Please post an Issue.".format(plus_shift, minus_shift))

    return plus_shift, minus_shift

def compute_shift_DNASE(ref_plus_pwms, ref_minus_pwms, plus_pwm, minus_pwm):
    plus_shifts = set()
    minus_shifts = set()

    for x in ref_plus_pwms:
        # 10 is the value when comparing unshifted BAM pwm
        shift = 10 - np.argmax(convolve(ic_scale(ref_plus_pwms[x]), ic_scale(plus_pwm)))
        plus_shifts.add(shift)
    for x in ref_minus_pwms:
        shift = 10 - np.argmax(convolve(ic_scale(ref_minus_pwms[x]), ic_scale(minus_pwm)))
        minus_shifts.add(shift)

    if len(plus_shifts) != 1 or len(minus_shifts) != 1:
        raise ValueError("Input file shifts inconsistent. Please post an Issue")

    plus_shift = list(plus_shifts)[0]
    minus_shift = list(minus_shifts)[0]

    if (plus_shift,minus_shift) not in [(0,0), (0,1)]:
        raise ValueError("Input shift is non-standard ({:+}/{:+}). Please post an Issue.".format(plus_shift, minus_shift))

    return plus_shift, minus_shift 

def compute_shift(
    input_bam_file: Optional[str],
    input_fragment_file: Optional[str],
    input_tagalign_file: Optional[str],
    num_samples,
    genome_fasta_path: str,
    data_type: str,
    ref_motifs_file: str,
    logger,
):
    # only one of the 3 inputs should be non None
    assert (input_bam_file is None) + (input_fragment_file is None) + (input_tagalign_file is None) == 2, "Only one input file must be specified."

    sampled_plus_reads, sampled_minus_reads = sample_reads(
        input_bam_file,
        input_fragment_file,
        input_tagalign_file,
        num_samples,
        genome_fasta_path,
        logger
    )

    plus_pwm, minus_pwm = get_pwms(sampled_plus_reads, sampled_minus_reads, genome_fasta_path)
    ref_plus_pwms, ref_minus_pwms = get_ref_pwms(ref_motifs_file)

    if data_type=="ATAC":
        plus_shift, minus_shift = compute_shift_ATAC(ref_plus_pwms, ref_minus_pwms, plus_pwm, minus_pwm)
    elif data_type=="DNASE":
        plus_shift, minus_shift = compute_shift_DNASE(ref_plus_pwms, ref_minus_pwms, plus_pwm, minus_pwm)

    return plus_shift, minus_shift
