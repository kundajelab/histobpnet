import torch
import numpy as np
import torch.nn.functional as F
import pyfaidx
import warnings
import subprocess
from .data_utils import dna_to_one_hot

def is_histone(type_t: str):
    return type_t in ['histobpnet_v1', 'histobpnet_v2']

def add_peak_id(df, chr_key: str = "chr", start_key: str = "start", end_key: str = "end", inplace: bool = True):
    if inplace:
        df["peak_id"] = df[chr_key].astype(str) + ":" + df[start_key].astype(str) + "-" + df[end_key].astype(str)
    else:
        df_copy = df.copy()
        df_copy["peak_id"] = df_copy[chr_key].astype(str) + ":" + df_copy[start_key].astype(str) + "-" + df_copy[end_key].astype(str)
        return df_copy

def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()

def softmax(x, temp: int = 1):
    # TODO_later: what s hte point of de-meaning? (though it s harmless AFAIU)
    # https://chatgpt.com/c/691bc0cc-2650-832a-bfd3-7ba6dcad2d26
    x_demeaned = x - np.mean(x, axis=1, keepdims=True)
    e = np.exp(temp * x_demeaned)
    return e / np.sum(e, axis=1, keepdims=True)

def pearson_corr(x: torch.Tensor, y: torch.Tensor, dim: int = -1, eps = 1e-8) -> torch.Tensor:
    """
    Compute the Pearson correlation coefficient along a given dimension for multi-dimensional tensors.

    Args:
        x (torch.Tensor): Input tensor of shape (..., N).
        y (torch.Tensor): Input tensor of shape (..., N).
        dim (int): The dimension along which to compute the Pearson correlation. Default is -1 (last dimension).
        eps: A small constant to prevent division by zero. Default is 1e-8.

    Returns:
        torch.Tensor: Pearson correlation coefficients along the specified dimension.
    """
    # Ensure x and y have the same shape
    assert x.shape == y.shape, "Input tensors must have the same shape"

    # Step 1: Center the data (subtract the mean along the given dimension)
    x_centered = x - torch.mean(x, dim=dim, keepdim=True)
    y_centered = y - torch.mean(y, dim=dim, keepdim=True)

    # Step 2: Compute covariance (sum of element-wise products of centered tensors)
    cov = torch.sum(x_centered * y_centered, dim=dim)

    # Step 3: Compute standard deviations for each tensor along the specified dimension
    std_x = torch.sqrt(torch.sum(x_centered ** 2, dim=dim))
    std_y = torch.sqrt(torch.sum(y_centered ** 2, dim=dim))
    
    # Step 4: Compute Pearson correlation (with numerical stability)
    corr = cov / (std_x * std_y + eps)

    return corr

def multinomial_nll(logits, true_counts):
    """Compute the multinomial negative log-likelihood in PyTorch.
    
    Args:
      true_counts: Tensor of observed counts (batch_size, num_classes) (integer counts)
      logits: Tensor of predicted logits (batch_size, num_classes)
    
    Returns:
      Mean negative log-likelihood across the batch.
    """
    # Ensure true_counts is an integer tensor
    true_counts = true_counts.to(torch.float)  # Keep as float to prevent conversion issues
    
    # Compute total counts per example (should already be integer-like)
    counts_per_example = true_counts.sum(dim=-1, keepdim=True)
    
    # Convert logits to log probabilities (Softmax + Log)
    log_probs = F.log_softmax(logits, dim=-1)
    
    # Compute log-probability of the observed counts
    log_likelihood = (true_counts * log_probs).sum(dim=-1)
    
    # Compute multinomial coefficient (log factorial term)
    log_factorial_counts = torch.lgamma(counts_per_example + 1) - torch.lgamma(true_counts + 1).sum(dim=-1)

    # Compute final NLL
    nll = -(log_factorial_counts + log_likelihood).mean()

    # TODO cf chatgpt for potential bug
    return nll

def get_pwms(plus_reads, minus_reads, genome_file):
    """
    This function grabs 40 bp sequences around the cut sites of plus and minus reads,
    encodes them, and computes strand‐specific sequence logos (PWMs) by averaging base
    frequencies.
    """
    plus_seqs = []
    minus_seqs = []
    with pyfaidx.Fasta(genome_file) as g:
        # for each read, extract a 40 bp window centered at the 5' start coordinate (which is the start coordinate
        # for plus strand reads, and the end coordinate for minus strand reads)
        for x in plus_reads.iter_rows(named=True):
            cur = str(g[x['chr']][int(x['start'])-20:int(x['start'])+20])
            # if the read is near a chromosome edge or a non-canonical chr ( e.g. chrEBV),
            # the slice may be shorter — those are skipped.
            if len(cur)==40:
                plus_seqs.append(cur)
        for x in minus_reads.iter_rows(named=True):
            cur = str(g[x['chr']][int(x['end'])-20:int(x['end'])+20])
            if len(cur)==40:
                minus_seqs.append(cur)
    
    # dna_to_one_hot converts each sequence into a one‐hot encoded array: shape = (num_sequences, L, 4)
    # where L=40 and 4 corresponds to A/C/G/T. Taking the .mean(0) collapses across all sequences, producing
    # the average base frequency at each position → a raw PWM.
    plus_pwm = dna_to_one_hot(plus_seqs).mean(0)
    # this is not strictly necessary since np.sum should be 1, but it might not be I think due to floating-point drift
    plus_pwm = plus_pwm/np.sum(plus_pwm, axis=-1, keepdims=True)
    minus_pwm = dna_to_one_hot(minus_seqs).mean(0)
    minus_pwm = minus_pwm/np.sum(minus_pwm, axis=-1, keepdims=True)

    return plus_pwm, minus_pwm

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

def strand_specific_start_site(df):
    df = df.copy()
    if set(df["Strand"]) != set(["+", "-"]):
        raise ValueError("Not all features are strand specific!")

    pos_strand = df.query("Strand == '+'").index
    neg_strand = df.query("Strand == '-'").index
    df.loc[pos_strand, "End"] = df.loc[pos_strand, "Start"] + 1
    df.loc[neg_strand, "Start"] = df.loc[neg_strand, "End"] - 1
    return df

# valeh: I think these are needed to do interpretability with DeepSHAP
class _Exp(torch.nn.Module):
    def __init__(self):
        super(_Exp, self).__init__()

    def forward(self, X):
        return torch.exp(X)

class _Log(torch.nn.Module):
    def __init__(self):
        super(_Log, self).__init__()

    def forward(self, X):
        return torch.log(X)
