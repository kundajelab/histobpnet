import warnings
import pyfaidx
import subprocess

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
