import numpy as np
import pandas as pd
import pyBigWig
import pyfaidx
import deepdish
import os
from toolbox.logger import cprint

def read_chrom_sizes(fname):
    with open(fname) as f:
        gs = [x.strip().split('\t') for x in f]
    gs = {x[0]: int(x[1]) for x in gs if len(x)==2}

    return gs

# format_region modifies the start and end columns of a DataFrame
# to center the region around the summit and set the width to a specified value.
# It also sets the summit to be half of the width.
# A better name for this function could be `center_region_around_summit`.
def format_region(df, width=500):
    df.loc[:, 'start'] = df.loc[:, 'start'].astype(np.int64) + df.loc[:, 'summit'] - width // 2
    df.loc[:, 'end'] = df.loc[:, 'start'] + width 
    df.loc[:, 'summit'] = width // 2
    return df

def expand_3col_to_10col(df):
    if df.shape[1] != 3:
        df = df.iloc[:, :3].copy()
    for i in ['name', 'score', 'strand', 'signalValue', 'pValue', 'qValue']: #range(4, 10):
        df[f'{i}'] = '.'
    df.iloc[:, 1] = df.iloc[:, 1].astype(int)
    df.iloc[:, 2] = df.iloc[:, 2].astype(int)
    df['summit'] = (df.iloc[:, 2] - df.iloc[:, 1]) // 2
    df.columns = ['chr', 'start', 'end', 'name', 'score', 'strand', 'signalValue', 'pValue', 'qValue', 'summit']
    df['start'] = df['start'].astype(int)
    df['end'] = df['end'].astype(int)
    df['summit'] = df['summit'].astype(int)
    return df

def load_region_df(regions_bed, chrom_sizes=None, in_window=2114, shift=0, is_peak: bool=True, logger=None, width=500):
    """
    Load the DataFrame and, optionally, filter regions in it that exceed defined chromosome sizes.
    """
    if isinstance(regions_bed, pd.DataFrame):
        df = regions_bed
    else:
        if not os.path.exists(regions_bed) and os.path.exists(regions_bed+'.gz'):
            regions_bed = regions_bed+'.gz'
        df = pd.read_csv(regions_bed, sep='\t', header=None)

    if isinstance(chrom_sizes, str):
        chrom_sizes = read_chrom_sizes(chrom_sizes)

    if df.shape[1] < 10:
        df = expand_3col_to_10col(df)
    df['is_peak'] = is_peak

    if chrom_sizes is not None:
        # assume column 0 is chr, column 9 is the summit, column 1 is start, column 2 is end
        flank_length = in_window // 2 + shift
        chrom_lengths = df.iloc[:, 0].map(lambda chrom: int(chrom_sizes.get(chrom, float('inf'))))
        assert (df.shape[1] >= 10), "DataFrame should have at least 10 columns after expanding to 10 columns."
        center = (df.iloc[:, 9] + df.iloc[:, 1])
        filtered_df = df[
            (center - flank_length > 0) &
            (center + flank_length <= chrom_lengths)
        ]
    else:
        cprint("Warning: No chromosome sizes provided, skipping filtering by chromosome length.", logger=logger)
        filtered_df = df
    filtered_df.columns = ['chr', 'start', 'end', 'name', 'score', 'strand', 'signalValue', 'pValue', 'qValue', 'summit', 'is_peak']

    cprint(f"Formatted {filtered_df.shape[0]} regions from {regions_bed} using width {width}.", logger=logger)
    filtered_df = format_region(filtered_df, width=width)

    # Reset index to avoid index errors
    return filtered_df.reset_index(drop=True)

def subsample_nonpeak_data(nonpeak_seqs, nonpeak_cts, nonpeak_coords, peak_data_size, negative_sampling_ratio):
    # Randomly samples a portion of the non-peak data to use in training
    num_nonpeak_samples = int(negative_sampling_ratio * peak_data_size)
    nonpeak_indices_to_keep = np.random.choice(len(nonpeak_seqs), size=min(num_nonpeak_samples, len(nonpeak_seqs)), replace=False)
    nonpeak_seqs = nonpeak_seqs[nonpeak_indices_to_keep]
    nonpeak_cts = nonpeak_cts[nonpeak_indices_to_keep]
    nonpeak_coords = nonpeak_coords[nonpeak_indices_to_keep]
    return nonpeak_seqs, nonpeak_cts, nonpeak_coords

def concat_peaks_and_subsampled_negatives(peaks, negatives=None, negative_sampling_ratio=0.1):
    if negatives is None:
        peaks, negatives = split_peak_and_nonpeak(peaks)
        # print(peaks.shape, negatives.shape)

    if len(negatives) > len(peaks) * negative_sampling_ratio and negative_sampling_ratio > 0:
        # TODO: do we need to pass random state here?
        negatives = negatives.sample(n=int(negative_sampling_ratio * len(peaks)), replace=False)
        
    data = pd.concat([peaks, negatives], ignore_index=True)
    return data

def split_peak_and_nonpeak(data):
    data['is_peak'] = data['is_peak'].astype(int).astype(bool)
    non_peaks = data[~data['is_peak']].copy()
    peaks = data[data['is_peak']].copy()
    return peaks, non_peaks

def get_cts(peaks_df, bw, width):
    """
    Fetches values from a bigwig bw, given a df with minimally
    chr, start and summit columns. Summit is relative to start.
    Retrieves values of specified width centered at summit.

    "cts" = per base counts across a region

    return shape: (len(peaks_df), width)
    """
    vals = []
    for _, r in peaks_df.iterrows():
        vals.append(np.nan_to_num(bw.values(r['chr'], 
                                            r['start'] + r['summit'] - width//2,
                                            r['start'] + r['summit'] + width//2)))
        
    return np.array(vals)

# valeh: REVIEWED ^^^^^^

def dna_to_one_hot(seqs):
    """
    Converts a list of DNA ("ACGT") sequences to one-hot encodings, where the
    position of 1s is ordered alphabetically by "ACGT". `seqs` must be a list
    of N strings, where every string is the same length L. Returns an N x L x 4
    NumPy array of one-hot encodings, in the same order as the input sequences.
    All bases will be converted to upper-case prior to performing the encoding.
    Any bases that are not "ACGT" will be given an encoding of all 0s.
    """
    seq_len = len(seqs[0])
    assert np.all(np.array([len(s) for s in seqs]) == seq_len)

    # Join all sequences together into one long string, all uppercase
    seq_concat = "".join(seqs).upper() + "ACGT"
    # Add one example of each base, so np.unique doesn't miss indices later

    one_hot_map = np.identity(5)[:, :-1].astype(np.int8)

    # Convert string into array of ASCII character codes;
    base_vals = np.frombuffer(bytearray(seq_concat, "utf8"), dtype=np.int8)

    # Anything that's not an A, C, G, or T gets assigned a higher code
    base_vals[~np.isin(base_vals, np.array([65, 67, 71, 84]))] = 85

    # Convert the codes into indices in [0, 4], in ascending order by code
    _, base_inds = np.unique(base_vals, return_inverse=True)

    # Get the one-hot encoding for those indices, and reshape back to separate
    return one_hot_map[base_inds[:-4]].reshape((len(seqs), seq_len, 4))

def one_hot_to_dna(one_hot):
    """
    Converts a one-hot encoding into a list of DNA ("ACGT") sequences, where the
    position of 1s is ordered alphabetically by "ACGT". `one_hot` must be an
    N x L x 4 array of one-hot encodings. Returns a lits of N "ACGT" strings,
    each of length L, in the same order as the input array. The returned
    sequences will only consist of letters "A", "C", "G", "T", or "N" (all
    upper-case). Any encodings that are all 0s will be translated to "N".
    """
    bases = np.array(["A", "C", "G", "T", "N"])
    # Create N x L array of all 5s
    one_hot_inds = np.tile(one_hot.shape[2], one_hot.shape[:2])

    # Get indices of where the 1s are
    batch_inds, seq_inds, base_inds = np.where(one_hot)

    # In each of the locations in the N x L array, fill in the location of the 1
    one_hot_inds[batch_inds, seq_inds] = base_inds

    # Fetch the corresponding base for each position using indexing
    seq_array = bases[one_hot_inds]
    return ["".join(seq) for seq in seq_array]

# https://stackoverflow.com/questions/46091111/python-slice-array-at-different-position-on-every-row
def take_per_row(A, indx, num_elem):
    """
    Matrix A, indx is a vector for each row which specifies 
    slice beginning for that row. Each has width num_elem.
    """

    all_indx = indx[:,None] + np.arange(num_elem)
    return A[np.arange(all_indx.shape[0])[:,None], all_indx]

def random_crop(seqs, labels, seq_crop_width, label_crop_width, coords):
    """
    Takes sequences and corresponding counts labels. They should have the same
    # of examples. The widths would correspond to inputlen and outputlen respectively,
    and any additional flanking width for jittering which should be the same
    for seqs and labels. Each example is cropped starting at a random offset. 

    seq_crop_width - label_crop_width should be equal to seqs width - labels width,
    essentially implying they should have the same flanking width.
    """
    assert(seqs.shape[1] >= seq_crop_width)
    assert(labels.shape[1] >= label_crop_width)
    assert(seqs.shape[1] - seq_crop_width == labels.shape[1] - label_crop_width)

    max_start = seqs.shape[1] - seq_crop_width # This should be the same for both input and output
    starts = np.random.choice(range(max_start+1), size=seqs.shape[0], replace=True)
    new_coords = coords.copy()
    new_coords[:,1] = new_coords[:,1].astype(int) - (seqs.shape[1]//2) + starts

    return take_per_row(seqs, starts, seq_crop_width), take_per_row(labels, starts, label_crop_width), new_coords

def random_rev_comp(seqs, labels, coords, frac=0.5):
    """
    Data augmentation: applies reverse complement randomly to a fraction of 
    sequences and labels.

    Assumes seqs are arranged in ACGT. Then ::-1 gives TGCA which is revcomp.

    NOTE: Performs in-place modification.
    """
    pos_to_rc = np.random.choice(range(seqs.shape[0]), size=int(seqs.shape[0]*frac), replace=False)
    seqs[pos_to_rc] = seqs[pos_to_rc, ::-1, ::-1]
    labels[pos_to_rc] = labels[pos_to_rc, ::-1]
    coords[pos_to_rc,2] =  "r"

    return seqs, labels, coords

def crop_revcomp_augment(seqs, labels, coords, add_revcomp, rc_frac=0.5, shuffle=False):
    """
    seqs: B x IL x 4
    labels: B x OL

    Applies random crop to seqs and labels and reverse complements rc_frac. 
    """
    assert(seqs.shape[0]==labels.shape[0])

    # this does not modify seqs and labels
    mod_seqs, mod_labels, mod_coords = seqs, labels, coords

    # this modifies mod_seqs, mod_labels in-place
    if add_revcomp:
        mod_seqs, mod_labels, mod_coords = random_rev_comp(mod_seqs, mod_labels, mod_coords, frac=rc_frac)

    if shuffle:
        perm = np.random.permutation(mod_seqs.shape[0])
        mod_seqs = mod_seqs[perm]
        mod_labels = mod_labels[perm]
        mod_coords = mod_coords[perm]

    return mod_seqs, mod_labels, mod_coords

def get_seq(peaks_df, genome, width):
    """
    Same as get_cts, but fetches sequence from a given genome.
    """
    vals = []
    for i, r in peaks_df.iterrows():
        sequence = str(genome[r['chr']][(r['start']+r['summit'] - width//2):(r['start'] + r['summit'] + width//2)])
        vals.append(sequence)

    return dna_to_one_hot(vals)

def get_coords(peaks_df, peaks_bool):
    """
    Fetch the co-ordinates of the regions in bed file
    returns a list of tuples with (chrom, summit)
    """
    vals = []
    for i, r in peaks_df.iterrows():
        vals.append([r['chr'], r['start']+r['summit'], "f", peaks_bool])

    return np.array(vals)

def get_seq_cts_coords(peaks_df, genome, bw, input_width, output_width, peaks_bool):
    seq = get_seq(peaks_df, genome, input_width)
    cts = get_cts(peaks_df, bw, output_width)
    coords = get_coords(peaks_df, peaks_bool)
    return seq, cts, coords

def load_data(bed_regions, nonpeak_regions, genome_fasta, cts_bw_file, inputlen, outputlen, max_jitter):
    """
    Load sequences and corresponding base resolution counts for training, 
    validation regions in peaks and nonpeaks (2 x 2 x 2 = 8 matrices).

    For training peaks/nonpeaks, values for inputlen + 2*max_jitter and outputlen + 2*max_jitter 
    are returned centered at peak summit. This allows for jittering examples by randomly
    cropping. Data of width inputlen/outputlen is returned for validation
    data.
    """
    cts_bw = pyBigWig.open(cts_bw_file)
    genome = pyfaidx.Fasta(genome_fasta)

    train_peaks_seqs=None
    train_peaks_cts=None
    train_peaks_coords=None
    train_nonpeaks_seqs=None
    train_nonpeaks_cts=None
    train_nonpeaks_coords=None

    if bed_regions is not None:
        if not set(['chr', 'start', 'summit']).issubset(bed_regions.columns):
            bed_regions = expand_3col_to_10col(bed_regions)
        train_peaks_seqs, train_peaks_cts, train_peaks_coords = get_seq_cts_coords(bed_regions,
                                                                                   genome,
                                                                                   cts_bw,
                                                                                   inputlen+2*max_jitter,
                                                                                   outputlen+2*max_jitter,
                                                                                   peaks_bool=1)
    
    if nonpeak_regions is not None:
        if not set(['chr', 'start', 'summit']).issubset(nonpeak_regions.columns):
            nonpeak_regions = expand_3col_to_10col(nonpeak_regions)
        train_nonpeaks_seqs, train_nonpeaks_cts, train_nonpeaks_coords = get_seq_cts_coords(nonpeak_regions,
                                                                                            genome,
                                                                                            cts_bw,
                                                                                            inputlen,
                                                                                            outputlen,
                                                                                            peaks_bool=0)

    cts_bw.close()
    genome.close()

    return (train_peaks_seqs, train_peaks_cts, train_peaks_coords,
            train_nonpeaks_seqs, train_nonpeaks_cts, train_nonpeaks_coords)

def write_bigwig(data, regions, gs, bw_out, debug_chr=None, use_tqdm=False, outstats_file=None):
    # regions may overlap but as we go in sorted order, at a given position,
    # we will pick the value from the interval whose summit is closest to 
    # current position
    chr_to_idx = {}
    for i,x in enumerate(gs):
        chr_to_idx[x[0]] = i

    bw = pyBigWig.open(bw_out, 'w')
    bw.addHeader(gs)
    
    # regions may not be sorted, so get their sorted order
    order_of_regs = sorted(range(len(regions)), key=lambda x:(chr_to_idx[regions[x][0]], regions[x][1]))

    all_entries = []
    cur_chr = ""
    cur_end = 0

    iterator = range(len(order_of_regs))
    if use_tqdm:
        from tqdm import tqdm
        iterator = tqdm(iterator)

    for itr in iterator:
        # subset to chromosome (debugging)
        if debug_chr and regions[i][0]!=debug_chr:
            continue

        i = order_of_regs[itr]
        i_chr, i_start, i_end, i_mid = regions[i]
    
        if i_chr != cur_chr: 
            cur_chr = i_chr
            cur_end = 0
    
        # bring current end to at least start of current region
        if cur_end < i_start:
            cur_end = i_start
    
        assert(regions[i][2]>=cur_end)
    
        # figure out where to stop for this region, get next region
        # which may partially overlap with this one
        next_end = i_end
    
        if itr+1 != len(order_of_regs):
            n = order_of_regs[itr+1]
            next_chr, next_start, _, next_mid = regions[n]
       
            if next_chr == i_chr and next_start < i_end:
                # if next region overlaps with this, end between their midpoints
                next_end = (i_mid+next_mid)//2
    
        vals = data[i][cur_end - i_start:next_end - i_start]

        bw.addEntries([i_chr]*(next_end-cur_end), 
                       list(range(cur_end,next_end)), 
                       ends = list(range(cur_end+1, next_end+1)), 
                       values=[float(x) for x in vals])
    
        all_entries.append(vals)
        
        cur_end = next_end

    bw.close()

    all_entries = np.hstack(all_entries)
    if outstats_file != None:
        with open(outstats_file, 'w') as f:
            f.write("Min\t{:.6f}\n".format(np.min(all_entries)))
            f.write(".1%\t{:.6f}\n".format(np.quantile(all_entries, 0.001)))
            f.write("1%\t{:.6f}\n".format(np.quantile(all_entries, 0.01)))
            f.write("50%\t{:.6f}\n".format(np.quantile(all_entries, 0.5)))
            f.write("99%\t{:.6f}\n".format(np.quantile(all_entries, 0.99)))
            f.write("99.9%\t{:.6f}\n".format(np.quantile(all_entries, 0.999)))
            f.write("99.95%\t{:.6f}\n".format(np.quantile(all_entries, 0.9995)))
            f.write("99.99%\t{:.6f}\n".format(np.quantile(all_entries, 0.9999)))
            f.write("Max\t{:.6f}\n".format(np.max(all_entries)))

def hdf5_to_bigwig(hdf5, regions, chrom_sizes, output_prefix, output_prefix_stats=None, debug_chr=None, tqdm=False):
    d = deepdish.io.load(hdf5, '/projected_shap/seq')

    SEQLEN = d.shape[2]
    assert(SEQLEN%2==0)

    # gs = bigwig_helper.read_chrom_sizes(chrom_sizes)
    # gs = chrom_sizes
    regions = bigwig_helper.get_regions(regions, SEQLEN)
    chr_list = set([region[0] for region in regions])
    chrom_sizes = [(x, v) for x, v in chrom_sizes.items() if x in chr_list]
    
    assert(d.shape[0] == len(regions))

    bigwig_helper.write_bigwig(d.sum(1), 
                        regions, 
                        chrom_sizes, 
                        output_prefix+".bw", 
                        outstats_file=output_prefix_stats, 
                        debug_chr=debug_chr, 
                        use_tqdm=tqdm)

def html_to_pdf(input_html, output_pdf):
    from weasyprint import HTML, CSS
    css = CSS(string='''
        @page {
            size: 1800mm 1300mm;
            margin: 0in 0in 0in 0in;
    }
    ''')
    HTML(input_html).write_pdf(output_pdf, stylesheets=[css])