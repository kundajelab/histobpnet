import gc
import numpy as np
import pandas as pd
import pyBigWig
import pyfaidx
import os
from toolbox.logger import cprint
from toolbox.one_hot import dna_to_one_hot
from histobpnet.utils.general_utils import add_peak_id

def html_to_pdf(input_html, output_pdf):
    from weasyprint import HTML, CSS
    css = CSS(string='''
        @page {
            size: 1800mm 1300mm;
            margin: 0in 0in 0in 0in;
        }
    ''')
    HTML(input_html).write_pdf(output_pdf, stylesheets=[css])

def read_chrom_sizes(fname):
    with open(fname) as f:
        gs = [x.strip().split('\t') for x in f]
    gs = {x[0]: int(x[1]) for x in gs if len(x)==2}

    return gs

# center_region_around_summit modifies the start and end columns of a DataFrame
# to center the region around the summit and set the width to a specified value.
# It also sets the summit to be half of the width.
def center_region_around_summit(df, width):
    df.loc[:, 'start'] = df.loc[:, 'start'].astype(np.int64) + df.loc[:, 'summit'] - width // 2
    df.loc[:, 'end'] = df.loc[:, 'start'] + width 
    df.loc[:, 'summit'] = width // 2
    return df

def expand_3col_to_10col(df):
    # Expands a 3-column DataFrame to a 10-column DataFrame by adding placeholder columns.
    # Assumes the first three columns are chr, start, end.
    # The summit is calculated as the midpoint between start and end.
    # The additional columns are filled with '.' as placeholders.
    # The final DataFrame has columns:
    # ['chr', 'start', 'end', 'name', 'score', 'strand', 'signalValue', 'pValue', 'qValue', 'summit']
    # where 'name', 'score', 'strand', 'signalValue', 'pValue', 'qValue' are placeholders.
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

# width used to default to 500, Lei said the only reason he added that was for RegNet so I've removed it
def load_region_df(
    regions_bed,
    chrom_sizes=None,
    in_window=2114,
    shift=0,
    is_peak: bool=True,
    logger=None,
    width=None,
    skip_missing_hist=False,
    atac_hgp_map="",
):
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

    width = width if width is not None else in_window
    cprint(f"Formatting {filtered_df.shape[0]} regions from {regions_bed} using width {width}.", logger=logger)
    filtered_df = center_region_around_summit(filtered_df, width)

    # Reset index to avoid index errors
    filtered_df = filtered_df.reset_index(drop=True)

    add_peak_id(filtered_df)

    # filter out any regions that dont have a matching histone peak
    if skip_missing_hist:
        assert atac_hgp_map != ""
        atac_hgp_df = pd.read_csv(atac_hgp_map, sep="\t", header=0)
        add_peak_id(atac_hgp_df, chr_key="chrom")
        merged = filtered_df.merge(
            atac_hgp_df[["peak_id", "hist_chrom", "hist_start", "hist_end"]],
            on="peak_id",
            how="left"
        )
        if len(merged) != len(filtered_df):
            raise ValueError("Some peaks in filtered_df have multiple matches in atac_hgp_df based on peak_id.")
        idx = merged[merged['hist_chrom'] != '.'].index
        pct_filtered = 100 * (len(filtered_df) - len(idx)) / len(filtered_df)
        cprint(f"Filtering out {len(filtered_df) - len(idx)} regions ({pct_filtered:.2f}%) with no matching histone peak.", logger=logger)
        filtered_df = filtered_df.loc[idx].reset_index(drop=True)

    return filtered_df

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

    if negatives is not None and len(negatives) > len(peaks) * negative_sampling_ratio and negative_sampling_ratio > 0:
        # TODO: do we need to pass random state here?
        negatives = negatives.sample(n=int(negative_sampling_ratio * len(peaks)), replace=False)
        data = pd.concat([peaks, negatives], ignore_index=True)
    else:
        data = peaks
        
    return data

def split_peak_and_nonpeak(data):
    data['is_peak'] = data['is_peak'].astype(int).astype(bool)
    non_peaks = data[~data['is_peak']].copy()
    if len(non_peaks) == 0:
        non_peaks = None
    peaks = data[data['is_peak']].copy()
    return peaks, non_peaks

def get_cts(peaks_df, bw, width, atac_hgp_df=None, get_total_cts: bool = False, skip_missing_hist: bool = False):
    """
    Fetches values from a bigwig bw, given a df with minimally
    chr, start and summit columns. Summit is relative to start.
    Retrieves values of specified width centered at summit.

    "cts" = per base counts across a region

    return shape: (len(peaks_df), width)
    """
    vals = []

    if atac_hgp_df is not None:
        u = (atac_hgp_df["end"] - atac_hgp_df["start"]).unique()
        assert u.shape[0] == 1, "All ATAC-Histone mapping regions must have the same length."
        peak_len_a = u[0]
        u2 = (peaks_df["end"] - peaks_df["start"]).unique()
        assert u2.shape[0] == 1, "All ATAC peaks must have the same length."
        peak_len_b = u2[0]
        assert peak_len_a == peak_len_b, "ATAC-Histone mapping regions must have the same length as ATAC peaks."

        merged = peaks_df.merge(
            atac_hgp_df[["peak_id", "hist_chrom", "hist_start", "hist_end"]],
            on="peak_id",
            how="left"
        )
        if len(merged) != len(peaks_df):
            raise ValueError("Some peaks in peaks_df have multiple matches in atac_hgp_df based on peak_id.")
        for _, r in merged.iterrows():
            if pd.isna(r['hist_chrom']):
                raise ValueError(f"No matching ATAC-Histone mapping found for region: {r['chr']}:{r['start']}-{r['end']}")
            elif r.hist_chrom == '.':
                assert not skip_missing_hist, "skip_missing_hist is True but found missing histone peak."
                if get_total_cts:
                    vals.append(np.array([0]))
                else:
                    vals.append(np.zeros(width))
            else:
                if not get_total_cts:
                    raw = bw.values(r.hist_chrom, r.hist_start, r.hist_end)
                    # pad to width w/ 0
                    padded = np.zeros(width, dtype=float)
                    padded[:len(raw)] = raw
                    # TODO_later should prob make this a sparse matrix. also validate 
                    vals.append(np.nan_to_num(padded))
                else:
                    vals.append(np.array([
                        np.nansum(bw.values(r.hist_chrom, r.hist_start, r.hist_end))
                    ]))
    else:
        for _, r in peaks_df.iterrows():
            if not get_total_cts:
                vals.append(
                    np.nan_to_num(bw.values(r['chr'],
                                            r['start'] + r['summit'] - width//2,
                                            r['start'] + r['summit'] + width//2))
                )
            else:
                vals.append(np.array([
                    np.nansum(bw.values(r['chr'],
                                        r['start'] + r['summit'] - width//2,
                                        r['start'] + r['summit'] + width//2))
                ]))

    return np.array(vals)

def get_seq(peaks_df, genome, width):
    """
    Same as get_cts, but fetches sequence from a given genome.
    """
    vals = []
    for _, r in peaks_df.iterrows():
        sequence = str(genome[r['chr']][(r['start']+r['summit'] - width//2):(r['start'] + r['summit'] + width//2)])
        vals.append(sequence)

    return dna_to_one_hot(vals)

def get_coords(peaks_df, peaks_bool):
    """
    Fetch the co-ordinates of the regions in bed file
    returns a list of tuples with (chrom, summit)
    """
    vals = []
    for _, r in peaks_df.iterrows():
        vals.append([r['chr'], r['start']+r['summit'], "f", peaks_bool])

    return np.array(vals)

def get_seq_cts_coords(
    peaks_df,
    genome,
    bw,
    bw_ctrl,
    input_width,
    output_width,
    peaks_bool,
    atac_hgp_df=None,
    get_total_cts: bool = False,
    skip_missing_hist: bool = False,
    mode: str = "train",
):
    # TODO_later remove this after im done debugging
    peaks_str = "peaks" if peaks_bool==1 else "nonpeaks"

    temp_p = f"/large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_hist/histobpnet_v2/train/seqs_{mode}_{peaks_str}.npy"
    if not os.path.isfile(temp_p):
        seq = get_seq(peaks_df, genome, input_width)
        np.save(temp_p, seq)
    else:
        seq = np.load(temp_p)

    temp_p = f"/large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_hist/histobpnet_v2/data/cts_{mode}_{peaks_str}_fast.npy"
    if not os.path.isfile(temp_p):
        cts = get_cts(peaks_df, bw, output_width, atac_hgp_df=atac_hgp_df, get_total_cts=get_total_cts, skip_missing_hist=skip_missing_hist)
        np.save(temp_p, cts)
    else:
        cts = np.load(temp_p)

    if bw_ctrl is None:
        cts_ctrl = None
    else:
        temp_p = f"/large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_hist/histobpnet_v2/data/cts_ctrl_{mode}_{peaks_str}_fast.npy"
        if not os.path.isfile(temp_p):
            cts_ctrl = get_cts(peaks_df, bw_ctrl, output_width, atac_hgp_df=atac_hgp_df, get_total_cts=get_total_cts, skip_missing_hist=skip_missing_hist)
            np.save(temp_p, cts_ctrl)
        else:
            cts_ctrl = np.load(temp_p)

    coords = get_coords(peaks_df, peaks_bool)
    return seq, cts, cts_ctrl, coords

def load_data(
    bed_regions,
    nonpeak_regions,
    genome_fasta,
    cts_bw_file,
    inputlen,
    outputlen,
    max_jitter,
    cts_ctrl_bw_file = None,
    output_bins = None,
    atac_hgp_df = None,
    get_total_cts = False,
    skip_missing_hist = False,
    mode: str = "train",
):
    """
    Load sequences and corresponding base resolution counts for training, 
    validation regions in peaks and nonpeaks (2 x 2 x 2 = 8 matrices).

    For training peaks/nonpeaks, values for inputlen + 2*max_jitter and outputlen + 2*max_jitter 
    are returned centered at peak summit. This allows for jittering examples by randomly
    cropping. Data of width inputlen/outputlen is returned for validation
    data.
    """
    cts_bw = pyBigWig.open(cts_bw_file)
    cts_ctrl_bw = pyBigWig.open(cts_ctrl_bw_file) if cts_ctrl_bw_file is not None else None
    genome = pyfaidx.Fasta(genome_fasta)

    if output_bins is not None:
        output_bins = [int(x) for x in output_bins.split(",")]
        output_len = max(output_bins)
        output_len_neg = output_len
    elif atac_hgp_df is not None:
        output_len = (atac_hgp_df['hist_end'] - atac_hgp_df['hist_start']).max()
        # we'll use this for nonpeak regions since we dont know what a "good" outputlen should be
        output_len_neg = int((atac_hgp_df['hist_end'] - atac_hgp_df['hist_start']).mean())
    else:
        output_len = outputlen
        output_len_neg = output_len
    
    # peaks
    train_peaks_seqs=None
    train_peaks_cts=None
    train_peaks_cts_ctrl=None
    train_peaks_coords=None
    # nonpeaks
    train_nonpeaks_seqs=None
    train_nonpeaks_cts=None
    train_nonpeaks_cts_ctrl=None
    train_nonpeaks_coords=None

    if bed_regions is not None:
        if not set(['chr', 'start', 'summit']).issubset(bed_regions.columns):
            bed_regions = expand_3col_to_10col(bed_regions)
        train_peaks_seqs, train_peaks_cts, train_peaks_cts_ctrl, train_peaks_coords = get_seq_cts_coords(
            bed_regions,
            genome,
            cts_bw,
            cts_ctrl_bw,
            inputlen+2*max_jitter,
            output_len+2*max_jitter,
            peaks_bool=1,
            atac_hgp_df=atac_hgp_df,
            get_total_cts=get_total_cts,
            skip_missing_hist=skip_missing_hist,
            mode=mode,
        )
    
    if nonpeak_regions is not None:
        if not set(['chr', 'start', 'summit']).issubset(nonpeak_regions.columns):
            nonpeak_regions = expand_3col_to_10col(nonpeak_regions)
        # no reason to pass atac_hgp_df here since nonpeaks shouldn't have entries in that df
        train_nonpeaks_seqs, train_nonpeaks_cts, train_nonpeaks_cts_ctrl, train_nonpeaks_coords = get_seq_cts_coords(
            nonpeak_regions,
            genome,
            cts_bw,
            cts_ctrl_bw,
            inputlen,
            output_len_neg,
            peaks_bool=0,
            get_total_cts=get_total_cts,
            skip_missing_hist=skip_missing_hist,
            mode=mode,
        )

    cts_bw.close()
    if cts_ctrl_bw is not None:
        cts_ctrl_bw.close()
    genome.close()

    if (train_peaks_seqs.sum(axis=-1) == 1).all():
        print('One-hot encoding verified for peak sequences.')
    else:
        # TODO figure out why this happens -> see get_seq in data_utils.py
        # I guess some of the corresponding sequences have letters other than ACGT?
        print('Warning: Peak sequences are not one-hot encoded?!')

    return (train_peaks_seqs, train_peaks_cts, train_peaks_cts_ctrl, train_peaks_coords,
            train_nonpeaks_seqs, train_nonpeaks_cts, train_nonpeaks_cts_ctrl, train_nonpeaks_coords)

# valeh: I skimmed through this
def write_bigwig(
    data,
    regions,
    chrom_sizes: list,
    bw_out: str,
    debug_chr: str = None,
    use_tqdm: bool = False,
    outstats_file: str = None
):
    # regions may overlap but as we go in sorted order, at a given position,
    # we will pick the value from the interval whose summit is closest to 
    # current position
    chr_to_idx = {}
    for i,x in enumerate(chrom_sizes):
        chr_to_idx[x[0]] = i

    bw = pyBigWig.open(bw_out, 'w')
    bw.addHeader(chrom_sizes)
    
    # regions may not be sorted, so get their sorted order (sort by chromosome index and then by start position).
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
        if debug_chr and (regions[i][0] != debug_chr):
            continue

        i = order_of_regs[itr]
        i_chr, i_start, i_end, i_mid = regions[i]
    
        if i_chr != cur_chr: 
            cur_chr = i_chr
            cur_end = 0
    
        # bring current end to at least start of current region
        if cur_end < i_start:
            cur_end = i_start
    
        assert(regions[i][2] >= cur_end)
    
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
        bw.addEntries(
            [i_chr]*(next_end-cur_end), 
            list(range(cur_end,next_end)), 
            ends = list(range(cur_end+1, next_end+1)), 
            values=[float(x) for x in vals]
        )
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

def get_regions(regions_file, seqlen, regions_used = None, regions_format: str = 'bed10'):
    assert(seqlen%2 == 0)

    regions = pd.read_csv(regions_file, sep='\t', header=None)
    rr = np.array(regions.values) if regions_used is None else np.array(regions.values)[regions_used]

    regions_out = []
    for x in rr:
        if regions_format == 'bed10':
            assert len(x) >= 10, "Expected at least 10 columns for bed10 format"
            chr, start, end, mid = x[0], int(x[1])+int(x[9])-seqlen//2, int(x[1])+int(x[9])+seqlen//2, int(x[1])+int(x[9])
        elif regions_format == 'bed':
            assert len(x) >= 3, "Expected at least 3 columns for bed format"
            chr, start, end, mid = x[0], int(x[1]), int(x[2]), (int(x[1]) + int(x[2])) // 2
        else:
            raise ValueError(f"Unsupported regions_format: {regions_format}. Supported formats are 'bed10' and 'bed'.")
        regions_out.append([chr, start, end, mid])

    return regions_out

def load_h5_and_sum(hdf5: str, key: str, chunk_n: int = 256):
    import h5py
    with h5py.File(hdf5, "r") as f:
        ds = f[key]
        shape = ds.shape

        # interpret orientation
        if shape[1] == 4:
            # (N, 4, seqlen)
            axis_channel = 1
            SEQLEN = shape[2]
        elif shape[2] == 4:
            # (N, seqlen, 4)
            axis_channel = 2
            SEQLEN = shape[1]
        else:
            raise ValueError("No channel dimension of size 4 found. We expect either (N,4,seqlen) or (N,seqlen,4).")

        N = shape[0]
        assert(SEQLEN%2 == 0)

        out = np.zeros((N, SEQLEN), dtype=ds.dtype)

        # chunk over N
        for i in range(0, N, chunk_n):
            j = min(i + chunk_n, N)

            if axis_channel == 1:
                # data already (N,4,L)
                # sometimes there is a trailing dim of 1 on the last axis for whatever reason
                # squeeze that out
                block = ds[i:j, :, :].squeeze()
                out[i:j] = np.sum(block, axis=1)
            else:
                # data is (N,L,4) — sum across last axis
                # sometimes there is a trailing dim of 1 on the last axis for whatever reason
                # squeeze that out
                block = ds[i:j, :, :].squeeze()
                out[i:j] = np.sum(block, axis=2)

    return out, SEQLEN

def hdf5_to_bigwig(
    hdf5: str,
    regions_file: str,
    chrom_sizes: dict,
    output_prefix,
    output_prefix_stats = None,
    debug_chr = None,
    tqdm: bool = False,
    h5_read_tool: str = "deepdish",
    hdf5_key: str = '/projected_shap/seq',
    regions_format: str = 'bed10',
    chunk_n: int = 256,
):
    if h5_read_tool == "deepdish":
        raise NotImplementedError("You should probably just switch to h5py, think about it")
    elif h5_read_tool == "h5py":
        d, SEQLEN = load_h5_and_sum(hdf5, hdf5_key, chunk_n=chunk_n)
    else:
        raise NotImplementedError(f"Unsupported h5_read_tool: {h5_read_tool}. Supported tools are 'deepdish' and 'h5py'.")

    regions = get_regions(regions_file, SEQLEN, regions_format = regions_format)
    assert(len(regions) == d.shape[0])

    chr_set = set([region[0] for region in regions])
    chr_sizes = [(x, v) for x, v in chrom_sizes.items() if x in chr_set]

    write_bigwig(
        d,
        regions, 
        chr_sizes, 
        output_prefix+".bw", 
        outstats_file=output_prefix_stats, 
        debug_chr=debug_chr, 
        use_tqdm=tqdm
    )

def debug_subsample(peak_regions, chrom=None):
    if peak_regions is None:
        return None

    if chrom is None:
        chrom = peak_regions['chr'].unique()[0]

    peak_regions = peak_regions[peak_regions['chr'] == chrom]
    # print('debugging on ', chrom, 'shape', peak_regions.shape)
    return peak_regions.reset_index(drop=True)

# https://stackoverflow.com/questions/46091111/python-slice-array-at-different-position-on-every-row
def take_per_row(A, indx, num_elem):
    """
    Matrix A, indx is a vector for each row which specifies 
    slice beginning for that row. Each has width num_elem.
    """

    all_indx = indx[:,None] + np.arange(num_elem)
    # valeh: this is basically taking all the rows and eveyrhting across the channel axis
    # (ACGT) and only slicing across the length axis using all_indx.
    # we can't do A[:, all_indx] for some reason but this syntax below wokrs as expected
    return A[np.arange(all_indx.shape[0])[:,None], all_indx]

def random_crop(seqs, labels, labels_ctrl, seq_crop_width, label_crop_width, coords):
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
    if labels_ctrl is not None:
        assert(labels_ctrl.shape[1] >= label_crop_width)

    # here is a graphic illustrating this
    #
    # |<------ seq_width ------>|
    #   |<--- label_width --->|
    # seq_width - alpha = label_width (alpha is "crop")
    #
    # after cropping both seq and label:
    # |<------ seq_crop_width ------>|
    #   |<--- label_crop_width --->|
    # we must still have:
    # seq_crop_width - alpha = label_crop_width
    #
    # which is the same as: 
    # seq_width - label_width = seq_crop_width - label_crop_width
    # aka
    # seq_width - seq_crop_width = label_width - label_crop_width
    #
    assert(seqs.shape[1] - seq_crop_width == labels.shape[1] - label_crop_width)
    if labels_ctrl is not None:
        assert(seqs.shape[1] - seq_crop_width == labels_ctrl.shape[1] - label_crop_width)

    max_start = seqs.shape[1] - seq_crop_width
    starts = np.random.choice(range(max_start+1), size=seqs.shape[0], replace=True)
    new_coords = coords.copy()
    new_coords[:,1] = new_coords[:,1].astype(int) - (seqs.shape[1]//2) + starts

    return (
        take_per_row(seqs, starts, seq_crop_width),
        take_per_row(labels, starts, label_crop_width),
        take_per_row(labels_ctrl, starts, label_crop_width) if labels_ctrl is not None else None,
        new_coords,
        starts
    )

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

def revcomp_shuffle_augment(seqs, labels, coords, add_revcomp, rc_frac=0.5, shuffle=False):
    """
    seqs: B x IL x 4
    labels: B x OL
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
    
def crop_revcomp_data(
    peak_seqs, peak_cts, peak_cts_ctrl, peak_coords, 
    nonpeak_seqs=None, nonpeak_cts=None, nonpeak_cts_ctrl=None, nonpeak_coords=None, 
    per_bin_peak_cts_dict=None, per_bin_peak_cts_ctrl_dict=None,
    per_bin_nonpeak_cts_dict=None, per_bin_nonpeak_cts_ctrl_dict=None,
    inputlen=2114, outputlen=1000, output_bins: list = None,
    add_revcomp=False, negative_sampling_ratio=0.1, shuffle=False, do_crop=True, rc_frac: float=0.5):
    """Apply random cropping and reverse complement augmentation to the data.
        
        This method:
        1. Randomly crops peak data to inputlen and outputlen
        2. Samples negative examples according to negative_sampling_ratio
        3. Applies reverse complement augmentation if enabled
        4. Shuffles data if shuffle is True
    """
    def crop_peak_data():
        if output_bins is None:
            cropped_peaks, cropped_cnts, cropped_cnts_ctrl, cropped_coords, _ = random_crop(
                peak_seqs, peak_cts, peak_cts_ctrl, inputlen, outputlen, peak_coords
            ) if do_crop else (peak_seqs, peak_cts, peak_cts_ctrl, peak_coords, None)
            return cropped_peaks, cropped_cnts, cropped_cnts_ctrl, cropped_coords
        else:
            # I dont expect a path where do_crop is False and output_bins is not None
            assert do_crop
            cropped_cnts = {}
            cropped_cnts_ctrl = {}
            starts = None
            for w in output_bins:
                if starts is None:
                    cropped_peaks, cropped_cnts[w], cropped_cnts_ctrl[w], cropped_coords, starts = random_crop(
                        peak_seqs, per_bin_peak_cts_dict[w], per_bin_peak_cts_ctrl_dict[w], inputlen, w, peak_coords
                    )
                else:
                    cropped_cnts[w] = take_per_row(per_bin_peak_cts_dict[w], starts, w)
                    cropped_cnts_ctrl[w] = take_per_row(per_bin_peak_cts_ctrl_dict[w], starts, w)
            return cropped_peaks, cropped_cnts, cropped_cnts_ctrl, cropped_coords

    if (peak_seqs is not None) and (nonpeak_seqs is not None):
        # Crop peak data
        cropped_peaks, cropped_cnts, cropped_cnts_ctrl, cropped_coords = crop_peak_data()
        
        # Sample negative examples
        # valeh: why is this needed btw? dont we do this during pre-processing?
        if negative_sampling_ratio > 0:
            if output_bins is not None:
                raise NotImplementedError("Subsampling non-peak data with output bins is not implemented yet.")
            sampled_nonpeak_seqs, sampled_nonpeak_cts, sampled_nonpeak_coords = subsample_nonpeak_data(
                nonpeak_seqs, nonpeak_cts, nonpeak_coords,
                len(peak_seqs), negative_sampling_ratio
            )
            seqs = np.vstack([cropped_peaks, sampled_nonpeak_seqs])
            coords = np.vstack([cropped_coords, sampled_nonpeak_coords])
            cts = np.vstack([cropped_cnts, sampled_nonpeak_cts])
            peak_status = np.array([1]*len(cropped_peaks) + [0]*len(sampled_nonpeak_seqs)).reshape(-1,1)
        else:
            seqs = np.vstack([cropped_peaks, nonpeak_seqs])
            coords = np.vstack([cropped_coords, nonpeak_coords])

            if output_bins is None:
                cts = np.vstack([cropped_cnts, nonpeak_cts])
                cts_ctrl = None if nonpeak_cts_ctrl is None else np.vstack([cropped_cnts_ctrl, nonpeak_cts_ctrl])
            else:
                # concatenate dicts
                cts, cts_ctrl = {}, {}
                for w in output_bins:
                    cts[w] = np.vstack([cropped_cnts[w], per_bin_nonpeak_cts_dict[w]])
                    cts_ctrl[w] = np.vstack([cropped_cnts_ctrl[w], per_bin_nonpeak_cts_ctrl_dict[w]])
            peak_status = np.array([1]*len(cropped_peaks) + [0]*len(nonpeak_seqs)).reshape(-1,1)
    elif peak_seqs is not None:
        # Only peak data
        cropped_peaks, cropped_cnts, cropped_cnts_ctrl, cropped_coords = crop_peak_data()
        seqs = cropped_peaks
        coords = cropped_coords
        cts = cropped_cnts
        cts_ctrl = cropped_cnts_ctrl
        peak_status = np.array([1]*len(cropped_peaks)).reshape(-1,1)
    elif nonpeak_seqs is not None:
        # Only non-peak data
        seqs = nonpeak_seqs
        coords = nonpeak_coords
        cts = nonpeak_cts if output_bins is None else per_bin_nonpeak_cts_dict
        cts_ctrl = nonpeak_cts_ctrl if output_bins is None else per_bin_nonpeak_cts_ctrl_dict
        peak_status = np.array([0]*len(nonpeak_seqs)).reshape(-1,1)
    else:
        raise ValueError("Both peak and non-peak arrays are empty")

    # Apply revcomp and shuffle augmentations
    # if rc_frac == 0, there is nothing to revcomp
    if (add_revcomp and rc_frac > 0) or shuffle:
        seqs, cts, coords = revcomp_shuffle_augment(
            seqs, cts, coords,
            add_revcomp, shuffle=shuffle, rc_frac=rc_frac
        )

    return seqs, cts, cts_ctrl, coords, peak_status