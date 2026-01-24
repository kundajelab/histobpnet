# Author: Lei Xiong <jsxlei@gmail.com>

import numpy as np 
from scipy.stats import spearmanr, pearsonr
from scipy.spatial.distance import jensenshannon
import matplotlib
from matplotlib import pyplot as plt
import os
import json
import h5py
from toolbox.plt_utils import density_scatter
from histobpnet.utils.general_utils import softmax
from histobpnet.utils.data_utils import write_bigwig, expand_3col_to_10col

def set_plotting_params(figsize = None):
    plt.rcParams["figure.figsize"] = figsize if figsize is not None else (10,5)
    font = {'weight' : 'bold',
            'size'   : 10}
    matplotlib.rc('font', **font)

#https://github.com/kundajelab/basepairmodels/blob/cf8e346e9df1bad9e55bd459041976b41207e6e5/basepairmodels/cli/fastpredict.py#L131
def jsd_min_max_bounds(profile):
    """
    Min Max bounds for the jsd metric
    
    Args:
        profile (numpy.ndarray): true profile 
        
    Returns:
        tuple: (min, max) bounds values
    """
    # uniform distribution profile
    uniform_profile = np.ones(len(profile)) * (1.0 / len(profile))

    # profile as probabilities
    profile_prob = profile / np.sum(profile)

    # jsd of profile with uniform profile
    max_jsd = jensenshannon(profile_prob, uniform_profile)

    # jsd of profile with itself (upper bound)
    min_jsd = 0.0

    return (min_jsd, max_jsd)

#https://github.com/kundajelab/basepairmodels/blob/cf8e346e9df1bad9e55bd459041976b41207e6e5/basepairmodels/cli/metrics.py#L129
def get_min_max_normalized_value(val, minimum, maximum):
    ret_val = (val - maximum) / (minimum - maximum)

    if ret_val < 0:
        return 0
    
    if ret_val > 1:
        return 1
    
    return ret_val

def load_output_to_regions(output, regions, out_dir: str):
    """
    Load the output to regions
    """
    # regions includes peaks and non-peaks
    regions = regions.reset_index(drop=True).copy()
    parsed_output = {key: np.concatenate([batch[key] for batch in output]) for key in output[0]}
    # cast 'is_peak' to int if it exists
    if 'is_peak' in regions.columns:
        regions['is_peak'] = regions['is_peak'].astype(int)
    regions['pred_count'] = parsed_output['pred_count']
    regions['true_count'] = parsed_output['true_count']
    if 'pred_lfc' in parsed_output:
        regions['pred_lfc'] = parsed_output['pred_lfc']
        regions['true_lfc'] = parsed_output['true_lfc']
    regions.to_csv(os.path.join(out_dir, 'regions.csv'), sep='\t', index=False)
    return regions, parsed_output

def compare_with_observed(regions, parsed_output, out_dir: str, tag='all_regions', skip_profile: bool = False,
                          model_wrapper=None, wandb_log_name: str = None):
    metrics_dictionary = {}

    # save count metrics (peaks and nonpeaks)
    metrics_dictionary["counts_metrics"] = {}
    spearman_cor, pearson_cor, mse = counts_metrics(
        regions['true_count'],
        regions['pred_count'],
        os.path.join(out_dir, 'all_regions'),
        model_wrapper=model_wrapper,
        wandb_log_name=wandb_log_name,
    )
    metrics_dictionary["counts_metrics"]["peaks_and_nonpeaks"] = {}
    metrics_dictionary["counts_metrics"]["peaks_and_nonpeaks"]["spearmanr"] = spearman_cor
    metrics_dictionary["counts_metrics"]["peaks_and_nonpeaks"]["pearsonr"] = pearson_cor
    metrics_dictionary["counts_metrics"]["peaks_and_nonpeaks"]["mse"] = mse

    # save lfc metrics if applicable
    if 'pred_lfc' in regions:
        metrics_dictionary["lfc_metrics"] = {}
        spearman_cor, pearson_cor, mse = counts_metrics(
            regions['true_lfc'],
            regions['pred_lfc'],
            os.path.join(out_dir, 'all_regions'),
            model_wrapper=model_wrapper,
            wandb_log_name=wandb_log_name,
            type_ = "lfc",
        )
        metrics_dictionary["lfc_metrics"]["peaks_and_nonpeaks"] = {}
        metrics_dictionary["lfc_metrics"]["peaks_and_nonpeaks"]["spearmanr"] = spearman_cor
        metrics_dictionary["lfc_metrics"]["peaks_and_nonpeaks"]["pearsonr"] = pearson_cor
        metrics_dictionary["lfc_metrics"]["peaks_and_nonpeaks"]["mse"] = mse

    # save profile metrics (peaks and nonpeaks)
    if not skip_profile:
        metrics_dictionary["profile_metrics"] = {}
        jsd_pw, jsd_norm, jsd_rnd, _ = profile_metrics(
            parsed_output['true_profile'],
            softmax(parsed_output['pred_profile'])
        )
        plot_histogram(jsd_pw, jsd_rnd, os.path.join(out_dir, 'all_regions_jsd'), '')
        metrics_dictionary["profile_metrics"]["peaks_and_nonpeaks"] = {}
        metrics_dictionary["profile_metrics"]["peaks_and_nonpeaks"]["median_jsd"] = np.nanmedian(jsd_pw)        
        metrics_dictionary["profile_metrics"]["peaks_and_nonpeaks"]["median_norm_jsd"] = np.nanmedian(jsd_norm)

    if 'is_peak' in regions.columns:
        # PEAK REGIONS
        peak_regions = regions[regions['is_peak'] == 1].copy()
        peak_index = peak_regions.index
        peak_regions = peak_regions.reset_index(drop=True)
        print('peak_regions', peak_regions.head())

        spearman_cor, pearson_cor, mse = counts_metrics(
            peak_regions['true_count'],
            peak_regions['pred_count'],
            os.path.join(out_dir, 'peaks'),
            model_wrapper=model_wrapper,
            wandb_log_name=wandb_log_name,
        )
        metrics_dictionary["counts_metrics"]["peaks"] = {}
        metrics_dictionary["counts_metrics"]["peaks"]["spearmanr"] = spearman_cor
        metrics_dictionary["counts_metrics"]["peaks"]["pearsonr"] = pearson_cor
        metrics_dictionary["counts_metrics"]["peaks"]["mse"] = mse

        if 'pred_lfc' in regions:
            spearman_cor, pearson_cor, mse = counts_metrics(
                peak_regions['true_lfc'],
                peak_regions['pred_lfc'],
                os.path.join(out_dir, 'peaks'),
                model_wrapper=model_wrapper,
                wandb_log_name=wandb_log_name,
                type_ = "lfc",
            )
            metrics_dictionary["lfc_metrics"]["peaks"] = {}
            metrics_dictionary["lfc_metrics"]["peaks"]["spearmanr"] = spearman_cor
            metrics_dictionary["lfc_metrics"]["peaks"]["pearsonr"] = pearson_cor
            metrics_dictionary["lfc_metrics"]["peaks"]["mse"] = mse

        if not skip_profile:
            jsd_pw, jsd_norm, jsd_rnd, _ = profile_metrics(
                parsed_output['true_profile'][peak_index],
                softmax(parsed_output['pred_profile'])[peak_index]
            )
            plot_histogram(jsd_pw, jsd_rnd, os.path.join(out_dir, 'peaks_jsd'), '')
            metrics_dictionary["profile_metrics"]["peaks"] = {}
            metrics_dictionary["profile_metrics"]["peaks"]["median_jsd"] = np.nanmedian(jsd_pw)        
            metrics_dictionary["profile_metrics"]["peaks"]["median_norm_jsd"] = np.nanmedian(jsd_norm)

        # NON-PEAK REGIONS
        nonpeak_regions = regions[regions['is_peak']==0].copy()
        nonpeak_index = nonpeak_regions.index
        nonpeak_regions = nonpeak_regions.reset_index(drop=True)
        print('nonpeak_regions', nonpeak_regions.head())

        spearman_cor, pearson_cor, mse = counts_metrics(
            nonpeak_regions['true_count'],
            nonpeak_regions['pred_count'],
            os.path.join(out_dir, 'nonpeaks'),
            model_wrapper=model_wrapper,
            wandb_log_name=wandb_log_name,
        )
        metrics_dictionary["counts_metrics"]["nonpeaks"] = {}
        metrics_dictionary["counts_metrics"]["nonpeaks"]["spearmanr"] = spearman_cor
        metrics_dictionary["counts_metrics"]["nonpeaks"]["pearsonr"] = pearson_cor
        metrics_dictionary["counts_metrics"]["nonpeaks"]["mse"] = mse

        if 'pred_lfc' in regions:
            spearman_cor, pearson_cor, mse = counts_metrics(
                nonpeak_regions['true_lfc'],
                nonpeak_regions['pred_lfc'],
                os.path.join(out_dir, 'nonpeaks'),
                model_wrapper=model_wrapper,
                wandb_log_name=wandb_log_name,
                type_ = "lfc",
            )
            metrics_dictionary["lfc_metrics"]["nonpeaks"] = {}
            metrics_dictionary["lfc_metrics"]["nonpeaks"]["spearmanr"] = spearman_cor
            metrics_dictionary["lfc_metrics"]["nonpeaks"]["pearsonr"] = pearson_cor
            metrics_dictionary["lfc_metrics"]["nonpeaks"]["mse"] = mse
        
        if not skip_profile:
            jsd_pw, jsd_norm, jsd_rnd, _ = profile_metrics(
                parsed_output['true_profile'][nonpeak_index],
                softmax(parsed_output['pred_profile'])[nonpeak_index]
            )
            plot_histogram(jsd_pw, jsd_rnd, os.path.join(out_dir, 'nonpeaks_jsd'), '')
            metrics_dictionary["profile_metrics"]["nonpeaks"] = {}
            metrics_dictionary["profile_metrics"]["nonpeaks"]["median_jsd"] = np.nanmedian(jsd_pw)
            metrics_dictionary["profile_metrics"]["nonpeaks"]["median_norm_jsd"] = np.nanmedian(jsd_norm)

    print(json.dumps(metrics_dictionary, indent=4, default=lambda o: float(o)))

    with open(os.path.join(out_dir, f'metrics_{tag}.json'), 'w') as fp:
        json.dump(metrics_dictionary, fp,  indent=4, default=lambda x: float(x))
    return metrics_dictionary

def counts_metrics(
    labels,
    preds,
    outf: str = None,
    fontsize = 20,
    xlab: str = None,
    ylab: str = None,
    model_wrapper=None,
    wandb_log_name: str = None,
    type_: str = 'counts',
):
    assert type_ in ['counts', 'lfc'], "type_ must be 'counts' or 'lfc'"
    
    if xlab is None:
        xlab = 'Log Count Labels' if type_ == 'counts' else 'LFC Labels'
    if ylab is None:
        ylab = 'Log Count Predictions' if type_ == 'counts' else 'LFC Predictions'

    spearman_cor = spearmanr(labels, preds)[0]
    pearson_cor = pearsonr(labels, preds)[0]  
    mse=((labels - preds)**2).mean(axis=0)

    set_plotting_params((8, 8))
    # fig=plt.figure()
    fig, _, _, _, _ = density_scatter(
        labels,
        preds,
        xlab=xlab,
        ylab=ylab,
        fontsize=fontsize,
    )
    fig.suptitle(
        "count: spearman R="+str(round(spearman_cor,3))+"\nPearson R="+str(round(pearson_cor,3))+"\nmse="+str(round(mse,3)),
        y=0.9,
        fontsize=20
    )
    # plt.legend(loc='best')

    filepath = outf+f'.{type_}_pearsonr.png' if outf is not None else None
    if outf is not None:
        plt.savefig(outf+f'.{type_}_pearsonr.png',format='png',dpi=300)
    
        if wandb_log_name is not None:
            assert model_wrapper is not None, "model_wrapper must be provided when wandb_log_name is not None"
            # if the name is /xxx/yyy/zzz.counts_pearsonr.png, get the zzz part
            region_type = os.path.basename(filepath).split(".", 1)[0]
            filename = f"{wandb_log_name}_scatter_{region_type}"
            model_wrapper._log_plot(fig, name=filename, close_fig = False)
    
    plt.show()
    plt.close()

    return spearman_cor, pearson_cor, mse

def profile_metrics(true_counts, pred_probs, pseudocount = 0.001):
    jsd_pw = []
    jsd_norm = []
    jsd_rnd = []
    jsd_rnd_norm = []

    num_regions = true_counts.shape[0]
    for idx in range(num_regions):
        true_probs = true_counts[idx,:]/(pseudocount + np.nansum(true_counts[idx,:]))

        # jsd
        cur_jsd = jensenshannon(true_probs, pred_probs[idx,:])
        jsd_pw.append(cur_jsd)
        
        # normalized jsd
        min_jsd, max_jsd = jsd_min_max_bounds(true_counts[idx,:])
        curr_jsd_norm = get_min_max_normalized_value(cur_jsd, min_jsd, max_jsd)
        jsd_norm.append(curr_jsd_norm)

        # get random shuffling on labels for a worst case performance on metrics - labels versus shuffled labels
        shuffled_labels = np.random.permutation(true_counts[idx,:])
        shuffled_labels_prob = shuffled_labels/(pseudocount + np.nansum(shuffled_labels))

        # jsd random
        curr_jsd_rnd = jensenshannon(true_probs, shuffled_labels_prob)
        jsd_rnd.append(curr_jsd_rnd)
        
        # normalized jsd random
        curr_rnd_jsd_norm = get_min_max_normalized_value(curr_jsd_rnd, min_jsd, max_jsd)
        jsd_rnd_norm.append(curr_rnd_jsd_norm)

    return np.array(jsd_pw), np.array(jsd_norm), np.array(jsd_rnd), np.array(jsd_rnd_norm)

def save_predictions(output, regions, chrom_sizes, out_dir: str, seqlen: int = 1000):
    """
    Save the predictions to an HDF5 file and write regions to a CSV file.
    """
    with open(chrom_sizes) as f:
        gs = [x.strip().split('\t') for x in f]
    gs = [(x[0], int(x[1])) for x in gs if len(x)==2]
    if regions.shape[1] < 10:
        regions = expand_3col_to_10col(regions)

    regions_array = [
        [
            x[0],
            int(x[1])+int(x[9])-seqlen//2,
            int(x[1])+int(x[9])+seqlen//2,
            int(x[1])+int(x[9])
        ]
        for x in np.array(regions.values)
    ]

    # parse output
    parsed_output = {key: np.concatenate([batch[key] for batch in output]) for key in output[0]}

    # what is this doing? -> “Convert profile logits → probabilities, convert log-count → total count,
    # and then form expected counts per position as (total count) × (probability per position).”
    data = softmax(parsed_output['pred_profile']) * np.expand_dims(np.exp(parsed_output['pred_count']), axis=1)

    write_bigwig(
        data,
        regions_array,
        gs,
        os.path.join(out_dir, "pred.bw"),
        outstats_file=None,
        debug_chr=None,
        use_tqdm=True
    )

    # save predictions into h5py file
    # write_predictions_h5py(parsed_output['pred_profile'], parsed_output['pred_count'], regions_array, out_dir)

    return

# valeh: REVIEWED ^^^^^^

def write_predictions_h5py(profile, logcts, coords, out_dir: str = './'):
    # open h5 file for writing predictions
    os.makedirs(out_dir, exist_ok=True)
    output_h5_fname = os.path.join(out_dir, "predictions.h5")
    h5_file = h5py.File(output_h5_fname, "w")
    # create groups
    coord_group = h5_file.create_group("coords")
    pred_group = h5_file.create_group("predictions")

    num_examples=len(coords)

    coords_chrom_dset =  [str(coords[i][0]) for i in range(num_examples)]
    coords_center_dset =  [int(coords[i][1]) for i in range(num_examples)]
    coords_peak_dset =  [int(coords[i][3]) for i in range(num_examples)]

    dt = h5py.special_dtype(vlen=str)

    # create the "coords" group datasets
    coords_chrom_dset = coord_group.create_dataset(
        "coords_chrom", data=np.array(coords_chrom_dset, dtype=dt),
        dtype=dt, compression="gzip")
    coords_start_dset = coord_group.create_dataset(
        "coords_center", data=coords_center_dset, dtype=int, compression="gzip")
    coords_end_dset = coord_group.create_dataset(
        "coords_peak", data=coords_peak_dset, dtype=int, compression="gzip")

    # create the "predictions" group datasets
    profs_dset = pred_group.create_dataset(
        "profs",
        data=profile,
        dtype=float, compression="gzip")
    logcounts_dset = pred_group.create_dataset(
        "logcounts", data=logcts,
        dtype=float, compression="gzip")

    # close hdf5 file
    h5_file.close()

def plot_histogram(region_jsd, shuffled_labels_jsd, output_prefix, title):
    num_bins = 100
    plt.rcParams["figure.figsize"] = 8,8
    plt.figure()
    plt.hist(region_jsd, num_bins, facecolor='blue', alpha=0.5, label="Predicted vs Labels")
    plt.hist(shuffled_labels_jsd, num_bins, facecolor='black', alpha=0.5, label='Shuffled Labels vs Labels')
    plt.xlabel('Jensen Shannon Distance Profile Labels and Predictions in Probability Space')
    plt.title("JSD Dist: " + title)
    plt.legend(loc='best')
    plt.savefig(output_prefix + ".profile_jsd.png", format='png', dpi=300)
    plt.close()

def flatten_dict(d, parent_key='', sep='/'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def compare_predictions(out_dir, chrom):
    import pandas as pd
    df_tf = pd.read_csv(os.path.join(out_dir, 'reproduce', chrom, 'regions.csv'), sep='\t')
    df_pt = pd.read_csv(os.path.join(out_dir, 'predict', chrom, 'regions.csv'), sep='\t')
    df_tf_peaks = df_tf[df_tf['is_peak']==1]
    df_pt_peaks = df_pt[df_pt['is_peak']==1]

    # counts_metrics(df_tf['pred_count'], df_pt['pred_count'],outf=os.path.join(out_dir, 'reproduce', chrom, 'compare_counts.png'), title='',
    #     fontsize=20, xlab='Log Count chrombpnet original', ylab='Log Count pytorch')
    counts_metrics(df_tf_peaks['pred_count'], df_pt_peaks['pred_count'],outf=os.path.join(out_dir, 'reproduce', chrom, 'compare'), title='',
        fontsize=20, xlab='Log Count chrombpnet original', ylab='Log Count pytorch')