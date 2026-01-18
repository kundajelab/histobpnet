# Author: Lei Xiong <jsxlei@gmail.com>

import pandas as pd
import numpy as np
import os
import torch
import pyfaidx
from tangermeme.deep_lift_shap import _nonlinear, deep_lift_shap
from tangermeme.utils import _validate_input

from histobpnet.utils.data_utils import (
    get_seq,
    load_region_df,
    hdf5_to_bigwig,
    html_to_pdf,
)

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

class _ProfileLogitScaling(torch.nn.Module):
    """This ugly class is necessary because of Captum.

    Captum internally registers classes as linear or non-linear. Because the
    profile wrapper performs some non-linear operations, those operations must
    be registered as such. However, the inputs to the wrapper are not the
    logits that are being modified in a non-linear manner but rather the
    original sequence that is subsequently run through the model. Hence, this
    object will contain all of the operations performed on the logits and
    can be registered.

    Parameters
    ----------
    logits: torch.Tensor, shape=(-1, -1)
        The logits as they come out of a Chrom/BPNet model.
    """

    def __init__(self):
        super(_ProfileLogitScaling, self).__init__()
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, logits):
        y_softmax = self.softmax(logits)
        y = logits * y_softmax
        return y
        #print("a") 
        #y_lsm = torch.nn.functional.log_softmax(logits, dim=-1)
        #return torch.sign(logits) * torch.exp(torch.log(abs(logits)) + y_lsm)
    
class ProfileWrapper(torch.nn.Module):
    """A wrapper class that returns transformed profiles.

    This class takes in a trained model and returns the weighted softmaxed
    outputs of the first dimension. Specifically, it takes the predicted
    "logits" and takes the dot product between them and the softmaxed versions
    of those logits. This is for convenience when using captum to calculate
    attribution scores.

    Parameters
    ----------
    model: torch.nn.Module
        A torch model to be wrapped.
    """

    def __init__(self, model):
        super(ProfileWrapper, self).__init__()
        self.model = model
        self.flatten = torch.nn.Flatten()
        self.scaling = _ProfileLogitScaling()

    def forward(self, x, x_ctl=None, **kwargs):
        logits = self.model(x, x_ctl=x_ctl, **kwargs)[0]
        logits = self.flatten(logits)
        logits = logits - torch.mean(logits, dim=-1, keepdims=True)
        return self.scaling(logits).sum(dim=-1, keepdims=True)

class CountWrapper(torch.nn.Module):
    """A wrapper class that only returns the predicted counts.

    This class takes in a trained model and returns only the second output.
    For BPNet models, this means that it is only returning the count
    predictions. This is for convenience when calculating attribution scores.

    Parameters
    ----------
    model: torch.nn.Module
        A torch model to be wrapped.
    """

    def __init__(self, model):
        super(CountWrapper, self).__init__()
        self.model = model

    # TODO fix for histobpnet models...
    def forward(self, x, x_ctl=None, **kwargs):
        return self.model(x, x_ctl=x_ctl, **kwargs)[1]

def run_modisco_and_shap(
    model,
    peaks,
    out_dir,
    in_window: str, 
    out_window: str,
    fasta: str,
    chrom_sizes: str,
    task: str = 'counts', 
    batch_size = 32, 
    n_shuffles = 20,
    sub_sample = None, 
    meme_file = None, 
    max_seqlets = 1000_000, 
    width = 500, 
    device = 'cuda',
    debug = False
):
    print("DeepLiftShap and modisco output directory: ", out_dir)
    if debug:
        sub_sample = 30_000
        max_seqlets = 50_000

    # n_control_tracks = model.n_control_tracks

    # # TODO also go through profile later
    # if task == 'profile':
    #     model = ProfileWrapper(model)
    # elif task == 'counts':
    #     model = CountWrapper(model)
    # else:
    #     raise ValueError(f"Task {task} not recognized. Must be 'profile' or 'counts'")

    # regions_df = load_region_df(peaks, chrom_sizes=chrom_sizes, in_window=in_window, is_peak=True)
    # if sub_sample is not None and len(regions_df) > sub_sample:
    #     regions_df = regions_df.sample(sub_sample, random_state=42).reset_index(drop=True)
    # print('Number of peaks:', len(regions_df))

    # seq = get_seq(regions_df, pyfaidx.Fasta(fasta), in_window)
    # if n_control_tracks > 0:
    #     args = [torch.zeros(seq.shape[0], n_control_tracks, out_window)]
    # else:
    #     args = None

    # if isinstance(seq, np.ndarray):
    #     seq = torch.tensor(seq.astype(np.float32))
    # if seq.shape[-1] == 4:
    #     seq = seq.permute(0, 2, 1)

    # # Mask those N values encoded as [0.25, 0.25, 0.25, 0.25]
    # mask = (seq == 0.25).any(dim=(1, 2))
    # seq = seq[~mask]
    # regions_df = regions_df[pd.Series(~mask)]

    # # Mask those sequences that are all 0s
    # mask = (seq == 0).all(dim=1).any(dim=1)  # Shape: (N, L), True where all 0s in dim=1  
    # seq = seq[~mask]  # Filter sequences
    # regions_df = regions_df[pd.Series(~mask)]  # Filter regions

    # for i in range(seq.shape[0]):
    #     try:
    #         X = _validate_input(seq[i], name='seq', shape=(-1, -1), ohe=True, ohe_dim=0)
    #     except:
    #         print(i)
    #         import pdb; pdb.set_trace()

    # # Note that by default the output is already projected (ie multiplied by seq) but it
    # # does not harm to do it again in generate_shap_dict as the seq is one-hot encoded
    # # so the re-projection/multiplication will efficetively be no-op
    # attr = deep_lift_shap(model, seq, batch_size=batch_size, n_shuffles=n_shuffles, verbose=True, args=args,
    #     additional_nonlinear_ops={
    #         _ProfileLogitScaling: _nonlinear,
    #         _Log: _nonlinear,
    #         _Exp: _nonlinear
    #     },
    #     warning_threshold=1e8, device=device
    # )
    # del model

    # shap_dict = generate_shap_dict(seq, attr)

    # print('Saving shap dict in h5 format')
    # # np.object = object <- valeh: why?
    # import deepdish
    # deepdish.io.save(os.path.join(out_dir, f'shap.h5'), shap_dict, compression='blosc')
    # np.save(os.path.join(out_dir, 'attr.npy'), attr)
    # np.save(os.path.join(out_dir, 'ohe.npy'), seq)

    # print('Saving peak regions in bed format')
    # regions_df.to_csv(os.path.join(out_dir, 'peaks.bed'), sep='\t', header=False, index=False)
    # print('Sorting and indexing peak bed file')
    # os.system("sort -k1,1 -k2,2n {} > {}".format(os.path.join(out_dir, 'peaks.bed'), os.path.join(out_dir, 'peaks.sorted.bed')))
    # os.system("bgzip -c {} > {}".format(os.path.join(out_dir, 'peaks.sorted.bed'), os.path.join(out_dir, 'peaks.bed.gz')))
    # os.system("tabix -p bed {}".format(os.path.join(out_dir, 'peaks.bed.gz')))

    # print('Converting shap h5 to bigwig')
    # hdf5_to_bigwig(
    #     os.path.join(out_dir, f'shap.h5'),
    #     os.path.join(out_dir, 'peaks.bed'),
    #     chrom_sizes,
    #     output_prefix=os.path.join(out_dir, f'shap'),
    #     debug_chr=None,
    #     tqdm=True
    # )

    t1 = "/large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_chrombpnet_tuto/interpretation/instance-20260118_102238/fold_0/counts/ohe.npy"
    t2 = "/large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_chrombpnet_tuto/interpretation/instance-20260118_102238/fold_0/counts/attr.npy"
    print('Running modisco')
    modisco_path = "/home/valehvpa/.local/bin/modisco"
    os.system('{} motifs -s {} -a {} -n {} -w {} -o {}'.format(
        modisco_path,
        t1,
        t2,
        # os.path.join(out_dir, 'ohe.npy'),  
        # os.path.join(out_dir, 'attr.npy'),
        max_seqlets,
        width, 
        os.path.join(out_dir, f'modisco.h5')
    ))

    if meme_file is None:
        from histobpnet.data_loader.genome import motifs_datasets
        meme_file = motifs_datasets().fetch("motifs.meme.txt")

    print('Generating modisco report')
    os.system('{} report -i {} -o {} -m {}'.format(
        modisco_path,
        os.path.join(out_dir, 'modisco.h5'),
        os.path.join(out_dir, 'modisco_report'),
        meme_file
    ))

    print('Converting modisco html report to pdf')
    html_to_pdf(
        os.path.join(out_dir, 'modisco_report/motifs.html'),
        os.path.join(out_dir, 'modisco_report.pdf')
    )

def generate_shap_dict(seqs, scores):
    if isinstance(seqs, torch.Tensor):
        seqs = seqs.cpu().numpy()
    if isinstance(scores, torch.Tensor):
        scores = scores.cpu().numpy()

    assert(seqs.shape==scores.shape)
    assert(seqs.shape[1]==4) # one hot encoding, which has been transposed

    # construct a dictionary for the raw shap scores and the
    # the projected shap scores
    # MODISCO workflow expects one hot sequences with shape (None,4,inputlen)
    d = {
        'raw': {'seq': seqs.astype(np.int8)},
        'shap': {'seq': scores.astype(np.float16)},
        'projected_shap': {'seq': (seqs*scores).astype(np.float16)}
    }

    return d
