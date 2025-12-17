python main.py train \
--out_dir /large_storage/goodarzilab/valehvpa/projects/scCisTrans/for_chrombpnet_tuto/training \
--fasta /large_storage/goodarzilab/valehvpa/refs/hg38/GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta \
--chrom_sizes /large_storage/goodarzilab/valehvpa/refs/hg38/hg38.chrom.sizes \
--peaks /large_storage/goodarzilab/valehvpa/projects/scCisTrans/for_chrombpnet_tuto/peaks_no_blacklist.bed \
--negatives /large_storage/goodarzilab/valehvpa/projects/scCisTrans/for_chrombpnet_tuto/negatives/lei_negatives.bed \
--fold_path /large_storage/goodarzilab/valehvpa/projects/scCisTrans/for_chrombpnet_tuto/splits/instance-20250727_104312/fold_0.json \
--max_epochs 2 \
--bias_scaled /large_storage/goodarzilab/valehvpa/projects/scCisTrans/for_chrombpnet_tuto/bias_model/ENCSR868FGK_bias_fold_0.h5 \
--gpu 0 1 2 3

# background ??

########################################

python /home/valehvpa/projects/scCisTrans/histobpnet/scripts/train/main.py \
    --command train \
    --name histobpnet_v2-finetune-0.2-skip_missing_hist \
    --model_type histobpnet_v2 \
    --output_dir /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_hist/histobpnet_v2/train \
    --fasta /large_storage/goodarzilab/valehvpa/refs/hg38/GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta \
    --chrom_sizes /large_storage/goodarzilab/valehvpa/refs/hg38/hg38.chrom.sizes \
    --peaks /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_hist/gm12878_atac_encode/chrombpnet_data/peaks.all_input_regions.ENCSR868FGK.bed.gz \
    --negatives /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_hist/gm12878_atac_encode/chrombpnet_data/fold_0/nonpeaks.all_input_regions.fold_0.ENCSR868FGK.bed.gz \
    --bigwig /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_borzoi/histone/peak_scramble/maya_coverage/all_reps.bw \
    --bigwig_ctrl /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_borzoi/histone/peak_scramble/maya_coverage/inputc/inputc.bw \
    --fold 0 \
    --gpu 0 \
    --cvd 2 \
    --num_workers 16 \
    --atac_hgp_map /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_hist/histobpnet_v2/data/reverse_extended_gapped_peak_histone_intersect_2114bp_atac_for_train.bed \
    --out_window 0 \
    --shift 0 \
    --rc_frac 0 \
    --negative_sampling_ratio -1 \
    --ctrl_scaling_factor 0.2 \
    --skip_missing_hist \
    --chrombpnet_wo_bias /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_hist/gm12878_atac_encode/chrombpnet_pretrained/fold_0/model.chrombpnet_nobias.fold_0.ENCSR637XSC.h5

########################################

python /home/valehvpa/projects/scCisTrans/histobpnet/scripts/train/main.py \
    --command train \
    --name histobpnet_v2-scratch-0.2-skip_missing_hist \
    --model_type histobpnet_v2 \
    --output_dir /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_hist/histobpnet_v2/train \
    --fasta /large_storage/goodarzilab/valehvpa/refs/hg38/GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta \
    --chrom_sizes /large_storage/goodarzilab/valehvpa/refs/hg38/hg38.chrom.sizes \
    --peaks /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_hist/gm12878_atac_encode/chrombpnet_data/peaks.all_input_regions.ENCSR868FGK.bed.gz \
    --negatives /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_hist/gm12878_atac_encode/chrombpnet_data/fold_0/nonpeaks.all_input_regions.fold_0.ENCSR868FGK.bed.gz \
    --bigwig /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_borzoi/histone/peak_scramble/maya_coverage/all_reps.bw \
    --bigwig_ctrl /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_borzoi/histone/peak_scramble/maya_coverage/inputc/inputc.bw \
    --fold 0 \
    --gpu 0 \
    --cvd 2 \
    --num_workers 16 \
    --atac_hgp_map /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_hist/histobpnet_v2/data/reverse_extended_gapped_peak_histone_intersect_2114bp_atac_for_train.bed \
    --out_window 0 \
    --shift 0 \
    --rc_frac 0 \
    --negative_sampling_ratio -1 \
    --ctrl_scaling_factor 0.2 \
    --skip_missing_hist

########################################

python /home/valehvpa/projects/scCisTrans/histobpnet/scripts/train/main.py \
    --command train \
    --name histobpnet_v2-finetune-5.0-skip_missing_hist \
    --model_type histobpnet_v2 \
    --output_dir /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_hist/histobpnet_v2/train \
    --fasta /large_storage/goodarzilab/valehvpa/refs/hg38/GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta \
    --chrom_sizes /large_storage/goodarzilab/valehvpa/refs/hg38/hg38.chrom.sizes \
    --peaks /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_hist/gm12878_atac_encode/chrombpnet_data/peaks.all_input_regions.ENCSR868FGK.bed.gz \
    --negatives /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_hist/gm12878_atac_encode/chrombpnet_data/fold_0/nonpeaks.all_input_regions.fold_0.ENCSR868FGK.bed.gz \
    --bigwig /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_borzoi/histone/peak_scramble/maya_coverage/all_reps.bw \
    --bigwig_ctrl /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_borzoi/histone/peak_scramble/maya_coverage/inputc/inputc.bw \
    --fold 0 \
    --gpu 0 \
    --cvd 2 \
    --num_workers 16 \
    --atac_hgp_map /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_hist/histobpnet_v2/data/reverse_extended_gapped_peak_histone_intersect_2114bp_atac_for_train.bed \
    --out_window 0 \
    --shift 0 \
    --rc_frac 0 \
    --negative_sampling_ratio -1 \
    --ctrl_scaling_factor 5.0 \
    --skip_missing_hist \
    --chrombpnet_wo_bias /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_hist/gm12878_atac_encode/chrombpnet_pretrained/fold_0/model.chrombpnet_nobias.fold_0.ENCSR637XSC.h5

########################################

python /home/valehvpa/projects/scCisTrans/histobpnet/scripts/train/main.py \
    --command train \
    --name histobpnet_v2-scratch-5.0-skip_missing_hist \
    --model_type histobpnet_v2 \
    --output_dir /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_hist/histobpnet_v2/train \
    --fasta /large_storage/goodarzilab/valehvpa/refs/hg38/GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta \
    --chrom_sizes /large_storage/goodarzilab/valehvpa/refs/hg38/hg38.chrom.sizes \
    --peaks /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_hist/gm12878_atac_encode/chrombpnet_data/peaks.all_input_regions.ENCSR868FGK.bed.gz \
    --negatives /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_hist/gm12878_atac_encode/chrombpnet_data/fold_0/nonpeaks.all_input_regions.fold_0.ENCSR868FGK.bed.gz \
    --bigwig /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_borzoi/histone/peak_scramble/maya_coverage/all_reps.bw \
    --bigwig_ctrl /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_borzoi/histone/peak_scramble/maya_coverage/inputc/inputc.bw \
    --fold 0 \
    --gpu 0 \
    --cvd 2 \
    --num_workers 16 \
    --atac_hgp_map /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_hist/histobpnet_v2/data/reverse_extended_gapped_peak_histone_intersect_2114bp_atac_for_train.bed \
    --out_window 0 \
    --shift 0 \
    --rc_frac 0 \
    --negative_sampling_ratio -1 \
    --ctrl_scaling_factor 5.0 \
    --skip_missing_hist

########################################

python /home/valehvpa/projects/scCisTrans/histobpnet/scripts/train/main.py \
    --command train \
    --name hv2-finetune-1.0-no_skip_missing_hist \
    --model_type histobpnet_v2 \
    --output_dir /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_hist/histobpnet_v2/train \
    --fasta /large_storage/goodarzilab/valehvpa/refs/hg38/GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta \
    --chrom_sizes /large_storage/goodarzilab/valehvpa/refs/hg38/hg38.chrom.sizes \
    --peaks /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_hist/gm12878_atac_encode/chrombpnet_data/peaks.all_input_regions.ENCSR868FGK.bed.gz \
    --negatives /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_hist/gm12878_atac_encode/chrombpnet_data/fold_0/nonpeaks.all_input_regions.fold_0.ENCSR868FGK.bed.gz \
    --bigwig /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_borzoi/histone/peak_scramble/maya_coverage/all_reps.bw \
    --bigwig_ctrl /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_borzoi/histone/peak_scramble/maya_coverage/inputc/inputc.bw \
    --fold 0 \
    --gpu 0 \
    --cvd 1 \
    --num_workers 16 \
    --atac_hgp_map /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_hist/histobpnet_v2/data/reverse_extended_gapped_peak_histone_intersect_2114bp_atac_for_train.bed \
    --out_window 0 \
    --shift 0 \
    --rc_frac 0 \
    --negative_sampling_ratio -1 \
    --ctrl_scaling_factor 1.0 \
    --chrombpnet_wo_bias /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_hist/gm12878_atac_encode/chrombpnet_pretrained/fold_0/model.chrombpnet_nobias.fold_0.ENCSR637XSC.h5

########################################

python /home/valehvpa/projects/scCisTrans/histobpnet/scripts/train/main.py \
    --command train \
    --name hv2-finetune-1.0-skip_missing_hist-olneg1k-nsr0.1 \
    --model_type histobpnet_v2 \
    --output_dir /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_hist/histobpnet_v2/train \
    --fasta /large_storage/goodarzilab/valehvpa/refs/hg38/GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta \
    --chrom_sizes /large_storage/goodarzilab/valehvpa/refs/hg38/hg38.chrom.sizes \
    --peaks /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_hist/gm12878_atac_encode/chrombpnet_data/peaks.all_input_regions.ENCSR868FGK.bed.gz \
    --negatives /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_hist/gm12878_atac_encode/chrombpnet_data/fold_0/nonpeaks.all_input_regions.fold_0.ENCSR868FGK.bed.gz \
    --bigwig /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_borzoi/histone/peak_scramble/maya_coverage/all_reps.bw \
    --bigwig_ctrl /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_borzoi/histone/peak_scramble/maya_coverage/inputc/inputc.bw \
    --fold 0 \
    --gpu 0 \
    --cvd 0 \
    --num_workers 16 \
    --atac_hgp_map /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_hist/histobpnet_v2/data/reverse_extended_gapped_peak_histone_intersect_2114bp_atac_for_train.bed \
    --out_window 0 \
    --shift 0 \
    --rc_frac 0 \
    --negative_sampling_ratio 0.1 \
    --ctrl_scaling_factor 1.0 \
    --chrombpnet_wo_bias /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_hist/gm12878_atac_encode/chrombpnet_pretrained/fold_0/model.chrombpnet_nobias.fold_0.ENCSR637XSC.h5 \
    --skip_missing_hist \
    --outputlen_neg 1000

########################################

python /home/valehvpa/projects/scCisTrans/histobpnet/scripts/train/main.py \
    --command train \
    --name hv2-scratch-1.0-skip_missing_hist-olneg1k-nsr0.1 \
    --model_type histobpnet_v2 \
    --output_dir /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_hist/histobpnet_v2/train \
    --fasta /large_storage/goodarzilab/valehvpa/refs/hg38/GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta \
    --chrom_sizes /large_storage/goodarzilab/valehvpa/refs/hg38/hg38.chrom.sizes \
    --peaks /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_hist/gm12878_atac_encode/chrombpnet_data/peaks.all_input_regions.ENCSR868FGK.bed.gz \
    --negatives /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_hist/gm12878_atac_encode/chrombpnet_data/fold_0/nonpeaks.all_input_regions.fold_0.ENCSR868FGK.bed.gz \
    --bigwig /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_borzoi/histone/peak_scramble/maya_coverage/all_reps.bw \
    --bigwig_ctrl /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_borzoi/histone/peak_scramble/maya_coverage/inputc/inputc.bw \
    --fold 0 \
    --gpu 0 \
    --cvd 0 \
    --num_workers 16 \
    --atac_hgp_map /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_hist/histobpnet_v2/data/reverse_extended_gapped_peak_histone_intersect_2114bp_atac_for_train.bed \
    --out_window 0 \
    --shift 0 \
    --rc_frac 0 \
    --negative_sampling_ratio 0.1 \
    --ctrl_scaling_factor 1.0 \
    --skip_missing_hist \
    --outputlen_neg 1000

########################################

python /home/valehvpa/projects/scCisTrans/histobpnet/scripts/train/main.py \
    --command train \
    --name hv2-finetune-csf1.0-nosmh-olneg1k-nsr0.1 \
    --model_type histobpnet_v2 \
    --output_dir /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_hist/histobpnet_v2/train \
    --fasta /large_storage/goodarzilab/valehvpa/refs/hg38/GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta \
    --chrom_sizes /large_storage/goodarzilab/valehvpa/refs/hg38/hg38.chrom.sizes \
    --peaks /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_hist/gm12878_atac_encode/chrombpnet_data/peaks.all_input_regions.ENCSR868FGK.bed.gz \
    --negatives /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_hist/gm12878_atac_encode/chrombpnet_data/fold_0/nonpeaks.all_input_regions.fold_0.ENCSR868FGK.bed.gz \
    --bigwig /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_borzoi/histone/peak_scramble/maya_coverage/all_reps.bw \
    --bigwig_ctrl /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_borzoi/histone/peak_scramble/maya_coverage/inputc/inputc.bw \
    --fold 0 \
    --gpu 0 \
    --cvd 0 \
    --num_workers 16 \
    --atac_hgp_map /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_hist/histobpnet_v2/data/reverse_extended_gapped_peak_histone_intersect_2114bp_atac_for_train.bed \
    --out_window 0 \
    --shift 0 \
    --rc_frac 0 \
    --negative_sampling_ratio 0.1 \
    --ctrl_scaling_factor 1.0 \
    --chrombpnet_wo_bias /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_hist/gm12878_atac_encode/chrombpnet_pretrained/fold_0/model.chrombpnet_nobias.fold_0.ENCSR637XSC.h5 \
    --outputlen_neg 1000

########################################

python /home/valehvpa/projects/scCisTrans/histobpnet/scripts/train/main.py \
    --command train \
    --name hv3-finetune-csf1.0-olneg1k-nsr0.1-ow1k \
    --model_type histobpnet_v3 \
    --output_dir /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_hist/histobpnet_v3/train \
    --fasta /large_storage/goodarzilab/valehvpa/refs/hg38/GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta \
    --chrom_sizes /large_storage/goodarzilab/valehvpa/refs/hg38/hg38.chrom.sizes \
    --peaks /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_hist/gm12878_atac_encode/chrombpnet_data/peaks.all_input_regions.ENCSR868FGK.bed.gz \
    --negatives /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_hist/gm12878_atac_encode/chrombpnet_data/fold_0/nonpeaks.all_input_regions.fold_0.ENCSR868FGK.bed.gz \
    --bigwig /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_borzoi/histone/peak_scramble/maya_coverage/all_reps.bw \
    --bigwig_ctrl /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_borzoi/histone/peak_scramble/maya_coverage/inputc/inputc.bw \
    --fold 0 \
    --gpu 0 \
    --cvd 1 \
    --num_workers 16 \
    --out_window 1000 \
    --shift 0 \
    --rc_frac 0 \
    --negative_sampling_ratio 0.1 \
    --ctrl_scaling_factor 1.0 \
    --outputlen_neg 1000

########################################

python /home/valehvpa/projects/scCisTrans/histobpnet/scripts/train/main.py \
    --command train \
    --name hv3-finetune-csf1.0-olneg1k-nsr0.1-ow1k \
    --model_type histobpnet_v3 \
    --output_dir /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_hist/histobpnet_v3/train \
    --fasta /large_storage/goodarzilab/valehvpa/refs/hg38/GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta \
    --chrom_sizes /large_storage/goodarzilab/valehvpa/refs/hg38/hg38.chrom.sizes \
    --peaks /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_hist/gm12878_atac_encode/chrombpnet_data/peaks.all_input_regions.ENCSR868FGK.bed.gz \
    --negatives /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_hist/gm12878_atac_encode/chrombpnet_data/fold_0/nonpeaks.all_input_regions.fold_0.ENCSR868FGK.bed.gz \
    --bigwig /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_borzoi/histone/peak_scramble/maya_coverage/all_reps.bw \
    --bigwig_ctrl /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_borzoi/histone/peak_scramble/maya_coverage/inputc/inputc.bw \
    --fold 0 \
    --gpu 0 \
    --cvd 1 \
    --num_workers 16 \
    --out_window 1000 \
    --shift 0 \
    --rc_frac 0 \
    --negative_sampling_ratio 0.1 \
    --ctrl_scaling_factor 1.0 \
    --chrombpnet_wo_bias /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_hist/gm12878_atac_encode/chrombpnet_pretrained/fold_0/model.chrombpnet_nobias.fold_0.ENCSR637XSC.h5 \
    --outputlen_neg 1000

########################################

python /home/valehvpa/projects/scCisTrans/histobpnet/scripts/train/main.py \
    --command train \
    --name hv3-scratch-csf1.0-olneg1k-nsr0.1-ow200 \
    --model_type histobpnet_v3 \
    --output_dir /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_hist/histobpnet_v3/train \
    --fasta /large_storage/goodarzilab/valehvpa/refs/hg38/GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta \
    --chrom_sizes /large_storage/goodarzilab/valehvpa/refs/hg38/hg38.chrom.sizes \
    --peaks /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_hist/gm12878_atac_encode/chrombpnet_data/peaks.all_input_regions.ENCSR868FGK.bed.gz \
    --negatives /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_hist/gm12878_atac_encode/chrombpnet_data/fold_0/nonpeaks.all_input_regions.fold_0.ENCSR868FGK.bed.gz \
    --bigwig /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_borzoi/histone/peak_scramble/maya_coverage/all_reps.bw \
    --bigwig_ctrl /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_borzoi/histone/peak_scramble/maya_coverage/inputc/inputc.bw \
    --fold 0 \
    --gpu 0 \
    --cvd 0 \
    --num_workers 16 \
    --out_window 200 \
    --shift 0 \
    --rc_frac 0 \
    --negative_sampling_ratio 0.1 \
    --ctrl_scaling_factor 1.0 \
    --outputlen_neg 1000

########################################

python /home/valehvpa/projects/scCisTrans/histobpnet/scripts/train/main.py \
    --command train \
    --name hv3-finetune-csf1.0-olneg1k-nsr0.1-ow200 \
    --model_type histobpnet_v3 \
    --output_dir /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_hist/histobpnet_v3/train \
    --fasta /large_storage/goodarzilab/valehvpa/refs/hg38/GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta \
    --chrom_sizes /large_storage/goodarzilab/valehvpa/refs/hg38/hg38.chrom.sizes \
    --peaks /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_hist/gm12878_atac_encode/chrombpnet_data/peaks.all_input_regions.ENCSR868FGK.bed.gz \
    --negatives /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_hist/gm12878_atac_encode/chrombpnet_data/fold_0/nonpeaks.all_input_regions.fold_0.ENCSR868FGK.bed.gz \
    --bigwig /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_borzoi/histone/peak_scramble/maya_coverage/all_reps.bw \
    --bigwig_ctrl /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_borzoi/histone/peak_scramble/maya_coverage/inputc/inputc.bw \
    --fold 0 \
    --gpu 0 \
    --cvd 0 \
    --num_workers 16 \
    --out_window 200 \
    --shift 0 \
    --rc_frac 0 \
    --negative_sampling_ratio 0.1 \
    --ctrl_scaling_factor 1.0 \
    --chrombpnet_wo_bias /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_hist/gm12878_atac_encode/chrombpnet_pretrained/fold_0/model.chrombpnet_nobias.fold_0.ENCSR637XSC.h5 \
    --outputlen_neg 1000

########################################

python /home/valehvpa/projects/scCisTrans/histobpnet/scripts/train/main.py \
    --command train \
    --name hv3-finetune-csf1.0-olneg1k-nsr0.1-ow500 \
    --model_type histobpnet_v3 \
    --output_dir /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_hist/histobpnet_v3/train \
    --fasta /large_storage/goodarzilab/valehvpa/refs/hg38/GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta \
    --chrom_sizes /large_storage/goodarzilab/valehvpa/refs/hg38/hg38.chrom.sizes \
    --peaks /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_hist/gm12878_atac_encode/chrombpnet_data/peaks.all_input_regions.ENCSR868FGK.bed.gz \
    --negatives /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_hist/gm12878_atac_encode/chrombpnet_data/fold_0/nonpeaks.all_input_regions.fold_0.ENCSR868FGK.bed.gz \
    --bigwig /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_borzoi/histone/peak_scramble/maya_coverage/all_reps.bw \
    --bigwig_ctrl /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_borzoi/histone/peak_scramble/maya_coverage/inputc/inputc.bw \
    --fold 0 \
    --gpu 0 \
    --cvd 1 \
    --num_workers 16 \
    --out_window 500 \
    --shift 0 \
    --rc_frac 0 \
    --negative_sampling_ratio 0.1 \
    --ctrl_scaling_factor 1.0 \
    --chrombpnet_wo_bias /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_hist/gm12878_atac_encode/chrombpnet_pretrained/fold_0/model.chrombpnet_nobias.fold_0.ENCSR637XSC.h5 \
    --outputlen_neg 1000

########################################

python /home/valehvpa/projects/scCisTrans/histobpnet/scripts/train/main.py \
    --command train \
    --name hv3-finetune-csf1.0-olneg1k-nsr0.1-ow5000 \
    --model_type histobpnet_v3 \
    --output_dir /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_hist/histobpnet_v3/train \
    --fasta /large_storage/goodarzilab/valehvpa/refs/hg38/GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta \
    --chrom_sizes /large_storage/goodarzilab/valehvpa/refs/hg38/hg38.chrom.sizes \
    --peaks /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_hist/gm12878_atac_encode/chrombpnet_data/peaks.all_input_regions.ENCSR868FGK.bed.gz \
    --negatives /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_hist/gm12878_atac_encode/chrombpnet_data/fold_0/nonpeaks.all_input_regions.fold_0.ENCSR868FGK.bed.gz \
    --bigwig /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_borzoi/histone/peak_scramble/maya_coverage/all_reps.bw \
    --bigwig_ctrl /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_borzoi/histone/peak_scramble/maya_coverage/inputc/inputc.bw \
    --fold 0 \
    --gpu 0 \
    --cvd 1 \
    --num_workers 16 \
    --out_window 5000 \
    --shift 0 \
    --rc_frac 0 \
    --negative_sampling_ratio 0.1 \
    --ctrl_scaling_factor 1.0 \
    --chrombpnet_wo_bias /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_hist/gm12878_atac_encode/chrombpnet_pretrained/fold_0/model.chrombpnet_nobias.fold_0.ENCSR637XSC.h5 \
    --outputlen_neg 1000

########################################

# COMAPRE LEI S TO MINE
python /home/valehvpa/projects/scCisTrans/histobpnet/scripts/train/main.py \
    --command train \
    --name chrombpnet_train_k562_lei_compare_seed_ev \
    --model_type chrombpnet \
    --output_dir /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_chrombpnet_tuto/training \
    --fasta /large_storage/goodarzilab/valehvpa/refs/hg38/GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta \
    --chrom_sizes /large_storage/goodarzilab/valehvpa/refs/hg38/hg38.chrom.sizes \
    --peaks /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_chrombpnet_tuto/lei_peaks_ss_40000.bed \
    --negatives /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_chrombpnet_tuto/negatives/lei_negatives_ss_40000.bed \
    --bigwig /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_chrombpnet_tuto/lei_unstranded.bw \
    --bias_scaled /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_chrombpnet_tuto/pretrained/ENCSR467RSV/fold_0/model.bias_scaled.fold_0.ENCSR868FGK.h5 \
    --adjust_bias \
    --fold 0 \
    --gpu 0 \
    --cvd 1 \
    --num_workers 8

# chrombpnet train --peaks /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_chrombpnet_tuto/lei_peaks_ss_40000.bed --negatives /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_chrombpnet_tuto/negatives/lei_negatives_ss_40000.bed --bigwig /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_chrombpnet_tuto/lei_unstranded.bw --bias /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_chrombpnet_tuto/pretrained/ENCSR467RSV/fold_0/model.bias_scaled.fold_0.ENCSR868FGK.h5 --fasta /large_storage/goodarzilab/valehvpa/refs/hg38/GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta --adjust_bias --num_workers 8

########################################
