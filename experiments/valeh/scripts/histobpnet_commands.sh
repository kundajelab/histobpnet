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
    --peaks /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_chrombpnet_tuto/lei_peaks_ss_40000.bed \
    --negatives /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_chrombpnet_tuto/negatives/lei_negatives_ss_40000.bed \
    --bigwig /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_chrombpnet_tuto/lei_unstranded.bw \
    --bias_scaled /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_chrombpnet_tuto/pretrained/ENCSR467RSV/fold_0/model.bias_scaled.fold_0.ENCSR868FGK.h5 \
    --adjust_bias \
    --gpu 0 \
    --cvd 1 \
    --num_workers 8

# chrombpnet train --peaks /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_chrombpnet_tuto/lei_peaks_ss_40000.bed --negatives /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_chrombpnet_tuto/negatives/lei_negatives_ss_40000.bed --bigwig /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_chrombpnet_tuto/lei_unstranded.bw --bias /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_chrombpnet_tuto/pretrained/ENCSR467RSV/fold_0/model.bias_scaled.fold_0.ENCSR868FGK.h5 --fasta /large_storage/goodarzilab/valehvpa/refs/hg38/GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta --adjust_bias --num_workers 8 --out_dir /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_chrombpnet_tuto/lei_training

########################################

# TEST RUN HISTOBPNET V3
python /home/valehvpa/projects/scCisTrans/histobpnet/scripts/train/main.py \
    --command train \
    --name hv3-test \
    --model_type histobpnet_v3 \
    --output_dir /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_hist/histobpnet_v3/train \
    --peaks /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_hist/gm12878_atac_encode/chrombpnet_data/peaks.all_input_regions.ENCSR868FGK_40000.bed \
    --negatives /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_hist/gm12878_atac_encode/chrombpnet_data/fold_0/nonpeaks.all_input_regions.fold_0.ENCSR868FGK_40000.bed \
    --bigwig /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_borzoi/histone/peak_scramble/maya_coverage/all_reps.bw \
    --bigwig_ctrl /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_borzoi/histone/peak_scramble/maya_coverage/inputc/inputc.bw \
    --gpu 0 \
    --cvd 1 \
    --out_window 5000 \
    --ctrl_scaling_factor 1.0 \
    --chrombpnet_wo_bias /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_hist/gm12878_atac_encode/chrombpnet_pretrained/fold_0/model.chrombpnet_nobias.fold_0.ENCSR637XSC.h5 \
    --skip_wandb \
    --max_epochs 1

########################################

# PREDICT WITH CHROMBPNET MODEL
python /home/valehvpa/projects/scCisTrans/histobpnet/scripts/train/main.py \
    --command predict \
    --name test_log_predict_wandb \
    --model_type chrombpnet \
    --output_dir /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_chrombpnet_tuto/prediction \
    --peaks /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_chrombpnet_tuto/lei_peaks_ss_40000.bed \
    --negatives /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_chrombpnet_tuto/negatives/lei_negatives_ss_40000.bed \
    --bigwig /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_chrombpnet_tuto/lei_unstranded.bw \
    --checkpoint /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_chrombpnet_tuto/training/instance-20251217_152811/pt_artifacts/best_model.ckpt \
    --gpu 0 \
    --cvd 1

python /home/valehvpa/projects/scCisTrans/histobpnet/scripts/train/main.py \
    --command train \
    --name test_log_predict_wandb \
    --model_type chrombpnet \
    --output_dir /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_chrombpnet_tuto/training \
    --peaks /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_chrombpnet_tuto/lei_peaks_ss_40000.bed \
    --negatives /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_chrombpnet_tuto/negatives/lei_negatives_ss_40000.bed \
    --bigwig /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_chrombpnet_tuto/lei_unstranded.bw \
    --bias_scaled /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_chrombpnet_tuto/pretrained/ENCSR467RSV/fold_0/model.bias_scaled.fold_0.ENCSR868FGK.h5 \
    --adjust_bias \
    --gpu 0 \
    --cvd 1 \
    --num_workers 8 \
    --max_epochs 1

########################################

# DIFFERENT Output lenghts for hv3, same olneg window

# 6k, 7k, 8k, 1k, 2k, 3k, 4k, 200, 500
# python /home/valehvpa/projects/scCisTrans/histobpnet/scripts/train/main.py \
#     --command train \
#     --name hv3-finetune-csf1.0-ow6k \
#     --model_type histobpnet_v3 \
#     --output_dir /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_hist/histobpnet_v3/train \
#     --peaks /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_hist/gm12878_atac_encode/chrombpnet_data/peaks.all_input_regions.ENCSR868FGK.bed.gz \
#     --negatives /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_hist/gm12878_atac_encode/chrombpnet_data/fold_0/nonpeaks.all_input_regions.fold_0.ENCSR868FGK.bed.gz \
#     --bigwig /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_borzoi/histone/peak_scramble/maya_coverage/all_reps.bw \
#     --bigwig_ctrl /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_borzoi/histone/peak_scramble/maya_coverage/inputc/inputc.bw \
#     --gpu 0 \
#     --cvd 0 \
#     --out_window 6000 \
#     --ctrl_scaling_factor 1.0 \
#     --chrombpnet_wo_bias /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_hist/gm12878_atac_encode/chrombpnet_pretrained/fold_0/model.chrombpnet_nobias.fold_0.ENCSR637XSC.h5

########################################

# DIFFERENT Output lenghts for hv3, same olneg window, POST ENCODE FIX

# 200, 500, 1k, 2k, 3k, 4k, 5k, 6k, 7k, 8k
python /home/valehvpa/projects/scCisTrans/histobpnet/scripts/train/main.py \
    --command train \
    --name hv3-finetune-encfix-csf1.0-ow8k \
    --model_type histobpnet_v3 \
    --output_dir /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_hist/histobpnet_v3/train \
    --peaks /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_hist/gm12878_atac_encode/ENCFF971WEQ/peaks.all_input_regions.ENCSR637XSC.bed.gz \
    --negatives /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_hist/gm12878_atac_encode/ENCFF971WEQ/fold_0/nonpeaks.all_input_regions.fold_0.ENCSR637XSC.bed.gz \
    --bigwig /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_borzoi/histone/peak_scramble/maya_coverage/all_reps.bw \
    --bigwig_ctrl /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_borzoi/histone/peak_scramble/maya_coverage/inputc/inputc.bw \
    --cvd 1 \
    --out_window 8000 \
    --ctrl_scaling_factor 1.0 \
    --chrombpnet_wo_bias /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_hist/gm12878_atac_encode/ENCFF142IOR/fold_0/model.chrombpnet_nobias.fold_0.ENCSR637XSC.h5

########################################

# HV2 POST ENCODE FIX

python /home/valehvpa/projects/scCisTrans/histobpnet/scripts/train/main.py \
    --command train \
    --name hv2-finetune-encfix-csf1.0-closest-olneg1k \
    --model_type histobpnet_v2 \
    --output_dir /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_hist/histobpnet_v2/train \
    --peaks /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_hist/gm12878_atac_encode/ENCFF971WEQ/peaks.all_input_regions.ENCSR637XSC.bed.gz \
    --negatives /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_hist/gm12878_atac_encode/ENCFF971WEQ/fold_0/nonpeaks.all_input_regions.fold_0.ENCSR637XSC.bed.gz \
    --bigwig /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_borzoi/histone/peak_scramble/maya_coverage/all_reps.bw \
    --bigwig_ctrl /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_borzoi/histone/peak_scramble/maya_coverage/inputc/inputc.bw \
    --cvd 0 \
    --atac_hgp_map /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_hist/histobpnet_v2/data/reverse_gapped_peak_histone_closest_2114bp_atac.bed \
    --ctrl_scaling_factor 1.0 \
    --chrombpnet_wo_bias /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_hist/gm12878_atac_encode/ENCFF142IOR/fold_0/model.chrombpnet_nobias.fold_0.ENCSR637XSC.h5 \
    --outputlen_neg 1000

########################################

python /home/valehvpa/projects/scCisTrans/histobpnet/scripts/train/main.py \
    --command train \
    --name hv2-finetune-encfix-csf1.0-closest-olneg1k-skip_oob_hgp \
    --model_type histobpnet_v2 \
    --output_dir /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_hist/histobpnet_v2/train \
    --peaks /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_hist/gm12878_atac_encode/ENCFF971WEQ/peaks.all_input_regions.ENCSR637XSC.bed.gz \
    --negatives /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_hist/gm12878_atac_encode/ENCFF971WEQ/fold_0/nonpeaks.all_input_regions.fold_0.ENCSR637XSC.bed.gz \
    --bigwig /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_borzoi/histone/peak_scramble/maya_coverage/all_reps.bw \
    --bigwig_ctrl /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_borzoi/histone/peak_scramble/maya_coverage/inputc/inputc.bw \
    --cvd 0 \
    --atac_hgp_map /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_hist/histobpnet_v2/data/reverse_gapped_peak_histone_closest_2114bp_atac_within_1k_bound.bed \
    --ctrl_scaling_factor 1.0 \
    --chrombpnet_wo_bias /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_hist/gm12878_atac_encode/ENCFF142IOR/fold_0/model.chrombpnet_nobias.fold_0.ENCSR637XSC.h5 \
    --outputlen_neg 1000 \
    --skip_missing_hist Yes

########################################

# v2 w/ use_linear_w_ctrl
python /home/valehvpa/projects/scCisTrans/histobpnet/scripts/train/main.py \
    --command train \
    --name hv2-finetune-encfix-csf1.0-closest-olneg1k-skip_oob_hgp \
    --model_type histobpnet_v2 \
    --output_dir /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_hist/histobpnet_v2/train \
    --peaks /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_hist/gm12878_atac_encode/ENCFF971WEQ/peaks.all_input_regions.ENCSR637XSC.bed.gz \
    --negatives /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_hist/gm12878_atac_encode/ENCFF971WEQ/fold_0/nonpeaks.all_input_regions.fold_0.ENCSR637XSC.bed.gz \
    --bigwig /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_borzoi/histone/peak_scramble/maya_coverage/all_reps.bw \
    --bigwig_ctrl /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_borzoi/histone/peak_scramble/maya_coverage/inputc/inputc.bw \
    --cvd 0 \
    --atac_hgp_map /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_hist/histobpnet_v2/data/reverse_gapped_peak_histone_closest_2114bp_atac_within_1k_bound.bed \
    --ctrl_scaling_factor 1.0 \
    --chrombpnet_wo_bias /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_hist/gm12878_atac_encode/ENCFF142IOR/fold_0/model.chrombpnet_nobias.fold_0.ENCSR637XSC.h5 \
    --outputlen_neg 1000 \
    --skip_missing_hist Yes \
    --use_linear_w_ctrl

########################################

# v2 w/ pass_zero_mode zero_seq
python /home/valehvpa/projects/scCisTrans/histobpnet/scripts/train/main.py \
    --command predict \
    --name hv2-finetune-csf1.0-closest-olneg1k-skip_oob_hgp-predict \
    --model_type histobpnet_v2 \
    --cvd 0 \
    --output_dir /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_hist/histobpnet_v2/predict \
    --checkpoint /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_hist/histobpnet_v2/train/instance-20260106_172930/pt_artifacts/best_model.ckpt \
    --peaks /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_hist/gm12878_atac_encode/ENCFF971WEQ/peaks.all_input_regions.ENCSR637XSC.bed.gz \
    --negatives /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_hist/gm12878_atac_encode/ENCFF971WEQ/fold_0/nonpeaks.all_input_regions.fold_0.ENCSR637XSC.bed.gz \
    --bigwig /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_borzoi/histone/peak_scramble/maya_coverage/all_reps.bw \
    --bigwig_ctrl /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_borzoi/histone/peak_scramble/maya_coverage/inputc/inputc.bw \
    --atac_hgp_map /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_hist/histobpnet_v2/data/reverse_gapped_peak_histone_closest_2114bp_atac_within_1k_bound.bed \
    --ctrl_scaling_factor 1.0 \
    --outputlen_neg 1000 \
    --skip_missing_hist Yes \
    --use_linear_w_ctrl \
    --pass_zero_mode zero_seq

# v2 w/ pass_zero_mode zero_ctl
python /home/valehvpa/projects/scCisTrans/histobpnet/scripts/train/main.py \
    --command predict \
    --name hv2-finetune-csf1.0-closest-olneg1k-skip_oob_hgp-predict \
    --model_type histobpnet_v2 \
    --cvd 0 \
    --output_dir /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_hist/histobpnet_v2/predict \
    --checkpoint /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_hist/histobpnet_v2/train/instance-20260106_172930/pt_artifacts/best_model.ckpt \
    --peaks /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_hist/gm12878_atac_encode/ENCFF971WEQ/peaks.all_input_regions.ENCSR637XSC.bed.gz \
    --negatives /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_hist/gm12878_atac_encode/ENCFF971WEQ/fold_0/nonpeaks.all_input_regions.fold_0.ENCSR637XSC.bed.gz \
    --bigwig /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_borzoi/histone/peak_scramble/maya_coverage/all_reps.bw \
    --bigwig_ctrl /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_borzoi/histone/peak_scramble/maya_coverage/inputc/inputc.bw \
    --atac_hgp_map /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_hist/histobpnet_v2/data/reverse_gapped_peak_histone_closest_2114bp_atac_within_1k_bound.bed \
    --ctrl_scaling_factor 1.0 \
    --outputlen_neg 1000 \
    --skip_missing_hist Yes \
    --use_linear_w_ctrl \
    --pass_zero_mode zero_ctl

########################################

# v2 w/ dont feed ctrl zero_seq (use_linear_w_ctrl is now True by default to match TF bpnet)
python /home/valehvpa/projects/scCisTrans/histobpnet/scripts/train/main.py \
    --command train \
    --name hv2-finetune-csf1.0-closest-olneg1k-skip_oob_hgp-no_feed_ctrl \
    --model_type histobpnet_v2 \
    --cvd 0 \
    --output_dir /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_hist/histobpnet_v2/train \
    --peaks /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_hist/gm12878_atac_encode/ENCFF971WEQ/peaks.all_input_regions.ENCSR637XSC.bed.gz \
    --negatives /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_hist/gm12878_atac_encode/ENCFF971WEQ/fold_0/nonpeaks.all_input_regions.fold_0.ENCSR637XSC.bed.gz \
    --bigwig /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_borzoi/histone/peak_scramble/maya_coverage/all_reps.bw \
    --bigwig_ctrl /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_borzoi/histone/peak_scramble/maya_coverage/inputc/inputc.bw \
    --atac_hgp_map /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_hist/histobpnet_v2/data/reverse_gapped_peak_histone_closest_2114bp_atac_within_1k_bound.bed \
    --chrombpnet_wo_bias /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_hist/gm12878_atac_encode/ENCFF142IOR/fold_0/model.chrombpnet_nobias.fold_0.ENCSR637XSC.h5 \
    --ctrl_scaling_factor 1.0 \
    --outputlen_neg 1000 \
    --skip_missing_hist Yes \
    --feed_ctrl False

python /home/valehvpa/projects/scCisTrans/histobpnet/scripts/train/main.py \
    --command train \
    --name hv2-scratch-csf1.0-closest-olneg1k-skip_oob_hgp-no_feed_ctrl \
    --model_type histobpnet_v2 \
    --cvd 0 \
    --output_dir /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_hist/histobpnet_v2/train \
    --peaks /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_hist/gm12878_atac_encode/ENCFF971WEQ/peaks.all_input_regions.ENCSR637XSC.bed.gz \
    --negatives /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_hist/gm12878_atac_encode/ENCFF971WEQ/fold_0/nonpeaks.all_input_regions.fold_0.ENCSR637XSC.bed.gz \
    --bigwig /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_borzoi/histone/peak_scramble/maya_coverage/all_reps.bw \
    --bigwig_ctrl /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_borzoi/histone/peak_scramble/maya_coverage/inputc/inputc.bw \
    --atac_hgp_map /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_hist/histobpnet_v2/data/reverse_gapped_peak_histone_closest_2114bp_atac_within_1k_bound.bed \
    --ctrl_scaling_factor 1.0 \
    --outputlen_neg 1000 \
    --skip_missing_hist Yes \
    --feed_ctrl False

python /home/valehvpa/projects/scCisTrans/histobpnet/scripts/train/main.py \
    --command predict \
    --name hv2-finetune-csf1.0-closest-olneg1k-skip_oob_hgp-predict \
    --model_type histobpnet_v2 \
    --cvd 0 \
    --output_dir /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_hist/histobpnet_v2/predict \
    --checkpoint /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_hist/histobpnet_v2/train/instance-20260108_111226/pt_artifacts/best_model.ckpt \
    --peaks /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_hist/gm12878_atac_encode/ENCFF971WEQ/peaks.all_input_regions.ENCSR637XSC.bed.gz \
    --negatives /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_hist/gm12878_atac_encode/ENCFF971WEQ/fold_0/nonpeaks.all_input_regions.fold_0.ENCSR637XSC.bed.gz \
    --bigwig /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_borzoi/histone/peak_scramble/maya_coverage/all_reps.bw \
    --bigwig_ctrl /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_borzoi/histone/peak_scramble/maya_coverage/inputc/inputc.bw \
    --atac_hgp_map /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_hist/histobpnet_v2/data/reverse_gapped_peak_histone_closest_2114bp_atac_within_1k_bound.bed \
    --ctrl_scaling_factor 1.0 \
    --outputlen_neg 1000 \
    --skip_missing_hist Yes \
    --feed_ctrl False \
    --pass_zero_mode zero_seq

########################################

# v2 with new datasource (5p coverage based)
python /home/valehvpa/projects/scCisTrans/histobpnet/scripts/train/main.py \
    --command train \
    --name hv2-finetune-csf1.0-closest-olneg1k-skip_oob_hgp-5prime \
    --model_type histobpnet_v2 \
    --cvd 0 \
    --output_dir /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_hist/histobpnet_v2/train \
    --peaks /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_hist/gm12878_atac_encode/ENCFF971WEQ/peaks.all_input_regions.ENCSR637XSC.bed.gz \
    --negatives /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_hist/gm12878_atac_encode/ENCFF971WEQ/fold_0/nonpeaks.all_input_regions.fold_0.ENCSR637XSC.bed.gz \
    --bigwig /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_borzoi/histone/peak_scramble/maya_coverage/5prime/chip/all_reps_shifted_5p_pooled.bw \
    --bigwig_ctrl /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_borzoi/histone/peak_scramble/maya_coverage/5prime/inputc/inputc_shifted_5p_pooled.bw \
    --atac_hgp_map /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_hist/histobpnet_v2/data/reverse_gapped_peak_histone_closest_2114bp_atac_within_1k_bound.bed \
    --chrombpnet_wo_bias /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_hist/gm12878_atac_encode/ENCFF142IOR/fold_0/model.chrombpnet_nobias.fold_0.ENCSR637XSC.h5 \
    --ctrl_scaling_factor 1.0 \
    --outputlen_neg 1000 \
    --skip_missing_hist Yes

########################################

# v3 with new datasource and zero ctrl
# 200, 500, 1k, 2k, 3k, 4k, 5k, 6k, 7k, 8k
python /home/valehvpa/projects/scCisTrans/histobpnet/scripts/train/main.py \
    --command train \
    --name hv3-finetune-csf1.0-5p-ow500-zero_ctrl \
    --model_type histobpnet_v3 \
    --cvd 0 \
    --output_dir /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_hist/histobpnet_v3/train \
    --peaks /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_hist/gm12878_atac_encode/ENCFF971WEQ/peaks.all_input_regions.ENCSR637XSC.bed.gz \
    --negatives /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_hist/gm12878_atac_encode/ENCFF971WEQ/fold_0/nonpeaks.all_input_regions.fold_0.ENCSR637XSC.bed.gz \
    --bigwig /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_borzoi/histone/peak_scramble/maya_coverage/5prime/chip/all_reps_5p_pooled.bw \
    --bigwig_ctrl /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_borzoi/histone/peak_scramble/maya_coverage/5prime/inputc/inputc_shifted_5p_pooled.bw \
    --out_window 500 \
    --ctrl_scaling_factor 1.0 \
    --chrombpnet_wo_bias /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_hist/gm12878_atac_encode/ENCFF142IOR/fold_0/model.chrombpnet_nobias.fold_0.ENCSR637XSC.h5 \
    --pass_zero_mode zero_ctl

########################################

# interpert with chrombpnet gm12878 atac
python /home/valehvpa/projects/scCisTrans/histobpnet/scripts/train/main.py \
    --command interpret \
    --name chrombpnet_gm12878_interpret \
    --model_type chrombpnet \
    --cvd 0 \
    --output_dir /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_chrombpnet_tuto/interpretation \
    --peaks /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_hist/gm12878_atac_encode/interpret/ENCFF877CRX_partial/logs.seq_contrib.counts.ENCSR637XSC/logs.seq_contrib.counts.input_regions.modisco.ENCSR637XSC.noLastCol.bed.gz \
    --chrombpnet_wo_bias /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_hist/gm12878_atac_encode/ENCFF142IOR/fold_0/model.chrombpnet_nobias.fold_0.ENCSR637XSC.h5 \
    --skip_wandb \
    --debug

########################################

# hv2 with unweighted_ctrl (5p coverage based)
python /home/valehvpa/projects/scCisTrans/histobpnet/scripts/train/main.py \
    --command train \
    --name hv2.finetune.csf1_56.closest.olneg1k.skip_oob_hgp.5prime.unweighted_ctrl \
    --model_type histobpnet_v2 \
    --cvd 0 \
    --output_dir /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_hist/histobpnet_v2/train \
    --peaks /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_hist/gm12878_atac_encode/ENCFF971WEQ/peaks.all_input_regions.ENCSR637XSC.bed.gz \
    --negatives /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_hist/gm12878_atac_encode/ENCFF971WEQ/fold_0/nonpeaks.all_input_regions.fold_0.ENCSR637XSC.bed.gz \
    --bigwig /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_borzoi/histone/peak_scramble/maya_coverage/GM12878/5prime/chip/all_reps_shifted_5p_pooled.bw \
    --bigwig_ctrl /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_borzoi/histone/peak_scramble/maya_coverage/GM12878/5prime/inputc/inputc_shifted_5p_pooled.bw \
    --atac_hgp_map /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_hist/gm12878_atac_encode/processed/reverse_narrow_peak_histone_closest_2114bp_atac_within_1k_bound.bed \
    --chrombpnet_wo_bias /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_hist/gm12878_atac_encode/ENCFF142IOR/fold_0/model.chrombpnet_nobias.fold_0.ENCSR637XSC.h5 \
    --ctrl_scaling_factor 1.56 \
    --outputlen_neg 1000 \
    --skip_missing_hist Yes \
    --unweighted_ctrl

# same as above but scratch not finetune
python /home/valehvpa/projects/scCisTrans/histobpnet/scripts/train/main.py \
    --command train \
    --name hv2.scratch.csf1_56.closest.olneg1k.skip_oob_hgp.5prime.unweighted_ctrl \
    --model_type histobpnet_v2 \
    --cvd 0 \
    --output_dir /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_hist/histobpnet_v2/train \
    --peaks /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_hist/gm12878_atac_encode/ENCFF971WEQ/peaks.all_input_regions.ENCSR637XSC.bed.gz \
    --negatives /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_hist/gm12878_atac_encode/ENCFF971WEQ/fold_0/nonpeaks.all_input_regions.fold_0.ENCSR637XSC.bed.gz \
    --bigwig /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_borzoi/histone/peak_scramble/maya_coverage/GM12878/5prime/chip/all_reps_shifted_5p_pooled.bw \
    --bigwig_ctrl /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_borzoi/histone/peak_scramble/maya_coverage/GM12878/5prime/inputc/inputc_shifted_5p_pooled.bw \
    --atac_hgp_map /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_hist/gm12878_atac_encode/processed/reverse_narrow_peak_histone_closest_2114bp_atac_within_1k_bound.bed \
    --ctrl_scaling_factor 1.56 \
    --outputlen_neg 1000 \
    --skip_missing_hist Yes \
    --unweighted_ctrl

# v2 same as above but pass_zero_mode zero_seq (sequence ablation)
python /home/valehvpa/projects/scCisTrans/histobpnet/scripts/train/main.py \
    --command predict \
    --name hv2.finetune.csf1_56.closest.olneg1k.skip_oob_hgp.5prime.unweighted_ctrl.predict \
    --model_type histobpnet_v2 \
    --cvd 0 \
    --output_dir /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_hist/histobpnet_v2/predict \
    --checkpoint /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_hist/histobpnet_v2/train/instance-20260121_130331/pt_artifacts/best_model.ckpt \
    --peaks /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_hist/gm12878_atac_encode/ENCFF971WEQ/peaks.all_input_regions.ENCSR637XSC.bed.gz \
    --negatives /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_hist/gm12878_atac_encode/ENCFF971WEQ/fold_0/nonpeaks.all_input_regions.fold_0.ENCSR637XSC.bed.gz \
    --bigwig /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_borzoi/histone/peak_scramble/maya_coverage/GM12878/5prime/chip/all_reps_shifted_5p_pooled.bw \
    --bigwig_ctrl /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_borzoi/histone/peak_scramble/maya_coverage/GM12878/5prime/inputc/inputc_shifted_5p_pooled.bw \
    --atac_hgp_map /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_hist/gm12878_atac_encode/processed/reverse_narrow_peak_histone_closest_2114bp_atac_within_1k_bound.bed \
    --ctrl_scaling_factor 1.56 \
    --outputlen_neg 1000 \
    --skip_missing_hist Yes \
    --unweighted_ctrl \
    --pass_zero_mode zero_seq

# v2 same as above but pass_zero_mode zero_ctl (control ablation)
python /home/valehvpa/projects/scCisTrans/histobpnet/scripts/train/main.py \
    --command predict \
    --name hv2.finetune.csf1_56.closest.olneg1k.skip_oob_hgp.5prime.unweighted_ctrl.predict \
    --model_type histobpnet_v2 \
    --cvd 0 \
    --output_dir /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_hist/histobpnet_v2/predict \
    --checkpoint /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_hist/histobpnet_v2/train/instance-20260121_130331/pt_artifacts/best_model.ckpt \
    --peaks /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_hist/gm12878_atac_encode/ENCFF971WEQ/peaks.all_input_regions.ENCSR637XSC.bed.gz \
    --negatives /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_hist/gm12878_atac_encode/ENCFF971WEQ/fold_0/nonpeaks.all_input_regions.fold_0.ENCSR637XSC.bed.gz \
    --bigwig /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_borzoi/histone/peak_scramble/maya_coverage/GM12878/5prime/chip/all_reps_shifted_5p_pooled.bw \
    --bigwig_ctrl /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_borzoi/histone/peak_scramble/maya_coverage/GM12878/5prime/inputc/inputc_shifted_5p_pooled.bw \
    --atac_hgp_map /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_hist/gm12878_atac_encode/processed/reverse_narrow_peak_histone_closest_2114bp_atac_within_1k_bound.bed \
    --ctrl_scaling_factor 1.56 \
    --outputlen_neg 1000 \
    --skip_missing_hist Yes \
    --unweighted_ctrl \
    --pass_zero_mode zero_ctl

########################################

# v3 with unweighted_ctrl (5p coverage based)
# 200, 500, 1k, 2k, 3k, 4k, 5k, 6k, 7k, 8k
python /home/valehvpa/projects/scCisTrans/histobpnet/scripts/train/main.py \
    --command train \
    --name hv3.finetune.csf1_56.5p.ow5k.unweighted_ctrl \
    --model_type histobpnet_v3 \
    --cvd 0 \
    --output_dir /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_hist/histobpnet_v3/train \
    --peaks /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_hist/gm12878_atac_encode/ENCFF971WEQ/peaks.all_input_regions.ENCSR637XSC.bed.gz \
    --negatives /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_hist/gm12878_atac_encode/ENCFF971WEQ/fold_0/nonpeaks.all_input_regions.fold_0.ENCSR637XSC.bed.gz \
    --bigwig /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_borzoi/histone/peak_scramble/maya_coverage/GM12878/5prime/chip/all_reps_shifted_5p_pooled.bw \
    --bigwig_ctrl /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_borzoi/histone/peak_scramble/maya_coverage/GM12878/5prime/inputc/inputc_shifted_5p_pooled.bw \
    --out_window 5000 \
    --ctrl_scaling_factor 1.56 \
    --chrombpnet_wo_bias /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_hist/gm12878_atac_encode/ENCFF142IOR/fold_0/model.chrombpnet_nobias.fold_0.ENCSR637XSC.h5 \
    --unweighted_ctrl

########################################

# v2 predict but also evaluate on lfc
python /home/valehvpa/projects/scCisTrans/histobpnet/scripts/train/main.py \
    --command predict \
    --name hv2.finetune.csf1_56.closest.olneg1k.skip_oob_hgp.5prime.unweighted_ctrl.predict_lfc \
    --model_type histobpnet_v2 \
    --cvd 1 \
    --output_dir /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_hist/histobpnet_v2/predict \
    --checkpoint /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_hist/histobpnet_v2/train/instance-20260121_130331/pt_artifacts/best_model.ckpt \
    --peaks /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_hist/gm12878_atac_encode/ENCFF971WEQ/peaks.all_input_regions.ENCSR637XSC.bed.gz \
    --negatives /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_hist/gm12878_atac_encode/ENCFF971WEQ/fold_0/nonpeaks.all_input_regions.fold_0.ENCSR637XSC.bed.gz \
    --bigwig /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_borzoi/histone/peak_scramble/maya_coverage/GM12878/5prime/chip/all_reps_shifted_5p_pooled.bw \
    --bigwig_ctrl /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_borzoi/histone/peak_scramble/maya_coverage/GM12878/5prime/inputc/inputc_shifted_5p_pooled.bw \
    --atac_hgp_map /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_hist/gm12878_atac_encode/processed/reverse_narrow_peak_histone_closest_2114bp_atac_within_1k_bound.bed \
    --ctrl_scaling_factor 1.56 \
    --outputlen_neg 1000 \
    --skip_missing_hist Yes \
    --unweighted_ctrl

# same as above for v3
python /home/valehvpa/projects/scCisTrans/histobpnet/scripts/train/main.py \
    --command predict \
    --name hv3.finetune.csf1_56.5p.ow1k.unweighted_ctrl.predict_lfc \
    --model_type histobpnet_v3 \
    --cvd 0 \
    --output_dir /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_hist/histobpnet_v3/predict \
    --checkpoint /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_hist/histobpnet_v3/train/instance-20260121_134835/pt_artifacts/best_model.ckpt \
    --peaks /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_hist/gm12878_atac_encode/ENCFF971WEQ/peaks.all_input_regions.ENCSR637XSC.bed.gz \
    --negatives /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_hist/gm12878_atac_encode/ENCFF971WEQ/fold_0/nonpeaks.all_input_regions.fold_0.ENCSR637XSC.bed.gz \
    --bigwig /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_borzoi/histone/peak_scramble/maya_coverage/GM12878/5prime/chip/all_reps_shifted_5p_pooled.bw \
    --bigwig_ctrl /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_borzoi/histone/peak_scramble/maya_coverage/GM12878/5prime/inputc/inputc_shifted_5p_pooled.bw \
    --atac_hgp_map /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_hist/gm12878_atac_encode/processed/reverse_narrow_peak_histone_closest_2114bp_atac_within_1k_bound.bed \
    --ctrl_scaling_factor 1.56 \
    --out_window 1000 \
    --unweighted_ctrl

########################################

# Literally repeate *all* of the stuff abvoe where I was using reverse_narrow_peak_histone_closest_2114bp_atac_within_1k_bound,
# and swap it for reverse_maya_gapped_peak_histone_closest_2114bp_atac_within_1k_bound.bed as that s the right one to use >:(

# hv2 with unweighted_ctrl (5p coverage based)
python /home/valehvpa/projects/scCisTrans/histobpnet/scripts/train/main.py \
    --command train \
    --name hv2.finetune.csf1_56.closest.olneg1k.skip_oob_hgp.5prime.unweighted_ctrl.fix_maya_peaks \
    --model_type histobpnet_v2 \
    --cvd 0 \
    --output_dir /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_hist/histobpnet_v2/train \
    --peaks /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_hist/gm12878_atac_encode/ENCFF971WEQ/peaks.all_input_regions.ENCSR637XSC.bed.gz \
    --negatives /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_hist/gm12878_atac_encode/ENCFF971WEQ/fold_0/nonpeaks.all_input_regions.fold_0.ENCSR637XSC.bed.gz \
    --bigwig /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_borzoi/histone/peak_scramble/maya_coverage/GM12878/5prime/chip/all_reps_shifted_5p_pooled.bw \
    --bigwig_ctrl /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_borzoi/histone/peak_scramble/maya_coverage/GM12878/5prime/inputc/inputc_shifted_5p_pooled.bw \
    --atac_hgp_map /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_hist/gm12878_atac_encode/processed/reverse_maya_gapped_peak_histone_closest_2114bp_atac_within_1k_bound.bed \
    --chrombpnet_wo_bias /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_hist/gm12878_atac_encode/ENCFF142IOR/fold_0/model.chrombpnet_nobias.fold_0.ENCSR637XSC.h5 \
    --ctrl_scaling_factor 1.56 \
    --outputlen_neg 1000 \
    --skip_missing_hist Yes \
    --unweighted_ctrl

# same as above but scratch not finetune
python /home/valehvpa/projects/scCisTrans/histobpnet/scripts/train/main.py \
    --command train \
    --name hv2.scratch.csf1_56.closest.olneg1k.skip_oob_hgp.5prime.unweighted_ctrl.fix_maya_peaks \
    --model_type histobpnet_v2 \
    --cvd 0 \
    --output_dir /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_hist/histobpnet_v2/train \
    --peaks /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_hist/gm12878_atac_encode/ENCFF971WEQ/peaks.all_input_regions.ENCSR637XSC.bed.gz \
    --negatives /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_hist/gm12878_atac_encode/ENCFF971WEQ/fold_0/nonpeaks.all_input_regions.fold_0.ENCSR637XSC.bed.gz \
    --bigwig /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_borzoi/histone/peak_scramble/maya_coverage/GM12878/5prime/chip/all_reps_shifted_5p_pooled.bw \
    --bigwig_ctrl /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_borzoi/histone/peak_scramble/maya_coverage/GM12878/5prime/inputc/inputc_shifted_5p_pooled.bw \
    --atac_hgp_map /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_hist/gm12878_atac_encode/processed/reverse_maya_gapped_peak_histone_closest_2114bp_atac_within_1k_bound.bed \
    --ctrl_scaling_factor 1.56 \
    --outputlen_neg 1000 \
    --skip_missing_hist Yes \
    --unweighted_ctrl

# v2 same as above but pass_zero_mode zero_seq (sequence ablation)
python /home/valehvpa/projects/scCisTrans/histobpnet/scripts/train/main.py \
    --command predict \
    --name hv2.finetune.csf1_56.closest.olneg1k.skip_oob_hgp.5prime.unweighted_ctrl.fix_maya_peaks.predict \
    --model_type histobpnet_v2 \
    --cvd 0 \
    --output_dir /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_hist/histobpnet_v2/predict \
    --checkpoint /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_hist/histobpnet_v2/train/instance-xxx/pt_artifacts/best_model.ckpt \
    --peaks /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_hist/gm12878_atac_encode/ENCFF971WEQ/peaks.all_input_regions.ENCSR637XSC.bed.gz \
    --negatives /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_hist/gm12878_atac_encode/ENCFF971WEQ/fold_0/nonpeaks.all_input_regions.fold_0.ENCSR637XSC.bed.gz \
    --bigwig /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_borzoi/histone/peak_scramble/maya_coverage/GM12878/5prime/chip/all_reps_shifted_5p_pooled.bw \
    --bigwig_ctrl /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_borzoi/histone/peak_scramble/maya_coverage/GM12878/5prime/inputc/inputc_shifted_5p_pooled.bw \
    --atac_hgp_map /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_hist/gm12878_atac_encode/processed/reverse_maya_gapped_peak_histone_closest_2114bp_atac_within_1k_bound.bed \
    --ctrl_scaling_factor 1.56 \
    --outputlen_neg 1000 \
    --skip_missing_hist Yes \
    --unweighted_ctrl \
    --pass_zero_mode zero_seq

# v2 same as above but pass_zero_mode zero_ctl (control ablation)
python /home/valehvpa/projects/scCisTrans/histobpnet/scripts/train/main.py \
    --command predict \
    --name hv2.finetune.csf1_56.closest.olneg1k.skip_oob_hgp.5prime.unweighted_ctrl.fix_maya_peaks.predict \
    --model_type histobpnet_v2 \
    --cvd 0 \
    --output_dir /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_hist/histobpnet_v2/predict \
    --checkpoint /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_hist/histobpnet_v2/train/instance-xxx/pt_artifacts/best_model.ckpt \
    --peaks /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_hist/gm12878_atac_encode/ENCFF971WEQ/peaks.all_input_regions.ENCSR637XSC.bed.gz \
    --negatives /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_hist/gm12878_atac_encode/ENCFF971WEQ/fold_0/nonpeaks.all_input_regions.fold_0.ENCSR637XSC.bed.gz \
    --bigwig /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_borzoi/histone/peak_scramble/maya_coverage/GM12878/5prime/chip/all_reps_shifted_5p_pooled.bw \
    --bigwig_ctrl /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_borzoi/histone/peak_scramble/maya_coverage/GM12878/5prime/inputc/inputc_shifted_5p_pooled.bw \
    --atac_hgp_map /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_hist/gm12878_atac_encode/processed/reverse_maya_gapped_peak_histone_closest_2114bp_atac_within_1k_bound.bed \
    --ctrl_scaling_factor 1.56 \
    --outputlen_neg 1000 \
    --skip_missing_hist Yes \
    --unweighted_ctrl \
    --pass_zero_mode zero_ctl
    
# v2 predict but also evaluate on lfc
python /home/valehvpa/projects/scCisTrans/histobpnet/scripts/train/main.py \
    --command predict \
    --name hv2.finetune.csf1_56.closest.olneg1k.skip_oob_hgp.5prime.unweighted_ctrl.fix_maya_peaks.predict_lfc \
    --model_type histobpnet_v2 \
    --cvd 1 \
    --output_dir /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_hist/histobpnet_v2/predict \
    --checkpoint /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_hist/histobpnet_v2/train/instance-xxx/pt_artifacts/best_model.ckpt \
    --peaks /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_hist/gm12878_atac_encode/ENCFF971WEQ/peaks.all_input_regions.ENCSR637XSC.bed.gz \
    --negatives /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_hist/gm12878_atac_encode/ENCFF971WEQ/fold_0/nonpeaks.all_input_regions.fold_0.ENCSR637XSC.bed.gz \
    --bigwig /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_borzoi/histone/peak_scramble/maya_coverage/GM12878/5prime/chip/all_reps_shifted_5p_pooled.bw \
    --bigwig_ctrl /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_borzoi/histone/peak_scramble/maya_coverage/GM12878/5prime/inputc/inputc_shifted_5p_pooled.bw \
    --atac_hgp_map /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_hist/gm12878_atac_encode/processed/reverse_maya_gapped_peak_histone_closest_2114bp_atac_within_1k_bound.bed \
    --ctrl_scaling_factor 1.56 \
    --outputlen_neg 1000 \
    --skip_missing_hist Yes \
    --unweighted_ctrl

########################################

# interpert with hv2
python /home/valehvpa/projects/scCisTrans/histobpnet/scripts/train/main.py \
    --command interpret \
    --name hv2_interpret \
    --model_type histobpnet_v2 \
    --cvd 0 \
    --output_dir /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_hist/histobpnet_v2/interpretation \
    --peaks /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_hist/gm12878_atac_encode/ENCFF971WEQ/peaks.all_input_regions.ENCSR637XSC.bed.gz \
    --chrombpnet_wo_bias /large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_hist/gm12878_atac_encode/ENCFF142IOR/fold_0/model.chrombpnet_nobias.fold_0.ENCSR637XSC.h5 \
    --skip_wandb \
    --debug

