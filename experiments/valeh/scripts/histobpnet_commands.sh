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

