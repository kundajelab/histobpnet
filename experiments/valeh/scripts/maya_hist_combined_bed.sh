#!/bin/sh

set -euo pipefail

t1=/large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_hist/maya_2013/align/rep1/rep1.bed
t2=/large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_hist/maya_2013/align/rep2/rep2.bed
t3=/large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_hist/maya_2013/align/rep3/rep3.bed
t4=/large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_hist/maya_2013/align/rep4/rep4.bed.gz
chrsz=/large_storage/goodarzilab/valehvpa/refs/hg38/hg38.chrom.sizes
outfile=/large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_hist/maya_2013/align/all_reps.sorted.bed

(cat "$t1" "$t2" "$t3"; zcat "$t4") | bedtools sort -faidx "$chrsz" -i - > "$outfile"

echo "bgziping $outfile..."
bgzip $outfile

echo "Indexing $outfile.gz with tabix..."
tabix -p bed $outfile.gz