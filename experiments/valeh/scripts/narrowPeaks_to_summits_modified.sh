#!/bin/sh

set -euo pipefail

dir=/large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_hist/gm12878_atac_encode/data
out_dir=/large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_hist/histobpnet_v2/data
file_name=peaks.all_input_regions.ENCSR868FGK

cd "$out_dir"

for X in 2114; do
    half=$((X / 2))
    out="${file_name}.summits.${X}bp.sorted.bed"

    # skip if compressed file already exists
    if [ -f "${out}.gz" ]; then
        echo "Skipping ${out} (already exists)"
        continue
    fi

    echo "Creating ${out}..."
    zcat "${dir}/${file_name}.bed.gz" \
        | awk -v h="$half" '{
            s = $2 + $10   # summit pos
            start = s - h
            end   = s + h
            print $1, start, end, $4, $5, $6
        }' OFS="\t" \
        | sort -k1,1V -k2,2n \
        > "${out}"

    echo "bgzipping ${out}..."
    bgzip "${out}"

    echo "indexing ${out}.gz with tabix..."
    tabix -p bed "${out}.gz"
done