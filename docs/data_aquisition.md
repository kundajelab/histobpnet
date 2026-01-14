Data was downloaded for the GM12878 LCL donor sequenced in this paper: "Extensive Variation in Chromatin States Across Humans
" (DOI: 10.1126/science.1242510).
Raw FASTQ_DIR were downloaded from SRA as follows (one replicate shown):
- `prefetch SRR998177 --output-directory /path/to/myfolder`
- `fasterq-dump /path/to/myfolder/SRR998177.sra \
  --outdir /path/to/myfolder \
  --split-files \
  --threads 16`
- `gzip /path/to/myfolder/SRR998177_*.fastq`

Then I concatenated the reads from the 4th replicate which was split into 4 technical replicates (4.1 ie SRR998180, 4.2 ie SRR998181, 4.3 ie SRR998182, 4.4 ie SRR998183):
```bash
FASTQ_DIR=/large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_hist/maya_2013/fastq

TECH_REP1_R1=${FASTQ_DIR}/SRR998180/SRR998180_1.fastq.gz
TECH_REP1_R2=${FASTQ_DIR}/SRR998180/SRR998180_2.fastq.gz

TECH_REP2_R1=${FASTQ_DIR}/SRR998181/SRR998181_1.fastq.gz
TECH_REP2_R2=${FASTQ_DIR}/SRR998181/SRR998181_2.fastq.gz

TECH_REP3_R1=${FASTQ_DIR}/SRR998182/SRR998182_1.fastq.gz
TECH_REP3_R2=${FASTQ_DIR}/SRR998182/SRR998182_2.fastq.gz

TECH_REP4_R1=${FASTQ_DIR}/SRR998183/SRR998183_1.fastq.gz
TECH_REP4_R2=${FASTQ_DIR}/SRR998183/SRR998183_2.fastq.gz

# arrays of R1 and R2 files
R1_LIST=("$TECH_REP1_R1" "$TECH_REP2_R1" "$TECH_REP3_R1" "$TECH_REP4_R1")
R2_LIST=("$TECH_REP1_R2" "$TECH_REP2_R2" "$TECH_REP3_R2" "$TECH_REP4_R2")

mkdir -p "${FASTQ_DIR}/rep4_merged"

# concatenate R1s and R2s separately
zcat "${R1_LIST[@]}" | gzip -c > "${FASTQ_DIR}/rep4_merged/rep4_R1_merged.fastq.gz"
zcat "${R2_LIST[@]}" | gzip -c > "${FASTQ_DIR}/rep4_merged/rep4_R2_merged.fastq.gz"

# quick sanity checks
# extract first 1 million read IDs from R1 and R2 and compare
zcat ${FASTQ_DIR}/rep4_merged/rep4_R1_merged.fastq.gz | awk 'NR%4==1 {print $1}' | head -n 1000000 > r1.ids
zcat ${FASTQ_DIR}/rep4_merged/rep4_R2_merged.fastq.gz | awk 'NR%4==1 {print $1}' | head -n 1000000 > r2.ids

# compare line by line
diff r1.ids r2.ids | head
rm r1.ids r2.ids

# verify equal read counts in R1 and R2
R1_lines=$(gzip -l ${FASTQ_DIR}/rep4_merged/rep4_R1_merged.fastq.gz | awk 'NR==2 {print $2}')
R2_lines=$(gzip -l ${FASTQ_DIR}/rep4_merged/rep4_R2_merged.fastq.gz | awk 'NR==2 {print $2}')
R1_N=$((R1_lines / 4))
R2_N=$((R2_lines / 4))
[[ "$R1_N" -eq "$R2_N" ]] || { echo "read count mismatch: R1=$R1_N R2=$R2_N"; exit 1; }
```

Note that producing an aligned SAM file for the merged replicate 4 OOM-ed so for now we use BED file outputs from
chromap for all replicates. I did however produce the aligned SAM files for the first three replicates.

Then chromap was used to align to the hg38 genome:
```bash
THREADS=16

# choose your reference fasta
REF=/large_storage/goodarzilab/valehvpa/refs/hg38/GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta
ALIGN_DIR=/large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_hist/maya_2013/align
IDX=${ALIGN_DIR}/hg38.chromap.idx

# build the chromap index
chromap -i -r "$REF" -o "$IDX"

# inputs: four replicates, paired R1/R2 FASTQ_DIR (gz ok)
FASTQ_DIR=/large_storage/goodarzilab/valehvpa/data/projects/scCisTrans/for_hist/maya_2013/fastq
REP1_R1=${FASTQ_DIR}/SRR998177/SRR998177_1.fastq.gz
REP1_R2=${FASTQ_DIR}/SRR998177/SRR998177_2.fastq.gz
REP2_R1=${FASTQ_DIR}/SRR998178/SRR998178_1.fastq.gz
REP2_R2=${FASTQ_DIR}/SRR998178/SRR998178_2.fastq.gz
REP3_R1=${FASTQ_DIR}/SRR998179/SRR998179_1.fastq.gz
REP3_R2=${FASTQ_DIR}/SRR998179/SRR998179_2.fastq.gz
REP4_R1=${FASTQ_DIR}/rep4_merged/rep4_R1_merged.fastq.gz
REP4_R2=${FASTQ_DIR}/rep4_merged/rep4_R2_merged.fastq.gz

# OPTION 1: produce SAMs, filter, then merge SAMs. <- I ended up choosing OPTION 2 below instead.
#
# align each replicate with the ChIP preset (-l 2000 --remove-pcr-duplicates --low-mem --BED).
# chromap --preset chip -x "$IDX" -r "$REF" \
#   -1 ${REP1_R1} -2 ${REP1_R2} \
#   -t "${THREADS}" \
#   --SAM -o ${ALIGN_DIR}/rep1/rep1.sam

# chromap --preset chip -x "$IDX" -r "$REF" \
#   -1 ${REP2_R1} -2 ${REP2_R2} \
#   -t "${THREADS}" \
#   --SAM -o ${ALIGN_DIR}/rep2/rep2.sam

# chromap --preset chip -x "$IDX" -r "$REF" \
#   -1 ${REP3_R1} -2 ${REP3_R2} \
#   -t "${THREADS}" \
#   --SAM -o ${ALIGN_DIR}/rep3/rep3.sam

# chromap --preset chip -x "$IDX" -r "$REF" \
#   -1 ${REP4_R1} -2 ${REP4_R2} \
#   -t "${THREADS}" \
#   --SAM -o ${ALIGN_DIR}/rep4/rep4.sam

# # minimal, ENCODE-style filters for ChIP:
# # - MAPQ >= 30 <- chromap by default does this
# # - proper pairs only
# # - remove secondary/supplementary/unmapped/QC-failed reads
# for r in rep1 rep2 rep3 rep4; do
#   samtools view -@ "${THREADS}" -b ${ALIGN_DIR}/$r/${r}.sam \
#     -q 30 \
#     -f 0x2 \
#     -F 0xB04 \
#     -o ${ALIGN_DIR}/$r/${r}_filtered.bam

#   # sort and index each filtered bam
#   samtools sort -@ "${THREADS}" -o ${ALIGN_DIR}/$r/${r}_filtered_sorted.bam ${ALIGN_DIR}/$r/${r}_filtered.bam
#   samtools index ${ALIGN_DIR}/$r/${r}_filtered_sorted.bam
# done

# # merge the filtered BAMs
# samtools merge -@ "${THREADS}" -o ${ALIGN_DIR}/merged_unsorted.bam \
#   ${ALIGN_DIR}/rep1/rep1_filtered_sorted.bam \
#   ${ALIGN_DIR}/rep2/rep2_filtered_sorted.bam \
#   ${ALIGN_DIR}/rep3/rep3_filtered_sorted.bam \
#   ${ALIGN_DIR}/rep4/rep4_filtered_sorted.bam

# # sort and index
# samtools sort -@ "${THREADS}" ${ALIGN_DIR}/merged_unsorted.bam -o ${ALIGN_DIR}/merged_sorted.bam
# samtools index ${ALIGN_DIR}/merged_sorted.bam

# OPTION 2: if you don't want to filter out improper pairs, secondary/supplementary/unmapped/QC-failed reads, you can
# have chromap produce BED files instead, then concat the BED files and sort them.
#
# align each replicate with the ChIP preset (-l 2000 --remove-pcr-duplicates --low-mem --BED).
chromap --preset chip -x "$IDX" -r "$REF" \
  -1 ${REP1_R1} -2 ${REP1_R2} \
  -t "${THREADS}" \
  -o ${ALIGN_DIR}/rep1/rep1.bed

chromap --preset chip -x "$IDX" -r "$REF" \
  -1 ${REP2_R1} -2 ${REP2_R2} \
  -t "${THREADS}" \
  -o ${ALIGN_DIR}/rep2/rep2.bed

chromap --preset chip -x "$IDX" -r "$REF" \
  -1 ${REP3_R1} -2 ${REP3_R2} \
  -t "${THREADS}" \
  -o ${ALIGN_DIR}/rep3/rep3.bed

chromap --preset chip -x "$IDX" -r "$REF" \
  -1 ${REP4_R1} -2 ${REP4_R2} \
  -t "${THREADS}" \
  -o ${ALIGN_DIR}/rep4/rep4.bed
```