- ATAC.ref.motifs.txt
Format: Contains motifs one position per line, 4 columns per base tab-separated. First 
line of each motif starts with ">" followed by name. "_" is used to name motifs 
and name should end with "_plus" or "_minus".

Contains reference motifs for the Tn5 transposase used in ATAC-seq. These motifs
were constructed using the get_pwms function (see general_utils.py) on +4/-4 shifted
(valeh: correct?) tagalign files from ATAC-seq experiments, and then taking central 20 bases.

- DNASE.ref.motifs.txt
Format: Same as ATAC.ref.motifs.txt.

Contains reference motifs for the DNASE enzyme used in DNASE-seq. These motifs
were constructed using the get_pwms function (see general_utils.py) on +0/+1 shifted
tagalign files from DNASE-seq experiments, and then taking central 20 bases.