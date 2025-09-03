from importlib import resources
from enum import Enum

class DataFile(Enum):
    atac_ref_motifs = "ATAC.ref.motifs.txt"
    dnase_ref_motifs = "DNASE.ref.motifs.txt"

def get_data_path(data_file_entry: DataFile):    
    with resources.path("histobpnet.data", data_file_entry.value) as f:
        data_file_path = f
    return data_file_path
