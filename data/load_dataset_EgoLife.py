'''
Dataset from https://huggingface.co/datasets/lmms-lab/EgoLife
'''

from huggingface_hub import snapshot_download
import os

save_dir = "./EgoLife"

snapshot_download(
    repo_id="lmms-lab/EgoLife",
    repo_type="dataset",
    local_dir=save_dir,
    allow_patterns=["A1_JAKE/DAY1/*"]
)

