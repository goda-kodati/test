"""
Normalize THUmanv2 scans with SMPL parameters for rendering.

THUmanv2 scans dataset
---------------------------
|----THuman2.0_Release
    |----0000
        |----0000.obj
        |----material0.jpeg
        |----material0.mtl
    |----0001     
        |----...  
    |----0525   
        |----...
THUmanv2 SMPL format
---------------------------
datasets
|----THuman2.0_smpl
    |----0000_smpl.pkl
    |----0001_smpl.pkl
    |----...
    |----0525_smpl.pkl
Run as `python3 prepare_thuman_scans_smpl.py`
"""

import os
import numpy as np
import argparse
from tqdm import tqdm
import trimesh
import pandas

def process_scans(smpl_folder, src_dir, dst_dir, index):
    """Process thuman scans and align them with SMPL parameters."""
    scan_idx = "%04d" % index
    scan_path = os.path.join(src_dir, scan_idx, scan_idx + '.obj')

    print(f"Trying to load mesh from: {scan_path}")  # Debugging line

    # Check if the scan path exists
    if not os.path.isfile(scan_path):
        print(f"Warning: {scan_path} does not exist. Skipping this index.")
        return

    output_folder = os.path.join(dst_dir, scan_idx)
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, scan_idx + '.obj')

    # Load the mesh
    try:
        mesh = trimesh.load(scan_path)
    except Exception as e:
        print(f"Error loading {scan_path}: {e}")
        return

    # Load SMPL file if it exists
    pickle_path = os.path.join(smpl_folder, '%04d_smpl.pkl' % index)
    if os.path.exists(pickle_path):
        smpl_file = pandas.read_pickle(pickle_path)

        # Process mesh with SMPL parameters
        scan_verts = mesh.vertices

        # Normalize the vertices
        scan_verts -= smpl_file['transl']
        scan_verts /= smpl_file['scale'][0]
        mesh.vertices = scan_verts
    else:
        print(f"SMPL file not found for {scan_idx}. Using original mesh without normalization.")

    # Export the processed mesh
    mesh.export(output_path)

def split(a, n):
    """Split the list into n nearly equal parts."""
    k, m = divmod(len(a), n)
    return [a[i*k + min(i, m):(i+1)*k + min(i+1, m)] for i in range(n)]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--id', type=int, default=0)
    parser.add_argument('--tot', type=int, default=1)

    args = parser.parse_args()

    src_dir = 'datasets/THuman2.0/THuman2.0_Release'
    dst_dir = 'datasets/THuman2.0/THuman2.0_aligned_scans'
    smpl_folder = 'datasets/THuman2.0/THuman2.0_smpl'

    scan_list = sorted(os.listdir(src_dir))

    # Filter for existing directories that match the expected format
    valid_indices = [int(d) for d in scan_list if d.isdigit()]
    valid_indices.sort()

    task = split(valid_indices, args.tot)[args.id]

    for idx in tqdm(task):
        process_scans(smpl_folder, src_dir, dst_dir, idx)
