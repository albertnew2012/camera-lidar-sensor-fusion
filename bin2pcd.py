import os
import numpy as np
import pcl
# import open3d as o3d


def load_bin_file(file_path, save_into_pcd=True):
    """
    Load a .bin file containing point cloud data.

    Args:
        file_path (str): Path to the .bin file.

    Returns:
        numpy.ndarray: Loaded point cloud data.
    """
    try:
        # Assuming each point is stored as 4 floats (x, y, z, intensity)
        points = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return None
    pcd_file = os.path.splitext(file_path)[0] + ".pcd"
    if save_into_pcd:
        pcd = pcl.PointCloud_PointXYZI(points)
        pcl.save(pcd, pcd_file)
    return points



if __name__ == "__main__":
    data_dir = "."
    
    # Find all .bin files in the data_dir
    bin_files = [
        os.path.join(root, file)
        for root, _, files in os.walk(data_dir)
        for file in files if file.endswith(".bin")
    ]
    
    # Process each .bin file
    for bin_file in bin_files:
        print(f"Processing {bin_file}...")
        load_bin_file(bin_file)
