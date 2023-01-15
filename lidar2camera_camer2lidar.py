import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import pcl
from typing import List


def projectLidar2Camera(instrinsics: np.ndarray, extrinsics: np.ndarray, points: np.ndarray, img: np.ndarray) -> List:
    """
    project lidar points onto image domain. For math theory, please refer to
    https://towardsdatascience.com/what-are-intrinsic-and-extrinsic-camera-parameters-in-computer-vision-7071b72fb8ec
    :param instrinsics: 3*3 matrix -> 3*4 matrix
    :param extrinsics: 4*4 matrix
    :param points: (?,4,1)
    :param img: RGB image
    :return: [indices, uv]
    """

    if instrinsics.shape == (3, 3):
        instrinsics = np.hstack([instrinsics, np.zeros((3, 1))])
    if instrinsics.shape != (3, 4):
        raise ValueError("instrinsic shape could not be converted to 3*4")
    if not np.array_equal(extrinsics[3, :3], np.array([0., 0., 0.])):
        extrinsics = extrinsics.T
    if not np.array_equal(extrinsics[3, :3], np.array([0., 0., 0.])):
        raise ValueError("extrinsics matrix is wrong")
    if points.ndim == 2 and points.shape[-1] == 3:
        points = np.hstack([points, np.ones((points.shape[0], 1))])
        points = np.expand_dims(points, axis=-1)

    uvw = instrinsics @ extrinsics @ points
    uvw = np.squeeze(uvw)
    uvw /= uvw[:, [2]]

    height, width = img.shape[:2]
    indices = np.asarray([points[:, 0, 0] > 0, 0 <= uvw[:, 0], uvw[:, 0] < width, 0 <= uvw[:, 1], uvw[:, 1] < height , uvw[:, -1] > 0]).all(axis=0)
    uv = uvw[indices][:, :2].astype(int)
    return [np.where(indices)[0], uv]


if __name__ == '__main__':
    # these camera info, image and pcd data are all from Matlab
    instrinsics = np.array([[1109, 0, 640],[0, 1109, 360],[0, 0, 1]])
    camera2lidar = np.array([[0, 0, 1, 0],[-1, 0, 0, 0],[0, -1, 0, 0],[0, 0 ,0, 1]])
    lidar2camera = camera2lidar.T
    extrinsics = lidar2camera

    # downsampling lidar points to be visualized on 2d image
    pcd = pcl.load("data/lidar-frame.pcd")
    points = pcd.to_array()
    filter = pcd.make_voxel_grid_filter()
    filter.set_leaf_size(0.95, 0.95, 0.95)
    pcd_downsampled= filter.filter()
    points_downsampled = pcd_downsampled.to_array()

    # load image and do projection
    img = plt.imread("data/camera-view.png")
    indices, uv = projectLidar2Camera(instrinsics, extrinsics, points_downsampled, img)
    plt.figure("project 3D points onto image")
    height, width = img.shape[:2]
    plt.axis([0, width,height, 0])
    plt.imshow(img)
    plt.plot(uv[:, 0], uv[:, 1], "r.")  # valid_indices = indices
    plt.savefig("result/project-3D-points-onto-image-street.png")

    # fuse camera image to 3d lidar points
    indices, uv = projectLidar2Camera(instrinsics, extrinsics, points, img)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[indices])  #
    colors = img[uv[:,1], uv[:,0], :]
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd],window_name="Fuse 2D camera with 3D lidar Points")
