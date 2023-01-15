import numpy as np

import matplotlib.pyplot as plt
import pcl
from typing import List
from mpl_toolkits.mplot3d import Axes3D


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
    indices = np.asarray([points[:, 0, 0] > 0, 0 <= uvw[:, 0], uvw[:, 0] < width, 0 <= uvw[:, 1], uvw[:, 1] < height,
                          uvw[:, -1] > 0]).all(axis=0)
    uv = uvw[indices][:, :2].astype(int)
    return [np.where(indices)[0], uv]


if __name__ == '__main__':
    # these extrinsics, instrinsics, image and pcd data are extracted from MATLAB
    extrinsics = np.array([[-0.00604699, -0.15869743, 0.98730875, 0.], [-0.99795, -0.06194804, -0.01606953, 0.],
                           [0.06371204, -0.98538194, -0.15799751, 0.], [-0.00310716, -0.39222156, -0.82649062, 1.]])
    instrinsics = np.array(
        [[1.61454623e+03, 0.00000000e+00, 6.41227636e+02], [0.00000000e+00, 1.61466901e+03, 4.80141056e+02],
         [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

    # load and visualize 3d lidar point cloud
    points = pcl.load("data/0005.pcd").to_array()
    fig = plt.figure("3d lidar point cloud")
    ax = fig.add_subplot(projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2])
    plt.savefig("result/3d-lidar-point-cloud.png")

    # project lidar points onto image and extract lidar points visible in camera file of view
    img = plt.imread("data/0005.png")
    indices, uv = projectLidar2Camera(instrinsics, extrinsics, points, img)

    plt.figure("project 3D points onto image")
    height, width = img.shape[:2]
    plt.axis([0, width, height, 0])
    plt.imshow(img)
    plt.plot(uv[:, 0], uv[:, 1], "r.")
    plt.savefig("result/project-3D-points-onto-image.png")

    # visualize lidar points that are visible in camera FOV
    fig = plt.figure("visualable lidar points in camera FOV")
    ax = fig.add_subplot(projection='3d')
    ax.scatter(points[indices, 0], points[indices, 1], points[indices, 2])
    plt.savefig("result/visualable-lidar-points-in-camera-FOV.png")
