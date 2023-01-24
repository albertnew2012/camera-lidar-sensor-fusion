import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pcl
import cv2
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

def annotate_obj_in_pcd(bbox: List[int],colors: np.ndarray, rgb:tuple)->None:
    """
    project lidar points onto image domain. For math theory, please refer to
    https://towardsdatascience.com/what-are-intrinsic-and-extrinsic-camera-parameters-in-computer-vision-7071b72fb8ec
    :param bbox: list of 4 elements
    :param colors: color of rach lidar points within camera field of view
    :param rgb: tuple of 3 elements
    :return: None
    """
    rgb = np.asarray(rgb,dtype="float32")
    if max(rgb) > 1:
        rgb /= 255.
    for idx, uv_ in enumerate(uv):
        if bbox[0]<=uv_[0]<bbox[2] and bbox[1]<=uv_[1]<bbox[3] :
            colors[idx] = rgb



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

    # here I used given bounding boxes of known objects in image domain
    bbox_traffic_light = [708,160,735, 212]
    bbox_car = [590,359,705,451]
    img = cv2.imread("data/camera-view.png")
    img_copy = np.copy(img)
    cv2.rectangle(img_copy,bbox_traffic_light[0:2],bbox_traffic_light[2:],(0,0,255),2)
    cv2.rectangle(img_copy,bbox_car[0:2],bbox_car[2:],(0,0,255),2)
    cv2.putText(img_copy,"Traffic light", (bbox_traffic_light[0],bbox_traffic_light[1]-5),0,0.5,(0,0,255))
    cv2.putText(img_copy,"car", (bbox_car[0],bbox_car[1]-5),0,0.5,(0,0,255))
    cv2.imshow("image", img_copy)
    cv2.imwrite("annotated_img.png",img_copy)
    cv2.waitKey(0)

    # fuse camera image to 3d lidar points
    indices, uv = projectLidar2Camera(instrinsics, extrinsics, points, img)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[indices])  #
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    colors = img[uv[:,1], uv[:,0], :]/255.

    # paint detected objects in different color
    annotate_obj_in_pcd(bbox_traffic_light, colors,(255,0,0))
    annotate_obj_in_pcd(bbox_car, colors, (0, 255, 0))
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd],window_name="Fuse 2D camera with 3D lidar Points")
    o3d.io.write_point_cloud("result/annotated.pcd",pcd)
    cv2.destroyAllWindows()
