import numpy as np
import open3d as o3d
import cv2
from tqdm import tqdm
from scipy.ndimage import distance_transform_edt


def visualize_rgb_map_3d(pc: np.ndarray, rgb: np.ndarray):
    grid_rgb = rgb / 255.0

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)
    pcd.colors = o3d.utility.Vector3dVector(grid_rgb)
    o3d.visualization.draw_geometries([pcd])


def visualize_rgb_map_3d_instances(pc: np.ndarray, rgb: np.ndarray, contours: list, centers: list, bboxs: list, mask):
    grid_rgb = rgb / 255.0

    clouds = []

    # pcd = o3d.geometry.PointCloud()
    # print("pc:", pc.shape, pc.dtype)
    # pcd.points = o3d.utility.Vector3dVector(pc)
    # pcd.colors = o3d.utility.Vector3dVector(grid_rgb)
    # clouds.append(pcd)

    z = 20

    # for contour in contours:
    #     contour = np.array(contour)
    #     newC = np.full((contour.shape[0], 3),z)
    #     newC[:,0:2] = contour
    #     contour = newC
    #     cloud = o3d.geometry.PointCloud()
    #     cloud.points = o3d.utility.Vector3dVector(contour)
    #     cloud.paint_uniform_color([1, 0, 0])
    #     clouds.append(cloud)

    # for center in centers:
    #     center = np.array(center)
    #     newC = np.full((center.shape[0], 3),z)
    #     newC[:, 0:2] = center
    #     center = newC
    #     cloud = o3d.geometry.PointCloud()
    #     cloud.points = o3d.utility.Vector3dVector(center)
    #     cloud.paint_uniform_color([0, 1, 0])
    #     clouds.append(cloud)

    # for bbox in bboxs:
    #     point1 = [bbox[0], bbox[2]]
    #     point2 = [bbox[1], bbox[2]]
    #     point3 = [bbox[0], bbox[3]]
    #     point4 = [bbox[1], bbox[3]]

    #     bbox = np.array([point1, point2, point3, point4])
    #     newC = np.full((bbox.shape[0], 3), z)
    #     newC[:, 0:2] = bbox
    #     bbox = newC
    #     cloud = o3d.geometry.PointCloud()
    #     cloud.points = o3d.utility.Vector3dVector(bbox)
    #     cloud.paint_uniform_color([1, 1, 0])
    #     clouds.append(cloud)

    minx = np.min(pc[:, 0])
    miny = np.min(pc[:, 1])

    masked = []
    masked_col = []
    for i in tqdm(range(pc.shape[0])):
        xyz = pc[i]
        xi = xyz[0]-minx
        yi = xyz[1]-miny
        if(mask[xi,yi]):
            masked.append(xyz)
            masked_col.append(grid_rgb[i])

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(masked)
    cloud.colors = o3d.utility.Vector3dVector(masked_col)
    #cloud.paint_uniform_color([1, 0, 0])
    clouds.append(cloud)

    o3d.visualization.draw_geometries(clouds)


def get_heatmap_from_mask_3d(
    pc: np.ndarray, mask: np.ndarray, cell_size: float = 0.05, decay_rate: float = 0.01
) -> np.ndarray:
    target_pc = pc[mask, :]
    other_ids = np.where(mask == 0)[0]
    other_pc = pc[other_ids, :]

    target_sim = np.ones((target_pc.shape[0], 1))
    other_sim = np.zeros((other_pc.shape[0], 1))
    pbar = tqdm(other_pc, desc="Computing heat", total=other_pc.shape[0])
    for other_p_i, p in enumerate(pbar):
        dist = np.linalg.norm(target_pc - p, axis=1) / cell_size
        min_dist_i = np.argmin(dist)
        min_dist = dist[min_dist_i]
        other_sim[other_p_i] = np.clip(1 - min_dist * decay_rate, 0, 1)

    new_pc = pc.copy()
    heatmap = np.ones((new_pc.shape[0], 1), dtype=np.float32)
    for s_i, s in enumerate(other_sim):
        heatmap[other_ids[s_i]] = s
    return heatmap.flatten()


def visualize_masked_map_3d(pc: np.ndarray, mask: np.ndarray, rgb: np.ndarray, transparency: float = 0.5):
    heatmap = mask.astype(np.float16)
    visualize_heatmap_3d(pc, heatmap, rgb, transparency)


def visualize_heatmap_3d(pc: np.ndarray, heatmap: np.ndarray, rgb: np.ndarray, transparency: float = 0.5):
    sim_new = (heatmap * 255).astype(np.uint8)
    heat = cv2.applyColorMap(sim_new, cv2.COLORMAP_JET)
    heat = heat.reshape(-1, 3)[:, ::-1].astype(np.float32)
    heat_rgb = heat * transparency + rgb * (1 - transparency)
    visualize_rgb_map_3d(pc, heat_rgb)


def pool_3d_label_to_2d(mask_3d: np.ndarray, grid_pos: np.ndarray, gs: int) -> np.ndarray:
    mask_2d = np.zeros((gs, gs), dtype=bool)
    for i, pos in enumerate(grid_pos):
        row, col, h = pos
        mask_2d[row, col] = mask_3d[i] or mask_2d[row, col]

    return mask_2d


def pool_3d_rgb_to_2d(rgb: np.ndarray, grid_pos: np.ndarray, gs: int) -> np.ndarray:
    rgb_2d = np.zeros((gs, gs, 3), dtype=np.uint8)
    height = -100 * np.ones((gs, gs), dtype=np.int32)
    for i, pos in enumerate(grid_pos):
        row, col, h = pos
        if h > height[row, col]:
            rgb_2d[row, col] = rgb[i]

    return rgb_2d


def get_heatmap_from_mask_2d(mask: np.ndarray, cell_size: float = 0.05, decay_rate: float = 0.01) -> np.ndarray:
    dists = distance_transform_edt(mask == 0) / cell_size
    tmp = np.ones_like(dists) - (dists * decay_rate)
    heatmap = np.where(tmp < 0, np.zeros_like(tmp), tmp)

    return heatmap


def visualize_rgb_map_2d(rgb: np.ndarray):
    """visualize rgb image

    Args:
        rgb (np.ndarray): (gs, gs, 3) element range [0, 255] np.uint8
    """
    rgb = rgb.astype(np.uint8)
    bgr = rgb[:, :, ::-1]
    cv2.imshow("rgb map", bgr)
    cv2.waitKey(0)


def visualize_heatmap_2d(rgb: np.ndarray, heatmap: np.ndarray, transparency: float = 0.5):
    """visualize heatmap

    Args:
        rgb (np.ndarray): (gs, gs, 3) element range [0, 255] np.uint8
        heatmap (np.ndarray): (gs, gs) element range [0, 1] np.float32
    """
    sim_new = (heatmap * 255).astype(np.uint8)
    heat = cv2.applyColorMap(sim_new, cv2.COLORMAP_JET)
    heat = heat[:, :, ::-1].astype(np.float32)  # convert to RGB
    heat_rgb = heat * transparency + rgb * (1 - transparency)
    visualize_rgb_map_2d(heat_rgb)


def visualize_masked_map_2d(rgb: np.ndarray, mask: np.ndarray):
    """visualize masked map

    Args:
        rgb (np.ndarray): (gs, gs, 3) element range [0, 255] np.uint8
        mask (np.ndarray): (gs, gs) element range [0, 1] np.uint8
    """
    visualize_heatmap_2d(rgb, mask.astype(np.float32))
