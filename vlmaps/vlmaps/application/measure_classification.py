from pathlib import Path
import hydra
from omegaconf import DictConfig
from vlmaps.map.vlmap import VLMap
import numpy as np
from vlmaps.utils.matterport3d_categories import mp3dcat
from vlmaps.utils.visualize_utils import (
    pool_3d_label_to_2d,
    pool_3d_rgb_to_2d,
    visualize_rgb_map_3d,
    visualize_masked_map_2d,
    visualize_heatmap_2d,
    visualize_heatmap_3d,
    visualize_masked_map_3d,
    get_heatmap_from_mask_2d,
    get_heatmap_from_mask_3d,
)
from vlmaps.map.vlmap import MapType
from tqdm import tqdm
from sklearn import metrics

import warnings

import utils.result_writing as results
import utils.classification as classification
import utils.common as common

@hydra.main(
    version_base=None,
    config_path="../config",
    config_name="measure_classification.yaml",
)
def main(config: DictConfig) -> None:
    data_dir = Path(config.data_paths.vlmaps_data_dir) / "vlmaps_dataset"
    data_dirs = sorted([x for x in data_dir.iterdir() if x.is_dir()])

    id = config.scene_id
    batch = config.batch
    out_path = config.output_path
    warnings.simplefilter("ignore")

    cl_out_file = results.getFile(out_path, f"queryability_{config.map_config.map_prefix}_{id}", "out")
    results.writeClassificationHeader(cl_out_file)

    classes, labels, names = common.parseClassFile(config.classes, config.delimiter)

    # GT
    gt = VLMap(config.map_config, data_dir=data_dirs[id])
    gt.load_map(data_dirs[id], MapType.INSTANCES)

    # COMPARISONS
    predictions = []

    # VLMAPS have to be custom compared because they don't segment the image
    vlmap = VLMap(config.map_config, data_dir=data_dirs[id])
    vlmap.load_map(data_dirs[id], MapType.PREDICTED)
    vlmap._init_clip(config.map_config.visual_encoder.vlm_version)
    vlmap.init_categories(mp3dcat[1:-1])
    obstacle_map = vlmap.generate_obstacle_map()
    _ = vlmap.generate_cropped_obstacle_map(obstacle_map)

    classifications = []
    for i in tqdm(range(len(labels)), desc="vlmap labels", leave=False):
        name = names[i]
        label = int(labels[i])

        contours, centers, bbox_list, mask = vlmap.get_pos(name)

        minx = np.min(vlmap.grid_pos[:, 0])
        miny = np.min(vlmap.grid_pos[:, 1])

        # masked area
        mask_3d = []
        for i in range(vlmap.grid_pos.shape[0]):
            xyz = vlmap.grid_pos[i]
            xi = xyz[0]-minx
            yi = xyz[1]-miny

            # In some cases, the grid_pos exceeds the size of the mask
            # and causes index out of bounds exception
            if (xi < mask.shape[0] and yi < mask.shape[1] and mask[xi, yi]):
                mask_3d.append(True)
            else:
                mask_3d.append(False)
        pred = np.array(mask_3d)

        if (np.any(gt.grid_semantic == label)):
            true = (gt.grid_semantic == label).squeeze()
        else:
            true = np.full_like(pred, False)

        cf = classification.classify_all(true, pred, label)
        classifications.append(cf)

    out = classification.aggregate(id, "vlmaps", classifications, False)
    results.writeClassificationLine(cl_out_file, [out])


    ############################################################################
    # THE REST
    ############################################################################
    # predicted semantics
    comp = VLMap(config.map_config, data_dir=data_dirs[id])
    comp.load_map(data_dirs[id], MapType.PREDICTED)
    predictions.append((comp.grid_semantic, "predicted"))

    # predicted postprocessed semantics
    comp = VLMap(config.map_config, data_dir=data_dirs[id])
    comp.load_map(data_dirs[id], MapType.PREDICTED_POSTPROCESSED)
    predictions.append((comp.grid_semantic, "predicted_postprocessed"))

    out = classification.classify(predictions, gt.grid_semantic, labels, id)
    results.writeClassificationLine(cl_out_file, out[0])



if __name__ == "__main__":
    main()
