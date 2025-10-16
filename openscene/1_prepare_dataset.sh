#!/bin/bash
data="$BASE_DIRhdd/datasets/matterport3d/v1/scans"

for scene in 5LpN3gDmAk7 gTV8FGcVJC9 jh4fc5c5qoQ JmbYfDe2QKZ mJXqzFtmKg4 ur6pFq6Qu1A UwV83HsGsw3 Vt2qJdWjCF2 YmJkqBEsHnH
do
    dir=$data/$scene

    cd $dir

    unzip -u region_segmentations.zip
    unzip -u matterport_camera_intrinsics.zip
    unzip -u matterport_camera_poses.zip
    unzip -u matterport_color_images.zip
    unzip -u matterport_hdr_images.zip
    unzip -u matterport_mesh.zip
    unzip -u matterport_skybox_images.zip
    unzip -u poisson_meshes.zip
    unzip -u sens.zip
    unzip -u undistorted_camera_parameters.zip
    unzip -u undistorted_color_images.zip
    unzip -u undistorted_depth_images.zip
    unzip -u undistorted_normal_images.zip
    unzip -u cameras.zip
    unzip -u house_segmentations
    unzip -u house_segmentations.zip

    mv $scene/* .
    rm -rf $scene
done