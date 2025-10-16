Changes made to the OpenScene repository are:
- scripts to automate running of experiments
- instance classification
- benchmarking

Complete diff:
****************************************************************************************
Only in latent_semantics/openscene: 0_run_all.sh
Only in latent_semantics/openscene: 1_prepare_dataset.sh
Only in latent_semantics/openscene: 2_preprocess.sh
Only in latent_semantics/openscene: 3_fuse_features.sh
Only in latent_semantics/openscene: 4_eval.sh
Only in latent_semantics/openscene: 5_instance_segmentation.sh
Only in latent_semantics/openscene: 6_benchmark.sh
Only in latent_semantics/openscene: 7_parse_files.sh
Only in latent_semantics/openscene: A_part1.sh
Only in latent_semantics/openscene: A_part2.sh
Only in latent_semantics/openscene: CHANGES.md
Only in latent_semantics/openscene: check_instance_labels.py
Only in latent_semantics/openscene: check_instances.py
Only in latent_semantics/openscene: check_preprocess.py
Only in latent_semantics/openscene: classification_benchmark.py
Only in latent_semantics/openscene: classification_benchmark.sh
Only in latent_semantics/openscene: class_summer.py
Only in latent_semantics/openscene: clear_links.sh
Only in latent_semantics/openscene/config/matterport: ours_lseg_pretrained_distill.yaml
Only in latent_semantics/openscene/config/matterport: ours_lseg_pretrained_fusion.yaml
diff --color -bur latent_semantics/openscene/config/matterport/ours_lseg_pretrained.yaml origs/openscene/config/matterport/ours_lseg_pretrained.yaml
--- latent_semantics/openscene/config/matterport/ours_lseg_pretrained.yaml	2025-10-14 15:43:44.318830015 +0300
+++ origs/openscene/config/matterport/ours_lseg_pretrained.yaml	2025-10-14 16:05:52.140870537 +0300
@@ -1,8 +1,8 @@
 DATA:
-  data_root: data/matterport_3d_40
-  data_root_2d_fused_feature: fusedfeatures-lseg
+  data_root: data/matterport_3d
+  data_root_2d_fused_feature: data/matterport_multiview_lseg
   feature_2d_extractor: lseg
-  classes: 42
+  classes: 21
   aug: True
   voxel_size: 0.02
   input_color: False
@@ -32,24 +32,19 @@

 TEST:
   split: test  # split in [train, val and test]
-  labelset: matterport_3d_40
   prompt_eng: True
   mark_no_feature_to_unknown: True
   feature_type: 'ensemble' # 'distill' | 'fusion' | 'ensemble'
-  save_feature_as_numpy: True
-  numpy_output_folder: 'embeddings-lseg'
-  save_comparable_results: True #! This is our comparison to VLMAPS
-  save_comparable_results_dir: 'save-lseg'
+  save_feature_as_numpy: False
   vis_input: True
   vis_pred: True
   vis_gt: True
   test_workers: 8
   test_gpu: [0]
   test_batch_size: 1
-  test_repeats: 1
-  model_path: 'https://cvg-data.inf.ethz.ch/openscene/models/matterport_lseg.pth.tar'
-  save_folder: 'save'
-  eval_iou: False
+  test_repeats: 5
+  model_path: 'https://cvg-data.inf.ethz.ch/openscene/models/matterport_openseg.pth.tar'
+  save_folder:

 Distributed:
   dist_url: tcp://127.0.0.1:6787
Only in latent_semantics/openscene/config/matterport: ours_openseg_pretrained_distill.yaml
Only in latent_semantics/openscene/config/matterport: ours_openseg_pretrained_fusion.yaml
diff --color -bur latent_semantics/openscene/config/matterport/ours_openseg_pretrained.yaml origs/openscene/config/matterport/ours_openseg_pretrained.yaml
--- latent_semantics/openscene/config/matterport/ours_openseg_pretrained.yaml	2025-10-14 15:43:44.318830015 +0300
+++ origs/openscene/config/matterport/ours_openseg_pretrained.yaml	2025-10-14 16:05:52.140870537 +0300
@@ -1,8 +1,8 @@
 DATA:
-  data_root: data/matterport_3d_40
-  data_root_2d_fused_feature: fusedfeatures-openseg #data/matterport_multiview_openseg_test
+  data_root: data/matterport_3d
+  data_root_2d_fused_feature: data/matterport_multiview_openseg_test
   feature_2d_extractor: openseg
-  classes: 42
+  classes: 21
   aug: True
   voxel_size: 0.02
   input_color: False
@@ -32,24 +32,19 @@

 TEST:
   split: test  # split in [train, val and test]
-  labelset: matterport_3d_40
   prompt_eng: True
   mark_no_feature_to_unknown: True
   feature_type: 'ensemble' # 'distill' | 'fusion' | 'ensemble'
-  save_feature_as_numpy: True
-  numpy_output_folder: 'embeddings-openseg'
-  save_comparable_results: True #! This is our comparison to VLMAPS
-  save_comparable_results_dir: 'save-openseg'
+  save_feature_as_numpy: False
   vis_input: True
   vis_pred: True
   vis_gt: True
   test_workers: 8
   test_gpu: [0]
   test_batch_size: 1
-  test_repeats: 1
+  test_repeats: 5
   model_path: 'https://cvg-data.inf.ethz.ch/openscene/models/matterport_openseg.pth.tar'
-  save_folder: 'save'
-  eval_iou: False
+  save_folder:

 Distributed:
   dist_url: tcp://127.0.0.1:6787
Only in latent_semantics/openscene: create_links.sh
diff --color -bur latent_semantics/openscene/dataset/label_constants.py origs/openscene/dataset/label_constants.py
--- latent_semantics/openscene/dataset/label_constants.py	2025-10-14 15:43:44.318830015 +0300
+++ origs/openscene/dataset/label_constants.py	2025-10-14 16:05:52.141870537 +0300
@@ -1,7 +1,7 @@
 '''Label file for different datasets.'''

 SCANNET_LABELS_20 = ('wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa',
-                     'table', 'door', 'window', 'bookshelf', 'picture', 'counter', 'desk', 'curtain', 'refrigerator', 'shower curtain',
+                     'table', 'door', 'window', 'bookshelf', 'picture','counter', 'desk', 'curtain', 'refrigerator', 'shower curtain',
                      'toilet', 'sink', 'bathtub', 'otherfurniture')


@@ -16,47 +16,6 @@
                         'coffee table', 'counter', 'bench', 'garbage bin', 'fireplace',
                         'clothes', 'bathtub', 'book', 'air vent', 'faucet')

-VLMAPS_LABELS = ('wall',
-'floor',
-'chair',
-'door',
-'table',
-'picture',
-'cabinet',
-'pillow',
-'window',
-'sofa',
-'bed ',
-'curtain',
-'night stand',
-'plant',
-'sink',
-'stairs',
-'ceiling',
-'toilet',
-'stool',
-'towel',
-'mirror',
-'television',
-'shower',
-'column',
-'bathtub',
-'counter',
-'fireplace',
-'lamp',
-'beam',
-'banister',
-'shelves',
-'blinds',
-'gym equipment',
-'bench',
-'board_panel',
-'furniture',
-'appliances',
-'clothes',
-'objects',
-'misc')
-
 MATTERPORT_LABELS_80 = ('wall', 'door', 'ceiling', 'floor', 'picture', 'window', 'chair', 'pillow', 'lamp',
                         'cabinet', 'curtain', 'table', 'plant', 'mirror', 'towel', 'sink', 'shelves', 'sofa',
                         'bed', 'night stand', 'toilet', 'column', 'banister', 'stairs', 'stool', 'vase',
@@ -94,7 +53,7 @@


 NUSCENES_LABELS_DETAILS = ('barrier', 'barricade', 'bicycle', 'bus', 'car', 'bulldozer', 'excavator', 'concrete mixer', 'crane', 'dump truck',
-                           'motorcycle', 'person', 'pedestrian', 'traffic cone', 'trailer', 'semi trailer', 'cargo container', 'shipping container', 'freight container',
+                           'motorcycle', 'person', 'pedestrian','traffic cone', 'trailer', 'semi trailer', 'cargo container', 'shipping container', 'freight container',
                            'truck', 'road', 'curb', 'traffic island', 'traffic median', 'sidewalk', 'grass', 'grassland', 'lawn', 'meadow', 'turf', 'sod',
                            'building', 'wall', 'pole', 'awning', 'tree', 'trunk', 'tree trunk', 'bush', 'shrub', 'plant', 'flower', 'woods')

@@ -107,24 +66,24 @@


 NUSCENES16_COLORMAP = {
-    1: (220, 220,  0),  # barrier
+    1: (220,220,  0), # barrier
     2: (119, 11, 32),  # bicycle
     3: (0, 60, 100),  # bus
     4: (0, 0, 250),  # car
-    5: (230, 230, 250),  # construction vehicle
+    5: (230,230,250), # construction vehicle
     6: (0, 0, 230),  # motorcycle
     7: (220, 20, 60),  # person
     8: (250, 170, 30),  # traffic cone
     9: (200, 150, 0),  # trailer
-    10: (0, 0, 110),  # truck
+    10: (0, 0, 110) , # truck
     11: (128, 64, 128),  # road
-    12: (0, 250, 250),  # other flat
+    12: (0,250, 250), # other flat
     13: (244, 35, 232),  # sidewalk
     14: (152, 251, 152),  # terrain
     15: (70, 70, 70),  # manmade
-    16: (107, 142, 35),  # vegetation
+    16: (107,142, 35), # vegetation
     17: (0, 0, 0),  # unknown
-}
+    }

 SCANNET_COLOR_MAP_20 = {
     1: (174., 199., 232.),
diff --color -bur latent_semantics/openscene/dataset/matterport/scenes_test.txt origs/openscene/dataset/matterport/scenes_test.txt
--- latent_semantics/openscene/dataset/matterport/scenes_test.txt	2025-10-14 15:43:44.319830015 +0300
+++ origs/openscene/dataset/matterport/scenes_test.txt	2025-10-14 16:05:52.141870537 +0300
@@ -1,9 +1,18 @@
-5LpN3gDmAk7
-gTV8FGcVJC9
-jh4fc5c5qoQ
-JmbYfDe2QKZ
-mJXqzFtmKg4
-ur6pFq6Qu1A
+2t7WUuJeko7
+5ZKStnWn8Zo
+ARNzJeq3xxb
+fzynW3qQPVF
+jtcxE69GiFV
+pa4otMbVnkk
+q9vSo1VnCiC
+rqfALeAoiTq
 UwV83HsGsw3
+wc2JMjhGNzB
+WYY7iVyf5p8
+YFuZgdQ5vWj
+yqstnuAEVhm
+YVUC4YcDtcY
+gxdoqLR6rwA
+gYvKGZ5eRqb
+RPmz2sHmrrY
 Vt2qJdWjCF2
-YmJkqBEsHnH
\ No newline at end of file
Only in latent_semantics/openscene/dataset/matterport: scenes_test.txt-orig
diff --color -bur latent_semantics/openscene/dataset/matterport/scenes_train.txt origs/openscene/dataset/matterport/scenes_train.txt
--- latent_semantics/openscene/dataset/matterport/scenes_train.txt	2025-10-14 15:43:44.319830015 +0300
+++ origs/openscene/dataset/matterport/scenes_train.txt	2025-10-14 16:05:52.141870537 +0300
@@ -1,9 +1,61 @@
+17DRP5sb8fy
+1LXtFkjw3qL
+1pXnuDYAj8r
+29hnd4uzFmX
 5LpN3gDmAk7
+5q7pvUzZiYa
+759xd9YjKW5
+7y3sRwLe3Va
+82sE5b5pLXE
+8WUmhLawc2A
+aayBHfsNo7d
+ac26ZMwG7aT
+B6ByNegPMKs
+b8cTxDM8gDG
+cV4RVeZvu5T
+D7N2EKCX4Sj
+e9zR4mvMWw7
+EDJbREhghzL
+GdvgFV5R1Z5
 gTV8FGcVJC9
+HxpKQynjfin
+i5noydFURQK
+JeFG25nYj2p
+JF19kD82Mey
 jh4fc5c5qoQ
-JmbYfDe2QKZ
+kEZ7cmS4wCh
 mJXqzFtmKg4
+p5wJjkQkbXX
+Pm6F8kyY3z2
+pRbA3pwrgk9
+PuKPg4mmafe
+PX4nDJXEHrG
+qoiz87JEwZ2
+rPc6DW4iMge
+s8pcmisQ38h
+S9hNv5qa7GM
+sKLMLpTHeUy
+SN83YJsR3w2
+sT4fr6TAbpF
+ULsKaCPVFJR
+uNb9QFRL6hY
+Uxmj2M2itWa
+V2XKFyX4ASd
+VFuaQ6m2Qom
+VVfe2KiqLaN
+Vvot9Ly1tCj
+vyrNrziPKCB
+VzqfbhrpDEA
+XcA2TqTSSAj
+2n8kARJN3HM
+D7G3Y4RVNrH
+dhjEzFoUFzH
+E9uDoFAP3SH
+gZ6f7yhEvPG
+JmbYfDe2QKZ
+r1Q1Z4BcV1o
+r47D5H71a5s
 ur6pFq6Qu1A
-UwV83HsGsw3
-Vt2qJdWjCF2
+VLzqgDo317F
 YmJkqBEsHnH
\ No newline at end of file
+ZMojNkEp431
\ No newline at end of file
Only in latent_semantics/openscene/dataset/matterport: scenes_train.txt-orig
diff --color -bur latent_semantics/openscene/dataset/matterport/scenes_val.txt origs/openscene/dataset/matterport/scenes_val.txt
--- latent_semantics/openscene/dataset/matterport/scenes_val.txt	2025-10-14 15:43:44.319830015 +0300
+++ origs/openscene/dataset/matterport/scenes_val.txt	2025-10-14 16:05:52.141870537 +0300
@@ -1,9 +1,11 @@
-5LpN3gDmAk7
-gTV8FGcVJC9
-jh4fc5c5qoQ
-JmbYfDe2QKZ
-mJXqzFtmKg4
-ur6pFq6Qu1A
-UwV83HsGsw3
-Vt2qJdWjCF2
-YmJkqBEsHnH
\ No newline at end of file
+2azQ1b91cZZ
+8194nk5LbLH
+EU6Fwq7SyZv
+oLBMNvg9in8
+QUCTc6BB5sX
+TbHJrupSAjP
+X7HyMhZNoso
+pLe4wQe7qrG
+x8F5xyUWy9e
+Z6MFQCViBuw
+zsNo4HB9uLZ
\ No newline at end of file
Only in latent_semantics/openscene/dataset/matterport: scenes_val.txt-orig
Only in origs/openscene: demo
Only in latent_semantics/openscene: demo_validation.py
Only in latent_semantics/openscene: eval-lseg.sh
Only in latent_semantics/openscene: eval-openseg.sh
Only in origs/openscene: .git
diff --color -bur latent_semantics/openscene/.gitignore origs/openscene/.gitignore
--- latent_semantics/openscene/.gitignore	2025-10-14 15:43:44.228830013 +0300
+++ origs/openscene/.gitignore	2025-10-14 16:05:52.140870537 +0300
@@ -133,57 +133,3 @@
 out
 3rdParty
 initmodel
-
-
-# vim
-*~
-
-results/
-results
-utils
-results-*/
-results-*
-exp/
-exp
-saved_feature/
-saved_feature
-.vscode/
-instances-openseg/
-save-openseg/
-save-openseg
-save-lseg/
-save-lseg
-fusedfeatures-openseg/
-fusedfeatures-openseg
-fusedfeatures-lseg/
-fusedfeatures-lseg
-eval-openseg/
-eval-openseg
-eval-lseg/
-eval-lseg
-instances-openseg/
-instances-openseg
-instances-lseg/
-instances-lseg
-embeddings-openseg/
-embeddings-openseg
-embeddings-lseg/
-embeddings-lseg
-parsed-openseg/
-parsed-openseg
-parsed-lseg/
-parsed-lseg
-*-old/
-*-bak/
-*-bak
-*-old
-*.orig
-*-old/
-*-bad/
-*-bad
-profiling/
-*.prof
-mp40*
-gaps/
-demo/tmp/
-ignored/
\ No newline at end of file
Only in latent_semantics/openscene: instance_classification_benchmark.py
Only in latent_semantics/openscene: instance_classification_benchmark.sh
Only in latent_semantics/openscene: instance_segmentation.py
Only in origs/openscene: media
Only in latent_semantics/openscene: openseg-validation-lseg.sh
Only in latent_semantics/openscene: openseg-validation.sh
Only in latent_semantics/openscene: parse_files.py
diff --color -bur latent_semantics/openscene/run/distill.py origs/openscene/run/distill.py
--- latent_semantics/openscene/run/distill.py	2025-10-14 15:43:50.268830197 +0300
+++ origs/openscene/run/distill.py	2025-10-14 16:05:52.225870540 +0300
@@ -260,8 +260,7 @@
         palette = get_palette()
         dataset_name = 'scannet'
     elif 'matterport' in args.data_root:
-        #labelset = list(MATTERPORT_LABELS_21)
-        labelset = list(VLMAPS_LABELS)
+        labelset = list(MATTERPORT_LABELS_21)
         palette = get_palette(colormap='matterport')
         dataset_name = 'matterport'
     elif 'nuscenes' in args.data_root:
diff --color -bur latent_semantics/openscene/run/eval.sh origs/openscene/run/eval.sh
--- latent_semantics/openscene/run/eval.sh	2025-10-14 15:43:50.268830197 +0300
+++ origs/openscene/run/eval.sh	2025-10-14 16:05:52.225870540 +0300
@@ -9,7 +9,6 @@
 result_dir=${exp_dir}/result_eval

 export PYTHONPATH=.
-
 python -u run/evaluate.py \
   --config=${config} \
   feature_type ${feature_type} \
diff --color -bur latent_semantics/openscene/run/evaluate.py origs/openscene/run/evaluate.py
--- latent_semantics/openscene/run/evaluate.py	2025-10-16 10:50:46.675854268 +0300
+++ origs/openscene/run/evaluate.py	2025-10-14 16:05:52.225870540 +0300
@@ -24,13 +24,6 @@

 from dataset.label_constants import *

-# ██╗  ██╗███████╗██╗     ██████╗ ███████╗██████╗ ███████╗
-# ██║  ██║██╔════╝██║     ██╔══██╗██╔════╝██╔══██╗██╔════╝
-# ███████║█████╗  ██║     ██████╔╝█████╗  ██████╔╝███████╗
-# ██╔══██║██╔══╝  ██║     ██╔═══╝ ██╔══╝  ██╔══██╗╚════██║
-# ██║  ██║███████╗███████╗██║     ███████╗██║  ██║███████║
-# ╚═╝  ╚═╝╚══════╝╚══════╝╚═╝     ╚══════╝╚═╝  ╚═╝╚══════╝
-

 def get_parser():
     '''Parse the config file.'''
@@ -63,17 +56,14 @@
     logger_in.addHandler(handler)
     return logger_in

-
 def is_url(url):
     scheme = urllib.parse.urlparse(url).scheme
     return scheme in ('http', 'https')

-
 def main_process():
     return not args.multiprocessing_distributed or (
         args.multiprocessing_distributed and args.rank % args.ngpus_per_node == 0)

-
 def precompute_text_related_properties(labelset_name):
     '''pre-compute text features, labelset, palette, and mapper.'''

@@ -82,12 +72,10 @@
         labelset[-1] = 'other'  # change 'other furniture' to 'other'
         palette = get_palette(colormap='scannet')
     elif labelset_name == 'matterport_3d' or labelset_name == 'matterport':
-        #note: we want vlmap labels for consistency
         labelset = list(MATTERPORT_LABELS_21)
         palette = get_palette(colormap='matterport')
     elif 'matterport_3d_40' in labelset_name or labelset_name == 'matterport40':
-        #labelset = list(MATTERPORT_LABELS_40)
-        labelset = list(VLMAPS_LABELS)
+        labelset = list(MATTERPORT_LABELS_40)
         palette = get_palette(colormap='matterport_160')
     elif 'matterport_3d_80' in labelset_name or labelset_name == 'matterport80':
         labelset = list(MATTERPORT_LABELS_80)
@@ -112,14 +100,6 @@
     labelset.append('unlabeled')
     return text_features, labelset, mapper, palette

-# ███╗   ███╗ █████╗ ██╗███╗   ██╗
-# ████╗ ████║██╔══██╗██║████╗  ██║
-# ██╔████╔██║███████║██║██╔██╗ ██║
-# ██║╚██╔╝██║██╔══██║██║██║╚██╗██║
-# ██║ ╚═╝ ██║██║  ██║██║██║ ╚████║
-# ╚═╝     ╚═╝╚═╝  ╚═╝╚═╝╚═╝  ╚═══╝
-
-
 def main():
     '''Main function.'''

@@ -146,6 +126,7 @@
         args.multiprocessing_distributed = False
         args.use_apex = False

+
     # By default we do not use shared memory for evaluation
     if not hasattr(args, 'use_shm'):
         args.use_shm = False
@@ -157,13 +138,6 @@
         main_worker(args.test_gpu, args.ngpus_per_node, args)


-# ██╗    ██╗ ██████╗ ██████╗ ██╗  ██╗███████╗██████╗
-# ██║    ██║██╔═══██╗██╔══██╗██║ ██╔╝██╔════╝██╔══██╗
-# ██║ █╗ ██║██║   ██║██████╔╝█████╔╝ █████╗  ██████╔╝
-# ██║███╗██║██║   ██║██╔══██╗██╔═██╗ ██╔══╝  ██╔══██╗
-# ╚███╔███╔╝╚██████╔╝██║  ██║██║  ██╗███████╗██║  ██║
-#  ╚══╝╚══╝  ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝
-
 def main_worker(gpu, ngpus_per_node, argss):
     global args
     args = argss
@@ -212,7 +186,7 @@
                     # add module
                     k = 'module.' + k

-                new_state_dict[k] = v
+                new_state_dict[k]=v
             model.load_state_dict(new_state_dict, strict=True)
             logger.info('Loaded a parallel model')

@@ -221,12 +195,6 @@
     else:
         raise RuntimeError("=> no checkpoint found at '{}'".format(args.model_path))

-    #
-    #
-    #
-    #
-    #
-    ####################################################################################################################
     # ####################### Data Loader ####################### #
     if not hasattr(args, 'input_color'):
         # by default we do not use the point color as input
@@ -245,12 +213,6 @@
                                              drop_last=False, collate_fn=collation_fn_eval_all,
                                              sampler=val_sampler)

-    #
-    #
-    #
-    #
-    #
-    ####################################################################################################################
     # ####################### Test ####################### #
     labelset_name = args.data_root.split('/')[-1]
     if hasattr(args, 'labelset'):
@@ -259,14 +221,6 @@

     evaluate(model, val_loader, labelset_name)

-
-# ███████╗██╗   ██╗ █████╗ ██╗
-# ██╔════╝██║   ██║██╔══██╗██║
-# █████╗  ██║   ██║███████║██║
-# ██╔══╝  ╚██╗ ██╔╝██╔══██║██║
-# ███████╗ ╚████╔╝ ██║  ██║███████╗
-# ╚══════╝  ╚═══╝  ╚═╝  ╚═╝╚══════╝
-
 def evaluate(model, val_data_loader, labelset_name='scannet_3d'):
     '''Evaluate our OpenScene model.'''

@@ -277,7 +231,7 @@

     if args.save_feature_as_numpy:  # save point features to folder
         out_root = os.path.commonprefix([args.save_folder, args.model_path])
-        saved_feature_folder = os.path.join(out_root, args.numpy_output_folder)
+        saved_feature_folder = os.path.join(out_root, 'saved_feature')
         os.makedirs(saved_feature_folder, exist_ok=True)

     # short hands
@@ -315,7 +269,7 @@

             # repeat the evaluation process
             # to account for the randomness in MinkowskiNet voxelization
-            if rep_i > 0:
+            if rep_i>0:
                 seed = np.random.randint(10000)
                 random.seed(seed)
                 np.random.seed(seed)
@@ -326,20 +280,11 @@
             if mark_no_feature_to_unknown:
                 masks = []

-            #
-            #
-            #
-            #
-            #
-            ############################################################################################################
-            # !main
-
             for i, (coords, feat, label, feat_3d, mask, inds_reverse) in enumerate(tqdm(val_data_loader)):
                 sinput = SparseTensor(feat.cuda(non_blocking=True), coords.cuda(non_blocking=True))
                 coords = coords[inds_reverse, :]
                 pcl = coords[:, 1:].cpu().numpy()

-                # if
                 if feature_type == 'distill':
                     predictions = model(sinput)
                     predictions = predictions[inds_reverse, :]
@@ -362,8 +307,7 @@
                     predictions = model(sinput)
                     predictions = predictions[inds_reverse, :]
                     # pred_distill = predictions.half() @ text_features.t()
-                    pred_distill = (predictions/(predictions.norm(dim=-1, keepdim=True)+1e-5)
-                                    ).half() @ text_features.t()
+                    pred_distill = (predictions/(predictions.norm(dim=-1, keepdim=True)+1e-5)).half() @ text_features.t()

                     # logits_distill = torch.max(pred_distill, 1)[1].detach().cpu()
                     # mask_ensem = pred_distill<pred_fusion # confidence-based ensemble
@@ -380,51 +324,16 @@
                     predictions = feat_ensemble  # if we need to save the features
                 else:
                     raise NotImplementedError
-                # endif

                 if args.save_feature_as_numpy:
                     scene_name = val_data_loader.dataset.data_paths[i].split('/')[-1].split('.pth')[0]
-                    path = os.path.join(saved_feature_folder, '{}_openscene_feat_{}.npy'.format(
-                        scene_name, feature_type))
-
-                    file_exists = os.path.exists(path)
-                    if(file_exists):
-                        print(path, "already exists")
-                    else:
-                        np.save(path, predictions.cpu().numpy())
+                    np.save(os.path.join(saved_feature_folder, '{}_openscene_feat_{}.npy'.format(scene_name, feature_type)), predictions.cpu().numpy())

-                #! here we create the parsed files
-                #! GT, prediction, X, Y, Z
-                if args.save_comparable_results:
-                    scene_name = val_data_loader.dataset.data_paths[i].split('/')[-1].split('.pth')[0]
-                    out_root = os.path.commonprefix([args.save_folder, args.model_path])
-                    savepath = os.path.join(out_root, args.save_comparable_results_dir)
-                    os.makedirs(savepath, exist_ok=True)
-                    path = os.path.join(savepath, '{}_openscene_feat_{}.npy'.format(scene_name, feature_type))
-
-                    gt_labels = label.cpu()
-                    xyz = coords[:,1:].cpu()
-
-                    outarray = np.stack((gt_labels, logits_pred, xyz[:, 0], xyz[:, 1], xyz[:, 2]))
-
-
-                    file_exists = os.path.exists(path)
-                    if(file_exists):
-                        print(path, "already exists")
-                    else:
-                        np.save(path, outarray)
-
-                #
-                #
-                #
-                #
-                #
-                ########################################################################################################
                 # Visualize the input, predictions and GT

                 # special case for nuScenes, evaluation points are only a subset of input
                 if 'nuscenes' in labelset_name:
-                    label_mask = (label != 255)
+                    label_mask = (label!=255)
                     label = label[label_mask]
                     logits_pred = logits_pred[label_mask]
                     pred = pred[label_mask]
@@ -433,49 +342,38 @@

                 if vis_input:
                     input_color = torch.load(val_data_loader.dataset.data_paths[i])[1]
-                    export_pointcloud(os.path.join(save_folder, '{}_input.ply'.format(scene_name)),
-                                      pcl, colors=(input_color+1)/2)
+                    export_pointcloud(os.path.join(save_folder, '{}_input.ply'.format(i)), pcl, colors=(input_color+1)/2)

                 if vis_pred:
                     if mapper is not None:
                         pred_label_color = convert_labels_with_palette(mapper[logits_pred].numpy(), palette)
-                        export_pointcloud(os.path.join(save_folder, '{}_{}.ply'.format(
-                            scene_name, feature_type)), pcl, colors=pred_label_color)
+                        export_pointcloud(os.path.join(save_folder, '{}_{}.ply'.format(i, feature_type)), pcl, colors=pred_label_color)
                     else:
                         pred_label_color = convert_labels_with_palette(logits_pred.numpy(), palette)
-                        export_pointcloud(os.path.join(save_folder, '{}_{}.ply'.format(
-                            scene_name, feature_type)), pcl, colors=pred_label_color)
+                        export_pointcloud(os.path.join(save_folder, '{}_{}.ply'.format(i, feature_type)), pcl, colors=pred_label_color)
                         visualize_labels(list(np.unique(logits_pred.numpy())),
                                          labelset,
                                          palette,
-                                         os.path.join(save_folder, '{}_labels_{}.jpg'.format(scene_name, feature_type)), ncol=5)
+                                    os.path.join(save_folder, '{}_labels_{}.jpg'.format(i, feature_type)), ncol=5)

                 # Visualize GT labels
                 if vis_gt:
                     # for points not evaluating
-                    label[label == 255] = len(labelset)-1
+                    label[label==255] = len(labelset)-1
                     gt_label_color = convert_labels_with_palette(label.cpu().numpy(), palette)
-                    export_pointcloud(os.path.join(save_folder, '{}_gt.ply'.format(scene_name)), pcl, colors=gt_label_color)
+                    export_pointcloud(os.path.join(save_folder, '{}_gt.ply'.format(i)), pcl, colors=gt_label_color)
                     visualize_labels(list(np.unique(label.cpu().numpy())),
                                      labelset,
                                      palette,
-                                     os.path.join(save_folder, '{}_labels_gt.jpg'.format(scene_name)), ncol=5)
+                                os.path.join(save_folder, '{}_labels_gt.jpg'.format(i)), ncol=5)

                     if 'nuscenes' in labelset_name:
-                        all_digits = np.unique(np.concatenate(
-                            [np.unique(mapper[logits_pred].numpy()), np.unique(label)]))
+                        all_digits = np.unique(np.concatenate([np.unique(mapper[logits_pred].numpy()), np.unique(label)]))
                         labelset = list(NUSCENES_LABELS_16)
                         labelset[4] = 'construct. vehicle'
                         labelset[10] = 'road'
                         visualize_labels(list(all_digits), labelset,
-                                         palette, os.path.join(save_folder, '{}_label.jpg'.format(scene_name)), ncol=all_digits.shape[0])
-
-                #
-                #
-                #
-                #
-                #
-                ########################################################################################################
+                            palette, os.path.join(save_folder, '{}_label.jpg'.format(i)), ncol=all_digits.shape[0])

                 if eval_iou:
                     if mark_no_feature_to_unknown:
@@ -484,7 +382,7 @@
                         else:
                             masks.append(mask[inds_reverse])

-                    if args.test_repeats == 1:
+                    if args.test_repeats==1:
                         # save directly the logits
                         preds.append(logits_pred)
                     else:
@@ -493,19 +391,12 @@

                     gts.append(label.cpu())

-            #
-            #
-            #
-            #
-            #
-            ############################################################################################################
-
             if eval_iou:
                 gt = torch.cat(gts)
                 pred = torch.cat(preds)

                 pred_logit = pred
-                if args.test_repeats > 1:
+                if args.test_repeats>1:
                     pred_logit = pred.float().max(1)[1]

                 if mapper is not None:
@@ -515,7 +406,7 @@
                     mask = torch.cat(masks)
                     pred_logit[~mask] = 256

-                if args.test_repeats == 1:
+                if args.test_repeats==1:
                     current_iou = metric.evaluate(pred_logit.numpy(),
                                                   gt.numpy(),
                                                   dataset=labelset_name,
@@ -533,6 +424,5 @@
                                                  stdout=True,
                                                  dataset=labelset_name)

-
 if __name__ == '__main__':
     main()
diff --color -bur latent_semantics/openscene/scripts/feature_fusion/matterport_openseg.py origs/openscene/scripts/feature_fusion/matterport_openseg.py
--- latent_semantics/openscene/scripts/feature_fusion/matterport_openseg.py	2025-10-14 15:43:50.269830197 +0300
+++ origs/openscene/scripts/feature_fusion/matterport_openseg.py	2025-10-14 16:05:52.225870540 +0300
@@ -55,6 +55,7 @@
     n_interval = num_rand_file_per_scene
     n_finished = 0
     for n in range(n_interval):
+
         if exists(join(out_dir, scene_id +'_%d.pt'%(n))):
             n_finished += 1
             print(scene_id +'_%d.pt'%(n) + ' already done!')
@@ -64,7 +65,6 @@

     # short hand for processing 2D features
     device = torch.device('cpu')
-    #device = torch.device('cuda:0')

     # extract image features and keep them in the memory
     # default: False (extract image on the fly)
@@ -136,7 +136,7 @@
     split = args.split
     data_dir = args.data_dir

-    data_root = join(data_dir, 'matterport_3d_40')
+    data_root = join(data_dir, 'matterport_3d')
     data_root_2d = join(data_dir,'matterport_2d')
     args.data_root_2d = data_root_2d
     out_dir = args.output_dir
diff --color -bur latent_semantics/openscene/scripts/preprocess/preprocess_2d_matterport.py origs/openscene/scripts/preprocess/preprocess_2d_matterport.py
--- latent_semantics/openscene/scripts/preprocess/preprocess_2d_matterport.py	2025-10-14 15:50:52.005843067 +0300
+++ origs/openscene/scripts/preprocess/preprocess_2d_matterport.py	2025-10-14 16:05:52.225870540 +0300
@@ -83,12 +83,11 @@
         lines = [line.rstrip() for line in lines]
     return lines

-BASE_DIR = os.environ.get("BASE_DIR")
 #! YOU NEED TO MODIFY THE FOLLOWING
 #####################################
 split = 'train' # 'train' | 'val' | 'test'
 out_dir = '../../data/matterport_2d/'
-in_path = f'{BASE_DIR}/hdd/datasets/matterport3d/v1/scans'  # downloaded original matterport data
+in_path = '../../data/matterport/scans' # downloaded original matterport data
 scene_list = process_txt('../../dataset/matterport/scenes_{}.txt'.format(split))
 #####################################

diff --color -bur latent_semantics/openscene/scripts/preprocess/preprocess_3d_matterport_K_num_classes.py origs/openscene/scripts/preprocess/preprocess_3d_matterport_K_num_classes.py
--- latent_semantics/openscene/scripts/preprocess/preprocess_3d_matterport_K_num_classes.py	2025-10-14 15:50:58.992843281 +0300
+++ origs/openscene/scripts/preprocess/preprocess_3d_matterport_K_num_classes.py	2025-10-14 16:05:52.225870540 +0300
@@ -6,6 +6,7 @@
 import pandas as pd


+
 def process_one_scene(fn):
     '''process one scene.'''

@@ -43,13 +44,12 @@
         lines = [line.rstrip() for line in lines]
     return lines

-BASE_DIR = os.environ.get("BASE_DIR")
 #! YOU NEED TO MODIFY THE FOLLOWING
 #####################################
 split = 'test' # 'train' | 'val' | 'test'
-num_classes = 40 # 40 | 80 | 160 # define the number of classes
+num_classes = 160 # 40 | 80 | 160 # define the number of classes
 out_dir = '../../data/matterport_3d_{}/{}'.format(num_classes, split)
-matterport_path = f'{BASE_DIR}/hdd/datasets/matterport3d/v1/scans'  # downloaded original matterport data
+matterport_path = '/PATH_TO/matterport/scans' # downloaded original matterport data
 tsv_file = '../../dataset/matterport/category_mapping.tsv'
 scene_list = process_txt('../../dataset/matterport/scenes_{}.txt'.format(split))
 #####################################
diff --color -bur latent_semantics/openscene/scripts/preprocess/preprocess_3d_matterport.py origs/openscene/scripts/preprocess/preprocess_3d_matterport.py
--- latent_semantics/openscene/scripts/preprocess/preprocess_3d_matterport.py	2025-10-14 15:51:08.383843567 +0300
+++ origs/openscene/scripts/preprocess/preprocess_3d_matterport.py	2025-10-14 16:05:52.225870540 +0300
@@ -4,7 +4,6 @@
 import plyfile
 import torch
 import pandas as pd
-from tqdm import tqdm

 # Map relevant classes to {0,1,...,19}, and ignored classes to 255
 remapper = np.ones(150) * (255)
@@ -80,28 +79,22 @@
         lines = [line.rstrip() for line in lines]
     return lines

-BASE_DIR = os.environ.get("BASE_DIR")
 #! YOU NEED TO MODIFY THE FOLLOWING
 #####################################
-split = 'train' # 'train' | 'val' | 'test'
-out_dir = '../../data/matterport_3d/{}'.format(split)
-matterport_path = f'{BASE_DIR}/hdd/datasets/matterport3d/v1/scans'  # downloaded original matterport data
+split = 'val' # 'train' | 'val' | 'test'
+out_dir = 'data/matterport_3d/{}'.format(split)
+matterport_path = '/PATH_TO/matterport/scans' # downloaded original matterport data
 tsv_file = '../../dataset/matterport/category_mapping.tsv'
-scene_list = process_txt('../../dataset/matterport/scenes_{}.txt'.format(split))
+scene_list = process_txt('../../dataset/scenes_{}.txt'.format(split))
 #####################################

 os.makedirs(out_dir, exist_ok=True)
 category_mapping = pd.read_csv(tsv_file, sep='\t', header=0)
-
-mping = category_mapping[['nyu40id']].to_numpy()
-mping = mping[~np.isnan(mping)]
-mping = mping.astype(int).flatten()
-mapping = np.insert(mping, 0, 0, axis=0)
+mapping = np.insert(category_mapping[['nyu40id']].to_numpy()
+                        .astype(int).flatten(), 0, 0, axis=0)
 files = []
-for scene in tqdm(scene_list):
-    path = os.path.join(matterport_path, scene, 'region_segmentations', '*.ply')
-    newFile = glob.glob(path)
-    files = files + newFile
+for scene in scene_list:
+    files = files + glob.glob(os.path.join(matterport_path, scene, 'region_segmentations', '*.ply'))

 p = mp.Pool(processes=mp.cpu_count())
 p.map(process_one_scene, files)
Only in latent_semantics/openscene/scripts/preprocess: preprocess_3d_matterport_vlmap_classes.py
Only in latent_semantics/openscene: show_map.py
Only in latent_semantics/openscene: test-eval.sh
Only in latent_semantics/openscene: unique_checker.py
diff --color -bur latent_semantics/openscene/util/metric.py origs/openscene/util/metric.py
--- latent_semantics/openscene/util/metric.py	2025-10-14 15:43:50.269830197 +0300
+++ origs/openscene/util/metric.py	2025-10-14 16:05:52.226870540 +0300
@@ -12,13 +12,17 @@
     assert pred_ids.shape == gt_ids.shape, (pred_ids.shape, gt_ids.shape)
     idxs = gt_ids != UNKNOWN_ID
     if NO_FEATURE_ID in pred_ids:  # some points have no feature assigned for prediction
-        pred_ids[pred_ids == NO_FEATURE_ID] = num_classes
-        bins = np.bincount(pred_ids[idxs] * (num_classes+1) + gt_ids[idxs], minlength=(num_classes+1)**2)
-        confusion = bins.reshape((num_classes+1, num_classes+1)).astype(np.ulonglong)
+        pred_ids[pred_ids==NO_FEATURE_ID] = num_classes
+        confusion = np.bincount(
+            pred_ids[idxs] * (num_classes+1) + gt_ids[idxs],
+            minlength=(num_classes+1)**2).reshape((
+            num_classes+1, num_classes+1)).astype(np.ulonglong)
         return confusion[:num_classes, :num_classes]

-    bins = np.bincount(pred_ids[idxs] * num_classes + gt_ids[idxs], minlength=num_classes**2)
-    return bins.reshape((num_classes, num_classes)).astype(np.ulonglong)
+    return np.bincount(
+        pred_ids[idxs] * num_classes + gt_ids[idxs],
+        minlength=num_classes**2).reshape((
+        num_classes, num_classes)).astype(np.ulonglong)


 def get_iou(label_id, confusion):
@@ -43,16 +47,13 @@
     if 'scannet_3d' in dataset:
         CLASS_LABELS = SCANNET_LABELS_20
     elif 'matterport_3d_40' in dataset:
-        #CLASS_LABELS = MATTERPORT_LABELS_40
-        CLASS_LABELS = VLMAPS_LABELS
+        CLASS_LABELS = MATTERPORT_LABELS_40
     elif 'matterport_3d_80' in dataset:
         CLASS_LABELS = MATTERPORT_LABELS_80
     elif 'matterport_3d_160' in dataset:
         CLASS_LABELS = MATTERPORT_LABELS_160
     elif 'matterport_3d' in dataset:
-        # note: here VLMAPS
         CLASS_LABELS = MATTERPORT_LABELS_21
-        #CLASS_LABELS = VLMAP_LABELS
     elif 'nuscenes_3d' in dataset:
         CLASS_LABELS = NUSCENES_LABELS_16
     else:
@@ -68,12 +69,12 @@
     count = 0
     for i in range(N_CLASSES):
         label_name = CLASS_LABELS[i]
-        if (gt_ids == i).sum() == 0:  # at least 1 point needs to be in the evaluation for this class
+        if (gt_ids==i).sum() == 0: # at least 1 point needs to be in the evaluation for this class
             continue

         class_ious[label_name] = get_iou(i, confusion)
-        class_accs[label_name] = class_ious[label_name][1] / (gt_ids == i).sum()
-        count += 1
+        class_accs[label_name] = class_ious[label_name][1] / (gt_ids==i).sum()
+        count+=1

         mean_iou += class_ious[label_name][0]
         mean_acc += class_accs[label_name]
diff --color -bur latent_semantics/openscene/util/util.py origs/openscene/util/util.py
--- latent_semantics/openscene/util/util.py	2025-10-14 15:43:50.269830197 +0300
+++ origs/openscene/util/util.py	2025-10-14 16:05:52.226870540 +0300
@@ -211,7 +211,6 @@
     elif colormap == 'matterport':
         scannet_palette = []
         for _, value in MATTERPORT_COLOR_MAP_21.items():
-        #for _, value in MATTERPORT_COLOR_MAP_160.items():
             scannet_palette.append(np.array(value))
         palette = np.concatenate(scannet_palette)
     elif colormap == 'matterport_160':
