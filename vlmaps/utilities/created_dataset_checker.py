import os

failure = False

scenes = ["5LpN3gDmAk7_1",
 "gTV8FGcVJC9_1",
 "jh4fc5c5qoQ_1",
 "JmbYfDe2QKZ_1",
 "JmbYfDe2QKZ_2",
 "mJXqzFtmKg4_1",
 "ur6pFq6Qu1A_1",
 "UwV83HsGsw3_1",
 "Vt2qJdWjCF2_1",
 "YmJkqBEsHnH_1"]

yaml_path = "../vlmaps/config/data_paths/default.yaml"

f = open(yaml_path, "r")
lines = f.readlines()
if(len(lines) < 3):
    print("Invalid configuration of file vlmaps/config/data_paths/default.yaml. Please see configuration instructions.")
    exit(1)

try:
    habitat_dir = lines[0].rstrip("\n").lstrip("habitat_scene_dir:").rstrip("\"").lstrip(" \"")
    out_dir = lines[1].lstrip("vlmaps_data_dir:").rstrip("\"").lstrip(" \"") + "vlmaps_dataset"
    map_prefix = lines[2].lstrip("map_prefix:").rstrip("\"").lstrip(" \"")
except:
    print("Invalid configuration of file vlmaps/config/data_paths/default.yaml. Please see configuration instructions.")
    exit(1)

outpath = out_dir
outfound = os.path.isdir(outpath)
if (not outfound):
    print("Missing directory:", outpath)
    exit(1)

for scene in scenes:
    dirpath = out_dir + "/" + scene
    dirfound = os.path.isdir(dirpath)
    if(not dirfound):
        print("Missing directory:", dirpath)
        failure = True

    depth = dirpath + "/" + "depth"
    depthfound = os.path.isdir(depth)
    if(not depthfound):
        print("Missing depth folder", depth)
        failure = True

    regions = dirpath + "/" + "regions"
    regionsfound = os.path.isdir(regions)
    if (not regionsfound):
        print("Missing regions folder", regions)
        failure = True

    semantic = dirpath + "/" + "semantic"
    semanticfound = os.path.isdir(semantic)
    if (not semanticfound):
        print("Missing semantic folder", semantic)
        failure = True

    rgb = dirpath + "/" + "rgb"
    rgbfound = os.path.isdir(rgb)
    if (not rgbfound):
        print("Missing rgb folder", rgb)
        failure = True

    semantic_rt = dirpath + "/" + "semantic-rt"
    semantic_rtfound = os.path.isdir(semantic_rt)
    if (not semantic_rtfound):
        print("Missing semantic-rt folder", semantic_rt)
        failure = True

    vlmap = dirpath + "/" + "vlmap"
    vlmapfound = os.path.isdir(vlmap)
    if (not vlmapfound):
        print("Missing vlmap folder", vlmap)
        failure = True

    map = dirpath + "/" + "vlmap" + f"/{map_prefix}.h5df"
    mapfound = os.path.exists(map)
    if (not mapfound):
        print("Missing map", map)
        failure = True

    postprocess = dirpath + "/" + "vlmap" + f"/{map_prefix}-postprocessed.h5df"
    postprocessfound = os.path.exists(postprocess)
    if (not postprocessfound):
        print("Missing postprocessed map", postprocess)
        failure = True

    instances = dirpath + "/" + "vlmap" + f"/{map_prefix}-instances.h5df"
    instancesfound = os.path.exists(instances)
    if (not instancesfound):
        print("Missing instances map", instances)
        failure = True


if (not failure):
    print("dataset ok!")
    exit(0)
else:
    exit(1)
