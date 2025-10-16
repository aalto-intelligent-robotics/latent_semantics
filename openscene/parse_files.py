import argparse
import numpy as np
import os
import glob
from tqdm import tqdm

mapids = {
    '5LpN3gDmAk7': 0,
    'gTV8FGcVJC9': 1,
    'jh4fc5c5qoQ': 2,
    'JmbYfDe2QKZ': 3,
    'mJXqzFtmKg4': 5,
    'ur6pFq6Qu1A': 6,
    'UwV83HsGsw3': 7,
    'Vt2qJdWjCF2': 8,
    'YmJkqBEsHnH': 9,
}

def find_files_with_string(folder_path, search_string):
    found_files = []
    # Iterate through all files and directories in the specified folder
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            # Check if the search string is present in the filename
            if search_string in file:
                found_files.append(os.path.join(root, file))
    return found_files

def find_file_in_folder(folder_path, file_name):
    for file in os.listdir(folder_path):
        if file == file_name:
            return os.path.join(folder_path, file)
    return None

def main():
    parser = argparse.ArgumentParser("./parse_files.py")
    parser.add_argument(
        '--input', '-i',
        dest="file_path",
        type=str,
        required=False,
        default=""
    )
    parser.add_argument(
        '--embeddings', '-e',
        dest="embeddings_path",
        type=str,
        required=False,
        default=""
    )
    parser.add_argument(
        '--embedding_size', '-es',
        dest="embedding_size",
        type=int,
        required=False,
        default=512
    )
    parser.add_argument(
        '--output', '-o',
        dest="output_file_path",
        type=str,
        required=False,
        default=""
    )
    FLAGS, unparsed = parser.parse_known_args()
    file_path = FLAGS.file_path
    embeddings_path = FLAGS.embeddings_path
    embedding_size = FLAGS.embedding_size
    output_file_path = FLAGS.output_file_path

    for mapid in tqdm(mapids, leave=False, desc="maps"):
        files = sorted(find_files_with_string(file_path, mapid))
        first = True
        file_i = 0
        for file in tqdm(files, leave=False, desc="files"):
            path, file_name = os.path.split(file)
            efile = find_file_in_folder(embeddings_path, file_name)

            d = np.load(file)
            embeddings_temp = np.load(efile)

            if(first):
                gt = d[0, :]
                pred = d[1, :]
                grid = d[2:5, :]
                instances = d[5, :]
                embeddings = embeddings_temp
                first = False
            else:
                gt_temp = d[0, :]
                pred_temp = d[1, :]
                grid_temp = d[2:5, :]
                instances_temp = d[5, :]
                instances_temp[instances_temp >= 0] = instances_temp[instances_temp >= 0] + (file_i * 1000)
                gt = np.concatenate((gt, gt_temp))
                pred = np.concatenate((pred, pred_temp))
                grid = np.concatenate((grid, grid_temp),axis=1)
                instances = np.concatenate((instances, instances_temp))
                embeddings = np.concatenate((embeddings, embeddings_temp))
            file_i += 1

        path = output_file_path + "/" + mapid + "_gt.data"
        file_exists = os.path.exists(path)
        if(file_exists):
            print(path, "already exists")
        else:
            np.save(path, gt)

        path = output_file_path + "/" + mapid + "_semantic.data"
        file_exists = os.path.exists(path)
        if(file_exists):
            print(path, "already exists")
        else:
            np.save(path, pred)

        path = output_file_path + "/" + mapid + "_grid.data"
        file_exists = os.path.exists(path)
        if(file_exists):
            print(path, "already exists")
        else:
            np.save(path, grid)

        path = output_file_path + "/" + mapid + "_instance.data"
        file_exists = os.path.exists(path)
        if(file_exists):
            print(path, "already exists")
        else:
            np.save(path, instances)

        path = output_file_path + "/" + mapid + "_embedding.data"
        file_exists = os.path.exists(path)
        if(file_exists):
            print(path, "already exists")
        else:
            np.save(path, embeddings)


if __name__ == '__main__':
    main()