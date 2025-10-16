import argparse
import numpy as np

###########################################
# MAIN
###########################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser("./2d_map_to_analysis_file.py")
    parser.add_argument(
        '--embeddings_file', '-e',
        dest="embeddings_file",
        type=str,
        required=True
    )
    parser.add_argument(
        '--labels_file', '-l',
        dest="labels_file",
        type=str,
        required=True
    )
    parser.add_argument(
        '--output_path', '-o',
        dest="output_path",
        type=str,
        required=True
    )
    parser.add_argument(
        '--width', '-w',
        dest="width",
        type=int,
        required=False,
        default=1000
    )
    parser.add_argument(
        '--height', '-ht',
        dest="height",
        type=int,
        required=False,
        default=1000
    )
    parser.add_argument(
        '--embedding_size', '-es',
        dest="embedding_size",
        type=int,
        required=False,
        default=512
    )
    parser.add_argument(
        '--trim_zeros', '-z',
        dest="trim_zeros",
        action="store_true",
        required=False,
        default=True
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda"],
        default="cuda",
        help="Use cpu or cuda.",
    )

    FLAGS, unparsed = parser.parse_known_args()
    embeddings_file = FLAGS.embeddings_file
    labels_file = FLAGS.labels_file
    output_path = FLAGS.output_path
    width = FLAGS.width
    height = FLAGS.height
    embedding_size = FLAGS.embedding_size
    trim_zeros = FLAGS.trim_zeros
    device = FLAGS.device

    if embeddings_file.endswith(".npy"):
        with open(embeddings_file, "rb") as f:
            embeddings = np.load(f)
    else:
        embeddings = np.fromfile(embeddings_file, "float32")

    if labels_file.endswith(".npy"):
        with open(labels_file, "rb") as f:
            labels = np.load(f)
    else:
        labels = np.fromfile(labels_file, "float32")

    print(f"Loaded: emb: {embeddings.shape}, labels: {labels.shape}")
    print(f"Labels min-max: {np.unique(labels).min()}-{np.unique(labels).max()}")

    embeddings = embeddings.reshape(height, width, embedding_size)
    labels = labels.reshape(height, width, 1)
    print(f"Reshaped: emb: {embeddings.shape}, labels: {labels.shape}")

    # Remove the empty space on the borders of the map
    if trim_zeros:
        x_indices, y_indices = np.where(embeddings.sum(-1) != 0)
        xmin = np.min(x_indices)
        xmax = np.max(x_indices)
        ymin = np.min(y_indices)
        ymax = np.max(y_indices)
        print(f"nonzeros: {xmin}:{xmax} x {ymin}:{ymax}")

        embeddings = embeddings[xmin:xmax+1, ymin:ymax+1, :]
        labels = labels[xmin:xmax+1, ymin:ymax+1, :]

        print(f"Zeros trimmed: emb: {embeddings.shape}, labels: {labels.shape}")

    embeddings = embeddings.reshape(-1, embedding_size)
    labels = labels.reshape(-1, 1).squeeze(1)
    print(f"Flattened: emb: {embeddings.shape}, labels: {labels.shape}")

    merged = np.full((embeddings.shape[0], 2, embedding_size), -1, dtype=np.float32)
    merged[:,0,0] = labels
    merged[:,1,:] = embeddings
    print(f"Merged: {merged.shape}")

    np.save(output_path, merged)
    print("Done.")
