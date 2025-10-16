# Do Visual-Language Grid Maps Capture Latent Semantics?

<img src="img/header.png" alt="header-image" class="center" width="400"/>

**Authors:** [Matti Pekkanen](https://research.aalto.fi/en/persons/matti-pekkanen) (<matti.pekkanen@aalto.fi>), [Tsvetomila Mihaylova](https://research.aalto.fi/en/persons/tsvetomila-mihaylova), [Francesco Verdoja](https://research.aalto.fi/en/persons/francesco-verdoja), and [Ville Kyrki](https://research.aalto.fi/en/persons/ville-kyrki)\
**Affiliation:** School of Electrical Engineering, Aalto University, Finland

Visual-language models (VLMs) have recently been introduced in robotic mapping using the latent representations, i.e., embeddings, of the VLMs to represent semantics in the map. They allow moving from a limited set of human-created labels toward open-vocabulary scene understanding, which is very useful for robots when operating in complex real-world environments and interacting with humans. While there is anecdotal evidence that maps built this way support downstream tasks, such as navigation, rigorous analysis of the quality of the maps using these embeddings is missing.
In this paper, we propose a way to analyze the quality of maps created using VLMs. We investigate two critical properties of map quality: queryability and distinctness. The evaluation of queryability addresses the ability to retrieve information from the embeddings. We investigate intra-map distinctness to study the ability of the embeddings to represent abstract semantic classes and inter-map distinctness to evaluate the generalization properties of the representation.
We propose metrics to evaluate these properties and evaluate two state-of-the-art mapping methods, VLMaps and OpenScene, using two encoders, LSeg and OpenSeg, using real-world data from the Matterport3D data set. Our findings show that while 3D features improve queryability, they are not scale invariant, whereas image-based embeddings generalize to multiple map resolutions. This allows the image-based methods to maintain smaller map sizes, which can be crucial for using these methods in real-world deployments. Furthermore, we show that the choice of the encoder has an effect on the results. The results imply that properly thresholding open-vocabulary queries is an open problem.

## Installation

The repository consists of three subrepositores: [vlmaps](https://github.com/vlmaps/vlmaps), [openscene](https://github.com/pengsongyou/openscene), [openscene-lseg](https://github.com/pengsongyou/lseg_feature_extraction). Follow the installation steps of the original works. Follow the dataset acquisition steps from [vlmaps](https://github.com/vlmaps/vlmaps) repository.

After installation, setup the environment via setup_environment.sh. The evaluation is performed in the vlmaps repository. First create all the maps with the three repositories, then run the analysis.

```bash
    # Create maps - VLMaps
    cd vlmaps
    ./setup_environment.sh
    ./0_1_run_benchmark_vlmaps.sh
    cd ..
    # Create maps - OpenScene
    # Fuse OpenSeg features (steps 1-3)
    cd openscene
    ./A_part1.sh
    cd ..
    # Fuse LSeg features
    cd openscene-lseg
    ./1.sh
    ./2.sh
    cd ..
    # Eval OpenScene maps and parse
    cd openscene
    ./A_part2.sh
    cd ..
    # Analyze results
    cd vlmaps
    ./run_C_analyze.sh
    cd ..
```

## Citation

If you find this work useful, please consider citing:

```bibtex
@inproceedings{pekkanen_2025_latent_semantics,
	title = {Do Visual-Language Grid Maps Capture Latent Semantics?},
	author = {Pekkanen, Matti and Verdoja, Francesco and Mihaylova, Tsvetomila and Kyrki, Ville},
	year = {2025},
	month = {October},
	booktitle = {IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
	publisher = {IEEE},
	address = {Hangzhou, China},
	pages = {XX--YY},
	isbn = {XXX},
	doi = {XXX},
}

```

## Acknowledgements

This software is built upon on the original works of [vlmaps](https://github.com/vlmaps/vlmaps), [openscene](https://github.com/pengsongyou/openscene), [openscene-lseg](https://github.com/pengsongyou/lseg_feature_extraction).

```bibtex
@inproceedings{huang23vlmaps,
               title={Visual Language Maps for Robot Navigation},
               author={Chenguang Huang and Oier Mees and Andy Zeng and Wolfram Burgard},
               booktitle = {Proceedings of the IEEE International Conference
                            on Robotics and Automation (ICRA)},
               year={2023},
               address = {London, UK}
}

@inproceedings{Peng2023OpenScene,
  title     = {OpenScene: 3D Scene Understanding with Open Vocabularies},
  author    = {Peng, Songyou and Genova, Kyle and Jiang, Chiyu "Max" and Tagliasacchi,
               Andrea and Pollefeys, Marc and Funkhouser, Thomas},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern
               Recognition (CVPR)},
  year      = {2023}
}

```

This work was supported by Business Finland (decision 9249/31/2021), the Research Council of Finland (decision 354909), Wallenberg AI, Autonomous Systems and Software Program, WASP and Saab AB. We gratefully acknowledge the support of NVIDIA Corporation with the donation of the Titan Xp GPUs used for this research.
