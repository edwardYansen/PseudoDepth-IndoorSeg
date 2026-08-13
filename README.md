# PseudoDepth-IndoorSeg

Benchmarking and evaluating pseudo-depth integration across modern deep learning frameworks (CNNs / Transformers) for indoor semantic segmentation on the NYUv2 dataset.

## Summary
This repository contains research code and notebooks that integrate monocular pseudo-depth cues into semantic segmentation models and evaluate their effect on the NYUv2 indoor dataset. Pseudo-depth is generated inside the project code and notebooks, so there is no separate pseudo-depth preprocessing step required.

## Table of Contents
- Overview
- Requirements
- Installation (pixi)
- Dataset (NYUv2)
- Usage
- Experiments and metrics
- Results and logging
- Contributing
- License
- Contact

## Overview
PseudoDepth-IndoorSeg is a research-oriented collection of notebooks and modules to test how adding pseudo-depth influences semantic segmentation performance on NYUv2. The project contains multiple model notebooks (each demonstrates a different backbone / fusion strategy) so you can try different approaches interactively.

## Requirements
- Python 3.8+
- PyTorch (compatible with your CUDA or CPU setup)
- torchvision
- numpy, scipy
- OpenCV (cv2), Pillow
- matplotlib, tensorboard (optional)

Pin exact versions in a dependency file (requirements.txt or a pixi lockfile) in the repo.

## Installation (pixi)
This project uses pixi for environment and dependency management (not conda). Install dependencies using your usual pixi workflow and the dependency files in the repository (e.g., requirements.txt, pixi.lock, pixi.toml). If you want, provide the exact pixi CLI commands you use and I can add them to this README verbatim.

## Dataset (NYUv2)
This project uses the NYUv2 dataset only. Use the extraction and loader scripts included under the module/nyuv2_python_toolkit_master directory to prepare the dataset. See module/nyuv2_python_toolkit_master/README.md for extraction instructions and expected folder layout.

Note: you will need to download the NYUv2 raw files (labels and images) from the official sources before running the extraction scripts.

## Usage
Open the notebooks provided for each model and run them interactively. Each notebook demonstrates a model and how pseudo-depth is incorporated (the pseudo-depth generation and fusion are implemented in the code). Typical workflow:
- Open the notebook for the model you want to try (notebooks/ or model-specific ipynb files).
- Follow the top cells to set dataset paths and dependency imports.
- Run the cells to prepare data, train or evaluate the model, and visualize results.

There is no separate pseudo-depth generation step to run manually—the notebooks and code will create / cache pseudo-depth internally when needed.

## Experiments and metrics
Typical experiments compare RGB-only baselines to models that incorporate pseudo-depth with different fusion strategies.

Reported metric:
- mean Intersection-over-Union (mIoU)

Record experiment configs, seeds, and checkpoints for reproducibility.

## Results and logging
Keep quantitative tables and qualitative visualizations in a results/ folder or in notebook outputs. Include experiment metadata (config, seed, dataset split, checkpoint) alongside results.

## Contributing
Contributions welcome. Open issues for bug reports or feature requests, and submit PRs for new models, fusion modules, or dataset utilities.

## License
Add a LICENSE file to the repository root and specify the project's license (e.g., MIT, Apache-2.0).

## Contact
Maintainer: Edward Yansen (https://github.com/edwardYansen)
For questions or collaboration, open an Issue in this repository.
