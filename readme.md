# WSI Normalizer: color stain normalization tool for whole slide images
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
## Description
Repository for normalizing whole slide images (WSI) patches.

**Support for:**
1. [Reinhard](https://ieeexplore.ieee.org/abstract/document/946629/) Reinhard, Erik, et al. "Color transfer between images." IEEE Computer graphics and applications 21.5 (2001): 34-41.
2. [Macenko](https://ieeexplore.ieee.org/abstract/document/5193250) Macenko, Marc, et al. "A method for normalizing histology slides for quantitative analysis." 2009 IEEE international symposium on biomedical imaging: from nano to macro. IEEE, 2009.
3. [Vahadane](https://ieeexplore.ieee.org/abstract/document/7460968/) Vahadane, Abhishek, et al. "Structure-preserving color normalization and sparse stain separation for histological images." IEEE transactions on medical imaging 35.8 (2016): 1962-1971.

## Installation
1. run `pip install -r requirements.txt` for installing dependencies.

2. make sure you have the following folder structure:
```
├── data
│   ├── slide_1
│   │   ├── patch_1.png
│   │   ├── patch_2.png
│   │   ├── ...
│   ├── slide_2
│   │   ├── patch_1.png
│   │   ├── patch_2.png
│   │   ├── ...
│   ├── ...
│   └── slide_n
│       ├── ...
│       └── patch_n.png
```
Note that png or jpg images are supported

3.  modify the following arguments: `<TARGET FOLDER> <OUTPUT FOLDER> <NORM METHOD> <TARGET IMAGE>`, and run:
```bash
python main.py --target_dir <TARGET FOLDER> --output_dir <OUTPUT FOLDER> --method <NORM METHOD> --target_img <TARGET IMAGE>
```
make sure the `<NORM METHOD>` is in **reinhard**, **macenko**, **vahadane**

4. each slide will be normalized and saved in the output folder with the same name as the original slide, with a suffix of `.jpg`.