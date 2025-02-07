# Diffusion 3D Features (Diff3F): Decorating Untextured Shapes with Distilled Semantic Features [CVPR 2024]
<a href='https://diff3f.github.io/'><img src='https://img.shields.io/badge/Project-Page-green'></a>  [![ArXiv](https://img.shields.io/badge/arXiv-2311.17024-b31b1b.svg)](https://arxiv.org/abs/2311.17024)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/diffusion-3d-features-diff3f-decorating/3d-dense-shape-correspondence-on-shrec-19)](https://paperswithcode.com/sota/3d-dense-shape-correspondence-on-shrec-19?p=diffusion-3d-features-diff3f-decorating) <a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
![](assets/teaser.jpg)

[Project Webpage](https://diff3f.github.io/) | [Paper](https://arxiv.org/abs/2311.17024)


## Setup
```shell
conda env create -f environment.yaml
conda activate diff3f
```

### Additional prerequisites
[Install pytorch3d](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md)

```shell
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
```

You might face difficulty in installing pytorch3d or encounter the error `ModuleNotFoundError: No module named 'pytorch3d` during run time. Unfortunately, this is because pytorch3d could not be installed properly. Please refer [here](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md) for alternate ways to install pytorch3d. 

## Usage
Please check the example notebook [test_correspondence.ipynb](test_correspondence.ipynb) for details on computing features for a mesh and finding correspondence/part segmentations. 

## Evaluation
Much of the code for evaluation is based upon [SE-ORNet](https://github.com/OpenSpaceAI/SE-ORNet) with some tweaks to decouple it from the method and make it work for precomputed correspondence (there is no change in the computation of core metrics).
Building the environment for evaluation can be painful as it involves building multiple CUDA packages for 3D, for installation please refer to [SE-ORNet](https://github.com/OpenSpaceAI/SE-ORNet/blob/main/setup.sh) or [DPC](https://github.com/dvirginz/DPC/blob/main/installation.sh). If you can get their code to run, this will also work flawlessly. The main issue is usually to get diffusers and pytorch3d working with the dependencies mentioned in SE-ORNet/DPC but this is not essential as you can keep two separate environemnts-- one to extract mesh features and another to perform the evaluation. Therefore, building the environment from SE-ORNet/DPC alone might be enough.

I have attched my [eval_environment](https://github.com/niladridutt/Diffusion-3D-Features/blob/main/eval_environment.yaml) but it may not work for you. 

```shell
# Extract features for SHREC'19 using
python extract_shrec.py

# Once features are extracted, run evaluation using
python evaluate_pipeline_shrec.py
```

```shell
# Extract features for TOSCA using
python extract_tosca.py

# Once features are extracted, run evaluation using
python evaluate_pipeline_tosca.py
```


## Additional Notes

The meshes provided in the [meshes](https://github.com/niladridutt/Diffusion-3D-Features/tree/main/meshes) directory are provided as examples from various sources and we do not claim any copyright.

## BibTeX

If you find our research useful, please consider citing it as follows.

```bibtex
@InProceedings{Dutt_2024_CVPR,
    author    = {Dutt, Niladri Shekhar and Muralikrishnan, Sanjeev and Mitra, Niloy J.},
    title     = {Diffusion 3D Features (Diff3F): Decorating Untextured Shapes with Distilled Semantic Features},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2024},
    pages     = {4494-4504}
} 
``` 
