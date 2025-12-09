# traiNNer-redux
![redux3](https://github.com/user-attachments/assets/d107b2fc-6b68-4d3e-b08d-82c8231796cb)

## Overview
[traiNNer-redux](https://trainner-redux.readthedocs.io/en/latest/index.html) is a deep learning training framework for image super resolution and restoration which allows you to train PyTorch models for upscaling and restoring images and videos. NVIDIA graphics card is recommended, but AMD works on Linux machines with ROCm.

## Usage Instructions
Please see the [getting started](https://trainner-redux.readthedocs.io/en/latest/getting_started.html) page for instructions on how to use traiNNer-redux.

## Contributing
Please see the [contributing](https://trainner-redux.readthedocs.io/en/latest/contributing.html) page for more info on how to contribute.

## Resources
- [OpenModelDB](https://openmodeldb.info/): Repository of AI upscaling models, which can be used as pretrain models to train new models. Models trained with this repo can be submitted to OMDB.
- [chaiNNer](https://github.com/chaiNNer-org/chaiNNer): General purpose tool for AI upscaling and image processing, models trained with this repo can be run on chaiNNer. chaiNNer can also assist with dataset preparation.
- [WTP Dataset Destroyer](https://github.com/umzi2/wtp_dataset_destroyer): Tool to degrade high quality images, which can be used to prepare the low quality images for the training dataset.
- [Video Destroyer](https://github.com/Kim2091/video-destroyer): A tool designed to create datasets from a single high quality input video for AI training purposes.
- [helpful-scripts](https://github.com/Kim2091/helpful-scripts): Collection of scripts written to improve experience training AI models.
- [Dataset_Preprocessing](https://github.com/umzi2/Dataset_Preprocessing): A small collection of scripts for initial dataset processing.
- [Enhance Everything! Discord Server](https://discord.gg/cpAUpDK): Get help training a model, share upscaling results, submit your trained models, and more.
- [vs_align](https://github.com/pifroggi/vs_align): Video Alignment and Synchonization for Vapoursynth, tool to align LR and HR datasets.
- [ImgAlign](https://github.com/sonic41592/ImgAlign): Tool for auto aligning, cropping, and scaling HR and LR images for training image based neural networks.
- [Deep Learning Tuning Playbook](https://developers.google.com/machine-learning/guides/deep-learning-tuning-playbook): Document to help you train deep learning models more effectively.

## License and Acknowledgement

traiNNer-redux is released under the [Apache License 2.0](LICENSE.txt). See [LICENSE](LICENSE/README.md) for individual licenses and acknowledgements.

- This repository is a fork of [joeyballentine/traiNNer-redux](https://github.com/joeyballentine/traiNNer-redux) which itself is a fork of [BasicSR](https://github.com/XPixelGroup/BasicSR).
- Network architectures are imported from [Spandrel](https://github.com/chaiNNer-org/spandrel).
- Several architectures are developed by [umzi2](https://github.com/umzi2): [ArtCNN-PyTorch](https://github.com/umzi2/ArtCNN-PyTorch), [DUnet](https://github.com/umzi2/DUnet), [FlexNet](https://github.com/umzi2/FlexNet), [GaterV3](https://github.com/umzi2/GaterV3), [MetaGan](https://github.com/umzi2/MetaGan), [MoESR](https://github.com/umzi2/MoESR), [MoSR](https://github.com/umzi2/MoSR), [RTMoSR](https://github.com/rewaifu/RTMoSR), [SPANPlus](https://github.com/umzi2/SPANPlus)
- The [ArtCNN](https://github.com/Artoriuz/ArtCNN) architecture is originally developed by [Artoriuz](https://github.com/Artoriuz).
- The TSCUNet architecture is from [aaf6aa/SCUNet](https://github.com/aaf6aa/SCUNet) which is a modification of [SCUNet](https://github.com/cszn/SCUNet), and parts of the training code for TSCUNet are adapted from [TSCUNet_Trainer](https://github.com/Demetter/TSCUNet_Trainer).
- The [FDAT](https://github.com/stinkybread/FDAT) architecture is created by [stinkybread (sharekhan)](https://github.com/stinkybread).
- The [TSPAN](https://github.com/Kim2091/TSPAN) and [TSPANv2](https://github.com/Kim2091/TSPANv2) architectures are temporally consistent variants of the SPAN architecture created by [Kim2091](https://github.com/Kim2091). TSPANv2 is an enhanced version of TSPAN with major improvements to temporal stability, Use [Vapourkit](https://github.com/Kim2091/vapourkit) to upscale videos with TSPAN and TSPANv2 models.
- The [DIS](https://github.com/Kim2091/DIS) architecture is an ultra lightweight arch created by [Kim2091](https://github.com/Kim2091).
- Several enhancements reference implementations from [Corpsecreate/neosr](https://github.com/Corpsecreate/neosr) and its original repo [neosr](https://github.com/muslll/neosr).
- Members of the Enhance Everything Discord server: [Corpsecreate](https://github.com/Corpsecreate), [joeyballentine](https://github.com/joeyballentine), [Kim2091](https://github.com/Kim2091).
