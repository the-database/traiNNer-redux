# traiNNer-redux
## Overview
A modern community fork of [BasicSR](https://github.com/XPixelGroup/BasicSR) and [traiNNer-redux](https://github.com/joeyballentine/traiNNer-redux).

## Usage Instructions

### Initial Setup
1. Install [Python](https://www.python.org/) if it's not already installed. A minimum of version Python 3.11 is required.
2. Clone the repository:
   - To use the git command line, navigate to where you want to install traiNNer-redux, and enter this command (install [git](https://git-scm.com/) first if it's not already installed):
      ```
      git clone https://github.com/the-database/traiNNer-redux.git
      ```
   - To use a GUI for git, follow the instructions for that git client. For [GitHub Desktop](https://desktop.github.com/), for example, click on the green Code button near the top of this page, click Open with GitHub Desktop and follow the instructions.
3. For Windows users, double click `install.bat`, and for Linux users, from the terminal in the traiNNer-redux folder run `chmod +x install.sh && ./install.sh` to install all Python dependencies to a new virtual environment.

### Training a Model
Refer to the [wiki](https://github.com/the-database/traiNNer-redux/wiki) for a [full training guide](https://github.com/the-database/traiNNer-redux/wiki/%F0%9F%93%88-Training%E2%80%90a%E2%80%90Model%E2%80%90in%E2%80%90traiNNer%E2%80%90redux) and [benchmarks](https://github.com/the-database/traiNNer-redux/wiki/PyTorch-Inference-Benchmarks-by-Architecture).

#### Do a quick test run
The repository comes with several configs that are ready to use out of the box, as well as a tiny dataset for testing purposes only. To confirm that your PC can run the training software successfully, run the following command from the `traiNNer-redux` folder:

```
venv\Scripts\activate
python train.py --auto_resume -opt ./options/train/SPAN/SPAN.yml
```

You should see the following output within a few minutes, depending on your GPU speed:

```
...
2024-07-02 21:40:56,593 INFO: Model [SRModel] is created.
2024-07-02 21:40:56,668 INFO: Start training from epoch: 0, iter: 0
2024-07-02 21:41:17,816 INFO: [4x_SP..][epoch:  0, iter:     100, lr:(1.000e-04,)] [performance: 4.729] [eta: 14:11:33] l_g_mssim: 1.0000e+00 l_g_percep: 3.5436e+00 l_g_hsluv: 4.3935e-01 l_g_gan: 2.4346e+00 l_g_total: 7.4175e+00 l_d_real: 2.4136e-01 out_d_real: 2.9309e+00 l_d_fake: 5.2773e-02 out_d_fake: -2.4104e+01
```

The last line shows the progress of training after 100 iterations. If you get this far without any errors, your PC is able to train successfully. Press `ctrl+C` to end the training run.

#### Set up config file
1. Navigate to `traiNNer-redux/options/train`, select the architecture you want to train, and open the `yml` file in that folder in a text editor. A text editor that supports YAML syntax highlighting is recommended, such as [VS Code](https://code.visualstudio.com/) or [Notepad++](https://notepad-plus-plus.org/). For example, to train SPAN, open `traiNNer-redux/options/train/SPAN/SPAN.yml`.
2. At the top of the file, set the `name` to the name of the model you want to train. Give it a unique name so you can differentiate it from other training runs.
3. Set the scale depending on what scale you want to train the model on. 2x doubles the width and height of the image, for example. Not all architectures support all scales. Supported scales appear next to the scale in a comment, so `# 2, 4` means the architecture only supports a scale of 2 or 4.
4. Set the paths to your dataset HR and LR images, at `dataroot_gt` and `dataroot_lq` under the `train:` section. The HR images and LR images should match in numer of images and filenames. For each matching LR/HR pair, the image resolutions should work with the selected scale, so if a scale of 2 is selected then each HR must be 2x the resolution of its matching LR image.
5. If you want to enable validation during training, set `val_enabled` to `true` and set the paths to your validation HR and LR images, at `dataroot_gt` and `dataroot_lq` under the `val` section.
6. If you want to use a pretrain model, set the path of the pretrain model at `pretrain_network_g` and remove the `#` to uncomment that line.

#### Run command to start training

Run the following command to start training. Change `./options/train/arch/config.yml` to point to the config file you set up in the previous step.
```
venv\Scripts\activate
python train.py --auto_resume -opt ./options/train/arch/config.yml
```

For example, to train with the SPAN config:
```
venv\Scripts\activate
python train.py --auto_resume -opt ./options/train/SPAN/SPAN.yml
```

To pause training, press `ctrl+C` or close the command window. To resume training, run the same command that was used to start training. The `--auto_resume` flag will resume training from when it was paused.

#### Test models

Models are saved in the `safetensors` format to `traiNNer-redux/experiments/<name>/models`, where `name` is whatever was used in the config file. [chaiNNer](https://github.com/chaiNNer-org/chaiNNer) can be used to run most models. If you want to run the model on images during training to monitor the progress of the model, set up validation in the config file, and find the validation results in `traiNNer-redux/experiments/<name>/visualization`.

The test script can also be used to test trained models, which is required to test models with architectures not yet supported by chaiNNer. For example, to test SPANPlus model, open the config file at `./options/test/SPANPlus/SPANPlus.yml`, and update the following:
1. Edit the `dataroot_lq` option to point to a folder that contains the images you want to run the model on.
2. Make sure the options under `network_g` match the options under `network_g` in the training config file that you used. For example, if you trained `SPANPlus_STS`, then set the type to `SPANPlus_STS` under `network_g` in the test config file as well.
3. Update `pretrain_network_g` to point to the path of the model you want to test.

Then run this command to run the model on the images as specified in the config file:
```
venv\Scripts\activate
python test.py -opt ./options/test/SPANPlus/SPANPlus.yml
```

## Resources
- [OpenModelDB](https://openmodeldb.info/): Repository of AI upscaling models, which can be used as pretrain models to train new models. Models trained with this repo can be submitted to OMDB.
- [chaiNNer](https://github.com/chaiNNer-org/chaiNNer): General purpose tool for AI upscaling and image processing, models trained with this repo can be run on chaiNNer. chaiNNer can also assist with dataset preparation.
- [WTP Dataset Destroyer](https://github.com/umzi2/wtp_dataset_destroyer): Tool to degrade high quality images, which can be used to prepare the low quality images for the training dataset.
- [helpful-scripts](https://github.com/Kim2091/helpful-scripts): Collection of scripts written to improve experience training AI models.
- [Enhance Everything! Discord Server](https://discord.gg/cpAUpDK): Get help training a model, share upscaling results, submit your trained models, and more.

## License and Acknowledgement

traiNNer-redux is released under the [Apache License 2.0](LICENSE.txt). See [LICENSE](LICENSE/README.md) for individual licenses and acknowledgements.

- This repository is a fork of [traiNNer-redux](https://github.com/joeyballentine/traiNNer-redux) which itself is a fork of [BasicSR](https://github.com/XPixelGroup/BasicSR).
- Network architectures are imported from [Spandrel](https://github.com/chaiNNer-org/spandrel).
- The SPANPlus architecture is from [umzi2/SPANPlus](https://github.com/umzi2/SPANPlus) which is a modification of [SPAN](https://github.com/hongyuanyu/SPAN).
- Several enhancements reference implementations from [Corpsecreate/neosr](https://github.com/Corpsecreate/neosr) and its original repo [neosr](https://github.com/muslll/neosr).
- Members of the Enhance Everything Discord server: [Corpsecreate](https://github.com/Corpsecreate), [joeyballentine](https://github.com/joeyballentine), [Kim2091](https://github.com/Kim2091).
