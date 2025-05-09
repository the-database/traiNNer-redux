## Introduction

### traiNNer-redux
traiNNer-redux is a deep learning training framework for image super resolution and restoration which allows you to train PyTorch models for upscaling and restoring images and videos. NVIDIA graphics card is recommended, but AMD works on Linux machines with ROCm.

### Applications

traiNNer-redux allows you to train customized image super resolution and restoration models optimized for a specific task. Once you have trained a model, you can upscale low resolution or compressed web images, videos, or game textures using a tool like such as:
   - [chaiNNer](https://github.com/chaiNNer-org/chaiNNer): for images including game textures
   - [VideoJaNai](https://github.com/the-database/VideoJaNai): to save upscaled videos
   - [mpv-upscale-2x_animejanai](https://github.com/the-database/mpv-upscale-2x_animejanai): to view upscaled videos in real-time
   - [Magpie](https://github.com/Blinue/Magpie): to upscale any window to fullscreen
   - [MangaJaNaiConverterGui](https://github.com/the-database/MangaJaNaiConverterGui): to upscale manga archives

### Do I really need traiNNer-redux?

Several models trained by members of the community already exist and work well for their intended task. Many of those models are available to download from [OpenModelDB](https://openmodeldb.info/). It's a good idea to try finding and testing available models to see if they already work well for your task. Consider using traiNNer-redux to train your own model only if existing models don't work well enough for your task, or if you just want to learn how model training works.
