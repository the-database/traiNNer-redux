# Inference Speed vs VRAM vs PSNR Charts

All charts were generated using [generate_charts.py](https://github.com/the-database/traiNNer-redux/blob/master/scripts/benchmarking/generate_charts.py). For this benchmark data in tabular format please see the [Benchmarks](/benchmarks) page. An interactive tableau chart of this data is also available [here](https://public.tableau.com/app/profile/vaibhav.bhat1737/viz/shared/R9CDXNN7X), created by Enhance Everything Discord member SharekhaN.

Architectures are separated into groups based on their inference speed. The grouping is arbitrary and shouldn't be considered as any official categorization. The groups are currently defined as follows:
- Small: On 4x model, inference speed of 640x480 input on RTX 4090 is more than 24 fps.
- Medium: On 4x model, inference speed of 640x480 input on RTX 4090 is between 2 and 24 fps.
- Large: On 4x model, inference speed of 640x480 input on RTX 4090 is less than 2 fps.

Only architectures which have metrics with the training set DF2K and validation set Urban100 are shown.

VRAM is depicted by the size of the shaded circle behind each dot, larger means higher VRAM consumption.

## 4x
### 4x Large Architectures
![large4x](resources/benchmark4x_large.png)

### 4x Medium Architectures
![medium4x](resources/benchmark4x_medium.png)

### 4x Small Architectures
![small4x](resources/benchmark4x_small.png)


## 3x
### 3x Large Architectures
![large3x](resources/benchmark3x_large.png)

### 3x Medium Architectures
![medium3x](resources/benchmark3x_medium.png)

### 3x Small Architectures
![small3x](resources/benchmark3x_small.png)

## 2x
### 2x Large Architectures
![large2x](resources/benchmark2x_large.png)
### 2x Medium Architectures
![medium2x](resources/benchmark2x_medium.png)
### 2x Small Architectures
![small2x](resources/benchmark2x_small.png)
