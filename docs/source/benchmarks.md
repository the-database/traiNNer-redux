# PyTorch Inference Benchmarks by Architecture (AMP & channels last)

All benchmarks were generated using [benchmark_archs.py](https://github.com/the-database/traiNNer-redux/blob/master/scripts/benchmarking/benchmark_archs.py). The benchmarks were done on a Windows 11 PC with RTX 4090 + i9-13000K.

Note that these benchmarks only measure the raw inference step of these architectures. In practice, several other factors may contribute to results not matching the benchmarks shown here. For example, when comparing two architectures with the same inference speed but one has double the VRAM usage, the one with less VRAM usage will be faster to upscale with for larger images, because the one with higher VRAM usage would require tiling to avoid running out of VRAM in order to upscale a large image while the one with lower VRAM usage could upscale the entire image at once without tiling.

PSNR and SSIM scores are a rough measure of quality, higher is better. These scores should not be taken as an absolute that one architecture is better than another. Metrics are calculated using the officially released models optimized on L1 loss, and are trained on either the DF2K or DIV2K training dataset. When comparing scores between architectures, only compare within the same dataset, so only compare DF2K scores with DF2K scores or DIV2K scores with DIV2K scores. DF2K scores are typically higher than DIV2K scores on the same architecture. PSNR and SSIM are calculated on the Y channel of the Urban100 validation dataset, one of the standard research validation sets.
## By Scale

### 4x scale
(1, 3, 480, 640) input, 1 warmup + 5 runs averaged
|Name|FPS|FPS ({abbr}`CL (channels_last)`)|{abbr}`CL (channels_last)` vs base|sec/img|VRAM ({abbr}`CL (channels_last)`)|PSNR (DF2K)|SSIM (DF2K)|PSNR (DIV2K)|SSIM (DIV2K)|Params|
|:-|-:|-:|-:|-:|-:|-:|-:|-:|-:|-:|
|superultracompact  4x fp16         |  730.61|  851.05|1.16x|  0.0014|    0.18 GB|-    |-     |-    |-     |    52,968|
|ultracompact  4x fp16              |  246.21|  359.57|1.46x|  0.0041|    0.18 GB|-    |-     |-    |-     |   325,488|
|compact  4x fp16                   |  136.54|  210.66|1.54x|  0.0073|    0.18 GB|-    |-     |-    |-     |   621,424|
|spanplus_sts  4x fp16              |  133.57|  154.85|1.16x|  0.0075|    0.23 GB|-    |-     |-    |-     |   693,160|
|spanplus_st  4x fp16               |   99.32|  105.79|1.07x|  0.0101|    0.32 GB|-    |-     |-    |-     | 2,236,728|
|span  4x fp16                      |   94.39|   95.91|1.02x|  0.0106|    0.60 GB|26.18|0.7879|-    |-     | 2,236,728|
|artcnn_r8f64  4x fp16              |   91.62|  138.61|1.51x|  0.0109|    0.23 GB|-    |-     |-    |-     |   952,688|
|spanplus_s  4x fp16                |   74.76|   67.28|0.90x|  0.0134|    1.55 GB|-    |-     |-    |-     |   687,707|
|spanplus  4x fp16                  |   54.73|   50.32|0.92x|  0.0183|    2.15 GB|-    |-     |-    |-     | 2,228,507|
|mosr_t  4x fp16                    |   53.80|   64.10|1.19x|  0.0186|    0.36 GB|-    |-     |-    |-     |   609,888|
|realcugan  4x fp16                 |   44.12|   61.28|1.39x|  0.0227|    0.95 GB|-    |-     |-    |-     | 1,406,812|
|plksr_tiny  4x fp16                |   36.96|   49.55|1.34x|  0.0271|    0.27 GB|26.34|0.7942|26.12|0.7888| 2,370,544|
|man_tiny  4x fp16                  |   35.02|   27.37|0.78x|  0.0286|    0.43 GB|25.84|0.7786|-    |-     |   150,024|
|lmlt_tiny  4x fp16                 |   28.70|   28.48|0.99x|  0.0348|    0.37 GB|26.08|0.7838|-    |-     |   251,040|
|artcnn_r16f96  4x fp16             |   17.38|   30.74|1.77x|  0.0575|    0.34 GB|-    |-     |-    |-     | 4,113,168|
|cfsr  4x bf16                      |   16.76|   14.21|0.85x|  0.0597|    0.48 GB|26.21|0.7897|-    |-     |   306,912|
|lmlt_base  4x fp16                 |   15.08|   15.06|1.00x|  0.0663|    0.63 GB|26.44|0.7949|-    |-     |   671,808|
|plksr  4x fp16                     |   11.95|   15.74|1.32x|  0.0836|    0.29 GB|26.85|0.8097|26.69|0.8054| 7,386,096|
|lmlt_large  4x fp16                |   10.33|   10.30|1.00x|  0.0968|    0.87 GB|26.63|0.8001|-    |-     | 1,295,328|
|metaflexnet  4x fp16               |    9.28|    9.63|1.04x|  0.1078|    1.29 GB|-    |-     |-    |-     |38,053,568|
|mosr  4x fp16                      |    9.27|   11.44|1.23x|  0.1079|    0.49 GB|-    |-     |-    |-     | 4,287,600|
|esrgan_lite  4x fp16               |    8.74|   12.17|1.39x|  0.1144|    1.24 GB|-    |-     |-    |-     | 5,021,155|
|scunet_aaf6aa  4x fp16             |    8.52|   10.92|1.28x|  0.1174|    3.37 GB|-    |-     |-    |-     |15,207,468|
|realplksr pixelshuffle layer_norm=True 4x fp16|    8.43|   10.06|1.19x|  0.1186|    0.37 GB|-    |-     |-    |-     | 7,389,680|
|eimn_a  4x fp16                    |    8.36|    6.09|0.73x|  0.1196|    0.89 GB|26.68|0.8027|-    |-     |   880,870|
|realplksr dysample layer_norm=True 4x fp16|    7.86|    9.09|1.16x|  0.1272|    2.21 GB|-    |-     |-    |-     | 7,402,243|
|omnisr  4x fp16                    |    7.80|    7.64|0.98x|  0.1283|    1.18 GB|26.95|0.8105|26.64|0.8018|   805,376|
|eimn_l  4x fp16                    |    7.32|    5.34|0.73x|  0.1366|    0.89 GB|26.88|0.8084|-    |-     | 1,002,496|
|man_light  4x fp16                 |    6.02|    4.42|0.73x|  0.1660|    0.53 GB|26.70|0.8052|-    |-     |   842,892|
|realplksr pixelshuffle layer_norm=False 4x fp16|    5.83|    6.13|1.05x|  0.1716|    0.44 GB|-    |-     |-    |-     | 7,389,680|
|realplksr dysample layer_norm=False 4x fp16|    5.55|    5.77|1.04x|  0.1801|    2.21 GB|-    |-     |-    |-     | 7,402,243|
|rcanspatialselayer  4x fp16        |    5.38|    7.23|1.34x|  0.1860|    1.40 GB|-    |-     |-    |-     |15,489,355|
|rcan  4x fp16                      |    5.36|    7.02|1.31x|  0.1866|    1.40 GB|-    |-     |26.82|0.8087|15,592,379|
|hit_sir  4x bf16                   |    5.26|    5.12|0.97x|  0.1900|    1.34 GB|-    |-     |26.71|0.8045|   791,540|
|camixersr  4x bf16                 |    4.78|    3.61|0.75x|  0.2091|    1.13 GB|-    |-     |26.63|0.8012|   765,322|
|hit_srf  4x bf16                   |    4.23|    4.19|0.99x|  0.2364|    1.34 GB|-    |-     |26.80|0.8069|   866,420|
|hit_sng  4x bf16                   |    4.09|    3.92|0.96x|  0.2444|    1.34 GB|-    |-     |26.75|0.8053| 1,032,060|
|moesr2  4x fp16                    |    3.93|    3.68|0.94x|  0.2542|    0.91 GB|-    |-     |-    |-     |16,547,008|
|flexnet  4x fp16                   |    3.45|    3.38|0.98x|  0.2900|    1.30 GB|-    |-     |-    |-     | 3,045,136|
|esrgan use_pixel_unshuffle=True 4x fp16|    3.28|    4.60|1.40x|  0.3051|    2.48 GB|27.03|0.8153|26.73|0.8072|16,697,987|
|esrgan use_pixel_unshuffle=False 4x fp16|    3.27|    4.60|1.40x|  0.3054|    2.48 GB|27.03|0.8153|26.73|0.8072|16,697,987|
|man  4x fp16                       |    1.26|    0.86|0.68x|  0.7928|    1.53 GB|27.26|0.8197|-    |-     | 8,712,612|
|atd_light  4x bf16                 |    1.22|    1.24|1.02x|  0.8189|    3.30 GB|26.97|0.8107|-    |-     |   768,792|
|srformer_light  4x bf16            |    1.04|    1.05|1.01x|  0.9627|    1.47 GB|26.67|0.8032|-    |-     |   872,748|
|swin2sr_s  4x bf16                 |    1.02|    1.02|1.00x|  0.9849|    1.45 GB|-    |-     |-    |-     | 1,024,824|
|swinir_s  4x bf16                  |    0.99|    0.98|0.99x|  1.0130|    1.55 GB|26.47|0.7980|-    |-     |   929,628|
|hat_s  4x bf16                     |    0.63|    0.66|1.05x|  1.5988|    9.85 GB|27.87|0.8346|-    |-     | 9,621,183|
|dat_light  4x bf16                 |    0.62|    0.62|0.99x|  1.6029|    2.71 GB|26.64|0.8033|-    |-     |   572,766|
|swinir_m  4x bf16                  |    0.62|    0.61|0.99x|  1.6084|    2.64 GB|27.45|0.8254|-    |-     |11,900,199|
|swin2sr_m  4x bf16                 |    0.62|    0.60|0.97x|  1.6154|    2.51 GB|27.51|0.8271|-    |-     |12,239,283|
|dat_s  4x bf16                     |    0.55|    0.56|1.02x|  1.8133|    2.85 GB|27.68|0.8300|-    |-     |11,212,131|
|hat_m  4x bf16                     |    0.54|    0.58|1.06x|  1.8367|   10.30 GB|27.97|0.8368|-    |-     |20,772,507|
|swinir_l  4x bf16                  |    0.46|    0.39|0.84x|  2.1642|    3.56 GB|-    |-     |-    |-     |28,013,059|
|swin2sr_l  4x bf16                 |    0.41|    0.40|1.00x|  2.4684|    3.37 GB|-    |-     |-    |-     |28,785,859|
|atd  4x bf16                       |    0.35|    0.37|1.06x|  2.8741|    6.42 GB|28.22|0.8414|-    |-     |20,260,929|
|hat_l  4x bf16                     |    0.29|    0.31|1.07x|  3.4823|   10.41 GB|28.60|0.8498|-    |-     |40,846,575|
|srformer  4x bf16                  |    0.29|    0.28|0.99x|  3.4944|    3.65 GB|27.68|0.8311|-    |-     |10,543,503|
|dat  4x bf16                       |    0.28|    0.28|0.98x|  3.5212|    3.94 GB|27.87|0.8343|-    |-     |14,802,051|
|dat_2  4x bf16                     |    0.28|    0.28|1.00x|  3.5391|    3.92 GB|27.86|0.8341|-    |-     |11,212,131|
|drct  4x bf16                      |    0.18|    0.18|1.00x|  5.4825|    6.28 GB|28.06|0.8378|-    |-     |14,139,579|
|drct_l  4x bf16                    |    0.09|    0.09|1.00x| 10.9431|    6.42 GB|28.70|0.8508|-    |-     |27,580,719|
|drct_xl  4x bf16                   |    0.08|    0.08|1.00x| 12.6742|    6.46 GB|-    |-     |-    |-     |32,061,099|
|rgt  4x bf16                       |-       |-       |-       |-       |-       |-|-|-||-         |
|rgt_s  4x bf16                     |-       |-       |-       |-       |-       |-|-|-||-         |
|ipt  4x fp16                       |-       |-       |-       |-       |-       |-|-|-||-         |
|tscunet  4x fp16                   |-       |-       |-       |-       |-       |-|-|-||-         |

### 3x scale
(1, 3, 480, 640) input, 1 warmup + 5 runs averaged
|Name|FPS|FPS ({abbr}`CL (channels_last)`)|{abbr}`CL (channels_last)` vs base|sec/img|VRAM ({abbr}`CL (channels_last)`)|PSNR (DF2K)|SSIM (DF2K)|PSNR (DIV2K)|SSIM (DIV2K)|Params|
|:-|-:|-:|-:|-:|-:|-:|-:|-:|-:|-:|
|ultracompact  3x fp16              |  241.36|   35.52|0.15x|  0.0041|    0.11 GB|-    |-     |-    |-     |   313,371|
|compact  3x fp16                   |  142.98|  206.06|1.44x|  0.0070|    0.11 GB|-    |-     |-    |-     |   609,307|
|superultracompact  3x fp16         |  129.11| 1070.15|8.29x|  0.0077|    0.11 GB|-    |-     |-    |-     |    48,411|
|spanplus_sts  3x fp16              |   98.29|  163.98|1.67x|  0.0102|    0.21 GB|-    |-     |-    |-     |   687,091|
|spanplus_s  3x fp16                |   96.68|   36.02|0.37x|  0.0103|    0.89 GB|-    |-     |-    |-     |   684,067|
|spanplus_st  3x fp16               |   94.71|  109.03|1.15x|  0.0106|    0.31 GB|-    |-     |-    |-     | 2,227,635|
|artcnn_r8f64  3x fp16              |   91.96|  143.36|1.56x|  0.0109|    0.22 GB|-    |-     |-    |-     |   940,571|
|span  3x fp16                      |   75.39|  104.90|1.39x|  0.0133|    0.59 GB|-    |-     |-    |-     | 2,227,635|
|spanplus  3x fp16                  |   56.88|   32.95|0.58x|  0.0176|    1.23 GB|-    |-     |-    |-     | 2,223,075|
|mosr_t  3x fp16                    |   54.82|   65.20|1.19x|  0.0182|    0.35 GB|-    |-     |-    |-     |   600,795|
|plksr_tiny  3x fp16                |   37.73|   40.54|1.07x|  0.0265|    0.24 GB|28.51|0.8599|28.35|0.8571| 2,358,427|
|man_tiny  3x fp16                  |   35.52|   27.12|0.76x|  0.0282|    0.41 GB|-    |-     |-    |-     |   140,931|
|realcugan  3x fp16                 |   22.19|   34.94|1.57x|  0.0451|    1.71 GB|-    |-     |-    |-     | 1,286,326|
|lmlt_tiny  3x fp16                 |   22.05|   26.12|1.18x|  0.0454|    0.36 GB|28.10|0.8503|-    |-     |   244,215|
|artcnn_r16f96  3x fp16             |   18.55|   29.34|1.58x|  0.0539|    0.32 GB|-    |-     |-    |-     | 4,095,003|
|lmlt_base  3x fp16                 |   15.33|   14.81|0.97x|  0.0652|    0.62 GB|28.48|0.8581|-    |-     |   660,447|
|cfsr  3x bf16                      |   12.87|    9.03|0.70x|  0.0777|    0.46 GB|28.29|0.8553|-    |-     |   297,819|
|plksr  3x fp16                     |   12.18|   15.89|1.30x|  0.0821|    0.27 GB|29.10|0.8713|28.86|0.8666| 7,373,979|
|scunet_aaf6aa  3x fp16             |   10.63|    5.60|0.53x|  0.0940|    1.02 GB|-    |-     |-    |-     |15,170,540|
|lmlt_large  3x fp16                |   10.25|   10.17|0.99x|  0.0976|    0.86 GB|28.72|0.8628|-    |-     | 1,279,431|
|esrgan_lite  3x fp16               |    9.50|   13.17|1.39x|  0.1053|    0.71 GB|-    |-     |-    |-     | 5,011,907|
|metaflexnet  3x fp16               |    9.37|    9.83|1.05x|  0.1067|    1.27 GB|-    |-     |-    |-     |38,035,403|
|realplksr pixelshuffle layer_norm=True 3x fp16|    8.05|    2.39|0.30x|  0.1242|    0.35 GB|-    |-     |-    |-     | 7,377,563|
|eimn_a  3x fp16                    |    8.04|    5.81|0.72x|  0.1243|    0.87 GB|28.87|0.8660|-    |-     |   868,753|
|realplksr dysample layer_norm=True 3x fp16|    7.90|    8.60|1.09x|  0.1265|    0.79 GB|-    |-     |-    |-     | 7,380,617|
|eimn_l  3x fp16                    |    7.48|    5.45|0.73x|  0.1337|    0.87 GB|29.05|0.8698|-    |-     |   990,379|
|omnisr  3x fp16                    |    7.29|    7.82|1.07x|  0.1371|    1.17 GB|29.12|0.8712|28.84|0.8656|   793,259|
|mosr  3x fp16                      |    7.13|    1.99|0.28x|  0.1403|    0.47 GB|-    |-     |-    |-     | 4,275,483|
|man_light  3x fp16                 |    6.13|    4.45|0.73x|  0.1631|    0.52 GB|-    |-     |-    |-     |   831,531|
|realplksr pixelshuffle layer_norm=False 3x fp16|    5.84|    6.18|1.06x|  0.1714|    0.42 GB|-    |-     |-    |-     | 7,377,563|
|rcan  3x fp16                      |    5.58|    7.58|1.36x|  0.1792|    0.86 GB|-    |-     |29.09|0.8702|15,629,307|
|rcanspatialselayer  3x fp16        |    5.57|    7.88|1.41x|  0.1794|    0.86 GB|-    |-     |-    |-     |15,526,283|
|hit_sir  3x bf16                   |    4.99|    4.81|0.96x|  0.2002|    1.32 GB|-    |-     |28.93|0.8673|   780,179|
|realplksr dysample layer_norm=False 3x fp16|    4.99|    3.67|0.73x|  0.2004|    0.79 GB|-    |-     |-    |-     | 7,380,617|
|camixersr  3x bf16                 |    4.93|    3.91|0.79x|  0.2030|    1.13 GB|-    |-     |-    |-     |   753,961|
|hit_srf  3x bf16                   |    4.26|    4.28|1.00x|  0.2346|    1.32 GB|-    |-     |28.99|0.8687|   855,059|
|moesr2  3x fp16                    |    3.81|    1.70|0.45x|  0.2625|    0.90 GB|-    |-     |-    |-     |16,534,891|
|esrgan use_pixel_unshuffle=True 3x fp16|    3.51|    4.91|1.40x|  0.2852|    1.44 GB|-    |-     |-    |-     |16,661,059|
|esrgan use_pixel_unshuffle=False 3x fp16|    3.45|    4.93|1.43x|  0.2901|    1.44 GB|-    |-     |-    |-     |16,661,059|
|atd_light  3x bf16                 |    1.13|    1.15|1.02x|  0.8870|    3.28 GB|29.17|0.8709|-    |-     |   759,699|
|hit_sng  3x bf16                   |    1.04|    2.43|2.34x|  0.9641|    1.32 GB|-    |-     |28.91|0.8671| 1,020,699|
|flexnet  3x fp16                   |    1.00|    2.65|2.65x|  0.9986|    1.29 GB|-    |-     |-    |-     | 3,020,923|
|srformer_light  3x bf16            |    0.88|    0.99|1.12x|  1.1332|    1.45 GB|28.81|0.8655|-    |-     |   861,387|
|swinir_s  3x bf16                  |    0.84|    1.00|1.19x|  1.1910|    1.53 GB|28.66|0.8624|-    |-     |   918,267|
|man  3x fp16                       |    0.63|    0.52|0.83x|  1.5960|    1.52 GB|29.52|0.8782|-    |-     | 8,678,571|
|dat_light  3x bf16                 |    0.58|    0.57|0.98x|  1.7114|    2.69 GB|28.89|0.8666|-    |-     |   561,405|
|dat_s  3x bf16                     |    0.56|    0.55|1.00x|  1.8015|    2.82 GB|29.98|0.8846|-    |-     |11,249,059|
|swin2sr_s  3x bf16                 |    0.53|    0.85|1.60x|  1.8918|    1.43 GB|-    |-     |-    |-     | 1,013,463|
|swinir_m  3x bf16                  |    0.43|    0.63|1.44x|  2.2997|    2.62 GB|29.75|0.8826|-    |-     |11,937,127|
|hat_s  3x bf16                     |    0.42|    0.43|1.04x|  2.4094|    9.83 GB|30.15|0.8879|-    |-     | 9,658,111|
|swin2sr_m  3x bf16                 |    0.39|    0.60|1.54x|  2.5719|    2.49 GB|-    |-     |-    |-     |12,276,211|
|atd  3x bf16                       |    0.34|    0.36|1.06x|  2.9453|    6.40 GB|30.52|0.8924|-    |-     |20,297,857|
|swinir_l  3x bf16                  |    0.32|    0.42|1.32x|  3.1135|    3.52 GB|-    |-     |-    |-     |27,976,131|
|hat_m  3x bf16                     |    0.32|    0.40|1.27x|  3.1324|   10.27 GB|30.23|0.8896|-    |-     |20,809,435|
|dat_2  3x bf16                     |    0.27|    0.27|1.02x|  3.7692|    3.90 GB|30.13|0.8878|-    |-     |11,249,059|
|dat  3x bf16                       |    0.26|    0.27|1.03x|  3.8234|    3.92 GB|30.18|0.8886|-    |-     |14,838,979|
|hat_l  3x bf16                     |    0.25|    0.24|0.98x|  4.0592|   10.39 GB|30.92|0.8981|-    |-     |40,883,503|
|srformer  3x bf16                  |    0.21|    0.22|1.03x|  4.7708|    3.62 GB|30.04|0.8865|-    |-     |10,580,431|
|drct  3x bf16                      |    0.19|    0.18|1.00x|  5.3984|    6.25 GB|30.34|0.8910|-    |-     |14,176,507|
|drct_l  3x bf16                    |    0.07|    0.07|1.05x| 14.1257|    6.39 GB|31.14|0.9004|-    |-     |27,617,647|
|drct_xl  3x bf16                   |    0.06|    0.06|1.00x| 17.1119|    6.44 GB|-    |-     |-    |-     |32,098,027|
|rgt  3x bf16                       |-       |-       |-       |-       |-       |-|-|-||-         |
|rgt_s  3x bf16                     |-       |-       |-       |-       |-       |-|-|-||-         |
|swin2sr_l  3x bf16                 |-       |-       |-       |-       |-       |-|-|-||-         |
|ipt  3x fp16                       |-       |-       |-       |-       |-       |-|-|-||-         |
|tscunet  3x fp16                   |-       |-       |-       |-       |-       |-|-|-||-         |

### 2x scale
(1, 3, 480, 640) input, 1 warmup + 5 runs averaged
|Name|FPS|FPS ({abbr}`CL (channels_last)`)|{abbr}`CL (channels_last)` vs base|sec/img|VRAM ({abbr}`CL (channels_last)`)|PSNR (DF2K)|SSIM (DF2K)|PSNR (DIV2K)|SSIM (DIV2K)|Params|
|:-|-:|-:|-:|-:|-:|-:|-:|-:|-:|-:|
|superultracompact  2x fp16         |  913.72| 1277.35|1.40x|  0.0011|    0.06 GB|-    |-     |-    |-     |    45,156|
|ultracompact  2x fp16              |  265.34|  427.63|1.61x|  0.0038|    0.09 GB|-    |-     |-    |-     |   304,716|
|spanplus_sts  2x fp16              |  161.06|  167.80|1.04x|  0.0062|    0.21 GB|-    |-     |-    |-     |   682,756|
|compact  2x fp16                   |  142.58|  224.47|1.57x|  0.0070|    0.10 GB|-    |-     |-    |-     |   600,652|
|spanplus_s  2x fp16                |  124.09|  122.62|0.99x|  0.0081|    0.42 GB|-    |-     |-    |-     |   681,467|
|spanplus_st  2x fp16               |  103.82|  113.71|1.10x|  0.0096|    0.30 GB|-    |-     |-    |-     | 2,221,140|
|span  2x fp16                      |   95.21|  102.49|1.08x|  0.0105|    0.58 GB|32.24|0.9294|-    |-     | 2,221,140|
|artcnn_r8f64  2x fp16              |   90.73|  144.84|1.60x|  0.0110|    0.21 GB|-    |-     |-    |-     |   931,916|
|spanplus  2x fp16                  |   79.79|   82.80|1.04x|  0.0125|    0.58 GB|-    |-     |-    |-     | 2,219,195|
|realcugan  2x fp16                 |   59.75|   81.17|1.36x|  0.0167|    0.78 GB|-    |-     |-    |-     | 1,284,598|
|mosr_t  2x fp16                    |   53.57|   65.56|1.22x|  0.0187|    0.34 GB|-    |-     |-    |-     |   594,300|
|esrgan_lite  2x fp16               |   42.00|   56.61|1.35x|  0.0238|    0.34 GB|-    |-     |-    |-     | 5,023,747|
|plksr_tiny  2x fp16                |   37.51|   50.63|1.35x|  0.0267|    0.22 GB|32.58|0.9328|32.43|0.9314| 2,349,772|
|man_tiny  2x fp16                  |   35.34|   27.75|0.79x|  0.0283|    0.40 GB|-    |-     |-    |-     |   134,436|
|lmlt_tiny  2x fp16                 |   28.71|   28.79|1.00x|  0.0348|    0.35 GB|32.04|0.9273|-    |-     |   239,340|
|artcnn_r16f96  2x fp16             |   18.37|   30.62|1.67x|  0.0545|    0.32 GB|-    |-     |-    |-     | 4,082,028|
|esrgan use_pixel_unshuffle=True 2x fp16|   17.98|   22.91|1.27x|  0.0556|    0.70 GB|-    |-     |-    |-     |16,703,171|
|cfsr  2x bf16                      |   16.89|   14.29|0.85x|  0.0592|    0.44 GB|32.28|0.9300|-    |-     |   291,324|
|lmlt_base  2x fp16                 |   15.20|   15.10|0.99x|  0.0658|    0.61 GB|32.52|0.9316|-    |-     |   652,332|
|plksr  2x fp16                     |   12.04|   15.87|1.32x|  0.0830|    0.25 GB|33.36|0.9395|32.99|0.9365| 7,365,324|
|scunet_aaf6aa  2x fp16             |   11.49|   14.17|1.23x|  0.0870|    1.02 GB|-    |-     |-    |-     |15,170,540|
|lmlt_large  2x fp16                |   10.38|   10.33|1.00x|  0.0964|    0.85 GB|32.75|0.9336|-    |-     | 1,268,076|
|metaflexnet  2x fp16               |    9.31|    9.64|1.03x|  0.1074|    1.27 GB|-    |-     |-    |-     |38,022,428|
|mosr  2x fp16                      |    9.29|   11.45|1.23x|  0.1077|    0.46 GB|-    |-     |-    |-     | 4,266,828|
|realplksr pixelshuffle layer_norm=True 2x fp16|    8.48|   10.15|1.20x|  0.1179|    0.33 GB|-    |-     |-    |-     | 7,368,908|
|eimn_a  2x fp16                    |    8.38|    6.12|0.73x|  0.1193|    0.86 GB|33.15|0.9373|-    |-     |   860,098|
|realplksr dysample layer_norm=True 2x fp16|    8.37|    9.96|1.19x|  0.1194|    0.32 GB|-    |-     |-    |-     | 7,369,747|
|omnisr  2x fp16                    |    7.80|    7.66|0.98x|  0.1282|    1.16 GB|33.30|0.9386|33.05|0.9363|   784,604|
|eimn_l  2x fp16                    |    7.33|    5.34|0.73x|  0.1364|    0.87 GB|33.23|0.9381|-    |-     |   981,724|
|man_light  2x fp16                 |    6.04|    4.42|0.73x|  0.1656|    0.51 GB|-    |-     |-    |-     |   823,416|
|realplksr pixelshuffle layer_norm=False 2x fp16|    5.84|    6.15|1.05x|  0.1713|    0.40 GB|-    |-     |-    |-     | 7,368,908|
|realplksr dysample layer_norm=False 2x fp16|    5.79|    6.08|1.05x|  0.1727|    0.39 GB|-    |-     |-    |-     | 7,369,747|
|rcanspatialselayer  2x fp16        |    5.68|    8.01|1.41x|  0.1761|    0.48 GB|-    |-     |-    |-     |15,341,643|
|rcan  2x fp16                      |    5.63|    7.77|1.38x|  0.1775|    0.48 GB|-    |-     |33.34|0.9384|15,444,667|
|hit_sir  2x bf16                   |    5.29|    5.13|0.97x|  0.1890|    1.30 GB|-    |-     |33.02|0.9365|   772,064|
|camixersr  2x bf16                 |    4.91|    3.84|0.78x|  0.2035|    1.13 GB|-    |-     |-    |-     |   745,846|
|hit_srf  2x bf16                   |    4.25|    4.20|0.99x|  0.2352|    1.30 GB|-    |-     |33.13|0.9372|   846,944|
|hit_sng  2x bf16                   |    4.11|    3.93|0.96x|  0.2432|    1.30 GB|-    |-     |33.01|0.9360| 1,012,584|
|moesr2  2x fp16                    |    3.93|    3.67|0.93x|  0.2544|    0.89 GB|-    |-     |-    |-     |16,526,236|
|esrgan use_pixel_unshuffle=False 2x fp16|    3.56|    4.99|1.40x|  0.2811|    0.70 GB|-    |-     |-    |-     |16,661,059|
|flexnet  2x fp16                   |    3.45|    3.37|0.98x|  0.2901|    1.28 GB|-    |-     |-    |-     | 3,003,628|
|man  2x fp16                       |    1.26|    0.86|0.69x|  0.7928|    1.51 GB|33.73|0.9422|-    |-     | 8,654,256|
|swinir_s  2x bf16                  |    1.06|    1.07|1.02x|  0.9470|    1.51 GB|32.76|0.9340|-    |-     |   910,152|
|srformer_light  2x bf16            |    1.06|    1.05|1.00x|  0.9475|    1.43 GB|32.91|0.9353|-    |-     |   853,272|
|swin2sr_s  2x bf16                 |    1.02|    1.02|1.00x|  0.9791|    1.41 GB|32.85|0.9349|-    |-     | 1,005,348|
|swinir_m  2x bf16                  |    0.70|    0.70|1.00x|  1.4290|    2.60 GB|33.81|0.9427|-    |-     |11,752,487|
|hat_s  2x bf16                     |    0.63|    0.67|1.06x|  1.5900|    9.81 GB|34.31|0.9459|-    |-     | 9,473,471|
|swin2sr_m  2x bf16                 |    0.63|    0.62|1.00x|  1.5970|    2.47 GB|33.89|0.9431|-    |-     |12,091,571|
|atd_light  2x bf16                 |    0.61|    1.26|2.08x|  1.6424|    3.26 GB|33.27|0.9375|-    |-     |   753,204|
|dat_light  2x bf16                 |    0.60|    0.60|1.00x|  1.6565|    2.67 GB|32.89|0.9346|-    |-     |   553,290|
|dat_s  2x bf16                     |    0.57|    0.57|0.99x|  1.7395|    2.80 GB|34.12|0.9444|-    |-     |11,064,419|
|hat_m  2x bf16                     |    0.55|    0.58|1.06x|  1.8203|   10.26 GB|34.45|0.9466|-    |-     |20,624,795|
|swinir_l  2x bf16                  |    0.47|    0.47|1.01x|  2.1330|    3.52 GB|-    |-     |-    |-     |27,976,131|
|hat_l  2x bf16                     |    0.29|    0.31|1.07x|  3.4783|   10.37 GB|35.09|0.9513|-    |-     |40,698,863|
|srformer  2x bf16                  |    0.29|    0.29|1.00x|  3.4968|    3.61 GB|34.09|0.9449|-    |-     |10,395,791|
|dat_2  2x bf16                     |    0.27|    0.27|1.01x|  3.7285|    3.88 GB|34.31|0.9457|-    |-     |11,064,419|
|atd  2x bf16                       |    0.25|    0.26|1.04x|  4.0063|    6.38 GB|34.73|0.9476|-    |-     |20,113,217|
|dat  2x bf16                       |    0.22|    0.24|1.10x|  4.6453|    3.90 GB|34.37|0.9458|-    |-     |14,654,339|
|drct  2x bf16                      |    0.18|    0.19|1.01x|  5.4143|    6.24 GB|34.54|0.9474|-    |-     |13,991,867|
|drct_l  2x bf16                    |    0.09|    0.09|0.98x| 10.6995|    6.37 GB|35.17|0.9516|-    |-     |27,433,007|
|drct_xl  2x bf16                   |    0.08|    0.08|1.00x| 12.5408|    6.42 GB|-    |-     |-    |-     |31,913,387|
|rgt  2x bf16                       |-       |-       |-       |-       |-       |-|-|-||-         |
|rgt_s  2x bf16                     |-       |-       |-       |-       |-       |-|-|-||-         |
|swin2sr_l  2x bf16                 |-       |-       |-       |-       |-       |-|-|-||-         |
|ipt  2x fp16                       |-       |-       |-       |-       |-       |-|-|-||-         |
|tscunet  2x fp16                   |-       |-       |-       |-       |-       |-|-|-||-         |

### 1x scale
(1, 3, 480, 640) input, 1 warmup + 5 runs averaged
|Name|FPS|FPS ({abbr}`CL (channels_last)`)|{abbr}`CL (channels_last)` vs base|sec/img|VRAM ({abbr}`CL (channels_last)`)|PSNR (DF2K)|SSIM (DF2K)|PSNR (DIV2K)|SSIM (DIV2K)|Params|
|:-|-:|-:|-:|-:|-:|-:|-:|-:|-:|-:|
|superultracompact  1x fp16         |  974.30| 1339.36|1.37x|  0.0010|    0.04 GB|-    |-     |-    |-     |    43,203|
|ultracompact  1x fp16              |  270.26|  435.08|1.61x|  0.0037|    0.09 GB|-    |-     |-    |-     |   299,523|
|spanplus_sts  1x fp16              |  158.99|  163.55|1.03x|  0.0063|    0.20 GB|-    |-     |-    |-     |   680,155|
|compact  1x fp16                   |  140.95|  230.37|1.63x|  0.0071|    0.09 GB|-    |-     |-    |-     |   595,459|
|spanplus_s  1x fp16                |  132.89|  136.83|1.03x|  0.0075|    0.20 GB|-    |-     |-    |-     |   679,907|
|spanplus_st  1x fp16               |  100.57|  110.92|1.10x|  0.0099|    0.30 GB|-    |-     |-    |-     | 2,217,243|
|artcnn_r8f64  1x fp16              |   92.09|  141.55|1.54x|  0.0109|    0.20 GB|-    |-     |-    |-     |   926,723|
|spanplus  1x fp16                  |   91.82|   97.73|1.06x|  0.0109|    0.30 GB|-    |-     |-    |-     | 2,216,867|
|span  1x fp16                      |   90.01|  102.47|1.14x|  0.0111|    0.58 GB|-    |-     |-    |-     | 2,217,243|
|esrgan_lite  1x fp16               |   74.28|  109.74|1.48x|  0.0135|    0.12 GB|-    |-     |-    |-     | 5,034,115|
|mosr_t  1x fp16                    |   54.78|   66.46|1.21x|  0.0183|    0.33 GB|-    |-     |-    |-     |   590,403|
|esrgan use_pixel_unshuffle=True 1x fp16|   37.78|   54.15|1.43x|  0.0265|    0.26 GB|-    |-     |-    |-     |16,723,907|
|plksr_tiny  1x fp16                |   37.49|   50.55|1.35x|  0.0267|    0.21 GB|-    |-     |-    |-     | 2,344,579|
|man_tiny  1x fp16                  |   35.45|   27.53|0.78x|  0.0282|    0.40 GB|-    |-     |-    |-     |   130,539|
|lmlt_tiny  1x fp16                 |   28.42|   28.49|1.00x|  0.0352|    0.35 GB|-    |-     |-    |-     |   236,415|
|scunet_aaf6aa  1x fp16             |   27.29|   27.73|1.02x|  0.0366|    0.80 GB|-    |-     |-    |-     | 9,699,756|
|artcnn_r16f96  1x fp16             |   18.55|   30.84|1.66x|  0.0539|    0.31 GB|-    |-     |-    |-     | 4,074,243|
|cfsr  1x bf16                      |   16.92|   14.30|0.84x|  0.0591|    0.43 GB|-    |-     |-    |-     |   287,427|
|lmlt_base  1x fp16                 |   15.12|   15.15|1.00x|  0.0662|    0.61 GB|-    |-     |-    |-     |   647,463|
|plksr  1x fp16                     |   12.04|   15.87|1.32x|  0.0831|    0.24 GB|-    |-     |-    |-     | 7,360,131|
|lmlt_large  1x fp16                |   10.38|   10.34|1.00x|  0.0963|    0.85 GB|-    |-     |-    |-     | 1,261,263|
|metaflexnet  1x fp16               |    9.31|    9.62|1.03x|  0.1075|    1.26 GB|-    |-     |-    |-     |38,014,643|
|mosr  1x fp16                      |    9.29|   11.48|1.24x|  0.1076|    0.46 GB|-    |-     |-    |-     | 4,261,635|
|realplksr pixelshuffle layer_norm=True 1x fp16|    8.46|   10.13|1.20x|  0.1182|    0.32 GB|-    |-     |-    |-     | 7,363,715|
|realplksr dysample layer_norm=True 1x fp16|    8.41|   10.05|1.19x|  0.1188|    0.32 GB|-    |-     |-    |-     | 7,363,757|
|eimn_a  1x fp16                    |    8.38|    6.12|0.73x|  0.1193|    0.86 GB|-    |-     |-    |-     |   854,905|
|omnisr  1x fp16                    |    7.78|    7.65|0.98x|  0.1285|    1.16 GB|-    |-     |-    |-     |   779,411|
|eimn_l  1x fp16                    |    7.35|    5.35|0.73x|  0.1360|    0.86 GB|-    |-     |-    |-     |   976,531|
|man_light  1x fp16                 |    6.03|    4.42|0.73x|  0.1658|    0.50 GB|-    |-     |-    |-     |   818,547|
|realplksr pixelshuffle layer_norm=False 1x fp16|    5.84|    6.15|1.05x|  0.1713|    0.39 GB|-    |-     |-    |-     | 7,363,715|
|realplksr dysample layer_norm=False 1x fp16|    5.83|    6.14|1.05x|  0.1717|    0.39 GB|-    |-     |-    |-     | 7,363,757|
|rcanspatialselayer  1x fp16        |    5.77|    8.19|1.42x|  0.1732|    0.29 GB|-    |-     |-    |-     |15,193,931|
|rcan  1x fp16                      |    5.72|    7.94|1.39x|  0.1749|    0.28 GB|-    |-     |-    |-     |15,296,955|
|hit_sir  1x bf16                   |    5.29|    5.12|0.97x|  0.1892|    1.29 GB|-    |-     |-    |-     |   767,195|
|camixersr  1x bf16                 |    4.96|    3.89|0.78x|  0.2016|    1.11 GB|-    |-     |-    |-     |   740,977|
|hit_srf  1x bf16                   |    4.24|    4.20|0.99x|  0.2357|    1.29 GB|-    |-     |-    |-     |   842,075|
|hit_sng  1x bf16                   |    4.11|    3.93|0.96x|  0.2433|    1.29 GB|-    |-     |-    |-     | 1,007,715|
|moesr2  1x fp16                    |    3.94|    3.68|0.93x|  0.2537|    0.89 GB|-    |-     |-    |-     |16,521,043|
|esrgan use_pixel_unshuffle=False 1x fp16|    3.56|    4.99|1.40x|  0.2810|    0.44 GB|-    |-     |-    |-     |16,624,131|
|flexnet  1x fp16                   |    3.44|    3.38|0.98x|  0.2904|    1.28 GB|-    |-     |-    |-     | 2,993,251|
|man  1x fp16                       |    1.26|    0.86|0.68x|  0.7921|    1.51 GB|-    |-     |-    |-     | 8,639,667|
|atd_light  1x bf16                 |    1.25|    1.26|1.01x|  0.7976|    3.25 GB|-    |-     |-    |-     |   749,307|
|swinir_s  1x bf16                  |    1.06|    1.07|1.01x|  0.9424|    1.50 GB|-    |-     |-    |-     |   905,283|
|srformer_light  1x bf16            |    1.03|    1.06|1.03x|  0.9756|    1.42 GB|-    |-     |-    |-     |   848,403|
|swin2sr_s  1x bf16                 |    1.02|    1.03|1.01x|  0.9796|    1.40 GB|-    |-     |-    |-     | 1,000,479|
|swinir_m  1x bf16                  |    0.70|    0.71|1.01x|  1.4287|    2.59 GB|-    |-     |-    |-     |11,604,775|
|hat_s  1x bf16                     |    0.63|    0.67|1.07x|  1.5879|    9.80 GB|-    |-     |-    |-     | 9,325,759|
|swin2sr_m  1x bf16                 |    0.63|    0.63|0.99x|  1.5907|    2.46 GB|-    |-     |-    |-     |11,943,859|
|hat_m  1x bf16                     |    0.55|    0.59|1.07x|  1.8278|   10.24 GB|-    |-     |-    |-     |20,477,083|
|swinir_l  1x bf16                  |    0.47|    0.47|1.01x|  2.1349|    3.52 GB|-    |-     |-    |-     |27,976,131|
|dat_light  1x bf16                 |    0.43|    0.59|1.39x|  2.3427|    2.66 GB|-    |-     |-    |-     |   548,421|
|dat_s  1x bf16                     |    0.36|    0.36|1.00x|  2.7647|    2.79 GB|-    |-     |-    |-     |10,916,707|
|atd  1x bf16                       |    0.35|    0.37|1.06x|  2.8328|    6.37 GB|-    |-     |-    |-     |19,965,505|
|dat  1x bf16                       |    0.28|    0.27|0.94x|  3.5294|    3.89 GB|-    |-     |-    |-     |14,506,627|
|srformer  1x bf16                  |    0.28|    0.29|1.03x|  3.5791|    3.60 GB|-    |-     |-    |-     |10,248,079|
|dat_2  1x bf16                     |    0.22|    0.27|1.24x|  4.5538|    3.87 GB|-    |-     |-    |-     |10,916,707|
|hat_l  1x bf16                     |    0.19|    0.31|1.64x|  5.3148|   10.36 GB|-    |-     |-    |-     |40,551,151|
|drct  1x bf16                      |    0.14|    0.15|1.11x|  7.3718|    6.23 GB|-    |-     |-    |-     |13,844,155|
|drct_l  1x bf16                    |    0.07|    0.07|0.99x| 13.8604|    6.36 GB|-    |-     |-    |-     |27,285,295|
|drct_xl  1x bf16                   |    0.06|    0.05|0.84x| 15.9074|    6.41 GB|-    |-     |-    |-     |31,765,675|
|realcugan  1x fp16                 |-       |-       |-       |-       |-       |-|-|-||-         |
|rgt  1x bf16                       |-       |-       |-       |-       |-       |-|-|-||-         |
|rgt_s  1x bf16                     |-       |-       |-       |-       |-       |-|-|-||-         |
|swin2sr_l  1x bf16                 |-       |-       |-       |-       |-       |-|-|-||-         |
|ipt  1x fp16                       |-       |-       |-       |-       |-       |-|-|-||-         |
|tscunet  1x fp16                   |-       |-       |-       |-       |-       |-|-|-||-         |

## By Architecture

### artcnn_r16f96 
(1, 3, 480, 640) input, 1 warmup + 5 runs averaged
|Name|FPS|FPS ({abbr}`CL (channels_last)`)|{abbr}`CL (channels_last)` vs base|sec/img|VRAM ({abbr}`CL (channels_last)`)|PSNR (DF2K)|SSIM (DF2K)|PSNR (DIV2K)|SSIM (DIV2K)|Params|
|:-|-:|-:|-:|-:|-:|-:|-:|-:|-:|-:|
|artcnn_r16f96  4x fp16             |   17.38|   30.74|1.77x|  0.0575|    0.34 GB|-    |-     |-    |-     | 4,113,168|
|artcnn_r16f96  3x fp16             |   18.55|   29.34|1.58x|  0.0539|    0.32 GB|-    |-     |-    |-     | 4,095,003|
|artcnn_r16f96  2x fp16             |   18.37|   30.62|1.67x|  0.0545|    0.32 GB|-    |-     |-    |-     | 4,082,028|
|artcnn_r16f96  1x fp16             |   18.55|   30.84|1.66x|  0.0539|    0.31 GB|-    |-     |-    |-     | 4,074,243|

### artcnn_r8f64 
(1, 3, 480, 640) input, 1 warmup + 5 runs averaged
|Name|FPS|FPS ({abbr}`CL (channels_last)`)|{abbr}`CL (channels_last)` vs base|sec/img|VRAM ({abbr}`CL (channels_last)`)|PSNR (DF2K)|SSIM (DF2K)|PSNR (DIV2K)|SSIM (DIV2K)|Params|
|:-|-:|-:|-:|-:|-:|-:|-:|-:|-:|-:|
|artcnn_r8f64  4x fp16              |   91.62|  138.61|1.51x|  0.0109|    0.23 GB|-    |-     |-    |-     |   952,688|
|artcnn_r8f64  3x fp16              |   91.96|  143.36|1.56x|  0.0109|    0.22 GB|-    |-     |-    |-     |   940,571|
|artcnn_r8f64  2x fp16              |   90.73|  144.84|1.60x|  0.0110|    0.21 GB|-    |-     |-    |-     |   931,916|
|artcnn_r8f64  1x fp16              |   92.09|  141.55|1.54x|  0.0109|    0.20 GB|-    |-     |-    |-     |   926,723|

### atd 
(1, 3, 480, 640) input, 1 warmup + 5 runs averaged
|Name|FPS|FPS ({abbr}`CL (channels_last)`)|{abbr}`CL (channels_last)` vs base|sec/img|VRAM ({abbr}`CL (channels_last)`)|PSNR (DF2K)|SSIM (DF2K)|PSNR (DIV2K)|SSIM (DIV2K)|Params|
|:-|-:|-:|-:|-:|-:|-:|-:|-:|-:|-:|
|atd  4x bf16                       |    0.35|    0.37|1.06x|  2.8741|    6.42 GB|28.22|0.8414|-    |-     |20,260,929|
|atd  3x bf16                       |    0.34|    0.36|1.06x|  2.9453|    6.40 GB|30.52|0.8924|-    |-     |20,297,857|
|atd  2x bf16                       |    0.25|    0.26|1.04x|  4.0063|    6.38 GB|34.73|0.9476|-    |-     |20,113,217|
|atd  1x bf16                       |    0.35|    0.37|1.06x|  2.8328|    6.37 GB|-    |-     |-    |-     |19,965,505|

### atd_light 
(1, 3, 480, 640) input, 1 warmup + 5 runs averaged
|Name|FPS|FPS ({abbr}`CL (channels_last)`)|{abbr}`CL (channels_last)` vs base|sec/img|VRAM ({abbr}`CL (channels_last)`)|PSNR (DF2K)|SSIM (DF2K)|PSNR (DIV2K)|SSIM (DIV2K)|Params|
|:-|-:|-:|-:|-:|-:|-:|-:|-:|-:|-:|
|atd_light  4x bf16                 |    1.22|    1.24|1.02x|  0.8189|    3.30 GB|26.97|0.8107|-    |-     |   768,792|
|atd_light  3x bf16                 |    1.13|    1.15|1.02x|  0.8870|    3.28 GB|29.17|0.8709|-    |-     |   759,699|
|atd_light  2x bf16                 |    0.61|    1.26|2.08x|  1.6424|    3.26 GB|33.27|0.9375|-    |-     |   753,204|
|atd_light  1x bf16                 |    1.25|    1.26|1.01x|  0.7976|    3.25 GB|-    |-     |-    |-     |   749,307|

### camixersr 
(1, 3, 480, 640) input, 1 warmup + 5 runs averaged
|Name|FPS|FPS ({abbr}`CL (channels_last)`)|{abbr}`CL (channels_last)` vs base|sec/img|VRAM ({abbr}`CL (channels_last)`)|PSNR (DF2K)|SSIM (DF2K)|PSNR (DIV2K)|SSIM (DIV2K)|Params|
|:-|-:|-:|-:|-:|-:|-:|-:|-:|-:|-:|
|camixersr  4x bf16                 |    4.78|    3.61|0.75x|  0.2091|    1.13 GB|-    |-     |26.63|0.8012|   765,322|
|camixersr  3x bf16                 |    4.93|    3.91|0.79x|  0.2030|    1.13 GB|-    |-     |-    |-     |   753,961|
|camixersr  2x bf16                 |    4.91|    3.84|0.78x|  0.2035|    1.13 GB|-    |-     |-    |-     |   745,846|
|camixersr  1x bf16                 |    4.96|    3.89|0.78x|  0.2016|    1.11 GB|-    |-     |-    |-     |   740,977|

### cfsr 
(1, 3, 480, 640) input, 1 warmup + 5 runs averaged
|Name|FPS|FPS ({abbr}`CL (channels_last)`)|{abbr}`CL (channels_last)` vs base|sec/img|VRAM ({abbr}`CL (channels_last)`)|PSNR (DF2K)|SSIM (DF2K)|PSNR (DIV2K)|SSIM (DIV2K)|Params|
|:-|-:|-:|-:|-:|-:|-:|-:|-:|-:|-:|
|cfsr  4x bf16                      |   16.76|   14.21|0.85x|  0.0597|    0.48 GB|26.21|0.7897|-    |-     |   306,912|
|cfsr  3x bf16                      |   12.87|    9.03|0.70x|  0.0777|    0.46 GB|28.29|0.8553|-    |-     |   297,819|
|cfsr  2x bf16                      |   16.89|   14.29|0.85x|  0.0592|    0.44 GB|32.28|0.9300|-    |-     |   291,324|
|cfsr  1x bf16                      |   16.92|   14.30|0.84x|  0.0591|    0.43 GB|-    |-     |-    |-     |   287,427|

### compact 
(1, 3, 480, 640) input, 1 warmup + 5 runs averaged
|Name|FPS|FPS ({abbr}`CL (channels_last)`)|{abbr}`CL (channels_last)` vs base|sec/img|VRAM ({abbr}`CL (channels_last)`)|PSNR (DF2K)|SSIM (DF2K)|PSNR (DIV2K)|SSIM (DIV2K)|Params|
|:-|-:|-:|-:|-:|-:|-:|-:|-:|-:|-:|
|compact  4x fp16                   |  136.54|  210.66|1.54x|  0.0073|    0.18 GB|-    |-     |-    |-     |   621,424|
|compact  3x fp16                   |  142.98|  206.06|1.44x|  0.0070|    0.11 GB|-    |-     |-    |-     |   609,307|
|compact  2x fp16                   |  142.58|  224.47|1.57x|  0.0070|    0.10 GB|-    |-     |-    |-     |   600,652|
|compact  1x fp16                   |  140.95|  230.37|1.63x|  0.0071|    0.09 GB|-    |-     |-    |-     |   595,459|

### dat 
(1, 3, 480, 640) input, 1 warmup + 5 runs averaged
|Name|FPS|FPS ({abbr}`CL (channels_last)`)|{abbr}`CL (channels_last)` vs base|sec/img|VRAM ({abbr}`CL (channels_last)`)|PSNR (DF2K)|SSIM (DF2K)|PSNR (DIV2K)|SSIM (DIV2K)|Params|
|:-|-:|-:|-:|-:|-:|-:|-:|-:|-:|-:|
|dat  4x bf16                       |    0.28|    0.28|0.98x|  3.5212|    3.94 GB|27.87|0.8343|-    |-     |14,802,051|
|dat  3x bf16                       |    0.26|    0.27|1.03x|  3.8234|    3.92 GB|30.18|0.8886|-    |-     |14,838,979|
|dat  2x bf16                       |    0.22|    0.24|1.10x|  4.6453|    3.90 GB|34.37|0.9458|-    |-     |14,654,339|
|dat  1x bf16                       |    0.28|    0.27|0.94x|  3.5294|    3.89 GB|-    |-     |-    |-     |14,506,627|

### dat_2 
(1, 3, 480, 640) input, 1 warmup + 5 runs averaged
|Name|FPS|FPS ({abbr}`CL (channels_last)`)|{abbr}`CL (channels_last)` vs base|sec/img|VRAM ({abbr}`CL (channels_last)`)|PSNR (DF2K)|SSIM (DF2K)|PSNR (DIV2K)|SSIM (DIV2K)|Params|
|:-|-:|-:|-:|-:|-:|-:|-:|-:|-:|-:|
|dat_2  4x bf16                     |    0.28|    0.28|1.00x|  3.5391|    3.92 GB|27.86|0.8341|-    |-     |11,212,131|
|dat_2  3x bf16                     |    0.27|    0.27|1.02x|  3.7692|    3.90 GB|30.13|0.8878|-    |-     |11,249,059|
|dat_2  2x bf16                     |    0.27|    0.27|1.01x|  3.7285|    3.88 GB|34.31|0.9457|-    |-     |11,064,419|
|dat_2  1x bf16                     |    0.22|    0.27|1.24x|  4.5538|    3.87 GB|-    |-     |-    |-     |10,916,707|

### dat_light 
(1, 3, 480, 640) input, 1 warmup + 5 runs averaged
|Name|FPS|FPS ({abbr}`CL (channels_last)`)|{abbr}`CL (channels_last)` vs base|sec/img|VRAM ({abbr}`CL (channels_last)`)|PSNR (DF2K)|SSIM (DF2K)|PSNR (DIV2K)|SSIM (DIV2K)|Params|
|:-|-:|-:|-:|-:|-:|-:|-:|-:|-:|-:|
|dat_light  4x bf16                 |    0.62|    0.62|0.99x|  1.6029|    2.71 GB|26.64|0.8033|-    |-     |   572,766|
|dat_light  3x bf16                 |    0.58|    0.57|0.98x|  1.7114|    2.69 GB|28.89|0.8666|-    |-     |   561,405|
|dat_light  2x bf16                 |    0.60|    0.60|1.00x|  1.6565|    2.67 GB|32.89|0.9346|-    |-     |   553,290|
|dat_light  1x bf16                 |    0.43|    0.59|1.39x|  2.3427|    2.66 GB|-    |-     |-    |-     |   548,421|

### dat_s 
(1, 3, 480, 640) input, 1 warmup + 5 runs averaged
|Name|FPS|FPS ({abbr}`CL (channels_last)`)|{abbr}`CL (channels_last)` vs base|sec/img|VRAM ({abbr}`CL (channels_last)`)|PSNR (DF2K)|SSIM (DF2K)|PSNR (DIV2K)|SSIM (DIV2K)|Params|
|:-|-:|-:|-:|-:|-:|-:|-:|-:|-:|-:|
|dat_s  4x bf16                     |    0.55|    0.56|1.02x|  1.8133|    2.85 GB|27.68|0.8300|-    |-     |11,212,131|
|dat_s  3x bf16                     |    0.56|    0.55|1.00x|  1.8015|    2.82 GB|29.98|0.8846|-    |-     |11,249,059|
|dat_s  2x bf16                     |    0.57|    0.57|0.99x|  1.7395|    2.80 GB|34.12|0.9444|-    |-     |11,064,419|
|dat_s  1x bf16                     |    0.36|    0.36|1.00x|  2.7647|    2.79 GB|-    |-     |-    |-     |10,916,707|

### drct 
(1, 3, 480, 640) input, 1 warmup + 5 runs averaged
|Name|FPS|FPS ({abbr}`CL (channels_last)`)|{abbr}`CL (channels_last)` vs base|sec/img|VRAM ({abbr}`CL (channels_last)`)|PSNR (DF2K)|SSIM (DF2K)|PSNR (DIV2K)|SSIM (DIV2K)|Params|
|:-|-:|-:|-:|-:|-:|-:|-:|-:|-:|-:|
|drct  4x bf16                      |    0.18|    0.18|1.00x|  5.4825|    6.28 GB|28.06|0.8378|-    |-     |14,139,579|
|drct  3x bf16                      |    0.19|    0.18|1.00x|  5.3984|    6.25 GB|30.34|0.8910|-    |-     |14,176,507|
|drct  2x bf16                      |    0.18|    0.19|1.01x|  5.4143|    6.24 GB|34.54|0.9474|-    |-     |13,991,867|
|drct  1x bf16                      |    0.14|    0.15|1.11x|  7.3718|    6.23 GB|-    |-     |-    |-     |13,844,155|

### drct_l 
(1, 3, 480, 640) input, 1 warmup + 5 runs averaged
|Name|FPS|FPS ({abbr}`CL (channels_last)`)|{abbr}`CL (channels_last)` vs base|sec/img|VRAM ({abbr}`CL (channels_last)`)|PSNR (DF2K)|SSIM (DF2K)|PSNR (DIV2K)|SSIM (DIV2K)|Params|
|:-|-:|-:|-:|-:|-:|-:|-:|-:|-:|-:|
|drct_l  4x bf16                    |    0.09|    0.09|1.00x| 10.9431|    6.42 GB|28.70|0.8508|-    |-     |27,580,719|
|drct_l  3x bf16                    |    0.07|    0.07|1.05x| 14.1257|    6.39 GB|31.14|0.9004|-    |-     |27,617,647|
|drct_l  2x bf16                    |    0.09|    0.09|0.98x| 10.6995|    6.37 GB|35.17|0.9516|-    |-     |27,433,007|
|drct_l  1x bf16                    |    0.07|    0.07|0.99x| 13.8604|    6.36 GB|-    |-     |-    |-     |27,285,295|

### drct_xl 
(1, 3, 480, 640) input, 1 warmup + 5 runs averaged
|Name|FPS|FPS ({abbr}`CL (channels_last)`)|{abbr}`CL (channels_last)` vs base|sec/img|VRAM ({abbr}`CL (channels_last)`)|PSNR (DF2K)|SSIM (DF2K)|PSNR (DIV2K)|SSIM (DIV2K)|Params|
|:-|-:|-:|-:|-:|-:|-:|-:|-:|-:|-:|
|drct_xl  4x bf16                   |    0.08|    0.08|1.00x| 12.6742|    6.46 GB|-    |-     |-    |-     |32,061,099|
|drct_xl  3x bf16                   |    0.06|    0.06|1.00x| 17.1119|    6.44 GB|-    |-     |-    |-     |32,098,027|
|drct_xl  2x bf16                   |    0.08|    0.08|1.00x| 12.5408|    6.42 GB|-    |-     |-    |-     |31,913,387|
|drct_xl  1x bf16                   |    0.06|    0.05|0.84x| 15.9074|    6.41 GB|-    |-     |-    |-     |31,765,675|

### eimn_a 
(1, 3, 480, 640) input, 1 warmup + 5 runs averaged
|Name|FPS|FPS ({abbr}`CL (channels_last)`)|{abbr}`CL (channels_last)` vs base|sec/img|VRAM ({abbr}`CL (channels_last)`)|PSNR (DF2K)|SSIM (DF2K)|PSNR (DIV2K)|SSIM (DIV2K)|Params|
|:-|-:|-:|-:|-:|-:|-:|-:|-:|-:|-:|
|eimn_a  4x fp16                    |    8.36|    6.09|0.73x|  0.1196|    0.89 GB|26.68|0.8027|-    |-     |   880,870|
|eimn_a  3x fp16                    |    8.04|    5.81|0.72x|  0.1243|    0.87 GB|28.87|0.8660|-    |-     |   868,753|
|eimn_a  2x fp16                    |    8.38|    6.12|0.73x|  0.1193|    0.86 GB|33.15|0.9373|-    |-     |   860,098|
|eimn_a  1x fp16                    |    8.38|    6.12|0.73x|  0.1193|    0.86 GB|-    |-     |-    |-     |   854,905|

### eimn_l 
(1, 3, 480, 640) input, 1 warmup + 5 runs averaged
|Name|FPS|FPS ({abbr}`CL (channels_last)`)|{abbr}`CL (channels_last)` vs base|sec/img|VRAM ({abbr}`CL (channels_last)`)|PSNR (DF2K)|SSIM (DF2K)|PSNR (DIV2K)|SSIM (DIV2K)|Params|
|:-|-:|-:|-:|-:|-:|-:|-:|-:|-:|-:|
|eimn_l  4x fp16                    |    7.32|    5.34|0.73x|  0.1366|    0.89 GB|26.88|0.8084|-    |-     | 1,002,496|
|eimn_l  3x fp16                    |    7.48|    5.45|0.73x|  0.1337|    0.87 GB|29.05|0.8698|-    |-     |   990,379|
|eimn_l  2x fp16                    |    7.33|    5.34|0.73x|  0.1364|    0.87 GB|33.23|0.9381|-    |-     |   981,724|
|eimn_l  1x fp16                    |    7.35|    5.35|0.73x|  0.1360|    0.86 GB|-    |-     |-    |-     |   976,531|

### esrgan use_pixel_unshuffle=False
(1, 3, 480, 640) input, 1 warmup + 5 runs averaged
|Name|FPS|FPS ({abbr}`CL (channels_last)`)|{abbr}`CL (channels_last)` vs base|sec/img|VRAM ({abbr}`CL (channels_last)`)|PSNR (DF2K)|SSIM (DF2K)|PSNR (DIV2K)|SSIM (DIV2K)|Params|
|:-|-:|-:|-:|-:|-:|-:|-:|-:|-:|-:|
|esrgan use_pixel_unshuffle=False 4x fp16|    3.27|    4.60|1.40x|  0.3054|    2.48 GB|27.03|0.8153|26.73|0.8072|16,697,987|
|esrgan use_pixel_unshuffle=False 3x fp16|    3.45|    4.93|1.43x|  0.2901|    1.44 GB|-    |-     |-    |-     |16,661,059|
|esrgan use_pixel_unshuffle=False 2x fp16|    3.56|    4.99|1.40x|  0.2811|    0.70 GB|-    |-     |-    |-     |16,661,059|
|esrgan use_pixel_unshuffle=False 1x fp16|    3.56|    4.99|1.40x|  0.2810|    0.44 GB|-    |-     |-    |-     |16,624,131|

### esrgan use_pixel_unshuffle=True
(1, 3, 480, 640) input, 1 warmup + 5 runs averaged
|Name|FPS|FPS ({abbr}`CL (channels_last)`)|{abbr}`CL (channels_last)` vs base|sec/img|VRAM ({abbr}`CL (channels_last)`)|PSNR (DF2K)|SSIM (DF2K)|PSNR (DIV2K)|SSIM (DIV2K)|Params|
|:-|-:|-:|-:|-:|-:|-:|-:|-:|-:|-:|
|esrgan use_pixel_unshuffle=True 4x fp16|    3.28|    4.60|1.40x|  0.3051|    2.48 GB|27.03|0.8153|26.73|0.8072|16,697,987|
|esrgan use_pixel_unshuffle=True 3x fp16|    3.51|    4.91|1.40x|  0.2852|    1.44 GB|-    |-     |-    |-     |16,661,059|
|esrgan use_pixel_unshuffle=True 2x fp16|   17.98|   22.91|1.27x|  0.0556|    0.70 GB|-    |-     |-    |-     |16,703,171|
|esrgan use_pixel_unshuffle=True 1x fp16|   37.78|   54.15|1.43x|  0.0265|    0.26 GB|-    |-     |-    |-     |16,723,907|

### esrgan_lite 
(1, 3, 480, 640) input, 1 warmup + 5 runs averaged
|Name|FPS|FPS ({abbr}`CL (channels_last)`)|{abbr}`CL (channels_last)` vs base|sec/img|VRAM ({abbr}`CL (channels_last)`)|PSNR (DF2K)|SSIM (DF2K)|PSNR (DIV2K)|SSIM (DIV2K)|Params|
|:-|-:|-:|-:|-:|-:|-:|-:|-:|-:|-:|
|esrgan_lite  4x fp16               |    8.74|   12.17|1.39x|  0.1144|    1.24 GB|-    |-     |-    |-     | 5,021,155|
|esrgan_lite  3x fp16               |    9.50|   13.17|1.39x|  0.1053|    0.71 GB|-    |-     |-    |-     | 5,011,907|
|esrgan_lite  2x fp16               |   42.00|   56.61|1.35x|  0.0238|    0.34 GB|-    |-     |-    |-     | 5,023,747|
|esrgan_lite  1x fp16               |   74.28|  109.74|1.48x|  0.0135|    0.12 GB|-    |-     |-    |-     | 5,034,115|

### flexnet 
(1, 3, 480, 640) input, 1 warmup + 5 runs averaged
|Name|FPS|FPS ({abbr}`CL (channels_last)`)|{abbr}`CL (channels_last)` vs base|sec/img|VRAM ({abbr}`CL (channels_last)`)|PSNR (DF2K)|SSIM (DF2K)|PSNR (DIV2K)|SSIM (DIV2K)|Params|
|:-|-:|-:|-:|-:|-:|-:|-:|-:|-:|-:|
|flexnet  4x fp16                   |    3.45|    3.38|0.98x|  0.2900|    1.30 GB|-    |-     |-    |-     | 3,045,136|
|flexnet  3x fp16                   |    1.00|    2.65|2.65x|  0.9986|    1.29 GB|-    |-     |-    |-     | 3,020,923|
|flexnet  2x fp16                   |    3.45|    3.37|0.98x|  0.2901|    1.28 GB|-    |-     |-    |-     | 3,003,628|
|flexnet  1x fp16                   |    3.44|    3.38|0.98x|  0.2904|    1.28 GB|-    |-     |-    |-     | 2,993,251|

### hat_l 
(1, 3, 480, 640) input, 1 warmup + 5 runs averaged
|Name|FPS|FPS ({abbr}`CL (channels_last)`)|{abbr}`CL (channels_last)` vs base|sec/img|VRAM ({abbr}`CL (channels_last)`)|PSNR (DF2K)|SSIM (DF2K)|PSNR (DIV2K)|SSIM (DIV2K)|Params|
|:-|-:|-:|-:|-:|-:|-:|-:|-:|-:|-:|
|hat_l  4x bf16                     |    0.29|    0.31|1.07x|  3.4823|   10.41 GB|28.60|0.8498|-    |-     |40,846,575|
|hat_l  3x bf16                     |    0.25|    0.24|0.98x|  4.0592|   10.39 GB|30.92|0.8981|-    |-     |40,883,503|
|hat_l  2x bf16                     |    0.29|    0.31|1.07x|  3.4783|   10.37 GB|35.09|0.9513|-    |-     |40,698,863|
|hat_l  1x bf16                     |    0.19|    0.31|1.64x|  5.3148|   10.36 GB|-    |-     |-    |-     |40,551,151|

### hat_m 
(1, 3, 480, 640) input, 1 warmup + 5 runs averaged
|Name|FPS|FPS ({abbr}`CL (channels_last)`)|{abbr}`CL (channels_last)` vs base|sec/img|VRAM ({abbr}`CL (channels_last)`)|PSNR (DF2K)|SSIM (DF2K)|PSNR (DIV2K)|SSIM (DIV2K)|Params|
|:-|-:|-:|-:|-:|-:|-:|-:|-:|-:|-:|
|hat_m  4x bf16                     |    0.54|    0.58|1.06x|  1.8367|   10.30 GB|27.97|0.8368|-    |-     |20,772,507|
|hat_m  3x bf16                     |    0.32|    0.40|1.27x|  3.1324|   10.27 GB|30.23|0.8896|-    |-     |20,809,435|
|hat_m  2x bf16                     |    0.55|    0.58|1.06x|  1.8203|   10.26 GB|34.45|0.9466|-    |-     |20,624,795|
|hat_m  1x bf16                     |    0.55|    0.59|1.07x|  1.8278|   10.24 GB|-    |-     |-    |-     |20,477,083|

### hat_s 
(1, 3, 480, 640) input, 1 warmup + 5 runs averaged
|Name|FPS|FPS ({abbr}`CL (channels_last)`)|{abbr}`CL (channels_last)` vs base|sec/img|VRAM ({abbr}`CL (channels_last)`)|PSNR (DF2K)|SSIM (DF2K)|PSNR (DIV2K)|SSIM (DIV2K)|Params|
|:-|-:|-:|-:|-:|-:|-:|-:|-:|-:|-:|
|hat_s  4x bf16                     |    0.63|    0.66|1.05x|  1.5988|    9.85 GB|27.87|0.8346|-    |-     | 9,621,183|
|hat_s  3x bf16                     |    0.42|    0.43|1.04x|  2.4094|    9.83 GB|30.15|0.8879|-    |-     | 9,658,111|
|hat_s  2x bf16                     |    0.63|    0.67|1.06x|  1.5900|    9.81 GB|34.31|0.9459|-    |-     | 9,473,471|
|hat_s  1x bf16                     |    0.63|    0.67|1.07x|  1.5879|    9.80 GB|-    |-     |-    |-     | 9,325,759|

### hit_sir 
(1, 3, 480, 640) input, 1 warmup + 5 runs averaged
|Name|FPS|FPS ({abbr}`CL (channels_last)`)|{abbr}`CL (channels_last)` vs base|sec/img|VRAM ({abbr}`CL (channels_last)`)|PSNR (DF2K)|SSIM (DF2K)|PSNR (DIV2K)|SSIM (DIV2K)|Params|
|:-|-:|-:|-:|-:|-:|-:|-:|-:|-:|-:|
|hit_sir  4x bf16                   |    5.26|    5.12|0.97x|  0.1900|    1.34 GB|-    |-     |26.71|0.8045|   791,540|
|hit_sir  3x bf16                   |    4.99|    4.81|0.96x|  0.2002|    1.32 GB|-    |-     |28.93|0.8673|   780,179|
|hit_sir  2x bf16                   |    5.29|    5.13|0.97x|  0.1890|    1.30 GB|-    |-     |33.02|0.9365|   772,064|
|hit_sir  1x bf16                   |    5.29|    5.12|0.97x|  0.1892|    1.29 GB|-    |-     |-    |-     |   767,195|

### hit_sng 
(1, 3, 480, 640) input, 1 warmup + 5 runs averaged
|Name|FPS|FPS ({abbr}`CL (channels_last)`)|{abbr}`CL (channels_last)` vs base|sec/img|VRAM ({abbr}`CL (channels_last)`)|PSNR (DF2K)|SSIM (DF2K)|PSNR (DIV2K)|SSIM (DIV2K)|Params|
|:-|-:|-:|-:|-:|-:|-:|-:|-:|-:|-:|
|hit_sng  4x bf16                   |    4.09|    3.92|0.96x|  0.2444|    1.34 GB|-    |-     |26.75|0.8053| 1,032,060|
|hit_sng  3x bf16                   |    1.04|    2.43|2.34x|  0.9641|    1.32 GB|-    |-     |28.91|0.8671| 1,020,699|
|hit_sng  2x bf16                   |    4.11|    3.93|0.96x|  0.2432|    1.30 GB|-    |-     |33.01|0.9360| 1,012,584|
|hit_sng  1x bf16                   |    4.11|    3.93|0.96x|  0.2433|    1.29 GB|-    |-     |-    |-     | 1,007,715|

### hit_srf 
(1, 3, 480, 640) input, 1 warmup + 5 runs averaged
|Name|FPS|FPS ({abbr}`CL (channels_last)`)|{abbr}`CL (channels_last)` vs base|sec/img|VRAM ({abbr}`CL (channels_last)`)|PSNR (DF2K)|SSIM (DF2K)|PSNR (DIV2K)|SSIM (DIV2K)|Params|
|:-|-:|-:|-:|-:|-:|-:|-:|-:|-:|-:|
|hit_srf  4x bf16                   |    4.23|    4.19|0.99x|  0.2364|    1.34 GB|-    |-     |26.80|0.8069|   866,420|
|hit_srf  3x bf16                   |    4.26|    4.28|1.00x|  0.2346|    1.32 GB|-    |-     |28.99|0.8687|   855,059|
|hit_srf  2x bf16                   |    4.25|    4.20|0.99x|  0.2352|    1.30 GB|-    |-     |33.13|0.9372|   846,944|
|hit_srf  1x bf16                   |    4.24|    4.20|0.99x|  0.2357|    1.29 GB|-    |-     |-    |-     |   842,075|

### ipt 
(1, 3, 480, 640) input, 1 warmup + 5 runs averaged
|Name|FPS|FPS ({abbr}`CL (channels_last)`)|{abbr}`CL (channels_last)` vs base|sec/img|VRAM ({abbr}`CL (channels_last)`)|PSNR (DF2K)|SSIM (DF2K)|PSNR (DIV2K)|SSIM (DIV2K)|Params|
|:-|-:|-:|-:|-:|-:|-:|-:|-:|-:|-:|
|ipt  4x fp16                       |-       |-       |-       |-       |-       |-|-|-||-         |
|ipt  3x fp16                       |-       |-       |-       |-       |-       |-|-|-||-         |
|ipt  2x fp16                       |-       |-       |-       |-       |-       |-|-|-||-         |
|ipt  1x fp16                       |-       |-       |-       |-       |-       |-|-|-||-         |

### lmlt_base 
(1, 3, 480, 640) input, 1 warmup + 5 runs averaged
|Name|FPS|FPS ({abbr}`CL (channels_last)`)|{abbr}`CL (channels_last)` vs base|sec/img|VRAM ({abbr}`CL (channels_last)`)|PSNR (DF2K)|SSIM (DF2K)|PSNR (DIV2K)|SSIM (DIV2K)|Params|
|:-|-:|-:|-:|-:|-:|-:|-:|-:|-:|-:|
|lmlt_base  4x fp16                 |   15.08|   15.06|1.00x|  0.0663|    0.63 GB|26.44|0.7949|-    |-     |   671,808|
|lmlt_base  3x fp16                 |   15.33|   14.81|0.97x|  0.0652|    0.62 GB|28.48|0.8581|-    |-     |   660,447|
|lmlt_base  2x fp16                 |   15.20|   15.10|0.99x|  0.0658|    0.61 GB|32.52|0.9316|-    |-     |   652,332|
|lmlt_base  1x fp16                 |   15.12|   15.15|1.00x|  0.0662|    0.61 GB|-    |-     |-    |-     |   647,463|

### lmlt_large 
(1, 3, 480, 640) input, 1 warmup + 5 runs averaged
|Name|FPS|FPS ({abbr}`CL (channels_last)`)|{abbr}`CL (channels_last)` vs base|sec/img|VRAM ({abbr}`CL (channels_last)`)|PSNR (DF2K)|SSIM (DF2K)|PSNR (DIV2K)|SSIM (DIV2K)|Params|
|:-|-:|-:|-:|-:|-:|-:|-:|-:|-:|-:|
|lmlt_large  4x fp16                |   10.33|   10.30|1.00x|  0.0968|    0.87 GB|26.63|0.8001|-    |-     | 1,295,328|
|lmlt_large  3x fp16                |   10.25|   10.17|0.99x|  0.0976|    0.86 GB|28.72|0.8628|-    |-     | 1,279,431|
|lmlt_large  2x fp16                |   10.38|   10.33|1.00x|  0.0964|    0.85 GB|32.75|0.9336|-    |-     | 1,268,076|
|lmlt_large  1x fp16                |   10.38|   10.34|1.00x|  0.0963|    0.85 GB|-    |-     |-    |-     | 1,261,263|

### lmlt_tiny 
(1, 3, 480, 640) input, 1 warmup + 5 runs averaged
|Name|FPS|FPS ({abbr}`CL (channels_last)`)|{abbr}`CL (channels_last)` vs base|sec/img|VRAM ({abbr}`CL (channels_last)`)|PSNR (DF2K)|SSIM (DF2K)|PSNR (DIV2K)|SSIM (DIV2K)|Params|
|:-|-:|-:|-:|-:|-:|-:|-:|-:|-:|-:|
|lmlt_tiny  4x fp16                 |   28.70|   28.48|0.99x|  0.0348|    0.37 GB|26.08|0.7838|-    |-     |   251,040|
|lmlt_tiny  3x fp16                 |   22.05|   26.12|1.18x|  0.0454|    0.36 GB|28.10|0.8503|-    |-     |   244,215|
|lmlt_tiny  2x fp16                 |   28.71|   28.79|1.00x|  0.0348|    0.35 GB|32.04|0.9273|-    |-     |   239,340|
|lmlt_tiny  1x fp16                 |   28.42|   28.49|1.00x|  0.0352|    0.35 GB|-    |-     |-    |-     |   236,415|

### man 
(1, 3, 480, 640) input, 1 warmup + 5 runs averaged
|Name|FPS|FPS ({abbr}`CL (channels_last)`)|{abbr}`CL (channels_last)` vs base|sec/img|VRAM ({abbr}`CL (channels_last)`)|PSNR (DF2K)|SSIM (DF2K)|PSNR (DIV2K)|SSIM (DIV2K)|Params|
|:-|-:|-:|-:|-:|-:|-:|-:|-:|-:|-:|
|man  4x fp16                       |    1.26|    0.86|0.68x|  0.7928|    1.53 GB|27.26|0.8197|-    |-     | 8,712,612|
|man  3x fp16                       |    0.63|    0.52|0.83x|  1.5960|    1.52 GB|29.52|0.8782|-    |-     | 8,678,571|
|man  2x fp16                       |    1.26|    0.86|0.69x|  0.7928|    1.51 GB|33.73|0.9422|-    |-     | 8,654,256|
|man  1x fp16                       |    1.26|    0.86|0.68x|  0.7921|    1.51 GB|-    |-     |-    |-     | 8,639,667|

### man_light 
(1, 3, 480, 640) input, 1 warmup + 5 runs averaged
|Name|FPS|FPS ({abbr}`CL (channels_last)`)|{abbr}`CL (channels_last)` vs base|sec/img|VRAM ({abbr}`CL (channels_last)`)|PSNR (DF2K)|SSIM (DF2K)|PSNR (DIV2K)|SSIM (DIV2K)|Params|
|:-|-:|-:|-:|-:|-:|-:|-:|-:|-:|-:|
|man_light  4x fp16                 |    6.02|    4.42|0.73x|  0.1660|    0.53 GB|26.70|0.8052|-    |-     |   842,892|
|man_light  3x fp16                 |    6.13|    4.45|0.73x|  0.1631|    0.52 GB|-    |-     |-    |-     |   831,531|
|man_light  2x fp16                 |    6.04|    4.42|0.73x|  0.1656|    0.51 GB|-    |-     |-    |-     |   823,416|
|man_light  1x fp16                 |    6.03|    4.42|0.73x|  0.1658|    0.50 GB|-    |-     |-    |-     |   818,547|

### man_tiny 
(1, 3, 480, 640) input, 1 warmup + 5 runs averaged
|Name|FPS|FPS ({abbr}`CL (channels_last)`)|{abbr}`CL (channels_last)` vs base|sec/img|VRAM ({abbr}`CL (channels_last)`)|PSNR (DF2K)|SSIM (DF2K)|PSNR (DIV2K)|SSIM (DIV2K)|Params|
|:-|-:|-:|-:|-:|-:|-:|-:|-:|-:|-:|
|man_tiny  4x fp16                  |   35.02|   27.37|0.78x|  0.0286|    0.43 GB|25.84|0.7786|-    |-     |   150,024|
|man_tiny  3x fp16                  |   35.52|   27.12|0.76x|  0.0282|    0.41 GB|-    |-     |-    |-     |   140,931|
|man_tiny  2x fp16                  |   35.34|   27.75|0.79x|  0.0283|    0.40 GB|-    |-     |-    |-     |   134,436|
|man_tiny  1x fp16                  |   35.45|   27.53|0.78x|  0.0282|    0.40 GB|-    |-     |-    |-     |   130,539|

### metaflexnet 
(1, 3, 480, 640) input, 1 warmup + 5 runs averaged
|Name|FPS|FPS ({abbr}`CL (channels_last)`)|{abbr}`CL (channels_last)` vs base|sec/img|VRAM ({abbr}`CL (channels_last)`)|PSNR (DF2K)|SSIM (DF2K)|PSNR (DIV2K)|SSIM (DIV2K)|Params|
|:-|-:|-:|-:|-:|-:|-:|-:|-:|-:|-:|
|metaflexnet  4x fp16               |    9.28|    9.63|1.04x|  0.1078|    1.29 GB|-    |-     |-    |-     |38,053,568|
|metaflexnet  3x fp16               |    9.37|    9.83|1.05x|  0.1067|    1.27 GB|-    |-     |-    |-     |38,035,403|
|metaflexnet  2x fp16               |    9.31|    9.64|1.03x|  0.1074|    1.27 GB|-    |-     |-    |-     |38,022,428|
|metaflexnet  1x fp16               |    9.31|    9.62|1.03x|  0.1075|    1.26 GB|-    |-     |-    |-     |38,014,643|

### moesr2 
(1, 3, 480, 640) input, 1 warmup + 5 runs averaged
|Name|FPS|FPS ({abbr}`CL (channels_last)`)|{abbr}`CL (channels_last)` vs base|sec/img|VRAM ({abbr}`CL (channels_last)`)|PSNR (DF2K)|SSIM (DF2K)|PSNR (DIV2K)|SSIM (DIV2K)|Params|
|:-|-:|-:|-:|-:|-:|-:|-:|-:|-:|-:|
|moesr2  4x fp16                    |    3.93|    3.68|0.94x|  0.2542|    0.91 GB|-    |-     |-    |-     |16,547,008|
|moesr2  3x fp16                    |    3.81|    1.70|0.45x|  0.2625|    0.90 GB|-    |-     |-    |-     |16,534,891|
|moesr2  2x fp16                    |    3.93|    3.67|0.93x|  0.2544|    0.89 GB|-    |-     |-    |-     |16,526,236|
|moesr2  1x fp16                    |    3.94|    3.68|0.93x|  0.2537|    0.89 GB|-    |-     |-    |-     |16,521,043|

### mosr 
(1, 3, 480, 640) input, 1 warmup + 5 runs averaged
|Name|FPS|FPS ({abbr}`CL (channels_last)`)|{abbr}`CL (channels_last)` vs base|sec/img|VRAM ({abbr}`CL (channels_last)`)|PSNR (DF2K)|SSIM (DF2K)|PSNR (DIV2K)|SSIM (DIV2K)|Params|
|:-|-:|-:|-:|-:|-:|-:|-:|-:|-:|-:|
|mosr  4x fp16                      |    9.27|   11.44|1.23x|  0.1079|    0.49 GB|-    |-     |-    |-     | 4,287,600|
|mosr  3x fp16                      |    7.13|    1.99|0.28x|  0.1403|    0.47 GB|-    |-     |-    |-     | 4,275,483|
|mosr  2x fp16                      |    9.29|   11.45|1.23x|  0.1077|    0.46 GB|-    |-     |-    |-     | 4,266,828|
|mosr  1x fp16                      |    9.29|   11.48|1.24x|  0.1076|    0.46 GB|-    |-     |-    |-     | 4,261,635|

### mosr_t 
(1, 3, 480, 640) input, 1 warmup + 5 runs averaged
|Name|FPS|FPS ({abbr}`CL (channels_last)`)|{abbr}`CL (channels_last)` vs base|sec/img|VRAM ({abbr}`CL (channels_last)`)|PSNR (DF2K)|SSIM (DF2K)|PSNR (DIV2K)|SSIM (DIV2K)|Params|
|:-|-:|-:|-:|-:|-:|-:|-:|-:|-:|-:|
|mosr_t  4x fp16                    |   53.80|   64.10|1.19x|  0.0186|    0.36 GB|-    |-     |-    |-     |   609,888|
|mosr_t  3x fp16                    |   54.82|   65.20|1.19x|  0.0182|    0.35 GB|-    |-     |-    |-     |   600,795|
|mosr_t  2x fp16                    |   53.57|   65.56|1.22x|  0.0187|    0.34 GB|-    |-     |-    |-     |   594,300|
|mosr_t  1x fp16                    |   54.78|   66.46|1.21x|  0.0183|    0.33 GB|-    |-     |-    |-     |   590,403|

### omnisr 
(1, 3, 480, 640) input, 1 warmup + 5 runs averaged
|Name|FPS|FPS ({abbr}`CL (channels_last)`)|{abbr}`CL (channels_last)` vs base|sec/img|VRAM ({abbr}`CL (channels_last)`)|PSNR (DF2K)|SSIM (DF2K)|PSNR (DIV2K)|SSIM (DIV2K)|Params|
|:-|-:|-:|-:|-:|-:|-:|-:|-:|-:|-:|
|omnisr  4x fp16                    |    7.80|    7.64|0.98x|  0.1283|    1.18 GB|26.95|0.8105|26.64|0.8018|   805,376|
|omnisr  3x fp16                    |    7.29|    7.82|1.07x|  0.1371|    1.17 GB|29.12|0.8712|28.84|0.8656|   793,259|
|omnisr  2x fp16                    |    7.80|    7.66|0.98x|  0.1282|    1.16 GB|33.30|0.9386|33.05|0.9363|   784,604|
|omnisr  1x fp16                    |    7.78|    7.65|0.98x|  0.1285|    1.16 GB|-    |-     |-    |-     |   779,411|

### plksr 
(1, 3, 480, 640) input, 1 warmup + 5 runs averaged
|Name|FPS|FPS ({abbr}`CL (channels_last)`)|{abbr}`CL (channels_last)` vs base|sec/img|VRAM ({abbr}`CL (channels_last)`)|PSNR (DF2K)|SSIM (DF2K)|PSNR (DIV2K)|SSIM (DIV2K)|Params|
|:-|-:|-:|-:|-:|-:|-:|-:|-:|-:|-:|
|plksr  4x fp16                     |   11.95|   15.74|1.32x|  0.0836|    0.29 GB|26.85|0.8097|26.69|0.8054| 7,386,096|
|plksr  3x fp16                     |   12.18|   15.89|1.30x|  0.0821|    0.27 GB|29.10|0.8713|28.86|0.8666| 7,373,979|
|plksr  2x fp16                     |   12.04|   15.87|1.32x|  0.0830|    0.25 GB|33.36|0.9395|32.99|0.9365| 7,365,324|
|plksr  1x fp16                     |   12.04|   15.87|1.32x|  0.0831|    0.24 GB|-    |-     |-    |-     | 7,360,131|

### plksr_tiny 
(1, 3, 480, 640) input, 1 warmup + 5 runs averaged
|Name|FPS|FPS ({abbr}`CL (channels_last)`)|{abbr}`CL (channels_last)` vs base|sec/img|VRAM ({abbr}`CL (channels_last)`)|PSNR (DF2K)|SSIM (DF2K)|PSNR (DIV2K)|SSIM (DIV2K)|Params|
|:-|-:|-:|-:|-:|-:|-:|-:|-:|-:|-:|
|plksr_tiny  4x fp16                |   36.96|   49.55|1.34x|  0.0271|    0.27 GB|26.34|0.7942|26.12|0.7888| 2,370,544|
|plksr_tiny  3x fp16                |   37.73|   40.54|1.07x|  0.0265|    0.24 GB|28.51|0.8599|28.35|0.8571| 2,358,427|
|plksr_tiny  2x fp16                |   37.51|   50.63|1.35x|  0.0267|    0.22 GB|32.58|0.9328|32.43|0.9314| 2,349,772|
|plksr_tiny  1x fp16                |   37.49|   50.55|1.35x|  0.0267|    0.21 GB|-    |-     |-    |-     | 2,344,579|

### rcan 
(1, 3, 480, 640) input, 1 warmup + 5 runs averaged
|Name|FPS|FPS ({abbr}`CL (channels_last)`)|{abbr}`CL (channels_last)` vs base|sec/img|VRAM ({abbr}`CL (channels_last)`)|PSNR (DF2K)|SSIM (DF2K)|PSNR (DIV2K)|SSIM (DIV2K)|Params|
|:-|-:|-:|-:|-:|-:|-:|-:|-:|-:|-:|
|rcan  4x fp16                      |    5.36|    7.02|1.31x|  0.1866|    1.40 GB|-    |-     |26.82|0.8087|15,592,379|
|rcan  3x fp16                      |    5.58|    7.58|1.36x|  0.1792|    0.86 GB|-    |-     |29.09|0.8702|15,629,307|
|rcan  2x fp16                      |    5.63|    7.77|1.38x|  0.1775|    0.48 GB|-    |-     |33.34|0.9384|15,444,667|
|rcan  1x fp16                      |    5.72|    7.94|1.39x|  0.1749|    0.28 GB|-    |-     |-    |-     |15,296,955|

### rcanspatialselayer 
(1, 3, 480, 640) input, 1 warmup + 5 runs averaged
|Name|FPS|FPS ({abbr}`CL (channels_last)`)|{abbr}`CL (channels_last)` vs base|sec/img|VRAM ({abbr}`CL (channels_last)`)|PSNR (DF2K)|SSIM (DF2K)|PSNR (DIV2K)|SSIM (DIV2K)|Params|
|:-|-:|-:|-:|-:|-:|-:|-:|-:|-:|-:|
|rcanspatialselayer  4x fp16        |    5.38|    7.23|1.34x|  0.1860|    1.40 GB|-    |-     |-    |-     |15,489,355|
|rcanspatialselayer  3x fp16        |    5.57|    7.88|1.41x|  0.1794|    0.86 GB|-    |-     |-    |-     |15,526,283|
|rcanspatialselayer  2x fp16        |    5.68|    8.01|1.41x|  0.1761|    0.48 GB|-    |-     |-    |-     |15,341,643|
|rcanspatialselayer  1x fp16        |    5.77|    8.19|1.42x|  0.1732|    0.29 GB|-    |-     |-    |-     |15,193,931|

### realcugan 
(1, 3, 480, 640) input, 1 warmup + 5 runs averaged
|Name|FPS|FPS ({abbr}`CL (channels_last)`)|{abbr}`CL (channels_last)` vs base|sec/img|VRAM ({abbr}`CL (channels_last)`)|PSNR (DF2K)|SSIM (DF2K)|PSNR (DIV2K)|SSIM (DIV2K)|Params|
|:-|-:|-:|-:|-:|-:|-:|-:|-:|-:|-:|
|realcugan  4x fp16                 |   44.12|   61.28|1.39x|  0.0227|    0.95 GB|-    |-     |-    |-     | 1,406,812|
|realcugan  3x fp16                 |   22.19|   34.94|1.57x|  0.0451|    1.71 GB|-    |-     |-    |-     | 1,286,326|
|realcugan  2x fp16                 |   59.75|   81.17|1.36x|  0.0167|    0.78 GB|-    |-     |-    |-     | 1,284,598|
|realcugan  1x fp16                 |-       |-       |-       |-       |-       |-|-|-||-         |

### realplksr dysample layer_norm=False
(1, 3, 480, 640) input, 1 warmup + 5 runs averaged
|Name|FPS|FPS ({abbr}`CL (channels_last)`)|{abbr}`CL (channels_last)` vs base|sec/img|VRAM ({abbr}`CL (channels_last)`)|PSNR (DF2K)|SSIM (DF2K)|PSNR (DIV2K)|SSIM (DIV2K)|Params|
|:-|-:|-:|-:|-:|-:|-:|-:|-:|-:|-:|
|realplksr dysample layer_norm=False 4x fp16|    5.55|    5.77|1.04x|  0.1801|    2.21 GB|-    |-     |-    |-     | 7,402,243|
|realplksr dysample layer_norm=False 3x fp16|    4.99|    3.67|0.73x|  0.2004|    0.79 GB|-    |-     |-    |-     | 7,380,617|
|realplksr dysample layer_norm=False 2x fp16|    5.79|    6.08|1.05x|  0.1727|    0.39 GB|-    |-     |-    |-     | 7,369,747|
|realplksr dysample layer_norm=False 1x fp16|    5.83|    6.14|1.05x|  0.1717|    0.39 GB|-    |-     |-    |-     | 7,363,757|

### realplksr dysample layer_norm=True
(1, 3, 480, 640) input, 1 warmup + 5 runs averaged
|Name|FPS|FPS ({abbr}`CL (channels_last)`)|{abbr}`CL (channels_last)` vs base|sec/img|VRAM ({abbr}`CL (channels_last)`)|PSNR (DF2K)|SSIM (DF2K)|PSNR (DIV2K)|SSIM (DIV2K)|Params|
|:-|-:|-:|-:|-:|-:|-:|-:|-:|-:|-:|
|realplksr dysample layer_norm=True 4x fp16|    7.86|    9.09|1.16x|  0.1272|    2.21 GB|-    |-     |-    |-     | 7,402,243|
|realplksr dysample layer_norm=True 3x fp16|    7.90|    8.60|1.09x|  0.1265|    0.79 GB|-    |-     |-    |-     | 7,380,617|
|realplksr dysample layer_norm=True 2x fp16|    8.37|    9.96|1.19x|  0.1194|    0.32 GB|-    |-     |-    |-     | 7,369,747|
|realplksr dysample layer_norm=True 1x fp16|    8.41|   10.05|1.19x|  0.1188|    0.32 GB|-    |-     |-    |-     | 7,363,757|

### realplksr pixelshuffle layer_norm=False
(1, 3, 480, 640) input, 1 warmup + 5 runs averaged
|Name|FPS|FPS ({abbr}`CL (channels_last)`)|{abbr}`CL (channels_last)` vs base|sec/img|VRAM ({abbr}`CL (channels_last)`)|PSNR (DF2K)|SSIM (DF2K)|PSNR (DIV2K)|SSIM (DIV2K)|Params|
|:-|-:|-:|-:|-:|-:|-:|-:|-:|-:|-:|
|realplksr pixelshuffle layer_norm=False 4x fp16|    5.83|    6.13|1.05x|  0.1716|    0.44 GB|-    |-     |-    |-     | 7,389,680|
|realplksr pixelshuffle layer_norm=False 3x fp16|    5.84|    6.18|1.06x|  0.1714|    0.42 GB|-    |-     |-    |-     | 7,377,563|
|realplksr pixelshuffle layer_norm=False 2x fp16|    5.84|    6.15|1.05x|  0.1713|    0.40 GB|-    |-     |-    |-     | 7,368,908|
|realplksr pixelshuffle layer_norm=False 1x fp16|    5.84|    6.15|1.05x|  0.1713|    0.39 GB|-    |-     |-    |-     | 7,363,715|

### realplksr pixelshuffle layer_norm=True
(1, 3, 480, 640) input, 1 warmup + 5 runs averaged
|Name|FPS|FPS ({abbr}`CL (channels_last)`)|{abbr}`CL (channels_last)` vs base|sec/img|VRAM ({abbr}`CL (channels_last)`)|PSNR (DF2K)|SSIM (DF2K)|PSNR (DIV2K)|SSIM (DIV2K)|Params|
|:-|-:|-:|-:|-:|-:|-:|-:|-:|-:|-:|
|realplksr pixelshuffle layer_norm=True 4x fp16|    8.43|   10.06|1.19x|  0.1186|    0.37 GB|-    |-     |-    |-     | 7,389,680|
|realplksr pixelshuffle layer_norm=True 3x fp16|    8.05|    2.39|0.30x|  0.1242|    0.35 GB|-    |-     |-    |-     | 7,377,563|
|realplksr pixelshuffle layer_norm=True 2x fp16|    8.48|   10.15|1.20x|  0.1179|    0.33 GB|-    |-     |-    |-     | 7,368,908|
|realplksr pixelshuffle layer_norm=True 1x fp16|    8.46|   10.13|1.20x|  0.1182|    0.32 GB|-    |-     |-    |-     | 7,363,715|

### rgt 
(1, 3, 480, 640) input, 1 warmup + 5 runs averaged
|Name|FPS|FPS ({abbr}`CL (channels_last)`)|{abbr}`CL (channels_last)` vs base|sec/img|VRAM ({abbr}`CL (channels_last)`)|PSNR (DF2K)|SSIM (DF2K)|PSNR (DIV2K)|SSIM (DIV2K)|Params|
|:-|-:|-:|-:|-:|-:|-:|-:|-:|-:|-:|
|rgt  4x bf16                       |-       |-       |-       |-       |-       |-|-|-||-         |
|rgt  3x bf16                       |-       |-       |-       |-       |-       |-|-|-||-         |
|rgt  2x bf16                       |-       |-       |-       |-       |-       |-|-|-||-         |
|rgt  1x bf16                       |-       |-       |-       |-       |-       |-|-|-||-         |

### rgt_s 
(1, 3, 480, 640) input, 1 warmup + 5 runs averaged
|Name|FPS|FPS ({abbr}`CL (channels_last)`)|{abbr}`CL (channels_last)` vs base|sec/img|VRAM ({abbr}`CL (channels_last)`)|PSNR (DF2K)|SSIM (DF2K)|PSNR (DIV2K)|SSIM (DIV2K)|Params|
|:-|-:|-:|-:|-:|-:|-:|-:|-:|-:|-:|
|rgt_s  4x bf16                     |-       |-       |-       |-       |-       |-|-|-||-         |
|rgt_s  3x bf16                     |-       |-       |-       |-       |-       |-|-|-||-         |
|rgt_s  2x bf16                     |-       |-       |-       |-       |-       |-|-|-||-         |
|rgt_s  1x bf16                     |-       |-       |-       |-       |-       |-|-|-||-         |

### scunet_aaf6aa 
(1, 3, 480, 640) input, 1 warmup + 5 runs averaged
|Name|FPS|FPS ({abbr}`CL (channels_last)`)|{abbr}`CL (channels_last)` vs base|sec/img|VRAM ({abbr}`CL (channels_last)`)|PSNR (DF2K)|SSIM (DF2K)|PSNR (DIV2K)|SSIM (DIV2K)|Params|
|:-|-:|-:|-:|-:|-:|-:|-:|-:|-:|-:|
|scunet_aaf6aa  4x fp16             |    8.52|   10.92|1.28x|  0.1174|    3.37 GB|-    |-     |-    |-     |15,207,468|
|scunet_aaf6aa  3x fp16             |   10.63|    5.60|0.53x|  0.0940|    1.02 GB|-    |-     |-    |-     |15,170,540|
|scunet_aaf6aa  2x fp16             |   11.49|   14.17|1.23x|  0.0870|    1.02 GB|-    |-     |-    |-     |15,170,540|
|scunet_aaf6aa  1x fp16             |   27.29|   27.73|1.02x|  0.0366|    0.80 GB|-    |-     |-    |-     | 9,699,756|

### span 
(1, 3, 480, 640) input, 1 warmup + 5 runs averaged
|Name|FPS|FPS ({abbr}`CL (channels_last)`)|{abbr}`CL (channels_last)` vs base|sec/img|VRAM ({abbr}`CL (channels_last)`)|PSNR (DF2K)|SSIM (DF2K)|PSNR (DIV2K)|SSIM (DIV2K)|Params|
|:-|-:|-:|-:|-:|-:|-:|-:|-:|-:|-:|
|span  4x fp16                      |   94.39|   95.91|1.02x|  0.0106|    0.60 GB|26.18|0.7879|-    |-     | 2,236,728|
|span  3x fp16                      |   75.39|  104.90|1.39x|  0.0133|    0.59 GB|-    |-     |-    |-     | 2,227,635|
|span  2x fp16                      |   95.21|  102.49|1.08x|  0.0105|    0.58 GB|32.24|0.9294|-    |-     | 2,221,140|
|span  1x fp16                      |   90.01|  102.47|1.14x|  0.0111|    0.58 GB|-    |-     |-    |-     | 2,217,243|

### spanplus 
(1, 3, 480, 640) input, 1 warmup + 5 runs averaged
|Name|FPS|FPS ({abbr}`CL (channels_last)`)|{abbr}`CL (channels_last)` vs base|sec/img|VRAM ({abbr}`CL (channels_last)`)|PSNR (DF2K)|SSIM (DF2K)|PSNR (DIV2K)|SSIM (DIV2K)|Params|
|:-|-:|-:|-:|-:|-:|-:|-:|-:|-:|-:|
|spanplus  4x fp16                  |   54.73|   50.32|0.92x|  0.0183|    2.15 GB|-    |-     |-    |-     | 2,228,507|
|spanplus  3x fp16                  |   56.88|   32.95|0.58x|  0.0176|    1.23 GB|-    |-     |-    |-     | 2,223,075|
|spanplus  2x fp16                  |   79.79|   82.80|1.04x|  0.0125|    0.58 GB|-    |-     |-    |-     | 2,219,195|
|spanplus  1x fp16                  |   91.82|   97.73|1.06x|  0.0109|    0.30 GB|-    |-     |-    |-     | 2,216,867|

### spanplus_s 
(1, 3, 480, 640) input, 1 warmup + 5 runs averaged
|Name|FPS|FPS ({abbr}`CL (channels_last)`)|{abbr}`CL (channels_last)` vs base|sec/img|VRAM ({abbr}`CL (channels_last)`)|PSNR (DF2K)|SSIM (DF2K)|PSNR (DIV2K)|SSIM (DIV2K)|Params|
|:-|-:|-:|-:|-:|-:|-:|-:|-:|-:|-:|
|spanplus_s  4x fp16                |   74.76|   67.28|0.90x|  0.0134|    1.55 GB|-    |-     |-    |-     |   687,707|
|spanplus_s  3x fp16                |   96.68|   36.02|0.37x|  0.0103|    0.89 GB|-    |-     |-    |-     |   684,067|
|spanplus_s  2x fp16                |  124.09|  122.62|0.99x|  0.0081|    0.42 GB|-    |-     |-    |-     |   681,467|
|spanplus_s  1x fp16                |  132.89|  136.83|1.03x|  0.0075|    0.20 GB|-    |-     |-    |-     |   679,907|

### spanplus_st 
(1, 3, 480, 640) input, 1 warmup + 5 runs averaged
|Name|FPS|FPS ({abbr}`CL (channels_last)`)|{abbr}`CL (channels_last)` vs base|sec/img|VRAM ({abbr}`CL (channels_last)`)|PSNR (DF2K)|SSIM (DF2K)|PSNR (DIV2K)|SSIM (DIV2K)|Params|
|:-|-:|-:|-:|-:|-:|-:|-:|-:|-:|-:|
|spanplus_st  4x fp16               |   99.32|  105.79|1.07x|  0.0101|    0.32 GB|-    |-     |-    |-     | 2,236,728|
|spanplus_st  3x fp16               |   94.71|  109.03|1.15x|  0.0106|    0.31 GB|-    |-     |-    |-     | 2,227,635|
|spanplus_st  2x fp16               |  103.82|  113.71|1.10x|  0.0096|    0.30 GB|-    |-     |-    |-     | 2,221,140|
|spanplus_st  1x fp16               |  100.57|  110.92|1.10x|  0.0099|    0.30 GB|-    |-     |-    |-     | 2,217,243|

### spanplus_sts 
(1, 3, 480, 640) input, 1 warmup + 5 runs averaged
|Name|FPS|FPS ({abbr}`CL (channels_last)`)|{abbr}`CL (channels_last)` vs base|sec/img|VRAM ({abbr}`CL (channels_last)`)|PSNR (DF2K)|SSIM (DF2K)|PSNR (DIV2K)|SSIM (DIV2K)|Params|
|:-|-:|-:|-:|-:|-:|-:|-:|-:|-:|-:|
|spanplus_sts  4x fp16              |  133.57|  154.85|1.16x|  0.0075|    0.23 GB|-    |-     |-    |-     |   693,160|
|spanplus_sts  3x fp16              |   98.29|  163.98|1.67x|  0.0102|    0.21 GB|-    |-     |-    |-     |   687,091|
|spanplus_sts  2x fp16              |  161.06|  167.80|1.04x|  0.0062|    0.21 GB|-    |-     |-    |-     |   682,756|
|spanplus_sts  1x fp16              |  158.99|  163.55|1.03x|  0.0063|    0.20 GB|-    |-     |-    |-     |   680,155|

### srformer 
(1, 3, 480, 640) input, 1 warmup + 5 runs averaged
|Name|FPS|FPS ({abbr}`CL (channels_last)`)|{abbr}`CL (channels_last)` vs base|sec/img|VRAM ({abbr}`CL (channels_last)`)|PSNR (DF2K)|SSIM (DF2K)|PSNR (DIV2K)|SSIM (DIV2K)|Params|
|:-|-:|-:|-:|-:|-:|-:|-:|-:|-:|-:|
|srformer  4x bf16                  |    0.29|    0.28|0.99x|  3.4944|    3.65 GB|27.68|0.8311|-    |-     |10,543,503|
|srformer  3x bf16                  |    0.21|    0.22|1.03x|  4.7708|    3.62 GB|30.04|0.8865|-    |-     |10,580,431|
|srformer  2x bf16                  |    0.29|    0.29|1.00x|  3.4968|    3.61 GB|34.09|0.9449|-    |-     |10,395,791|
|srformer  1x bf16                  |    0.28|    0.29|1.03x|  3.5791|    3.60 GB|-    |-     |-    |-     |10,248,079|

### srformer_light 
(1, 3, 480, 640) input, 1 warmup + 5 runs averaged
|Name|FPS|FPS ({abbr}`CL (channels_last)`)|{abbr}`CL (channels_last)` vs base|sec/img|VRAM ({abbr}`CL (channels_last)`)|PSNR (DF2K)|SSIM (DF2K)|PSNR (DIV2K)|SSIM (DIV2K)|Params|
|:-|-:|-:|-:|-:|-:|-:|-:|-:|-:|-:|
|srformer_light  4x bf16            |    1.04|    1.05|1.01x|  0.9627|    1.47 GB|26.67|0.8032|-    |-     |   872,748|
|srformer_light  3x bf16            |    0.88|    0.99|1.12x|  1.1332|    1.45 GB|28.81|0.8655|-    |-     |   861,387|
|srformer_light  2x bf16            |    1.06|    1.05|1.00x|  0.9475|    1.43 GB|32.91|0.9353|-    |-     |   853,272|
|srformer_light  1x bf16            |    1.03|    1.06|1.03x|  0.9756|    1.42 GB|-    |-     |-    |-     |   848,403|

### superultracompact 
(1, 3, 480, 640) input, 1 warmup + 5 runs averaged
|Name|FPS|FPS ({abbr}`CL (channels_last)`)|{abbr}`CL (channels_last)` vs base|sec/img|VRAM ({abbr}`CL (channels_last)`)|PSNR (DF2K)|SSIM (DF2K)|PSNR (DIV2K)|SSIM (DIV2K)|Params|
|:-|-:|-:|-:|-:|-:|-:|-:|-:|-:|-:|
|superultracompact  4x fp16         |  730.61|  851.05|1.16x|  0.0014|    0.18 GB|-    |-     |-    |-     |    52,968|
|superultracompact  3x fp16         |  129.11| 1070.15|8.29x|  0.0077|    0.11 GB|-    |-     |-    |-     |    48,411|
|superultracompact  2x fp16         |  913.72| 1277.35|1.40x|  0.0011|    0.06 GB|-    |-     |-    |-     |    45,156|
|superultracompact  1x fp16         |  974.30| 1339.36|1.37x|  0.0010|    0.04 GB|-    |-     |-    |-     |    43,203|

### swin2sr_l 
(1, 3, 480, 640) input, 1 warmup + 5 runs averaged
|Name|FPS|FPS ({abbr}`CL (channels_last)`)|{abbr}`CL (channels_last)` vs base|sec/img|VRAM ({abbr}`CL (channels_last)`)|PSNR (DF2K)|SSIM (DF2K)|PSNR (DIV2K)|SSIM (DIV2K)|Params|
|:-|-:|-:|-:|-:|-:|-:|-:|-:|-:|-:|
|swin2sr_l  4x bf16                 |    0.41|    0.40|1.00x|  2.4684|    3.37 GB|-    |-     |-    |-     |28,785,859|
|swin2sr_l  3x bf16                 |-       |-       |-       |-       |-       |-|-|-||-         |
|swin2sr_l  2x bf16                 |-       |-       |-       |-       |-       |-|-|-||-         |
|swin2sr_l  1x bf16                 |-       |-       |-       |-       |-       |-|-|-||-         |

### swin2sr_m 
(1, 3, 480, 640) input, 1 warmup + 5 runs averaged
|Name|FPS|FPS ({abbr}`CL (channels_last)`)|{abbr}`CL (channels_last)` vs base|sec/img|VRAM ({abbr}`CL (channels_last)`)|PSNR (DF2K)|SSIM (DF2K)|PSNR (DIV2K)|SSIM (DIV2K)|Params|
|:-|-:|-:|-:|-:|-:|-:|-:|-:|-:|-:|
|swin2sr_m  4x bf16                 |    0.62|    0.60|0.97x|  1.6154|    2.51 GB|27.51|0.8271|-    |-     |12,239,283|
|swin2sr_m  3x bf16                 |    0.39|    0.60|1.54x|  2.5719|    2.49 GB|-    |-     |-    |-     |12,276,211|
|swin2sr_m  2x bf16                 |    0.63|    0.62|1.00x|  1.5970|    2.47 GB|33.89|0.9431|-    |-     |12,091,571|
|swin2sr_m  1x bf16                 |    0.63|    0.63|0.99x|  1.5907|    2.46 GB|-    |-     |-    |-     |11,943,859|

### swin2sr_s 
(1, 3, 480, 640) input, 1 warmup + 5 runs averaged
|Name|FPS|FPS ({abbr}`CL (channels_last)`)|{abbr}`CL (channels_last)` vs base|sec/img|VRAM ({abbr}`CL (channels_last)`)|PSNR (DF2K)|SSIM (DF2K)|PSNR (DIV2K)|SSIM (DIV2K)|Params|
|:-|-:|-:|-:|-:|-:|-:|-:|-:|-:|-:|
|swin2sr_s  4x bf16                 |    1.02|    1.02|1.00x|  0.9849|    1.45 GB|-    |-     |-    |-     | 1,024,824|
|swin2sr_s  3x bf16                 |    0.53|    0.85|1.60x|  1.8918|    1.43 GB|-    |-     |-    |-     | 1,013,463|
|swin2sr_s  2x bf16                 |    1.02|    1.02|1.00x|  0.9791|    1.41 GB|32.85|0.9349|-    |-     | 1,005,348|
|swin2sr_s  1x bf16                 |    1.02|    1.03|1.01x|  0.9796|    1.40 GB|-    |-     |-    |-     | 1,000,479|

### swinir_l 
(1, 3, 480, 640) input, 1 warmup + 5 runs averaged
|Name|FPS|FPS ({abbr}`CL (channels_last)`)|{abbr}`CL (channels_last)` vs base|sec/img|VRAM ({abbr}`CL (channels_last)`)|PSNR (DF2K)|SSIM (DF2K)|PSNR (DIV2K)|SSIM (DIV2K)|Params|
|:-|-:|-:|-:|-:|-:|-:|-:|-:|-:|-:|
|swinir_l  4x bf16                  |    0.46|    0.39|0.84x|  2.1642|    3.56 GB|-    |-     |-    |-     |28,013,059|
|swinir_l  3x bf16                  |    0.32|    0.42|1.32x|  3.1135|    3.52 GB|-    |-     |-    |-     |27,976,131|
|swinir_l  2x bf16                  |    0.47|    0.47|1.01x|  2.1330|    3.52 GB|-    |-     |-    |-     |27,976,131|
|swinir_l  1x bf16                  |    0.47|    0.47|1.01x|  2.1349|    3.52 GB|-    |-     |-    |-     |27,976,131|

### swinir_m 
(1, 3, 480, 640) input, 1 warmup + 5 runs averaged
|Name|FPS|FPS ({abbr}`CL (channels_last)`)|{abbr}`CL (channels_last)` vs base|sec/img|VRAM ({abbr}`CL (channels_last)`)|PSNR (DF2K)|SSIM (DF2K)|PSNR (DIV2K)|SSIM (DIV2K)|Params|
|:-|-:|-:|-:|-:|-:|-:|-:|-:|-:|-:|
|swinir_m  4x bf16                  |    0.62|    0.61|0.99x|  1.6084|    2.64 GB|27.45|0.8254|-    |-     |11,900,199|
|swinir_m  3x bf16                  |    0.43|    0.63|1.44x|  2.2997|    2.62 GB|29.75|0.8826|-    |-     |11,937,127|
|swinir_m  2x bf16                  |    0.70|    0.70|1.00x|  1.4290|    2.60 GB|33.81|0.9427|-    |-     |11,752,487|
|swinir_m  1x bf16                  |    0.70|    0.71|1.01x|  1.4287|    2.59 GB|-    |-     |-    |-     |11,604,775|

### swinir_s 
(1, 3, 480, 640) input, 1 warmup + 5 runs averaged
|Name|FPS|FPS ({abbr}`CL (channels_last)`)|{abbr}`CL (channels_last)` vs base|sec/img|VRAM ({abbr}`CL (channels_last)`)|PSNR (DF2K)|SSIM (DF2K)|PSNR (DIV2K)|SSIM (DIV2K)|Params|
|:-|-:|-:|-:|-:|-:|-:|-:|-:|-:|-:|
|swinir_s  4x bf16                  |    0.99|    0.98|0.99x|  1.0130|    1.55 GB|26.47|0.7980|-    |-     |   929,628|
|swinir_s  3x bf16                  |    0.84|    1.00|1.19x|  1.1910|    1.53 GB|28.66|0.8624|-    |-     |   918,267|
|swinir_s  2x bf16                  |    1.06|    1.07|1.02x|  0.9470|    1.51 GB|32.76|0.9340|-    |-     |   910,152|
|swinir_s  1x bf16                  |    1.06|    1.07|1.01x|  0.9424|    1.50 GB|-    |-     |-    |-     |   905,283|

### tscunet 
(1, 3, 480, 640) input, 1 warmup + 5 runs averaged
|Name|FPS|FPS ({abbr}`CL (channels_last)`)|{abbr}`CL (channels_last)` vs base|sec/img|VRAM ({abbr}`CL (channels_last)`)|PSNR (DF2K)|SSIM (DF2K)|PSNR (DIV2K)|SSIM (DIV2K)|Params|
|:-|-:|-:|-:|-:|-:|-:|-:|-:|-:|-:|
|tscunet  4x fp16                   |-       |-       |-       |-       |-       |-|-|-||-         |
|tscunet  3x fp16                   |-       |-       |-       |-       |-       |-|-|-||-         |
|tscunet  2x fp16                   |-       |-       |-       |-       |-       |-|-|-||-         |
|tscunet  1x fp16                   |-       |-       |-       |-       |-       |-|-|-||-         |

### ultracompact 
(1, 3, 480, 640) input, 1 warmup + 5 runs averaged
|Name|FPS|FPS ({abbr}`CL (channels_last)`)|{abbr}`CL (channels_last)` vs base|sec/img|VRAM ({abbr}`CL (channels_last)`)|PSNR (DF2K)|SSIM (DF2K)|PSNR (DIV2K)|SSIM (DIV2K)|Params|
|:-|-:|-:|-:|-:|-:|-:|-:|-:|-:|-:|
|ultracompact  4x fp16              |  246.21|  359.57|1.46x|  0.0041|    0.18 GB|-    |-     |-    |-     |   325,488|
|ultracompact  3x fp16              |  241.36|   35.52|0.15x|  0.0041|    0.11 GB|-    |-     |-    |-     |   313,371|
|ultracompact  2x fp16              |  265.34|  427.63|1.61x|  0.0038|    0.09 GB|-    |-     |-    |-     |   304,716|
|ultracompact  1x fp16              |  270.26|  435.08|1.61x|  0.0037|    0.09 GB|-    |-     |-    |-     |   299,523|
