# PyTorch Inference Benchmarks by Architecture (AMP & channels last)

All benchmarks were generated using [benchmark_archs.py](https://github.com/the-database/traiNNer-redux/blob/master/scripts/benchmarking/benchmark_archs.py). The benchmarks were done on a Windows 11 PC with RTX 4090 + i9-13000K.

Note that these benchmarks only measure the raw inference step of these architectures. In practice, several other factors may contribute to results not matching the benchmarks shown here. For example, when comparing two architectures with the same inference speed but one has double the VRAM usage, the one with less VRAM usage will be faster to upscale with for larger images, because the one with higher VRAM usage would require tiling to avoid running out of VRAM in order to upscale a large image while the one with lower VRAM usage could upscale the entire image at once without tiling.

PSNR and SSIM scores are a rough measure of quality, higher is better. These scores should not be taken as an absolute that one architecture is better than another. Metrics are calculated using the officially released models optimized on L1 loss, and are trained on either the DF2K or DIV2K training dataset. When comparing scores between architectures, only compare within the same dataset, so only compare DF2K scores with DF2K scores or DIV2K scores with DIV2K scores. DF2K scores are typically higher than DIV2K scores on the same architecture. PSNR and SSIM are calculated on the Y channel of the Urban100 validation dataset, one of the standard research validation sets.
## By Scale

### 4x scale
(1, 3, 480, 640) input, 1 warmup + 5 runs averaged
|Name|FPS|FPS ({abbr}`CL (channels_last)`)|{abbr}`CL (channels_last)` vs base|sec/img|VRAM ({abbr}`CL (channels_last)`)|PSNR (DF2K)|SSIM (DF2K)|PSNR (DIV2K)|SSIM (DIV2K)|Params|
|:-|-:|-:|-:|-:|-:|-:|-:|-:|-:|-:|
|rgt  4x bf16                       |-       |-       |-       |-       |-       |-|-|-||-         |
|rgt_s  4x bf16                     |-       |-       |-       |-       |-       |-|-|-||-         |
|tscunet  4x fp16                   |-       |-       |-       |-       |-       |-|-|-||-         |
|superultracompact  4x fp16         |  736.22|  854.77|1.16x|  0.0014|    0.18 GB|-    |-     |-    |-     |    52,968|
|rtmosr_s  4x fp16                  |  370.62|  344.31|0.93x|  0.0027|    0.30 GB|-    |-     |-    |-     |   460,294|
|rtmosr  4x fp16                    |  369.63|  345.34|0.93x|  0.0027|    0.30 GB|-    |-     |-    |-     |   460,294|
|ultracompact  4x fp16              |  247.53|  362.11|1.46x|  0.0040|    0.18 GB|-    |-     |-    |-     |   325,488|
|compact  4x fp16                   |  135.60|  208.25|1.54x|  0.0074|    0.18 GB|-    |-     |-    |-     |   621,424|
|spanplus_sts  4x fp16              |  159.16|  170.84|1.07x|  0.0063|    0.23 GB|-    |-     |-    |-     |   693,160|
|artcnn_r8f64  4x fp16              |   90.59|  141.86|1.57x|  0.0110|    0.23 GB|-    |-     |-    |-     |   952,688|
|span  4x fp16                      |  101.93|  112.30|1.10x|  0.0098|    0.60 GB|26.18|0.7879|-    |-     | 2,236,728|
|spanplus_st  4x fp16               |  100.06|  112.20|1.12x|  0.0100|    0.32 GB|-    |-     |-    |-     | 2,236,728|
|spanplus_s  4x fp16                |   73.33|   67.22|0.92x|  0.0136|    1.55 GB|-    |-     |-    |-     |   687,707|
|mosr_t  4x fp16                    |   53.89|   64.80|1.20x|  0.0186|    0.36 GB|-    |-     |-    |-     |   609,888|
|realcugan  4x fp16                 |   44.13|   61.32|1.39x|  0.0227|    0.95 GB|-    |-     |-    |-     | 1,406,812|
|spanplus  4x fp16                  |   54.79|   50.94|0.93x|  0.0182|    2.15 GB|-    |-     |-    |-     | 2,228,507|
|plksr_tiny  4x fp16                |   36.91|   49.47|1.34x|  0.0271|    0.27 GB|26.34|0.7942|26.12|0.7888| 2,370,544|
|omnisr  4x fp16                    |   37.78|   37.50|0.99x|  0.0265|    1.14 GB|26.95|0.8105|26.64|0.8018|   214,208|
|man_tiny  4x fp16                  |   34.75|   27.41|0.79x|  0.0288|    0.43 GB|25.84|0.7786|-    |-     |   150,024|
|artcnn_r16f96  4x fp16             |   17.49|   30.38|1.74x|  0.0572|    0.34 GB|-    |-     |-    |-     | 4,113,168|
|lmlt_tiny  4x fp16                 |   28.42|   28.67|1.01x|  0.0352|    0.37 GB|26.08|0.7838|-    |-     |   251,040|
|plksr  4x fp16                     |   11.97|   15.68|1.31x|  0.0835|    0.29 GB|26.85|0.8097|26.69|0.8054| 7,386,096|
|lmlt_base  4x fp16                 |   15.12|   15.01|0.99x|  0.0661|    0.63 GB|26.44|0.7949|-    |-     |   671,808|
|esrgan_lite  4x fp16               |    8.57|   12.13|1.42x|  0.1167|    1.24 GB|-    |-     |-    |-     | 5,021,155|
|mosr  4x fp16                      |    9.27|   11.42|1.23x|  0.1078|    0.49 GB|-    |-     |-    |-     | 4,287,600|
|scunet_aaf6aa  4x fp16             |    8.52|   10.91|1.28x|  0.1174|    3.37 GB|-    |-     |-    |-     |15,207,468|
|lmlt_large  4x fp16                |   10.36|   10.29|0.99x|  0.0965|    0.87 GB|26.63|0.8001|-    |-     | 1,295,328|
|realplksr pixelshuffle layer_norm=True 4x fp16|    8.42|   10.02|1.19x|  0.1187|    0.37 GB|26.94|0.8140|-    |-     | 7,389,680|
|realplksr dysample layer_norm=True 4x fp16|    7.88|    9.09|1.15x|  0.1270|    2.21 GB|-    |-     |-    |-     | 7,402,243|
|eimn_a  4x fp16                    |    8.39|    6.10|0.73x|  0.1193|    0.89 GB|26.68|0.8027|-    |-     |   880,870|
|eimn_l  4x fp16                    |    7.32|    5.33|0.73x|  0.1366|    0.89 GB|26.88|0.8084|-    |-     | 1,002,496|
|rcan  4x fp16                      |    5.35|    7.04|1.32x|  0.1868|    1.40 GB|27.16|0.8168|26.82|0.8087|15,592,379|
|metaflexnet  4x fp16               |    6.61|    6.75|1.02x|  0.1514|    1.80 GB|-    |-     |-    |-     |67,205,424|
|realplksr pixelshuffle layer_norm=False 4x fp16|    5.83|    6.11|1.05x|  0.1715|    0.44 GB|-    |-     |-    |-     | 7,389,680|
|man_light  4x fp16                 |    6.01|    4.42|0.73x|  0.1663|    0.53 GB|26.70|0.8052|-    |-     |   842,892|
|realplksr dysample layer_norm=False 4x fp16|    5.54|    5.76|1.04x|  0.1804|    2.21 GB|-    |-     |-    |-     | 7,402,243|
|hit_sir  4x bf16                   |    5.28|    5.13|0.97x|  0.1893|    1.34 GB|-    |-     |26.71|0.8045|   791,540|
|esrgan use_pixel_unshuffle=False 4x fp16|    3.24|    4.52|1.40x|  0.3088|    2.48 GB|-    |-     |-    |-     |16,697,987|
|esrgan use_pixel_unshuffle=True 4x fp16|    3.15|    4.45|1.41x|  0.3170|    2.48 GB|27.03|0.8153|26.73|0.8072|16,697,987|
|hit_srf  4x bf16                   |    4.24|    4.20|0.99x|  0.2359|    1.34 GB|-    |-     |26.80|0.8069|   866,420|
|hit_sng  4x bf16                   |    4.11|    3.93|0.96x|  0.2435|    1.34 GB|-    |-     |26.75|0.8053| 1,032,060|
|moesr2  4x fp16                    |    3.93|    3.69|0.94x|  0.2543|    0.91 GB|27.05|0.8177|-    |-     |16,547,008|
|flexnet  4x fp16                   |    3.45|    3.61|1.05x|  0.2902|    1.31 GB|-    |-     |-    |-     | 3,045,136|
|man  4x fp16                       |    1.26|    0.86|0.69x|  0.7929|    1.53 GB|27.26|0.8197|-    |-     | 8,712,612|
|swinir_s  4x bf16                  |    1.06|    1.05|0.99x|  0.9405|    1.55 GB|26.47|0.7980|-    |-     |   929,628|
|atd_light  4x bf16                 |    1.01|    1.02|1.01x|  0.9917|    4.48 GB|26.97|0.8107|-    |-     |   814,920|
|srformer_light  4x bf16            |    1.01|    1.01|1.00x|  0.9933|    2.36 GB|26.67|0.8032|-    |-     | 6,894,948|
|swin2sr_s  4x bf16                 |    0.98|    0.99|1.00x|  1.0172|    1.45 GB|-    |-     |-    |-     | 1,024,824|
|swinir_m  4x bf16                  |    0.70|    0.69|0.98x|  1.4329|    2.64 GB|27.45|0.8254|-    |-     |11,900,199|
|hat_s  4x bf16                     |    0.62|    0.66|1.06x|  1.6021|    9.85 GB|27.87|0.8346|-    |-     | 9,621,183|
|dat_light  4x bf16                 |    0.62|    0.60|0.96x|  1.6046|    2.72 GB|26.64|0.8033|-    |-     |   878,577|
|swin2sr_m  4x bf16                 |    0.59|    0.59|0.99x|  1.6817|    2.51 GB|27.51|0.8271|-    |-     |12,239,283|
|hat_m  4x bf16                     |    0.54|    0.58|1.07x|  1.8418|   10.30 GB|27.97|0.8368|-    |-     |20,772,507|
|dat_s  4x bf16                     |    0.56|    0.54|0.97x|  1.7947|    2.85 GB|27.68|0.8300|-    |-     |11,212,131|
|swinir_l  4x bf16                  |    0.45|    0.46|1.02x|  2.2096|    3.56 GB|-    |-     |-    |-     |28,013,059|
|swin2sr_l  4x bf16                 |    0.39|    0.40|1.02x|  2.5339|    3.37 GB|-    |-     |-    |-     |28,785,859|
|atd  4x bf16                       |    0.35|    0.37|1.07x|  2.8767|    6.42 GB|28.22|0.8414|-    |-     |20,260,929|
|hat_l  4x bf16                     |    0.29|    0.31|1.07x|  3.4795|   10.41 GB|28.60|0.8498|-    |-     |40,846,575|
|srformer  4x bf16                  |    0.28|    0.28|1.02x|  3.6103|    3.65 GB|27.68|0.8311|-    |-     |10,543,503|
|dat_2  4x bf16                     |    0.28|    0.28|0.99x|  3.5597|    3.92 GB|27.86|0.8341|-    |-     |11,212,131|
|dat  4x bf16                       |    0.27|    0.28|1.01x|  3.6400|    3.94 GB|27.87|0.8343|-    |-     |14,802,051|
|drct  4x bf16                      |    0.18|    0.18|0.99x|  5.6127|    6.28 GB|28.06|0.8378|-    |-     |14,139,579|
|drct_l  4x bf16                    |    0.08|    0.09|1.04x| 12.0505|    6.42 GB|28.70|0.8508|-    |-     |27,580,719|
|drct_xl  4x bf16                   |    0.08|    0.08|1.00x| 13.0764|    6.46 GB|-    |-     |-    |-     |32,061,099|

### 3x scale
(1, 3, 480, 640) input, 1 warmup + 5 runs averaged
|Name|FPS|FPS ({abbr}`CL (channels_last)`)|{abbr}`CL (channels_last)` vs base|sec/img|VRAM ({abbr}`CL (channels_last)`)|PSNR (DF2K)|SSIM (DF2K)|PSNR (DIV2K)|SSIM (DIV2K)|Params|
|:-|-:|-:|-:|-:|-:|-:|-:|-:|-:|-:|
|rgt  3x bf16                       |-       |-       |-       |-       |-       |-|-|-||-         |
|rgt_s  3x bf16                     |-       |-       |-       |-       |-       |-|-|-||-         |
|swin2sr_l  3x bf16                 |-       |-       |-       |-       |-       |-|-|-||-         |
|rtmosr_s  3x fp16                  |-       |-       |-       |-       |-       |-|-|-||-         |
|tscunet  3x fp16                   |-       |-       |-       |-       |-       |-|-|-||-         |
|superultracompact  3x fp16         |  841.56| 1090.61|1.30x|  0.0012|    0.11 GB|-    |-     |-    |-     |    48,411|
|rtmosr  3x fp16                    |  402.14|  405.48|1.01x|  0.0025|    0.27 GB|-    |-     |-    |-     |   454,225|
|ultracompact  3x fp16              |  262.17|  397.15|1.51x|  0.0038|    0.11 GB|-    |-     |-    |-     |   313,371|
|compact  3x fp16                   |  137.73|  215.70|1.57x|  0.0073|    0.11 GB|-    |-     |-    |-     |   609,307|
|spanplus_sts  3x fp16              |  158.59|  164.46|1.04x|  0.0063|    0.21 GB|-    |-     |-    |-     |   687,091|
|artcnn_r8f64  3x fp16              |   91.43|  144.61|1.58x|  0.0109|    0.22 GB|-    |-     |-    |-     |   940,571|
|spanplus_st  3x fp16               |  101.12|  111.51|1.10x|  0.0099|    0.31 GB|-    |-     |-    |-     | 2,227,635|
|span  3x fp16                      |   98.51|  109.16|1.11x|  0.0102|    0.59 GB|-    |-     |-    |-     | 2,227,635|
|spanplus_s  3x fp16                |   89.45|   86.85|0.97x|  0.0112|    0.89 GB|-    |-     |-    |-     |   684,067|
|spanplus  3x fp16                  |   64.88|   64.24|0.99x|  0.0154|    1.23 GB|-    |-     |-    |-     | 2,223,075|
|mosr_t  3x fp16                    |   53.71|   64.79|1.21x|  0.0186|    0.35 GB|-    |-     |-    |-     |   600,795|
|plksr_tiny  3x fp16                |   37.39|   50.13|1.34x|  0.0267|    0.24 GB|28.51|0.8599|28.35|0.8571| 2,358,427|
|omnisr  3x fp16                    |   37.90|   37.29|0.98x|  0.0264|    1.13 GB|29.12|0.8712|28.84|0.8656|   202,091|
|man_tiny  3x fp16                  |   34.69|   27.60|0.80x|  0.0288|    0.41 GB|-    |-     |-    |-     |   140,931|
|realcugan  3x fp16                 |   25.16|   34.40|1.37x|  0.0398|    1.71 GB|-    |-     |-    |-     | 1,286,326|
|artcnn_r16f96  3x fp16             |   17.14|   30.52|1.78x|  0.0584|    0.32 GB|-    |-     |-    |-     | 4,095,003|
|lmlt_tiny  3x fp16                 |   28.64|   28.36|0.99x|  0.0349|    0.36 GB|28.10|0.8503|-    |-     |   244,215|
|plksr  3x fp16                     |   12.01|   15.86|1.32x|  0.0832|    0.27 GB|29.10|0.8713|28.86|0.8666| 7,373,979|
|lmlt_base  3x fp16                 |   15.11|   15.08|1.00x|  0.0662|    0.62 GB|28.48|0.8581|-    |-     |   660,447|
|scunet_aaf6aa  3x fp16             |   11.52|   14.15|1.23x|  0.0868|    1.02 GB|-    |-     |-    |-     |15,170,540|
|esrgan_lite  3x fp16               |    9.32|   13.01|1.40x|  0.1073|    0.71 GB|-    |-     |-    |-     | 5,011,907|
|mosr  3x fp16                      |    9.28|   11.48|1.24x|  0.1078|    0.47 GB|-    |-     |-    |-     | 4,275,483|
|lmlt_large  3x fp16                |   10.37|   10.33|1.00x|  0.0965|    0.86 GB|28.72|0.8628|-    |-     | 1,279,431|
|realplksr pixelshuffle layer_norm=True 3x fp16|    8.46|   10.11|1.20x|  0.1182|    0.35 GB|-    |-     |-    |-     | 7,377,563|
|realplksr dysample layer_norm=True 3x fp16|    8.23|    9.70|1.18x|  0.1215|    0.79 GB|-    |-     |-    |-     | 7,380,617|
|eimn_a  3x fp16                    |    8.38|    6.11|0.73x|  0.1193|    0.87 GB|28.87|0.8660|-    |-     |   868,753|
|rcan  3x fp16                      |    5.57|    7.59|1.36x|  0.1795|    0.86 GB|-    |-     |29.09|0.8702|15,629,307|
|eimn_l  3x fp16                    |    7.33|    5.35|0.73x|  0.1364|    0.87 GB|29.05|0.8698|-    |-     |   990,379|
|metaflexnet  3x fp16               |    6.61|    6.75|1.02x|  0.1513|    1.79 GB|-    |-     |-    |-     |67,181,211|
|realplksr pixelshuffle layer_norm=False 3x fp16|    5.84|    6.13|1.05x|  0.1713|    0.42 GB|-    |-     |-    |-     | 7,377,563|
|man_light  3x fp16                 |    6.02|    4.41|0.73x|  0.1662|    0.52 GB|-    |-     |-    |-     |   831,531|
|realplksr dysample layer_norm=False 3x fp16|    5.72|    6.00|1.05x|  0.1749|    0.79 GB|-    |-     |-    |-     | 7,380,617|
|hit_sir  3x bf16                   |    5.29|    5.13|0.97x|  0.1891|    1.32 GB|-    |-     |28.93|0.8673|   780,179|
|esrgan use_pixel_unshuffle=False 3x fp16|    3.37|    4.85|1.44x|  0.2964|    1.44 GB|-    |-     |-    |-     |16,661,059|
|esrgan use_pixel_unshuffle=True 3x fp16|    3.27|    4.66|1.43x|  0.3060|    1.44 GB|-    |-     |-    |-     |16,661,059|
|hit_srf  3x bf16                   |    4.24|    4.20|0.99x|  0.2358|    1.32 GB|-    |-     |28.99|0.8687|   855,059|
|hit_sng  3x bf16                   |    4.11|    3.94|0.96x|  0.2431|    1.32 GB|-    |-     |28.91|0.8671| 1,020,699|
|moesr2  3x fp16                    |    3.94|    3.69|0.94x|  0.2539|    0.90 GB|-    |-     |-    |-     |16,534,891|
|flexnet  3x fp16                   |    3.45|    3.62|1.05x|  0.2895|    1.29 GB|-    |-     |-    |-     | 3,020,923|
|man  3x fp16                       |    1.26|    0.87|0.69x|  0.7920|    1.52 GB|29.52|0.8782|-    |-     | 8,678,571|
|swinir_s  3x bf16                  |    1.04|    1.06|1.01x|  0.9572|    1.53 GB|28.66|0.8624|-    |-     |   918,267|
|atd_light  3x bf16                 |    1.01|    1.03|1.01x|  0.9860|    4.45 GB|29.17|0.8709|-    |-     |   805,827|
|srformer_light  3x bf16            |    1.00|    1.02|1.02x|  1.0033|    2.34 GB|28.81|0.8655|-    |-     | 6,860,907|
|swin2sr_s  3x bf16                 |    1.00|    0.98|0.98x|  0.9963|    1.43 GB|-    |-     |-    |-     | 1,013,463|
|swinir_m  3x bf16                  |    0.69|    0.69|1.00x|  1.4557|    2.62 GB|29.75|0.8826|-    |-     |11,937,127|
|hat_s  3x bf16                     |    0.63|    0.67|1.06x|  1.5760|    9.83 GB|30.15|0.8879|-    |-     | 9,658,111|
|dat_light  3x bf16                 |    0.62|    0.63|1.02x|  1.6246|    2.69 GB|28.89|0.8666|-    |-     |   915,505|
|swin2sr_m  3x bf16                 |    0.61|    0.61|1.00x|  1.6432|    2.49 GB|-    |-     |-    |-     |12,276,211|
|hat_m  3x bf16                     |    0.55|    0.59|1.07x|  1.8147|   10.27 GB|30.23|0.8896|-    |-     |20,809,435|
|dat_s  3x bf16                     |    0.55|    0.55|1.00x|  1.8229|    2.82 GB|29.98|0.8846|-    |-     |11,249,059|
|swinir_l  3x bf16                  |    0.46|    0.47|1.02x|  2.1709|    3.52 GB|-    |-     |-    |-     |27,976,131|
|atd  3x bf16                       |    0.35|    0.37|1.06x|  2.8521|    6.40 GB|30.52|0.8924|-    |-     |20,297,857|
|hat_l  3x bf16                     |    0.29|    0.31|1.08x|  3.4702|   10.39 GB|30.92|0.8981|-    |-     |40,883,503|
|dat_2  3x bf16                     |    0.28|    0.28|1.00x|  3.5149|    3.90 GB|30.13|0.8878|-    |-     |11,249,059|
|dat  3x bf16                       |    0.28|    0.27|0.97x|  3.5417|    3.92 GB|30.18|0.8886|-    |-     |14,838,979|
|srformer  3x bf16                  |    0.27|    0.28|1.01x|  3.6462|    3.62 GB|30.04|0.8865|-    |-     |10,580,431|
|drct  3x bf16                      |    0.18|    0.18|0.99x|  5.5078|    6.25 GB|30.34|0.8910|-    |-     |14,176,507|
|drct_l  3x bf16                    |    0.07|    0.05|0.78x| 14.8329|    6.39 GB|31.14|0.9004|-    |-     |27,617,647|
|drct_xl  3x bf16                   |    0.04|    0.05|1.22x| 22.6431|    6.44 GB|-    |-     |-    |-     |32,098,027|

### 2x scale
(1, 3, 480, 640) input, 1 warmup + 5 runs averaged
|Name|FPS|FPS ({abbr}`CL (channels_last)`)|{abbr}`CL (channels_last)` vs base|sec/img|VRAM ({abbr}`CL (channels_last)`)|PSNR (DF2K)|SSIM (DF2K)|PSNR (DIV2K)|SSIM (DIV2K)|Params|
|:-|-:|-:|-:|-:|-:|-:|-:|-:|-:|-:|
|rgt  2x bf16                       |-       |-       |-       |-       |-       |-|-|-||-         |
|rgt_s  2x bf16                     |-       |-       |-       |-       |-       |-|-|-||-         |
|swin2sr_l  2x bf16                 |-       |-       |-       |-       |-       |-|-|-||-         |
|tscunet  2x fp16                   |-       |-       |-       |-       |-       |-|-|-||-         |
|superultracompact  2x fp16         |  915.62| 1292.01|1.41x|  0.0011|    0.06 GB|-    |-     |-    |-     |    45,156|
|rtmosr_s  2x fp16                  | 1087.94|  920.15|0.85x|  0.0009|    0.08 GB|-    |-     |-    |-     |   462,886|
|ultracompact  2x fp16              |  265.03|  429.91|1.62x|  0.0038|    0.09 GB|-    |-     |-    |-     |   304,716|
|rtmosr  2x fp16                    |  423.64|  423.91|1.00x|  0.0024|    0.26 GB|-    |-     |-    |-     |   449,890|
|compact  2x fp16                   |  141.20|  229.95|1.63x|  0.0071|    0.10 GB|-    |-     |-    |-     |   600,652|
|spanplus_sts  2x fp16              |  164.49|  169.55|1.03x|  0.0061|    0.21 GB|-    |-     |-    |-     |   682,756|
|artcnn_r8f64  2x fp16              |   92.30|  146.62|1.59x|  0.0108|    0.21 GB|-    |-     |-    |-     |   931,916|
|spanplus_s  2x fp16                |  115.40|  121.08|1.05x|  0.0087|    0.42 GB|-    |-     |-    |-     |   681,467|
|spanplus_st  2x fp16               |  102.62|  113.23|1.10x|  0.0097|    0.30 GB|-    |-     |-    |-     | 2,221,140|
|span  2x fp16                      |  102.29|  110.88|1.08x|  0.0098|    0.58 GB|32.24|0.9294|-    |-     | 2,221,140|
|spanplus  2x fp16                  |   80.25|   82.59|1.03x|  0.0125|    0.58 GB|-    |-     |-    |-     | 2,219,195|
|realcugan  2x fp16                 |   59.73|   81.21|1.36x|  0.0167|    0.78 GB|-    |-     |-    |-     | 1,284,598|
|mosr_t  2x fp16                    |   53.09|   65.02|1.22x|  0.0188|    0.34 GB|-    |-     |-    |-     |   594,300|
|esrgan_lite  2x fp16               |   43.35|   59.03|1.36x|  0.0231|    0.34 GB|-    |-     |-    |-     | 5,023,747|
|plksr_tiny  2x fp16                |   37.56|   50.83|1.35x|  0.0266|    0.22 GB|32.58|0.9328|32.43|0.9314| 2,349,772|
|omnisr  2x fp16                    |   37.80|   37.81|1.00x|  0.0265|    1.12 GB|33.30|0.9386|33.05|0.9363|   193,436|
|man_tiny  2x fp16                  |   35.11|   27.85|0.79x|  0.0285|    0.40 GB|-    |-     |-    |-     |   134,436|
|artcnn_r16f96  2x fp16             |   17.75|   30.64|1.73x|  0.0563|    0.32 GB|-    |-     |-    |-     | 4,082,028|
|lmlt_tiny  2x fp16                 |   28.66|   28.58|1.00x|  0.0349|    0.35 GB|32.04|0.9273|-    |-     |   239,340|
|esrgan use_pixel_unshuffle=True 2x fp16|   17.88|   22.91|1.28x|  0.0559|    0.70 GB|-    |-     |-    |-     |16,703,171|
|plksr  2x fp16                     |   12.10|   15.96|1.32x|  0.0827|    0.25 GB|33.36|0.9395|32.99|0.9365| 7,365,324|
|lmlt_base  2x fp16                 |   15.11|   15.14|1.00x|  0.0662|    0.61 GB|32.52|0.9316|-    |-     |   652,332|
|scunet_aaf6aa  2x fp16             |   11.60|   14.24|1.23x|  0.0862|    1.03 GB|-    |-     |-    |-     |15,170,540|
|mosr  2x fp16                      |    9.29|   11.47|1.23x|  0.1076|    0.46 GB|-    |-     |-    |-     | 4,266,828|
|lmlt_large  2x fp16                |   10.37|   10.33|1.00x|  0.0964|    0.85 GB|32.75|0.9336|-    |-     | 1,268,076|
|realplksr pixelshuffle layer_norm=True 2x fp16|    8.47|   10.11|1.19x|  0.1181|    0.33 GB|33.44|0.9412|-    |-     | 7,368,908|
|realplksr dysample layer_norm=True 2x fp16|    8.37|    9.96|1.19x|  0.1194|    0.32 GB|-    |-     |-    |-     | 7,369,747|
|eimn_a  2x fp16                    |    8.36|    6.10|0.73x|  0.1197|    0.86 GB|33.15|0.9373|-    |-     |   860,098|
|rcan  2x fp16                      |    5.63|    7.76|1.38x|  0.1775|    0.48 GB|-    |-     |33.34|0.9384|15,444,667|
|eimn_l  2x fp16                    |    7.31|    5.35|0.73x|  0.1368|    0.87 GB|33.23|0.9381|-    |-     |   981,724|
|metaflexnet  2x fp16               |    6.61|    6.77|1.02x|  0.1514|    1.78 GB|-    |-     |-    |-     |67,163,916|
|realplksr pixelshuffle layer_norm=False 2x fp16|    5.85|    6.12|1.05x|  0.1710|    0.40 GB|-    |-     |-    |-     | 7,368,908|
|realplksr dysample layer_norm=False 2x fp16|    5.78|    6.07|1.05x|  0.1729|    0.39 GB|-    |-     |-    |-     | 7,369,747|
|man_light  2x fp16                 |    6.04|    4.41|0.73x|  0.1657|    0.51 GB|-    |-     |-    |-     |   823,416|
|hit_sir  2x bf16                   |    5.27|    5.13|0.97x|  0.1896|    1.30 GB|-    |-     |33.02|0.9365|   772,064|
|esrgan use_pixel_unshuffle=False 2x fp16|    3.39|    4.88|1.44x|  0.2952|    0.70 GB|-    |-     |-    |-     |16,661,059|
|hit_srf  2x bf16                   |    4.24|    4.19|0.99x|  0.2359|    1.30 GB|-    |-     |33.13|0.9372|   846,944|
|hit_sng  2x bf16                   |    4.11|    3.93|0.96x|  0.2435|    1.30 GB|-    |-     |33.01|0.9360| 1,012,584|
|moesr2  2x fp16                    |    3.93|    3.69|0.94x|  0.2544|    0.89 GB|-    |-     |-    |-     |16,526,236|
|flexnet  2x fp16                   |    3.45|    3.61|1.05x|  0.2902|    1.28 GB|-    |-     |-    |-     | 3,003,628|
|man  2x fp16                       |    1.26|    0.86|0.68x|  0.7923|    1.51 GB|33.73|0.9422|-    |-     | 8,654,256|
|swinir_s  2x bf16                  |    1.04|    1.05|1.01x|  0.9658|    1.51 GB|32.76|0.9340|-    |-     |   910,152|
|atd_light  2x bf16                 |    1.02|    1.03|1.01x|  0.9822|    4.44 GB|33.27|0.9375|-    |-     |   799,332|
|srformer_light  2x bf16            |    1.00|    1.02|1.02x|  0.9973|    2.32 GB|32.91|0.9353|-    |-     | 6,836,592|
|swin2sr_s  2x bf16                 |    0.99|    1.02|1.04x|  1.0140|    1.41 GB|32.85|0.9349|-    |-     | 1,005,348|
|swinir_m  2x bf16                  |    0.67|    0.69|1.02x|  1.4863|    2.60 GB|33.81|0.9427|-    |-     |11,752,487|
|hat_s  2x bf16                     |    0.63|    0.67|1.06x|  1.5879|    9.81 GB|34.31|0.9459|-    |-     | 9,473,471|
|dat_light  2x bf16                 |    0.61|    0.62|1.01x|  1.6333|    2.67 GB|32.89|0.9346|-    |-     |   730,865|
|swin2sr_m  2x bf16                 |    0.61|    0.61|1.01x|  1.6375|    2.47 GB|33.89|0.9431|-    |-     |12,091,571|
|hat_m  2x bf16                     |    0.55|    0.58|1.07x|  1.8294|   10.26 GB|34.45|0.9466|-    |-     |20,624,795|
|dat_s  2x bf16                     |    0.56|    0.55|0.99x|  1.7985|    2.80 GB|34.12|0.9444|-    |-     |11,064,419|
|swinir_l  2x bf16                  |    0.46|    0.46|1.00x|  2.1683|    3.52 GB|-    |-     |-    |-     |27,976,131|
|atd  2x bf16                       |    0.35|    0.37|1.06x|  2.8377|    6.38 GB|34.73|0.9476|-    |-     |20,113,217|
|hat_l  2x bf16                     |    0.29|    0.31|1.07x|  3.4752|   10.37 GB|35.09|0.9513|-    |-     |40,698,863|
|srformer  2x bf16                  |    0.28|    0.27|0.97x|  3.5674|    3.61 GB|34.09|0.9449|-    |-     |10,395,791|
|dat  2x bf16                       |    0.27|    0.28|1.01x|  3.6626|    3.90 GB|34.37|0.9458|-    |-     |14,654,339|
|dat_2  2x bf16                     |    0.28|    0.28|1.00x|  3.6257|    3.88 GB|34.31|0.9457|-    |-     |11,064,419|
|drct  2x bf16                      |    0.18|    0.18|1.01x|  5.5716|    6.24 GB|34.54|0.9474|-    |-     |13,991,867|
|drct_l  2x bf16                    |    0.09|    0.09|1.00x| 11.0373|    6.37 GB|35.17|0.9516|-    |-     |27,433,007|
|drct_xl  2x bf16                   |    0.08|    0.08|0.99x| 13.0431|    6.42 GB|-    |-     |-    |-     |31,913,387|

### 1x scale
(1, 3, 480, 640) input, 1 warmup + 5 runs averaged
|Name|FPS|FPS ({abbr}`CL (channels_last)`)|{abbr}`CL (channels_last)` vs base|sec/img|VRAM ({abbr}`CL (channels_last)`)|PSNR (DF2K)|SSIM (DF2K)|PSNR (DIV2K)|SSIM (DIV2K)|Params|
|:-|-:|-:|-:|-:|-:|-:|-:|-:|-:|-:|
|realcugan  1x fp16                 |-       |-       |-       |-       |-       |-|-|-||-         |
|rgt  1x bf16                       |-       |-       |-       |-       |-       |-|-|-||-         |
|rgt_s  1x bf16                     |-       |-       |-       |-       |-       |-|-|-||-         |
|swin2sr_l  1x bf16                 |-       |-       |-       |-       |-       |-|-|-||-         |
|tscunet  1x fp16                   |-       |-       |-       |-       |-       |-|-|-||-         |
|superultracompact  1x fp16         |  954.87| 1388.69|1.45x|  0.0010|    0.04 GB|-    |-     |-    |-     |    43,203|
|rtmosr_s  1x fp16                  | 1249.20|  942.58|0.75x|  0.0008|    0.03 GB|-    |-     |-    |-     |   473,254|
|rtmosr  1x fp16                    |  422.67|  438.16|1.04x|  0.0024|    0.25 GB|-    |-     |-    |-     |   447,289|
|ultracompact  1x fp16              |  266.31|  434.57|1.63x|  0.0038|    0.09 GB|-    |-     |-    |-     |   299,523|
|compact  1x fp16                   |  141.42|  231.73|1.64x|  0.0071|    0.09 GB|-    |-     |-    |-     |   595,459|
|spanplus_sts  1x fp16              |  163.83|  175.44|1.07x|  0.0061|    0.20 GB|-    |-     |-    |-     |   680,155|
|artcnn_r8f64  1x fp16              |   90.70|  146.78|1.62x|  0.0110|    0.20 GB|-    |-     |-    |-     |   926,723|
|spanplus_s  1x fp16                |  140.19|  140.83|1.00x|  0.0071|    0.20 GB|-    |-     |-    |-     |   679,907|
|span  1x fp16                      |  102.00|  112.77|1.11x|  0.0098|    0.57 GB|-    |-     |-    |-     | 2,217,243|
|spanplus_st  1x fp16               |  101.69|  112.33|1.10x|  0.0098|    0.30 GB|-    |-     |-    |-     | 2,217,243|
|esrgan_lite  1x fp16               |   86.02|  107.18|1.25x|  0.0116|    0.12 GB|-    |-     |-    |-     | 5,034,115|
|spanplus  1x fp16                  |   92.06|   98.31|1.07x|  0.0109|    0.30 GB|-    |-     |-    |-     | 2,216,867|
|mosr_t  1x fp16                    |   53.90|   63.14|1.17x|  0.0186|    0.33 GB|-    |-     |-    |-     |   590,403|
|esrgan use_pixel_unshuffle=True 1x fp16|   40.21|   61.54|1.53x|  0.0249|    0.26 GB|-    |-     |-    |-     |16,723,907|
|plksr_tiny  1x fp16                |   37.55|   50.74|1.35x|  0.0266|    0.21 GB|-    |-     |-    |-     | 2,344,579|
|omnisr  1x fp16                    |   38.02|   37.48|0.99x|  0.0263|    1.12 GB|-    |-     |-    |-     |   188,243|
|man_tiny  1x fp16                  |   35.22|   27.47|0.78x|  0.0284|    0.40 GB|-    |-     |-    |-     |   130,539|
|artcnn_r16f96  1x fp16             |   16.54|   30.61|1.85x|  0.0604|    0.31 GB|-    |-     |-    |-     | 4,074,243|
|lmlt_tiny  1x fp16                 |   27.71|   28.67|1.03x|  0.0361|    0.35 GB|-    |-     |-    |-     |   236,415|
|scunet_aaf6aa  1x fp16             |   27.32|   27.77|1.02x|  0.0366|    0.80 GB|-    |-     |-    |-     | 9,699,756|
|plksr  1x fp16                     |   12.05|   15.98|1.33x|  0.0830|    0.24 GB|-    |-     |-    |-     | 7,360,131|
|lmlt_base  1x fp16                 |   14.94|   15.02|1.01x|  0.0669|    0.61 GB|-    |-     |-    |-     |   647,463|
|mosr  1x fp16                      |    9.29|   11.31|1.22x|  0.1076|    0.46 GB|-    |-     |-    |-     | 4,261,635|
|lmlt_large  1x fp16                |   10.34|   10.28|0.99x|  0.0967|    0.85 GB|-    |-     |-    |-     | 1,261,263|
|realplksr pixelshuffle layer_norm=True 1x fp16|    8.47|   10.15|1.20x|  0.1181|    0.32 GB|-    |-     |-    |-     | 7,363,715|
|realplksr dysample layer_norm=True 1x fp16|    8.44|   10.04|1.19x|  0.1184|    0.32 GB|-    |-     |-    |-     | 7,363,757|
|eimn_a  1x fp16                    |    8.38|    6.10|0.73x|  0.1193|    0.86 GB|-    |-     |-    |-     |   854,905|
|rcan  1x fp16                      |    5.72|    7.98|1.40x|  0.1749|    0.28 GB|-    |-     |-    |-     |15,296,955|
|eimn_l  1x fp16                    |    7.34|    5.34|0.73x|  0.1363|    0.86 GB|-    |-     |-    |-     |   976,531|
|metaflexnet  1x fp16               |    6.60|    6.73|1.02x|  0.1516|    1.77 GB|-    |-     |-    |-     |67,153,539|
|realplksr pixelshuffle layer_norm=False 1x fp16|    5.84|    6.15|1.05x|  0.1712|    0.39 GB|-    |-     |-    |-     | 7,363,715|
|realplksr dysample layer_norm=False 1x fp16|    5.82|    6.11|1.05x|  0.1718|    0.39 GB|-    |-     |-    |-     | 7,363,757|
|man_light  1x fp16                 |    6.02|    4.43|0.74x|  0.1661|    0.50 GB|-    |-     |-    |-     |   818,547|
|hit_sir  1x bf16                   |    5.27|    5.12|0.97x|  0.1897|    1.29 GB|-    |-     |-    |-     |   767,195|
|esrgan use_pixel_unshuffle=False 1x fp16|    3.50|    5.00|1.43x|  0.2856|    0.44 GB|-    |-     |-    |-     |16,624,131|
|hit_srf  1x bf16                   |    4.24|    4.19|0.99x|  0.2359|    1.29 GB|-    |-     |-    |-     |   842,075|
|hit_sng  1x bf16                   |    4.10|    3.92|0.95x|  0.2438|    1.29 GB|-    |-     |-    |-     | 1,007,715|
|moesr2  1x fp16                    |    3.94|    3.70|0.94x|  0.2540|    0.89 GB|-    |-     |-    |-     |16,521,043|
|flexnet  1x fp16                   |    3.45|    3.61|1.05x|  0.2901|    1.28 GB|-    |-     |-    |-     | 2,993,251|
|man  1x fp16                       |    1.26|    0.87|0.69x|  0.7921|    1.51 GB|-    |-     |-    |-     | 8,639,667|
|srformer_light  1x bf16            |    1.02|    1.03|1.01x|  0.9822|    2.31 GB|-    |-     |-    |-     | 6,822,003|
|atd_light  1x bf16                 |    1.02|    1.02|1.01x|  0.9848|    4.42 GB|-    |-     |-    |-     |   795,435|
|swin2sr_s  1x bf16                 |    0.99|    0.99|0.99x|  1.0054|    1.40 GB|-    |-     |-    |-     | 1,000,479|
|swinir_s  1x bf16                  |    0.91|    0.91|1.01x|  1.1039|    1.50 GB|-    |-     |-    |-     |   905,283|
|hat_s  1x bf16                     |    0.64|    0.67|1.06x|  1.5748|    9.80 GB|-    |-     |-    |-     | 9,325,759|
|swinir_m  1x bf16                  |    0.66|    0.62|0.95x|  1.5257|    2.59 GB|-    |-     |-    |-     |11,604,775|
|dat_light  1x bf16                 |    0.61|    0.63|1.02x|  1.6327|    2.66 GB|-    |-     |-    |-     |   583,153|
|swin2sr_m  1x bf16                 |    0.62|    0.61|0.98x|  1.6136|    2.46 GB|-    |-     |-    |-     |11,943,859|
|hat_m  1x bf16                     |    0.55|    0.59|1.07x|  1.8091|   10.24 GB|-    |-     |-    |-     |20,477,083|
|dat_s  1x bf16                     |    0.55|    0.54|0.98x|  1.8039|    2.79 GB|-    |-     |-    |-     |10,916,707|
|swinir_l  1x bf16                  |    0.46|    0.47|1.01x|  2.1560|    3.52 GB|-    |-     |-    |-     |27,976,131|
|atd  1x bf16                       |    0.35|    0.37|1.06x|  2.8402|    6.37 GB|-    |-     |-    |-     |19,965,505|
|hat_l  1x bf16                     |    0.29|    0.31|1.08x|  3.4572|   10.36 GB|-    |-     |-    |-     |40,551,151|
|dat_2  1x bf16                     |    0.28|    0.28|1.00x|  3.5622|    3.87 GB|-    |-     |-    |-     |10,916,707|
|srformer  1x bf16                  |    0.28|    0.28|1.01x|  3.5898|    3.60 GB|-    |-     |-    |-     |10,248,079|
|dat  1x bf16                       |    0.28|    0.28|1.01x|  3.6036|    3.89 GB|-    |-     |-    |-     |14,506,627|
|drct  1x bf16                      |    0.18|    0.18|1.01x|  5.6035|    6.23 GB|-    |-     |-    |-     |13,844,155|
|drct_l  1x bf16                    |    0.09|    0.09|1.00x| 11.1585|    6.36 GB|-    |-     |-    |-     |27,285,295|
|drct_xl  1x bf16                   |    0.08|    0.08|0.98x| 12.8536|    6.41 GB|-    |-     |-    |-     |31,765,675|

## By Architecture

### artcnn_r16f96
(1, 3, 480, 640) input, 1 warmup + 5 runs averaged
|Name|FPS|FPS ({abbr}`CL (channels_last)`)|{abbr}`CL (channels_last)` vs base|sec/img|VRAM ({abbr}`CL (channels_last)`)|PSNR (DF2K)|SSIM (DF2K)|PSNR (DIV2K)|SSIM (DIV2K)|Params|
|:-|-:|-:|-:|-:|-:|-:|-:|-:|-:|-:|
|artcnn_r16f96  4x fp16             |   17.49|   30.38|1.74x|  0.0572|    0.34 GB|-    |-     |-    |-     | 4,113,168|
|artcnn_r16f96  3x fp16             |   17.14|   30.52|1.78x|  0.0584|    0.32 GB|-    |-     |-    |-     | 4,095,003|
|artcnn_r16f96  2x fp16             |   17.75|   30.64|1.73x|  0.0563|    0.32 GB|-    |-     |-    |-     | 4,082,028|
|artcnn_r16f96  1x fp16             |   16.54|   30.61|1.85x|  0.0604|    0.31 GB|-    |-     |-    |-     | 4,074,243|

### artcnn_r8f64
(1, 3, 480, 640) input, 1 warmup + 5 runs averaged
|Name|FPS|FPS ({abbr}`CL (channels_last)`)|{abbr}`CL (channels_last)` vs base|sec/img|VRAM ({abbr}`CL (channels_last)`)|PSNR (DF2K)|SSIM (DF2K)|PSNR (DIV2K)|SSIM (DIV2K)|Params|
|:-|-:|-:|-:|-:|-:|-:|-:|-:|-:|-:|
|artcnn_r8f64  4x fp16              |   90.59|  141.86|1.57x|  0.0110|    0.23 GB|-    |-     |-    |-     |   952,688|
|artcnn_r8f64  3x fp16              |   91.43|  144.61|1.58x|  0.0109|    0.22 GB|-    |-     |-    |-     |   940,571|
|artcnn_r8f64  2x fp16              |   92.30|  146.62|1.59x|  0.0108|    0.21 GB|-    |-     |-    |-     |   931,916|
|artcnn_r8f64  1x fp16              |   90.70|  146.78|1.62x|  0.0110|    0.20 GB|-    |-     |-    |-     |   926,723|

### atd
(1, 3, 480, 640) input, 1 warmup + 5 runs averaged
|Name|FPS|FPS ({abbr}`CL (channels_last)`)|{abbr}`CL (channels_last)` vs base|sec/img|VRAM ({abbr}`CL (channels_last)`)|PSNR (DF2K)|SSIM (DF2K)|PSNR (DIV2K)|SSIM (DIV2K)|Params|
|:-|-:|-:|-:|-:|-:|-:|-:|-:|-:|-:|
|atd  4x bf16                       |    0.35|    0.37|1.07x|  2.8767|    6.42 GB|28.22|0.8414|-    |-     |20,260,929|
|atd  3x bf16                       |    0.35|    0.37|1.06x|  2.8521|    6.40 GB|30.52|0.8924|-    |-     |20,297,857|
|atd  2x bf16                       |    0.35|    0.37|1.06x|  2.8377|    6.38 GB|34.73|0.9476|-    |-     |20,113,217|
|atd  1x bf16                       |    0.35|    0.37|1.06x|  2.8402|    6.37 GB|-    |-     |-    |-     |19,965,505|

### atd_light
(1, 3, 480, 640) input, 1 warmup + 5 runs averaged
|Name|FPS|FPS ({abbr}`CL (channels_last)`)|{abbr}`CL (channels_last)` vs base|sec/img|VRAM ({abbr}`CL (channels_last)`)|PSNR (DF2K)|SSIM (DF2K)|PSNR (DIV2K)|SSIM (DIV2K)|Params|
|:-|-:|-:|-:|-:|-:|-:|-:|-:|-:|-:|
|atd_light  4x bf16                 |    1.01|    1.02|1.01x|  0.9917|    4.48 GB|26.97|0.8107|-    |-     |   814,920|
|atd_light  3x bf16                 |    1.01|    1.03|1.01x|  0.9860|    4.45 GB|29.17|0.8709|-    |-     |   805,827|
|atd_light  2x bf16                 |    1.02|    1.03|1.01x|  0.9822|    4.44 GB|33.27|0.9375|-    |-     |   799,332|
|atd_light  1x bf16                 |    1.02|    1.02|1.01x|  0.9848|    4.42 GB|-    |-     |-    |-     |   795,435|

### compact
(1, 3, 480, 640) input, 1 warmup + 250 runs averaged
|Name|FPS|FPS ({abbr}`CL (channels_last)`)|{abbr}`CL (channels_last)` vs base|sec/img|VRAM ({abbr}`CL (channels_last)`)|PSNR (DF2K)|SSIM (DF2K)|PSNR (DIV2K)|SSIM (DIV2K)|Params|
|:-|-:|-:|-:|-:|-:|-:|-:|-:|-:|-:|
|compact  4x fp16                   |  135.60|  208.25|1.54x|  0.0074|    0.18 GB|-    |-     |-    |-     |   621,424|
|compact  3x fp16                   |  137.73|  215.70|1.57x|  0.0073|    0.11 GB|-    |-     |-    |-     |   609,307|
|compact  2x fp16                   |  141.20|  229.95|1.63x|  0.0071|    0.10 GB|-    |-     |-    |-     |   600,652|
|compact  1x fp16                   |  141.42|  231.73|1.64x|  0.0071|    0.09 GB|-    |-     |-    |-     |   595,459|

### dat
(1, 3, 480, 640) input, 1 warmup + 5 runs averaged
|Name|FPS|FPS ({abbr}`CL (channels_last)`)|{abbr}`CL (channels_last)` vs base|sec/img|VRAM ({abbr}`CL (channels_last)`)|PSNR (DF2K)|SSIM (DF2K)|PSNR (DIV2K)|SSIM (DIV2K)|Params|
|:-|-:|-:|-:|-:|-:|-:|-:|-:|-:|-:|
|dat  4x bf16                       |    0.27|    0.28|1.01x|  3.6400|    3.94 GB|27.87|0.8343|-    |-     |14,802,051|
|dat  3x bf16                       |    0.28|    0.27|0.97x|  3.5417|    3.92 GB|30.18|0.8886|-    |-     |14,838,979|
|dat  2x bf16                       |    0.27|    0.28|1.01x|  3.6626|    3.90 GB|34.37|0.9458|-    |-     |14,654,339|
|dat  1x bf16                       |    0.28|    0.28|1.01x|  3.6036|    3.89 GB|-    |-     |-    |-     |14,506,627|

### dat_2
(1, 3, 480, 640) input, 1 warmup + 5 runs averaged
|Name|FPS|FPS ({abbr}`CL (channels_last)`)|{abbr}`CL (channels_last)` vs base|sec/img|VRAM ({abbr}`CL (channels_last)`)|PSNR (DF2K)|SSIM (DF2K)|PSNR (DIV2K)|SSIM (DIV2K)|Params|
|:-|-:|-:|-:|-:|-:|-:|-:|-:|-:|-:|
|dat_2  4x bf16                     |    0.28|    0.28|0.99x|  3.5597|    3.92 GB|27.86|0.8341|-    |-     |11,212,131|
|dat_2  3x bf16                     |    0.28|    0.28|1.00x|  3.5149|    3.90 GB|30.13|0.8878|-    |-     |11,249,059|
|dat_2  2x bf16                     |    0.28|    0.28|1.00x|  3.6257|    3.88 GB|34.31|0.9457|-    |-     |11,064,419|
|dat_2  1x bf16                     |    0.28|    0.28|1.00x|  3.5622|    3.87 GB|-    |-     |-    |-     |10,916,707|

### dat_light
(1, 3, 480, 640) input, 1 warmup + 5 runs averaged
|Name|FPS|FPS ({abbr}`CL (channels_last)`)|{abbr}`CL (channels_last)` vs base|sec/img|VRAM ({abbr}`CL (channels_last)`)|PSNR (DF2K)|SSIM (DF2K)|PSNR (DIV2K)|SSIM (DIV2K)|Params|
|:-|-:|-:|-:|-:|-:|-:|-:|-:|-:|-:|
|dat_light  4x bf16                 |    0.62|    0.60|0.96x|  1.6046|    2.72 GB|26.64|0.8033|-    |-     |   878,577|
|dat_light  3x bf16                 |    0.62|    0.63|1.02x|  1.6246|    2.69 GB|28.89|0.8666|-    |-     |   915,505|
|dat_light  2x bf16                 |    0.61|    0.62|1.01x|  1.6333|    2.67 GB|32.89|0.9346|-    |-     |   730,865|
|dat_light  1x bf16                 |    0.61|    0.63|1.02x|  1.6327|    2.66 GB|-    |-     |-    |-     |   583,153|

### dat_s
(1, 3, 480, 640) input, 1 warmup + 5 runs averaged
|Name|FPS|FPS ({abbr}`CL (channels_last)`)|{abbr}`CL (channels_last)` vs base|sec/img|VRAM ({abbr}`CL (channels_last)`)|PSNR (DF2K)|SSIM (DF2K)|PSNR (DIV2K)|SSIM (DIV2K)|Params|
|:-|-:|-:|-:|-:|-:|-:|-:|-:|-:|-:|
|dat_s  4x bf16                     |    0.56|    0.54|0.97x|  1.7947|    2.85 GB|27.68|0.8300|-    |-     |11,212,131|
|dat_s  3x bf16                     |    0.55|    0.55|1.00x|  1.8229|    2.82 GB|29.98|0.8846|-    |-     |11,249,059|
|dat_s  2x bf16                     |    0.56|    0.55|0.99x|  1.7985|    2.80 GB|34.12|0.9444|-    |-     |11,064,419|
|dat_s  1x bf16                     |    0.55|    0.54|0.98x|  1.8039|    2.79 GB|-    |-     |-    |-     |10,916,707|

### drct
(1, 3, 480, 640) input, 1 warmup + 5 runs averaged
|Name|FPS|FPS ({abbr}`CL (channels_last)`)|{abbr}`CL (channels_last)` vs base|sec/img|VRAM ({abbr}`CL (channels_last)`)|PSNR (DF2K)|SSIM (DF2K)|PSNR (DIV2K)|SSIM (DIV2K)|Params|
|:-|-:|-:|-:|-:|-:|-:|-:|-:|-:|-:|
|drct  4x bf16                      |    0.18|    0.18|0.99x|  5.6127|    6.28 GB|28.06|0.8378|-    |-     |14,139,579|
|drct  3x bf16                      |    0.18|    0.18|0.99x|  5.5078|    6.25 GB|30.34|0.8910|-    |-     |14,176,507|
|drct  2x bf16                      |    0.18|    0.18|1.01x|  5.5716|    6.24 GB|34.54|0.9474|-    |-     |13,991,867|
|drct  1x bf16                      |    0.18|    0.18|1.01x|  5.6035|    6.23 GB|-    |-     |-    |-     |13,844,155|

### drct_l
(1, 3, 480, 640) input, 1 warmup + 5 runs averaged
|Name|FPS|FPS ({abbr}`CL (channels_last)`)|{abbr}`CL (channels_last)` vs base|sec/img|VRAM ({abbr}`CL (channels_last)`)|PSNR (DF2K)|SSIM (DF2K)|PSNR (DIV2K)|SSIM (DIV2K)|Params|
|:-|-:|-:|-:|-:|-:|-:|-:|-:|-:|-:|
|drct_l  4x bf16                    |    0.08|    0.09|1.04x| 12.0505|    6.42 GB|28.70|0.8508|-    |-     |27,580,719|
|drct_l  3x bf16                    |    0.07|    0.05|0.78x| 14.8329|    6.39 GB|31.14|0.9004|-    |-     |27,617,647|
|drct_l  2x bf16                    |    0.09|    0.09|1.00x| 11.0373|    6.37 GB|35.17|0.9516|-    |-     |27,433,007|
|drct_l  1x bf16                    |    0.09|    0.09|1.00x| 11.1585|    6.36 GB|-    |-     |-    |-     |27,285,295|

### drct_xl
(1, 3, 480, 640) input, 1 warmup + 5 runs averaged
|Name|FPS|FPS ({abbr}`CL (channels_last)`)|{abbr}`CL (channels_last)` vs base|sec/img|VRAM ({abbr}`CL (channels_last)`)|PSNR (DF2K)|SSIM (DF2K)|PSNR (DIV2K)|SSIM (DIV2K)|Params|
|:-|-:|-:|-:|-:|-:|-:|-:|-:|-:|-:|
|drct_xl  4x bf16                   |    0.08|    0.08|1.00x| 13.0764|    6.46 GB|-    |-     |-    |-     |32,061,099|
|drct_xl  3x bf16                   |    0.04|    0.05|1.22x| 22.6431|    6.44 GB|-    |-     |-    |-     |32,098,027|
|drct_xl  2x bf16                   |    0.08|    0.08|0.99x| 13.0431|    6.42 GB|-    |-     |-    |-     |31,913,387|
|drct_xl  1x bf16                   |    0.08|    0.08|0.98x| 12.8536|    6.41 GB|-    |-     |-    |-     |31,765,675|

### eimn_a
(1, 3, 480, 640) input, 1 warmup + 5 runs averaged
|Name|FPS|FPS ({abbr}`CL (channels_last)`)|{abbr}`CL (channels_last)` vs base|sec/img|VRAM ({abbr}`CL (channels_last)`)|PSNR (DF2K)|SSIM (DF2K)|PSNR (DIV2K)|SSIM (DIV2K)|Params|
|:-|-:|-:|-:|-:|-:|-:|-:|-:|-:|-:|
|eimn_a  4x fp16                    |    8.39|    6.10|0.73x|  0.1193|    0.89 GB|26.68|0.8027|-    |-     |   880,870|
|eimn_a  3x fp16                    |    8.38|    6.11|0.73x|  0.1193|    0.87 GB|28.87|0.8660|-    |-     |   868,753|
|eimn_a  2x fp16                    |    8.36|    6.10|0.73x|  0.1197|    0.86 GB|33.15|0.9373|-    |-     |   860,098|
|eimn_a  1x fp16                    |    8.38|    6.10|0.73x|  0.1193|    0.86 GB|-    |-     |-    |-     |   854,905|

### eimn_l
(1, 3, 480, 640) input, 1 warmup + 5 runs averaged
|Name|FPS|FPS ({abbr}`CL (channels_last)`)|{abbr}`CL (channels_last)` vs base|sec/img|VRAM ({abbr}`CL (channels_last)`)|PSNR (DF2K)|SSIM (DF2K)|PSNR (DIV2K)|SSIM (DIV2K)|Params|
|:-|-:|-:|-:|-:|-:|-:|-:|-:|-:|-:|
|eimn_l  4x fp16                    |    7.32|    5.33|0.73x|  0.1366|    0.89 GB|26.88|0.8084|-    |-     | 1,002,496|
|eimn_l  3x fp16                    |    7.33|    5.35|0.73x|  0.1364|    0.87 GB|29.05|0.8698|-    |-     |   990,379|
|eimn_l  2x fp16                    |    7.31|    5.35|0.73x|  0.1368|    0.87 GB|33.23|0.9381|-    |-     |   981,724|
|eimn_l  1x fp16                    |    7.34|    5.34|0.73x|  0.1363|    0.86 GB|-    |-     |-    |-     |   976,531|

### esrgan use_pixel_unshuffle=False
(1, 3, 480, 640) input, 1 warmup + 5 runs averaged
|Name|FPS|FPS ({abbr}`CL (channels_last)`)|{abbr}`CL (channels_last)` vs base|sec/img|VRAM ({abbr}`CL (channels_last)`)|PSNR (DF2K)|SSIM (DF2K)|PSNR (DIV2K)|SSIM (DIV2K)|Params|
|:-|-:|-:|-:|-:|-:|-:|-:|-:|-:|-:|
|esrgan use_pixel_unshuffle=False 4x fp16|    3.24|    4.52|1.40x|  0.3088|    2.48 GB|-    |-     |-    |-     |16,697,987|
|esrgan use_pixel_unshuffle=False 3x fp16|    3.37|    4.85|1.44x|  0.2964|    1.44 GB|-    |-     |-    |-     |16,661,059|
|esrgan use_pixel_unshuffle=False 2x fp16|    3.39|    4.88|1.44x|  0.2952|    0.70 GB|-    |-     |-    |-     |16,661,059|
|esrgan use_pixel_unshuffle=False 1x fp16|    3.50|    5.00|1.43x|  0.2856|    0.44 GB|-    |-     |-    |-     |16,624,131|

### esrgan use_pixel_unshuffle=True
(1, 3, 480, 640) input, 1 warmup + 5 runs averaged
|Name|FPS|FPS ({abbr}`CL (channels_last)`)|{abbr}`CL (channels_last)` vs base|sec/img|VRAM ({abbr}`CL (channels_last)`)|PSNR (DF2K)|SSIM (DF2K)|PSNR (DIV2K)|SSIM (DIV2K)|Params|
|:-|-:|-:|-:|-:|-:|-:|-:|-:|-:|-:|
|esrgan use_pixel_unshuffle=True 4x fp16|    3.15|    4.45|1.41x|  0.3170|    2.48 GB|27.03|0.8153|26.73|0.8072|16,697,987|
|esrgan use_pixel_unshuffle=True 3x fp16|    3.27|    4.66|1.43x|  0.3060|    1.44 GB|-    |-     |-    |-     |16,661,059|
|esrgan use_pixel_unshuffle=True 2x fp16|   17.88|   22.91|1.28x|  0.0559|    0.70 GB|-    |-     |-    |-     |16,703,171|
|esrgan use_pixel_unshuffle=True 1x fp16|   40.21|   61.54|1.53x|  0.0249|    0.26 GB|-    |-     |-    |-     |16,723,907|

### esrgan_lite
(1, 3, 480, 640) input, 1 warmup + 5 runs averaged
|Name|FPS|FPS ({abbr}`CL (channels_last)`)|{abbr}`CL (channels_last)` vs base|sec/img|VRAM ({abbr}`CL (channels_last)`)|PSNR (DF2K)|SSIM (DF2K)|PSNR (DIV2K)|SSIM (DIV2K)|Params|
|:-|-:|-:|-:|-:|-:|-:|-:|-:|-:|-:|
|esrgan_lite  4x fp16               |    8.57|   12.13|1.42x|  0.1167|    1.24 GB|-    |-     |-    |-     | 5,021,155|
|esrgan_lite  3x fp16               |    9.32|   13.01|1.40x|  0.1073|    0.71 GB|-    |-     |-    |-     | 5,011,907|
|esrgan_lite  2x fp16               |   43.35|   59.03|1.36x|  0.0231|    0.34 GB|-    |-     |-    |-     | 5,023,747|
|esrgan_lite  1x fp16               |   86.02|  107.18|1.25x|  0.0116|    0.12 GB|-    |-     |-    |-     | 5,034,115|

### flexnet
(1, 3, 480, 640) input, 1 warmup + 5 runs averaged
|Name|FPS|FPS ({abbr}`CL (channels_last)`)|{abbr}`CL (channels_last)` vs base|sec/img|VRAM ({abbr}`CL (channels_last)`)|PSNR (DF2K)|SSIM (DF2K)|PSNR (DIV2K)|SSIM (DIV2K)|Params|
|:-|-:|-:|-:|-:|-:|-:|-:|-:|-:|-:|
|flexnet  4x fp16                   |    3.45|    3.61|1.05x|  0.2902|    1.31 GB|-    |-     |-    |-     | 3,045,136|
|flexnet  3x fp16                   |    3.45|    3.62|1.05x|  0.2895|    1.29 GB|-    |-     |-    |-     | 3,020,923|
|flexnet  2x fp16                   |    3.45|    3.61|1.05x|  0.2902|    1.28 GB|-    |-     |-    |-     | 3,003,628|
|flexnet  1x fp16                   |    3.45|    3.61|1.05x|  0.2901|    1.28 GB|-    |-     |-    |-     | 2,993,251|

### hat_l
(1, 3, 480, 640) input, 1 warmup + 5 runs averaged
|Name|FPS|FPS ({abbr}`CL (channels_last)`)|{abbr}`CL (channels_last)` vs base|sec/img|VRAM ({abbr}`CL (channels_last)`)|PSNR (DF2K)|SSIM (DF2K)|PSNR (DIV2K)|SSIM (DIV2K)|Params|
|:-|-:|-:|-:|-:|-:|-:|-:|-:|-:|-:|
|hat_l  4x bf16                     |    0.29|    0.31|1.07x|  3.4795|   10.41 GB|28.60|0.8498|-    |-     |40,846,575|
|hat_l  3x bf16                     |    0.29|    0.31|1.08x|  3.4702|   10.39 GB|30.92|0.8981|-    |-     |40,883,503|
|hat_l  2x bf16                     |    0.29|    0.31|1.07x|  3.4752|   10.37 GB|35.09|0.9513|-    |-     |40,698,863|
|hat_l  1x bf16                     |    0.29|    0.31|1.08x|  3.4572|   10.36 GB|-    |-     |-    |-     |40,551,151|

### hat_m
(1, 3, 480, 640) input, 1 warmup + 5 runs averaged
|Name|FPS|FPS ({abbr}`CL (channels_last)`)|{abbr}`CL (channels_last)` vs base|sec/img|VRAM ({abbr}`CL (channels_last)`)|PSNR (DF2K)|SSIM (DF2K)|PSNR (DIV2K)|SSIM (DIV2K)|Params|
|:-|-:|-:|-:|-:|-:|-:|-:|-:|-:|-:|
|hat_m  4x bf16                     |    0.54|    0.58|1.07x|  1.8418|   10.30 GB|27.97|0.8368|-    |-     |20,772,507|
|hat_m  3x bf16                     |    0.55|    0.59|1.07x|  1.8147|   10.27 GB|30.23|0.8896|-    |-     |20,809,435|
|hat_m  2x bf16                     |    0.55|    0.58|1.07x|  1.8294|   10.26 GB|34.45|0.9466|-    |-     |20,624,795|
|hat_m  1x bf16                     |    0.55|    0.59|1.07x|  1.8091|   10.24 GB|-    |-     |-    |-     |20,477,083|

### hat_s
(1, 3, 480, 640) input, 1 warmup + 5 runs averaged
|Name|FPS|FPS ({abbr}`CL (channels_last)`)|{abbr}`CL (channels_last)` vs base|sec/img|VRAM ({abbr}`CL (channels_last)`)|PSNR (DF2K)|SSIM (DF2K)|PSNR (DIV2K)|SSIM (DIV2K)|Params|
|:-|-:|-:|-:|-:|-:|-:|-:|-:|-:|-:|
|hat_s  4x bf16                     |    0.62|    0.66|1.06x|  1.6021|    9.85 GB|27.87|0.8346|-    |-     | 9,621,183|
|hat_s  3x bf16                     |    0.63|    0.67|1.06x|  1.5760|    9.83 GB|30.15|0.8879|-    |-     | 9,658,111|
|hat_s  2x bf16                     |    0.63|    0.67|1.06x|  1.5879|    9.81 GB|34.31|0.9459|-    |-     | 9,473,471|
|hat_s  1x bf16                     |    0.64|    0.67|1.06x|  1.5748|    9.80 GB|-    |-     |-    |-     | 9,325,759|

### hit_sir
(1, 3, 480, 640) input, 1 warmup + 5 runs averaged
|Name|FPS|FPS ({abbr}`CL (channels_last)`)|{abbr}`CL (channels_last)` vs base|sec/img|VRAM ({abbr}`CL (channels_last)`)|PSNR (DF2K)|SSIM (DF2K)|PSNR (DIV2K)|SSIM (DIV2K)|Params|
|:-|-:|-:|-:|-:|-:|-:|-:|-:|-:|-:|
|hit_sir  4x bf16                   |    5.28|    5.13|0.97x|  0.1893|    1.34 GB|-    |-     |26.71|0.8045|   791,540|
|hit_sir  3x bf16                   |    5.29|    5.13|0.97x|  0.1891|    1.32 GB|-    |-     |28.93|0.8673|   780,179|
|hit_sir  2x bf16                   |    5.27|    5.13|0.97x|  0.1896|    1.30 GB|-    |-     |33.02|0.9365|   772,064|
|hit_sir  1x bf16                   |    5.27|    5.12|0.97x|  0.1897|    1.29 GB|-    |-     |-    |-     |   767,195|

### hit_sng
(1, 3, 480, 640) input, 1 warmup + 5 runs averaged
|Name|FPS|FPS ({abbr}`CL (channels_last)`)|{abbr}`CL (channels_last)` vs base|sec/img|VRAM ({abbr}`CL (channels_last)`)|PSNR (DF2K)|SSIM (DF2K)|PSNR (DIV2K)|SSIM (DIV2K)|Params|
|:-|-:|-:|-:|-:|-:|-:|-:|-:|-:|-:|
|hit_sng  4x bf16                   |    4.11|    3.93|0.96x|  0.2435|    1.34 GB|-    |-     |26.75|0.8053| 1,032,060|
|hit_sng  3x bf16                   |    4.11|    3.94|0.96x|  0.2431|    1.32 GB|-    |-     |28.91|0.8671| 1,020,699|
|hit_sng  2x bf16                   |    4.11|    3.93|0.96x|  0.2435|    1.30 GB|-    |-     |33.01|0.9360| 1,012,584|
|hit_sng  1x bf16                   |    4.10|    3.92|0.95x|  0.2438|    1.29 GB|-    |-     |-    |-     | 1,007,715|

### hit_srf
(1, 3, 480, 640) input, 1 warmup + 5 runs averaged
|Name|FPS|FPS ({abbr}`CL (channels_last)`)|{abbr}`CL (channels_last)` vs base|sec/img|VRAM ({abbr}`CL (channels_last)`)|PSNR (DF2K)|SSIM (DF2K)|PSNR (DIV2K)|SSIM (DIV2K)|Params|
|:-|-:|-:|-:|-:|-:|-:|-:|-:|-:|-:|
|hit_srf  4x bf16                   |    4.24|    4.20|0.99x|  0.2359|    1.34 GB|-    |-     |26.80|0.8069|   866,420|
|hit_srf  3x bf16                   |    4.24|    4.20|0.99x|  0.2358|    1.32 GB|-    |-     |28.99|0.8687|   855,059|
|hit_srf  2x bf16                   |    4.24|    4.19|0.99x|  0.2359|    1.30 GB|-    |-     |33.13|0.9372|   846,944|
|hit_srf  1x bf16                   |    4.24|    4.19|0.99x|  0.2359|    1.29 GB|-    |-     |-    |-     |   842,075|

### lmlt_base
(1, 3, 480, 640) input, 1 warmup + 5 runs averaged
|Name|FPS|FPS ({abbr}`CL (channels_last)`)|{abbr}`CL (channels_last)` vs base|sec/img|VRAM ({abbr}`CL (channels_last)`)|PSNR (DF2K)|SSIM (DF2K)|PSNR (DIV2K)|SSIM (DIV2K)|Params|
|:-|-:|-:|-:|-:|-:|-:|-:|-:|-:|-:|
|lmlt_base  4x fp16                 |   15.12|   15.01|0.99x|  0.0661|    0.63 GB|26.44|0.7949|-    |-     |   671,808|
|lmlt_base  3x fp16                 |   15.11|   15.08|1.00x|  0.0662|    0.62 GB|28.48|0.8581|-    |-     |   660,447|
|lmlt_base  2x fp16                 |   15.11|   15.14|1.00x|  0.0662|    0.61 GB|32.52|0.9316|-    |-     |   652,332|
|lmlt_base  1x fp16                 |   14.94|   15.02|1.01x|  0.0669|    0.61 GB|-    |-     |-    |-     |   647,463|

### lmlt_large
(1, 3, 480, 640) input, 1 warmup + 5 runs averaged
|Name|FPS|FPS ({abbr}`CL (channels_last)`)|{abbr}`CL (channels_last)` vs base|sec/img|VRAM ({abbr}`CL (channels_last)`)|PSNR (DF2K)|SSIM (DF2K)|PSNR (DIV2K)|SSIM (DIV2K)|Params|
|:-|-:|-:|-:|-:|-:|-:|-:|-:|-:|-:|
|lmlt_large  4x fp16                |   10.36|   10.29|0.99x|  0.0965|    0.87 GB|26.63|0.8001|-    |-     | 1,295,328|
|lmlt_large  3x fp16                |   10.37|   10.33|1.00x|  0.0965|    0.86 GB|28.72|0.8628|-    |-     | 1,279,431|
|lmlt_large  2x fp16                |   10.37|   10.33|1.00x|  0.0964|    0.85 GB|32.75|0.9336|-    |-     | 1,268,076|
|lmlt_large  1x fp16                |   10.34|   10.28|0.99x|  0.0967|    0.85 GB|-    |-     |-    |-     | 1,261,263|

### lmlt_tiny
(1, 3, 480, 640) input, 1 warmup + 5 runs averaged
|Name|FPS|FPS ({abbr}`CL (channels_last)`)|{abbr}`CL (channels_last)` vs base|sec/img|VRAM ({abbr}`CL (channels_last)`)|PSNR (DF2K)|SSIM (DF2K)|PSNR (DIV2K)|SSIM (DIV2K)|Params|
|:-|-:|-:|-:|-:|-:|-:|-:|-:|-:|-:|
|lmlt_tiny  4x fp16                 |   28.42|   28.67|1.01x|  0.0352|    0.37 GB|26.08|0.7838|-    |-     |   251,040|
|lmlt_tiny  3x fp16                 |   28.64|   28.36|0.99x|  0.0349|    0.36 GB|28.10|0.8503|-    |-     |   244,215|
|lmlt_tiny  2x fp16                 |   28.66|   28.58|1.00x|  0.0349|    0.35 GB|32.04|0.9273|-    |-     |   239,340|
|lmlt_tiny  1x fp16                 |   27.71|   28.67|1.03x|  0.0361|    0.35 GB|-    |-     |-    |-     |   236,415|

### man
(1, 3, 480, 640) input, 1 warmup + 5 runs averaged
|Name|FPS|FPS ({abbr}`CL (channels_last)`)|{abbr}`CL (channels_last)` vs base|sec/img|VRAM ({abbr}`CL (channels_last)`)|PSNR (DF2K)|SSIM (DF2K)|PSNR (DIV2K)|SSIM (DIV2K)|Params|
|:-|-:|-:|-:|-:|-:|-:|-:|-:|-:|-:|
|man  4x fp16                       |    1.26|    0.86|0.69x|  0.7929|    1.53 GB|27.26|0.8197|-    |-     | 8,712,612|
|man  3x fp16                       |    1.26|    0.87|0.69x|  0.7920|    1.52 GB|29.52|0.8782|-    |-     | 8,678,571|
|man  2x fp16                       |    1.26|    0.86|0.68x|  0.7923|    1.51 GB|33.73|0.9422|-    |-     | 8,654,256|
|man  1x fp16                       |    1.26|    0.87|0.69x|  0.7921|    1.51 GB|-    |-     |-    |-     | 8,639,667|

### man_light
(1, 3, 480, 640) input, 1 warmup + 5 runs averaged
|Name|FPS|FPS ({abbr}`CL (channels_last)`)|{abbr}`CL (channels_last)` vs base|sec/img|VRAM ({abbr}`CL (channels_last)`)|PSNR (DF2K)|SSIM (DF2K)|PSNR (DIV2K)|SSIM (DIV2K)|Params|
|:-|-:|-:|-:|-:|-:|-:|-:|-:|-:|-:|
|man_light  4x fp16                 |    6.01|    4.42|0.73x|  0.1663|    0.53 GB|26.70|0.8052|-    |-     |   842,892|
|man_light  3x fp16                 |    6.02|    4.41|0.73x|  0.1662|    0.52 GB|-    |-     |-    |-     |   831,531|
|man_light  2x fp16                 |    6.04|    4.41|0.73x|  0.1657|    0.51 GB|-    |-     |-    |-     |   823,416|
|man_light  1x fp16                 |    6.02|    4.43|0.74x|  0.1661|    0.50 GB|-    |-     |-    |-     |   818,547|

### man_tiny
(1, 3, 480, 640) input, 1 warmup + 5 runs averaged
|Name|FPS|FPS ({abbr}`CL (channels_last)`)|{abbr}`CL (channels_last)` vs base|sec/img|VRAM ({abbr}`CL (channels_last)`)|PSNR (DF2K)|SSIM (DF2K)|PSNR (DIV2K)|SSIM (DIV2K)|Params|
|:-|-:|-:|-:|-:|-:|-:|-:|-:|-:|-:|
|man_tiny  4x fp16                  |   34.75|   27.41|0.79x|  0.0288|    0.43 GB|25.84|0.7786|-    |-     |   150,024|
|man_tiny  3x fp16                  |   34.69|   27.60|0.80x|  0.0288|    0.41 GB|-    |-     |-    |-     |   140,931|
|man_tiny  2x fp16                  |   35.11|   27.85|0.79x|  0.0285|    0.40 GB|-    |-     |-    |-     |   134,436|
|man_tiny  1x fp16                  |   35.22|   27.47|0.78x|  0.0284|    0.40 GB|-    |-     |-    |-     |   130,539|

### metaflexnet
(1, 3, 480, 640) input, 1 warmup + 5 runs averaged
|Name|FPS|FPS ({abbr}`CL (channels_last)`)|{abbr}`CL (channels_last)` vs base|sec/img|VRAM ({abbr}`CL (channels_last)`)|PSNR (DF2K)|SSIM (DF2K)|PSNR (DIV2K)|SSIM (DIV2K)|Params|
|:-|-:|-:|-:|-:|-:|-:|-:|-:|-:|-:|
|metaflexnet  4x fp16               |    6.61|    6.75|1.02x|  0.1514|    1.80 GB|-    |-     |-    |-     |67,205,424|
|metaflexnet  3x fp16               |    6.61|    6.75|1.02x|  0.1513|    1.79 GB|-    |-     |-    |-     |67,181,211|
|metaflexnet  2x fp16               |    6.61|    6.77|1.02x|  0.1514|    1.78 GB|-    |-     |-    |-     |67,163,916|
|metaflexnet  1x fp16               |    6.60|    6.73|1.02x|  0.1516|    1.77 GB|-    |-     |-    |-     |67,153,539|

### moesr2
(1, 3, 480, 640) input, 1 warmup + 5 runs averaged
|Name|FPS|FPS ({abbr}`CL (channels_last)`)|{abbr}`CL (channels_last)` vs base|sec/img|VRAM ({abbr}`CL (channels_last)`)|PSNR (DF2K)|SSIM (DF2K)|PSNR (DIV2K)|SSIM (DIV2K)|Params|
|:-|-:|-:|-:|-:|-:|-:|-:|-:|-:|-:|
|moesr2  4x fp16                    |    3.93|    3.69|0.94x|  0.2543|    0.91 GB|27.05|0.8177|-    |-     |16,547,008|
|moesr2  3x fp16                    |    3.94|    3.69|0.94x|  0.2539|    0.90 GB|-    |-     |-    |-     |16,534,891|
|moesr2  2x fp16                    |    3.93|    3.69|0.94x|  0.2544|    0.89 GB|-    |-     |-    |-     |16,526,236|
|moesr2  1x fp16                    |    3.94|    3.70|0.94x|  0.2540|    0.89 GB|-    |-     |-    |-     |16,521,043|

### mosr
(1, 3, 480, 640) input, 1 warmup + 5 runs averaged
|Name|FPS|FPS ({abbr}`CL (channels_last)`)|{abbr}`CL (channels_last)` vs base|sec/img|VRAM ({abbr}`CL (channels_last)`)|PSNR (DF2K)|SSIM (DF2K)|PSNR (DIV2K)|SSIM (DIV2K)|Params|
|:-|-:|-:|-:|-:|-:|-:|-:|-:|-:|-:|
|mosr  4x fp16                      |    9.27|   11.42|1.23x|  0.1078|    0.49 GB|-    |-     |-    |-     | 4,287,600|
|mosr  3x fp16                      |    9.28|   11.48|1.24x|  0.1078|    0.47 GB|-    |-     |-    |-     | 4,275,483|
|mosr  2x fp16                      |    9.29|   11.47|1.23x|  0.1076|    0.46 GB|-    |-     |-    |-     | 4,266,828|
|mosr  1x fp16                      |    9.29|   11.31|1.22x|  0.1076|    0.46 GB|-    |-     |-    |-     | 4,261,635|

### mosr_t
(1, 3, 480, 640) input, 1 warmup + 5 runs averaged
|Name|FPS|FPS ({abbr}`CL (channels_last)`)|{abbr}`CL (channels_last)` vs base|sec/img|VRAM ({abbr}`CL (channels_last)`)|PSNR (DF2K)|SSIM (DF2K)|PSNR (DIV2K)|SSIM (DIV2K)|Params|
|:-|-:|-:|-:|-:|-:|-:|-:|-:|-:|-:|
|mosr_t  4x fp16                    |   53.89|   64.80|1.20x|  0.0186|    0.36 GB|-    |-     |-    |-     |   609,888|
|mosr_t  3x fp16                    |   53.71|   64.79|1.21x|  0.0186|    0.35 GB|-    |-     |-    |-     |   600,795|
|mosr_t  2x fp16                    |   53.09|   65.02|1.22x|  0.0188|    0.34 GB|-    |-     |-    |-     |   594,300|
|mosr_t  1x fp16                    |   53.90|   63.14|1.17x|  0.0186|    0.33 GB|-    |-     |-    |-     |   590,403|

### omnisr
(1, 3, 480, 640) input, 1 warmup + 5 runs averaged
|Name|FPS|FPS ({abbr}`CL (channels_last)`)|{abbr}`CL (channels_last)` vs base|sec/img|VRAM ({abbr}`CL (channels_last)`)|PSNR (DF2K)|SSIM (DF2K)|PSNR (DIV2K)|SSIM (DIV2K)|Params|
|:-|-:|-:|-:|-:|-:|-:|-:|-:|-:|-:|
|omnisr  4x fp16                    |   37.78|   37.50|0.99x|  0.0265|    1.14 GB|26.95|0.8105|26.64|0.8018|   214,208|
|omnisr  3x fp16                    |   37.90|   37.29|0.98x|  0.0264|    1.13 GB|29.12|0.8712|28.84|0.8656|   202,091|
|omnisr  2x fp16                    |   37.80|   37.81|1.00x|  0.0265|    1.12 GB|33.30|0.9386|33.05|0.9363|   193,436|
|omnisr  1x fp16                    |   38.02|   37.48|0.99x|  0.0263|    1.12 GB|-    |-     |-    |-     |   188,243|

### plksr
(1, 3, 480, 640) input, 1 warmup + 5 runs averaged
|Name|FPS|FPS ({abbr}`CL (channels_last)`)|{abbr}`CL (channels_last)` vs base|sec/img|VRAM ({abbr}`CL (channels_last)`)|PSNR (DF2K)|SSIM (DF2K)|PSNR (DIV2K)|SSIM (DIV2K)|Params|
|:-|-:|-:|-:|-:|-:|-:|-:|-:|-:|-:|
|plksr  4x fp16                     |   11.97|   15.68|1.31x|  0.0835|    0.29 GB|26.85|0.8097|26.69|0.8054| 7,386,096|
|plksr  3x fp16                     |   12.01|   15.86|1.32x|  0.0832|    0.27 GB|29.10|0.8713|28.86|0.8666| 7,373,979|
|plksr  2x fp16                     |   12.10|   15.96|1.32x|  0.0827|    0.25 GB|33.36|0.9395|32.99|0.9365| 7,365,324|
|plksr  1x fp16                     |   12.05|   15.98|1.33x|  0.0830|    0.24 GB|-    |-     |-    |-     | 7,360,131|

### plksr_tiny
(1, 3, 480, 640) input, 1 warmup + 250 runs averaged
|Name|FPS|FPS ({abbr}`CL (channels_last)`)|{abbr}`CL (channels_last)` vs base|sec/img|VRAM ({abbr}`CL (channels_last)`)|PSNR (DF2K)|SSIM (DF2K)|PSNR (DIV2K)|SSIM (DIV2K)|Params|
|:-|-:|-:|-:|-:|-:|-:|-:|-:|-:|-:|
|plksr_tiny  4x fp16                |   36.91|   49.47|1.34x|  0.0271|    0.27 GB|26.34|0.7942|26.12|0.7888| 2,370,544|
|plksr_tiny  3x fp16                |   37.39|   50.13|1.34x|  0.0267|    0.24 GB|28.51|0.8599|28.35|0.8571| 2,358,427|
|plksr_tiny  2x fp16                |   37.56|   50.83|1.35x|  0.0266|    0.22 GB|32.58|0.9328|32.43|0.9314| 2,349,772|
|plksr_tiny  1x fp16                |   37.55|   50.74|1.35x|  0.0266|    0.21 GB|-    |-     |-    |-     | 2,344,579|

### rcan
(1, 3, 480, 640) input, 1 warmup + 5 runs averaged
|Name|FPS|FPS ({abbr}`CL (channels_last)`)|{abbr}`CL (channels_last)` vs base|sec/img|VRAM ({abbr}`CL (channels_last)`)|PSNR (DF2K)|SSIM (DF2K)|PSNR (DIV2K)|SSIM (DIV2K)|Params|
|:-|-:|-:|-:|-:|-:|-:|-:|-:|-:|-:|
|rcan  4x fp16                      |    5.35|    7.04|1.32x|  0.1868|    1.40 GB|27.16|0.8168|26.82|0.8087|15,592,379|
|rcan  3x fp16                      |    5.57|    7.59|1.36x|  0.1795|    0.86 GB|-    |-     |29.09|0.8702|15,629,307|
|rcan  2x fp16                      |    5.63|    7.76|1.38x|  0.1775|    0.48 GB|-    |-     |33.34|0.9384|15,444,667|
|rcan  1x fp16                      |    5.72|    7.98|1.40x|  0.1749|    0.28 GB|-    |-     |-    |-     |15,296,955|

### realcugan
(1, 3, 480, 640) input, 1 warmup + 250 runs averaged
|Name|FPS|FPS ({abbr}`CL (channels_last)`)|{abbr}`CL (channels_last)` vs base|sec/img|VRAM ({abbr}`CL (channels_last)`)|PSNR (DF2K)|SSIM (DF2K)|PSNR (DIV2K)|SSIM (DIV2K)|Params|
|:-|-:|-:|-:|-:|-:|-:|-:|-:|-:|-:|
|realcugan  4x fp16                 |   44.13|   61.32|1.39x|  0.0227|    0.95 GB|-    |-     |-    |-     | 1,406,812|
|realcugan  3x fp16                 |   25.16|   34.40|1.37x|  0.0398|    1.71 GB|-    |-     |-    |-     | 1,286,326|
|realcugan  2x fp16                 |   59.73|   81.21|1.36x|  0.0167|    0.78 GB|-    |-     |-    |-     | 1,284,598|
|realcugan  1x fp16                 |-       |-       |-       |-       |-       |-|-|-||-         |

### realplksr dysample layer_norm=False
(1, 3, 480, 640) input, 1 warmup + 5 runs averaged
|Name|FPS|FPS ({abbr}`CL (channels_last)`)|{abbr}`CL (channels_last)` vs base|sec/img|VRAM ({abbr}`CL (channels_last)`)|PSNR (DF2K)|SSIM (DF2K)|PSNR (DIV2K)|SSIM (DIV2K)|Params|
|:-|-:|-:|-:|-:|-:|-:|-:|-:|-:|-:|
|realplksr dysample layer_norm=False 4x fp16|    5.54|    5.76|1.04x|  0.1804|    2.21 GB|-    |-     |-    |-     | 7,402,243|
|realplksr dysample layer_norm=False 3x fp16|    5.72|    6.00|1.05x|  0.1749|    0.79 GB|-    |-     |-    |-     | 7,380,617|
|realplksr dysample layer_norm=False 2x fp16|    5.78|    6.07|1.05x|  0.1729|    0.39 GB|-    |-     |-    |-     | 7,369,747|
|realplksr dysample layer_norm=False 1x fp16|    5.82|    6.11|1.05x|  0.1718|    0.39 GB|-    |-     |-    |-     | 7,363,757|

### realplksr dysample layer_norm=True
(1, 3, 480, 640) input, 1 warmup + 5 runs averaged
|Name|FPS|FPS ({abbr}`CL (channels_last)`)|{abbr}`CL (channels_last)` vs base|sec/img|VRAM ({abbr}`CL (channels_last)`)|PSNR (DF2K)|SSIM (DF2K)|PSNR (DIV2K)|SSIM (DIV2K)|Params|
|:-|-:|-:|-:|-:|-:|-:|-:|-:|-:|-:|
|realplksr dysample layer_norm=True 4x fp16|    7.88|    9.09|1.15x|  0.1270|    2.21 GB|-    |-     |-    |-     | 7,402,243|
|realplksr dysample layer_norm=True 3x fp16|    8.23|    9.70|1.18x|  0.1215|    0.79 GB|-    |-     |-    |-     | 7,380,617|
|realplksr dysample layer_norm=True 2x fp16|    8.37|    9.96|1.19x|  0.1194|    0.32 GB|-    |-     |-    |-     | 7,369,747|
|realplksr dysample layer_norm=True 1x fp16|    8.44|   10.04|1.19x|  0.1184|    0.32 GB|-    |-     |-    |-     | 7,363,757|

### realplksr pixelshuffle layer_norm=False
(1, 3, 480, 640) input, 1 warmup + 5 runs averaged
|Name|FPS|FPS ({abbr}`CL (channels_last)`)|{abbr}`CL (channels_last)` vs base|sec/img|VRAM ({abbr}`CL (channels_last)`)|PSNR (DF2K)|SSIM (DF2K)|PSNR (DIV2K)|SSIM (DIV2K)|Params|
|:-|-:|-:|-:|-:|-:|-:|-:|-:|-:|-:|
|realplksr pixelshuffle layer_norm=False 4x fp16|    5.83|    6.11|1.05x|  0.1715|    0.44 GB|-    |-     |-    |-     | 7,389,680|
|realplksr pixelshuffle layer_norm=False 3x fp16|    5.84|    6.13|1.05x|  0.1713|    0.42 GB|-    |-     |-    |-     | 7,377,563|
|realplksr pixelshuffle layer_norm=False 2x fp16|    5.85|    6.12|1.05x|  0.1710|    0.40 GB|-    |-     |-    |-     | 7,368,908|
|realplksr pixelshuffle layer_norm=False 1x fp16|    5.84|    6.15|1.05x|  0.1712|    0.39 GB|-    |-     |-    |-     | 7,363,715|

### realplksr pixelshuffle layer_norm=True
(1, 3, 480, 640) input, 1 warmup + 5 runs averaged
|Name|FPS|FPS ({abbr}`CL (channels_last)`)|{abbr}`CL (channels_last)` vs base|sec/img|VRAM ({abbr}`CL (channels_last)`)|PSNR (DF2K)|SSIM (DF2K)|PSNR (DIV2K)|SSIM (DIV2K)|Params|
|:-|-:|-:|-:|-:|-:|-:|-:|-:|-:|-:|
|realplksr pixelshuffle layer_norm=True 4x fp16|    8.42|   10.02|1.19x|  0.1187|    0.37 GB|26.94|0.8140|-    |-     | 7,389,680|
|realplksr pixelshuffle layer_norm=True 3x fp16|    8.46|   10.11|1.20x|  0.1182|    0.35 GB|-    |-     |-    |-     | 7,377,563|
|realplksr pixelshuffle layer_norm=True 2x fp16|    8.47|   10.11|1.19x|  0.1181|    0.33 GB|33.44|0.9412|-    |-     | 7,368,908|
|realplksr pixelshuffle layer_norm=True 1x fp16|    8.47|   10.15|1.20x|  0.1181|    0.32 GB|-    |-     |-    |-     | 7,363,715|

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

### rtmosr
(1, 3, 480, 640) input, 1 warmup + 250 runs averaged
|Name|FPS|FPS ({abbr}`CL (channels_last)`)|{abbr}`CL (channels_last)` vs base|sec/img|VRAM ({abbr}`CL (channels_last)`)|PSNR (DF2K)|SSIM (DF2K)|PSNR (DIV2K)|SSIM (DIV2K)|Params|
|:-|-:|-:|-:|-:|-:|-:|-:|-:|-:|-:|
|rtmosr  4x fp16                    |  369.63|  345.34|0.93x|  0.0027|    0.30 GB|-    |-     |-    |-     |   460,294|
|rtmosr  3x fp16                    |  402.14|  405.48|1.01x|  0.0025|    0.27 GB|-    |-     |-    |-     |   454,225|
|rtmosr  2x fp16                    |  423.64|  423.91|1.00x|  0.0024|    0.26 GB|-    |-     |-    |-     |   449,890|
|rtmosr  1x fp16                    |  422.67|  438.16|1.04x|  0.0024|    0.25 GB|-    |-     |-    |-     |   447,289|

### rtmosr_s
(1, 3, 480, 640) input, 1 warmup + 250 runs averaged
|Name|FPS|FPS ({abbr}`CL (channels_last)`)|{abbr}`CL (channels_last)` vs base|sec/img|VRAM ({abbr}`CL (channels_last)`)|PSNR (DF2K)|SSIM (DF2K)|PSNR (DIV2K)|SSIM (DIV2K)|Params|
|:-|-:|-:|-:|-:|-:|-:|-:|-:|-:|-:|
|rtmosr_s  4x fp16                  |  370.62|  344.31|0.93x|  0.0027|    0.30 GB|-    |-     |-    |-     |   460,294|
|rtmosr_s  3x fp16                  |-       |-       |-       |-       |-       |-|-|-||-         |
|rtmosr_s  2x fp16                  | 1087.94|  920.15|0.85x|  0.0009|    0.08 GB|-    |-     |-    |-     |   462,886|
|rtmosr_s  1x fp16                  | 1249.20|  942.58|0.75x|  0.0008|    0.03 GB|-    |-     |-    |-     |   473,254|

### scunet_aaf6aa
(1, 3, 480, 640) input, 1 warmup + 5 runs averaged
|Name|FPS|FPS ({abbr}`CL (channels_last)`)|{abbr}`CL (channels_last)` vs base|sec/img|VRAM ({abbr}`CL (channels_last)`)|PSNR (DF2K)|SSIM (DF2K)|PSNR (DIV2K)|SSIM (DIV2K)|Params|
|:-|-:|-:|-:|-:|-:|-:|-:|-:|-:|-:|
|scunet_aaf6aa  4x fp16             |    8.52|   10.91|1.28x|  0.1174|    3.37 GB|-    |-     |-    |-     |15,207,468|
|scunet_aaf6aa  3x fp16             |   11.52|   14.15|1.23x|  0.0868|    1.02 GB|-    |-     |-    |-     |15,170,540|
|scunet_aaf6aa  2x fp16             |   11.60|   14.24|1.23x|  0.0862|    1.03 GB|-    |-     |-    |-     |15,170,540|
|scunet_aaf6aa  1x fp16             |   27.32|   27.77|1.02x|  0.0366|    0.80 GB|-    |-     |-    |-     | 9,699,756|

### span
(1, 3, 480, 640) input, 1 warmup + 250 runs averaged
|Name|FPS|FPS ({abbr}`CL (channels_last)`)|{abbr}`CL (channels_last)` vs base|sec/img|VRAM ({abbr}`CL (channels_last)`)|PSNR (DF2K)|SSIM (DF2K)|PSNR (DIV2K)|SSIM (DIV2K)|Params|
|:-|-:|-:|-:|-:|-:|-:|-:|-:|-:|-:|
|span  4x fp16                      |  101.93|  112.30|1.10x|  0.0098|    0.60 GB|26.18|0.7879|-    |-     | 2,236,728|
|span  3x fp16                      |   98.51|  109.16|1.11x|  0.0102|    0.59 GB|-    |-     |-    |-     | 2,227,635|
|span  2x fp16                      |  102.29|  110.88|1.08x|  0.0098|    0.58 GB|32.24|0.9294|-    |-     | 2,221,140|
|span  1x fp16                      |  102.00|  112.77|1.11x|  0.0098|    0.57 GB|-    |-     |-    |-     | 2,217,243|

### spanplus
(1, 3, 480, 640) input, 1 warmup + 250 runs averaged
|Name|FPS|FPS ({abbr}`CL (channels_last)`)|{abbr}`CL (channels_last)` vs base|sec/img|VRAM ({abbr}`CL (channels_last)`)|PSNR (DF2K)|SSIM (DF2K)|PSNR (DIV2K)|SSIM (DIV2K)|Params|
|:-|-:|-:|-:|-:|-:|-:|-:|-:|-:|-:|
|spanplus  4x fp16                  |   54.79|   50.94|0.93x|  0.0182|    2.15 GB|-    |-     |-    |-     | 2,228,507|
|spanplus  3x fp16                  |   64.88|   64.24|0.99x|  0.0154|    1.23 GB|-    |-     |-    |-     | 2,223,075|
|spanplus  2x fp16                  |   80.25|   82.59|1.03x|  0.0125|    0.58 GB|-    |-     |-    |-     | 2,219,195|
|spanplus  1x fp16                  |   92.06|   98.31|1.07x|  0.0109|    0.30 GB|-    |-     |-    |-     | 2,216,867|

### spanplus_s
(1, 3, 480, 640) input, 1 warmup + 250 runs averaged
|Name|FPS|FPS ({abbr}`CL (channels_last)`)|{abbr}`CL (channels_last)` vs base|sec/img|VRAM ({abbr}`CL (channels_last)`)|PSNR (DF2K)|SSIM (DF2K)|PSNR (DIV2K)|SSIM (DIV2K)|Params|
|:-|-:|-:|-:|-:|-:|-:|-:|-:|-:|-:|
|spanplus_s  4x fp16                |   73.33|   67.22|0.92x|  0.0136|    1.55 GB|-    |-     |-    |-     |   687,707|
|spanplus_s  3x fp16                |   89.45|   86.85|0.97x|  0.0112|    0.89 GB|-    |-     |-    |-     |   684,067|
|spanplus_s  2x fp16                |  115.40|  121.08|1.05x|  0.0087|    0.42 GB|-    |-     |-    |-     |   681,467|
|spanplus_s  1x fp16                |  140.19|  140.83|1.00x|  0.0071|    0.20 GB|-    |-     |-    |-     |   679,907|

### spanplus_st
(1, 3, 480, 640) input, 1 warmup + 250 runs averaged
|Name|FPS|FPS ({abbr}`CL (channels_last)`)|{abbr}`CL (channels_last)` vs base|sec/img|VRAM ({abbr}`CL (channels_last)`)|PSNR (DF2K)|SSIM (DF2K)|PSNR (DIV2K)|SSIM (DIV2K)|Params|
|:-|-:|-:|-:|-:|-:|-:|-:|-:|-:|-:|
|spanplus_st  4x fp16               |  100.06|  112.20|1.12x|  0.0100|    0.32 GB|-    |-     |-    |-     | 2,236,728|
|spanplus_st  3x fp16               |  101.12|  111.51|1.10x|  0.0099|    0.31 GB|-    |-     |-    |-     | 2,227,635|
|spanplus_st  2x fp16               |  102.62|  113.23|1.10x|  0.0097|    0.30 GB|-    |-     |-    |-     | 2,221,140|
|spanplus_st  1x fp16               |  101.69|  112.33|1.10x|  0.0098|    0.30 GB|-    |-     |-    |-     | 2,217,243|

### spanplus_sts
(1, 3, 480, 640) input, 1 warmup + 250 runs averaged
|Name|FPS|FPS ({abbr}`CL (channels_last)`)|{abbr}`CL (channels_last)` vs base|sec/img|VRAM ({abbr}`CL (channels_last)`)|PSNR (DF2K)|SSIM (DF2K)|PSNR (DIV2K)|SSIM (DIV2K)|Params|
|:-|-:|-:|-:|-:|-:|-:|-:|-:|-:|-:|
|spanplus_sts  4x fp16              |  159.16|  170.84|1.07x|  0.0063|    0.23 GB|-    |-     |-    |-     |   693,160|
|spanplus_sts  3x fp16              |  158.59|  164.46|1.04x|  0.0063|    0.21 GB|-    |-     |-    |-     |   687,091|
|spanplus_sts  2x fp16              |  164.49|  169.55|1.03x|  0.0061|    0.21 GB|-    |-     |-    |-     |   682,756|
|spanplus_sts  1x fp16              |  163.83|  175.44|1.07x|  0.0061|    0.20 GB|-    |-     |-    |-     |   680,155|

### srformer
(1, 3, 480, 640) input, 1 warmup + 5 runs averaged
|Name|FPS|FPS ({abbr}`CL (channels_last)`)|{abbr}`CL (channels_last)` vs base|sec/img|VRAM ({abbr}`CL (channels_last)`)|PSNR (DF2K)|SSIM (DF2K)|PSNR (DIV2K)|SSIM (DIV2K)|Params|
|:-|-:|-:|-:|-:|-:|-:|-:|-:|-:|-:|
|srformer  4x bf16                  |    0.28|    0.28|1.02x|  3.6103|    3.65 GB|27.68|0.8311|-    |-     |10,543,503|
|srformer  3x bf16                  |    0.27|    0.28|1.01x|  3.6462|    3.62 GB|30.04|0.8865|-    |-     |10,580,431|
|srformer  2x bf16                  |    0.28|    0.27|0.97x|  3.5674|    3.61 GB|34.09|0.9449|-    |-     |10,395,791|
|srformer  1x bf16                  |    0.28|    0.28|1.01x|  3.5898|    3.60 GB|-    |-     |-    |-     |10,248,079|

### srformer_light
(1, 3, 480, 640) input, 1 warmup + 5 runs averaged
|Name|FPS|FPS ({abbr}`CL (channels_last)`)|{abbr}`CL (channels_last)` vs base|sec/img|VRAM ({abbr}`CL (channels_last)`)|PSNR (DF2K)|SSIM (DF2K)|PSNR (DIV2K)|SSIM (DIV2K)|Params|
|:-|-:|-:|-:|-:|-:|-:|-:|-:|-:|-:|
|srformer_light  4x bf16            |    1.01|    1.01|1.00x|  0.9933|    2.36 GB|26.67|0.8032|-    |-     | 6,894,948|
|srformer_light  3x bf16            |    1.00|    1.02|1.02x|  1.0033|    2.34 GB|28.81|0.8655|-    |-     | 6,860,907|
|srformer_light  2x bf16            |    1.00|    1.02|1.02x|  0.9973|    2.32 GB|32.91|0.9353|-    |-     | 6,836,592|
|srformer_light  1x bf16            |    1.02|    1.03|1.01x|  0.9822|    2.31 GB|-    |-     |-    |-     | 6,822,003|

### superultracompact
(1, 3, 480, 640) input, 1 warmup + 250 runs averaged
|Name|FPS|FPS ({abbr}`CL (channels_last)`)|{abbr}`CL (channels_last)` vs base|sec/img|VRAM ({abbr}`CL (channels_last)`)|PSNR (DF2K)|SSIM (DF2K)|PSNR (DIV2K)|SSIM (DIV2K)|Params|
|:-|-:|-:|-:|-:|-:|-:|-:|-:|-:|-:|
|superultracompact  4x fp16         |  736.22|  854.77|1.16x|  0.0014|    0.18 GB|-    |-     |-    |-     |    52,968|
|superultracompact  3x fp16         |  841.56| 1090.61|1.30x|  0.0012|    0.11 GB|-    |-     |-    |-     |    48,411|
|superultracompact  2x fp16         |  915.62| 1292.01|1.41x|  0.0011|    0.06 GB|-    |-     |-    |-     |    45,156|
|superultracompact  1x fp16         |  954.87| 1388.69|1.45x|  0.0010|    0.04 GB|-    |-     |-    |-     |    43,203|

### swin2sr_l
(1, 3, 480, 640) input, 1 warmup + 5 runs averaged
|Name|FPS|FPS ({abbr}`CL (channels_last)`)|{abbr}`CL (channels_last)` vs base|sec/img|VRAM ({abbr}`CL (channels_last)`)|PSNR (DF2K)|SSIM (DF2K)|PSNR (DIV2K)|SSIM (DIV2K)|Params|
|:-|-:|-:|-:|-:|-:|-:|-:|-:|-:|-:|
|swin2sr_l  4x bf16                 |    0.39|    0.40|1.02x|  2.5339|    3.37 GB|-    |-     |-    |-     |28,785,859|
|swin2sr_l  3x bf16                 |-       |-       |-       |-       |-       |-|-|-||-         |
|swin2sr_l  2x bf16                 |-       |-       |-       |-       |-       |-|-|-||-         |
|swin2sr_l  1x bf16                 |-       |-       |-       |-       |-       |-|-|-||-         |

### swin2sr_m
(1, 3, 480, 640) input, 1 warmup + 5 runs averaged
|Name|FPS|FPS ({abbr}`CL (channels_last)`)|{abbr}`CL (channels_last)` vs base|sec/img|VRAM ({abbr}`CL (channels_last)`)|PSNR (DF2K)|SSIM (DF2K)|PSNR (DIV2K)|SSIM (DIV2K)|Params|
|:-|-:|-:|-:|-:|-:|-:|-:|-:|-:|-:|
|swin2sr_m  4x bf16                 |    0.59|    0.59|0.99x|  1.6817|    2.51 GB|27.51|0.8271|-    |-     |12,239,283|
|swin2sr_m  3x bf16                 |    0.61|    0.61|1.00x|  1.6432|    2.49 GB|-    |-     |-    |-     |12,276,211|
|swin2sr_m  2x bf16                 |    0.61|    0.61|1.01x|  1.6375|    2.47 GB|33.89|0.9431|-    |-     |12,091,571|
|swin2sr_m  1x bf16                 |    0.62|    0.61|0.98x|  1.6136|    2.46 GB|-    |-     |-    |-     |11,943,859|

### swin2sr_s
(1, 3, 480, 640) input, 1 warmup + 5 runs averaged
|Name|FPS|FPS ({abbr}`CL (channels_last)`)|{abbr}`CL (channels_last)` vs base|sec/img|VRAM ({abbr}`CL (channels_last)`)|PSNR (DF2K)|SSIM (DF2K)|PSNR (DIV2K)|SSIM (DIV2K)|Params|
|:-|-:|-:|-:|-:|-:|-:|-:|-:|-:|-:|
|swin2sr_s  4x bf16                 |    0.98|    0.99|1.00x|  1.0172|    1.45 GB|-    |-     |-    |-     | 1,024,824|
|swin2sr_s  3x bf16                 |    1.00|    0.98|0.98x|  0.9963|    1.43 GB|-    |-     |-    |-     | 1,013,463|
|swin2sr_s  2x bf16                 |    0.99|    1.02|1.04x|  1.0140|    1.41 GB|32.85|0.9349|-    |-     | 1,005,348|
|swin2sr_s  1x bf16                 |    0.99|    0.99|0.99x|  1.0054|    1.40 GB|-    |-     |-    |-     | 1,000,479|

### swinir_l
(1, 3, 480, 640) input, 1 warmup + 5 runs averaged
|Name|FPS|FPS ({abbr}`CL (channels_last)`)|{abbr}`CL (channels_last)` vs base|sec/img|VRAM ({abbr}`CL (channels_last)`)|PSNR (DF2K)|SSIM (DF2K)|PSNR (DIV2K)|SSIM (DIV2K)|Params|
|:-|-:|-:|-:|-:|-:|-:|-:|-:|-:|-:|
|swinir_l  4x bf16                  |    0.45|    0.46|1.02x|  2.2096|    3.56 GB|-    |-     |-    |-     |28,013,059|
|swinir_l  3x bf16                  |    0.46|    0.47|1.02x|  2.1709|    3.52 GB|-    |-     |-    |-     |27,976,131|
|swinir_l  2x bf16                  |    0.46|    0.46|1.00x|  2.1683|    3.52 GB|-    |-     |-    |-     |27,976,131|
|swinir_l  1x bf16                  |    0.46|    0.47|1.01x|  2.1560|    3.52 GB|-    |-     |-    |-     |27,976,131|

### swinir_m
(1, 3, 480, 640) input, 1 warmup + 5 runs averaged
|Name|FPS|FPS ({abbr}`CL (channels_last)`)|{abbr}`CL (channels_last)` vs base|sec/img|VRAM ({abbr}`CL (channels_last)`)|PSNR (DF2K)|SSIM (DF2K)|PSNR (DIV2K)|SSIM (DIV2K)|Params|
|:-|-:|-:|-:|-:|-:|-:|-:|-:|-:|-:|
|swinir_m  4x bf16                  |    0.70|    0.69|0.98x|  1.4329|    2.64 GB|27.45|0.8254|-    |-     |11,900,199|
|swinir_m  3x bf16                  |    0.69|    0.69|1.00x|  1.4557|    2.62 GB|29.75|0.8826|-    |-     |11,937,127|
|swinir_m  2x bf16                  |    0.67|    0.69|1.02x|  1.4863|    2.60 GB|33.81|0.9427|-    |-     |11,752,487|
|swinir_m  1x bf16                  |    0.66|    0.62|0.95x|  1.5257|    2.59 GB|-    |-     |-    |-     |11,604,775|

### swinir_s
(1, 3, 480, 640) input, 1 warmup + 5 runs averaged
|Name|FPS|FPS ({abbr}`CL (channels_last)`)|{abbr}`CL (channels_last)` vs base|sec/img|VRAM ({abbr}`CL (channels_last)`)|PSNR (DF2K)|SSIM (DF2K)|PSNR (DIV2K)|SSIM (DIV2K)|Params|
|:-|-:|-:|-:|-:|-:|-:|-:|-:|-:|-:|
|swinir_s  4x bf16                  |    1.06|    1.05|0.99x|  0.9405|    1.55 GB|26.47|0.7980|-    |-     |   929,628|
|swinir_s  3x bf16                  |    1.04|    1.06|1.01x|  0.9572|    1.53 GB|28.66|0.8624|-    |-     |   918,267|
|swinir_s  2x bf16                  |    1.04|    1.05|1.01x|  0.9658|    1.51 GB|32.76|0.9340|-    |-     |   910,152|
|swinir_s  1x bf16                  |    0.91|    0.91|1.01x|  1.1039|    1.50 GB|-    |-     |-    |-     |   905,283|

### tscunet
(1, 3, 480, 640) input, 1 warmup + 5 runs averaged
|Name|FPS|FPS ({abbr}`CL (channels_last)`)|{abbr}`CL (channels_last)` vs base|sec/img|VRAM ({abbr}`CL (channels_last)`)|PSNR (DF2K)|SSIM (DF2K)|PSNR (DIV2K)|SSIM (DIV2K)|Params|
|:-|-:|-:|-:|-:|-:|-:|-:|-:|-:|-:|
|tscunet  4x fp16                   |-       |-       |-       |-       |-       |-|-|-||-         |
|tscunet  3x fp16                   |-       |-       |-       |-       |-       |-|-|-||-         |
|tscunet  2x fp16                   |-       |-       |-       |-       |-       |-|-|-||-         |
|tscunet  1x fp16                   |-       |-       |-       |-       |-       |-|-|-||-         |

### ultracompact
(1, 3, 480, 640) input, 1 warmup + 250 runs averaged
|Name|FPS|FPS ({abbr}`CL (channels_last)`)|{abbr}`CL (channels_last)` vs base|sec/img|VRAM ({abbr}`CL (channels_last)`)|PSNR (DF2K)|SSIM (DF2K)|PSNR (DIV2K)|SSIM (DIV2K)|Params|
|:-|-:|-:|-:|-:|-:|-:|-:|-:|-:|-:|
|ultracompact  4x fp16              |  247.53|  362.11|1.46x|  0.0040|    0.18 GB|-    |-     |-    |-     |   325,488|
|ultracompact  3x fp16              |  262.17|  397.15|1.51x|  0.0038|    0.11 GB|-    |-     |-    |-     |   313,371|
|ultracompact  2x fp16              |  265.03|  429.91|1.62x|  0.0038|    0.09 GB|-    |-     |-    |-     |   304,716|
|ultracompact  1x fp16              |  266.31|  434.57|1.63x|  0.0038|    0.09 GB|-    |-     |-    |-     |   299,523|
