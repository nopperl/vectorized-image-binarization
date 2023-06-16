# Vectorized Image Binarization

A vectorized implementation of the image binarization algorithm of [Su et al. (2010)][1] using numpy. Example images from the [DIBCO2009][2] dataset are provided in `dibco2009`. The dataset consists of scans of handwritten and printed documents.

## Usage
### Split image into channels (optional)
If the color channels are neatly orthogonal, it is possible to binarize each color channel individually. This is not recommended for the provided [DIBCO2009][2] dataset.

```
./main.py split $IMAGE_FILE
```

### Binarize image
Binarize an image using the [Su et al. (2010)][1] algorithm. The resulting file will be stored next to the input file.
```
./main.py binarize dibco2009/DIBC02009_Test_images-handwritten/dibco_img0004.tif
```

### Evaluate
It is also possible to evaluate the binarization algorithm on either the handwritten or the printed portion of the provided [DIBCO2009][2] dataset. The below command shows the `F1` and `PSNR` metrics on the handwritten documents.

```
./main.py evaluate dibco2009/DIBC02009_Test_images-handwritten
```

Again, the parameters of the algorithm can be changed by setting command line parameters (see `main.py evaluate -h`). 

A comparison of the implementation with default parameters and the values stated in [Su et al. (2010)][1] is given below:


| Algorithm | F1 (%) | PSNR |
| --- | --- | --- |
| [Su et al. (2010)][1] | 89.93 | 19.94 |
| This | 86.01 | 18.37 |


[1]: https://doi.org/10.1145/1815330.1815351
[2]: https://doi.org/10.1109/ICDAR.2009.246
