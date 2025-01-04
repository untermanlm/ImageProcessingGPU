# ImageProcessingGPU

This project compares the runtime performance of different image processing techniques on the CPU vs the GPU in C++. These image processing techniques, which were implemented from scratch, include binarization via Sauvola's methodology and segmentation via the 4-connectivity two-pass connected components labeling algorithm.

![Image processing pipeline given stream of images]([https://github.com/untermanlm/ImageProcessingGPU/blob/main/image_segmentation_pipeline.png])

**Abstract:**
This report summarizes the results achieved for
improving the serialized versions of different image processing
algorithms using NVIDIA’s CUDA architecture. These different
image processing algorithms include algorithms that are used
to 1) convert three-channel RGB images into single-channel
grayscale images (Grayscale), 2) binarize grayscale images via
Sauvola’s adaptive thresholding, 3) return the inverse of a binary
image (BitwiseNot), 4) pad an image, and 5) find the connected
components from within the image (Connected Component
Labeling, CCL). When ranking each operation by its range
of runtimes in increasing order, it is clear that BitwiseFlip
< Padding < Grayscale < Sauvola Thresholding < CCL as
a result of the Grayscale algorithm’s requirement of handling
the pixels of an RGB image as opposed to a grayscale image,
Sauvola Thresholding’s additional requirement of calculating the
local mean and standard deviation to devise a local threshold
value, and parallel CCL’s reliance upon serial code to complete
the first pass and resolve step(s) of the 4-connectivity two-pass
connected components algorithm. Furthermore, the most scalable
of the parallel image-processing implementations is the Sauvola
Thresholding algorithm, once again as a result of its parallel time
complexity of O(W2) instead of O(1) like the other algorithms.
