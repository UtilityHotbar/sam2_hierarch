# Hierarchical Clustering of Latent Representations
Using [SAM2](https://github.com/facebookresearch/segment-anything-2) and [CLIP](https://huggingface.co/transformers/v4.8.0/model_doc/clip.html#clipvisionmodel). Highly experimental code released under MIT license.

Modules you'll need (file issue if you spot any I missed):

* `transformer`
* `sam2`
* `torch`
* `scipy`
* `numpy`

## Files in this repository
* `hierarch2.py` - Hierarchical clustering of CLIP object embeddings using `scipy` This is not online clustering, the images come in and are sorted as a batch. Prepare images with `img_preprocess.py`.
* `sam_hierarch.py` - Hierarchical clustering of CLIP object embeddings extracted from images of scenes via SAM2. This is online clustering, and will output a folder with the "world model" of the system (each folder is what the system thinks is the same object). Prepare images with `sam_preprocess.py` (this can be done as images come in, its just to avoid repeated high GPU usage from SAM2 cutting up the same image over and over again).
