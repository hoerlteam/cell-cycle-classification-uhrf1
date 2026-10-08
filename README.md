# CNN-based Cell cycle / S-phase classification

This repository contains code for the paper **[TODO: paper title, authors, preprint link]**.

In our study, we first performed instance segmentation of nuclei using Cellpose, followed by classification of detected cells into G1/G2 or various sub-phases of the S-phase based on PCNA signal using a small CNN model. Many parts (notebook "recipes") of the pipeline are implemented in a generic way and can be reused, e.g. for other per-object classification tasks with a retrained model. All notebooks are designed to work with one directory of experimental data and will typically save their results in a subdirectory, so they can be used by the next step.

## Pipeline overview

1. **Re-save nd2 image data as TIFF:** For subsequent steps, we first re-save the nd2 image data as TIFF, splitting multi-position images into single files. Optionally, we can also perform flatfield correction of the image and save a corrected version at this step.
    
    - use `1a_resave_nd2_as_tiff.ipynb` to re-save without illumination correction
    - use `1b_resave_with_illumination_correction.ipynb` to re-save with illumination correction (files used can be found in `FlatfieldCorrection`)

2. **Detect cells using Cellpose:** Next, we use Cellpose to perform instance segmentation of nuclei in the images. The notebook `2_cellpose_segmentation.ipynb` can be used to perform this for all files in a folder. We used Cellpose 3 with a model finetuned on manually annotated nuclei of ESCs. The model weights are available under `TODO: models/esc`.

3. **Per-cell classification & feature extraction:**  Next, we perform the S-phase classification of detected cells using a small CNN model plus extract simple features.

    - use `3a_classify_regions.ipynb` to classify cells, creating a CSV file with predicted class (e.g. S-phase subphase) and probability for each class. The model information and weights can be found in `TODO: models/`.
    - use `3b_simple_regionprops.ipynb` to use scikit-image's `regionprops_table` to extract shape / intensity features for all cells

4. **Analysis of combined results:** Finally, we use `4_analysis_cellcycle_rprops` to perform statistical analysis on the combined data frames from step 3 and produce plots for the manuscript.

## Training the CNN model

The training code for our ResNet models can be found under `resnet/train_resnet.ipynb`, napari-based annotation to create new training datasets can be done with `annotate_cellcycle.ipynb` 

