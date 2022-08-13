# IMAGE SEGMENTATION FOR OVERHEAD DRONE IMAGERY

------

## Main Usage

This package includes a range of functions for processing large orthomosaic raster images into a dataset for machine learning, and training neural networks for semantic segmentation of these images.

Relevant functions are included in the "segmenter" folder, and a few scripts for specific uses are in "scripts".

If you have a large image that you want to segment using one of the provided models, simply call `stitching.py` as follows:

```bash
python stitching.py -i PATH/TO/IMAGE -m PATH/TO/MODEL -o OUTPUT/DIRECTORY
```

This script will tile your image into the appropriate size, run it through the specified network, stitch the outputs together, and save the results.

## Pretrained Models

There are a number of pre-trained models in the `models/` folder that can be loaded for use in any of the available scripts. Most notable are `model_new_retrained.hdf5` used to locate glare in overhead drone imagery in Giles et al. 2021. And `model_mres_3class.hdf5`, used to differentiate bleached and unbleached coral, as well as sun glint. Both models should function in any of the analysis scripts (aside from predict_multiclass, which will only work with the multiclass models).

## Other Scripts

You can train new or existing models using train_model.py, and train_model_finetune.py, where the finetune script freezes the early layers of the model.

You can obtain detailed statistics on a model's performance using predict.py, using a set of test data with known classifications.

You can examine your input to the data pipeline (the preprocessing before the network) using `examine_bigtiles.py` and the output using `examine_dataset.py`. These scripts plot your images so you can check for image alignment and correct classification of training data.

### EDIT: (30/06/22)

This package includes pretrained models for the segmentation of coral features (null, coral, bleached coral, sun glint). To use this model, place any image you'd like to segment into `input/test_images` and run `python predict_multiclass.py -i INPUT_PATH -o OUTPUT_PATH` to start. Alternatively, you can test this script out by using predefined defaults with `python predict_multiclass.py`.

A second script added into "scripts" is the model finetuner, which can be run with `python train_model_finetune.py -i INPUT_PATH`. Please take a look into the script itself to make adjustments variables, such as training image sizes, trainable layers and etc. The default points to the predefined MRESUnet used for the multiclass segmentation. To effectively utilise this script, you would need to tile your images using the tile_images script in "segmenter" or via another method of your choice. Make sure that the mask is sparse encoded (each pixel represents a class) - the dataloader will one-hot encode for you.
