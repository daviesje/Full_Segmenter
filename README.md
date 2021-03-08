# IMAGE SEGMENTATION FOR OVERHEAD DRONE IMAGERY

This package includes a range of functions for processing large orthomosaic raster images into a dataset for machine learning, and training neural networks for semantic segmentation of these images.

Relevant functions are included in the "segmenter" folder, and a few scripts for specific uses are in "scripts".

If you have a large image that you want to segment using one of the provided models, simply call `stitching.py` as follows:

```bash
python stitching.py -i PATH/TO/IMAGE -m PATH/TO/MODEL -o OUTPUT/DIRECTORY
```

This script will tile your image into the appropriate size, run it through the specified network, stitch the outputs together, and save the results.

## stubs

you can train new models using train_model.py, and provide detailed statistics on a model's performance using predict.py

you can examine your input to the data pipeline (the preprocessing before the network) using `examine_bigtiles.py` and the output using `examine_dataset.py`
