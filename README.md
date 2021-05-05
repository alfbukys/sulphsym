# Simplified kinetic model of the sulphur assimilation pathway in Arabidopsis.
#### By Alfredas Bukys, 2021

## The code that was used to create the model ensemble is in `Parameterisation_relaxed_fit.py.` 
This requires the accessory files provided in `Accessory.zip.`
The amount of models to be generated can be determined by adjusting the range of the for loop that runs the differential evolution parameter fitting.
50 parameter sets have already been generated and are stored as .txt files in `Model ensemble.zip`

## Data analysis can be accessed via the Jupyter Notebook `Data analysis.ipynb`
This requires parameter sets to be generated beforehand. While the calculations can be re-ran, we already include .npy files for all the major calculations needed for the same model ensemble we provide stored within `Precalculated.zip`
