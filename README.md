# T5-Sentinel-public

Release repo for our work "Token Prediction as Implicit Classification to Identify LLM-Generated Text"

## Requirement

As shown in `requirements.txt` in the root directory.

## Evaluate

1. Run `./data/download.py` to automatically download Dataset & model checkpoint
2. Run the following files in need
   1. `./evaluator/calc/calc_accuracy.py` to calculate the accuracy under different settings for each module
   2. `./evaluator/interpret/integrated_gradient.ipynb` to calculate the integrated gradient for samples
   3. `./evaluator/interpret/sample_pca.py` to calculate the PCA analysis for hidden layers of the test subset
   4. `./evaluator/plot/*.py` to generate plots of related metrics (confusion matrix, roc, det, etc.)

## Train

1. Use the `./detector/t5/arbitrary/__main__.py` to train the T5-Sentinel Model

    (The detailed hyperparameter setup we used for training the T5-Sentinel model in paper is presented in `settings_0613_full.yaml`)

2. Use the `./detector/t5/arbitrary_hidden/__main__.py` to train the T5-Hidden Model

