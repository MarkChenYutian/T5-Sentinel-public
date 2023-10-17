# T5-Sentinel-public

Release repo for our work "Token Prediction as Implicit Classification to Identify LLM-Generated Text"

## Requirement

As shown in `requirements.txt` in the root directory.

## Evaluate

1. Download the checkpoints `0622.hidden.a.pt`, `t5-small.0613.a.pt` and `solaiman-detector-base.pt` and place in the `./data/checkpoint` directory. The models can be found in Release page of this repository.
2. Download the OpenLLMText dataset in the `./data/split` directory

3. Run the following files
   1. `./evaluator/calc/calc_accuracy.py` to calculate the accuracy under different settings for each module
   2. `./evaluator/interpret/integrated_gradient.ipynb` to calculate the integrated gradient for samples
   3. `./evaluator/interpret/sample_pca.py` to calculate the PCA analysis for hidden layers of the test subset
   4. `./evaluator/plot/*.py` to generate plots of related metrics (confusion matrix, roc, det, etc.)

## Train

1. Use the `./detector/t5/arbitrary/__main__.py` to train the T5-Sentinel Model

    (The detailed hyperparameter setup we used for training the T5-Sentinel model in paper is presented in `settings_0613_full.yaml`)

2. Use the `./detector/t5/arbitrary_hidden/__main__.py` to train the T5-Hidden Model

