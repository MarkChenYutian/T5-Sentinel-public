<img width="1204" alt="image" src="https://github.com/MarkChenYutian/T5-Sentinel-public/assets/47029019/ebbab6c4-e1c4-4cca-9df4-78a72fe79135">


**Release repo for our work "Token Prediction as Implicit Classification to Identify LLM-Generated Text"**

## Abstract

This paper introduces a novel approach for identifying the possible large language models (LLMs) involved in text generation. Instead of adding an additional classification layer to a base LM, we reframe the classification task as a next-token prediction task and directly fine-tune the base LM to perform it. We utilize the Text-to-Text Transfer Transformer (T5) model as the backbone for our experiments. We compared our approach to the more direct approach of utilizing hidden states for classification. Evaluation shows the exceptional performance of our method in the text classification task, highlighting its simplicity and efficiency. Furthermore, interpretability studies on the features extracted by our model reveal its ability to differentiate distinctive writing styles among various LLMs even in the absence of an explicit classifier. We also collected a dataset named OpenLLMText, containing approximately 340k text samples from human and LLMs, including GPT3.5, PaLM, LLaMA, and GPT2.

## Evaluation Result Overview

<img width="1413" alt="image" src="https://github.com/MarkChenYutian/T5-Sentinel-public/assets/47029019/94bc13a8-e164-4e5f-ba75-91139b05c167">


## Requirement

Run `pip install -r requirements.txt` to install dependencies.

> Note that the baseline model proposed by Solaiman et al. requires a legacy version of library `transformers`, the detailed environment requirements
> for baseline model is placed in [here](https://github.com/MarkChenYutian/T5-Sentinel-public/blob/main/detector/solaiman_classifier/solaiman_requirements.txt)

## Evaluate

1. Run `./data/download.py` to automatically download dataset & model checkpoints
2. Run the following files in need
   1. `./evaluator/calc/calc_accuracy.py` to calculate the accuracy under different settings for each module
   2. `./evaluator/interpret/integrated_gradient.ipynb` to calculate the integrated gradient for samples
   3. `./evaluator/interpret/sample_pca.py` to calculate the PCA analysis for hidden layers of the test subset
   4. `./evaluator/plot/*.py` to generate plots of related metrics (confusion matrix, roc, det, etc.)

Note that python files are in module, so to use `./evaluator/calc/calc_accuracy.py`, you need to run `python3 -m evaluator.calc.calc_accuracy`.

## Train

1. Use the `./detector/t5/arbitrary/__main__.py` to train the T5-Sentinel Model

    (The detailed hyperparameter setup we used for training the T5-Sentinel model in paper is presented in `settings_0613_full.yaml`)

2. Use the `./detector/t5/arbitrary_hidden/__main__.py` to train the T5-Hidden Model

