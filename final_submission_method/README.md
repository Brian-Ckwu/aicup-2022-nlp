# AICUP 2022 NLP - Final solution (First Prize)
The natural language explanatory sequence labeling contest of the AICUP 2022 competition.

## Overview
Our final solution is based on a question-answering framework.<br />
We trained models for q' and r' respectively and find the best model pair for submission.

## How to run
### Preprocess
Run cells of **Initialization**, **Functions**, **Tokenizer**
### Train
Run cells of **Model / Collator / Trainer**
### Predict for Evaluation
Run cells of **Predict Answer from model checkpoints**<br />
Remember to set <code>real_predict_test = 0</code>
### Evaluation
Run cells of **Evaluation**, **Model combination test**
### Predict for Test
Run cells of **Predict Answer from model checkpoints**<br />
Remember to set <code>real_predict_test = 1</code>

## Acknowledgement
We used pretrained-model from: <https://huggingface.co/janeel/muppet-roberta-base-finetuned-squad/><br />
Part of the code is adjusted from: <https://github.com/huggingface/notebooks/blob/main/examples/question_answering.ipynb>