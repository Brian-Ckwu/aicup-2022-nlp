import os
import sys
os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.append("../")

from pathlib import Path
from argparse import Namespace, ArgumentParser

import pandas as pd

import torch
from transformers import logging
logging.set_verbosity_error()

from data import BertEncoderNetDataset
from model import BertEncoderNet
from utils import ensemble_model_outputs
from trainer import BertEncoderNetTrainer
from arguments import PreprocessArgs, MyTrainingArguments
from preprocess import Preprocessor
from postprocess import Postprocessor

def make_prediction_file(args: Namespace) -> None:
    # Configuration
    print("Setting configurations...")
    preprocess_args = PreprocessArgs(
        use_nltk=False,
        model_tokenizer_name=args.model_name,
        input_scheme="qrs",
        output_scheme="q'r'",
        labeling_scheme="IO1",
        filter_long_text=False
    )

    train_args = MyTrainingArguments(
        num_train_epochs=4,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=4,
        learning_rate=3e-5,
        lr_scheduler_type="linear",
        warmup_ratio=0.0,
        seed=42,

        evaluation_strategy="epoch",
        eval_steps=None,
        logging_strategy="epoch",
        logging_first_step=True,
        output_dir=f"../experiments/notebook_run",
        overwrite_output_dir=True,
        save_strategy="epoch",
        save_total_limit=5,

        fp16=False,
        load_best_model_at_end=False,
        metric_for_best_model=None,
        greater_is_better=None,
        
        device_str=args.device,
    )

    # Data
    # load data
    print("Loading data...")
    data = pd.read_csv(args.data_path) # NOTE: for validation purpose .drop(["Unnamed: 6", "total no.: 7987"], axis=1)
    data["q'"] = ['' for _ in range(len(data))]
    data["r'"] = ['' for _ in range(len(data))]

    # preprocess data
    preprocessor = Preprocessor(preprocess_args)
    dataset = BertEncoderNetDataset(data, preprocessor, do_eval=True)

    # Model
    print("Loading model checkpoint...")
    model = BertEncoderNet(
        model_name=args.model_name, 
        num_tags=len(Preprocessor.scheme2tags[preprocess_args.labeling_scheme]),
        multitask=(preprocess_args.output_scheme == "sq'r'"),
    )

    # Trainer
    trainer = BertEncoderNetTrainer(
        model=model,
        args=train_args,
        data_collator=dataset.collate_fn,
        train_dataset=dataset,
        eval_dataset=dataset
    )

    # Inference
    print("Making inference...")

    if args.ensemble:
        print(f"Ensembling using strategy '{args.ensemble_strategy}'...")
        model_ckpt_paths = args.ckpt_paths_file.read_text().split('\n')
        pred_token_mask_l = ensemble_model_outputs(
            dataset,
            trainer,
            model,
            model_ckpt_paths,
            args.ensemble_strategy,
            args.device
        )
    else:
        model.load_state_dict(torch.load(args.ckpt_path / "pytorch_model.bin", map_location=args.device))
        test_outputs = trainer.predict(dataset)
        pred_token_mask_l = test_outputs.predictions.argmax(axis=-1)
    
    # Make the final prediction DataFrame
    print("Making the final prediction DataFrame...")
    pred_sents = Postprocessor.predict_sents(dataset, pred_token_mask_l)
    pred_df = pd.DataFrame(data={
        "id": dataset.p_data.id.unique().tolist(),
        **pred_sents
    })

    # Save the prediction file
    print("Saving the prediction file...")
    pred_df = pred_df.rename({"q'": 'q', "r'": 'r'}, axis=1)
    pred_df.loc[:, ['q', 'r']] = pred_df[['q', 'r']].applymap(lambda s: '"' + str(s).strip('"') + '"')
    pred_df.to_csv(Path("./csv_for_submission") / args.file_name, quotechar='"', header=True, index=False, encoding="utf-8")

    return None

def parse_args() -> Namespace:
    parser = ArgumentParser()

    parser.add_argument(
        "--model_name",
        type=str,
        required=True
    )
    parser.add_argument(
        "--ckpt_path",
        type=Path,
        default=""
    )
    parser.add_argument(
        "--ensemble",
        action="store_true"
    )
    parser.add_argument(
        "--ensemble_strategy",
        type=str,
        default="logits"
    )
    parser.add_argument(
        "--ckpt_paths_file",
        type=Path,
        default=""
    )
    parser.add_argument(
        "--file_name",
        type=str,
        required=True
    )
    parser.add_argument(
        "--data_path",
        type=Path,
        default="./dataset/submission_set/Batch_answers - test_data(no_label).csv"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0"
    )

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    make_prediction_file(args)