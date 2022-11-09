"""
    Import the packages
"""
import pandas as pd

from utils import load_json
from arguments import Args, PreprocessArgs, FileArgs, ModelArgs, OptimizationArgs, ExperimentArgs
from preprocess import Preprocessor

class Experiment(object):
    debug_nsamples = 1000

    def __init__(self, args: ExperimentArgs):
        # Initialize the objects that will be used in this experiment
        self.args = args
        self.split_ids = load_json(file=args.file_args.split_ids_path)
        self.preprocessor = Preprocessor(args=args.preprocess_args)

    def run(self, debug_mode: bool): # debug_mode --> forward a small proportion of samples (e.g., 10 - 100)
        # Load data
        data = pd.read_csv(self.args.file_args.data_path).drop(["Unnamed: 6", "total no.: 7987"], axis=1)
        if debug_mode:
            data = data[:self.debug_nsamples]
        
        # Preprocess data (tokenization, sequence labeling, and formatting)
        print("Preprocessing the data...") # TODO: change to logging
        p_data = self.preprocessor(data)

        # Split into train/valid/test by ID
        train_data, valid_data, test_data = [p_data[p_data.id.isin(self.split_ids[split])] for split in ["train", "valid", "test"]]
        print(f"Sample size: train = {len(train_data)} / valid = {len(valid_data)} / test = {len(test_data)}")
        
        # Load model (new or from a checkpoint)

        # Optimization (Trainer?): training + validation
            # log to wandb

        # Testing
            # post-processing
            # LCS

        raise NotImplementedError

# Experiment Arguments

exp_args = ExperimentArgs(
    file_args=FileArgs(
        data_path="./dataset/train.csv",
        split_ids_path="./dataset/splitIds__splitBy-id_stratifyBy-s_train-0.6_valid-0.2_test-0.2_seed-42.json"
    ),
    preprocess_args=PreprocessArgs(
        use_nltk=True,
        model_tokenizer_name="bert-base-uncased",
        input_scheme="qr",
        output_scheme="q'r'",
        labeling_scheme="IO1"
    ),
    model_args=ModelArgs(
        model_type="bert",
        model_name="bert-base-uncased"
    ),
    opt_args=OptimizationArgs(
        lr=3e-5,
        nepochs=5,
        train_batch_size=16,
        eval_batch_size=16,
        grad_accum_steps=1,
        optimizer="AdamW",
        scheduler=None,
        stopping_strategy=None
    ),
)

if __name__ == "__main__":
    exp = Experiment(exp_args)
    exp.run(debug_mode=True) # -> save experiment arguments and results to a specified directory