import os
import datetime

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import TrainingArguments, IntervalStrategy, Trainer

# from transformers import Traniner

from experiments.dataset.tolkien_dataset_builder import TolkienDatasetBuilder

# TODO: instrument this code with flwr
# TODO: Refer to https://github.com/huggingface/transformers/blob/v4.30.0/src/transformers/integrations.py#L670
#   Implement a custom callback for HF trainer.

# from src.training.trainer import Trainer

os.environ["WANDB_DISABLED"] = "true"

# MODEL_NAME = "EleutherAI/gpt-neo-125M"
# MODEL_NAME = "facebook/opt-350m"

from argparse import ArgumentParser


class TrainingSessionArgParser:
    def __init__(self):
        self.parser = ArgumentParser()
        self.add_args()

    def add_args(self):
        self.parser.add_argument(
            "--filename",
            type=str,
            default="experiments/dataset/lotr-paragraphs.json",
            help="Path to the dataset",
        )
        self.parser.add_argument(
            "--model_name",
            type=str,
            default="EleutherAI/gpt-neo-125M",
            help="HuggingFace CausalLM pre-trained model to be fine tuned",
        )
        self.parser.add_argument(
            "--epochs",
            type=int,
            default=3,
            help="Number of epochs to train the model",
        )
        self.parser.add_argument(
            "--batch_size",
            type=int,
            default=32,
            help="Batch size to use for training",
        )
        self.parser.add_argument(
            "--learning_rate",
            type=float,
            default=2e-5,
            help="Learning rate to use for training",
        )

    def parse_args(self):
        return self.parser.parse_args()


class TrainingSession:
    def __init__(self, args):
        self.args = args

        self.bos_token = "<|startoftext|>"
        self.eos_token = "<|endoftext|>"
        self.pad_token = "<|pad|>"
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.args.model_name,
            bos_token=self.bos_token,
            eos_token=self.eos_token,
            pad_token=self.pad_token,
        )

    def run(self):
        self.create_datasets()
        # self.create_dataloaders()
        self.create_model()
        self.create_trainer()
        self.trainer.train()

    def create_datasets(self):
        builder = TolkienDatasetBuilder(self.args.filename, self.args.model_name)
        self.train_dataset, self.val_dataset = builder.build_datasets()

    def create_dataloaders(self):
        self.train_dataloader = DataLoader(
            self.train_dataset, batch_size=self.args.batch_size, shuffle=True
        )
        self.val_dataloader = DataLoader(
            self.val_dataset, batch_size=self.args.batch_size
        )

    def create_model(self):
        self.model = AutoModelForCausalLM.from_pretrained(self.args.model_name).cuda()
        self.model.resize_token_embeddings(len(self.tokenizer))

    def create_trainer(self):
        # self.trainer = Trainer(
        #     model=self.model,
        #     train_dataloader=self.train_dataloader,
        #     val_dataloader=self.val_dataloader,
        #     learning_rate=self.args.learning_rate,
        #     epochs=self.args.epochs,
        # )
        time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        training_args = TrainingArguments(
            output_dir=f"./results/{time}",
            num_train_epochs=self.args.epochs,
            per_device_train_batch_size=16,
            # per_device_eval_batch_size=8,
            # logging_steps=5000,
            # logging_dir="./logs",
            # save_strategy=IntervalStrategy.NO,
            # warmup_steps=100,
            # weight_decay=0.01,
        )
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            data_collator=lambda data: {
                "input_ids": torch.stack([f["input_ids"] for f in data]),
                "attention_mask": torch.stack([f["attention_mask"] for f in data]),
                "labels": torch.stack([f["input_ids"] for f in data]),
            },
        )


if __name__ == "__main__":
    args = TrainingSessionArgParser().parse_args()
    session = TrainingSession(args)
    session.run()
