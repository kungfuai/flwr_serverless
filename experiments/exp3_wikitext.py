import os
import datetime
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, is_torch_tpu_available
from transformers import TrainingArguments, IntervalStrategy, Trainer, TrainerCallback
from transformers import DataCollatorForLanguageModeling
import datasets
import evaluate
from flwr.common import (
    Parameters,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
import wandb

# from transformers import Traniner

from experiments.dataset.tolkien_dataset_builder import TolkienDatasetBuilder

# TODO: instrument this code with flwr
# TODO: Refer to https://github.com/huggingface/transformers/blob/v4.30.0/src/transformers/integrations.py#L670
#   Implement a custom callback for HF trainer.

# from src.training.trainer import Trainer

os.environ["WANDB_DISABLED"] = "true"

from argparse import ArgumentParser


class FederatedLearningCallback(TrainerCallback):
    def __init__(self, args, federated_node, **kwargs):
        super().__init__(args, **kwargs)
        self.node = federated_node

    def on_epoch_end(self, args, state, control, **kwargs):
        # ??
        epoch = state.epoch
        metrics = state.metrics

        node_id = self.node.node_id
        # get model weights
        torch_model = self.model
        model_weights = torch_model.parameters()
        params: Parameters = ndarrays_to_parameters(model_weights)
        updated_params, updated_metrics = self.node.update_parameters(
            params,
            num_examples=self.num_examples_per_epoch,
            epoch=epoch,
            metrics=metrics,
        )
        self._federated_metrics = updated_metrics

        if updated_params is not None:
            # set model weights
            for param, updated_param in zip(model_weights, updated_params):
                param.data = updated_param.data  # ?


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


@dataclass
class TrainingSession:
    model_name: str = "EleutherAI/gpt-neo-125M"
    # model_name: str = "EleutherAI/pythia-14M"
    epochs: int = 3
    batch_size: int = 16
    lr: float = 5e-5
    context_length: int = 128

    def __post_init__(self):
        # self.bos_token = "<|startoftext|>"
        # self.eos_token = "<|endoftext|>"
        # self.pad_token = "<|pad|>"
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            # bos_token=self.bos_token,
            # eos_token=self.eos_token,
            # pad_token=self.pad_token,
        )

    def run(self):
        if self.args.track:
            with wandb.init(project="wikitext"):
                wandb.config.update(self.__dict__)
                self._run()
        else:
            self._run()

    def _run(self):
        self.create_datasets()
        self.create_model()
        self.create_trainer()
        self.trainer.train()

    def create_datasets(self):
        raw_datasets = datasets.load_dataset(
            "wikitext",
            "wikitext-103-v1",
            split=["train[:100000]", "validation[:1000]"],
            # streaming=True
        )
        print("raw datasets:")
        print(raw_datasets)
        context_length = self.context_length

        def tokenize(element):
            outputs = self.tokenizer(
                element["text"],
                truncation=True,
                max_length=context_length,
                return_overflowing_tokens=False,
                return_length=True,
            )
            input_batch = []
            for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
                if length == context_length:
                    input_batch.append(input_ids)
            return {"input_ids": input_batch}

        tokenized_train = raw_datasets[0].map(
            tokenize, batched=True, remove_columns=raw_datasets[0].column_names
        )
        tokenized_test = raw_datasets[1].map(
            tokenize, batched=True, remove_columns=raw_datasets[1].column_names
        )

        print("tokenized:")
        print(tokenized_train)
        print("iterating:")
        for x in tokenized_train:
            print(x)
            break

        self.train_dataset = tokenized_train
        self.val_dataset = tokenized_test

    def create_model(self):
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name).cuda()
        self.model.resize_token_embeddings(len(self.tokenizer))

    def create_trainer(self):
        time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        training_args = TrainingArguments(
            learning_rate=self.lr,
            output_dir=f"./results/{time}",
            num_train_epochs=self.epochs,
            per_device_train_batch_size=self.batch_size,
            evaluation_strategy="steps",
            logging_strategy="steps",
            gradient_accumulation_steps=10,
            eval_steps=50,
            logging_steps=50,
            save_strategy=IntervalStrategy.NO,
            # evaluation_strategy="epoch",
            # logging_strategy="epoch",
            report_to=["wandb"],
            eval_delay=0,
            per_device_eval_batch_size=self.batch_size,
            eval_accumulation_steps=10,
            # per_device_eval_batch_size=8,
            # logging_steps=5000,
            # logging_dir="./logs",
            # save_strategy=IntervalStrategy.NO,
            # warmup_steps=100,
            # weight_decay=0.01,
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        data_collator = DataCollatorForLanguageModeling(self.tokenizer, mlm=False)

        def preprocess_logits_for_metrics(logits, labels):
            if isinstance(logits, tuple):
                # Depending on the model and config, logits may contain extra tensors,
                # like past_key_values, but logits always come first
                logits = logits[0]
            return logits.argmax(dim=-1)

        metric = evaluate.load("accuracy")

        def compute_metrics(eval_preds):
            preds, labels = eval_preds
            # preds have the same shape as the labels, after the argmax(-1) has been calculated
            # by preprocess_logits_for_metrics but we need to shift the labels
            labels = labels[:, 1:].reshape(-1)
            preds = preds[:, :-1].reshape(-1)
            return metric.compute(predictions=preds, references=labels)

        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics
            if training_args.do_eval and not is_torch_tpu_available()
            else None,
            # data_collator=lambda data: {
            #     "input_ids": torch.stack([f["input_ids"] for f in data]),
            #     "attention_mask": torch.stack([f["attention_mask"] for f in data]),
            #     "labels": torch.stack([f["input_ids"] for f in data]),
            # },
        )


@dataclass
class FederatedTrainingSession:
    model_name: str = "EleutherAI/gpt-neo-125M"
    num_nodes: int = 2

    def run(self):
        self.create_datasets()
        self.create_models()
        self.train_concurrently()

    def create_random_partitioned_datasets(self):
        # load wikitext
        raw_datasets = datasets.load_dataset(
            "wikitext",
            "wikitext-103-v1",
            split=["train[:120000]", "validation[:1000]"],
            # streaming=True
        )
        # partition them
        partitioned_datasets = []
        for i in range(self.num_nodes):
            partitioned_datasets.append(
                raw_datasets[0].shard(num_shards=self.num_nodes, index=i)
            )
        self.train_datasets = partitioned_datasets
        self.test_dataset = raw_datasets[1]

    def create_models(self):
        self.federated_models = []
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.data_collator = DataCollatorForLanguageModeling(self.tokenizer, mlm=False)
        for i in range(self.num_nodes):
            model = AutoModelForCausalLM.from_pretrained(self.model_name).cuda()
            model.resize_token_embeddings(len(self.tokenizer))
            self.federated_models.append(model)
        return self.federated_models

    def train_concurrently(self):
        training_args = TrainingArguments(
            learning_rate=2e-5,
            # output_dir="./results",
            num_train_epochs=3,
            per_device_train_batch_size=16,
            evaluation_strategy="steps",
            logging_strategy="steps",
            gradient_accumulation_steps=10,
            eval_steps=50,
            logging_steps=50,
            save_strategy=IntervalStrategy.NO,
            # report_to=["wandb"],
            eval_delay=0,
            per_device_eval_batch_size=16,
            eval_accumulation_steps=10,
        )
        trainers = []

        def preprocess_logits_for_metrics(logits, labels):
            if isinstance(logits, tuple):
                # Depending on the model and config, logits may contain extra tensors,
                # like past_key_values, but logits always come first
                logits = logits[0]
            return logits.argmax(dim=-1)

        accuracy_metric = evaluate.load("accuracy")
        # perplexity_metric = evaluate.load("perplexity")

        def compute_metrics(eval_preds):
            preds, labels = eval_preds
            # preds have the same shape as the labels, after the argmax(-1) has been calculated
            # by preprocess_logits_for_metrics but we need to shift the labels
            labels = labels[:, 1:].reshape(-1)
            preds = preds[:, :-1].reshape(-1)
            return accuracy_metric.compute(predictions=preds, references=labels)

        for i in range(self.num_nodes):
            trainer = Trainer(
                model=self.federated_models[i],
                args=training_args,
                train_dataset=self.train_datasets[i],
                eval_dataset=self.test_dataset,
                data_collator=self.data_collator,
                compute_metrics=compute_metrics,
                preprocess_logits_for_metrics=preprocess_logits_for_metrics
                if training_args.do_eval and not is_torch_tpu_available()
                else None,
            )
            trainers.append(trainer)

        with ThreadPoolExecutor(max_workers=self.num_nodes) as executor:
            executor.map(lambda trainer: trainer.train(), trainers)


if __name__ == "__main__":
    args = TrainingSessionArgParser().parse_args()
    session = TrainingSession(
        model_name=args.model_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.learning_rate,
    )
    session.run()
