#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for sequence classification on NLI datasets."""

import argparse
import copy
import logging
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Optional, Tuple, Union

import numpy as np
import torch
import transformers
from datasets import load_dataset, load_metric
from PIL import Image
from pixel import (
    AutoConfig,
    AutoModelForSequenceClassification,
    Modality,
    PangoCairoTextRenderer,
    PIXELForSequenceClassification,
    PIXELTrainerForContrastiveWithEval,
    PIXELTrainingArguments,
    PoolingMode,
    PyGameTextRenderer,
    get_attention_mask,
    get_transforms,
    glue_strip_spaces,
    log_sequence_classification_predictions,
    resize_model_embeddings,
)
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PreTrainedTokenizerFast,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

import datasets

check_min_version("4.17.0")

require_version("datasets>=1.8.0", "To fix: pip install ./datasets")


datasets_keys = {
    "snli": ("rungalileo/snli", "premise", "hypothesis"),   # currently not support anymore, use 'allnli' instead
    "mnli": ("SetFit/mnli", "text1", "text2"),              # currently not support anymore, use 'allnli' instead
    "stsb": ('SetFit/stsb', "text1", "text2"),
    "mteb": ('mteb/stsbenchmark-sts', "sentence1", "sentence2"),
    "allnli": ("gowitheflow/allnli-sup", "sentence1", "sentence2"),
    "allnlineg": ("gowitheflow/allnli-withnegs", "sentence1", "sentence2&sentence3"),
    "unsup-pixel_aug": ("gowitheflow/wiki1M-character-level-all", "sentence1", None),
    "unsup-c": ("gowitheflow/wiki1M-character-level-all", "sentence1", "sentence2"),
    "unsup-wr": ("gowitheflow/wiki1M-word-random-shuffle", "sentence1", "sentence2"),
    "unsup-wc": ("gowitheflow/wiki1M-word-condition-shuffle", "sentence1", "sentence2"),
    "unsup-wa": ("gowitheflow/wiki1M-word-character-all-multiple", "sentence1", "sentence2"),
    "para":("sentence-transformers/parallel-sentences", "Membership of Parliament: see Minutes", "Състав на Парламента: вж. протоколи"),
    "wikispan":("gowitheflow/wiki-span", "sentence1", "sentence2"),
    "en-de": ("gowitheflowlab/nli-sts-en-de", "sentence1", "sentence2"),
    "en-es": ("gowitheflowlab/nli-sts-en-es", "sentence1", "sentence2"),
    "en-fr": ("gowitheflowlab/nli-sts-en-fr", "sentence1", "sentence2"),
    "en-it": ("gowitheflowlab/nli-sts-en-it", "sentence1", "sentence2"),
    "en-nl": ("gowitheflowlab/nli-sts-en-nl", "sentence1", "sentence2"),
    "en-pl": ("gowitheflowlab/nli-sts-en-pl", "sentence1", "sentence2"),
    "en-pt": ("gowitheflowlab/nli-sts-en-pt", "sentence1", "sentence2"),
    "en-ru": ("gowitheflowlab/nli-sts-en-ru", "sentence1", "sentence2"),
    "en-zh": ("gowitheflowlab/nli-sts-en-zh", "sentence1", "sentence2"),
    "parallel-pt-nl-pl":("gowitheflowlab/parallel-pt-nl-pl","sentence1","sentence2"),
    "parallel-9":("gowitheflowlab/parallel-9","sentence1","sentence2"),
    "parallel-all":("gowitheflowlab/parallel-all","sentence1","sentence2"),
    "parallel-small":("gowitheflowlab/parallel-small","English","Other Language"),
    "parallel-medium":("gowitheflowlab/parallel-medium","English","Other Language"),
    "parallel-small-nli":("gowitheflowlab/parallel-small-w-nli","English","Other Language"),
    "parallel-medium-nli":("gowitheflowlab/parallel-medium-w-nli","English","Other Language"),
}


logger = logging.getLogger(__name__)

def get_sentence_keys(example):
    for name, val in datasets_keys.items():
        this_dataset_name, sentence1_key, sentence2_key = val
        if sentence1_key in example and sentence2_key in example:
            return sentence1_key, sentence2_key
        else:
            continue

def select_datasets(_dataset, id2label, do_select=False):
    key = 'id' if 'id' in _dataset[0] else 'idx'
    if do_select:
        logger.info("Select positive samples only.")
        _dataset = _dataset.select(
            data[key] for data in _dataset if id2label[int(data['label'])].capitalize() == 'Entailment'
        )
    return _dataset

def condition(example):
    return example['label'].capitalize() == 'Entailment'


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the NLI dataset to use (via the datasets library)."}
    )
    dataset_name_val: Optional[str] = field(
        default="mteb", metadata={"help": "The name of the NLI dataset to use (via the datasets library)."}
    )
    dataset_config_name: str = field(
        default=None, metadata={"help": "Subset of the NLI dataset, e.g language ISO code"}
    )
    max_seq_length: Optional[int] = field(
        default=196,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
            "value if set."
        },
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation data."}
    )
    test_file: Optional[str] = field(default=None, metadata={"help": "A csv or a json file containing the test data."})

    def __post_init__(self):
        if self.dataset_name is not None:
            pass
        elif self.train_file is None or self.validation_file is None:
            raise ValueError("Need either a NLI task, a training/validation file or a dataset name.")
        # else:
        #     train_extension = self.train_file.split(".")[-1]
        #     assert train_extension in ["csv", "json"], "`train_file` should be a csv or a json file."
        #     validation_extension = self.validation_file.split(".")[-1]
        #     assert (
        #         validation_extension == train_extension
        #     ), "`validation_file` should have the same extension (csv or json) as `train_file`."


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    processor_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained processor name or path if not the same as model_name"}
    )
    rendering_backend: Optional[str] = field(
        default="pangocairo", metadata={
            "help": "Rendering backend to use. Options are 'pygame' or 'pangocairo'. For most applications it is "
                    "recommended to use the default 'pangocairo'."}
    )
    fallback_fonts_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Directory containing fallback font files used by the text renderer for PIXEL. "
                          "PyGame does not support fallback fonts so this argument is ignored when using the "
                          "PyGame backend."},
    )
    render_rgb: bool = field(
        default=False,
        metadata={
            "help": "Whether or not to render images in RGB. RGB rendering can be useful when working with emoji "
            "but it makes rendering a bit slower, so it is recommended to turn on RGB rendering only "
            "when there is need for it. PyGame does not support fallback fonts so this argument is ignored "
            "when using the PyGame backend."
        }
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: str = field(
        default=None,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )
    pooling_mode: str = field(
        default="mean",
        metadata={
            "help": f"Pooling mode to use in classification head (options are {[e.value for e in PoolingMode]}."
        },
    )
    pooler_add_layer_norm: bool = field(
        default=True,
        metadata={
            "help": "Whether to add layer normalization to the classification head pooler. Note that this flag is"
            "ignored and no layer norm is added when using CLS pooling mode."
        },
    )
    dropout_prob: float = field(
        default=0.1, metadata={"help": "Dropout probability for attention blocks and classification head"}
    )

    def __post_init__(self):
        self.pooling_mode = PoolingMode.from_string(self.pooling_mode)

        if not self.rendering_backend.lower() in ["pygame", "pangocairo"]:
            raise ValueError("Invalid rendering backend. Supported backends are 'pygame' and 'pangocairo'.")
        else:
            self.rendering_backend = self.rendering_backend.lower()


def get_processor(model_args: argparse.Namespace, modality: Modality):
    if modality == Modality.TEXT:
        processor = AutoTokenizer.from_pretrained(
            model_args.processor_name if model_args.processor_name else model_args.model_name_or_path,
            use_fast=True,
            add_prefix_space=True if model_args.model_name_or_path == "roberta-base" else False,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=model_args.use_auth_token if model_args.use_auth_token else None,
        )
    elif modality == Modality.IMAGE:
        renderer_cls = PyGameTextRenderer if model_args.rendering_backend == "pygame" else PangoCairoTextRenderer
        processor = renderer_cls.from_pretrained(
            model_args.processor_name if model_args.processor_name else model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=model_args.use_auth_token if model_args.use_auth_token else None,
            fallback_fonts_dir=model_args.fallback_fonts_dir,
            rgb=model_args.render_rgb,
        )
    else:
        raise ValueError("Modality not supported.")
    return processor


def get_model_and_config(model_args: argparse.Namespace, num_labels: int, dataset_name: str):
    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": model_args.use_auth_token if model_args.use_auth_token else None,
    }

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=dataset_name,
        attention_probs_dropout_prob=model_args.dropout_prob,
        hidden_dropout_prob=model_args.dropout_prob,
        **config_kwargs,
    )

    logger.info(f"Using dropout with probability {model_args.dropout_prob}")

    if config.model_type in ["bert", "roberta"]:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            **config_kwargs,
        )
    elif config.model_type in ["vit_mae", "pixel"]:
        model = PIXELForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            pooling_mode=model_args.pooling_mode,
            add_layer_norm=model_args.pooler_add_layer_norm,
            **config_kwargs,
        )
    else:
        raise ValueError(f"Model type {config.model_type} not supported.")

    return model, config


def get_collator(
    training_args: argparse.Namespace,
    processor: Union[Union[PyGameTextRenderer, PangoCairoTextRenderer], PreTrainedTokenizerFast],
    modality: Modality,
):
    def image_collate_fn(examples):

        # two sentences for contrastive learning

        pixel_values1 = torch.stack([example["pixel_values1"] for example in examples])
        attention_mask1 = torch.stack([example["attention_mask1"] for example in examples])

        pixel_values2 = torch.stack([example["pixel_values2"] for example in examples])
        attention_mask2 = torch.stack([example["attention_mask2"] for example in examples])

        if 'pixel_values3' in examples[0]:
            pixel_values3 = torch.stack([example["pixel_values3"] for example in examples])
            attention_mask3 = torch.stack([example["attention_mask3"] for example in examples])

            pixel_values2 = torch.cat([pixel_values2, pixel_values3], dim=0)
            attention_mask2 = torch.cat([attention_mask2, attention_mask3], dim=0)

        labels = torch.LongTensor([-1 for example in examples])  # no use in contrastive

        return {
            'pixel_values': labels,  # for ignore warning obly
            'sentence1': {"pixel_values": pixel_values1, "attention_mask": attention_mask1},
            'sentence2': {"pixel_values": pixel_values2, "attention_mask": attention_mask2},
            'labels': labels
        }

    if modality == Modality.IMAGE:
        collator = image_collate_fn
    elif modality == Modality.TEXT:
        collator = DataCollatorWithPadding(processor, pad_to_multiple_of=8) if training_args.fp16 else None
    else:
        raise ValueError(f"Modality {modality} not supported.")

    return collator


def get_preprocess_fn(
    data_args: argparse.Namespace,
    processor: Union[Union[PyGameTextRenderer, PangoCairoTextRenderer], PreTrainedTokenizerFast],
    modality: Modality,
    sentence_keys: Tuple[str, Optional[str]],
):
    sentence1_key, sentence2_key = sentence_keys
    if sentence2_key is not None:
        if "&" in sentence2_key: 
            sentence2_key, sentence3_key = sentence2_key.split('&')
        else:
            sentence3_key = None
    else:
        sentence3_key = None

    if modality == Modality.IMAGE:

        transforms = get_transforms(
            do_resize=True,
            size=(processor.pixels_per_patch, processor.pixels_per_patch * processor.max_seq_length),
        )
        transforms2 = get_transforms(
            do_resize=True,
            size=(processor.pixels_per_patch, processor.pixels_per_patch * processor.max_seq_length),
            do_vision_aug=True,
            do_random_choise=False,
            vis_aug_choise=3,
        )
        
        format_fn = glue_strip_spaces

        def image_preprocess_fn(examples):

            encodings = [processor(text=format_fn(a)) for a in examples[sentence1_key]]
            examples["pixel_values1"] = [transforms(Image.fromarray(e.pixel_values)) for e in encodings]
            examples["attention_mask1"] = [
                get_attention_mask(e.num_text_patches, seq_length=data_args.max_seq_length) for e in encodings
            ]

            if not sentence2_key:
                encodings = [processor(text=format_fn(a)) for a in examples[sentence1_key]]
                examples["pixel_values2"] = [transforms2(Image.fromarray(e.pixel_values)) for e in encodings]
            else:
                encodings = [processor(text=format_fn(a)) for a in examples[sentence2_key]]
                examples["pixel_values2"] = [transforms(Image.fromarray(e.pixel_values)) for e in encodings]

            
            examples["attention_mask2"] = [
                get_attention_mask(e.num_text_patches, seq_length=data_args.max_seq_length) for e in encodings
            ]

            if sentence3_key is not None:  # may have problem here
                encodings = [processor(text=format_fn(a)) for a in examples[sentence3_key]]
                examples["pixel_values3"] = [transforms(Image.fromarray(e.pixel_values)) for e in encodings]
                examples["attention_mask3"] = [
                    get_attention_mask(e.num_text_patches, seq_length=data_args.max_seq_length) for e in encodings
                ]

            if "label" in examples:
                examples["label"] = [l if l != -1 else -100 for l in examples["label"]]
            if "score" in examples:
                examples["label"] = [l if l != -1 else -100 for l in examples["score"]]

            return examples

        preprocess_fn = image_preprocess_fn

    elif modality == Modality.TEXT:

        def text_preprocess_fn(examples):
            # Tokenize the texts
            args = (
                (examples[sentence1_key],)
                if sentence2_key is None
                else (examples[sentence1_key], examples[sentence2_key])
            )
            result = processor(*args, padding="max_length", max_length=data_args.max_seq_length, truncation=True)

            if "label" in examples:
                result["label"] = [l if l != -1 else -100 for l in examples["label"]]

            return result

        preprocess_fn = text_preprocess_fn
    else:
        raise ValueError(f"Modality {modality} not supported.")

    return preprocess_fn

def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, PIXELTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    log_level = logging.INFO
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        level=log_level,
    )
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    train_dataset_name, sentence1_key, sentence2_key = datasets_keys[data_args.dataset_name]
    val_dataset_name, val_sentence1_key, val_sentence2_key = datasets_keys[data_args.dataset_name_val]

    if training_args.do_train:
        if data_args.train_file:
            # Settings fixed mainly for our robustness experiments
            train_dataset = load_dataset(
                train_dataset_name,
                data_files=data_args.train_file,
                delimiter="\t",
                split="train",
                cache_dir=model_args.cache_dir,
                use_auth_token=True if model_args.use_auth_token else None,
            )
            train_dataset = train_dataset.class_encode_column("label")
        else:
            train_dataset = load_dataset(
                train_dataset_name,
                data_args.dataset_config_name,
                split="train",
                cache_dir=model_args.cache_dir,
                use_auth_token=True if model_args.use_auth_token else None,
            )
        try:
            label_list = train_dataset.features["label"].names
        except (AttributeError, KeyError):
            label_list = None

    if training_args.do_eval:
        if data_args.validation_file:
            # Settings fixed mainly for our robustness experiments
            eval_dataset = load_dataset(
                val_dataset_name,
                data_files=data_args.validation_file,
                delimiter="\t",
                split="test",
                cache_dir=model_args.cache_dir,
                use_auth_token=True if model_args.use_auth_token else None,
            )
            eval_dataset = eval_dataset.class_encode_column("label")
        else:
            eval_dataset = load_dataset(
                val_dataset_name,
                data_args.dataset_config_name,
                split="test",
                cache_dir=model_args.cache_dir,
                use_auth_token=True if model_args.use_auth_token else None,
            )

    if training_args.do_predict:
        if data_args.test_file:
            # Settings fixed mainly for our robustness experiments
            predict_dataset = load_dataset(
                this_dataset_name,
                data_files=data_args.test_file,
                delimiter="\t",
                split="train",
                cache_dir=model_args.cache_dir,
                use_auth_token=True if model_args.use_auth_token else None,
            )
            predict_dataset = predict_dataset.class_encode_column("label")
        else:
            predict_dataset = load_dataset(
                this_dataset_name,
                data_args.dataset_config_name,
                split="test",
                cache_dir=model_args.cache_dir,
                use_auth_token=True if model_args.use_auth_token else None,
            )

    # Labels
    num_labels = 0  # len(label_list) no need
    if label_list:
        label_to_id = {v: i for i, v in enumerate(label_list)}
    else:
        label_to_id = {'Entailment': 0, 'Neutral': 1, 'Contradiction': 2}

    # Load pretrained model and config
    model, config = get_model_and_config(model_args, num_labels, data_args.dataset_name)

    model.config.label2id = label_to_id
    model.config.id2label = {id: label for label, id in config.label2id.items()}

    
    key_list = ['parallel','xnli','unsup','neg', 'span']
    if not any(key in data_args.dataset_name for key in key_list):
        logger.info("Select positive samples only.")
        train_dataset = train_dataset.filter(condition)

    modality = Modality.TEXT if config.model_type in ["bert", "roberta"] else Modality.IMAGE
    processor = get_processor(model_args, modality)

    if modality == Modality.IMAGE:
        if processor.max_seq_length != data_args.max_seq_length:
            processor.max_seq_length = data_args.max_seq_length

        resize_model_embeddings(model, processor.max_seq_length)

    preprocess_fn = get_preprocess_fn(data_args, processor, modality, (sentence1_key, sentence2_key))
    preprocess_fn_eval = get_preprocess_fn(data_args, processor, modality, (val_sentence1_key, val_sentence2_key))

    if training_args.do_train:
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))
        if modality == Modality.IMAGE:
            train_dataset.features["pixel_values"] = datasets.Image()
        train_dataset.set_transform(preprocess_fn)

    if training_args.do_eval:
        if data_args.max_eval_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))
        if modality == Modality.IMAGE:
            eval_dataset.features["pixel_values"] = datasets.Image()
        eval_examples = copy.deepcopy(eval_dataset)
        eval_dataset.set_transform(preprocess_fn_eval)

    if training_args.do_predict or data_args.test_file is not None:
        if data_args.max_predict_samples is not None:
            predict_dataset = predict_dataset.select(range(data_args.max_predict_samples))
        if modality == Modality.IMAGE:
            predict_dataset.features["pixel_values"] = datasets.Image()
        predict_examples = copy.deepcopy(predict_dataset)
        predict_dataset.set_transform(preprocess_fn)

    # Log a few random samples from the training set:
    if training_args.do_train:
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    if training_args.do_eval:
        for index in random.sample(range(len(eval_dataset)), 3):
            logger.info(f"Sample {index} of the eval set: {eval_dataset[index]}.")

    # Get the metric function
    metric = load_metric("xnli")

    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.argmax(preds, axis=1)
        return metric.compute(predictions=preds, references=p.label_ids)
    
    # try shuffle
    # if 'allnli' in data_args.dataset_name:
    #     train_dataset = train_dataset.shuffle(seed=42)

    # Initialize our Trainer
    trainer = PIXELTrainerForContrastiveWithEval(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=processor,
        data_collator=get_collator(training_args, processor, modality),
        # compute_metrics=compute_metrics,
        # callbacks=[EarlyStoppingCallback(early_stopping_patience=training_args.early_stopping_patience)]
        # if training_args.early_stopping
        # else None,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.save_model()

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval and False:
        logger.info("*** Evaluate ***")

        outputs = trainer.predict(test_dataset=eval_dataset, metric_key_prefix="eval")
        metrics = outputs.metrics

        if training_args.log_predictions:
            log_sequence_classification_predictions(
                training_args=training_args,
                features=eval_dataset,
                examples=eval_examples,
                predictions=outputs.predictions,
                sentence1_key=sentence1_key,
                sentence2_key=sentence2_key,
                modality=modality,
                prefix="eval",
            )

        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.do_predict:
        logger.info("*** Predict ***")

        outputs = trainer.predict(predict_dataset, metric_key_prefix="test")
        metrics = outputs.metrics

        trainer.log_metrics("test", metrics)
        trainer.save_metrics("test", metrics)

        # Log predictions to understand where model goes wrong
        if training_args.log_predictions:
            log_sequence_classification_predictions(
                training_args=training_args,
                features=predict_dataset,
                examples=predict_examples,
                predictions=outputs.predictions,
                sentence1_key=sentence1_key,
                sentence2_key=sentence2_key,
                modality=modality,
                prefix="test",
            )

        max_test_samples = (
            data_args.max_predict_samples if data_args.max_predict_samples is not None else len(predict_dataset)
        )
        metrics["test_samples"] = min(max_test_samples, len(predict_dataset))

    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "text-classification"}
    if data_args.dataset_name is not None:
        kwargs["language"] = data_args.dataset_config_name
        kwargs["dataset_tags"] = data_args.dataset_name
        kwargs["dataset_args"] = data_args.dataset_name
        kwargs["dataset"] = f"{data_args.dataset_name.upper()}"

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()