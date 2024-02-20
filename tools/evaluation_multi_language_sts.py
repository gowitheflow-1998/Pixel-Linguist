import logging
from sklearn.metrics.pairwise import paired_cosine_distances, paired_euclidean_distances, paired_manhattan_distances
from scipy.stats import pearsonr, spearmanr
import logging
from tqdm import tqdm
import numpy as np
import torch
from datasets import load_dataset
from PIL import Image
from pixel import (
    AutoConfig,
    Modality,
    PangoCairoTextRenderer,
    PIXELForSequenceClassification,
    PoolingMode,
    get_attention_mask,
    get_transforms,
    glue_strip_spaces,
    resize_model_embeddings,
)

from transformers.utils import check_min_version
from transformers.utils.versions import require_version

import datasets

check_min_version("4.17.0")

require_version("datasets>=1.8.0", "To fix: pip install ./datasets")


def sts_evaluation(model_name, language):
    LANGUAGE = language
    DATASET_NAME = 'multi-sts'
    POOLING_MODE = "mean"
    FALLBACK_FONTS_DIR = "data/fallback_fonts"
    SEQ_LEN = 64
    BSZ = 16

    datasets_keys = {
        "snli": ("rungalileo/snli", "premise", "hypothesis"),
        "mnli": ("SetFit/mnli", "text1", "text2"),
        "mteb": ('mteb/stsbenchmark-sts', "sentence1", "sentence2"),
        "multi-sts": ('stsb_multi_mt', "sentence1", "sentence2")
    }
    this_dataset_name, sentence1_key, sentence2_key = datasets_keys[DATASET_NAME]
    train_dataset = load_dataset(
        this_dataset_name,
        LANGUAGE,
        split="test",
        cache_dir=None,
        use_auth_token=None,
    )

    def image_preprocess_fn(examples):

        # two sentences for contrastive learning
        if not sentence2_key:
            raise ValueError(f"two sentences needed, but got one.")

        encodings = [processor(text=format_fn(a)) for a in examples[sentence1_key]]
        examples["pixel_values1"] = [transforms(Image.fromarray(e.pixel_values)) for e in encodings]
        examples["attention_mask1"] = [
            get_attention_mask(e.num_text_patches, seq_length=SEQ_LEN) for e in encodings
        ]

        encodings = [processor(text=format_fn(a)) for a in examples[sentence2_key]]
        examples["pixel_values2"] = [transforms(Image.fromarray(e.pixel_values)) for e in encodings]
        examples["attention_mask2"] = [
            get_attention_mask(e.num_text_patches, seq_length=SEQ_LEN) for e in encodings
        ]

        if "label" in examples:
            examples["label"] = [l if l != -1 else -100 for l in examples["label"]]
        if "score" in examples:
            examples["label"] = [l if l != -1 else -100 for l in examples["score"]]
        if "similarity_score" in examples:
            examples["label"] = [l if l != -1 else -100 for l in examples["similarity_score"]]
        return examples

    def image_collate_fn(examples):

        # two sentences for contrastive learning

        pixel_values1 = torch.stack([example["pixel_values1"] for example in examples])
        attention_mask1 = torch.stack([example["attention_mask1"] for example in examples])

        pixel_values2 = torch.stack([example["pixel_values2"] for example in examples])
        attention_mask2 = torch.stack([example["attention_mask2"] for example in examples])

        if "label" in examples[0]:
            labels = torch.LongTensor([example["label"] for example in examples])
        else:
            labels = None

        return {
            'pixel_values': labels,  # for ignore warning obly
            'sentence1': {"pixel_values": pixel_values1, "attention_mask": attention_mask1},
            'sentence2': {"pixel_values": pixel_values2, "attention_mask": attention_mask2},
            'labels': labels
        }

    try:
        label_list = train_dataset.features["similarity_score"].names
    except AttributeError:
        label_list = None

    # Labels
    num_labels = 0  # len(label_list) no need
    if label_list:
        label_to_id = {v: i for i, v in enumerate(label_list)}
    else:  # for mnli
        label_to_id = {'entailment': 0, 'neutral': 1, 'contradiction': 2}


    print(f'Building models for {model_name}')
    config_kwargs = {
        "cache_dir": None,
        "revision": "main",
        "use_auth_token": None,
    }

    config = AutoConfig.from_pretrained(
        model_name,
        num_labels=num_labels,
        finetuning_task=this_dataset_name,
        attention_probs_dropout_prob=0.1,
        hidden_dropout_prob=0.1,
        **config_kwargs,
    )

    print(f'model type: {config.model_type}')

    model = PIXELForSequenceClassification.from_pretrained(
        model_name,
        config=config,
        pooling_mode=PoolingMode.from_string(POOLING_MODE),
        add_layer_norm=True,
        **config_kwargs,
    )

    model.config.label2id = label_to_id
    model.config.id2label = {id: label for label, id in config.label2id.items()}

    modality = Modality.IMAGE
    renderer_cls = PangoCairoTextRenderer
    processor = renderer_cls.from_pretrained(
        model_name,
        cache_dir=None,
        revision="main",
        use_auth_token=None,
        fallback_fonts_dir=FALLBACK_FONTS_DIR,
        rgb=False,
    )

    processor.max_seq_length = SEQ_LEN
    resize_model_embeddings(model, processor.max_seq_length)

    transforms = get_transforms(
        do_resize=True,
        size=(processor.pixels_per_patch, processor.pixels_per_patch * processor.max_seq_length),
    )
    format_fn = glue_strip_spaces

    preprocess_fn = image_preprocess_fn

    train_dataset.features["pixel_values"] = datasets.Image()
    train_dataset.set_transform(preprocess_fn)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    model.to(device)

    total_output_a = []
    total_output_b = []

    model.eval()
    with torch.no_grad():
        for step in tqdm(range(0, len(train_dataset), BSZ)):
            inputs = [train_dataset[step + idx] for idx in range(0, min(BSZ, len(train_dataset)-step))]
            inputs = image_collate_fn(inputs)
            sentence1 = inputs.pop("sentence1")
            sentence2 = inputs.pop("sentence2")

            sentence1 = {k: v.to(device) for k, v in sentence1.items()} 
            sentence2 = {k: v.to(device) for k, v in sentence2.items()} 

            outputs_a = model(**sentence1).logits
            outputs_b = model(**sentence2).logits

            total_output_a.append(outputs_a.detach().cpu())
            total_output_b.append(outputs_b.detach().cpu())


    embeddings1  = torch.cat(total_output_a, dim=0)
    embeddings2 = torch.cat(total_output_b, dim=0)


    labels = [n['label'] for n in train_dataset]

    cosine_scores = 1 - (paired_cosine_distances(embeddings1, embeddings2))
    manhattan_distances = -paired_manhattan_distances(embeddings1, embeddings2)
    euclidean_distances = -paired_euclidean_distances(embeddings1, embeddings2)
    dot_products = [np.dot(emb1, emb2) for emb1, emb2 in zip(embeddings1, embeddings2)]

    eval_pearson_cosine, _ = pearsonr(labels, cosine_scores)
    eval_spearman_cosine, _ = spearmanr(labels, cosine_scores)

    eval_pearson_manhattan, _ = pearsonr(labels, manhattan_distances)
    eval_spearman_manhattan, _ = spearmanr(labels, manhattan_distances)

    eval_pearson_euclidean, _ = pearsonr(labels, euclidean_distances)
    eval_spearman_euclidean, _ = spearmanr(labels, euclidean_distances)

    eval_pearson_dot, _ = pearsonr(labels, dot_products)
    eval_spearman_dot, _ = spearmanr(labels, dot_products)

    logger = logging.getLogger(__name__)
    logger.info("Cosine-Similarity :\tPearson: {:.4f}\tSpearman: {:.4f}".format(
        eval_pearson_cosine, eval_spearman_cosine))
    logger.info("Manhattan-Distance:\tPearson: {:.4f}\tSpearman: {:.4f}".format(
        eval_pearson_manhattan, eval_spearman_manhattan))
    logger.info("Euclidean-Distance:\tPearson: {:.4f}\tSpearman: {:.4f}".format(
        eval_pearson_euclidean, eval_spearman_euclidean))
    logger.info("Dot-Product-Similarity:\tPearson: {:.4f}\tSpearman: {:.4f}".format(
        eval_pearson_dot, eval_spearman_dot))
    return eval_pearson_cosine, eval_spearman_cosine

if __name__ == "__main__":
    model_pearson_results = []
    model_spearman_results = []
    model_name = # replace with model name
    for eval_language in ["en"]:
        eval_pearson_cosine, eval_spearman_cosine = sts_evaluation(model_name, eval_language)
        model_pearson_results.append(eval_pearson_cosine)
        model_spearman_results.append(eval_spearman_cosine)
    print(model_pearson_results)
    print(model_spearman_results)
    print(np.mean(model_pearson_results))
    print(np.mean(model_spearman_results))