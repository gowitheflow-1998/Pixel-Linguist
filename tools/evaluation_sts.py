from sklearn.metrics.pairwise import paired_cosine_distances
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm
import numpy as np
import torch
from datasets import load_dataset
from PIL import Image
from pixel import (
    AutoConfig,
    PangoCairoTextRenderer,
    PIXELForSequenceClassification,
    PIXELForRepresentation,
    PoolingMode,
    get_attention_mask,
    get_transforms,
    glue_strip_spaces,
    resize_model_embeddings,
)
import datasets


def sts_evaluation(model_name, language):
    LANGUAGE = language
    DATASET_NAME = 'multi-sts'
    POOLING_MODE = "mean"
    FALLBACK_FONTS_DIR = "data/fallback_fonts"
    SEQ_LEN = 64
    BSZ = 16

    datasets_keys = {
        "multi-sts": ('stsb_multi_mt', "sentence1", "sentence2")
    }
    this_dataset_name, sentence1_key, sentence2_key = datasets_keys[DATASET_NAME]
    train_dataset = load_dataset(
        this_dataset_name,
        LANGUAGE,
        split="test",
    )

    def image_preprocess_fn(examples):

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
        return examples

    def image_collate_fn(examples):

        pixel_values1 = torch.stack([example["pixel_values1"] for example in examples])
        attention_mask1 = torch.stack([example["attention_mask1"] for example in examples])

        pixel_values2 = torch.stack([example["pixel_values2"] for example in examples])
        attention_mask2 = torch.stack([example["attention_mask2"] for example in examples])

        return {
            'sentence1': {"pixel_values": pixel_values1, "attention_mask": attention_mask1},
            'sentence2': {"pixel_values": pixel_values2, "attention_mask": attention_mask2},
        }


    num_labels = 0 

    print(f'Building models for {model_name}')

    config = AutoConfig.from_pretrained(
        model_name,
        num_labels=num_labels,
        finetuning_task=this_dataset_name,
        attention_probs_dropout_prob=0.1,
        hidden_dropout_prob=0.1,
    )

    print(f'model type: {config.model_type}')

    if "Pixel" in model_name:
        model = PIXELForRepresentation.from_pretrained(
            model_name,
            config=config,
            pooling_mode=PoolingMode.from_string(POOLING_MODE),
            add_layer_norm=True,
        )

    else:  
        model = PIXELForSequenceClassification.from_pretrained(
            model_name,
            config=config,
            pooling_mode=PoolingMode.from_string(POOLING_MODE),
            add_layer_norm=True,
        )


    renderer_cls = PangoCairoTextRenderer
    processor = renderer_cls.from_pretrained(
        model_name,
        cache_dir=None,
        revision="main",
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

    device = "cuda:1" if torch.cuda.is_available() else "cpu"

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

    labels = [n['similarity_score'] for n in train_dataset]

    cosine_scores = 1 - (paired_cosine_distances(embeddings1, embeddings2))
    eval_pearson_cosine, _ = pearsonr(labels, cosine_scores)
    eval_spearman_cosine, _ = spearmanr(labels, cosine_scores)

    cos = cosine_similarity(embeddings1, embeddings1)
    ani = round((np.sum(cos) - len(cos))/(len(cos) * (len(cos)-1)),3)

    return eval_pearson_cosine, eval_spearman_cosine, ani

if __name__ == "__main__":
    model_pearson_results = []
    model_spearman_results = []
    anisotropy = []
    model_name = "Pixel-Linguist/Pixel-Linguist-v0"
    for eval_language in ["de"]:
        eval_pearson_cosine, eval_spearman_cosine, ani = sts_evaluation(model_name, eval_language)
        model_pearson_results.append(eval_pearson_cosine)
        model_spearman_results.append(eval_spearman_cosine)
        anisotropy.append(ani)
    print("spearman all languages:", model_spearman_results)
    print("anisotropy all languages:",anisotropy)
    print("mean spearman except English:",np.mean(model_spearman_results[1:]))
    print("mean anisotropy except English:",np.mean(anisotropy[1:]))