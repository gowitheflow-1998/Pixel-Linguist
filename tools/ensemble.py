import copy 
import random
import numpy as np
import torch
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


seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed) 
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


datasets_keys = {
    "snli": ("rungalileo/snli", "premise", "hypothesis"),
    "mnli": ("SetFit/mnli", "text1", "text2"),
    "stsb": ('SetFit/stsb', "text1", "text2"),
    "mteb": ('mteb/stsbenchmark-sts', "sentence1","sentence2"),
    "multi": ("stsb_multi_mt", "sentence1","sentence2")
}


POOLING_MODE = "mean"
FALLBACK_FONTS_DIR = "data/fallback_fonts"
SEQ_LEN = 64
BSZ = 16

DATASET_NAME = 'mteb'
this_dataset_name, sentence1_key, sentence2_key = datasets_keys[DATASET_NAME]


num_labels = 0  

model_name = #model name

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

    return examples

preprocess_fn = image_preprocess_fn


def initialize_model(model_name):
    model = PIXELForSequenceClassification.from_pretrained(
        model_name,
        config=config,
        pooling_mode=PoolingMode.from_string(POOLING_MODE),
        add_layer_norm=True,
        **config_kwargs,
    )
    return model

def average_models_state_dicts(model_state_dicts):
    num_models = len(model_state_dicts)
    averaged_model = copy.deepcopy(models[0])
    averaged_state_dict = averaged_model.state_dict()

    for key in averaged_state_dict.keys():
        layer_weights = sum([model_state_dict[key] for model_state_dict in model_state_dicts]) / num_models
        averaged_state_dict[key] = layer_weights
    averaged_model.load_state_dict(averaged_state_dict)
    return averaged_model

model_names = ["0-unsup-c-64-768-3e-6-1300/checkpoint-1300",
               "0-unsup-wc-64-768-3e-6-1300/checkpoint-1300",
               "0-unsup-wr-64-768-3e-6-1300/checkpoint-1300",
               "0-unsup-wa-64-768-3e-6-1300/checkpoint-1300"
               ]

models = [initialize_model(model_name) for model_name in model_names]
model_state_dicts = [model.state_dict() for model in models]

averaged_model = average_models_state_dicts(model_state_dicts)

averaged_model.save_pretrained("0-unsup-ensemble-4-last")