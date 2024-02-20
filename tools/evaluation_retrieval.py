from tqdm import tqdm
import torch
import numpy as np
from tqdm import tqdm
import numpy as np
import torch
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
import os
from beir import util
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from typing import List, Dict
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES

class PixelForRetrieval:
    def __init__(self, model_path=None, **kwargs):
        self.SEQ_LEN = 64  # Adjust as per your model's requirements
        self.FALLBACK_FONTS_DIR = "data/fallback_fonts"  # Update as needed
        self.sep = " "
        config_kwargs = {
            "cache_dir": None,
            "revision": "main",
            "use_auth_token": None,
        }

        # Load model configuration
        config = AutoConfig.from_pretrained(
            model_path,
            num_labels=0,
            finetuning_task="Placeholder",
            attention_probs_dropout_prob=0.1,
            hidden_dropout_prob=0.1,
            **config_kwargs,
        )

        # Load the model
        if "pixel" in model_path:
            self.model = PIXELForRepresentation.from_pretrained(
                model_path,
                config=config,
                pooling_mode=PoolingMode.from_string("mean"),  # Adjust pooling mode as needed
                add_layer_norm=True,
                **config_kwargs,
            )
        else:
            self.model = PIXELForSequenceClassification.from_pretrained(
                model_path,
                config=config,
                pooling_mode=PoolingMode.from_string("mean"),  # Adjust pooling mode as needed
                add_layer_norm=True,
                **config_kwargs,
            )            
        # Load the text renderer processor
        self.processor = PangoCairoTextRenderer.from_pretrained(
            model_path,
            cache_dir=None,
            revision="main",
            use_auth_token=None,
            fallback_fonts_dir=self.FALLBACK_FONTS_DIR,
            rgb=False,
        )

        self.processor.max_seq_length = self.SEQ_LEN
        resize_model_embeddings(self.model, self.processor.max_seq_length)
        self.transforms = get_transforms(
            do_resize=True,
            size=(self.processor.pixels_per_patch, self.processor.pixels_per_patch * self.processor.max_seq_length),
        )
        self.format_fn = glue_strip_spaces
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()

    def encode_text(self, texts: List[str], batch_size: int) -> np.ndarray:
        embeddings = []

        for text_batch in tqdm([texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]):
            batch_embeddings = []
            for text in text_batch:
                encoding = self.processor(text=self.format_fn(text))
                pixel_values = self.transforms(Image.fromarray(encoding.pixel_values))
                attention_mask = get_attention_mask(encoding.num_text_patches, seq_length=self.SEQ_LEN)

                # Move tensors to the same device as the model
                pixel_values = pixel_values.to(self.device)
                attention_mask = attention_mask.to(self.device)

                with torch.no_grad():
                    model_output = self.model(pixel_values=pixel_values.unsqueeze(0), attention_mask=attention_mask.unsqueeze(0))
                    batch_embeddings.append(model_output.logits.detach().cpu().numpy())

            embeddings.append(np.concatenate(batch_embeddings, axis=0))
        return np.concatenate(embeddings, axis=0)

    def encode_queries(self, queries: List[str], batch_size: int, **kwargs) -> np.ndarray:
        return self.encode_text(queries, batch_size)

    def encode_corpus(self, corpus: List[Dict[str, str]], batch_size: int, **kwargs) -> np.ndarray:
        if type(corpus) is dict:
            sentences = [(corpus["title"][i] + self.sep + corpus["text"][i]).strip() if "title" in corpus else corpus["text"][i].strip() for i in range(len(corpus['text']))]
        else:
            sentences = [(doc["title"] + self.sep + doc["text"]).strip() if "title" in doc else doc["text"].strip() for doc in corpus]
        return self.encode_text(sentences, batch_size)
    
dataset = "nq"
url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
out_dir = os.path.join("datasets")
data_path = util.download_and_unzip(url, out_dir)
corpus, queries, qrels = GenericDataLoader(data_folder=f"datasets/{dataset}").load(split="test")

model_name = # replace with model name
model = DRES(PixelForRetrieval(model_path=model_name),batch_size=64)
retriever = EvaluateRetrieval(model, score_function="dot") # or "cos_sim" for cosine similarity
results = retriever.retrieve(corpus, queries)
ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)

print(model_name)
print("--------------------------")
print(ndcg, recall)