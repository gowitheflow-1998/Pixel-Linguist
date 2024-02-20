from typing import List, Optional

import numpy as np
import torch
from tqdm import tqdm

from transformers import Trainer, is_torch_tpu_available
from transformers.utils import logging

from sklearn.metrics.pairwise import paired_cosine_distances, paired_euclidean_distances, paired_manhattan_distances
from scipy.stats import pearsonr, spearmanr

from .utils.optimization import get_cosine_schedule_to_min_lr_with_warmup
from .utils.training import debug_log_inputs

if is_torch_tpu_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.parallel_loader as pl

logger = logging.get_logger(__name__)


class PIXELTrainer(Trainer):
    """
    Same as a regular Trainer but with the option to visualize inputs before they are fed into the model
    for debugging purposes
    """
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None

        # Uncomment this to visualize inputs
        # debug_log_inputs(inputs)

        outputs = model(**inputs)

        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            loss = self.label_smoother(outputs, labels)
        else:
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss

class PIXELTrainerForContrastive(Trainer):
    """
    Same as a regular Trainer but with the option to visualize inputs before they are fed into the model
    for debugging purposes
    """
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if "labels" in inputs:
            labels = inputs.pop("labels")

        mask = [model.config.id2label[int(label)].capitalize() == 'Entailment' for label in labels]

        sentence1 = inputs.pop("sentence1")
        sentence2 = inputs.pop("sentence2")

        outputs_a = model(**sentence1)
        outputs_b = model(**sentence2)

        embeddings_a = outputs_a['logits'][mask]
        embeddings_b = outputs_b['logits'][mask]

        # after pool
        scores = torch.mm(embeddings_a, embeddings_b.transpose(0, 1)) / 0.05

        labels = torch.tensor(range(len(scores)), dtype=torch.long,
                              device=embeddings_a.device)  # Example a[i] should match with b[i]

        loss = (model.loss(scores, labels) + model.loss(scores.transpose(0, 1), labels)) / 2

        outputs = (outputs_a, outputs_b)

        return (loss, outputs) if return_outputs else loss

class PIXELTrainerForContrastiveWithEval(Trainer):
    """
    Same as a regular Trainer but with the option to visualize inputs before they are fed into the model
    for debugging purposes
    """
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if "labels" in inputs:
            labels = inputs.pop("labels")

        # mask = [model.config.id2label[int(label)].capitalize() == 'Entailment' for label in labels]

        sentence1 = inputs.pop("sentence1")
        sentence2 = inputs.pop("sentence2")

        # if 'sentence3' in inputs:
        #     sentence3 = inputs.pop("sentence3")
        #     sentence2 = torch.cat([sentence2, sentence3], dim=0)

        outputs_a = model(**sentence1)
        outputs_b = model(**sentence2)

        embeddings_a = outputs_a['logits']
        embeddings_b = outputs_b['logits']

        # after pool
        scores = torch.mm(embeddings_a, embeddings_b.transpose(0, 1)) / 0.05

        labels = torch.tensor(range(len(scores)), dtype=torch.long,
                              device=embeddings_a.device)  # Example a[i] should match with b[i]

        loss = (model.loss(scores, labels) + model.loss(scores.transpose(0, 1), labels)) / 2
        # loss = model.loss(scores, labels)

        outputs = (outputs_a, outputs_b)

        return (loss, outputs) if return_outputs else loss

    def evaluate(self, ignore_keys=None, metric_key_prefix: str = "eval"):

        logger.info("*** Training Evaluate ***")

        total_output_a = []
        total_output_b = []

        args = self.args
        model = self.model.to(args.device)

        model.eval()
        eval_dataloader = self.get_eval_dataloader(self.eval_dataset)
        batch_size = eval_dataloader.batch_size
        with torch.no_grad():
            for step, inputs in enumerate(tqdm(eval_dataloader)):
            # for step in tqdm(range(0, len(self.eval_dataset), bs)):
                # inputs = [self.eval_dataset[step + idx] for idx in range(0, min(bs, len(self.eval_dataset) - step))]
                sentence1 = inputs.pop("sentence1")
                sentence2 = inputs.pop("sentence2")

                sentence1 = {k: v.to(args.device) for k, v in sentence1.items()}
                sentence2 = {k: v.to(args.device) for k, v in sentence2.items()}

                outputs_a = model(**sentence1).logits
                outputs_b = model(**sentence2).logits

                total_output_a.append(outputs_a.detach().cpu())
                total_output_b.append(outputs_b.detach().cpu())

        embeddings1 = torch.cat(total_output_a, dim=0)
        embeddings2 = torch.cat(total_output_b, dim=0)
        labels = [n['label'] for n in self.eval_dataset]

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

        metrics = {}
        metrics['eval_loss'] = 0  # for ignore error obly
        metrics['pearson_cosine'] = eval_pearson_cosine
        metrics['spearman_cosine'] = eval_spearman_cosine
        metrics['pearson_manhattan'] = eval_pearson_manhattan
        metrics['spearman_manhattan'] = eval_spearman_manhattan
        metrics['pearson_euclidean'] = eval_pearson_euclidean
        metrics['spearman_euclidean'] = eval_spearman_euclidean
        metrics['pearson_dot'] = eval_pearson_dot
        metrics['spearman_dot'] = eval_spearman_dot

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        # self.log(metrics)
        logger.info("Cosine-Similarity :\tPearson: {:.4f}\tSpearman: {:.4f}".format(
            eval_pearson_cosine, eval_spearman_cosine))
        logger.info("Manhattan-Distance:\tPearson: {:.4f}\tSpearman: {:.4f}".format(
            eval_pearson_manhattan, eval_spearman_manhattan))
        logger.info("Euclidean-Distance:\tPearson: {:.4f}\tSpearman: {:.4f}".format(
            eval_pearson_euclidean, eval_spearman_euclidean))
        logger.info("Dot-Product-Similarity:\tPearson: {:.4f}\tSpearman: {:.4f}".format(
            eval_pearson_dot, eval_spearman_dot))

        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
        return metrics


class PIXELTrainerForPretraining(PIXELTrainer):
    """
    PIXELTrainer for pretraining
    """

    def create_scheduler(self, num_training_steps: int, optimizer: torch.optim.Optimizer = None):
        """
        Setup the scheduler. The optimizer of the trainer must have been set up either before this method is called or
        passed as an argument.
        Args:
            num_training_steps (int): The number of training steps to do.
        """
        if self.lr_scheduler is None:
            self.lr_scheduler = get_cosine_schedule_to_min_lr_with_warmup(
                self.optimizer if optimizer is None else optimizer,
                self.args.get_warmup_steps(num_training_steps),
                num_training_steps,
                self.args.learning_rate,
            )
        return self.lr_scheduler