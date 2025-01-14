import torch

import math
from fairseq import metrics, utils
from dataclasses import dataclass
from fairseq.dataclass import FairseqDataclass
from fairseq.criterions import FairseqCriterion, register_criterion


@register_criterion("bce-loss")
class BCELoss(FairseqCriterion):

    def __init__(self, task):
        super().__init__(task)

    def forward(self, model, sample, reduce=True):

        features, targets = sample
        outputs = model(features)

        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            outputs, targets, reduction="sum" if reduce else "none"
        )
        sample_size = targets.numel()

        decoded_outputs = torch.sigmoid(outputs) > 0.5
        tp = torch.sum((decoded_outputs == 1) & (targets == 1)).item()
        fp = torch.sum((decoded_outputs == 1) & (targets == 0)).item()
        fn = torch.sum((decoded_outputs == 0) & (targets == 1)).item()
        tn = torch.sum((decoded_outputs == 0) & (targets == 0)).item()
        
        logging_output = {
            "loss": loss.data,
            "ntokens": sample_size,
            "nsentences": outputs.size(0),
            "sample_size": sample_size,
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
        }

        return loss, sample_size, logging_output
    
    def logging_outputs_can_be_summed(self):
        return True
    
    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        nsentences = sum(log.get("nsentences", 0) for log in logging_outputs)
        
        tp = sum(log.get("tp", 0) for log in logging_outputs)
        fp = sum(log.get("fp", 0) for log in logging_outputs)
        tn = sum(log.get("tn", 0) for log in logging_outputs)
        fn = sum(log.get("fn", 0) for log in logging_outputs)
        
        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        if sample_size != ntokens:
            metrics.log_scalar(
                "nll_loss", loss_sum / ntokens / math.log(2), ntokens, round=3
            )
            metrics.log_derived(
                "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
            )
        else:
            metrics.log_derived(
                "ppl", lambda meters: utils.get_perplexity(meters["loss"].avg)
            )
            
        # accuracy, recall, precision, f1
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        recall = tp / (tp + fn) if tp + tp else 0
        precision = tp / (tp + fp) if tp + fp else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
        
        metrics.log_scalar("accuracy", accuracy, sample_size, round=3)
        metrics.log_scalar("recall", recall, sample_size if tp + tp else 0, round=3)
        metrics.log_scalar("precision", precision, sample_size if tp + fp else 0, round=3)
        metrics.log_scalar("f1", f1, sample_size if precision + recall else 0, round=3)