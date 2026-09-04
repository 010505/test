from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from .progressive import ClassDiffusionModel


ONE_STEP_EXPERIMENTS = (
    "07_one_step_direct",
    "08_one_step_distilled",
    "09_one_step_conditional_drift",
)


class OneStepClassDiffusionModel(ClassDiffusionModel):
    """Class-diffusion student trained and evaluated with one reverse NFE.

    The architecture intentionally remains checkpoint-compatible with the formal
    four-step teacher.  Only the inference path and denoising objective are
    specialized to the direct x_4 -> x_0 prediction used at deployment.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.inference_steps = 1

    def denoising_loss(self, conditions, targets):
        flat = conditions.flatten(0, 1)
        repeated_targets = targets[:, None].expand(-1, conditions.shape[1]).reshape(-1)
        step = self.steps
        steps = torch.full(
            (len(flat),), step, dtype=torch.long, device=flat.device
        )
        clean = F.one_hot(repeated_targets, self.num_classes).to(flat.dtype)
        probabilities = torch.einsum("bi,bij->bj", clean, self.cumulative[steps])
        noisy = torch.multinomial(probabilities, 1).squeeze(1)
        logits = self._denoise_logits(flat, noisy, step)
        return F.cross_entropy(logits, repeated_targets)


@dataclass(frozen=True)
class DriftEntry:
    context: torch.Tensor
    positive: torch.Tensor
    negative: torch.Tensor


class ConditionalDriftMemoryBank:
    """FIFO teacher/student samples separated by class and observation step.

    The bank is a training-only estimator of conditional distributions.  It is
    deliberately not an ``nn.Module`` and is never serialized into deployment
    checkpoints.
    """

    def __init__(self, capacity_per_bucket: int = 64):
        if capacity_per_bucket < 1:
            raise ValueError("capacity_per_bucket must be positive")
        self.capacity_per_bucket = int(capacity_per_bucket)
        self._buckets: dict[tuple[int, int], deque[DriftEntry]] = defaultdict(
            lambda: deque(maxlen=self.capacity_per_bucket)
        )

    def __len__(self):
        return sum(len(bucket) for bucket in self._buckets.values())

    def add(
        self,
        contexts: torch.Tensor,
        positives: torch.Tensor,
        negatives: torch.Tensor,
        labels: torch.Tensor,
    ):
        if contexts.ndim != 3 or positives.ndim != 3 or negatives.ndim != 3:
            raise ValueError("contexts, positives and negatives must have shape [B, S, D]")
        if contexts.shape[:2] != positives.shape[:2] or positives.shape != negatives.shape:
            raise ValueError("drift tensors must agree on batch and observation dimensions")
        if len(labels) != contexts.shape[0]:
            raise ValueError("labels must agree with the drift batch")
        contexts = contexts.detach().to("cpu", dtype=torch.float32)
        positives = positives.detach().to("cpu", dtype=torch.float32)
        negatives = negatives.detach().to("cpu", dtype=torch.float32)
        labels = labels.detach().to("cpu", dtype=torch.long)
        for batch_index in range(contexts.shape[0]):
            label = int(labels[batch_index])
            for update in range(contexts.shape[1]):
                self._buckets[(label, update)].append(DriftEntry(
                    contexts[batch_index, update].clone(),
                    positives[batch_index, update].clone(),
                    negatives[batch_index, update].clone(),
                ))

    def get(
        self,
        label: int,
        update: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        bucket = self._buckets.get((int(label), int(update)))
        if not bucket:
            return None
        contexts = torch.stack([entry.context for entry in bucket]).to(device=device, dtype=dtype)
        positives = torch.stack([entry.positive for entry in bucket]).to(device=device, dtype=dtype)
        negatives = torch.stack([entry.negative for entry in bucket]).to(device=device, dtype=dtype)
        return contexts, positives, negatives


def _normalized_kernel_mean(
    query_probability: torch.Tensor,
    query_context: torch.Tensor,
    candidates: torch.Tensor,
    candidate_contexts: torch.Tensor,
    radius: float,
    context_weight: float,
) -> torch.Tensor:
    probability_distance = torch.linalg.vector_norm(
        candidates - query_probability.unsqueeze(0), dim=-1
    )
    query_context = F.normalize(query_context.unsqueeze(0), dim=-1)
    candidate_contexts = F.normalize(candidate_contexts, dim=-1)
    context_distance = 1.0 - (candidate_contexts * query_context).sum(dim=-1)
    distance = probability_distance + float(context_weight) * context_distance
    scale = distance.detach().mean().clamp_min(1e-3)
    weights = torch.softmax(-distance / (scale * float(radius)), dim=0)
    return torch.einsum("n,nk->k", weights, candidates)


def conditional_categorical_drift_target(
    student_log_probabilities: torch.Tensor,
    teacher_probabilities: torch.Tensor,
    contexts: torch.Tensor,
    labels: torch.Tensor,
    bank: ConditionalDriftMemoryBank,
    radii: Sequence[float] = (0.2, 0.5, 1.0),
    drift_strength: float = 0.25,
    context_weight: float = 0.25,
) -> torch.Tensor:
    """Construct a frozen mirror-descent target on the class simplex.

    Positives are four-step teacher posteriors and negatives are one-step
    student posteriors.  Both are restricted to the same true class and
    observation step, while the kernel additionally favors similar causal
    encoder features.  The field is applied as a logit tilt and projected back
    to the simplex with softmax.
    """

    if student_log_probabilities.shape != teacher_probabilities.shape:
        raise ValueError("student and teacher distributions must have the same shape")
    if student_log_probabilities.ndim != 3 or contexts.ndim != 3:
        raise ValueError("distributions and contexts must have shape [B, S, D]")
    if student_log_probabilities.shape[:2] != contexts.shape[:2]:
        raise ValueError("contexts must align with distributions")
    if not radii or any(radius <= 0 for radius in radii):
        raise ValueError("radii must contain positive values")

    with torch.no_grad():
        output_device = student_log_probabilities.device
        output_dtype = student_log_probabilities.dtype
        # The field consists of many very small kernel reductions.  Computing
        # them on CPU avoids hundreds of tiny GPU launches and repeated bank
        # transfers; only the final frozen target returns to the training GPU.
        student_log_cpu = student_log_probabilities.detach().to("cpu", dtype=torch.float32)
        student_probabilities = student_log_cpu.exp()
        teacher_probabilities = teacher_probabilities.detach().to("cpu", dtype=torch.float32)
        contexts = contexts.detach().to("cpu", dtype=torch.float32)
        labels = labels.detach().to("cpu", dtype=torch.long)
        targets = []
        for batch_index in range(student_probabilities.shape[0]):
            sample_targets = []
            label = int(labels[batch_index])
            for update in range(student_probabilities.shape[1]):
                query = student_probabilities[batch_index, update]
                query_context = contexts[batch_index, update]

                same_label = labels == labels[batch_index]
                current_contexts = contexts[same_label, update]
                current_positives = teacher_probabilities[same_label, update]
                current_negatives = student_probabilities[same_label, update]
                stored = bank.get(label, update, torch.device("cpu"), torch.float32)
                if stored is not None:
                    stored_contexts, stored_positives, stored_negatives = stored
                    candidate_contexts = torch.cat([current_contexts, stored_contexts], dim=0)
                    positives = torch.cat([current_positives, stored_positives], dim=0)
                    negatives = torch.cat([current_negatives, stored_negatives], dim=0)
                else:
                    candidate_contexts = current_contexts
                    positives = current_positives
                    negatives = current_negatives

                field = torch.zeros_like(query)
                for radius in radii:
                    attraction = _normalized_kernel_mean(
                        query, query_context, positives, candidate_contexts,
                        radius, context_weight,
                    )
                    repulsion = _normalized_kernel_mean(
                        query, query_context, negatives, candidate_contexts,
                        radius, context_weight,
                    )
                    component = attraction - repulsion
                    component = component / component.square().mean().sqrt().clamp_min(1e-4)
                    field = field + component
                field = field / len(radii)
                sample_targets.append(torch.softmax(
                    student_log_cpu[batch_index, update]
                    + float(drift_strength) * field,
                    dim=-1,
                ))
            targets.append(torch.stack(sample_targets))
        target = torch.stack(targets).detach()
        bank.add(contexts, teacher_probabilities, student_probabilities, labels)
    return target.to(device=output_device, dtype=output_dtype)


def distillation_loss(
    student_log_probabilities: torch.Tensor,
    teacher_log_probabilities: torch.Tensor,
    temperature: float = 2.0,
) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    student_log_soft = F.log_softmax(student_log_probabilities / temperature, dim=-1)
    teacher_soft = F.softmax(teacher_log_probabilities.detach() / temperature, dim=-1)
    return F.kl_div(
        student_log_soft,
        teacher_soft,
        reduction="batchmean",
    ) * (temperature ** 2) / student_log_probabilities.shape[1]


def build_one_step_model(num_classes: int = 14, dropout: float = 0.15) -> nn.Module:
    return OneStepClassDiffusionModel(num_classes=num_classes, dropout=dropout)
