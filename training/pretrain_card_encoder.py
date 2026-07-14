from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.card_dataset import CardDataset, collate_cards
from data.card_preprocessing import DEFAULT_CACHE_DIR, MASK_TOKEN, NULL_TOKEN, SCHEMA_VERSION
from models.card_encoder import CardEncoder
from models.card_pretrain_heads import (
    CardDetailOwnershipHead,
    CardRelationHead,
    MaskedCardFieldHeads,
    MaskedDetailHeads,
    card_detail_ownership_loss,
    card_relation_loss,
    masked_card_field_loss,
    masked_detail_loss,
)


TRAINING_SCHEMA_VERSION = "static_card_training_v3"
SPLIT_SCHEMA_VERSION = "static_card_component_split_v3"
TASK_NAMES = (
    "field_recovery",
    "detail_attributes",
    "text_mlm",
    "structure_reference",
    "card_detail_matching",
)
TINY_PENDING_COVERAGE = "PENDING_V3_COVERAGE_SCAN"
DATA_SCHEMA_VERSION = SCHEMA_VERSION


def load_config(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - Kaggle and the project env include PyYAML.
        raise RuntimeError("PyYAML is required to read the static v2 training config") from exc
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} did not contain a mapping")
    return value


def validate_formal_config(config: dict[str, Any]) -> None:
    if config.get("schema_version") != TRAINING_SCHEMA_VERSION:
        raise ValueError(f"config must declare schema_version: {TRAINING_SCHEMA_VERSION}")
    if config.get("run_mode") != "static_v3_formal":
        raise ValueError("formal v3 config must declare run_mode: static_v3_formal")
    tokenizer = config.get("tokenizer") or {}
    expected_tokenizer = {
        "tokenizer_type": "sentencepiece",
        "model_type": "unigram",
        "vocab_size": 1024,
        "hard_vocab_limit": False,
        "character_coverage": 1.0,
        "byte_fallback": False,
        "pad_id": 0,
        "unk_id": 1,
        "bos_id": -1,
        "eos_id": -1,
        "user_defined_symbols": ["[MASK_TEXT]"],
        "input_sentence_size": 0,
        "shuffle_input_sentence": False,
        "max_text_subword_tokens": 256,
    }
    for name, expected in expected_tokenizer.items():
        if tokenizer.get(name) != expected:
            raise ValueError(f"formal v3 tokenizer.{name} must be {expected!r}")
    if Path(str(tokenizer.get("output_dir"))) != Path("outputs/static_v3_formal/tokenizer"):
        raise ValueError("formal v3 tokenizer output_dir is not isolated")
    data = config.get("data") or {}
    if Path(str(data.get("cache_dir"))) != Path("artifacts/card_data_v3"):
        raise ValueError("formal v3 cache_dir must be artifacts/card_data_v3")
    if data.get("split_mode") != "connected_component":
        raise ValueError("formal v3 split_mode must be connected_component")
    if float(data.get("validation_ratio", -1)) != 0.10 or float(data.get("test_ratio", -1)) != 0.10:
        raise ValueError("formal v3 split must be 80/10/10")

    expected_weights = {
        "field_recovery": 1.0,
        "detail_attributes": 1.0,
        "text_mlm": 0.5,
        "structure_reference": 0.5,
        "card_detail_matching": 1.0,
    }
    tasks = config.get("tasks") or {}
    if set(tasks) != set(TASK_NAMES) or not all(tasks.get(name) is True for name in TASK_NAMES):
        raise ValueError(f"formal v3 must enable exactly the five frozen tasks: {TASK_NAMES}")
    weights = config.get("loss_weights") or {}
    if {name: float(weights.get(name, math.nan)) for name in TASK_NAMES} != expected_weights:
        raise ValueError(f"formal v3 loss weights must equal {expected_weights}")

    training = config.get("training") or {}
    micro_batch_size = int(training.get("micro_batch_size", 0))
    accumulation = int(training.get("gradient_accumulation_steps", 0))
    effective_batch_size = int(training.get("effective_batch_size", 0))
    if micro_batch_size <= 0 or accumulation <= 0:
        raise ValueError("micro batch size and gradient accumulation must be positive")
    if micro_batch_size * accumulation != 32 or effective_batch_size != 32:
        raise ValueError(
            "formal v3 requires micro_batch_size * gradient_accumulation_steps == "
            "effective_batch_size == 32; OOM changes must be explicit in config"
        )
    if int(training.get("max_epochs", 0)) != 100:
        raise ValueError("formal v3 selection must run a 100-epoch horizon")
    if int(training.get("eval_every_epochs", 0)) != 1:
        raise ValueError("formal v3 must validate after every epoch")
    optimizer = training.get("optimizer") or {}
    if optimizer.get("name") != "AdamW":
        raise ValueError("formal v3 optimizer must be AdamW")
    if float(optimizer.get("weight_decay", -1)) != 0.01:
        raise ValueError("formal v3 AdamW weight_decay must be 0.01")
    if [float(value) for value in optimizer.get("betas", [])] != [0.9, 0.95]:
        raise ValueError("formal v3 AdamW betas must be [0.9, 0.95]")
    scheduler = training.get("scheduler") or {}
    if (
        scheduler.get("name") != "linear_warmup_cosine"
        or scheduler.get("update_unit") != "optimizer_step"
        or int(scheduler.get("horizon_epochs", 0)) != 100
        or float(scheduler.get("warmup_ratio", -1)) != 0.05
    ):
        raise ValueError(
            "formal v3 requires 5% linear warmup followed by optimizer-step cosine "
            "over a 100-epoch horizon"
        )
    if float(training.get("gradient_clip_norm", -1)) != 1.0:
        raise ValueError("formal v3 gradient_clip_norm must be 1.0")
    if str(training.get("device")) != "cuda":
        raise ValueError("formal v3 training must explicitly require CUDA")
    masking = training.get("masking") or {}
    expected_masking = {
        "card_field_group_probability": 0.15,
        "text_token_probability": 0.15,
        "structure_reference_probability": 0.15,
    }
    if set(masking) != set(expected_masking) or any(
        float(masking.get(name, -1)) != value for name, value in expected_masking.items()
    ):
        raise ValueError(f"formal v3 masking contract must equal {expected_masking}")
    early = training.get("early_stopping") or {}
    if (
        early.get("enabled") is not True
        or early.get("metric") != "validation_weighted_total_loss"
        or int(early.get("patience", 0)) != 12
        or float(early.get("min_delta", -1)) != 0.0001
        or early.get("restore_best_checkpoint") is not True
    ):
        raise ValueError("formal v3 early-stopping contract does not match the frozen values")

    tiny = config.get("tiny_overfit") or {}
    expected_tiny = {
        "card_count": 16,
        "max_steps": 500,
        "batch_size_cards": 16,
        "learning_rate": 1.0e-3,
        "weight_decay": 0.0,
        "gradient_clip_norm": 1.0,
    }
    for name, value in expected_tiny.items():
        actual = tiny.get(name)
        if isinstance(value, float):
            if float(actual if actual is not None else math.nan) != value:
                raise ValueError(f"tiny_overfit.{name} must be {value}")
        elif int(actual or 0) != value:
            raise ValueError(f"tiny_overfit.{name} must be {value}")
    thresholds = tiny.get("thresholds") or {}
    if float(thresholds.get("total_loss_ratio", -1)) != 0.30:
        raise ValueError("tiny total-loss ratio threshold must be 0.30")
    if float(thresholds.get("per_active_task_loss_ratio", -1)) != 0.50:
        raise ValueError("tiny per-task loss ratio threshold must be 0.50")
    if not bool(tiny.get("require_finite")) or not bool(tiny.get("require_save_reload")):
        raise ValueError("tiny finite and save/reload gates must be enabled")
    # This is deliberately a formal-entry guard. A pending or dynamically
    # selected tiny set is never accepted by the single cloud kernel.
    if tiny.get("selection_state") != "frozen_v3_coverage_scan":
        raise ValueError("tiny card IDs are pending the required v3 coverage scan")
    if len(tiny.get("card_ids") or []) != 16:
        raise ValueError("formal v3 requires exactly 16 frozen tiny-overfit Card IDs")
    if tiny.get("coverage_sha256") in (None, "", TINY_PENDING_COVERAGE):
        raise ValueError("formal v3 requires the frozen tiny-overfit coverage SHA256")

    refit = config.get("production_refit") or {}
    if not bool(refit.get("enabled")) or refit.get("epoch_source") != "best_validation_epoch":
        raise ValueError("production refit must use the selected best validation epoch")
    if int(refit.get("scheduler_horizon_epochs", 0)) != 100:
        raise ValueError("production refit cosine horizon must remain 100 epochs")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def clone_batch(batch: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value.clone() if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def _move_masks(masks: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in masks.items()}


def _sample_mask(
    valid: torch.Tensor,
    probability: float,
    generator: torch.Generator,
    *,
    ensure_one: bool = True,
) -> torch.Tensor:
    valid_cpu = valid.detach().bool().cpu()
    selected = (torch.rand(valid_cpu.shape, generator=generator) < float(probability)) & valid_cpu
    if ensure_one and bool(valid_cpu.any().item()) and not bool(selected.any().item()):
        first = torch.nonzero(valid_cpu, as_tuple=False)[0]
        selected[tuple(int(value) for value in first.tolist())] = True
    return selected.to(valid.device)


def _vocab_id(schema: dict[str, Any], field: str, token: str) -> int:
    vocab = schema["vocab"][field]
    if token not in vocab:
        raise KeyError(f"{field} vocab has no {token!r} token")
    return int(vocab[token])


@dataclass
class MaskedInputs:
    batch: dict[str, Any]
    card_masks: dict[str, torch.Tensor]
    detail_masks: dict[str, torch.Tensor]
    mlm_labels: torch.Tensor
    mlm_mask: torch.Tensor

    def to(self, device: torch.device) -> "MaskedInputs":
        return MaskedInputs(
            batch=move_batch(self.batch, device),
            card_masks=_move_masks(self.card_masks, device),
            detail_masks=_move_masks(self.detail_masks, device),
            mlm_labels=self.mlm_labels.to(device),
            mlm_mask=self.mlm_mask.to(device),
        )


def mask_training_inputs(
    batch: dict[str, Any],
    schema: dict[str, Any],
    masking: dict[str, Any],
    *,
    seed: int,
) -> MaskedInputs:
    """Create explicit v2 masks and corrupt only the corresponding inputs."""

    generator = torch.Generator().manual_seed(int(seed))
    masked = clone_batch(batch)
    category = batch["card_category_ids"]
    pokemon_id = _vocab_id(schema, "card_category", "POKEMON")
    trainer_id = _vocab_id(schema, "card_category", "TRAINER")
    energy_id = _vocab_id(schema, "card_category", "ENERGY")
    pokemon = category == pokemon_id
    trainer = category == trainer_id
    energy = category == energy_id

    categorical_probability = float(masking.get("categorical_probability", 0.10))
    numeric_probability = float(masking.get("numeric_probability", 0.10))
    rule_probability = float(masking.get("rule_probability", categorical_probability))
    detail_probability = float(masking.get("detail_field_probability", 0.10))
    text_probability = float(masking.get("text_token_probability", 0.15))

    card_masks: dict[str, torch.Tensor] = {}
    categorical_fields = {
        "stage": ("stage_ids", pokemon),
        "pokemon_type": ("pokemon_type_ids", pokemon),
        "weakness_type": ("weakness_type_ids", pokemon),
        "resistance_type": ("resistance_type_ids", pokemon),
        "trainer_subtype": ("trainer_subtype_ids", trainer),
        "energy_subtype": ("energy_subtype_ids", energy),
    }
    for field, (batch_key, applicable) in categorical_fields.items():
        selected = _sample_mask(
            applicable,
            categorical_probability,
            generator,
            ensure_one=categorical_probability > 0,
        )
        masked[batch_key][selected] = _vocab_id(schema, field, MASK_TOKEN)
        card_masks[field] = selected

    all_cards = torch.ones_like(category, dtype=torch.bool)
    rule_selected = _sample_mask(
        all_cards,
        rule_probability,
        generator,
        ensure_one=rule_probability > 0,
    )
    masked["rule_flag_multihot"][rule_selected] = 0.0
    card_masks["rule_flags"] = rule_selected

    # TERA is represented both as a canonical rule flag and as source-derived
    # card tags (TERA / TERA_TYPE_*).  Leaving either tag visible would make
    # masked rule recovery a direct alias lookup.  Resolve columns from the
    # schema rather than assuming any fixed multihot ordering.
    card_tag_vocab = schema.get("card_tag_vocab")
    if not isinstance(card_tag_vocab, dict):
        card_tag_vocab = schema.get("vocab", {}).get("card_tags", {})
    tera_tag_indices = sorted(
        int(index)
        for token, index in card_tag_vocab.items()
        if str(token).upper() == "TERA" or str(token).upper().startswith("TERA_TYPE_")
    )
    if tera_tag_indices and bool(rule_selected.any().item()):
        selected_rows = torch.nonzero(rule_selected, as_tuple=False).flatten()
        selected_tags = torch.tensor(
            tera_tag_indices,
            dtype=torch.long,
            device=masked["card_tag_multihot"].device,
        )
        masked["card_tag_multihot"][selected_rows[:, None], selected_tags[None, :]] = 0.0

    card_tag_probability = float(masking.get("card_tag_probability", rule_probability))
    card_tag_selected = _sample_mask(
        all_cards,
        card_tag_probability,
        generator,
        ensure_one=card_tag_probability > 0,
    )
    masked["card_tag_multihot"][card_tag_selected] = 0.0
    card_masks["card_tags"] = card_tag_selected

    for field, value_key, applicability_key in (
        ("printed_hp", "printed_hp", "printed_hp_mask"),
        ("retreat", "retreat", "retreat_mask"),
    ):
        selected = _sample_mask(
            batch[applicability_key] > 0,
            numeric_probability,
            generator,
            ensure_one=numeric_probability > 0,
        )
        masked[value_key][selected] = 0.0
        masked[applicability_key][selected] = 0.0
        card_masks[field] = selected

    detail_valid = batch["detail_mask"] > 0
    attack_valid = batch["attack_energy_mask"] > 0
    damage_valid = batch["attack_damage_mask"] > 0
    detail_masks: dict[str, torch.Tensor] = {}

    subtype_selected = _sample_mask(
        detail_valid,
        detail_probability,
        generator,
        ensure_one=detail_probability > 0,
    )
    masked["detail_subtype_ids"][subtype_selected] = _vocab_id(schema, "detail_subtype", MASK_TOKEN)
    detail_masks["detail_subtype"] = subtype_selected

    energy_selected = _sample_mask(
        attack_valid,
        detail_probability,
        generator,
        ensure_one=detail_probability > 0,
    )
    masked["attack_energy_counts"][energy_selected] = 0.0
    detail_masks["energy_counts"] = energy_selected

    damage_selected = _sample_mask(
        damage_valid,
        detail_probability,
        generator,
        ensure_one=detail_probability > 0,
    )
    masked["attack_base_damage"][damage_selected] = 0.0
    masked["attack_damage_mask"][damage_selected] = 0.0
    detail_masks["base_damage"] = damage_selected

    mode_selected = _sample_mask(
        attack_valid,
        detail_probability,
        generator,
        ensure_one=detail_probability > 0,
    )
    masked["attack_damage_mode"][mode_selected] = _vocab_id(schema, "damage_mode", MASK_TOKEN)
    detail_masks["damage_mode"] = mode_selected

    mlm_labels = batch["detail_text_ids"].clone()
    token_valid = (batch["detail_text_mask"] > 0) & detail_valid.unsqueeze(-1)
    mlm_mask = _sample_mask(
        token_valid,
        text_probability,
        generator,
        ensure_one=text_probability > 0,
    )
    masked["detail_text_ids"][mlm_mask] = int(schema["text_mask_id"])

    return MaskedInputs(masked, card_masks, detail_masks, mlm_labels, mlm_mask)


class StaticPretrainingModel(nn.Module):
    def __init__(self, schema: dict[str, Any], model_config: dict[str, Any]) -> None:
        super().__init__()
        embedding_dim = int(model_config.get("embedding_dim", 128))
        detail_dim = int(model_config.get("detail_token_dim", embedding_dim))
        self.encoder = CardEncoder(
            schema,
            embedding_dim=embedding_dim,
            detail_token_dim=detail_dim,
            num_heads=int(model_config.get("attention_heads", 4)),
            transformer_layers=int(model_config.get("transformer_layers", 2)),
            ffn_dim=int(model_config.get("ffn_dim", 256)),
            dropout=float(model_config.get("dropout", 0.0)),
        )
        self.card_heads = MaskedCardFieldHeads(schema, embedding_dim)
        self.detail_heads = MaskedDetailHeads(schema, detail_dim, text_state_dim=detail_dim)
        self.ownership_head = CardDetailOwnershipHead(embedding_dim)
        self.relation_head = CardRelationHead(embedding_dim)


def _item_for_index(dataset: CardDataset, index: int) -> dict[str, Any]:
    start = dataset.detail_offsets[index]
    end = dataset.detail_offsets[index + 1]
    card = dataset.cards[index]
    return {
        "index": index,
        "card_id": card["card_id"],
        "card": card,
        "record": card,
        "details": dataset.details[start:end],
        "schema": dataset.schema,
    }


@dataclass
class RelationInputs:
    left: dict[str, Any]
    right: dict[str, Any]
    labels: dict[str, torch.Tensor]
    masks: dict[str, torch.Tensor]
    diagnostics: dict[str, float] = field(default_factory=dict)

    def to(self, device: torch.device) -> "RelationInputs":
        return RelationInputs(
            left=move_batch(self.left, device),
            right=move_batch(self.right, device),
            labels=_move_masks(self.labels, device),
            masks=_move_masks(self.masks, device),
            diagnostics=dict(self.diagnostics),
        )


def build_relation_inputs(
    dataset: CardDataset,
    *,
    batch_size: int,
    seed: int,
) -> RelationInputs | None:
    rng = random.Random(seed)
    relation_pairs = dataset.relation_samples()
    positive_by_name = {
        "same_name": set(relation_pairs.get("same_name", [])),
        "same_species": set(relation_pairs.get("same_species", [])),
        "direct_evolution": set(relation_pairs.get("evolves_to", [])),
    }
    allowed = list(dataset.indices)
    if len(allowed) < 2:
        return None
    per_relation = max(1, int(batch_size) // 6)
    examples: list[tuple[int, int, str, float]] = []
    diagnostics: dict[str, float] = {}

    signature_tiers: dict[str, list[tuple[str, tuple[str, ...], bool]]] = {
        "same_name": [
            ("category_card_type_stage", ("card_category", "card_type", "stage"), False),
            ("category_card_type", ("card_category", "card_type"), False),
        ],
        "same_species": [
            ("pokemon_stage_type", ("stage", "pokemon_type"), True),
            ("pokemon_stage", ("stage",), True),
            ("pokemon_type", ("pokemon_type",), True),
        ],
        "direct_evolution": [
            ("category_card_type_stage", ("card_category", "card_type", "stage"), False),
            ("category_card_type", ("card_category", "card_type"), False),
            ("category_stage", ("card_category", "stage"), False),
        ],
    }

    def matching_indices(reference: int, keys: tuple[str, ...], pokemon_only: bool) -> list[int]:
        reference_card = dataset.cards[reference]
        signature = tuple(reference_card.get(key) for key in keys)
        return [
            index
            for index in allowed
            if (not pokemon_only or dataset.cards[index].get("card_category") == "POKEMON")
            and tuple(dataset.cards[index].get(key) for key in keys) == signature
        ]

    def matched_negative(
        relation_name: str,
        reference_pair: tuple[int, int],
        used: set[tuple[int, int]],
    ) -> tuple[tuple[int, int] | None, str | None]:
        for tier_name, keys, pokemon_only in signature_tiers[relation_name]:
            left_candidates = matching_indices(reference_pair[0], keys, pokemon_only)
            right_candidates = matching_indices(reference_pair[1], keys, pokemon_only)
            rng.shuffle(left_candidates)
            rng.shuffle(right_candidates)
            for left_index in left_candidates:
                for right_index in right_candidates:
                    pair = (left_index, right_index)
                    if (
                        left_index == right_index
                        or pair in positive_by_name[relation_name]
                        or pair in used
                    ):
                        continue
                    return pair, tier_name
        return None, None

    for name in CardRelationHead.RELATIONS:
        positives = sorted(positive_by_name[name])
        if not positives:
            continue
        selected = rng.sample(positives, min(per_relation, len(positives)))
        used: set[tuple[int, int]] = set()
        for reference_pair in selected:
            pair, tier_name = matched_negative(name, reference_pair, used)
            if pair is None or tier_name is None:
                diagnostics[f"relation_negative_{name}_unmatched"] = (
                    diagnostics.get(f"relation_negative_{name}_unmatched", 0.0) + 1.0
                )
                continue
            used.add(pair)
            examples.append((reference_pair[0], reference_pair[1], name, 1.0))
            examples.append((pair[0], pair[1], name, 0.0))
            metric_name = f"relation_negative_{name}_{tier_name}"
            diagnostics[metric_name] = diagnostics.get(metric_name, 0.0) + 1.0
    if not examples:
        return None
    rng.shuffle(examples)
    left = collate_cards([_item_for_index(dataset, row[0]) for row in examples], dataset.schema)
    right = collate_cards([_item_for_index(dataset, row[1]) for row in examples], dataset.schema)

    relation_names = [row[2] for row in examples]
    labels = {name: torch.zeros(len(examples), dtype=torch.float32) for name in CardRelationHead.RELATIONS}
    masks = {name: torch.zeros(len(examples), dtype=torch.bool) for name in CardRelationHead.RELATIONS}
    for index, (_left, _right, name, label) in enumerate(examples):
        labels[name][index] = float(label)
        masks[name][index] = True

    # Prevent identity aliases from directly giving away Task D. Card names and
    # species are derived from the same source name, so masking only the target
    # id would leave a near-label-equivalent shortcut in the other field.
    for row_index, relation_name in enumerate(relation_names):
        if relation_name == "same_name":
            fields = (
                ("card_name_ids", "card_name"),
                ("species_ids", "species"),
            )
        else:
            fields = (
                ("card_name_ids", "card_name"),
                ("species_ids", "species"),
                ("previous_species_ids", "previous_species"),
                ("evolves_from_name_ids", "evolves_from_name"),
                ("evolves_to_name_ids", "evolves_to_name"),
            )
        for batch_key, vocab_key in fields:
            left[batch_key][row_index] = _vocab_id(dataset.schema, vocab_key, MASK_TOKEN)
            right[batch_key][row_index] = _vocab_id(dataset.schema, vocab_key, MASK_TOKEN)
    diagnostics["relation_negative_count"] = float(
        sum(1 for _left, _right, _name, label in examples if label == 0.0)
    )
    return RelationInputs(left, right, labels, masks, diagnostics)


def ownership_task(
    model: StaticPretrainingModel,
    batch: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, float]]:
    detail_mask = batch["detail_mask"] > 0
    selected_rows: list[int] = []
    selected_slots: list[int] = []
    for row in range(detail_mask.shape[0]):
        positions = torch.nonzero(detail_mask[row], as_tuple=False).flatten()
        if positions.numel():
            selected_rows.append(row)
            selected_slots.append(int(positions[0].item()))
    if len(selected_rows) < 2:
        zero = next(model.ownership_head.parameters()).sum() * 0.0
        return zero, {"ownership_examples": 0.0, "ownership_accuracy": 0.0}

    full_output = model.encoder(batch, return_details=True)
    leave_one_out = clone_batch(batch)
    for row, slot in zip(selected_rows, selected_slots):
        leave_one_out["detail_mask"][row, slot] = 0.0
        leave_one_out["detail_text_mask"][row, slot] = 0.0
    owner_output = model.encoder(leave_one_out, return_details=True)

    summaries = torch.stack([owner_output.card_summary[row] for row in selected_rows])
    candidates = torch.stack(
        [full_output.pre_fusion_detail_tokens[row, slot] for row, slot in zip(selected_rows, selected_slots)]
    )
    types = [int(batch["detail_type_ids"][row, slot].item()) for row, slot in zip(selected_rows, selected_slots)]

    owner_rows: list[torch.Tensor] = []
    candidate_rows: list[torch.Tensor] = []
    labels: list[float] = []
    for index, detail_type in enumerate(types):
        owner_rows.append(summaries[index])
        candidate_rows.append(candidates[index])
        labels.append(1.0)
        negative = next(
            (other for other, other_type in enumerate(types) if other != index and other_type == detail_type),
            None,
        )
        if negative is not None:
            owner_rows.append(summaries[index])
            candidate_rows.append(candidates[negative])
            labels.append(0.0)
    owner_tensor = torch.stack(owner_rows)
    candidate_tensor = torch.stack(candidate_rows)
    label_tensor = torch.tensor(labels, dtype=torch.float32, device=owner_tensor.device)
    logits = model.ownership_head(owner_tensor, candidate_tensor)
    mask = torch.ones_like(label_tensor, dtype=torch.bool)
    loss = card_detail_ownership_loss(logits, label_tensor, mask)
    accuracy = ((logits >= 0) == (label_tensor >= 0.5)).float().mean()
    return loss, {
        "ownership_examples": float(label_tensor.numel()),
        "ownership_accuracy": float(accuracy.detach().cpu()),
    }


def relation_task(
    model: StaticPretrainingModel,
    relation: RelationInputs | None,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    if relation is None:
        zero = next(model.relation_head.parameters()).sum() * 0.0
        return zero, {"relation_examples": 0.0}
    relation = relation.to(device)
    left = model.encoder(relation.left)
    right = model.encoder(relation.right)
    predictions = model.relation_head(left, right)
    loss, metrics = card_relation_loss(predictions, relation.labels, relation.masks)
    metrics["relation_examples"] = float(sum(int(mask.sum().item()) for mask in relation.masks.values()))
    metrics.update(relation.diagnostics)
    return loss, metrics


def compute_losses(
    model: StaticPretrainingModel,
    original_batch: dict[str, Any],
    masked: MaskedInputs,
    relation: RelationInputs | None,
    config: dict[str, Any],
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, float]]:
    output = model.encoder(masked.batch, return_details=True)
    card_predictions = model.card_heads(output.card_summary)
    card_loss, card_metrics = masked_card_field_loss(card_predictions, original_batch, masked.card_masks)

    detail_predictions = model.detail_heads(output.detail_tokens, output.text_token_states)
    detail_loss, detail_metrics = masked_detail_loss(
        detail_predictions,
        original_batch,
        masked.detail_masks,
    )
    mlm_loss, mlm_metrics = masked_detail_loss(
        detail_predictions,
        original_batch,
        {},
        mlm_labels=masked.mlm_labels,
        mlm_mask=masked.mlm_mask,
    )
    ownership_loss, ownership_metrics = ownership_task(model, original_batch)
    relation_loss, relation_metrics = relation_task(model, relation, device)

    task_losses = {
        "field_recovery": card_loss,
        "detail_attributes": detail_loss,
        "text_mlm": mlm_loss,
        "structure_reference": relation_loss,
        "card_detail_matching": ownership_loss,
    }
    weights = config.get("loss_weights", {})
    total = sum(float(weights.get(name, 1.0)) * value for name, value in task_losses.items())

    task_sample_counts = {
        "field_recovery": float(sum(int(mask.sum().item()) for mask in masked.card_masks.values())),
        "detail_attributes": float(sum(int(mask.sum().item()) for mask in masked.detail_masks.values())),
        "text_mlm": float(masked.mlm_mask.sum().item()),
        "structure_reference": float(relation_metrics.get("relation_examples", 0.0)),
        "card_detail_matching": float(ownership_metrics.get("ownership_examples", 0.0)),
    }
    metrics = {
        "total_loss": float(total.detach().cpu()),
        **{f"{name}_loss": float(value.detach().cpu()) for name, value in task_losses.items()},
        **{f"{name}_valid_samples": value for name, value in task_sample_counts.items()},
        **{f"card_{key}": value for key, value in card_metrics.items()},
        **{f"detail_{key}": value for key, value in detail_metrics.items()},
        **{f"mlm_{key}": value for key, value in mlm_metrics.items()},
        **ownership_metrics,
        **relation_metrics,
    }
    return total, task_losses, metrics


def average_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    values: dict[str, list[float]] = {}
    for row in rows:
        for key, value in row.items():
            if math.isfinite(float(value)):
                values.setdefault(key, []).append(float(value))
    return {key: sum(items) / len(items) for key, items in values.items() if items}


def validate_metric_integrity(
    metrics: dict[str, Any],
    *,
    require_task_samples: bool,
    context: str,
) -> dict[str, Any]:
    non_finite = [
        name
        for name, value in metrics.items()
        if isinstance(value, (int, float)) and not math.isfinite(float(value))
    ]
    missing_losses = [
        name for name in ("total_loss", *(f"{task}_loss" for task in TASK_NAMES)) if name not in metrics
    ]
    sample_counts = {
        task: float(metrics.get(f"{task}_valid_samples", 0.0))
        for task in TASK_NAMES
    }
    missing_samples = [task for task, value in sample_counts.items() if value <= 0]
    passed = not non_finite and not missing_losses and (not require_task_samples or not missing_samples)
    report = {
        "context": context,
        "passed": passed,
        "non_finite_metrics": non_finite,
        "missing_loss_metrics": missing_losses,
        "task_valid_samples": sample_counts,
        "tasks_without_valid_samples": missing_samples,
    }
    if not passed:
        raise RuntimeError(f"{context} integrity gate failed: {json.dumps(report, ensure_ascii=False)}")
    return report


def gradient_norm(parameters: Iterable[nn.Parameter]) -> float:
    total = 0.0
    for parameter in parameters:
        if parameter.grad is not None:
            total += float(parameter.grad.detach().float().pow(2).sum().cpu())
    return total**0.5


def _assert_finite_tensor(value: torch.Tensor, name: str) -> None:
    if not bool(torch.isfinite(value.detach()).all().item()):
        raise FloatingPointError(f"non-finite tensor in {name}")


def _assert_finite_gradients(model: nn.Module) -> float:
    squared_norm = 0.0
    gradient_count = 0
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        _assert_finite_tensor(parameter.grad, f"gradient:{name}")
        squared_norm += float(parameter.grad.detach().float().pow(2).sum().cpu())
        gradient_count += 1
    if gradient_count == 0:
        raise RuntimeError("optimizer step has no gradients")
    norm = squared_norm**0.5
    if not math.isfinite(norm):
        raise FloatingPointError("aggregate gradient norm is non-finite")
    return norm


def optimizer_steps_per_epoch(loader_length: int, accumulation_steps: int) -> int:
    if loader_length <= 0 or accumulation_steps <= 0:
        raise ValueError("loader length and accumulation steps must be positive")
    return math.ceil(loader_length / accumulation_steps)


def build_optimizer_scheduler(
    model: nn.Module,
    training_config: dict[str, Any],
    *,
    steps_per_epoch: int,
    horizon_epochs: int | None = None,
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]:
    optimizer_config = training_config.get("optimizer") or {}
    scheduler_config = training_config.get("scheduler") or {}
    if optimizer_config.get("name") != "AdamW":
        raise ValueError("only the frozen AdamW optimizer is supported")
    if scheduler_config.get("name") != "linear_warmup_cosine":
        raise ValueError("only the frozen linear_warmup_cosine scheduler is supported")
    if scheduler_config.get("update_unit") != "optimizer_step":
        raise ValueError("cosine scheduler must update on each optimizer step")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(optimizer_config["learning_rate"]),
        weight_decay=float(optimizer_config["weight_decay"]),
        betas=tuple(float(value) for value in optimizer_config.get("betas", [0.9, 0.95])),
    )
    horizon = int(horizon_epochs or scheduler_config.get("horizon_epochs", 100))
    if horizon != 100:
        raise ValueError("formal v3 cosine horizon must be exactly 100 epochs")
    total_steps = int(steps_per_epoch) * horizon
    warmup_ratio = float(scheduler_config.get("warmup_ratio", 0.05))
    if warmup_ratio != 0.05:
        raise ValueError("formal v3 warmup_ratio must be 0.05")
    if float(scheduler_config.get("eta_min", 0.0)) != 0.0:
        raise ValueError("formal v3 eta_min must be 0.0")
    warmup_steps = max(1, round(total_steps * warmup_ratio))

    def lr_factor(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = min(1.0, float(step - warmup_steps) / float(max(1, total_steps - warmup_steps)))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_factor)
    # Durable attributes make the frozen horizon auditable without relying on
    # the non-serializable lambda implementation.
    scheduler.v3_total_steps = total_steps  # type: ignore[attr-defined]
    scheduler.v3_warmup_steps = warmup_steps  # type: ignore[attr-defined]
    scheduler.v3_horizon_epochs = horizon  # type: ignore[attr-defined]
    return optimizer, scheduler


def _structural_edges(dataset: CardDataset) -> list[tuple[int, int, str]]:
    """Return every relationship that must stay inside one split component."""

    edges: set[tuple[int, int, str]] = set()
    relation_aliases = {
        "same_name": "same_name",
        "same_species": "same_species",
        "evolves_to": "direct_evolution",
        "direct_evolution": "direct_evolution",
    }
    for source_name, pairs in dataset.relation_samples().items():
        relation_name = relation_aliases.get(str(source_name), str(source_name))
        for left, right in pairs:
            left_index, right_index = int(left), int(right)
            if left_index == right_index:
                continue
            edges.add((min(left_index, right_index), max(left_index, right_index), relation_name))

    mapping = {str(card_id): int(index) for card_id, index in dataset.card_id_to_index.items()}
    for detail in dataset.details:
        source_index = int(detail["card_index"])
        for reference in detail.get("text_references") or []:
            payload = reference.get("payload") or {}
            for target_card_id in payload.get("matching_target_card_ids") or []:
                target_index = mapping.get(str(target_card_id))
                if target_index is None:
                    raise ValueError(
                        f"detail {detail.get('global_detail_index')} references missing card "
                        f"{target_card_id!r}"
                    )
                if source_index != target_index:
                    edges.add(
                        (
                            min(source_index, target_index),
                            max(source_index, target_index),
                            f"text_reference:{reference.get('reference_type')}",
                        )
                    )
    return sorted(edges)


def split_connected_components(
    dataset: CardDataset,
    *,
    validation_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[list[int], list[int], list[int], dict[str, Any]]:
    """Create a deterministic 80/10/10 split without structural-edge leakage."""

    if validation_ratio <= 0 or test_ratio <= 0 or validation_ratio + test_ratio >= 1:
        raise ValueError("validation_ratio and test_ratio must be positive and sum to less than one")
    card_count = len(dataset.cards)
    if card_count < 3:
        raise ValueError("component split requires at least three cards")

    parent = list(range(card_count))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    edges = _structural_edges(dataset)
    for left, right, _relation in edges:
        if not 0 <= left < card_count or not 0 <= right < card_count:
            raise ValueError(f"structural edge contains an invalid card index: {(left, right)}")
        union(left, right)

    grouped: dict[int, list[int]] = {}
    for index in range(card_count):
        grouped.setdefault(find(index), []).append(index)
    components = [sorted(indices) for indices in grouped.values()]
    if len(components) < 3:
        raise ValueError("structural graph has fewer than three connected components")

    # Seed controls equal-size ordering; placing larger components first keeps
    # target deficits stable and avoids a late large component dominating a
    # validation/test partition.
    rng = random.Random(int(seed))
    rng.shuffle(components)
    components.sort(key=len, reverse=True)
    train_ratio = 1.0 - float(validation_ratio) - float(test_ratio)
    targets = {
        "train": float(card_count) * train_ratio,
        "validation": float(card_count) * float(validation_ratio),
        "test": float(card_count) * float(test_ratio),
    }
    assignments: dict[str, list[list[int]]] = {name: [] for name in targets}
    counts = {name: 0 for name in targets}
    seeded_split_order = list(targets)
    rng.shuffle(seeded_split_order)
    order_rank = {name: rank for rank, name in enumerate(seeded_split_order)}
    for component in components:
        destination = max(
            targets,
            key=lambda name: (targets[name] - counts[name], -order_rank[name]),
        )
        assignments[destination].append(component)
        counts[destination] += len(component)
    if any(not groups for groups in assignments.values()):
        raise ValueError(f"component split produced an empty partition: {counts}")

    split_indices = {
        name: sorted(index for component in groups for index in component)
        for name, groups in assignments.items()
    }
    all_indices = set().union(*(set(values) for values in split_indices.values()))
    if all_indices != set(range(card_count)):
        raise AssertionError("component split is not a complete card partition")
    if any(
        set(split_indices[left]) & set(split_indices[right])
        for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"))
    ):
        raise AssertionError("component split partitions overlap")
    owner = {
        index: split_name
        for split_name, indices in split_indices.items()
        for index in indices
    }
    leaking_edges = [edge for edge in edges if owner[edge[0]] != owner[edge[1]]]
    if leaking_edges:
        raise AssertionError(f"structural edges cross component splits: {leaking_edges[:10]}")

    component_rows: list[dict[str, Any]] = []
    for split_name, groups in assignments.items():
        for indices in groups:
            card_ids = [str(dataset.cards[index]["card_id"]) for index in indices]
            component_rows.append(
                {
                    "component_id": sha256_json(card_ids),
                    "split": split_name,
                    "card_indices": indices,
                    "card_ids": card_ids,
                }
            )
    component_rows.sort(key=lambda row: str(row["component_id"]))
    audit = {
        "component_count": len(component_rows),
        "components": component_rows,
        "structural_edge_count": len(edges),
        "structural_edges_sha256": sha256_json(edges),
        "component_membership_sha256": sha256_json(component_rows),
        "target_ratios": {
            "train": train_ratio,
            "validation": float(validation_ratio),
            "test": float(test_ratio),
        },
        "actual_counts": counts,
        "cross_split_structural_edge_count": 0,
    }
    return (
        split_indices["train"],
        split_indices["validation"],
        split_indices["test"],
        audit,
    )


def split_train_validation_test(
    records: list[dict[str, Any]],
    *,
    validation_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[list[int], list[int], list[int]]:
    """Legacy Card-ID-only splitter retained for callers outside formal v3."""

    if validation_ratio <= 0 or test_ratio <= 0 or validation_ratio + test_ratio >= 1:
        raise ValueError("validation_ratio and test_ratio must be positive and sum to less than one")
    groups: dict[str, list[int]] = {}
    for index, record in enumerate(records):
        groups.setdefault(str(record["card_id"]), []).append(index)
    items = list(groups.values())
    random.Random(seed).shuffle(items)
    validation_target = max(1, round(len(records) * validation_ratio))
    test_target = max(1, round(len(records) * test_ratio))
    test, validation, train = [], [], []
    for indices in items:
        if len(test) < test_target:
            test.extend(indices)
        elif len(validation) < validation_target:
            validation.extend(indices)
        else:
            train.extend(indices)
    return sorted(train), sorted(validation), sorted(test)


def tiny_coverage_contract(dataset: CardDataset, indices: list[int]) -> dict[str, Any]:
    selected = set(indices)
    selected_details = [
        detail for detail in dataset.details if int(detail["card_index"]) in selected
    ]
    relation_counts: dict[str, int] = {}
    for name, pairs in dataset.relation_samples().items():
        count = sum(int(left) in selected and int(right) in selected for left, right in pairs)
        if count:
            relation_counts[str(name)] = int(count)
    reference_type_counts: dict[str, int] = {}
    text_token_count = 0
    for detail in selected_details:
        text_token_count += len(detail.get("model_text_tokens") or [])
        for reference in detail.get("text_references") or []:
            name = str(reference.get("reference_type"))
            reference_type_counts[name] = reference_type_counts.get(name, 0) + 1
    detail_type_counts: dict[str, int] = {}
    for detail in selected_details:
        name = str(detail.get("detail_type"))
        detail_type_counts[name] = detail_type_counts.get(name, 0) + 1
    card_category_counts: dict[str, int] = {}
    for index in indices:
        name = str(dataset.cards[index].get("card_category"))
        card_category_counts[name] = card_category_counts.get(name, 0) + 1
    task_coverage = {
        "field_recovery": len(indices),
        "detail_attributes": len(selected_details),
        "text_mlm": text_token_count,
        "structure_reference": sum(relation_counts.values()) + sum(reference_type_counts.values()),
        "card_detail_matching": len(selected_details),
    }
    return {
        "card_ids": [str(dataset.cards[index]["card_id"]) for index in indices],
        "card_category_counts": dict(sorted(card_category_counts.items())),
        "detail_type_counts": dict(sorted(detail_type_counts.items())),
        "relation_counts": dict(sorted(relation_counts.items())),
        "reference_type_counts": dict(sorted(reference_type_counts.items())),
        "task_valid_sample_lower_bounds": task_coverage,
    }


def resolve_tiny_indices(
    dataset: CardDataset,
    tiny_config: dict[str, Any],
) -> tuple[list[int], dict[str, Any], str]:
    """Resolve only an explicitly frozen 16-card contract; never auto-select."""

    state = str(tiny_config.get("selection_state", ""))
    card_ids = [str(value) for value in tiny_config.get("card_ids") or []]
    expected_count = int(tiny_config.get("card_count", 16))
    expected_hash = str(tiny_config.get("coverage_sha256", ""))
    if state != "frozen_v3_coverage_scan":
        raise ValueError(
            "tiny_overfit.selection_state must be frozen_v3_coverage_scan; "
            "run the v3 full-catalog coverage scan and commit its 16 IDs first"
        )
    if expected_count != 16 or len(card_ids) != 16 or len(set(card_ids)) != 16:
        raise ValueError("tiny_overfit.card_ids must contain exactly 16 unique Card IDs")
    if not expected_hash or expected_hash == TINY_PENDING_COVERAGE:
        raise ValueError("tiny_overfit.coverage_sha256 is still pending")
    missing = sorted(set(card_ids) - set(dataset.card_id_to_index))
    if missing:
        raise ValueError(f"tiny-overfit Card IDs are absent from the v3 cache: {missing}")
    indices = [int(dataset.card_id_to_index[card_id]) for card_id in card_ids]
    coverage = tiny_coverage_contract(dataset, indices)
    actual_hash = sha256_json(coverage)
    if actual_hash != expected_hash:
        raise ValueError(
            "tiny-overfit coverage changed since the frozen scan: "
            f"config={expected_hash} actual={actual_hash}"
        )
    missing_tasks = [
        name
        for name in TASK_NAMES
        if int(coverage["task_valid_sample_lower_bounds"].get(name, 0)) <= 0
    ]
    if missing_tasks:
        raise ValueError(f"tiny-overfit coverage has no valid samples for tasks: {missing_tasks}")
    return indices, coverage, actual_hash


def run_tiny_overfit(
    base_model: StaticPretrainingModel,
    dataset: CardDataset,
    config: dict[str, Any],
    device: torch.device,
    *,
    selected_indices: list[int],
    coverage: dict[str, Any],
    coverage_sha256: str,
) -> tuple[dict[str, Any], StaticPretrainingModel]:
    tiny_config = config.get("tiny_overfit", {})
    steps = int(tiny_config.get("max_steps", 500))
    selected = list(selected_indices)
    if len(selected) != int(tiny_config.get("batch_size_cards", 16)):
        raise ValueError("tiny-overfit frozen IDs must form one 16-card batch")
    tiny_dataset = dataset.subset(selected)
    batch = collate_cards([_item_for_index(tiny_dataset, index) for index in selected], tiny_dataset.schema)
    masked = mask_training_inputs(
        batch,
        tiny_dataset.schema,
        config["training"].get("masking", {}),
        seed=int(config["seed"]) + 991,
    )
    relation = build_relation_inputs(
        tiny_dataset,
        batch_size=int(config["training"].get("relation_batch_size", 64)),
        seed=int(config["seed"]) + 992,
    )
    batch = move_batch(batch, device)
    masked = masked.to(device)
    model = copy.deepcopy(base_model).to(device)
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(tiny_config.get("learning_rate", 1e-3)),
        weight_decay=float(tiny_config.get("weight_decay", 0.0)),
    )

    with torch.no_grad():
        initial_total, initial_tasks, initial_metrics = compute_losses(
            model, batch, masked, relation, config, device
        )
    history = [float(initial_total.detach().cpu())]
    print(
        json.dumps(
            {
                "tiny_step": 0,
                "tiny_steps": steps,
                "total_loss": history[0],
                "device": str(device),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    first_gradient_norms: dict[str, float] = {}
    final_metrics = initial_metrics
    final_tasks = initial_tasks
    for step in range(steps):
        total, task_losses, metrics = compute_losses(model, batch, masked, relation, config, device)
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        if step == 0:
            first_gradient_norms = {
                "card_fields": gradient_norm(model.card_heads.parameters()),
                "detail_fields_and_mlm": gradient_norm(model.detail_heads.parameters()),
                "ownership": gradient_norm(model.ownership_head.parameters()),
                "relation": gradient_norm(model.relation_head.parameters()),
                "encoder": gradient_norm(model.encoder.parameters()),
            }
        _assert_finite_tensor(total, "tiny_overfit.total_loss")
        _assert_finite_gradients(model)
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            float(tiny_config.get("gradient_clip_norm", 1.0)),
        )
        optimizer.step()
        final_metrics = metrics
        final_tasks = task_losses
        if step == steps - 1 or (step + 1) % max(1, steps // 10) == 0:
            current_loss = float(total.detach().cpu())
            history.append(current_loss)
            print(
                json.dumps(
                    {
                        "tiny_step": step + 1,
                        "tiny_steps": steps,
                        "total_loss": current_loss,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    model.eval()
    with torch.no_grad():
        final_total, final_tasks, final_metrics = compute_losses(
            model, batch, masked, relation, config, device
        )
    initial_value = float(initial_total.detach().cpu())
    final_value = float(final_total.detach().cpu())
    ratio = final_value / max(initial_value, 1e-12)
    thresholds = tiny_config.get("thresholds") or {}
    required_ratio = float(thresholds.get("total_loss_ratio", 0.30))
    required_task_ratio = float(thresholds.get("per_active_task_loss_ratio", 0.50))
    gradient_ok = all(value > 0 and math.isfinite(value) for value in first_gradient_norms.values())
    task_finite = all(math.isfinite(float(value.detach().cpu())) for value in final_tasks.values())
    initial_task_values = {
        name: float(value.detach().cpu()) for name, value in initial_tasks.items()
    }
    final_task_values = {
        name: float(value.detach().cpu()) for name, value in final_tasks.items()
    }
    task_ratios = {
        name: final_task_values[name] / max(initial_task_values[name], 1.0e-12)
        for name in TASK_NAMES
    }
    task_sample_counts = {
        name: float(initial_metrics.get(f"{name}_valid_samples", 0.0))
        for name in TASK_NAMES
    }
    valid_samples = all(value > 0 for value in task_sample_counts.values())
    tasks_improved = all(task_ratios[name] <= required_task_ratio for name in TASK_NAMES)
    finite = (
        math.isfinite(initial_value)
        and math.isfinite(final_value)
        and task_finite
        and all(math.isfinite(value) for value in initial_task_values.values())
        and all(math.isfinite(value) for value in task_ratios.values())
    )
    success = ratio <= required_ratio and tasks_improved and gradient_ok and valid_samples and finite
    result = {
        "schema_version": "static_card_tiny_overfit_v3",
        "card_count": len(selected),
        "max_steps": steps,
        "selected_card_ids": [tiny_dataset.cards[index]["card_id"] for index in selected],
        "coverage": coverage,
        "coverage_sha256": coverage_sha256,
        "initial_total_loss": initial_value,
        "final_total_loss": final_value,
        "loss_ratio": ratio,
        "thresholds": {
            "total_loss_ratio": required_ratio,
            "per_active_task_loss_ratio": required_task_ratio,
        },
        "first_gradient_norms": first_gradient_norms,
        "initial_task_losses": initial_task_values,
        "final_task_losses": final_task_values,
        "task_loss_ratios": task_ratios,
        "task_valid_samples": task_sample_counts,
        "finite": finite,
        "all_heads_have_valid_samples": valid_samples,
        "final_metrics": final_metrics,
        "history": history,
        "success": success,
    }
    if not success:
        raise RuntimeError(f"static v3 tiny-overfit gate failed: {json.dumps(result, ensure_ascii=False)}")
    return result, model


def run_epoch(
    model: StaticPretrainingModel,
    dataset: CardDataset,
    loader: DataLoader,
    config: dict[str, Any],
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    *,
    epoch: int,
    seed_offset: int = 0,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    rows: list[dict[str, float]] = []
    accumulation_steps = int(config["training"].get("gradient_accumulation_steps", 1))
    if training and accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be positive")
    if not training and scheduler is not None:
        raise ValueError("evaluation cannot update a scheduler")
    if training:
        optimizer.zero_grad(set_to_none=True)
    optimizer_step_count = 0
    micro_batch_count = len(loader)
    for batch_index, raw_batch in enumerate(loader):
        # Validation/test must compare the same masked targets and relation
        # samples at every checkpoint. Only training varies masks by epoch.
        seed_epoch = epoch if training else 0
        mask_seed = int(config["seed"]) + int(seed_offset) + seed_epoch * 100_003 + batch_index
        masked = mask_training_inputs(
            raw_batch,
            dataset.schema,
            config["training"].get("masking", {}),
            seed=mask_seed,
        )
        relation = build_relation_inputs(
            dataset,
            batch_size=int(config["training"].get("relation_batch_size", 64)),
            seed=mask_seed + 50_000,
        )
        batch = move_batch(raw_batch, device)
        masked = masked.to(device)
        context = torch.enable_grad() if training else torch.no_grad()
        with context:
            total, _tasks, metrics = compute_losses(model, batch, masked, relation, config, device)
            if training:
                _assert_finite_tensor(total, "training.total_loss")
                group_start = (batch_index // accumulation_steps) * accumulation_steps
                group_size = min(accumulation_steps, micro_batch_count - group_start)
                (total / float(group_size)).backward()
                is_step = ((batch_index + 1) % accumulation_steps == 0) or (
                    batch_index + 1 == micro_batch_count
                )
                if is_step:
                    raw_gradient_norm = _assert_finite_gradients(model)
                    clipped_gradient_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        float(config["training"].get("gradient_clip_norm", 1.0)),
                    )
                    if not math.isfinite(float(clipped_gradient_norm)):
                        raise FloatingPointError("gradient clipping returned a non-finite norm")
                    optimizer.step()
                    if scheduler is not None:
                        scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    optimizer_step_count += 1
                    metrics["raw_gradient_norm"] = raw_gradient_norm
                    metrics["learning_rate"] = float(optimizer.param_groups[0]["lr"])
        rows.append(metrics)
    averaged = average_metrics(rows)
    averaged["micro_batch_count"] = float(micro_batch_count)
    averaged["optimizer_step_count"] = float(optimizer_step_count)
    if training:
        expected_steps = optimizer_steps_per_epoch(micro_batch_count, accumulation_steps)
        if optimizer_step_count != expected_steps:
            raise AssertionError(
                f"optimizer-step count mismatch: {optimizer_step_count} != {expected_steps}"
            )
    return averaged


def run_fixed_evaluation(
    model: StaticPretrainingModel,
    dataset: CardDataset,
    loader: DataLoader,
    config: dict[str, Any],
    device: torch.device,
    *,
    seed_offsets: Iterable[int],
) -> dict[str, float]:
    """Average a fixed set of masks/relations for comparable checkpoints."""

    rows = [
        run_epoch(
            model,
            dataset,
            loader,
            config,
            device,
            None,
            None,
            epoch=0,
            seed_offset=int(seed_offset),
        )
        for seed_offset in seed_offsets
    ]
    return average_metrics(rows)


def save_checkpoint(
    path: Path,
    model: StaticPretrainingModel,
    optimizer: torch.optim.Optimizer | None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    *,
    epoch: int,
    config: dict[str, Any],
    schema: dict[str, Any],
    metrics: dict[str, Any],
    stage: str = "evaluation_training",
    lineage: dict[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "training_schema_version": TRAINING_SCHEMA_VERSION,
            "data_schema_version": schema.get("schema_version"),
            "model_state": model.state_dict(),
            "encoder": model.encoder.state_dict(),
            "card_field_heads": model.card_heads.state_dict(),
            "detail_heads": model.detail_heads.state_dict(),
            "ownership_head": model.ownership_head.state_dict(),
            "relation_head": model.relation_head.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "epoch": int(epoch),
            "config": config,
            "schema": schema,
            "metrics": metrics,
            "stage": stage,
            "lineage": lineage or {},
            "rng_state": {
                "python": random.getstate(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
        },
        temporary_path,
    )
    temporary_path.replace(path)


def load_checkpoint(
    path: Path,
    model: StaticPretrainingModel,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
) -> int:
    checkpoint = torch.load(path, map_location=device)
    if checkpoint.get("training_schema_version") != TRAINING_SCHEMA_VERSION:
        raise ValueError("legacy static checkpoint cannot be resumed by static v3 training")
    if checkpoint.get("data_schema_version") != DATA_SCHEMA_VERSION:
        raise ValueError("checkpoint data schema does not match static_card_v3")
    model.load_state_dict(checkpoint["model_state"])
    if optimizer is not None and checkpoint.get("optimizer") is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and checkpoint.get("scheduler") is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
    return int(checkpoint.get("epoch", -1)) + 1


def verify_checkpoint_reload(
    path: Path,
    source_model: StaticPretrainingModel,
    schema: dict[str, Any],
    model_config: dict[str, Any],
) -> dict[str, Any]:
    reloaded = StaticPretrainingModel(schema, model_config).cpu()
    load_checkpoint(path, reloaded, None, torch.device("cpu"))
    source_state = source_model.state_dict()
    reloaded_state = reloaded.state_dict()
    if source_state.keys() != reloaded_state.keys():
        raise RuntimeError("checkpoint save/reload changed model-state keys")
    mismatched = [
        name
        for name in source_state
        if not torch.equal(source_state[name].detach().cpu(), reloaded_state[name].detach().cpu())
    ]
    if mismatched:
        raise RuntimeError(f"checkpoint save/reload changed tensors: {mismatched[:10]}")
    return {
        "passed": True,
        "path": str(path),
        "sha256": sha256_file(path),
        "tensor_count": len(source_state),
    }


def _device_from_config(config: dict[str, Any]) -> torch.device:
    requested = str(config["training"].get("device", "auto"))
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def main() -> None:
    parser = argparse.ArgumentParser(description="Train static CardEncoder v2")
    parser.add_argument("--config", type=Path, default=Path("configs/card_pretrain.yaml"))
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--tiny-overfit-only", action="store_true")
    parser.add_argument("--skip-tiny-overfit", action="store_true")
    parser.add_argument("--tiny-steps", type=int)
    parser.add_argument("--tiny-card-count", type=int)
    args = parser.parse_args()

    config = load_config(args.config)
    if config.get("schema_version") != TRAINING_SCHEMA_VERSION:
        raise ValueError(f"config must declare schema_version: {TRAINING_SCHEMA_VERSION}")
    set_seed(int(config["seed"]))
    cache_dir = Path(config["data"].get("cache_dir", DEFAULT_CACHE_DIR))
    dataset = CardDataset.from_cache(cache_dir, rebuild=args.rebuild_cache)
    if dataset.schema.get("schema_version") != DATA_SCHEMA_VERSION:
        raise ValueError("static v2 dataset is required")

    train_indices, validation_indices, test_indices = split_train_validation_test(
        dataset.records,
        validation_ratio=float(config["data"].get("validation_ratio", 0.15)),
        test_ratio=float(config["data"].get("test_ratio", 0.15)),
        seed=int(config["seed"]),
    )
    train_dataset = dataset.subset(train_indices)
    validation_dataset = dataset.subset(validation_indices)
    test_dataset = dataset.subset(test_indices)
    print(
        json.dumps(
            {
                "schema_version": DATA_SCHEMA_VERSION,
                "device": str(_device_from_config(config)),
                "cards": len(dataset.records),
                "details": len(dataset.details),
                "split": {
                    "train": len(train_indices),
                    "validation": len(validation_indices),
                    "test": len(test_indices),
                },
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    output_root = Path(config["training"]["checkpoint_dir"]).parent
    output_root.mkdir(parents=True, exist_ok=True)
    preprocess_manifest_path = cache_dir / "preprocess_manifest.json"
    if not preprocess_manifest_path.is_file():
        raise FileNotFoundError("static v2 cache is missing preprocess_manifest.json")
    split_manifest = {
        "schema_version": "static_card_split_v2",
        "seed": int(config["seed"]),
        "mode": "card_id",
        "transductive_catalog_schema": True,
        "transductive_note": (
            "vocabularies and numeric normalization are fixed from the complete known card catalog; "
            "optimization examples and relation pairs remain split-local"
        ),
        "train_indices": train_indices,
        "validation_indices": validation_indices,
        "test_indices": test_indices,
        "train_card_ids": [str(dataset.cards[index]["card_id"]) for index in train_indices],
        "validation_card_ids": [str(dataset.cards[index]["card_id"]) for index in validation_indices],
        "test_card_ids": [str(dataset.cards[index]["card_id"]) for index in test_indices],
    }
    split_manifest_path = output_root / "split_manifest.json"
    split_manifest_path.write_text(
        json.dumps(split_manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    lineage = {
        "config_path": str(args.config),
        "config_sha256": sha256_file(args.config),
        "preprocess_manifest_path": str(preprocess_manifest_path),
        "preprocess_manifest_sha256": sha256_file(preprocess_manifest_path),
        "split_manifest_path": str(split_manifest_path),
        "split_manifest_sha256": sha256_file(split_manifest_path),
    }

    device = _device_from_config(config)
    model = StaticPretrainingModel(dataset.schema, config["model"]).to(device)
    tiny_result: dict[str, Any] | None = None

    if not args.skip_tiny_overfit and bool(config.get("tiny_overfit", {}).get("enabled", True)):
        tiny_result, tiny_model = run_tiny_overfit(
            model,
            train_dataset,
            config,
            device,
            steps_override=args.tiny_steps,
            card_count_override=args.tiny_card_count,
        )
        tiny_dir = output_root / "tiny_overfit"
        tiny_dir.mkdir(parents=True, exist_ok=True)
        (tiny_dir / "metrics.json").write_text(
            json.dumps(tiny_result, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        save_checkpoint(
            tiny_dir / "tiny_overfit.pt",
            tiny_model,
            None,
            epoch=-1,
            config=config,
            schema=dataset.schema,
            metrics=tiny_result,
            stage="tiny_overfit",
            lineage=lineage,
        )
        print(json.dumps({"tiny_overfit": tiny_result}, ensure_ascii=False, indent=2), flush=True)
    if args.tiny_overfit_only:
        return

    training_config = config["training"]
    batch_size = int(training_config.get("batch_size", 32))
    collator = lambda items: collate_cards(items, dataset.schema)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collator)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, collate_fn=collator)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collator)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_config.get("learning_rate", 3e-4)),
        weight_decay=float(training_config.get("weight_decay", 1e-4)),
    )
    start_epoch = 0
    if args.resume is not None and bool(config.get("production_refit", {}).get("enabled", False)):
        raise ValueError("formal production-refit runs must start from a fresh output directory")
    if args.resume is not None:
        start_epoch = load_checkpoint(args.resume, model, optimizer, device)

    checkpoint_dir = Path(training_config["checkpoint_dir"])
    log_dir = Path(training_config["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    metrics_log_path = log_dir / "card_pretrain_metrics.jsonl"
    if start_epoch == 0:
        metrics_log_path.write_text("", encoding="utf-8")
    max_epochs = int(training_config.get("max_epochs", 400))
    if max_epochs <= 0:
        raise ValueError("formal training requires max_epochs > 0")
    eval_every = max(1, int(training_config.get("eval_every_epochs", 5)))
    early = training_config.get("early_stopping", {}) or {}
    patience_epochs = int(early.get("patience_epochs", early.get("patience", 0)))
    patience = (
        max(1, math.ceil(patience_epochs / eval_every))
        if bool(early.get("enabled", False)) and patience_epochs > 0
        else 0
    )
    restore_best = bool(early.get("restore_best_checkpoint", True))
    if not restore_best:
        raise ValueError("static v2 formal evaluation requires restore_best_checkpoint: true")
    min_epochs = int(training_config.get("min_epochs", 0))
    validation_seed_offsets = [
        int(value)
        for value in training_config.get(
            "evaluation_seed_offsets",
            [1_000_000, 2_000_000, 3_000_000],
        )
    ]
    if not validation_seed_offsets:
        raise ValueError("at least one fixed validation seed offset is required")
    test_seed_offsets = [value + 10_000_000 for value in validation_seed_offsets]
    best_validation = float("inf")
    best_epoch: int | None = None
    stale = 0
    last_epoch = start_epoch - 1
    stopped_early = False
    for epoch in range(start_epoch, max_epochs):
        train_metrics = run_epoch(model, train_dataset, train_loader, config, device, optimizer, epoch=epoch)
        last_epoch = epoch
        row: dict[str, Any] = {"epoch": epoch, "train": train_metrics}
        if (epoch + 1) % eval_every == 0 or epoch + 1 == max_epochs:
            validation_metrics = run_fixed_evaluation(
                model,
                validation_dataset,
                validation_loader,
                config,
                device,
                seed_offsets=validation_seed_offsets,
            )
            row["validation"] = validation_metrics
            current = float(validation_metrics["total_loss"])
            if current < best_validation:
                best_validation = current
                best_epoch = epoch
                stale = 0
                save_checkpoint(
                    checkpoint_dir / "card_encoder_best.pt",
                    model,
                    None,
                    epoch=epoch,
                    config=config,
                    schema=dataset.schema,
                    metrics=validation_metrics,
                    stage="split_selection_best",
                    lineage={**lineage, "best_epoch": epoch},
                )
            else:
                stale += 1
            save_checkpoint(
                checkpoint_dir / "card_encoder_last.pt",
                model,
                optimizer,
                epoch=epoch,
                config=config,
                schema=dataset.schema,
                metrics=validation_metrics,
                stage="split_selection_last",
                lineage={
                    **lineage,
                    "best_epoch": best_epoch,
                    "best_validation": best_validation,
                    "stale_evaluations": stale,
                },
            )
        print(json.dumps(row, ensure_ascii=False), flush=True)
        with metrics_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        if patience and epoch + 1 >= min_epochs and stale >= patience:
            stopped_early = True
            break

    best_path = checkpoint_dir / "card_encoder_best.pt"
    if not best_path.exists() or best_epoch is None:
        raise RuntimeError("training ended without a best checkpoint")
    load_checkpoint(best_path, model, None, device)
    test_metrics = run_fixed_evaluation(
        model,
        test_dataset,
        test_loader,
        config,
        device,
        seed_offsets=test_seed_offsets,
    )
    (log_dir / "test_metrics.json").write_text(
        json.dumps(test_metrics, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"test": test_metrics}, ensure_ascii=False, indent=2), flush=True)

    production_refit_summary: dict[str, Any] | None = None
    refit_config = config.get("production_refit", {}) or {}
    if bool(refit_config.get("enabled", False)):
        if refit_config.get("epoch_source") != "best_validation_epoch":
            raise ValueError("production_refit.epoch_source must be best_validation_epoch")
        refit_epochs = best_epoch + 1
        if refit_epochs <= 0:
            raise RuntimeError("best validation epoch produced an invalid refit length")
        model.to("cpu")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        set_seed(int(config["seed"]))
        production_model = StaticPretrainingModel(dataset.schema, config["model"]).to(device)
        production_optimizer = torch.optim.AdamW(
            production_model.parameters(),
            lr=float(training_config.get("learning_rate", 3e-4)),
            weight_decay=float(training_config.get("weight_decay", 1e-4)),
        )
        production_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collator,
        )
        production_log_path = Path(
            refit_config.get(
                "log_path",
                log_dir / "production_refit_metrics.jsonl",
            )
        )
        production_log_path.parent.mkdir(parents=True, exist_ok=True)
        production_log_path.write_text("", encoding="utf-8")
        production_metrics: dict[str, float] = {}
        for refit_epoch in range(refit_epochs):
            production_metrics = run_epoch(
                production_model,
                dataset,
                production_loader,
                config,
                device,
                production_optimizer,
                epoch=refit_epoch,
                seed_offset=30_000_000,
            )
            refit_row = {"epoch": refit_epoch, "train": production_metrics}
            print(json.dumps({"production_refit": refit_row}, ensure_ascii=False), flush=True)
            with production_log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(refit_row, ensure_ascii=False) + "\n")
        production_checkpoint = Path(refit_config["checkpoint_path"])
        save_checkpoint(
            production_checkpoint,
            production_model,
            None,
            epoch=refit_epochs - 1,
            config=config,
            schema=dataset.schema,
            metrics=production_metrics,
            stage="full_catalog_production_refit",
            lineage={
                **lineage,
                "selected_best_epoch": best_epoch,
                "refit_epochs": refit_epochs,
                "refit_card_count": len(dataset),
            },
        )
        production_refit_summary = {
            "enabled": True,
            "card_count": len(dataset),
            "epochs": refit_epochs,
            "checkpoint": str(production_checkpoint),
            "checkpoint_sha256": sha256_file(production_checkpoint),
            "log": str(production_log_path),
            "final_train_metrics": production_metrics,
        }

    formal_summary = {
        "schema_version": "static_card_formal_training_v2",
        "success": True,
        "run_mode": config.get("run_mode", "static_v2_formal"),
        "selection_training": {
            "completed_epochs": last_epoch + 1,
            "best_epoch": best_epoch,
            "best_validation_loss": best_validation,
            "stopped_early": stopped_early,
            "patience_epochs": patience_epochs,
            "patience_evaluations": patience,
            "validation_seed_offsets": validation_seed_offsets,
            "best_checkpoint": str(best_path),
            "best_checkpoint_sha256": sha256_file(best_path),
            "last_checkpoint": str(checkpoint_dir / "card_encoder_last.pt"),
        },
        "test_seed_offsets": test_seed_offsets,
        "test_metrics": test_metrics,
        "production_refit": production_refit_summary,
        "tiny_overfit": tiny_result,
        "lineage": lineage,
    }
    (output_root / "formal_training_summary.json").write_text(
        json.dumps(formal_summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"formal_training": formal_summary}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
