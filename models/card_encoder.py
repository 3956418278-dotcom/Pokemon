from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn


STATIC_V3_SCHEMA = "static_card_v3"


def _vocab_size(schema: dict[str, Any], name: str) -> int:
    vocab = schema.get("vocab", {}).get(name)
    if not isinstance(vocab, dict) or not vocab:
        raise ValueError(f"schema is missing the {name!r} vocabulary")
    return len(vocab)


def _token_id(vocab: dict[str, int], label: str) -> int:
    if label not in vocab:
        raise ValueError(f"required token {label!r} is absent from the frozen vocabulary")
    return int(vocab[label])


def _required_tensor(batch: dict[str, Any], name: str) -> torch.Tensor:
    value = batch.get(name)
    if not isinstance(value, torch.Tensor):
        raise KeyError(f"CardEncoder batch is missing tensor {name!r}")
    return value


@dataclass
class CardEncoderOutput:
    card_summary: torch.Tensor
    base_card_token: torch.Tensor
    independent_detail_tokens: torch.Tensor
    contextualized_detail_tokens: torch.Tensor
    detail_mask: torch.Tensor
    detail_type_ids: torch.Tensor
    text_token_states: torch.Tensor
    card_field_tokens: torch.Tensor

    @property
    def detail_tokens(self) -> torch.Tensor:
        """The permanent detail export is the parent-independent token."""

        return self.independent_detail_tokens

    @property
    def pre_fusion_detail_tokens(self) -> torch.Tensor:
        return self.independent_detail_tokens

    @property
    def contextual_detail_states(self) -> torch.Tensor:
        return self.contextualized_detail_tokens


class SharedCardValueEmbeddings(nn.Module):
    """One embedding table per semantic field group.

    Card name, evolves-from and individual evolves-to targets share the same
    name table.  No species or detail-name table exists here.
    """

    def __init__(self, schema: dict[str, Any], embedding_dim: int) -> None:
        super().__init__()
        groups = schema.get("value_vocabs")
        if not isinstance(groups, dict):
            raise ValueError("static-v3 schema is missing value_vocabs")
        self.field_to_group = dict(schema["field_to_value_group"])
        self.tables = nn.ModuleDict(
            {
                group: nn.Embedding(len(vocab), embedding_dim)
                for group, vocab in groups.items()
            }
        )

    def embed_field(self, field: str, ids: torch.Tensor) -> torch.Tensor:
        return self.tables[self.field_to_group[field]](ids.long())

    def embed_card_names(self, ids: torch.Tensor) -> torch.Tensor:
        return self.tables["card_name"](ids.long())


class EnergyPrintedProfileEncoder(nn.Module):
    """Encode 13 printed energy symbol counts with symbol-specific tables."""

    def __init__(self, schema: dict[str, Any], embedding_dim: int) -> None:
        super().__init__()
        symbols = tuple(schema["energy_symbols"])
        if len(symbols) != 13:
            raise ValueError("printed energy profile must contain exactly 13 symbols")
        self.symbols = symbols
        count_vocab_size = int(schema["profile_energy_count_vocab_size"])
        self.count_embeddings = nn.ModuleList(
            [nn.Embedding(count_vocab_size, embedding_dim) for _ in symbols]
        )
        self.output_norm = nn.LayerNorm(embedding_dim)

    def forward(self, count_ids: torch.Tensor) -> torch.Tensor:
        if count_ids.dim() != 2 or count_ids.shape[1] != len(self.count_embeddings):
            raise ValueError("provided_energy_count_ids must have shape [B, 13]")
        values = [table(count_ids[:, index].long()) for index, table in enumerate(self.count_embeddings)]
        return self.output_norm(torch.stack(values, dim=1).sum(dim=1))


class CardFieldEncoder(nn.Module):
    def __init__(
        self,
        schema: dict[str, Any],
        values: SharedCardValueEmbeddings,
        embedding_dim: int,
    ) -> None:
        super().__init__()
        self.fields = tuple(schema["card_field_slots"])
        if len(self.fields) != 15:
            raise ValueError("static-v3 requires exactly 15 card field slots")
        self.values = values
        self.energy_profile = EnergyPrintedProfileEncoder(schema, embedding_dim)
        self.slot_embedding = nn.Embedding(len(self.fields), embedding_dim)
        self.output_norm = nn.LayerNorm(embedding_dim)
        self.energy_index = self.fields.index("energy_printed_type")
        self.evolves_to_index = self.fields.index("evolves_to")
        self.numeric_fields = ("hp", "retreat")
        self.numeric_indices = {field: self.fields.index(field) for field in self.numeric_fields}
        self.numeric_projection = nn.ModuleDict(
            {
                field: nn.Sequential(
                    nn.Linear(1, embedding_dim),
                    nn.GELU(),
                    nn.LayerNorm(embedding_dim),
                )
                for field in self.numeric_fields
            }
        )
        self.numeric_mask_tokens = nn.ParameterDict(
            {field: nn.Parameter(torch.zeros(embedding_dim)) for field in self.numeric_fields}
        )
        self.numeric_mask_ids = {
            field: _token_id(
                schema["value_vocabs"][schema["field_to_value_group"][field]],
                "<MASK>",
            )
            for field in self.numeric_fields
        }

    def forward(
        self,
        field_value_ids: torch.Tensor,
        evolves_to_name_ids: torch.Tensor,
        evolves_to_name_mask: torch.Tensor,
        provided_energy_count_ids: torch.Tensor,
        card_numeric_values: torch.Tensor,
    ) -> torch.Tensor:
        if field_value_ids.dim() != 2 or field_value_ids.shape[1] != len(self.fields):
            raise ValueError("card_field_value_ids must have shape [B, 15]")
        tokens = torch.stack(
            [self.values.embed_field(field, field_value_ids[:, index]) for index, field in enumerate(self.fields)],
            dim=1,
        )
        tokens[:, self.energy_index] = self.energy_profile(provided_energy_count_ids)

        if card_numeric_values.shape != (field_value_ids.shape[0], len(self.numeric_fields)):
            raise ValueError("card_numeric_values must have shape [B, 2] for HP and Retreat")
        for numeric_position, field in enumerate(self.numeric_fields):
            field_index = self.numeric_indices[field]
            projected = self.numeric_projection[field](card_numeric_values[:, numeric_position : numeric_position + 1].float())
            is_masked = field_value_ids[:, field_index].long() == self.numeric_mask_ids[field]
            mask_token = self.numeric_mask_tokens[field].reshape(1, -1).expand_as(projected)
            tokens[:, field_index] = torch.where(is_masked.unsqueeze(-1), mask_token, projected)

        if evolves_to_name_ids.dim() != 2 or evolves_to_name_mask.shape != evolves_to_name_ids.shape:
            raise ValueError("evolves_to_name_ids/mask must align as [B, E]")
        target_states = self.values.embed_card_names(evolves_to_name_ids)
        target_mask = evolves_to_name_mask.bool()
        pooled_targets = (
            (target_states * target_mask.unsqueeze(-1).to(target_states.dtype)).sum(dim=1)
            / target_mask.sum(dim=1, keepdim=True).clamp_min(1).to(target_states.dtype)
        )
        has_target = target_mask.any(dim=1)
        tokens[:, self.evolves_to_index] = torch.where(
            has_target.unsqueeze(-1), pooled_targets, tokens[:, self.evolves_to_index]
        )

        positions = torch.arange(len(self.fields), device=field_value_ids.device)
        return self.output_norm(tokens + self.slot_embedding(positions).unsqueeze(0))


class CardCategoryBranches(nn.Module):
    """Route applicable card fields through Pokémon, Trainer or Energy branches."""

    def __init__(self, schema: dict[str, Any], embedding_dim: int) -> None:
        super().__init__()
        fields = tuple(schema["card_field_slots"])
        self.card_kind_index = fields.index("card_kind")
        kind_vocab = schema["value_vocabs"][schema["field_to_value_group"]["card_kind"]]
        self.pokemon_id = _token_id(kind_vocab, "POKEMON")
        self.trainer_id = _token_id(kind_vocab, "TRAINER")
        self.energy_id = _token_id(kind_vocab, "ENERGY")

        def branch() -> nn.Module:
            return nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim),
                nn.GELU(),
                nn.LayerNorm(embedding_dim),
            )

        self.pokemon_branch = branch()
        self.trainer_branch = branch()
        self.energy_branch = branch()

    def forward(
        self,
        field_tokens: torch.Tensor,
        card_kind_route_ids: torch.Tensor,
        applicability_mask: torch.Tensor,
    ) -> torch.Tensor:
        if applicability_mask.shape != field_tokens.shape[:2]:
            raise ValueError("card field tokens and applicability mask must align")
        if card_kind_route_ids.shape != (field_tokens.shape[0],):
            raise ValueError("card_kind_route_ids must have shape [B]")
        valid = applicability_mask.bool()
        if not bool(valid.any(dim=1).all().item()):
            raise ValueError("every card must expose at least one applicable field")
        pooled = (
            (field_tokens * valid.unsqueeze(-1).to(field_tokens.dtype)).sum(dim=1)
            / valid.sum(dim=1, keepdim=True).to(field_tokens.dtype)
        )
        kinds = card_kind_route_ids.long()
        known = (kinds == self.pokemon_id) | (kinds == self.trainer_id) | (kinds == self.energy_id)
        if not bool(known.all().item()):
            raise ValueError("card_kind must be POKEMON, TRAINER or ENERGY")
        result = torch.where((kinds == self.pokemon_id).unsqueeze(-1), self.pokemon_branch(pooled), self.trainer_branch(pooled))
        return torch.where((kinds == self.energy_id).unsqueeze(-1), self.energy_branch(pooled), result)


class RuleTextEncoder(nn.Module):
    """Encode ordinary text and typed structure references in one sequence."""

    def __init__(
        self,
        schema: dict[str, Any],
        embedding_dim: int,
        num_heads: int,
        ffn_dim: int,
        layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.max_length = int(schema["max_text_subword_tokens"])
        self.kind_embedding = nn.Embedding(len(schema["text_token_kind_vocab"]), embedding_dim, padding_idx=0)
        self.text_embedding = nn.Embedding(_vocab_size(schema, "text"), embedding_dim, padding_idx=int(schema["text_pad_id"]))
        self.reference_type_embedding = nn.Embedding(
            len(schema["reference_type_vocab"]), embedding_dim, padding_idx=0
        )
        self.reference_fields = tuple(schema["reference_fields"])
        self.reference_field_embedding = nn.Embedding(len(self.reference_fields) + 1, embedding_dim, padding_idx=0)
        self.reference_value_embeddings = nn.ModuleDict(
            {
                field: nn.Embedding(len(schema["reference_value_vocabs"][field]), embedding_dim, padding_idx=0)
                for field in self.reference_fields
            }
        )
        self.position_embedding = nn.Embedding(self.max_length, embedding_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=layers)
        self.output_norm = nn.LayerNorm(embedding_dim)

    def forward(self, batch: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        kind_ids = _required_tensor(batch, "detail_text_token_kind_ids").long()
        token_mask = _required_tensor(batch, "detail_text_token_mask").bool()
        if kind_ids.dim() != 3 or token_mask.shape != kind_ids.shape:
            raise ValueError("detail text tensors must have shape [B, D, T]")
        batch_size, detail_count, token_count = kind_ids.shape
        if token_count > self.max_length:
            raise ValueError(f"detail text length {token_count} exceeds {self.max_length}")

        states = self.kind_embedding(kind_ids)
        plain_mask = _required_tensor(batch, "detail_plain_text_mask").bool()
        text_ids = _required_tensor(batch, "detail_text_ids").long()
        states = states + plain_mask.unsqueeze(-1) * self.text_embedding(text_ids)

        reference_mask = _required_tensor(batch, "detail_structure_reference_mask").bool()
        reference_types = _required_tensor(batch, "detail_structure_reference_type_ids").long()
        reference_fields = _required_tensor(batch, "detail_structure_reference_field_ids").long()
        reference_values = _required_tensor(batch, "detail_structure_reference_value_ids").long()
        states = states + reference_mask.unsqueeze(-1) * (
            self.reference_type_embedding(reference_types) + self.reference_field_embedding(reference_fields)
        )
        for field_index, field in enumerate(self.reference_fields, start=1):
            field_mask = reference_mask & (reference_fields == field_index)
            if bool(field_mask.any().item()):
                table = self.reference_value_embeddings[field]
                safe_values = reference_values.clamp_min(0).clamp_max(table.num_embeddings - 1)
                states = states + field_mask.unsqueeze(-1) * table(safe_values)

        positions = torch.arange(token_count, device=kind_ids.device).reshape(1, 1, token_count)
        states = states + self.position_embedding(positions)
        flat_states = states.reshape(batch_size * detail_count, token_count, self.embedding_dim)
        original_mask = token_mask.reshape(batch_size * detail_count, token_count)
        safe_mask = original_mask.clone()
        empty = ~safe_mask.any(dim=1)
        if bool(empty.any().item()):
            safe_mask[empty, 0] = True
        encoded = self.transformer(flat_states, src_key_padding_mask=~safe_mask)
        encoded = self.output_norm(encoded)
        encoded = encoded * original_mask.unsqueeze(-1).to(encoded.dtype)
        pooled = encoded.sum(dim=1) / original_mask.sum(dim=1, keepdim=True).clamp_min(1).to(encoded.dtype)
        return (
            pooled.reshape(batch_size, detail_count, self.embedding_dim),
            encoded.reshape(batch_size, detail_count, token_count, self.embedding_dim),
        )


class AttackCostEncoder(nn.Module):
    def __init__(self, schema: dict[str, Any], embedding_dim: int) -> None:
        super().__init__()
        symbols = tuple(schema["attack_energy_symbols"])
        if len(symbols) != 12:
            raise ValueError("Attack cost must contain exactly 12 energy symbols")
        size = int(schema["attack_energy_count_vocab_size"])
        self.tables = nn.ModuleList([nn.Embedding(size, embedding_dim) for _ in symbols])
        self.output_norm = nn.LayerNorm(embedding_dim)

    def forward(self, count_ids: torch.Tensor) -> torch.Tensor:
        if count_ids.dim() != 3 or count_ids.shape[-1] != len(self.tables):
            raise ValueError("attack_energy_count_ids must have shape [B, D, 12]")
        values = [table(count_ids[..., index].long()) for index, table in enumerate(self.tables)]
        return self.output_norm(torch.stack(values, dim=-2).sum(dim=-2))


class AttackEncoder(nn.Module):
    def __init__(self, schema: dict[str, Any], embedding_dim: int) -> None:
        super().__init__()
        self.cost_encoder = AttackCostEncoder(schema, embedding_dim)
        self.damage_value_embedding = nn.Embedding(_vocab_size(schema, "damage_value"), embedding_dim, padding_idx=0)
        self.damage_mode_embedding = nn.Embedding(_vocab_size(schema, "damage_mode"), embedding_dim, padding_idx=0)
        self.projection = nn.Sequential(
            nn.Linear(embedding_dim * 5, embedding_dim),
            nn.GELU(),
            nn.LayerNorm(embedding_dim),
        )

    def forward(
        self,
        text: torch.Tensor,
        detail_type: torch.Tensor,
        cost_count_ids: torch.Tensor,
        damage_value_ids: torch.Tensor,
        damage_mode_ids: torch.Tensor,
    ) -> torch.Tensor:
        return self.projection(
            torch.cat(
                [
                    text,
                    detail_type,
                    self.cost_encoder(cost_count_ids),
                    self.damage_value_embedding(damage_value_ids.long()),
                    self.damage_mode_embedding(damage_mode_ids.long()),
                ],
                dim=-1,
            )
        )


class AbilityEncoder(nn.Module):
    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.GELU(),
            nn.LayerNorm(embedding_dim),
        )

    def forward(self, text: torch.Tensor, detail_type: torch.Tensor) -> torch.Tensor:
        return self.projection(torch.cat([text, detail_type], dim=-1))


class CardEffectEncoder(AbilityEncoder):
    pass


class IndependentDetailEncoder(nn.Module):
    def __init__(
        self,
        schema: dict[str, Any],
        embedding_dim: int,
        num_heads: int,
        ffn_dim: int,
        layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.text_encoder = RuleTextEncoder(
            schema, embedding_dim, num_heads, ffn_dim, layers, dropout
        )
        detail_vocab = schema["vocab"]["detail_type"]
        self.attack_id = _token_id(detail_vocab, "ATTACK")
        self.ability_id = _token_id(detail_vocab, "ABILITY")
        self.card_effect_id = _token_id(detail_vocab, "CARD_EFFECT")
        self.detail_type_embedding = nn.Embedding(len(detail_vocab), embedding_dim, padding_idx=0)
        self.attack_encoder = AttackEncoder(schema, embedding_dim)
        self.ability_encoder = AbilityEncoder(embedding_dim)
        self.card_effect_encoder = CardEffectEncoder(embedding_dim)

    def forward(self, batch: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        text, token_states = self.text_encoder(batch)
        detail_type_ids = _required_tensor(batch, "detail_type_ids").long()
        detail_type = self.detail_type_embedding(detail_type_ids)
        attack = self.attack_encoder(
            text,
            detail_type,
            _required_tensor(batch, "attack_energy_count_ids"),
            _required_tensor(batch, "attack_damage_value_ids"),
            _required_tensor(batch, "attack_damage_mode_ids"),
        )
        ability = self.ability_encoder(text, detail_type)
        card_effect = self.card_effect_encoder(text, detail_type)
        is_attack = detail_type_ids == self.attack_id
        is_ability = detail_type_ids == self.ability_id
        is_effect = detail_type_ids == self.card_effect_id
        valid_type = is_attack | is_ability | is_effect | (detail_type_ids == 0)
        if not bool(valid_type.all().item()):
            raise ValueError("detail_type_ids contains an unknown non-padding type")
        result = torch.where(is_attack.unsqueeze(-1), attack, card_effect)
        result = torch.where(is_ability.unsqueeze(-1), ability, result)
        detail_mask = _required_tensor(batch, "detail_mask").bool()
        return result * detail_mask.unsqueeze(-1).to(result.dtype), token_states


class RelationAwareCardLayer(nn.Module):
    def __init__(self, embedding_dim: int, num_heads: int, ffn_dim: int, dropout: float) -> None:
        super().__init__()
        self.num_heads = int(num_heads)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.attention = nn.MultiheadAttention(embedding_dim, num_heads, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embedding_dim),
        )
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, states: torch.Tensor, valid: torch.Tensor, attention_bias: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, _ = states.shape
        if attention_bias.shape != (batch_size, self.num_heads, sequence_length, sequence_length):
            raise ValueError("same-card attention bias has the wrong shape")
        normalized = self.norm1(states)
        attended, _ = self.attention(
            normalized,
            normalized,
            normalized,
            key_padding_mask=~valid.bool(),
            attn_mask=attention_bias.reshape(batch_size * self.num_heads, sequence_length, sequence_length),
            need_weights=False,
        )
        states = states + self.dropout1(attended)
        states = states + self.dropout2(self.ffn(self.norm2(states)))
        return states * valid.unsqueeze(-1).to(states.dtype)


class CardDetailTransformer(nn.Module):
    def __init__(
        self,
        schema: dict[str, Any],
        embedding_dim: int,
        num_heads: int,
        layers: int,
        ffn_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.num_heads = int(num_heads)
        self.position_embedding = nn.Embedding(max(2, int(schema["max_details"]) + 1), embedding_dim)
        self.same_card_reference_bias = nn.Parameter(torch.zeros(num_heads))
        self.layers = nn.ModuleList(
            [RelationAwareCardLayer(embedding_dim, num_heads, ffn_dim, dropout) for _ in range(layers)]
        )
        self.output_norm = nn.LayerNorm(embedding_dim)

    def forward(
        self,
        card_token: torch.Tensor,
        detail_tokens: torch.Tensor,
        detail_mask: torch.Tensor,
        same_card_reference_matrix: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, detail_count, hidden = detail_tokens.shape
        sequence_length = detail_count + 1
        if sequence_length > self.position_embedding.num_embeddings:
            raise ValueError("card has more detail positions than the frozen schema")
        states = torch.cat([card_token.unsqueeze(1), detail_tokens], dim=1)
        positions = torch.arange(sequence_length, device=states.device)
        states = states + self.position_embedding(positions).unsqueeze(0)
        valid = torch.cat(
            [torch.ones((batch_size, 1), dtype=torch.bool, device=states.device), detail_mask.bool()], dim=1
        )
        if same_card_reference_matrix.shape != (batch_size, detail_count, detail_count):
            raise ValueError("same_card_detail_reference_matrix must have shape [B, D, D]")
        attention_bias = states.new_zeros((batch_size, self.num_heads, sequence_length, sequence_length))
        relation = same_card_reference_matrix.bool().unsqueeze(1).to(states.dtype)
        attention_bias[:, :, 1:, 1:] = relation * self.same_card_reference_bias.reshape(1, -1, 1, 1)
        for layer in self.layers:
            states = layer(states, valid, attention_bias)
        states = self.output_norm(states)
        return states[:, 0], states[:, 1:] * detail_mask.unsqueeze(-1).to(states.dtype)


class CardEncoder(nn.Module):
    """The single static-v3 CardEncoder implementation.

    It produces one 128-dimensional base/card summary per card and one
    128-dimensional parent-independent token per detail.  Numeric reference IDs,
    detail names, species and derived subtypes are not accepted as inputs.
    """

    def __init__(
        self,
        schema: dict[str, Any],
        embedding_dim: int = 128,
        detail_token_dim: int = 128,
        num_heads: int = 4,
        transformer_layers: int = 2,
        ffn_dim: int = 256,
        dropout: float = 0.1,
        freeze_text_encoder: bool = False,
        **_unused: Any,
    ) -> None:
        super().__init__()
        if schema.get("schema_version") != STATIC_V3_SCHEMA:
            raise ValueError(f"CardEncoder requires schema {STATIC_V3_SCHEMA}")
        if embedding_dim != 128 or detail_token_dim != 128:
            raise ValueError("static-v3 card and detail dimensions are fixed at 128")
        if num_heads != 4 or transformer_layers != 2 or ffn_dim != 256:
            raise ValueError("static-v3 Card Transformer is frozen at heads=4, layers=2, ffn_dim=256")
        if freeze_text_encoder:
            raise ValueError("static-v3 freezes tokenizer bytes, not neural text embeddings")
        self.schema = schema
        self.embedding_dim = int(embedding_dim)
        self.detail_token_dim = int(detail_token_dim)
        self.value_embeddings = SharedCardValueEmbeddings(schema, embedding_dim)
        self.card_field_encoder = CardFieldEncoder(schema, self.value_embeddings, embedding_dim)
        self.card_branches = CardCategoryBranches(schema, embedding_dim)
        self.detail_encoder = IndependentDetailEncoder(
            schema,
            embedding_dim,
            num_heads,
            ffn_dim,
            transformer_layers,
            dropout,
        )
        self.card_transformer = CardDetailTransformer(
            schema,
            embedding_dim,
            num_heads,
            transformer_layers,
            ffn_dim,
            dropout,
        )

    def forward(self, batch: dict[str, Any], return_details: bool = False) -> torch.Tensor | CardEncoderOutput:
        field_value_ids = _required_tensor(batch, "card_field_value_ids")
        field_tokens = self.card_field_encoder(
            field_value_ids,
            _required_tensor(batch, "evolves_to_name_ids"),
            _required_tensor(batch, "evolves_to_name_mask"),
            _required_tensor(batch, "provided_energy_count_ids"),
            _required_tensor(batch, "card_numeric_values"),
        )
        base_card_token = self.card_branches(
            field_tokens,
            _required_tensor(batch, "card_kind_route_ids"),
            _required_tensor(batch, "card_field_applicability_mask"),
        )
        independent_details, text_states = self.detail_encoder(batch)
        detail_mask = _required_tensor(batch, "detail_mask").bool()
        card_summary, contextualized_details = self.card_transformer(
            base_card_token,
            independent_details,
            detail_mask,
            _required_tensor(batch, "same_card_detail_reference_matrix"),
        )
        if not return_details:
            return card_summary
        return CardEncoderOutput(
            card_summary=card_summary,
            base_card_token=base_card_token,
            independent_detail_tokens=independent_details,
            contextualized_detail_tokens=contextualized_details,
            detail_mask=detail_mask,
            detail_type_ids=_required_tensor(batch, "detail_type_ids").long(),
            text_token_states=text_states,
            card_field_tokens=field_tokens,
        )


def infer_head_sizes(schema: dict[str, Any]) -> dict[str, Any]:
    """Return only the five-task output sizes defined by static-v3."""

    return {
        "card_fields": {
            field: len(schema["value_vocabs"][schema["field_to_value_group"][field]])
            for field in schema["card_field_slots"]
            if field not in {"card_name", "energy_printed_type"}
        },
        "profile_energy_count": int(schema["profile_energy_count_vocab_size"]) - 1,
        "attack_energy_count": int(schema["attack_energy_count_vocab_size"]),
        "damage_value": _vocab_size(schema, "damage_value"),
        "damage_mode": _vocab_size(schema, "damage_mode"),
        "text": _vocab_size(schema, "text"),
        "reference_type": len(schema["reference_type_vocab"]),
        "reference_values": {
            field: len(schema["reference_value_vocabs"][field]) for field in schema["reference_fields"]
        },
    }
