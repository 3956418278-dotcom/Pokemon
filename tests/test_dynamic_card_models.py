from __future__ import annotations

from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F

from data.state_schema import AREA_IDS, CardInstanceState, collate_card_dynamic
from models.card_instance_fusion import CardInstanceFusion, CardInstanceFusionOutput
from models.dynamic_card_auxiliary import DynamicCardAuxiliaryHeads
from models.dynamic_instance_encoder import DynamicInstanceEncoder
from models.static_card_adapter import StaticCardEmbeddingAdapter


def _instance() -> CardInstanceState:
    energy = [0] * 12
    energy[5] = 1
    return CardInstanceState(
        card_id=21,
        serial=1,
        player_index=0,
        relative_player=0,
        area=AREA_IDS["ACTIVE"],
        zone="active",
        slot=0,
        is_pokemon=True,
        hp=100,
        max_hp=120,
        appear_this_turn=False,
        appear_this_turn_valid=True,
        energy_counts=energy,
        energy_counts_valid=True,
        energy_card_count=1,
        energy_cards_valid=True,
        energy_card_ids=[5],
        tool_count=0,
        tools_valid=True,
        pre_evolution_count=0,
        pre_evolution_valid=True,
        special_conditions_valid=True,
        copy_count=1,
        energy_payment_resolved=True,
        detail_exists=True,
        static_artifact_known=True,
    )


def _assert_effective_finite_grad(module) -> None:
    gradients = [parameter.grad for parameter in module.parameters() if parameter.grad is not None]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert sum(float(gradient.abs().sum().item()) for gradient in gradients) > 0.0


def test_dynamic_encoder_handles_normal_and_empty_batches() -> None:
    encoder = DynamicInstanceEncoder(dropout=0.0)
    batch = collate_card_dynamic([_instance()])
    output = encoder(batch)
    assert output.shape == (1, 64)
    assert torch.isfinite(output).all()

    empty_batch = collate_card_dynamic([])
    empty_dynamic = encoder(empty_batch)
    fusion = CardInstanceFusion(dropout=0.0)
    empty_summary = torch.empty(0, 128, requires_grad=True)
    empty_details = torch.empty(0, 2, 128)
    empty_mask = torch.empty(0, 2)
    empty_result = fusion(
        empty_summary,
        empty_dynamic,
        empty_details,
        empty_mask,
        torch.empty(0, 2, dtype=torch.long),
        return_attention=True,
    )
    assert isinstance(empty_result, CardInstanceFusionOutput)
    assert empty_dynamic.shape == (0, 64)
    assert empty_result.card_instance_token.shape == (0, 128)
    assert empty_result.attention_weights.shape == (0, 4, 2)
    empty_loss = empty_result.card_instance_token.sum()
    assert torch.isfinite(empty_loss)
    empty_loss.backward()
    assert empty_summary.grad is not None and torch.isfinite(empty_summary.grad).all()
    assert all(parameter.grad is not None for parameter in encoder.parameters())
    assert all(
        torch.isfinite(parameter.grad).all()
        for parameter in encoder.parameters()
        if parameter.grad is not None
    )


def test_dynamic_encoder_sanitizes_non_finite_validity_masks() -> None:
    batch = collate_card_dynamic([_instance()])
    batch.static_known_mask.fill_(float("nan"))
    batch.detail_exists_mask.fill_(float("inf"))
    batch.energy_resolved_mask.fill_(float("-inf"))
    batch.visibility_mask.fill_(float("nan"))
    output = DynamicInstanceEncoder(dropout=0.0)(batch)
    assert output.shape == (1, 64)
    assert torch.isfinite(output).all()


def test_cross_attention_all_mask_is_finite_and_backward_safe() -> None:
    fusion = CardInstanceFusion(dropout=0.0)
    summary = torch.randn(3, 128, requires_grad=True)
    dynamic = torch.randn(3, 64, requires_grad=True)
    details = torch.randn(3, 4, 128, requires_grad=True)
    mask = torch.zeros(3, 4)
    result = fusion(summary, dynamic, details, mask, torch.zeros(3, 4, dtype=torch.long), return_attention=True)
    assert isinstance(result, CardInstanceFusionOutput)
    assert result.card_instance_token.shape == (3, 128)
    assert result.attention_weights.shape == (3, 4, 4)
    assert result.attention_weights.abs().sum().item() == 0.0
    assert torch.isfinite(result.card_instance_token).all()
    result.card_instance_token.square().mean().backward()
    assert summary.grad is not None and torch.isfinite(summary.grad).all()
    assert dynamic.grad is not None and torch.isfinite(dynamic.grad).all()


def test_padding_does_not_change_valid_detail_output() -> None:
    torch.manual_seed(7)
    fusion = CardInstanceFusion(dropout=0.0).eval()
    summary = torch.randn(2, 128)
    dynamic = torch.randn(2, 64)
    valid_details = torch.randn(2, 2, 128)
    base = fusion(
        summary,
        dynamic,
        valid_details,
        torch.ones(2, 2),
        torch.tensor([[1, 2], [1, 3]]),
    )
    padded_details = torch.cat([valid_details, torch.randn(2, 3, 128) * 100.0], dim=1)
    padded = fusion(
        summary,
        dynamic,
        padded_details,
        torch.tensor([[1, 1, 0, 0, 0], [1, 1, 0, 0, 0]], dtype=torch.float32),
        torch.tensor([[1, 2, 3, 3, 3], [1, 3, 2, 2, 2]]),
    )
    assert torch.allclose(base, padded, atol=1e-6, rtol=1e-5)


def test_masked_non_finite_padding_is_isolated() -> None:
    fusion = CardInstanceFusion(dropout=0.0).eval()
    heads = DynamicCardAuxiliaryHeads().eval()
    summary = torch.randn(1, 128)
    dynamic = torch.randn(1, 64)
    details = torch.randn(1, 3, 128)
    details[:, 1] = float("nan")
    details[:, 2] = float("inf")
    mask = torch.tensor([[1.0, 0.0, 0.0]])
    token = fusion(summary, dynamic, details, mask, torch.tensor([[1, 2, 3]]))
    output = heads(token, details, mask)
    assert torch.isfinite(token).all()
    assert torch.isfinite(output.payable_logits).all()
    assert torch.isfinite(output.energy_remaining).all()


def test_same_card_changes_token_with_hp_energy_and_zone() -> None:
    torch.manual_seed(11)
    base = _instance()
    no_energy = replace(base, serial=2, energy_counts=[0] * 12, energy_card_count=0, energy_card_ids=[])
    damaged = replace(base, serial=3, hp=20)
    bench = replace(base, serial=4, area=AREA_IDS["BENCH"], zone="bench")
    batch = collate_card_dynamic([base, no_energy, damaged, bench])
    encoder = DynamicInstanceEncoder(dropout=0.0).eval()
    fusion = CardInstanceFusion(dropout=0.0).eval()
    dynamic = encoder(batch)
    summary = torch.randn(1, 128).expand(4, -1).clone()
    details = torch.randn(1, 3, 128).expand(4, -1, -1).clone()
    tokens = fusion(summary, dynamic, details, torch.ones(4, 3), torch.ones(4, 3, dtype=torch.long))
    assert not torch.allclose(tokens[0], tokens[1])
    assert not torch.allclose(tokens[0], tokens[2])
    assert not torch.allclose(tokens[0], tokens[3])


def test_unknown_and_anonymous_hidden_instances_are_finite_and_backward_safe() -> None:
    unknown = replace(
        _instance(),
        card_id=9999,
        serial=2,
        static_artifact_known=False,
        detail_exists=False,
    )
    hidden = CardInstanceState(
        card_id=None,
        serial=None,
        player_index=1,
        relative_player=1,
        area=AREA_IDS["HAND"],
        zone="hand",
        slot=0,
        is_visible=False,
        is_face_down=True,
        attachment_kind=None,
        static_artifact_known=False,
        detail_exists=False,
    )
    batch = collate_card_dynamic([unknown, hidden])
    encoder = DynamicInstanceEncoder(dropout=0.0)
    fusion = CardInstanceFusion(dropout=0.0)
    summary = torch.zeros(2, 128, requires_grad=True)
    details = torch.zeros(2, 3, 128)
    detail_mask = torch.zeros(2, 3)
    tokens = fusion(
        summary,
        encoder(batch),
        details,
        detail_mask,
        torch.zeros(2, 3, dtype=torch.long),
    )
    assert batch.card_ids.tolist() == [9999, 0]
    assert batch.visibility_mask.tolist() == [1.0, 0.0]
    assert tokens.shape == (2, 128)
    assert torch.isfinite(tokens).all()
    tokens[:, 0].sum().backward()
    assert summary.grad is not None and torch.isfinite(summary.grad).all()
    _assert_effective_finite_grad(encoder)
    _assert_effective_finite_grad(fusion)


@pytest.mark.parametrize("task", ["payable", "energy", "hp", "zone_role"])
def test_each_auxiliary_task_backpropagates_to_instance_token(task: str) -> None:
    torch.manual_seed(17)
    token = torch.randn(3, 128, requires_grad=True)
    details = torch.randn(3, 2, 128)
    mask = torch.ones(3, 2)
    heads = DynamicCardAuxiliaryHeads()
    output = heads(token, details, mask, torch.ones(3, 2, dtype=torch.long))
    if task == "payable":
        loss = F.binary_cross_entropy_with_logits(output.payable_logits, torch.ones_like(output.payable_logits))
    elif task == "energy":
        loss = F.smooth_l1_loss(output.energy_remaining, torch.zeros_like(output.energy_remaining))
    elif task == "hp":
        loss = F.mse_loss(output.hp_state, torch.zeros_like(output.hp_state))
    else:
        loss = F.cross_entropy(output.zone_logits, torch.tensor([4, 5, 2]))
        loss = loss + F.cross_entropy(output.role_logits, torch.tensor([2, 3, 1]))
    loss.backward()
    assert token.grad is not None
    assert token.grad.abs().sum().item() > 0


def test_combined_auxiliary_loss_reaches_encoder_attention_and_all_heads() -> None:
    torch.manual_seed(29)
    first = _instance()
    second = replace(
        first,
        serial=2,
        area=AREA_IDS["BENCH"],
        zone="bench",
        hp=60,
        energy_counts=[0] * 12,
        energy_card_count=0,
        energy_card_ids=[],
    )
    batch = collate_card_dynamic([first, second])
    encoder = DynamicInstanceEncoder(dropout=0.0)
    fusion = CardInstanceFusion(dropout=0.0)
    heads = DynamicCardAuxiliaryHeads()
    summary = torch.randn(2, 128)
    details = torch.randn(2, 3, 128)
    detail_mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
    detail_type_ids = torch.tensor([[1, 1, 0], [1, 1, 0]])
    tokens = fusion(summary, encoder(batch), details, detail_mask, detail_type_ids)
    output = heads(tokens, details, detail_mask, detail_type_ids)
    loss = F.binary_cross_entropy_with_logits(
        output.payable_logits[:, :2],
        torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
    )
    loss = loss + F.smooth_l1_loss(
        output.energy_remaining[:, :2],
        torch.ones(2, 2, 12),
    )
    loss = loss + F.mse_loss(
        output.hp_state,
        torch.tensor([[0.75, 0.25], [0.5, 0.5]]),
    )
    loss = loss + F.cross_entropy(output.zone_logits, batch.zone_ids)
    loss = loss + F.cross_entropy(output.role_logits, batch.field_role_ids)
    assert torch.isfinite(loss)
    loss.backward()
    _assert_effective_finite_grad(encoder)
    _assert_effective_finite_grad(fusion.cross_attention)
    _assert_effective_finite_grad(heads.payable_head)
    _assert_effective_finite_grad(heads.energy_remaining_head)
    _assert_effective_finite_grad(heads.hp_head)
    _assert_effective_finite_grad(heads.zone_head)
    _assert_effective_finite_grad(heads.role_head)


def test_auxiliary_heads_support_distinct_token_and_detail_dimensions() -> None:
    heads = DynamicCardAuxiliaryHeads(token_dim=64, detail_dim=32)
    tokens = torch.randn(2, 64, requires_grad=True)
    no_details = heads(tokens, None)
    assert no_details.payable_logits.shape == (2, 0)
    assert no_details.energy_remaining.shape == (2, 0, 12)
    assert no_details.hp_state.shape == (2, 2)

    details = torch.randn(2, 3, 32)
    output = heads(tokens, details, torch.ones(2, 3))
    assert output.payable_logits.shape == (2, 3)
    assert output.energy_remaining.shape == (2, 3, 12)
    output.payable_logits.sum().backward()
    assert tokens.grad is not None and torch.isfinite(tokens.grad).all()
    with pytest.raises(ValueError, match="detail token width"):
        heads(tokens.detach(), torch.randn(2, 3, 31))


def test_static_artifacts_remain_frozen_during_dynamic_backward() -> None:
    adapter = StaticCardEmbeddingAdapter(embedding_dim=128, max_details=2, detail_dim=128)
    batch = collate_card_dynamic([_instance()])
    static = adapter.forward_features(batch.card_ids)
    dynamic_encoder = DynamicInstanceEncoder(dropout=0.0)
    fusion = CardInstanceFusion(dropout=0.0)
    token = fusion(
        static.summary,
        dynamic_encoder(batch),
        static.detail_tokens,
        static.detail_mask,
        static.detail_type_ids,
    )
    token.square().mean().backward()
    assert not adapter.dummy_param.requires_grad
    assert adapter.dummy_param.grad is None
    assert any(parameter.grad is not None for parameter in dynamic_encoder.parameters())
    assert any(parameter.grad is not None for parameter in fusion.parameters())
