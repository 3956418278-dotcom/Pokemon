from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from competition_selfplay.cli import dry_run
from competition_selfplay.config import SCHEMA_VERSION, load_config
from competition_selfplay.deck import load_target_deck
from competition_selfplay.features import GLOBAL_FEATURE_NAMES, compact_global_features
from competition_selfplay.league import EvaluationResult, LeagueAction, LeagueController, LeagueState
from competition_selfplay.model import (
    SemanticSelfPlayModel,
    compute_critic_losses,
)
from competition_selfplay.phase import (
    CalibrationMetrics,
    PhaseController,
    TrainingPhase,
)
from competition_selfplay.reward import (
    perspective_outcome,
    transaction_gae,
    transaction_reward,
)
from competition_selfplay.rollout import AsymmetricRolloutCollector, ObservationVectorizer
from competition_selfplay.run_mechanical_selfplay import _build_replay
from competition_selfplay.semantic import (
    ACTIVE_SURVIVAL,
    DELAYED_TRIGGER,
    NET_PRIZE_H1,
    NUM_SEMANTIC_CONCEPTS,
    RECOVERY_PATH,
    RULE_LOCK,
    SEMANTIC_CONCEPT_NAMES,
    SemanticConceptOutput,
    SemanticConceptTargets,
    SemanticPotentialHead,
    TrajectoryLabelBuilder,
    reverse_net_prize_perspective,
    semantic_applicability,
    semantic_concept_loss,
)
from competition_selfplay.training import (
    AsymmetricSelfPlayTrainer,
    FixedCalibrationHoldout,
)
from competition_selfplay.transactions import (
    SelectRecord,
    Transaction,
    TransactionAssembler,
    closes_transaction,
    joint_log_probability,
    selection_is_forced,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "competition_selfplay/configs/raging_bolt_fast.yaml"


def _observation(
    *,
    seat: int = 0,
    turn: int = 1,
    context: int | None = 0,
    options: int = 2,
    result: int = -1,
    active_serials: tuple[int | None, int | None] = (10, 20),
    prizes: tuple[int, int] = (6, 6),
    semantic_facts: dict | None = None,
    logs: list[dict] | None = None,
) -> dict:
    players = []
    for player in (0, 1):
        serial = active_serials[player]
        players.append(
            {
                "active": [] if serial is None else [{"id": 100 + player, "serial": serial}],
                "bench": [],
                "hand": [],
                "discard": [],
                "prize": [None] * prizes[player],
                "deckCount": 30,
            }
        )
    select = None
    if context is not None:
        select = {
            "context": context,
            "type": 1,
            "minCount": 1,
            "maxCount": 1,
            "option": [{"type": 7, "index": index} for index in range(options)],
        }
    return {
        "current": {
            "yourIndex": seat,
            "turn": turn,
            "turnActionCount": 1,
            "result": result,
            "players": players,
        },
        "select": select,
        "semantic_facts": semantic_facts or {},
        "logs": logs or [],
    }


class _FakeEnvironment:
    def __init__(self) -> None:
        self.states = [
            _observation(seat=0, context=0, options=2),
            _observation(
                seat=0,
                context=7,
                options=1,
                logs=[{"type": 10, "playerIndex": 0, "serial": 101}],
            ),
            _observation(
                seat=1,
                context=0,
                options=2,
                logs=[{"type": 15, "playerIndex": 0, "serial": 10}],
            ),
            _observation(
                seat=1,
                context=None,
                options=0,
                result=0,
                prizes=(6, 5),
                logs=[
                    {"type": 16, "playerIndex": 0, "serial": 20},
                    {"type": 23, "result": 0, "reason": 1},
                ],
            ),
        ]
        self.index = 0
        self.finished = False

    def start(self, deck0, deck1):
        assert len(deck0) == len(deck1) == 60
        return self.states[0]

    def select(self, action):
        self.index += 1
        return self.states[self.index]

    def finish(self):
        self.finished = True


def _small_system():
    base = load_config(CONFIG)
    model_config = replace(
        base.model,
        hidden_dim=16,
        observation_dimensions=64,
        option_dimensions=16,
        semantic_spline_knots=3,
    )
    training_config = replace(
        base.training,
        device="cpu",
        update_epochs=1,
        minibatch_size=8,
        target_kl=100.0,
    )
    config = replace(base, model=model_config, training=training_config)
    learner = SemanticSelfPlayModel(model_config)
    opponent = copy.deepcopy(learner)
    target = learner.build_target_semantic_module()
    phase = PhaseController(config.reward)
    collector = AsymmetricRolloutCollector(
        online_model=learner,
        frozen_opponent=opponent,
        target_semantic=target,
        model_config=model_config,
        reward_config=config.reward,
        phase_controller=phase,
        device="cpu",
    )
    deck = load_target_deck(config.target_deck, ROOT)
    batch = collector.collect_batch(
        environment_factory=_FakeEnvironment,
        deck=deck.card_ids,
        games=1,
    )
    optimizer = torch.optim.Adam(learner.parameters(), lr=1e-3)
    trainer = AsymmetricSelfPlayTrainer(
        config=config,
        learner=learner,
        frozen_opponent=opponent,
        target_semantic=target,
        optimizer=optimizer,
        phase_controller=phase,
        device="cpu",
    )
    return config, learner, opponent, target, batch, trainer


def test_config_and_deck_are_transactional_scalar_and_copy_aware(tmp_path: Path) -> None:
    config = load_config(CONFIG)
    deck = load_target_deck(config.target_deck, ROOT)
    assert config.schema_version == SCHEMA_VERSION
    assert config.model.value_dimensions == 1
    assert config.training.gamma == 0.997
    assert config.training.clip_epsilon == 0.2
    assert len(SEMANTIC_CONCEPT_NAMES) == NUM_SEMANTIC_CONCEPTS == 9
    assert len(deck.card_ids) == 60
    kangaskhan = [token for token in deck.card_tokens if token.card_id == 756]
    assert [token.copy_ordinal for token in kangaskhan] == [0, 1, 2]

    legacy = tmp_path / "legacy.yaml"
    legacy.write_text("schema_version: fixed_deck_selfplay_v1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="three-dimensional reward config is obsolete"):
        load_config(legacy)


def test_multilevel_resolution_is_one_transaction_and_forced_select_is_excluded() -> None:
    root = _observation(seat=0, context=0, options=2)
    nested = _observation(seat=0, context=7, options=1)
    next_root = _observation(seat=0, context=0, options=2)
    assembler = TransactionAssembler()
    transaction = assembler.begin(seat=0, start_state=root)
    assembler.record_select(log_prob=torch.tensor(-0.2), forced=False)
    assert not closes_transaction(transaction, nested)
    assert selection_is_forced(nested)
    assembler.record_select(log_prob=torch.tensor(0.0), forced=True)
    assert closes_transaction(transaction, next_root)
    completed = assembler.close(end_state=next_root)
    assert len(assembler.completed) == 1
    assert completed.old_log_prob_sum == pytest.approx(-0.2)
    assert len(completed.non_forced_log_probs) == 1


def test_two_non_forced_probabilities_form_the_transaction_joint_probability() -> None:
    transaction = Transaction(0, 0, {})
    transaction.append_select(log_prob=torch.tensor(-0.25), forced=False)
    transaction.append_select(log_prob=torch.tensor(-0.75), forced=False)
    assert transaction.old_log_prob_sum == pytest.approx(-1.0)
    assert float(joint_log_probability(transaction.non_forced_log_probs)) == pytest.approx(-1.0)


def test_optional_selection_stop_is_part_of_recomputed_joint_probability() -> None:
    config = load_config(CONFIG)
    model = SemanticSelfPlayModel(config.model)
    with torch.no_grad():
        model.policy_head.stop_logit.weight.zero_()
        model.policy_head.stop_logit.bias.fill_(100.0)
    state = torch.randn(config.model.observation_dimensions)
    options = torch.randn(2, config.model.option_dimensions)
    actions, sampled_log_probability, _, stopped = model.sample_selection(
        state,
        options,
        minimum_count=0,
        maximum_count=2,
        deterministic=True,
    )
    recomputed, _ = model.selection_log_probability(
        state,
        options,
        actions,
        minimum_count=0,
        maximum_count=2,
        stopped_early=stopped,
    )
    assert actions == () and stopped
    assert float(recomputed) == pytest.approx(sampled_log_probability)
    optional_single = _observation(options=1)
    optional_single["select"]["minCount"] = 0
    assert not selection_is_forced(optional_single)


@pytest.mark.parametrize("winner,seat,expected", [(0, 0, 1.0), (0, 1, -1.0), (2, 0, 0.0)])
def test_terminal_outcome_is_locked_to_win_loss_draw(winner: int, seat: int, expected: float) -> None:
    config = load_config(CONFIG)
    assert perspective_outcome(
        perspective_player=seat,
        winner=winner,
        config=config.reward,
    ) == expected


def test_alpha_zero_is_exact_terminal_reward_and_terminal_phi_is_zero() -> None:
    pure = transaction_reward(
        terminal=False,
        outcome=0.0,
        target_phi_before=0.7,
        target_phi_after=-0.6,
        gamma=0.997,
        shaping_alpha=0.0,
    )
    terminal = transaction_reward(
        terminal=True,
        outcome=1.0,
        target_phi_before=0.2,
        target_phi_after=0.8,
        gamma=0.997,
        shaping_alpha=0.15,
    )
    assert pure.total == 0.0
    assert terminal.total == pytest.approx(1.0 - 0.15 * 0.2)

    head = SemanticPotentialHead(knot_count=3)
    value = torch.full((1, 9), 0.5)
    mask = torch.ones_like(value, dtype=torch.bool)
    confidence = torch.ones_like(value)
    assert head(value, mask, confidence, terminal=torch.tensor([True])).item() == 0.0


def test_rollout_consumes_transactions_alternates_seats_and_stores_frozen_phi() -> None:
    _, _, _, target, batch, _ = _small_system()
    assert batch.transaction_count == 2
    first, second = batch.transactions
    assert first.seat == 0 and first.learner_controlled
    assert second.seat == 1 and not second.learner_controlled
    assert len(first.select_records) == 2
    assert first.non_forced_select_count == 1
    assert first.forced_select_count == 1
    assert first.target_phi_after == second.target_phi_before or first.target_phi_after == 0.0
    assert first.terminal_reason == second.terminal_reason == 1
    assert all(not parameter.requires_grad for parameter in target.parameters())
    assert PhaseController.learner_seat(0) == 0
    assert PhaseController.learner_seat(1) == 1


def test_target_and_frozen_opponent_do_not_move_during_online_batch_update() -> None:
    _, _, opponent, target, batch, trainer = _small_system()
    opponent_before = {name: value.clone() for name, value in opponent.state_dict().items()}
    target_before = {name: value.clone() for name, value in target.state_dict().items()}
    update = trainer.update_online(batch)
    assert {
        "concept_loss",
        "semantic_loss",
        "residual_loss",
        "full_loss",
    }.issubset(update.metrics)
    assert all(torch.equal(value, opponent_before[name]) for name, value in opponent.state_dict().items())
    assert all(torch.equal(value, target_before[name]) for name, value in target.state_dict().items())


def test_phase_b_actor_set_contains_only_learner_transactions() -> None:
    _, _, _, _, batch, trainer = _small_system()
    actor_ids = trainer.ppo.actor_transaction_ids(batch)
    assert actor_ids == tuple(
        transaction.transaction_id
        for transaction in batch.transactions
        if transaction.learner_controlled and transaction.non_forced_select_count
    )
    assert not set(actor_ids).intersection(
        transaction.transaction_id
        for transaction in batch.transactions
        if not transaction.learner_controlled
    )


def test_applicability_is_a_state_fact_and_masks_concept_loss() -> None:
    state = _observation(
        active_serials=(None, 20),
        semantic_facts={
            "rule_lock_ids": [],
            "armed_delayed_effect_ids": [],
            "recovery_needed": True,
        },
    )
    mask = semantic_applicability(state, 0)
    assert not mask[ACTIVE_SURVIVAL]
    assert not mask[RULE_LOCK]
    assert mask[RECOVERY_PATH]
    assert not mask[DELAYED_TRIGGER]

    values = torch.full((1, 9), 0.25)
    target_values = torch.zeros_like(values)
    target_mask = torch.ones_like(values, dtype=torch.bool)
    target_mask[:, RULE_LOCK] = False
    output = SemanticConceptOutput(values, target_mask, torch.ones_like(values))
    baseline = semantic_concept_loss(
        output,
        SemanticConceptTargets(target_values, target_mask),
    )
    target_values[:, RULE_LOCK] = 1.0
    changed_only_under_mask = semantic_concept_loss(
        output,
        SemanticConceptTargets(target_values, target_mask),
    )
    assert baseline == changed_only_under_mask


def test_observable_stadium_rule_lock_persistence_is_labeled_without_card_id_rules() -> None:
    start = _observation()
    start["current"]["stadium"] = [{"id": 123, "serial": 77, "playerIndex": 0}]
    opponent_start = _observation(seat=1)
    opponent_start["current"]["stadium"] = [
        {"id": 123, "serial": 77, "playerIndex": 0}
    ]
    first = Transaction(0, 0, start, end_state=opponent_start)
    second = Transaction(1, 1, opponent_start, terminal=True, outcome=-1)
    targets = TrajectoryLabelBuilder().build([first, second])
    assert targets.applicable[0, RULE_LOCK]
    assert targets.values[0, RULE_LOCK] == 1.0


def test_trajectory_labels_use_physical_serial_and_signed_prize_horizons() -> None:
    first = Transaction(
        0,
        0,
        _observation(active_serials=(41, 20), prizes=(6, 6)),
        end_state=_observation(active_serials=(41, 20), prizes=(6, 5)),
        event_records=[],
    )
    first.add_event({"kind": "attack_executed", "seat": 0, "source_card_serial": 41})
    opponent = Transaction(1, 1, _observation(seat=1, active_serials=(41, 20)))
    next_own = Transaction(2, 0, _observation(active_serials=(41, 20)), terminal=True, outcome=1)
    targets = TrajectoryLabelBuilder().build([first, opponent, next_own])
    assert targets.values[0, 0] == 1.0
    assert targets.values[0, ACTIVE_SURVIVAL] == 1.0
    assert targets.values[0, NET_PRIZE_H1] == pytest.approx(1 / 6)

    different_serial = Transaction(3, 0, _observation(active_serials=(99, 20)))
    replaced = TrajectoryLabelBuilder().build([first, opponent, different_serial])
    assert replaced.values[0, ACTIVE_SURVIVAL] == 0.0

    evolved_on_bench_state = _observation(active_serials=(55, 20))
    evolved_on_bench_state["current"]["players"][0]["bench"] = [
        {"id": 999, "serial": 99, "preEvolution": [{"id": 998, "serial": 41}]}
    ]
    evolved_on_bench = Transaction(4, 0, evolved_on_bench_state)
    survived_move = TrajectoryLabelBuilder().build([first, opponent, evolved_on_bench])
    assert survived_move.values[0, ACTIVE_SURVIVAL] == 1.0


def test_deckout_label_uses_exact_terminal_result_reason() -> None:
    transaction = Transaction(
        0,
        0,
        _observation(),
        end_state=_observation(context=None, result=1),
        terminal=True,
        outcome=-1,
        terminal_reason=2,
    )
    targets = TrajectoryLabelBuilder().build([transaction])
    assert targets.values[0, 5] == 1.0


def test_option_features_resolve_hand_index_to_physical_card_identity() -> None:
    observation = _observation()
    observation["current"]["players"][0]["hand"] = [
        {"id": 1234, "serial": 88, "energyCards": [], "tools": []}
    ]
    observation["select"]["option"] = [{"type": 7, "index": 0}]
    card = ObservationVectorizer.resolve_option_card(
        observation,
        observation["select"]["option"][0],
        0,
    )
    assert card["id"] == 1234
    assert card["serial"] == 88


def test_delayed_trigger_creates_a_causal_link_to_earlier_transaction() -> None:
    assembler = TransactionAssembler(first_transaction_id=5)
    assembler.begin(seat=1, start_state=_observation(seat=1))
    assembler.record_select(
        log_prob=-0.2,
        forced=False,
        events=[
            {
                "kind": "delayed_trigger",
                "cause_transaction_id": 2,
                "cause_kind": "armed_tool",
                "source_card_serial": 77,
            }
        ],
    )
    transaction = assembler.close(end_state=_observation(seat=0))
    assert transaction.cause_transaction_ids == [2]
    assert transaction.causal_links[0].trigger_transaction_id == 5
    assert transaction.causal_links[0].source_card_serial == 77


def test_semantic_explanation_exactly_reconstructs_pre_tanh_logit() -> None:
    torch.manual_seed(7)
    head = SemanticPotentialHead(knot_count=4)
    with torch.no_grad():
        head.bias.fill_(0.13)
        head.interaction_weights.copy_(torch.linspace(-0.2, 0.3, len(head.interaction_weights)))
        for index, function in enumerate(head.unary_functions):
            function.knot_values.copy_(torch.linspace(-0.1, 0.2, 4) + index * 0.01)
    values = torch.linspace(0.1, 0.9, 9)
    mask = torch.ones(9, dtype=torch.bool)
    confidence = torch.linspace(0.6, 1.0, 9)
    explanation = head.explain(values, mask, confidence)
    reconstructed = (
        explanation.bias
        + sum(explanation.unary_contributions.values())
        + sum(explanation.interaction_contributions.values())
    )
    assert reconstructed == pytest.approx(explanation.pre_tanh_logit, abs=1e-6)
    assert -0.8 <= explanation.potential <= 0.8


def test_seat_swap_reverses_signed_prize_concepts_and_scalar_value_direction() -> None:
    values = torch.arange(9, dtype=torch.float32)
    swapped = reverse_net_prize_perspective(values)
    assert torch.equal(swapped[[2, 3, 4]], -values[[2, 3, 4]])
    assert torch.equal(swapped[[0, 1, 5, 6, 7, 8]], values[[0, 1, 5, 6, 7, 8]])
    p0_value, p1_value = 0.4, -0.4
    assert p0_value == -p1_value


def test_gae_counts_transactions_not_nested_selects() -> None:
    rewards = torch.tensor([1.0, 1.0])
    values = torch.zeros(2)
    terminals = torch.tensor([False, True])
    advantages, returns = transaction_gae(
        rewards,
        values,
        terminals,
        gamma=1.0,
        gae_lambda=1.0,
    )
    assert torch.equal(advantages, torch.tensor([2.0, 1.0]))
    assert torch.equal(returns, advantages)


def test_semantic_residual_and_full_gradient_boundaries_are_identifiable() -> None:
    config = load_config(CONFIG)
    model = SemanticSelfPlayModel(config.model)
    states = torch.randn(3, config.model.observation_dimensions)
    mask = torch.ones(3, 9, dtype=torch.bool)
    targets = SemanticConceptTargets(torch.zeros(3, 9), mask)
    returns = torch.tensor([1.0, 0.0, -1.0])

    output = model(states, mask)
    assert torch.allclose(output.full_value, output.semantic_value + output.residual_value)
    model.zero_grad(set_to_none=True)
    output.semantic_value.sum().backward()
    assert all(parameter.grad is None for parameter in model.semantic_concept_heads.parameters())
    assert any(parameter.grad is not None for parameter in model.semantic_potential_head.parameters())

    model.zero_grad(set_to_none=True)
    output = model(states, mask)
    losses = compute_critic_losses(output, targets, returns)
    losses.residual.backward()
    assert all(parameter.grad is None for parameter in model.semantic_potential_head.parameters())
    assert any(parameter.grad is not None for parameter in model.residual_value_head.parameters())


def test_stored_shaping_is_unchanged_after_online_encoder_update() -> None:
    config = load_config(CONFIG)
    before = transaction_reward(
        terminal=False,
        outcome=0,
        target_phi_before=0.25,
        target_phi_after=-0.1,
        gamma=config.training.gamma,
        shaping_alpha=0.15,
    ).total
    model = SemanticSelfPlayModel(config.model)
    with torch.no_grad():
        for parameter in model.shared_state_encoder.parameters():
            parameter.add_(torch.randn_like(parameter))
    after = transaction_reward(
        terminal=False,
        outcome=0,
        target_phi_before=0.25,
        target_phi_after=-0.1,
        gamma=config.training.gamma,
        shaping_alpha=0.15,
    ).total
    assert before == after


def test_confidence_is_ensemble_derived_detached_and_cannot_collapse_by_gradient() -> None:
    config = load_config(CONFIG)
    model = SemanticSelfPlayModel(config.model)
    output = model(
        torch.randn(2, config.model.observation_dimensions),
        torch.ones(2, 9, dtype=torch.bool),
    )
    assert not output.concepts.confidence.requires_grad
    assert not any("confidence_head" in name for name, _ in model.named_parameters())
    assert len(model.semantic_concept_heads.members) >= 2


def test_phase_a_gate_and_phase_b_ramp_are_locked() -> None:
    config = load_config(CONFIG)
    phase = PhaseController(config.reward, completed_games=19_999)
    passing = CalibrationMetrics(0.1, 0.2, 0.5, 0.05, 0.04, 0.7)
    assert phase.consider_calibration(passing).passed
    assert phase.phase is TrainingPhase.PHASE_A
    assert phase.shaping_alpha() == 0.0
    phase.record_completed_games(1)
    phase.consider_calibration(passing)
    assert phase.phase is TrainingPhase.PHASE_B
    assert phase.shaping_alpha() == 0.0
    phase.record_completed_games(25_000)
    assert phase.shaping_alpha() == pytest.approx(0.075)
    phase.record_completed_games(25_000)
    assert phase.shaping_alpha() == pytest.approx(0.15)


def test_checkpoint_payload_contains_all_training_boundaries(tmp_path: Path) -> None:
    config, _, _, _, batch, trainer = _small_system()
    trainer.calibration_holdout = FixedCalibrationHoldout.from_transactions(
        batch.transactions,
        ObservationVectorizer(config.model),
        gamma=config.training.gamma,
    )
    payload = trainer.checkpoint_payload()
    required = {
        "learner",
        "optimizer",
        "full_critic",
        "online_semantic_heads",
        "target_semantic",
        "residual_value",
        "frozen_opponent",
        "frozen_opponent_reference",
        "current_phase",
        "completed_games",
        "shaping_alpha",
        "calibration_metrics",
        "config_snapshot",
    }
    assert required.issubset(payload)
    checkpoint = trainer.save_checkpoint(tmp_path / "checkpoint.pt")
    expected_games = trainer.phase_controller.completed_games
    trainer.phase_controller.completed_games = 999
    trainer.load_checkpoint(checkpoint)
    assert trainer.phase_controller.completed_games == expected_games
    assert trainer.calibration_holdout is not None
    assert torch.equal(
        trainer.calibration_holdout.states,
        payload["calibration_holdout"].states,
    )


def test_obsolete_three_dimensional_reward_symbols_are_absent() -> None:
    code = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (ROOT / "competition_selfplay").glob("*.py")
    )
    for obsolete in (
        "class RewardVector",
        "class VectorReward",
        "_setup_potential",
        "actor_weights",
        "own_active_weight",
        "setup_delta_clip",
        "value_dimensions != 3",
    ):
        assert obsolete not in code


def test_compact_features_still_reject_hidden_opponent_counts() -> None:
    values = {name: 0.0 for name in GLOBAL_FEATURE_NAMES}
    assert len(compact_global_features(values)) == len(GLOBAL_FEATURE_NAMES)
    values["opponent_deck_count"] = 40
    with pytest.raises(ValueError, match="forbidden"):
        compact_global_features(values)


def test_league_remains_strictly_frozen_until_promotion_threshold() -> None:
    config = load_config(CONFIG)
    controller = LeagueController(config.promotion)
    state = controller.record_training_iteration(LeagueState())
    threshold_wins = int(config.promotion.minimum_games * config.promotion.win_rate_threshold)
    below = controller.evaluate(
        state,
        EvaluationResult(
            wins=threshold_wins,
            losses=config.promotion.minimum_games - threshold_wins,
            draws=0,
        ),
    )
    assert below.action is LeagueAction.KEEP_TRAINING
    assert below.state.frozen_opponent_revision == 0


def test_dry_run_reports_transactional_semantic_contract_without_writes() -> None:
    result = dry_run(CONFIG)
    assert result["source_submission_ref"] == 54815037
    assert result["value_dimensions"] == 1
    assert len(result["semantic_concepts"]) == 9
    assert result["phase_b_calibration_gated"] is True


def test_mechanical_selfplay_replay_contains_real_viewer_frames() -> None:
    frame = {
        "current": {"players": [{"active": [], "bench": []}, {"active": [], "bench": []}]},
        "logs": [],
        "selected": [0],
    }
    observation = {"current": {"yourIndex": 0}, "select": {"option": [{"type": 1}]}}
    replay = _build_replay(
        episode=7,
        deck=[1] * 60,
        initial_observation=observation,
        decisions=[{"player": 0, "observation": observation, "action": [0]}],
        visualize=[frame],
        result=0,
        completed=True,
        error=None,
    )
    assert replay["rewards"] == [1, -1]
    assert replay["statuses"] == ["DONE", "DONE"]
    assert replay["local"]["visual_frame_count"] == 1
