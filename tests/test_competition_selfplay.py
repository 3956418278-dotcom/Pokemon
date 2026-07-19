from __future__ import annotations

from pathlib import Path

import pytest

from competition_selfplay.cli import dry_run
from competition_selfplay.config import load_config
from competition_selfplay.deck import load_target_deck
from competition_selfplay.features import GLOBAL_FEATURE_NAMES, compact_global_features
from competition_selfplay.league import EvaluationResult, LeagueAction, LeagueController, LeagueState
from competition_selfplay.reward import BattleSnapshot, TerminalReason, VectorReward
from competition_selfplay.run_mechanical_selfplay import _build_replay


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "competition_selfplay/configs/raging_bolt_fast.yaml"


def _state(**overrides: int) -> BattleSnapshot:
    values = {
        "turn": 3,
        "self_prize_count": 6,
        "opponent_prize_count": 6,
        "self_active_count": 1,
        "self_bench_count": 1,
        "self_attached_energy_count": 1,
        "opponent_active_count": 1,
        "opponent_bench_count": 1,
    }
    values.update(overrides)
    return BattleSnapshot(**values)


def test_scored_target_deck_is_exact_and_copy_aware() -> None:
    config = load_config(CONFIG)
    deck = load_target_deck(config.target_deck, ROOT)
    assert deck.name == "Raging Bolt Ogerpon"
    assert len(deck.card_ids) == 60
    assert len(set(deck.card_ids)) == 27
    kangaskhan = [token for token in deck.card_tokens if token.card_id == 756]
    assert [token.copy_ordinal for token in kangaskhan] == [0, 1, 2]
    assert {token.copies_in_deck for token in kangaskhan} == {3}
    indices = {token.card_index for token in deck.card_tokens}
    assert indices == set(range(2, 2 + len(set(deck.card_ids))))


def test_reward_is_vector_and_deck_out_is_asymmetric() -> None:
    config = load_config(CONFIG)
    reward = VectorReward(config.reward)
    before = _state()
    after = _state(opponent_prize_count=4, self_bench_count=3, self_attached_energy_count=3)

    normal_win = reward.transition(
        before, after, perspective_player=0, winner=0, terminal_reason=TerminalReason.PRIZES
    )
    deck_out_win = reward.transition(
        before, after, perspective_player=0, winner=0, terminal_reason=TerminalReason.DECK_OUT
    )
    self_deck_out = reward.transition(
        before, after, perspective_player=0, winner=1, terminal_reason=TerminalReason.DECK_OUT
    )

    assert len(normal_win.as_tuple()) == 3
    assert normal_win.outcome == 1.0
    assert normal_win.prize_progress == pytest.approx(2 / 6)
    assert normal_win.setup_tempo > 0
    assert deck_out_win.outcome == 0.0
    assert self_deck_out.outcome == -1.25


def test_compact_features_reject_non_field_card_counts() -> None:
    values = {name: 0.0 for name in GLOBAL_FEATURE_NAMES}
    assert len(compact_global_features(values)) == len(GLOBAL_FEATURE_NAMES)
    values["opponent_deck_count"] = 40
    with pytest.raises(ValueError, match="forbidden"):
        compact_global_features(values)


def test_frozen_opponent_changes_only_after_strict_threshold() -> None:
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

    above_wins = threshold_wins + 2
    above = controller.evaluate(
        state,
        EvaluationResult(
            wins=above_wins,
            losses=config.promotion.minimum_games - above_wins,
            draws=0,
        ),
    )
    assert above.action is LeagueAction.PROMOTE_AND_COPY
    assert above.state.frozen_opponent_revision == state.learner_revision
    assert above.state.generation == 1


def test_dry_run_validates_without_writing_outputs() -> None:
    result = dry_run(CONFIG)
    assert result["source_submission_ref"] == 54815037
    assert result["value_dimensions"] == 3
    assert result["passed"] is True


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
    assert replay["steps"][0][0]["visualize"][0]["current"]["players"] == frame["current"]["players"]
    assert replay["steps"][1][0]["action"] == [0]
