from __future__ import annotations

import torch

from decision_agent_v1.adapters.simulator_adapter import LiveDecisionView
from decision_agent_v1.baseline.deterministic_legal_agent import DeterministicLegalAgent
from decision_agent_v1.contracts.schemas import DecisionSampleV1, SelectionMode, TerminalOutcome
from decision_agent_v1.data.collate import collate_decision_samples


class PolicyValueAgent:
    def __init__(self, model: torch.nn.Module, device: str = "cpu") -> None:
        self.model = model.to(device)
        self.device = device
        self.legal_baseline = DeterministicLegalAgent()

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        *,
        device: str = "cpu",
        expected_hashes: dict[str, str] | None = None,
    ) -> "PolicyValueAgent":
        from decision_agent_v1.models.policy_value_model import PolicyValueModel

        checkpoint = torch.load(checkpoint_path, map_location=device)
        if expected_hashes is not None:
            for key in (
                "data_schema_hash",
                "action_contract_hash",
                "card_vocabulary_hash",
            ):
                if checkpoint.get(key) != expected_hashes.get(key):
                    raise RuntimeError(f"checkpoint {key} mismatch")
        model = PolicyValueModel(**checkpoint["model_config"])
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        return cls(model, device=device)

    @torch.no_grad()
    def act(self, view: LiveDecisionView) -> list[int]:
        if view.selection_mode is SelectionMode.UNKNOWN:
            return self.legal_baseline.act(view)
        sample = DecisionSampleV1(
            episode_id="live",
            source_date=None,
            agent_index=0,
            decision_index=0,
            step=0,
            turn=view.global_state.turn,
            turn_action_count=view.global_state.turn_action_count,
            cards=view.cards,
            global_state=view.global_state,
            history=view.history,
            options=view.options,
            selected_option_indices=(),
            selected_equivalence_groups=(),
            min_count=view.global_state.min_count,
            max_count=view.global_state.max_count,
            selection_mode=view.selection_mode,
            terminal_outcome=TerminalOutcome.DRAW,
            episode_decision_count=1,
            policy_supervision=False,
            policy_mask_reason="LIVE_INFERENCE",
        )
        batch = collate_decision_samples([sample])
        batch = {
            key: value.to(self.device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }
        self.model.eval()
        outputs = self.model(batch)
        count = 1 if view.selection_mode is SelectionMode.SINGLE else view.global_state.min_count
        positions = self.model.multiselect_decoder.decode(
            outputs["policy_logits"], outputs["option_embeddings"], batch["option_mask"], count
        )[0]
        return [view.options[position].original_option_index for position in positions]
