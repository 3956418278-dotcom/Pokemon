from __future__ import annotations

import argparse
import time

import torch

from data.game_memory import GameMemoryState
from data.observation_parser import parse_observation
from data.state_schema import collate_card_dynamic
from models.board_tokenizer import BoardTokenizer
from models.board_transformer import BoardTransformer
from models.card_instance_fusion import CardInstanceFusion
from models.dynamic_instance_encoder import DynamicInstanceEncoder
from models.static_card_adapter import StaticCardEmbeddingAdapter


def _card(card_id: int, serial: int, player: int) -> dict:
    return {"id": card_id, "serial": serial, "playerIndex": player}


def _pokemon(card_id: int, serial: int, player: int, hp: int = 100) -> dict:
    return {
        "id": card_id,
        "serial": serial,
        "playerIndex": player,
        "hp": hp,
        "maxHp": 120,
        "appearThisTurn": False,
        "energies": [0, 2],
        "energyCards": [_card(1, serial + 100, player)],
        "tools": [],
        "preEvolution": [],
    }


def _observation() -> dict:
    players = []
    for player in [0, 1]:
        players.append(
            {
                "active": [_pokemon(21 + player, 10 + player, player)],
                "bench": [_pokemon(30 + player * 10 + index, 20 + player * 10 + index, player) for index in range(3)],
                "benchMax": 5,
                "deckCount": 42,
                "discard": [_card(80 + player, 70 + player, player)],
                "prize": [None for _ in range(6)],
                "handCount": 5,
                "hand": [_card(100 + index, 1000 + index, player) for index in range(5)] if player == 0 else None,
                "poisoned": False,
                "burned": False,
                "asleep": False,
                "paralyzed": False,
                "confused": False,
            }
        )
    return {
        "current": {
            "turn": 5,
            "turnActionCount": 3,
            "yourIndex": 0,
            "firstPlayer": 0,
            "supporterPlayed": False,
            "stadiumPlayed": False,
            "energyAttached": True,
            "retreated": False,
            "result": -1,
            "stadium": [],
            "looking": None,
            "players": players,
        },
        "select": {
            "type": 0,
            "context": 0,
            "minCount": 1,
            "maxCount": 1,
            "remainDamageCounter": 0,
            "remainEnergyCost": 0,
            "option": [{"type": 14}],
            "deck": None,
            "contextCard": None,
            "effect": None,
        },
        "logs": [{"type": 2, "playerIndex": 0}, {"type": 11, "playerIndex": 0, "cardId": 1, "serial": 110}],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()

    parsed = parse_observation(_observation())
    memory = GameMemoryState().update_from_parsed(parsed)
    adapter = StaticCardEmbeddingAdapter(torch.randn(160, 128), {str(i): i - 1 for i in range(1, 161)})
    dynamic_encoder = DynamicInstanceEncoder()
    fusion = CardInstanceFusion()
    tokenizer = BoardTokenizer()
    transformer = BoardTransformer(dropout=0.0)

    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(args.iterations):
            parsed = parse_observation(_observation())
            memory.update_from_parsed(parsed)
            batch = collate_card_dynamic(parsed.card_instances, memory.appearance_features(parsed.card_instances))
            static_embeddings, _known = adapter(batch.card_ids)
            dynamic_embeddings = dynamic_encoder(batch)
            instance_embeddings = fusion(static_embeddings, dynamic_embeddings)
            tokenized = tokenizer(
                instance_embeddings,
                batch.visibility_mask,
                torch.tensor(parsed.global_snapshot.features(), dtype=torch.float32),
                area_ids=torch.tensor([instance.area for instance in parsed.card_instances], dtype=torch.long),
                decision_features=torch.tensor(parsed.global_snapshot.decision_features(), dtype=torch.float32),
                match_features=torch.tensor(parsed.global_snapshot.match_features(), dtype=torch.float32),
                ledger_features=torch.tensor(memory.ledger_features(parsed.global_snapshot.your_index), dtype=torch.float32),
                event_features=torch.tensor(memory.recent_event_features(), dtype=torch.float32)
                if memory.recent_event_features()
                else None,
            )
            transformer(tokenized.tokens, tokenized.mask)
    elapsed = time.perf_counter() - start
    print(
        {
            "iterations": args.iterations,
            "elapsed_sec": round(elapsed, 4),
            "ms_per_iteration": round(elapsed * 1000.0 / args.iterations, 4),
            "instance_count": len(parsed.card_instances),
            "recent_event_count": len(memory.recent_events),
        }
    )


if __name__ == "__main__":
    main()
