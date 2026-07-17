# Static card data

`static_card` owns the canonical, print-level card corpus. It does not parse replay observations or implement encoders/training.

## Build and load

Run preprocessing from the repository root:

```bash
python static_card/data/card_preprocessing.py
```

The canonical cache is written to `static_card/artifacts/card_data/` as `cards.json`, `details.json`, `detail_offsets.json`, `card_id_to_index.json`, and `preprocess_manifest.json`. `card_preprocessing.py` remains the only CSV/`cg.api` reader and Card ID aggregator.

Load model-facing data with:

```python
from static_card.data.card_dataset import CardDataset, collate_cards

dataset = CardDataset.from_cache()
sample = dataset[0]
batch = collate_cards([sample])
```

Each sample contains `index`, metadata-only `card_id`, an encoded `card`, and zero or more independently encoded `details`. Card features cover identity/role/type/category/evolution, masked HP/type/energy/weakness/resistance/retreat fields, structured provided-energy amount/counts/allowed-type-mask/mode/restriction semantics, and stable vocab IDs. Fixed energy profiles retain numeric counts; choice-based profiles keep unresolved counts at zero and expose their legal types through the separate boolean mask. Detail features cover type/subtype/name/joint identity, explicit text-token IDs, attack-local energy cost, damage/mode, and source IDs. No whole-card text, hashed text buckets, random masks, aggregate attack statistics, or training targets are produced here.

`collate_cards` dynamically pads evolution targets, details, and detail text. A Basic Energy retains zero permanent details and receives only masked batch padding. Future static encoders should consume this batch contract without reading raw CSV or reconstructing details.

## Split semantics

`split_train_validation_test` is a deterministic Card-ID partition used only for
optimization and regression monitoring within the fixed canonical card pool. The
feature schema, vocabularies, and normalization statistics are intentionally built
from that fixed pool before splitting. Consequently, validation/test metrics must
not be reported as unseen-card generalization. A future unseen-Card-ID benchmark
would require a train-only schema and explicit UNK handling and is not implemented
by this split.
