# Companion CSV PPO Experiment

## Purpose

This historical experiment tested two independent ideas: deterministic fixed-width vectors built directly from the competition card CSV, and PPO utilities that score a variable set of legal action candidates.

It is not part of the current replay, dynamic-instance, memory, or Board mainline. It does not define the colleague static artifact contract and must not be imported by production code.

## Retained Conclusions

- Stable column-scoped hashing can produce reproducible flat vectors without Python's randomized `hash()`.
- Candidate scoring naturally supports variable legal-action counts and preserves permutation equivariance.
- PPO rollouts must retain the selected candidate index, old log probability, state value, reward, and terminal boundary.

Reusable concepts are the deterministic CSV validation rules, candidate-scoring actor interface, GAE boundary handling, and focused unit tests. Reuse requires promotion into a formal module with current interfaces and tests.

## Tests

From the repository root:

```bash
python -m pytest experiments/companion_csv_ppo/tests -q
```
