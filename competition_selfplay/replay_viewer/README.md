# Replay viewer

This static viewer is for human inspection of Kaggle PTCG replay JSON files. It reads
`steps[*][*].visualize[*].current`, renders both players' Active and Bench state, and supports
frame-by-frame or timed playback. Files stay in the browser and are not uploaded.

## Direct local use

Open `index.html` in a browser, then select or drag in a replay JSON file.

Local mechanical self-play replays are generated under
`outputs/competition_selfplay/mechanical_v2_selfplay/<run>/replays/`, one JSON per game. The sibling
`manifest.json` lists results and exact paths for the run.

## URL loading

From the repository root, start any static HTTP server, for example:

```bash
python -m http.server 8000
```

Then open:

```text
http://127.0.0.1:8000/competition_selfplay/replay_viewer/?replay=/replays/86823089.json
```

Keyboard controls: Space toggles playback, arrow keys move one frame, and Shift plus an arrow key
moves ten frames.
