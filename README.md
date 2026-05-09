# Video Pair Rating Tool

Run a Gradio interface for collecting human preferences between paired videos.

```bash
python3 -m pip install -r requirements.txt
python3 app.py
```

By default, the app compares matching filenames in:

- `wan_vs_ours_new/wan`
- `wan_vs_ours_new/ours`

Each user session shuffles the pair order and randomly swaps which model appears on the left. Ratings are appended to `ratings.csv` with the true left/right mapping.

Useful options:

```bash
python3 app.py --output results/ratings.csv
python3 app.py --baseline-dir path/to/baseline --ours-dir path/to/ours
python3 app.py --show-model-names
python3 app.py --server-port 7960
python3 app.py --share
```

To select videos by the order they appear on `index.html`, use `--include-indices`.
Ranges and comma-separated values are supported:

```bash
python3 app.py --include-indices "1-6,11,14,16,17,19,22,24-27"
```

This automatically reads pairs from `index.html` in website order.
