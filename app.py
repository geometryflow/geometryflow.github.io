from __future__ import annotations

import argparse
import csv
import random
import re
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import gradio as gr


Rating = Literal["baseline", "ours", "same"]

CSV_FIELDS = [
    "timestamp_utc",
    "rater_id",
    "pair_index",
    "pair_id",
    "rating",
    "chosen_side",
    "left_model",
    "right_model",
    "baseline_path",
    "ours_path",
]

WRITE_LOCK = threading.Lock()


@dataclass(frozen=True)
class VideoPair:
    pair_id: str
    baseline_path: Path
    ours_path: Path


@dataclass(frozen=True)
class DisplayPair:
    pair: VideoPair
    left_path: Path
    right_path: Path
    left_model: str
    right_model: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rate paired video generations with Gradio.")
    parser.add_argument(
        "--baseline-dir",
        type=Path,
        default=Path("wan_vs_ours_new/wan"),
        help="Directory containing baseline videos.",
    )
    parser.add_argument(
        "--ours-dir",
        type=Path,
        default=Path("wan_vs_ours_new/ours"),
        help="Directory containing our generated videos.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("ratings.csv"),
        help="CSV file where ratings are appended.",
    )
    parser.add_argument(
        "--show-model-names",
        action="store_true",
        help="Show true model labels in the UI instead of blinded A/B labels.",
    )
    parser.add_argument(
        "--website-order",
        action="store_true",
        help="Load video pairs from index.html in the same order as the website.",
    )
    parser.add_argument(
        "--include-indices",
        default=None,
        help='Website-order indices to include, for example "1-6,11,14,16,17,19,22,24-27".',
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public Gradio share link.",
    )
    parser.add_argument(
        "--server-name",
        default=None,
        help="Host/interface for Gradio to bind, for example 0.0.0.0.",
    )
    parser.add_argument(
        "--server-port",
        type=int,
        default=None,
        help="Port for Gradio to bind. Useful when the default range is occupied.",
    )
    return parser.parse_args()


def find_pairs(baseline_dir: Path, ours_dir: Path) -> list[VideoPair]:
    baseline_dir = baseline_dir.expanduser().resolve()
    ours_dir = ours_dir.expanduser().resolve()

    if not baseline_dir.exists():
        raise FileNotFoundError(f"Baseline directory does not exist: {baseline_dir}")
    if not ours_dir.exists():
        raise FileNotFoundError(f"Ours directory does not exist: {ours_dir}")

    baseline_files = {
        path.name: path
        for path in baseline_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".mp4", ".webm", ".mov", ".m4v"}
    }
    ours_files = {
        path.name: path
        for path in ours_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".mp4", ".webm", ".mov", ".m4v"}
    }

    shared_names = sorted(baseline_files.keys() & ours_files.keys())
    if not shared_names:
        raise ValueError(
            f"No matching video filenames found in {baseline_dir} and {ours_dir}."
        )

    return [
        VideoPair(
            pair_id=Path(name).stem,
            baseline_path=baseline_files[name],
            ours_path=ours_files[name],
        )
        for name in shared_names
    ]


def find_website_pairs(index_path: Path = Path("index.html")) -> list[VideoPair]:
    index_path = index_path.expanduser().resolve()
    html = index_path.read_text()
    root = index_path.parent
    pairs: list[VideoPair] = []

    for card_number, match in enumerate(
        re.finditer(r'<div class="comparison-card" data-scene="([^"]+)">', html),
        start=1,
    ):
        start = match.start()
        end = html.find('<div class="annotation"', start)
        if end == -1:
            continue

        chunk = html[start:end]
        sources = re.findall(r'<source src="([^"]+\.(?:mp4|webm|mov|m4v))"', chunk)
        if len(sources) < 2:
            continue

        scene = match.group(1)
        pairs.append(
            VideoPair(
                pair_id=f"{card_number:02d}_{scene}",
                baseline_path=(root / sources[0]).resolve(),
                ours_path=(root / sources[1]).resolve(),
            )
        )

    if not pairs:
        raise ValueError(f"No video pairs found in {index_path}.")

    return pairs


def parse_index_selection(selection: str, total: int) -> list[int]:
    selected: list[int] = []
    seen: set[int] = set()

    for raw_part in selection.split(","):
        part = raw_part.strip()
        if not part:
            continue

        if "-" in part:
            start_text, end_text = part.split("-", 1)
            start = int(start_text.strip())
            end = int(end_text.strip())
            if start > end:
                raise ValueError(f"Invalid descending range: {part}")
            values = range(start, end + 1)
        else:
            values = [int(part)]

        for value in values:
            if value < 1 or value > total:
                raise ValueError(f"Video index {value} is outside 1-{total}.")
            if value not in seen:
                selected.append(value)
                seen.add(value)

    if not selected:
        raise ValueError("No video indices were selected.")

    return selected


def select_pairs_by_indices(pairs: list[VideoPair], selection: str) -> list[VideoPair]:
    indices = parse_index_selection(selection, len(pairs))
    return [pairs[index - 1] for index in indices]


def build_session_pairs(pairs: list[VideoPair]) -> list[DisplayPair]:
    session_pairs: list[DisplayPair] = []
    shuffled_pairs = pairs[:]
    random.shuffle(shuffled_pairs)

    for pair in shuffled_pairs:
        if random.choice([True, False]):
            session_pairs.append(
                DisplayPair(
                    pair=pair,
                    left_path=pair.baseline_path,
                    right_path=pair.ours_path,
                    left_model="baseline",
                    right_model="ours",
                )
            )
        else:
            session_pairs.append(
                DisplayPair(
                    pair=pair,
                    left_path=pair.ours_path,
                    right_path=pair.baseline_path,
                    left_model="ours",
                    right_model="baseline",
                )
            )
    return session_pairs


def ensure_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with path.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
            writer.writeheader()


def append_rating(
    output_path: Path,
    rater_id: str,
    index: int,
    display_pair: DisplayPair,
    rating: Rating,
) -> None:
    if rating == "same":
        chosen_side = "same"
    elif display_pair.left_model == rating:
        chosen_side = "left"
    else:
        chosen_side = "right"

    row = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "rater_id": rater_id,
        "pair_index": index,
        "pair_id": display_pair.pair.pair_id,
        "rating": rating,
        "chosen_side": chosen_side,
        "left_model": display_pair.left_model,
        "right_model": display_pair.right_model,
        "baseline_path": str(display_pair.pair.baseline_path),
        "ours_path": str(display_pair.pair.ours_path),
    }

    with WRITE_LOCK:
        ensure_csv(output_path)
        with output_path.open("a", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
            writer.writerow(row)


def make_app(
    pairs: list[VideoPair],
    output_path: Path,
    show_model_names: bool,
) -> gr.Blocks:
    def start_session() -> tuple[str, list[DisplayPair], int, str, str, str, str, str]:
        session_pairs = build_session_pairs(pairs)
        return render_pair(str(uuid.uuid4()), session_pairs, 0)

    def render_pair(
        rater_id: str,
        session_pairs: list[DisplayPair],
        index: int,
    ) -> tuple[str, list[DisplayPair], int, str, str, str, str, str]:
        display_pair = session_pairs[index]
        total = len(session_pairs)

        if show_model_names:
            left_label = f"Left video: {display_pair.left_model}"
            right_label = f"Right video: {display_pair.right_model}"
        else:
            left_label = "Left video: A"
            right_label = "Right video: B"

        progress = f"Pair {index + 1} of {total}"
        return (
            rater_id,
            session_pairs,
            index,
            str(display_pair.left_path),
            str(display_pair.right_path),
            left_label,
            right_label,
            progress,
        )

    def rate(
        rating: Rating,
        rater_id: str,
        session_pairs: list[DisplayPair],
        index: int,
    ) -> tuple[str, list[DisplayPair], int, str | None, str | None, str, str, str, str]:
        if index >= len(session_pairs):
            return (
                rater_id,
                session_pairs,
                index,
                None,
                None,
                "Complete",
                "Complete",
                f"Finished {len(session_pairs)} ratings.",
                "This session is already complete.",
            )

        display_pair = session_pairs[index]
        append_rating(output_path, rater_id, index, display_pair, rating)

        next_index = index + 1
        if next_index >= len(session_pairs):
            return (
                rater_id,
                session_pairs,
                next_index,
                None,
                None,
                "Complete",
                "Complete",
                f"Finished {len(session_pairs)} ratings.",
                "Thanks. Your ratings were recorded.",
            )

        rendered = render_pair(rater_id, session_pairs, next_index)
        return (*rendered, "Recorded.")

    with gr.Blocks(title="Video Pair Rating") as app:
        rater_id = gr.State()
        session_pairs = gr.State()
        index = gr.State()

        gr.Markdown("# Video Pair Rating")
        gr.Markdown("Compare each video pair and choose which one is better, or mark them as the same.")

        with gr.Row():
            progress = gr.Markdown()
            status = gr.Markdown()

        with gr.Row(equal_height=True):
            with gr.Column():
                left_label = gr.Markdown()
                left_video = gr.Video(label=None, autoplay=True, loop=True)
            with gr.Column():
                right_label = gr.Markdown()
                right_video = gr.Video(label=None, autoplay=True, loop=True)

        with gr.Row():
            left_better = gr.Button("Left is better", variant="primary")
            same = gr.Button("Same", variant="secondary")
            right_better = gr.Button("Right is better", variant="primary")

        def left_rating(
            rater_id_value: str,
            session_pairs_value: list[DisplayPair],
            index_value: int,
        ):
            if index_value >= len(session_pairs_value):
                return rate("same", rater_id_value, session_pairs_value, index_value)
            rating: Rating = session_pairs_value[index_value].left_model  # type: ignore[assignment]
            return rate(rating, rater_id_value, session_pairs_value, index_value)

        def right_rating(
            rater_id_value: str,
            session_pairs_value: list[DisplayPair],
            index_value: int,
        ):
            if index_value >= len(session_pairs_value):
                return rate("same", rater_id_value, session_pairs_value, index_value)
            rating: Rating = session_pairs_value[index_value].right_model  # type: ignore[assignment]
            return rate(rating, rater_id_value, session_pairs_value, index_value)

        outputs = [
            rater_id,
            session_pairs,
            index,
            left_video,
            right_video,
            left_label,
            right_label,
            progress,
            status,
        ]
        inputs = [rater_id, session_pairs, index]

        app.load(
            start_session,
            inputs=None,
            outputs=[
                rater_id,
                session_pairs,
                index,
                left_video,
                right_video,
                left_label,
                right_label,
                progress,
            ],
        )
        left_better.click(left_rating, inputs=inputs, outputs=outputs)
        same.click(lambda *args: rate("same", *args), inputs=inputs, outputs=outputs)
        right_better.click(right_rating, inputs=inputs, outputs=outputs)

    return app


def main() -> None:
    args = parse_args()
    if args.website_order or args.include_indices:
        pairs = find_website_pairs()
    else:
        pairs = find_pairs(args.baseline_dir, args.ours_dir)

    if args.include_indices:
        pairs = select_pairs_by_indices(pairs, args.include_indices)

    ensure_csv(args.output)
    app = make_app(pairs, args.output.expanduser().resolve(), args.show_model_names)
    app.launch(
        share=args.share,
        server_name=args.server_name,
        server_port=args.server_port,
    )


if __name__ == "__main__":
    main()
