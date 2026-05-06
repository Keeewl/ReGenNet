import argparse
import csv
import random
import shutil
from pathlib import Path
import re


QUESTION_CONTACT = (
    "Contact Quality:\n"
    "Which video shows more natural and accurate physical contact? Please focus on hand contact "
    "alignment, floating hands, penetration, or obvious misalignment.\n\n"
    "接触质量：\n"
    "哪个视频中的身体接触更自然、更准确？请重点关注手部接触是否对齐，是否存在悬空、穿模或明显错位。"
)

QUESTION_REACTION = (
    "Reaction Appropriateness:\n"
    "Which video shows a more appropriate reaction to the other person's motion? Please focus on "
    "timing, direction, and whether the response matches the interaction semantics.\n\n"
    "反应合理性：\n"
    "哪个视频对另一人的动作做出了更合适的反应？请重点关注反应的时机、方向，以及是否符合交互语义。"
)

QUESTION_REALISM = (
    "Overall Realism:\n"
    "Which video looks more realistic, natural, and coherent overall? Please consider motion "
    "smoothness, body pose plausibility, interaction consistency, and visual credibility.\n\n"
    "整体真实感：\n"
    "哪个视频整体看起来更真实、自然、连贯？请综合考虑动作平滑性、姿态合理性、交互一致性和视觉可信度。"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Randomize user-study videos into A/B/C/D and export mapping/questionnaire CSVs."
    )
    parser.add_argument("--input_root", required=True, help="Root directory containing per-clip videos.")
    parser.add_argument("--output_root", required=True, help="Root directory for anonymized outputs.")
    parser.add_argument("--seed", type=int, default=24, help="Random seed for reproducible shuffling.")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["baseline", "hireact_star", "hireact", "gt"],
        help="Method names to randomize. Default: baseline hireact_star hireact gt",
    )
    parser.add_argument(
        "--copy_mode",
        choices=["copy", "symlink"],
        default="copy",
        help="Whether to copy or symlink anonymized videos.",
    )
    parser.add_argument(
        "--video_ext",
        default=".mp4",
        help="Video extension to search for. Default: .mp4",
    )
    parser.add_argument(
        "--layout",
        choices=["auto", "user_study_pack", "flat"],
        default="auto",
        help=(
            "Input layout. 'user_study_pack' expects per-clip directories; "
            "'flat' expects files like <dataset_key>-baseline_0.mp4 under input_root; "
            "'auto' detects automatically."
        ),
    )
    parser.add_argument(
        "--selected_keys_file",
        help="Optional text/markdown file containing one dataset_key per line; only selected clips will be randomized.",
    )
    parser.add_argument(
        "--gt_fixed_option",
        choices=["A", "B", "C", "D"],
        help="If set, always place method 'gt' at this anonymous option, and shuffle the remaining methods over the remaining options.",
    )
    return parser.parse_args()


def normalize_ext(ext: str) -> str:
    return ext if ext.startswith(".") else f".{ext}"


def first_video_under(root: Path, video_ext: str):
    if not root.exists():
        return None
    videos = sorted(p for p in root.rglob(f"*{video_ext}") if p.is_file())
    return videos[0] if videos else None


def resolve_method_root(clip_dir: Path, method: str):
    if method == "baseline":
        candidates = [
            clip_dir / "baseline",
            clip_dir / "baseline" / "motions",
        ]
    elif method == "hireact_star":
        candidates = [
            clip_dir / "hireact_star",
            clip_dir / "stage1",
            clip_dir / "stage1" / "motions",
        ]
    elif method == "hireact":
        candidates = [
            clip_dir / "hireact",
            clip_dir / "stage2" / "refined",
        ]
    elif method == "gt":
        candidates = [
            clip_dir / "gt",
        ]
    else:
        candidates = [
            clip_dir / method,
        ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def resolve_method_video(clip_dir: Path, method: str, video_ext: str):
    method_root = resolve_method_root(clip_dir, method)
    if method_root is None:
        return None
    return first_video_under(method_root, video_ext)


def iter_clip_dirs(input_root: Path):
    return sorted(p for p in input_root.iterdir() if p.is_dir() and not p.name.startswith("."))


def canonical_method_name(method: str) -> str:
    aliases = {
        "hireact_star": "stage1",
        "stage1": "stage1",
        "hireact": "stage2",
        "stage2": "stage2",
        "baseline": "baseline",
        "gt": "gt",
    }
    return aliases.get(method, method)


def detect_layout(input_root: Path, video_ext: str) -> str:
    clip_dirs = iter_clip_dirs(input_root)
    if clip_dirs:
        return "user_study_pack"
    flat_videos = sorted(input_root.glob(f"*{video_ext}"))
    if flat_videos:
        return "flat"
    return "user_study_pack"


def build_flat_index(input_root: Path, video_ext: str):
    pattern = re.compile(r"^(?P<clip>.+)-(?P<method>baseline|stage1|stage2|gt)(?:_[^/\\]+)?$")
    index = {}
    for video_path in sorted(input_root.glob(f"*{video_ext}")):
        stem = video_path.stem
        match = pattern.match(stem)
        if not match:
            continue
        clip_key = match.group("clip")
        method = match.group("method")
        index.setdefault(clip_key, {})[method] = video_path.resolve()
    return index


def canonical_clip_key(name: str) -> str:
    return re.sub(r"-pack\d+$", "", name)


def load_selected_keys(path: Path | None):
    if path is None:
        return None
    selected = set()
    text = path.read_text(encoding="utf-8-sig")
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        selected.add(line)
    return selected


def ensure_parent(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)


def write_video(src: Path, dst: Path, copy_mode: str):
    ensure_parent(dst)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy_mode == "copy":
        shutil.copy2(src, dst)
    else:
        dst.symlink_to(src.resolve())


def main():
    args = parse_args()
    rng = random.Random(args.seed)
    input_root = Path(args.input_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    videos_root = output_root / "videos"
    mapping_path = output_root / "mapping.csv"
    questionnaire_path = output_root / "questionnaire_template.csv"
    video_ext = normalize_ext(args.video_ext)
    methods = list(args.methods)
    options = ["A", "B", "C", "D"]
    canonical_methods = [canonical_method_name(m) for m in methods]
    selected_keys = load_selected_keys(Path(args.selected_keys_file).expanduser().resolve() if args.selected_keys_file else None)

    output_root.mkdir(parents=True, exist_ok=True)
    videos_root.mkdir(parents=True, exist_ok=True)

    layout = args.layout
    if layout == "auto":
        layout = detect_layout(input_root, video_ext)

    flat_index = {}
    if layout == "flat":
        flat_index = build_flat_index(input_root, video_ext)
        found_clip_keys = sorted(flat_index.keys())
        if selected_keys is not None:
            found_clip_keys = [
                clip_key for clip_key in found_clip_keys
                if canonical_clip_key(clip_key) in selected_keys
            ]
        found_clip_dirs = [input_root / clip_key for clip_key in found_clip_keys]
    else:
        found_clip_dirs = iter_clip_dirs(input_root)
        if selected_keys is not None:
            found_clip_dirs = [
                clip_dir for clip_dir in found_clip_dirs
                if canonical_clip_key(clip_dir.name) in selected_keys
            ]

    mapping_rows = []
    questionnaire_rows = []
    exported_count = 0
    skipped_count = 0

    for clip_idx, clip_dir in enumerate(found_clip_dirs, start=1):
        clip_source_name = clip_dir.name
        clip_dataset_key = canonical_clip_key(clip_source_name)
        clip_id = f"clip_{clip_idx:03d}"
        resolved = {}
        missing = []
        for method, canonical_method in zip(methods, canonical_methods):
            if layout == "flat":
                video_path = flat_index.get(clip_source_name, {}).get(canonical_method)
            else:
                video_path = resolve_method_video(clip_dir, method, video_ext)
            if video_path is None:
                missing.append(method)
            else:
                resolved[method] = video_path.resolve()

        if missing:
            print(
                f"WARNING: skip {clip_source_name} because missing videos for methods: "
                f"{', '.join(missing)}"
            )
            skipped_count += 1
            continue

        if args.gt_fixed_option:
            if "gt" not in methods:
                raise ValueError("--gt_fixed_option requires 'gt' to be present in --methods")
            remaining_methods = [m for m in methods if m != "gt"]
            rng.shuffle(remaining_methods)
            option_to_method = {}
            remaining_options = [opt for opt in options if opt != args.gt_fixed_option]
            for opt, method in zip(remaining_options, remaining_methods):
                option_to_method[opt] = method
            option_to_method[args.gt_fixed_option] = "gt"
            shuffled_pairs = [(opt, option_to_method[opt]) for opt in options]
        else:
            shuffled_methods = methods[:]
            rng.shuffle(shuffled_methods)
            shuffled_pairs = list(zip(options, shuffled_methods))
        clip_out_dir = videos_root / clip_id
        clip_out_dir.mkdir(parents=True, exist_ok=True)

        question_row = {"clip_id": clip_id}
        for option, method in shuffled_pairs:
            src_path = resolved[method]
            anon_path = (clip_out_dir / f"{option}{video_ext}").resolve()
            write_video(src_path, anon_path, args.copy_mode)
            mapping_rows.append(
                {
                    "clip_id": clip_id,
                    "dataset_key": clip_dataset_key,
                    "source_name": clip_source_name,
                    "option": option,
                    "method": method,
                    "src_path": str(src_path),
                    "anon_path": str(anon_path),
                }
            )
            question_row[f"video_{option}"] = str(anon_path)

        question_row["question_contact"] = QUESTION_CONTACT
        question_row["question_reaction"] = QUESTION_REACTION
        question_row["question_realism"] = QUESTION_REALISM
        question_row["dataset_key"] = clip_dataset_key
        questionnaire_rows.append(question_row)
        exported_count += 1

    with mapping_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f, fieldnames=["clip_id", "dataset_key", "source_name", "option", "method", "src_path", "anon_path"]
        )
        writer.writeheader()
        writer.writerows(mapping_rows)

    with questionnaire_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "clip_id",
                "dataset_key",
                "video_A",
                "video_B",
                "video_C",
                "video_D",
                "question_contact",
                "question_reaction",
                "question_realism",
            ],
        )
        writer.writeheader()
        writer.writerows(questionnaire_rows)

    print(f"found clips: {len(found_clip_dirs)}")
    print(f"successfully exported clips: {exported_count}")
    print(f"skipped clips: {skipped_count}")
    print(f"mapping.csv: {mapping_path.resolve()}")
    print(f"questionnaire_template.csv: {questionnaire_path.resolve()}")


if __name__ == "__main__":
    main()
