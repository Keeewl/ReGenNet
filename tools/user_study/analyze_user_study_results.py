import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Decode user-study A/B/C/D responses back to real methods and summarize votes."
    )
    parser.add_argument(
        "--mapping_csv",
        default="visualize/user_study/randomized_videos/mapping.csv",
        help="Path to mapping.csv produced by randomize_user_study_videos.py",
    )
    parser.add_argument(
        "--responses_csv",
        default="visualize/user_study/results/responses.csv",
        help="Path to participant response CSV",
    )
    parser.add_argument(
        "--output_dir",
        default="visualize/user_study/results/analysis",
        help="Directory for decoded and summary CSV outputs",
    )
    return parser.parse_args()


def load_mapping(mapping_csv: Path):
    mapping = {}
    with mapping_csv.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mapping[(row["clip_id"], row["option"])] = row["method"]
    return mapping


def decode_responses(responses_csv: Path, mapping):
    decoded_rows = []
    with responses_csv.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            clip_id = row["clip_id"]
            contact_option = row["contact_choice"]
            reaction_option = row["reaction_choice"]
            realism_option = row["realism_choice"]

            decoded_rows.append(
                {
                    "participant_id": row["participant_id"],
                    "timestamp": row["timestamp"],
                    "clip_id": clip_id,
                    "contact_option": contact_option,
                    "contact_method": mapping.get((clip_id, contact_option), ""),
                    "reaction_option": reaction_option,
                    "reaction_method": mapping.get((clip_id, reaction_option), ""),
                    "realism_option": realism_option,
                    "realism_method": mapping.get((clip_id, realism_option), ""),
                }
            )
    return decoded_rows


def summarize(decoded_rows):
    criterion_counters = {
        "contact": Counter(),
        "reaction": Counter(),
        "realism": Counter(),
    }
    per_clip = defaultdict(lambda: {"contact": Counter(), "reaction": Counter(), "realism": Counter()})
    participant_counter = Counter()

    for row in decoded_rows:
        participant_counter[row["participant_id"]] += 1
        clip_id = row["clip_id"]

        criterion_counters["contact"][row["contact_method"]] += 1
        criterion_counters["reaction"][row["reaction_method"]] += 1
        criterion_counters["realism"][row["realism_method"]] += 1

        per_clip[clip_id]["contact"][row["contact_method"]] += 1
        per_clip[clip_id]["reaction"][row["reaction_method"]] += 1
        per_clip[clip_id]["realism"][row["realism_method"]] += 1

    return criterion_counters, per_clip, participant_counter


def write_csv(path: Path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    mapping_csv = Path(args.mapping_csv).expanduser().resolve()
    responses_csv = Path(args.responses_csv).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    mapping = load_mapping(mapping_csv)
    decoded_rows = decode_responses(responses_csv, mapping)
    criterion_counters, per_clip, participant_counter = summarize(decoded_rows)

    decoded_csv = output_dir / "decoded_responses.csv"
    criterion_csv = output_dir / "summary_by_criterion.csv"
    clip_csv = output_dir / "summary_by_clip.csv"

    write_csv(
        decoded_csv,
        [
            "participant_id",
            "timestamp",
            "clip_id",
            "contact_option",
            "contact_method",
            "reaction_option",
            "reaction_method",
            "realism_option",
            "realism_method",
        ],
        decoded_rows,
    )

    criterion_rows = []
    for criterion, counter in criterion_counters.items():
        total = sum(counter.values())
        for method, votes in sorted(counter.items()):
            criterion_rows.append(
                {
                    "criterion": criterion,
                    "method": method,
                    "votes": votes,
                    "ratio": f"{(votes / total):.4f}" if total else "0.0000",
                }
            )
    write_csv(criterion_csv, ["criterion", "method", "votes", "ratio"], criterion_rows)

    clip_rows = []
    for clip_id in sorted(per_clip.keys()):
        for criterion in ["contact", "reaction", "realism"]:
            total = sum(per_clip[clip_id][criterion].values())
            for method, votes in sorted(per_clip[clip_id][criterion].items()):
                clip_rows.append(
                    {
                        "clip_id": clip_id,
                        "criterion": criterion,
                        "method": method,
                        "votes": votes,
                        "ratio": f"{(votes / total):.4f}" if total else "0.0000",
                    }
                )
    write_csv(clip_csv, ["clip_id", "criterion", "method", "votes", "ratio"], clip_rows)

    print(f"participants: {len(participant_counter)}")
    print(f"response rows: {len(decoded_rows)}")
    for criterion in ["contact", "reaction", "realism"]:
        print(f"\n[{criterion}]")
        total = sum(criterion_counters[criterion].values())
        for method, votes in criterion_counters[criterion].most_common():
            ratio = votes / total if total else 0.0
            print(f"  {method}: {votes} ({ratio:.2%})")
    print(f"\ndecoded_responses.csv: {decoded_csv}")
    print(f"summary_by_criterion.csv: {criterion_csv}")
    print(f"summary_by_clip.csv: {clip_csv}")


if __name__ == "__main__":
    main()
