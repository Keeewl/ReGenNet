import argparse
import csv
import re
from collections import Counter, defaultdict
from pathlib import Path


CRITERIA = {
    "contact": ["contact_rank1", "contact_rank2", "contact_rank3"],
    "reaction": ["reaction_rank1", "reaction_rank2", "reaction_rank3"],
    "realism": ["realism_rank1", "realism_rank2", "realism_rank3"],
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Analyze ranking-style user study results with GT as reference. "
            "By default analyzes the whole table; optionally filter a specific participant_id."
        )
    )
    parser.add_argument(
        "--mapping_csv",
        default="visualize/user_study/randomized_videos_selected_gtd/mapping.csv",
        help="Path to mapping.csv produced by randomize_user_study_videos.py",
    )
    parser.add_argument(
        "--responses_csv",
        default="visualize/user_study/results/ranking_responses_gtd.csv",
        help="Path to ranking response CSV",
    )
    parser.add_argument(
        "--output_dir",
        default="visualize/user_study/results/ranking_analysis",
        help="Directory for analysis outputs",
    )
    parser.add_argument(
        "--participant_id",
        default="",
        help="Optional participant_id to analyze only a single participant",
    )
    parser.add_argument(
        "--save_results",
        action="store_true",
        help="Save decoded and summary CSV files. By default only print to terminal.",
    )
    return parser.parse_args()


def slugify(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text.strip()) or "empty"


def load_mapping(mapping_csv: Path):
    mapping = {}
    with mapping_csv.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mapping[(row["clip_id"], row["option"])] = row["method"]
    return mapping


def load_responses(responses_csv: Path, participant_id: str = ""):
    with responses_csv.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if participant_id:
        rows = [r for r in rows if r.get("participant_id", "") == participant_id]
    return rows


def decode_rows(rows, mapping):
    decoded = []
    for row in rows:
        clip_id = row["clip_id"]
        decoded_row = {
            "participant_id": row["participant_id"],
            "timestamp": row["timestamp"],
            "clip_id": clip_id,
            "dataset_key": row.get("dataset_key", ""),
            "reference_option": row.get("reference_option", ""),
            "candidate_options": row.get("candidate_options", ""),
        }
        for criterion, cols in CRITERIA.items():
            for rank_idx, col in enumerate(cols, start=1):
                option = row[col]
                decoded_row[f"{criterion}_rank{rank_idx}_option"] = option
                decoded_row[f"{criterion}_rank{rank_idx}_method"] = mapping.get((clip_id, option), "")
        decoded.append(decoded_row)
    return decoded


def summarize(rows, mapping):
    avg_rank_total = {criterion: defaultdict(int) for criterion in CRITERIA}
    avg_rank_count = {criterion: defaultdict(int) for criterion in CRITERIA}
    top1 = {criterion: Counter() for criterion in CRITERIA}
    top2 = {criterion: Counter() for criterion in CRITERIA}
    top3 = {criterion: Counter() for criterion in CRITERIA}
    pairwise = {criterion: Counter() for criterion in CRITERIA}
    per_clip = defaultdict(
        lambda: {
            criterion: {
                "rank_total": defaultdict(int),
                "rank_count": defaultdict(int),
                "top1": Counter(),
            }
            for criterion in CRITERIA
        }
    )

    for row in rows:
        clip_id = row["clip_id"]
        for criterion, cols in CRITERIA.items():
            ordered_methods = [mapping.get((clip_id, row[col]), "") for col in cols]
            for rank_idx, method in enumerate(ordered_methods, start=1):
                avg_rank_total[criterion][method] += rank_idx
                avg_rank_count[criterion][method] += 1
                per_clip[clip_id][criterion]["rank_total"][method] += rank_idx
                per_clip[clip_id][criterion]["rank_count"][method] += 1
            if ordered_methods:
                top1[criterion][ordered_methods[0]] += 1
                per_clip[clip_id][criterion]["top1"][ordered_methods[0]] += 1
            if len(ordered_methods) > 1:
                top2[criterion][ordered_methods[1]] += 1
            if len(ordered_methods) > 2:
                top3[criterion][ordered_methods[2]] += 1
            for i in range(len(ordered_methods)):
                for j in range(i + 1, len(ordered_methods)):
                    pairwise[criterion][(ordered_methods[i], ordered_methods[j])] += 1

    return {
        "avg_rank_total": avg_rank_total,
        "avg_rank_count": avg_rank_count,
        "top1": top1,
        "top2": top2,
        "top3": top3,
        "pairwise": pairwise,
        "per_clip": per_clip,
    }


def write_csv(path: Path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_summary_rows(rows, summary):
    result = []
    total_rows = len(rows)
    for criterion in CRITERIA:
        methods = sorted(summary["avg_rank_count"][criterion].keys())
        for method in methods:
            count = summary["avg_rank_count"][criterion][method]
            avg_rank = summary["avg_rank_total"][criterion][method] / count if count else 0.0
            first_count = summary["top1"][criterion][method]
            second_count = summary["top2"][criterion][method]
            third_count = summary["top3"][criterion][method]
            result.append(
                {
                    "criterion": criterion,
                    "method": method,
                    "avg_rank": f"{avg_rank:.4f}",
                    "first_count": first_count,
                    "first_ratio": f"{(first_count / total_rows):.4f}" if total_rows else "0.0000",
                    "second_count": second_count,
                    "third_count": third_count,
                }
            )
    return result


def build_pairwise_rows(summary):
    result = []
    for criterion in CRITERIA:
        for (winner, loser), wins in sorted(summary["pairwise"][criterion].items()):
            result.append(
                {
                    "criterion": criterion,
                    "winner": winner,
                    "loser": loser,
                    "wins": wins,
                }
            )
    return result


def build_clip_rows(summary):
    result = []
    for clip_id in sorted(summary["per_clip"].keys()):
        for criterion in CRITERIA:
            rank_total = summary["per_clip"][clip_id][criterion]["rank_total"]
            rank_count = summary["per_clip"][clip_id][criterion]["rank_count"]
            top1 = summary["per_clip"][clip_id][criterion]["top1"]
            for method in sorted(rank_count.keys()):
                count = rank_count[method]
                avg_rank = rank_total[method] / count if count else 0.0
                result.append(
                    {
                        "clip_id": clip_id,
                        "criterion": criterion,
                        "method": method,
                        "avg_rank": f"{avg_rank:.4f}",
                        "first_count": top1[method],
                    }
                )
    return result


def build_combined_table_rows(rows, summary):
    result = []
    total_rows = len(rows)
    all_methods = sorted(
        {
            method
            for criterion in CRITERIA
            for method in summary["avg_rank_count"][criterion].keys()
        }
    )
    for method in all_methods:
        row = {"method": method}
        for criterion in CRITERIA:
            count = summary["avg_rank_count"][criterion][method]
            avg_rank = summary["avg_rank_total"][criterion][method] / count if count else 0.0
            first_count = summary["top1"][criterion][method]
            first_ratio = first_count / total_rows if total_rows else 0.0
            second_count = summary["top2"][criterion][method]
            third_count = summary["top3"][criterion][method]
            row[f"{criterion}_avg_rank"] = f"{avg_rank:.4f}"
            row[f"{criterion}_first_count"] = first_count
            row[f"{criterion}_first_ratio"] = f"{first_ratio:.4f}"
            row[f"{criterion}_second_count"] = second_count
            row[f"{criterion}_third_count"] = third_count
        result.append(row)
    return result


def print_console_summary(rows, summary, participant_id):
    participant_label = participant_id if participant_id else "ALL"
    participants = sorted({r["participant_id"] for r in rows})
    print(f"scope: {participant_label}")
    print(f"participants: {participants}")
    print(f"response rows: {len(rows)}")
    for criterion in CRITERIA:
        print(f"\n[{criterion}]")
        methods = sorted(summary["avg_rank_count"][criterion].keys())
        for method in methods:
            count = summary["avg_rank_count"][criterion][method]
            avg_rank = summary["avg_rank_total"][criterion][method] / count if count else 0.0
            first_count = summary["top1"][criterion][method]
            ratio = first_count / len(rows) if rows else 0.0
            print(f"  {method}: avg_rank={avg_rank:.3f}, first={first_count} ({ratio:.2%})")


def main():
    args = parse_args()
    mapping_csv = Path(args.mapping_csv).expanduser().resolve()
    responses_csv = Path(args.responses_csv).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    participant_id = args.participant_id.strip()
    mapping = load_mapping(mapping_csv)
    rows = load_responses(responses_csv, participant_id=participant_id)
    if not rows:
        scope = participant_id or "ALL"
        raise RuntimeError(f"No ranking responses found for scope: {scope}")

    decoded = decode_rows(rows, mapping)
    summary = summarize(rows, mapping)

    suffix = "all" if not participant_id else f"participant_{slugify(participant_id)}"
    decoded_csv = output_dir / f"decoded_ranking_{suffix}.csv"
    summary_csv = output_dir / f"summary_ranking_{suffix}.csv"
    pairwise_csv = output_dir / f"pairwise_ranking_{suffix}.csv"
    clip_csv = output_dir / f"clip_ranking_{suffix}.csv"
    combined_csv = output_dir / f"combined_ranking_{suffix}.csv"

    decoded_fieldnames = [
        "participant_id",
        "timestamp",
        "clip_id",
        "dataset_key",
        "reference_option",
        "candidate_options",
    ]
    for criterion in CRITERIA:
        for rank_idx in range(1, 4):
            decoded_fieldnames.append(f"{criterion}_rank{rank_idx}_option")
            decoded_fieldnames.append(f"{criterion}_rank{rank_idx}_method")

    print_console_summary(rows, summary, participant_id)
    if args.save_results:
        output_dir.mkdir(parents=True, exist_ok=True)
        write_csv(decoded_csv, decoded_fieldnames, decoded)
        write_csv(
            summary_csv,
            ["criterion", "method", "avg_rank", "first_count", "first_ratio", "second_count", "third_count"],
            build_summary_rows(rows, summary),
        )
        write_csv(pairwise_csv, ["criterion", "winner", "loser", "wins"], build_pairwise_rows(summary))
        write_csv(clip_csv, ["clip_id", "criterion", "method", "avg_rank", "first_count"], build_clip_rows(summary))
        write_csv(
            combined_csv,
            [
                "method",
                "contact_avg_rank",
                "contact_first_count",
                "contact_first_ratio",
                "contact_second_count",
                "contact_third_count",
                "reaction_avg_rank",
                "reaction_first_count",
                "reaction_first_ratio",
                "reaction_second_count",
                "reaction_third_count",
                "realism_avg_rank",
                "realism_first_count",
                "realism_first_ratio",
                "realism_second_count",
                "realism_third_count",
            ],
            build_combined_table_rows(rows, summary),
        )
        print(f"\ndecoded: {decoded_csv}")
        print(f"summary: {summary_csv}")
        print(f"pairwise: {pairwise_csv}")
        print(f"per_clip: {clip_csv}")
        print(f"combined: {combined_csv}")


if __name__ == "__main__":
    main()
