import argparse
import csv
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import gradio as gr


APP_TITLE = "Motion Interaction Ranking Study"
INTRO_TEXT = (
    "## Motion Interaction Ranking Study\n\n"
    "Each page shows one reference video and three anonymized candidate videos for the same interaction.\n\n"
    "- The reference video is provided for context only and is not part of the ranking.\n"
    "- Focus on the person responding to the interaction.\n"
    "- For each criterion, rank the three candidates from best to worst.\n"
    "- Please replace the default participant ID before you start."
)
CONTACT_PROMPT = (
    "### Contact Quality\n"
    "Rank the three candidate videos from best to worst in terms of physical contact quality. "
    "Focus on contact alignment, floating hands, penetration, and obvious spatial mismatch."
)
REACTION_PROMPT = (
    "### Reaction Appropriateness\n"
    "Rank the three candidate videos from best to worst in terms of response quality. "
    "Focus on timing, direction, and whether the response matches the interaction."
)
REALISM_PROMPT = (
    "### Overall Realism\n"
    "Rank the three candidate videos from best to worst in overall realism. "
    "Consider motion smoothness, pose plausibility, interaction consistency, and visual credibility."
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Gradio ranking user study with GT as reference and three anonymized candidate videos."
    )
    parser.add_argument(
        "--study_root",
        default="visualize/user_study/randomized_videos_selected",
        help="Root directory containing videos/, mapping.csv, questionnaire_template.csv",
    )
    parser.add_argument(
        "--output_csv",
        default="visualize/user_study/results/ranking_responses.csv",
        help="CSV file to append ranking responses to",
    )
    parser.add_argument("--server_name", default="0.0.0.0")
    parser.add_argument("--server_port", type=int, default=7861)
    parser.add_argument("--share", action="store_true")
    return parser.parse_args()


def load_mapping_by_clip(mapping_csv: Path):
    by_clip = {}
    with mapping_csv.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            by_clip.setdefault(row["clip_id"], []).append(row)
    return by_clip


def load_questionnaire(questionnaire_csv: Path):
    rows = {}
    with questionnaire_csv.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows[row["clip_id"]] = row
    return rows


def build_items(study_root: Path):
    mapping_csv = study_root / "mapping.csv"
    questionnaire_csv = study_root / "questionnaire_template.csv"
    if not mapping_csv.exists():
        raise FileNotFoundError(f"Missing mapping.csv: {mapping_csv}")
    if not questionnaire_csv.exists():
        raise FileNotFoundError(f"Missing questionnaire_template.csv: {questionnaire_csv}")

    mapping_by_clip = load_mapping_by_clip(mapping_csv)
    questionnaire = load_questionnaire(questionnaire_csv)
    items = []

    for clip_id in sorted(mapping_by_clip.keys()):
        mapping_rows = mapping_by_clip[clip_id]
        qrow = questionnaire.get(clip_id, {})
        option_to_video = {}
        option_to_method = {}
        gt_option = None

        for row in mapping_rows:
            option = row["option"]
            method = row["method"]
            option_to_method[option] = method
            option_to_video[option] = str((study_root / "videos" / clip_id / f"{option}.mp4").resolve())
            if method == "gt":
                gt_option = option

        if gt_option is None:
            raise RuntimeError(f"No GT option found for {clip_id} in {mapping_csv}")

        candidate_options = [opt for opt in ["A", "B", "C", "D"] if opt != gt_option]
        for opt in [gt_option] + candidate_options:
            if not Path(option_to_video[opt]).exists():
                raise FileNotFoundError(f"Missing video for {clip_id} option {opt}: {option_to_video[opt]}")

        items.append(
            {
                "clip_id": clip_id,
                "dataset_key": qrow.get("dataset_key", ""),
                "reference_option": gt_option,
                "reference_video": option_to_video[gt_option],
                "candidate_options": candidate_options,
                "candidate_videos": {opt: option_to_video[opt] for opt in candidate_options},
            }
        )

    if not items:
        raise RuntimeError(f"No clips found in {study_root}")
    return items


def append_results(output_csv: Path, participant_id: str, items, responses):
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    write_header = not output_csv.exists()
    timestamp = datetime.now().isoformat(timespec="seconds")
    fieldnames = [
        "participant_id",
        "timestamp",
        "clip_id",
        "dataset_key",
        "reference_option",
        "candidate_options",
        "contact_rank1",
        "contact_rank2",
        "contact_rank3",
        "reaction_rank1",
        "reaction_rank2",
        "reaction_rank3",
        "realism_rank1",
        "realism_rank2",
        "realism_rank3",
    ]

    with output_csv.open("a", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for item in items:
            clip_id = item["clip_id"]
            row = responses[clip_id]
            writer.writerow(
                {
                    "participant_id": participant_id,
                    "timestamp": timestamp,
                    "clip_id": clip_id,
                    "dataset_key": item["dataset_key"],
                    "reference_option": item["reference_option"],
                    "candidate_options": ",".join(item["candidate_options"]),
                    "contact_rank1": row["contact"][0],
                    "contact_rank2": row["contact"][1],
                    "contact_rank3": row["contact"][2],
                    "reaction_rank1": row["reaction"][0],
                    "reaction_rank2": row["reaction"][1],
                    "reaction_rank3": row["reaction"][2],
                    "realism_rank1": row["realism"][0],
                    "realism_rank2": row["realism"][1],
                    "realism_rank3": row["realism"][2],
                }
            )


def build_page_payload(items, index, responses):
    item = items[index]
    existing = responses.get(item["clip_id"], {})
    is_last = index == len(items) - 1
    candidates = item["candidate_options"]
    return {
        "progress": f"Clip {index + 1} of {len(items)}",
        "clip_label": f"Interaction Clip {index + 1}",
        "reference_label": "Reference Video",
        "reference_video": item["reference_video"],
        "cand_labels": [f"Candidate {opt}" for opt in candidates],
        "cand_videos": [item["candidate_videos"][opt] for opt in candidates],
        "choices": candidates,
        "contact": existing.get("contact", [None, None, None]),
        "reaction": existing.get("reaction", [None, None, None]),
        "realism": existing.get("realism", [None, None, None]),
        "next_interactive": not is_last,
        "submit_interactive": is_last,
    }


def is_valid_ranking(values, choices):
    return (
        len(values) == len(choices)
        and all(v in choices for v in values)
        and len(set(values)) == len(choices)
    )


def make_app(items, output_csv: Path):
    initial_participant_id = str(uuid4())
    initial_responses = {}
    initial = build_page_payload(items, 0, initial_responses)

    def validate_participant_id(participant_id):
        pid = (participant_id or "").strip()
        if not pid:
            return False, "Please enter a participant ID."
        if pid == initial_participant_id:
            return False, "Please replace the default participant ID before continuing."
        return True, ""

    def save_current(current_index, responses, contact1, contact2, contact3, reaction1, reaction2, reaction3, realism1, realism2, realism3):
        item = items[current_index]
        choices = item["candidate_options"]
        contact = [contact1, contact2, contact3]
        reaction = [reaction1, reaction2, reaction3]
        realism = [realism1, realism2, realism3]
        if not is_valid_ranking(contact, choices):
            return None, "Please provide a valid non-repeating ranking for Contact Quality."
        if not is_valid_ranking(reaction, choices):
            return None, "Please provide a valid non-repeating ranking for Reaction Appropriateness."
        if not is_valid_ranking(realism, choices):
            return None, "Please provide a valid non-repeating ranking for Overall Realism."

        responses = deepcopy(responses or {})
        responses[item["clip_id"]] = {
            "contact": contact,
            "reaction": reaction,
            "realism": realism,
        }
        return responses, ""

    def go_next(participant_id, current_index, responses, contact1, contact2, contact3, reaction1, reaction2, reaction3, realism1, realism2, realism3):
        ok, msg = validate_participant_id(participant_id)
        if not ok:
            return (
                current_index,
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                msg,
            )
        responses, msg = save_current(
            current_index, responses,
            contact1, contact2, contact3,
            reaction1, reaction2, reaction3,
            realism1, realism2, realism3,
        )
        if responses is None:
            return (
                current_index,
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                msg,
            )

        new_index = current_index + 1
        payload = build_page_payload(items, new_index, responses)
        return (
            new_index,
            responses,
            payload["progress"],
            payload["clip_label"],
            payload["reference_label"],
            gr.update(value=payload["reference_video"]),
            gr.update(value=payload["cand_videos"][0], label=payload["cand_labels"][0]),
            gr.update(value=payload["cand_videos"][1], label=payload["cand_labels"][1]),
            gr.update(value=payload["cand_videos"][2], label=payload["cand_labels"][2]),
            gr.update(choices=payload["choices"], value=payload["contact"][0]),
            gr.update(choices=payload["choices"], value=payload["contact"][1]),
            gr.update(choices=payload["choices"], value=payload["contact"][2]),
            gr.update(choices=payload["choices"], value=payload["reaction"][0]),
            gr.update(choices=payload["choices"], value=payload["reaction"][1]),
            gr.update(choices=payload["choices"], value=payload["reaction"][2]),
            gr.update(choices=payload["choices"], value=payload["realism"][0]),
            gr.update(choices=payload["choices"], value=payload["realism"][1]),
            gr.update(choices=payload["choices"], value=payload["realism"][2]),
            gr.update(interactive=payload["next_interactive"]),
            gr.update(interactive=payload["submit_interactive"]),
            "",
        )

    def submit(participant_id, current_index, responses, contact1, contact2, contact3, reaction1, reaction2, reaction3, realism1, realism2, realism3):
        ok, msg = validate_participant_id(participant_id)
        if not ok:
            return msg
        responses, msg = save_current(
            current_index, responses,
            contact1, contact2, contact3,
            reaction1, reaction2, reaction3,
            realism1, realism2, realism3,
        )
        if responses is None:
            return msg
        if len(responses) != len(items):
            return "Submission failed because some clips are still missing rankings."
        participant_id = participant_id.strip()
        append_results(output_csv, participant_id, items, responses)
        return (
            f"Submission saved successfully.\n\n"
            f"Participant ID: {participant_id}\n"
            f"Responses written to: {output_csv.resolve()}"
        )

    with gr.Blocks(title=APP_TITLE) as demo:
        idx_state = gr.State(0)
        responses_state = gr.State(initial_responses)

        gr.Markdown(INTRO_TEXT)

        participant_id = gr.Textbox(label="Participant ID", value=initial_participant_id, interactive=True)
        progress = gr.Markdown(f"**{initial['progress']}**")
        clip_label = gr.Markdown(f"**{initial['clip_label']}**")

        ref_label = gr.Markdown(f"**{initial['reference_label']}**")
        ref_video = gr.Video(value=initial["reference_video"], label="Reference Video", interactive=False)

        with gr.Row():
            with gr.Column():
                cand1 = gr.Video(value=initial["cand_videos"][0], label=initial["cand_labels"][0], interactive=False)
            with gr.Column():
                cand2 = gr.Video(value=initial["cand_videos"][1], label=initial["cand_labels"][1], interactive=False)
            with gr.Column():
                cand3 = gr.Video(value=initial["cand_videos"][2], label=initial["cand_labels"][2], interactive=False)

        choices = initial["choices"]

        gr.Markdown(CONTACT_PROMPT)
        with gr.Row():
            contact1 = gr.Dropdown(choices=choices, value=initial["contact"][0], label="1st")
            contact2 = gr.Dropdown(choices=choices, value=initial["contact"][1], label="2nd")
            contact3 = gr.Dropdown(choices=choices, value=initial["contact"][2], label="3rd")

        gr.Markdown(REACTION_PROMPT)
        with gr.Row():
            reaction1 = gr.Dropdown(choices=choices, value=initial["reaction"][0], label="1st")
            reaction2 = gr.Dropdown(choices=choices, value=initial["reaction"][1], label="2nd")
            reaction3 = gr.Dropdown(choices=choices, value=initial["reaction"][2], label="3rd")

        gr.Markdown(REALISM_PROMPT)
        with gr.Row():
            realism1 = gr.Dropdown(choices=choices, value=initial["realism"][0], label="1st")
            realism2 = gr.Dropdown(choices=choices, value=initial["realism"][1], label="2nd")
            realism3 = gr.Dropdown(choices=choices, value=initial["realism"][2], label="3rd")

        status = gr.Markdown("")
        with gr.Row():
            next_btn = gr.Button("Next", interactive=initial["next_interactive"], variant="primary")
            submit_btn = gr.Button("Submit", interactive=initial["submit_interactive"], variant="primary")

        next_btn.click(
            fn=go_next,
            inputs=[participant_id, idx_state, responses_state, contact1, contact2, contact3, reaction1, reaction2, reaction3, realism1, realism2, realism3],
            outputs=[
                idx_state, responses_state, progress, clip_label, ref_label, ref_video,
                cand1, cand2, cand3,
                contact1, contact2, contact3,
                reaction1, reaction2, reaction3,
                realism1, realism2, realism3,
                next_btn, submit_btn,
                status,
            ],
        )

        submit_btn.click(
            fn=submit,
            inputs=[participant_id, idx_state, responses_state, contact1, contact2, contact3, reaction1, reaction2, reaction3, realism1, realism2, realism3],
            outputs=[status],
        )

    return demo


def main():
    args = parse_args()
    study_root = Path(args.study_root).expanduser().resolve()
    output_csv = Path(args.output_csv).expanduser().resolve()
    items = build_items(study_root)
    demo = make_app(items, output_csv)
    demo.launch(server_name=args.server_name, server_port=args.server_port, share=args.share)


if __name__ == "__main__":
    main()
