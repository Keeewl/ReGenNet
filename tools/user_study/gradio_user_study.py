import argparse
import csv
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import gradio as gr


DEFAULT_CONTACT_QUESTION = (
    "Contact Quality:\n"
    "Which video shows more natural and accurate physical contact? Please focus on hand contact "
    "alignment, floating hands, penetration, or obvious misalignment."
)
DEFAULT_REACTION_QUESTION = (
    "Reaction Appropriateness:\n"
    "Which video shows a more appropriate reaction to the other person's motion? Please focus on "
    "timing, direction, and whether the response matches the interaction semantics."
)
DEFAULT_REALISM_QUESTION = (
    "Overall Realism:\n"
    "Which video looks more realistic, natural, and coherent overall? Please consider motion "
    "smoothness, body pose plausibility, interaction consistency, and visual credibility."
)


def parse_args():
    parser = argparse.ArgumentParser(description="Gradio user study interface for randomized A/B/C/D videos.")
    parser.add_argument(
        "--study_root",
        default="visualize/user_study/randomized_videos",
        help="Root directory containing videos/, mapping.csv, questionnaire_template.csv",
    )
    parser.add_argument(
        "--output_csv",
        default="visualize/user_study/results/responses.csv",
        help="CSV file to append study responses to",
    )
    parser.add_argument("--server_name", default="0.0.0.0")
    parser.add_argument("--server_port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    return parser.parse_args()


def _abs_video(path_str: str, study_root: Path) -> str:
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = (study_root / path).resolve()
    else:
        path = path.resolve()
    return str(path)


def load_study_items(study_root: Path):
    questionnaire_csv = study_root / "questionnaire_template.csv"
    videos_root = study_root / "videos"
    items = []

    if questionnaire_csv.exists():
        with questionnaire_csv.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                clip_id = row["clip_id"]
                clip_video_dir = videos_root / clip_id
                videos = {}
                for option in ["A", "B", "C", "D"]:
                    candidate = row.get(f"video_{option}", "").strip()
                    if candidate:
                        candidate_abs = Path(_abs_video(candidate, study_root))
                    else:
                        candidate_abs = (clip_video_dir / f"{option}.mp4").resolve()
                    if not candidate_abs.exists():
                        candidate_abs = (clip_video_dir / f"{option}.mp4").resolve()
                    if not candidate_abs.exists():
                        raise FileNotFoundError(f"Missing video for {clip_id} option {option}: {candidate_abs}")
                    videos[option] = str(candidate_abs)

                items.append(
                    {
                        "clip_id": clip_id,
                        "videos": videos,
                        "question_contact": row.get("question_contact") or DEFAULT_CONTACT_QUESTION,
                        "question_reaction": row.get("question_reaction") or DEFAULT_REACTION_QUESTION,
                        "question_realism": row.get("question_realism") or DEFAULT_REALISM_QUESTION,
                    }
                )

    else:
        if not videos_root.exists():
            raise FileNotFoundError(f"Study videos directory not found: {videos_root}")
        for clip_dir in sorted(p for p in videos_root.iterdir() if p.is_dir()):
            videos = {}
            for option in ["A", "B", "C", "D"]:
                path = (clip_dir / f"{option}.mp4").resolve()
                if not path.exists():
                    raise FileNotFoundError(f"Missing video for {clip_dir.name} option {option}: {path}")
                videos[option] = str(path)
            items.append(
                {
                    "clip_id": clip_dir.name,
                    "videos": videos,
                    "question_contact": DEFAULT_CONTACT_QUESTION,
                    "question_reaction": DEFAULT_REACTION_QUESTION,
                    "question_realism": DEFAULT_REALISM_QUESTION,
                }
            )

    if not items:
        raise RuntimeError(f"No study items found under {study_root}")
    return items


def append_results_csv(output_csv: Path, participant_id: str, items, responses):
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    write_header = not output_csv.exists()
    timestamp = datetime.now().isoformat(timespec="seconds")

    with output_csv.open("a", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "participant_id",
                "timestamp",
                "clip_id",
                "contact_choice",
                "reaction_choice",
                "realism_choice",
            ],
        )
        if write_header:
            writer.writeheader()
        for item in items:
            clip_id = item["clip_id"]
            choice_row = responses[clip_id]
            writer.writerow(
                {
                    "participant_id": participant_id,
                    "timestamp": timestamp,
                    "clip_id": clip_id,
                    "contact_choice": choice_row["contact_choice"],
                    "reaction_choice": choice_row["reaction_choice"],
                    "realism_choice": choice_row["realism_choice"],
                }
            )


def build_page_payload(items, index, responses):
    item = items[index]
    existing = responses.get(item["clip_id"], {})
    is_last = index == len(items) - 1
    return {
        "progress_text": f"Clip {index + 1} / {len(items)}",
        "clip_label": f"{item['clip_id']}",
        "video_A": item["videos"]["A"],
        "video_B": item["videos"]["B"],
        "video_C": item["videos"]["C"],
        "video_D": item["videos"]["D"],
        "question_contact": item["question_contact"],
        "question_reaction": item["question_reaction"],
        "question_realism": item["question_realism"],
        "contact_value": existing.get("contact_choice"),
        "reaction_value": existing.get("reaction_choice"),
        "realism_value": existing.get("realism_choice"),
        "next_visible": not is_last,
        "submit_visible": is_last,
    }


def make_app(items, output_csv: Path):
    initial_participant_id = str(uuid4())
    initial_responses = {}
    initial_page = build_page_payload(items, 0, initial_responses)

    def validate_choices(contact_choice, reaction_choice, realism_choice):
        return all([contact_choice, reaction_choice, realism_choice])

    def next_clip(participant_id, current_index, responses, contact_choice, reaction_choice, realism_choice):
        if not validate_choices(contact_choice, reaction_choice, realism_choice):
            return (
                current_index,
                responses,
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
                "Please complete all three questions before continuing.",
            )

        responses = deepcopy(responses or {})
        clip_id = items[current_index]["clip_id"]
        responses[clip_id] = {
            "contact_choice": contact_choice,
            "reaction_choice": reaction_choice,
            "realism_choice": realism_choice,
        }
        new_index = current_index + 1
        payload = build_page_payload(items, new_index, responses)
        return (
            new_index,
            responses,
            payload["progress_text"],
            payload["clip_label"],
            payload["video_A"],
            payload["video_B"],
            payload["video_C"],
            payload["video_D"],
            payload["question_contact"],
            payload["question_reaction"],
            payload["question_realism"],
            payload["contact_value"],
            payload["reaction_value"],
            payload["realism_value"],
            "",
        )

    def submit_study(participant_id, current_index, responses, contact_choice, reaction_choice, realism_choice):
        if not validate_choices(contact_choice, reaction_choice, realism_choice):
            return "Please complete all three questions before submitting."

        participant_id = participant_id.strip() or str(uuid4())
        responses = deepcopy(responses or {})
        clip_id = items[current_index]["clip_id"]
        responses[clip_id] = {
            "contact_choice": contact_choice,
            "reaction_choice": reaction_choice,
            "realism_choice": realism_choice,
        }

        if len(responses) != len(items):
            return "Submission failed because some clips are still missing responses."

        append_results_csv(output_csv, participant_id, items, responses)
        return (
            f"Submission saved successfully.\n\n"
            f"Participant ID: {participant_id}\n"
            f"Responses written to: {output_csv.resolve()}"
        )

    with gr.Blocks(title="HiReact User Study") as demo:
        current_index_state = gr.State(0)
        responses_state = gr.State(initial_responses)

        gr.Markdown(
            "## User Study\n\n"
            "This study contains 10 interaction clips. Each clip shows four anonymized videos "
            "A/B/C/D of the same interaction. Please choose the best video for each criterion."
        )

        participant_id_box = gr.Textbox(
            label="Participant ID",
            value=initial_participant_id,
            interactive=True,
        )

        progress_text = gr.Markdown(f"**{initial_page['progress_text']}**")
        clip_label = gr.Markdown(f"**{initial_page['clip_label']}**")

        with gr.Row():
            with gr.Column():
                video_a = gr.Video(value=initial_page["video_A"], label="Video A", interactive=False)
                video_c = gr.Video(value=initial_page["video_C"], label="Video C", interactive=False)
            with gr.Column():
                video_b = gr.Video(value=initial_page["video_B"], label="Video B", interactive=False)
                video_d = gr.Video(value=initial_page["video_D"], label="Video D", interactive=False)

        question_contact_md = gr.Markdown(initial_page["question_contact"])
        contact_radio = gr.Radio(choices=["A", "B", "C", "D"], value=initial_page["contact_value"], label="Contact Quality")

        question_reaction_md = gr.Markdown(initial_page["question_reaction"])
        reaction_radio = gr.Radio(
            choices=["A", "B", "C", "D"],
            value=initial_page["reaction_value"],
            label="Reaction Appropriateness",
        )

        question_realism_md = gr.Markdown(initial_page["question_realism"])
        realism_radio = gr.Radio(
            choices=["A", "B", "C", "D"],
            value=initial_page["realism_value"],
            label="Overall Realism",
        )

        status_box = gr.Markdown("")

        with gr.Row():
            next_btn = gr.Button("Next", visible=initial_page["next_visible"], variant="primary")
            submit_btn = gr.Button("Submit", visible=initial_page["submit_visible"], variant="primary")

        next_btn.click(
            fn=next_clip,
            inputs=[
                participant_id_box,
                current_index_state,
                responses_state,
                contact_radio,
                reaction_radio,
                realism_radio,
            ],
            outputs=[
                current_index_state,
                responses_state,
                progress_text,
                clip_label,
                video_a,
                video_b,
                video_c,
                video_d,
                question_contact_md,
                question_reaction_md,
                question_realism_md,
                contact_radio,
                reaction_radio,
                realism_radio,
                status_box,
            ],
        ).then(
            fn=lambda idx, resp: (
                gr.update(visible=idx < len(items) - 1),
                gr.update(visible=idx == len(items) - 1),
            ),
            inputs=[current_index_state, responses_state],
            outputs=[next_btn, submit_btn],
        )

        submit_btn.click(
            fn=submit_study,
            inputs=[
                participant_id_box,
                current_index_state,
                responses_state,
                contact_radio,
                reaction_radio,
                realism_radio,
            ],
            outputs=[status_box],
        )

    return demo


def main():
    args = parse_args()
    study_root = Path(args.study_root).expanduser().resolve()
    output_csv = Path(args.output_csv).expanduser().resolve()
    items = load_study_items(study_root)
    demo = make_app(items, output_csv)
    demo.launch(server_name=args.server_name, server_port=args.server_port, share=args.share)


if __name__ == "__main__":
    main()
