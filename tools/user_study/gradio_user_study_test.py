import argparse
from pathlib import Path
from uuid import uuid4

import gradio as gr


APP_TITLE = "Motion Interaction Ranking Study"
INTRO_TEXT = (
    "## Motion Interaction Ranking Study\n\n"
    "This test page keeps the original ranking-study layout while showing only one clip.\n\n"
    "- The reference video is provided for context only and is not part of the ranking.\n"
    "- Focus on the person responding to the interaction.\n"
    "- For each criterion, rank the three candidates from best to worst.\n"
    "- This page is intended for preview and screenshot capture."
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
    parser = argparse.ArgumentParser(description="Single-clip test page with the full ranking-study layout.")
    parser.add_argument(
        "--reference_video",
        default="visualize/user_study/randomized_videos_selected_gtd/videos/clip_001/D.mp4",
        help="Reference video shown above the three candidates",
    )
    parser.add_argument(
        "--candidate_a",
        default="visualize/export/show_test/G002T000A001R005-baseline_1.mp4",
        help="Candidate A video path",
    )
    parser.add_argument(
        "--candidate_b",
        default="visualize/export/show_test/G002T000A001R005-stage1_1.mp4",
        help="Candidate B video path",
    )
    parser.add_argument(
        "--candidate_c",
        default="visualize/export/show_test/G002T000A001R005-stage2_1.mp4",
        help="Candidate C video path",
    )
    parser.add_argument("--server_name", default="127.0.0.1")
    parser.add_argument("--server_port", type=int, default=7862)
    parser.add_argument("--share", action="store_true")
    return parser.parse_args()


def ensure_file(path: Path, label: str) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")
    return str(path.resolve())


def is_valid_ranking(values, choices):
    return (
        len(values) == len(choices)
        and all(v in choices for v in values)
        and len(set(values)) == len(choices)
    )


def make_app(reference_video: str, candidate_a: str, candidate_b: str, candidate_c: str):
    initial_participant_id = str(uuid4())
    choices = ["A", "B", "C"]

    def validate_participant_id(participant_id):
        pid = (participant_id or "").strip()
        if not pid:
            return False, "Please enter a participant ID."
        if pid == initial_participant_id:
            return False, "Please replace the default participant ID before continuing."
        return True, ""

    def validate_rankings(contact1, contact2, contact3, reaction1, reaction2, reaction3, realism1, realism2, realism3):
        contact = [contact1, contact2, contact3]
        reaction = [reaction1, reaction2, reaction3]
        realism = [realism1, realism2, realism3]
        if not is_valid_ranking(contact, choices):
            return "Please provide a valid non-repeating ranking for Contact Quality."
        if not is_valid_ranking(reaction, choices):
            return "Please provide a valid non-repeating ranking for Reaction Appropriateness."
        if not is_valid_ranking(realism, choices):
            return "Please provide a valid non-repeating ranking for Overall Realism."
        return ""

    def submit(participant_id, contact1, contact2, contact3, reaction1, reaction2, reaction3, realism1, realism2, realism3):
        ok, msg = validate_participant_id(participant_id)
        if not ok:
            return msg
        msg = validate_rankings(contact1, contact2, contact3, reaction1, reaction2, reaction3, realism1, realism2, realism3)
        if msg:
            return msg
        return "Inputs look valid. This test page does not write any results."

    with gr.Blocks(title=APP_TITLE) as demo:
        gr.Markdown(INTRO_TEXT)

        participant_id = gr.Textbox(label="Participant ID", value=initial_participant_id, interactive=True)
        progress = gr.Markdown("**Clip 1 of 1**")
        clip_label = gr.Markdown("**Interaction Clip 1**")

        ref_label = gr.Markdown("**Reference Video**")
        ref_video = gr.Video(value=reference_video, label="Reference Video", interactive=False)

        with gr.Row():
            with gr.Column():
                cand1 = gr.Video(value=candidate_a, label="Candidate A", interactive=False)
            with gr.Column():
                cand2 = gr.Video(value=candidate_b, label="Candidate B", interactive=False)
            with gr.Column():
                cand3 = gr.Video(value=candidate_c, label="Candidate C", interactive=False)

        gr.Markdown(CONTACT_PROMPT)
        with gr.Row():
            contact1 = gr.Dropdown(choices=choices, value=None, label="1st")
            contact2 = gr.Dropdown(choices=choices, value=None, label="2nd")
            contact3 = gr.Dropdown(choices=choices, value=None, label="3rd")

        gr.Markdown(REACTION_PROMPT)
        with gr.Row():
            reaction1 = gr.Dropdown(choices=choices, value=None, label="1st")
            reaction2 = gr.Dropdown(choices=choices, value=None, label="2nd")
            reaction3 = gr.Dropdown(choices=choices, value=None, label="3rd")

        gr.Markdown(REALISM_PROMPT)
        with gr.Row():
            realism1 = gr.Dropdown(choices=choices, value=None, label="1st")
            realism2 = gr.Dropdown(choices=choices, value=None, label="2nd")
            realism3 = gr.Dropdown(choices=choices, value=None, label="3rd")

        status = gr.Markdown("")
        with gr.Row():
            next_btn = gr.Button("Next", interactive=False, variant="secondary")
            submit_btn = gr.Button("Submit", variant="primary")

        submit_btn.click(
            fn=submit,
            inputs=[participant_id, contact1, contact2, contact3, reaction1, reaction2, reaction3, realism1, realism2, realism3],
            outputs=[status],
        )

    return demo


def main():
    args = parse_args()
    reference_video = ensure_file(Path(args.reference_video).expanduser().resolve(), "reference video")
    candidate_a = ensure_file(Path(args.candidate_a).expanduser().resolve(), "candidate A video")
    candidate_b = ensure_file(Path(args.candidate_b).expanduser().resolve(), "candidate B video")
    candidate_c = ensure_file(Path(args.candidate_c).expanduser().resolve(), "candidate C video")
    demo = make_app(reference_video, candidate_a, candidate_b, candidate_c)
    demo.launch(server_name=args.server_name, server_port=args.server_port, share=args.share)


if __name__ == "__main__":
    main()
