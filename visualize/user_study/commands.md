# User Study Commands

This note summarizes the main commands and parameters for preparing videos, randomizing them, launching the web UI, and analyzing results.

## 1. Export viewer-ready clips

Script:
- [export_user_study_pack.sh](/Users/keweiou/Desktop/Project/ReGenNet/visualize/user_study/export_user_study_pack.sh)

Purpose:
- export `gt / baseline / stage1 / stage2` clip folders under `outputs/user_study_pack` or similar.

Typical usage:

```bash
cp visualize/user_study/dataset_keys.txt.example visualize/user_study/dataset_keys.txt

export CUDA_VISIBLE_DEVICES=0
export BASELINE_MODEL_PATH=save/cmdm/interx_smplx_online_exp1/model000140000.pt
export STAGE1_MODEL_PATH=save/cnet_v5_256/interx_smplx_online_exp1/model000209455.pt
export STAGE2_CHECKPOINT=refine_v2/save/train/refiner_v2_exp8_interaction_v1_10k/model_best.pt
export RESTORATION_META_PATH=dataset/interx/cache/interx_restoration_meta.npz

bash visualize/user_study/export_user_study_pack.sh
```

Important environment variables:
- `BASELINE_MODEL_PATH`
- `STAGE1_MODEL_PATH`
- `STAGE2_CHECKPOINT`
- `RESTORATION_META_PATH`
- optional:
  - `RAW_MOTIONS_ROOT`
  - `GT_DATA_DIR`
  - `USER_STUDY_KEYS_FILE`
  - `USER_STUDY_OUT_ROOT`

Notes:
- Stage1 export uses `DDIM-5`.
- Stage2 export currently uses the `refined` result as the final user-study candidate.

## 2. Render mp4 from exported clip folders

Script:
- [render_pack_to_mp4.py](/Users/keweiou/Desktop/Project/ReGenNet/visualize/user_study/render_pack_to_mp4.py)

Purpose:
- convert exported `P1.npz / P2.npz` clips into mp4.

Typical usage:

```bash
python visualize/user_study/render_pack_to_mp4.py \
  --root_dir outputs/user_study_pack
```

Optional arguments:
- `--root_dir`
- `--output_dir`
- `--overwrite`
- `--device`

Notes:
- the current offline renderer is based on the legacy renderer path
- for macOS local use, compatibility may vary; server-side rendering is usually more stable

## 3. Randomize rendered mp4 into A/B/C/D

Script:
- [randomize_user_study_videos.py](/Users/keweiou/Desktop/Project/ReGenNet/visualize/user_study/randomize_user_study_videos.py)

Purpose:
- anonymize method videos into `A/B/C/D`
- export:
  - `mapping.csv`
  - `questionnaire_template.csv`

### 3.1 Flat video layout

Supported input format:

```text
visualize/export/videos/
  G002T000A001R005-baseline_0.mp4
  G002T000A001R005-stage1_0.mp4
  G002T000A001R005-stage2_0.mp4
  G002T000A001R005-gt_0.mp4
  ...
```

Typical usage:

```bash
python visualize/user_study/randomize_user_study_videos.py \
  --input_root visualize/export/videos \
  --output_root visualize/user_study/randomized_videos \
  --copy_mode copy \
  --layout flat
```

### 3.2 Selected clip subset

If you want to only randomize a selected key list:

```bash
python visualize/user_study/randomize_user_study_videos.py \
  --input_root visualize/export/videos \
  --output_root visualize/user_study/randomized_videos_selected \
  --copy_mode copy \
  --layout flat \
  --selected_keys_file visualize/user_study/selected_dataset_key.md
```

Important arguments:
- `--input_root`
- `--output_root`
- `--seed`
- `--methods`
- `--copy_mode`
- `--video_ext`
- `--layout`
  - `auto`
  - `user_study_pack`
  - `flat`
- `--selected_keys_file`

Output structure:

```text
output_root/
  videos/
    clip_001/
      A.mp4
      B.mp4
      C.mp4
      D.mp4
  mapping.csv
  questionnaire_template.csv
```

## 4. Launch the 4-way single-choice study

Script:
- [tools/user_study/gradio_user_study.py](/Users/keweiou/Desktop/Project/ReGenNet/tools/user_study/gradio_user_study.py)

Purpose:
- show `A/B/C/D`
- choose one best video for each criterion

Typical usage:

```bash
python tools/user_study/gradio_user_study.py \
  --study_root visualize/user_study/randomized_videos \
  --output_csv visualize/user_study/results/responses.csv \
  --server_name 0.0.0.0 \
  --server_port 7860
```

Optional:

```bash
python tools/user_study/gradio_user_study.py --share
```

Important arguments:
- `--study_root`
- `--output_csv`
- `--server_name`
- `--server_port`
- `--share`

## 5. Analyze the 4-way single-choice results

Script:
- [tools/user_study/analyze_user_study_results.py](/Users/keweiou/Desktop/Project/ReGenNet/tools/user_study/analyze_user_study_results.py)

Purpose:
- decode `A/B/C/D` back to true methods
- summarize vote counts and ratios

Typical usage:

```bash
python tools/user_study/analyze_user_study_results.py \
  --mapping_csv visualize/user_study/randomized_videos/mapping.csv \
  --responses_csv visualize/user_study/results/responses.csv \
  --output_dir visualize/user_study/results/analysis
```

Outputs:
- `decoded_responses.csv`
- `summary_by_criterion.csv`
- `summary_by_clip.csv`

## 6. Launch the GT-reference ranking study

Script:
- [tools/user_study/gradio_user_study_ranking.py](/Users/keweiou/Desktop/Project/ReGenNet/tools/user_study/gradio_user_study_ranking.py)

Purpose:
- display one `GT` reference video
- rank the remaining three anonymized candidate videos from best to worst

Typical usage:

```bash
python tools/user_study/gradio_user_study_ranking.py \
  --study_root visualize/user_study/randomized_videos_selected \
  --output_csv visualize/user_study/results/ranking_responses.csv \
  --server_name 0.0.0.0 \
  --server_port 7861
```

Optional:

```bash
python tools/user_study/gradio_user_study_ranking.py --share
```

Important arguments:
- `--study_root`
- `--output_csv`
- `--server_name`
- `--server_port`
- `--share`

Notes:
- `GT` is used only as a reference
- the participant ranks the other three candidate videos
- the page explicitly explains:
  - blue = actor
  - orange = generated reactor
  - focus on differences in the orange person

## 7. Manual clip inspection

File:
- [user_study.md](/Users/keweiou/Desktop/Project/ReGenNet/visualize/user_study/user_study.md)

Purpose:
- record manual `data_viewer.py` commands for:
  - `pack01`
  - `pack02`

Each command already includes:
- `--title`
- `--playback_fps 30`

## 8. Current main file roles

- `README.md`: overall organization of the two user-study settings
- `commands.md`: practical command reference
- `user_study.md`: manual viewer commands per clip
- `selected_dataset_key.md`: selected clip list for the GT-reference ranking study
