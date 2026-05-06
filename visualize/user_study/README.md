# User Study Overview

This directory contains the assets and scripts used to build and run the user-study pipeline for HiReact.

## Two study settings

### 1. Four-video single-choice study
This is the first implementation.

- Inputs per clip:
  - `baseline`
  - `hireact_star` (`stage1`)
  - `hireact` (`stage2/refined`)
  - `gt`
- Randomization:
  - each clip is anonymized into `A/B/C/D`
- Frontend:
  - [tools/user_study/gradio_user_study.py](/Users/keweiou/Desktop/Project/ReGenNet/tools/user_study/gradio_user_study.py)
- Response format:
  - for each clip, choose the best video for:
    - Contact Quality
    - Reaction Appropriateness
    - Overall Realism

Artifacts:
- randomized pack:
  - [randomized_videos](/Users/keweiou/Desktop/Project/ReGenNet/visualize/user_study/randomized_videos)
- mapping:
  - [randomized_videos/mapping.csv](/Users/keweiou/Desktop/Project/ReGenNet/visualize/user_study/randomized_videos/mapping.csv)
- questionnaire template:
  - [randomized_videos/questionnaire_template.csv](/Users/keweiou/Desktop/Project/ReGenNet/visualize/user_study/randomized_videos/questionnaire_template.csv)
- result decoder:
  - [tools/user_study/analyze_user_study_results.py](/Users/keweiou/Desktop/Project/ReGenNet/tools/user_study/analyze_user_study_results.py)

### 2. GT-reference ranking study
This is the second implementation.

- Inputs per clip:
  - one `GT` reference video
  - three candidate videos:
    - `baseline`
    - `hireact_star`
    - `hireact`
- Randomization:
  - `GT` is identified from the anonymized `A/B/C/D` pack using `mapping.csv`
  - the remaining three anonymous videos are ranked
- Frontend:
  - [tools/user_study/gradio_user_study_ranking.py](/Users/keweiou/Desktop/Project/ReGenNet/tools/user_study/gradio_user_study_ranking.py)
- Response format:
  - for each clip, rank the three candidate videos from best to worst for:
    - Contact Quality
    - Reaction Appropriateness
    - Overall Realism

Artifacts:
- selected randomized pack:
  - [randomized_videos_selected](/Users/keweiou/Desktop/Project/ReGenNet/visualize/user_study/randomized_videos_selected)
- selected key list:
  - [selected_dataset_key.md](/Users/keweiou/Desktop/Project/ReGenNet/visualize/user_study/selected_dataset_key.md)
- ranking results:
  - `visualize/user_study/results/ranking_responses.csv`

## Main local scripts

### Asset preparation
- [export_user_study_pack.sh](/Users/keweiou/Desktop/Project/ReGenNet/visualize/user_study/export_user_study_pack.sh)
  - batch-export viewer-ready clips from checkpoints
- [render_pack_to_mp4.py](/Users/keweiou/Desktop/Project/ReGenNet/visualize/user_study/render_pack_to_mp4.py)
  - offline conversion from `P1.npz/P2.npz` clips to mp4
- [randomize_user_study_videos.py](/Users/keweiou/Desktop/Project/ReGenNet/visualize/user_study/randomize_user_study_videos.py)
  - anonymize rendered mp4 videos into `A/B/C/D`

### Study frontends
- [tools/user_study/gradio_user_study.py](/Users/keweiou/Desktop/Project/ReGenNet/tools/user_study/gradio_user_study.py)
  - 4-way single-choice study
- [tools/user_study/gradio_user_study_ranking.py](/Users/keweiou/Desktop/Project/ReGenNet/tools/user_study/gradio_user_study_ranking.py)
  - GT-reference ranking study

### Analysis
- [tools/user_study/analyze_user_study_results.py](/Users/keweiou/Desktop/Project/ReGenNet/tools/user_study/analyze_user_study_results.py)
  - decode anonymized `A/B/C/D` results back to true methods and summarize votes

## Supporting files

- [user_study.md](/Users/keweiou/Desktop/Project/ReGenNet/visualize/user_study/user_study.md)
  - manual viewer commands for pack01 and pack02 clips
- [dataset_keys.txt.example](/Users/keweiou/Desktop/Project/ReGenNet/visualize/user_study/dataset_keys.txt.example)
  - example list for `pack01`
- [dataset_keys_02.txt.example](/Users/keweiou/Desktop/Project/ReGenNet/visualize/user_study/dataset_keys_02.txt.example)
  - example list for `pack02`

## Recommended usage

### If you want a standard 4-way comparison
Use:
- `randomized_videos`
- `gradio_user_study.py`

### If you want GT as reference and only compare model outputs
Use:
- `randomized_videos_selected`
- `gradio_user_study_ranking.py`

This second setting is usually more suitable for comparing:
- `baseline`
- `HiReact*`
- `HiReact`

while using `GT` only as a perceptual reference.
