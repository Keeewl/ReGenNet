conda activate regennet5090

####### refine_v2: run module-1 minimal loop #######
export CUDA_VISIBLE_DEVICES=7
bash refine_v2/commands/00_build_contact_labels.sh
bash refine_v2/commands/01_select_windows.sh
bash refine_v2/commands/02_audit_windows.sh

