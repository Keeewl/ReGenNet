

cmdm:

```bash
cd visualize/viewer
python data_viewer.py \
  --dataset interx \
  --data_dir ../../outputs/single_stage1_cmdm_G038T003A016R005/motions \
  --texts_dir '' \
  --title 'cnetv5-single-G038T003A016R005'
```

cnetv5:

```bash
cd visualize/viewer
python data_viewer.py \
  --dataset interx \
  --data_dir ../../outputs/single_stage1_cnetv5_G038T003A016R005/motions \
  --texts_dir '' \
  --title 'cnetv5-single-G038T003A016R005'
```

HiReact:

```bash
cd visualize/viewer
python data_viewer.py \
  --dataset interx \
  --data_dir ../../outputs/single_stage2_hireact_G038T003A016R005/refined \
  --texts_dir '' \
  --title 'hireact-exp8-G038T003A016R005'
```


GT Video:

```bash
cd visualize/viewer
python data_viewer.py \
  --dataset interx \
  --data_dir ../../outputs/interx_regen_train_restored_height \
  --texts_dir ''
```