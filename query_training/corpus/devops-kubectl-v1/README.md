---
dataset_info:
  features:
  - name: prompt
    dtype: string
  - name: response
    dtype: string
  splits:
  - name: train
    num_bytes: 40729093
    num_examples: 34535
  - name: test
    num_bytes: 411595
    num_examples: 349
  download_size: 9146060
  dataset_size: 41140688
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
  - split: test
    path: data/test-*
---
