| Dataset              | Usage    | Link                                                                         |
| -------------------- | -------- | ---------------------------------------------------------------------------- |
| **ImageNet**         | Training | [Download](https://image-net.org/download.php)                               |
| **ImageNet-NeurIPS** | Testing  | [Kaggle](https://www.kaggle.com/c/nips-2017-non-targeted-adversarial-attack) |
| **MSCOCO val2017**   | Testing  | [Download](http://images.cocodataset.org/zips/val2017.zip)                   |  



 Training

```bash
python train.py \
  --pretrained_model_name_or_path "your_path" \
  --dataset_dir "your_path" \
  --label_to_caption_path "/path/imagenet_class_index.json" \
  --output_root "output" \
  --model_type res152 \
  --eps 16 \
  --label_flag N8 \
  --num_train_epochs 10 \
  --train_batch_size 16 \
  --learning_rate 2e-05 \
  --s_for 6 \
  --s_gen 6 \
  --freeze_ratio 0.5
```

 Test  Models

```bash
python inference.py \
  --pretrained_model_name_or_path "your_path" \
  --dataset_dir "your dataset" \
  --lora_path "your_path.safetensors" \
  --label_to_caption_path "/path/imagenet_class_index.json" \
  --excel_output_path "output/results.xlsx" \
  --model_type res152 \
  --eps 16 \
  --label_flag N8 \
  --s_for 6 \
  --s_gen 6 \
  --freeze_ratio 0.5 \
  --include_standard_models
```
