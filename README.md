# QUASAR

This repository contais implenetation of the paper [QUASAR: QUality and Aesthetics Scoring with Advanced Representations](https://arxiv.org/abs/2403.06866) 
by Sergey Kastryulin, Denis Prokopenko, Artem Babenko and Dmitry V. Dylov.

QUASAR is a no-reference fully unsupervised metric for Image Quality and Aesthetics Assessment. It is designed to improve
the quality of its text-based predicessors while keeping their flexibility and generalizability.

## Reproducibility

The metric is based on pre-computed centroids of image embeddings. The following code downloads them automaticaly if you do not have them in place. You can also download them manually from the [releases page](https://github.com/photosynthesis-team/QUASAR/releases/tag/embeds) or refer to the [Preprocessing](#reprocessing) section to compute them yourself.

To reproduce results reported in the paper use the `main.py` script. 

```bash
python main.py [-h] [--prompt_data {text,KADIS700k,PIPAL,AVA}] [--prompt_backbone {CLIP-RN50_no-pos}] [--prompt_ratio PROMPT_RATIO] [--target_data {TID2013,KonIQ10k,KADID10k,LIVEitW,SPAQ,TAD66k,AADB,PieAPP}]
               [--target_data_subset {None,all,train,test}] [--target_backbone {CLIP-RN50_no-pos}] [--target_cv_folds TARGET_CV_FOLDS] [--aggregation_type {mean,clustering}] [--batch_size BATCH_SIZE]
               [--device {cpu,cuda}] [--seed SEED] [--median_offset_ratio MEDIAN_OFFSET_RATIO]

Script to run an experiment

optional arguments:
  -h, --help            show this help message and exit
  --prompt_data {text,KADIS700k,PIPAL,AVA}
                        The data to form anchors. `text` stands for CLIP-IQA
  --prompt_backbone {CLIP-RN50_no-pos}
                        Embeddings extractor for image-based prompt data
  --prompt_ratio PROMPT_RATIO
                        Fraction of embeddings to take for anchor forming
  --target_data {TID2013,KonIQ10k,KADID10k,LIVEitW,SPAQ,TAD66k,AADB,PieAPP}
                        The target dataset to compute scores and SRCC values on
  --target_data_subset {None,all,train,test}
                        Select which subset of target data will be used to compute SRCC values. Each dataset has its own default value because it varies in literature. Use None if not sure for a default value.
  --target_backbone {CLIP-RN50_no-pos}
                        Embeddings extractor for image-based prompt data
  --target_cv_folds TARGET_CV_FOLDS
                        Number of cross validation folds
  --aggregation_type {mean,clustering}
                        The way to aggregate embeddings into anchors
  --batch_size BATCH_SIZE
                        mind large batches for low VRAM GPUs
  --device {cpu,cuda}
  --seed SEED
  --median_offset_ratio MEDIAN_OFFSET_RATIO
                        If offset aggreation is used, this one determince the offset from the median score
```

```bash
python main.py \
  --prompt_data <anchor dataset name> \
  --prompt_backbone <backbone of a model to compute anchor embeds> \
  --prompt_ratio <how much data to take> \
  --target_data <target dataset name> \
  --target_backbone <backbone of a model to compute target embeds> \
  --batch_size <from 1 to any, flex with your high VRAM GPU here> \
  --device cuda:0 \
  --seed 42
```

## Preprocessing 

The preprocessing step transforms the raw data to embeddings in lattent space.


```bash
python3 generate_embeddigs.py -h
usage: generate_embeddigs.py [-h] [--dataset] [--dataset_dir] [--batch_size] [--resolution] [--backbone]
                             [--backbone_type] [--pretrain] [--positional_embedding] [--no-positional_embedding]
                             [--embeddings_dir] [--device] [--seed] [--verbose]

Script to generate the features

options:
  -h, --help            show this help message and exit
  --dataset             supported datasets: kadis700k, pipal, tid2013, koniq10k, kadid10k, liveitw, spaq, tad66k,
                        pieapp, sac, coyo700m.
  --dataset_dir         path to the dataset
  --batch_size          batch size. Choose 1 to use native resolution of the dataset.
  --resolution          resize the images of the dataset: None, 224, 512
  --backbone            extractor backbone: RN50, ViT-H-14, ViT-bigG-14, ViT-L-14, coca_ViT-L-14, vitl14
  --backbone_type       extractor backbone type: clip, open-clip, dinov2
  --pretrain            pretrain version
  --positional_embedding
                        enable positional embeddings
  --no-positional_embedding
                        disable positional embeddings
  --embeddings_dir      path to embeddings directory
  --device              device to use
  --seed                random seed
  --verbose             logs
```

```bash
python3 generate_embeddigs.py \
  --dataset <dataset-title> \
  --dataset_dir <path ti dataset directory> \
  --batch_size 1 \
  --backbone <encoding backbone> \
  --backbone_type <encodign backbone type> \
  --pretrain <encoding backbone weights> \
  --embeddings_dir <directory to store embeddings> \
  --seed 42 \
  --verbose \
  --device cuda:0 \
  --resolution 224 
```

Currently supported parameters:
```
# Datasets
kadis700k, pipal, tid2013, koniq10k, kadid10k, liveitw, spaq, tad66k, pieapp, sac, coyo700m

# Combinations of backbone_type|backbone|preptrain|positional_embedding
- clip|RN50|openai|True
- clip|RN50|openai|False
- open-clip|ViT-H-14|laion2b_s32b_b79k|None
- open-clip|ViT-bigG-14|laion2b_s39b_b160k|None
- open-clip|ViT-L-14|laion2B-s32B-b82K|None
- open-clip|coca_ViT-L-14|mscoco_finetuned_laion2B-s13B-b90k|None
- dinov2|vitl14|dinov2_vitl14|None 
```

## Citation

```
@article{kastryulin2024quasar,
  title={QUASAR: QUality and Aesthetics Scoring with Advanced Representations},
  author={Kastryulin, Sergey and Prokopenko, Denis and Babenko, Artem and Dylov, Dmitry V},
  journal={arXiv preprint arXiv:2403.06866},
  year={2024}
}
```