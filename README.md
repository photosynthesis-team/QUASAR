# QUASAR


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
  --resolution          Resize the images of the dataset: None, 224, 512
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
python3 generate_embeddigs.py --dataset <dataset-title> --dataset_dir <path ti dataset directory> --batch_size 1 --backbone <encoding backbone> --backbone_type <encodign backbone type> --pretrain <encoding backbone weights> --embeddings_dir <directory to store embeddings> --seed 42 --verbose --device cuda:0 --resolution 224
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

