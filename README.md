# Russian-bashkirian translator
This repository contains the source code for machine translation from Russian into Bashkir. T5 language model is used. The model is trained from scratch on [bashkir russian parallel corpora](https://huggingface.co/datasets/AigizK/bashkir-russian-parallel-corpora).

## Current results
|model|BLEU|train size|
|-----|----|----------|
|T5-small|0.195|50(100)k|

## Develop in docker
```bash
docker compose up --build -d
```