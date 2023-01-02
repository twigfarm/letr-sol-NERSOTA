# NERSOTA-SLNER-ELECTRA

2022 TWIGFARM SOL Project VOL.03
"한국어 구어체 NER 모델 성능 향상 연구 프로젝트"

# Data

## 1. Pretraining Corpus

[AIHub] 방송 콘텐츠 대본 요약 데이터] <br>
[AIHub] 일상생활 및 구어체 한-중, 한-일 번역 병렬 말뭉치 데이터 <br>
[AIHub] 다국어 구어체 번역 병렬 말뭉치 데이터 <br>
[모두의 말뭉치] 구어체 말뭉치 <br>
[모두의 말뭉치] 일상 대화 말뭉치 2020 <br>

## 2. Finetuning Corpus

[AIHub] 방송 콘텐츠 한-중, 한-일 번역 병렬 말뭉치 데이터 <br>
[AIHub] 일상생활 및 구어체 한-영 번역 병렬 말뭉치 데이터 <br>
[모두의 말뭉치] 개체명 분석 말뭉치 2021 <br>

# Pretraining ELECTRA-small

<img width="1191" alt="스크린샷 2022-12-20 오후 12 23 56" src="https://user-images.githubusercontent.com/91872769/209758741-3f2873c7-14eb-423c-8d8e-cb9d008130db.png">

# Result

| Model                     | Macro F1 score |
| :------------------------ | -------------: |
| `KoBERT-NER(Naver NER)`   |           0.34 |
| `KoELECTRA-NER(Finetuned)`|           0.80 |
| `NERSOTA-ELECTRA-NER`     |           0.77 |


# NERSOTA-ELECTRA-small on 🤗 Transformers 🤗

## 1. Pytorch Model & Tokenizer

```python
from transformers import ElectraModel, ElectraTokenizer

model = ElectraModel.from_pretrained("jieun0115/nersota-electra-small-discriminator")  # KoELECTRA-Small
```

## 2. Tokenizer Example

```python
>>> from transformers import ElectraTokenizer
>>> tokenizer = ElectraTokenizer.from_pretrained("jieun0115/nersota-electra-small-discriminator")
>>> tokenizer.tokenize("[CLS] 한국어 구어체 특화 개체명 인식기입니다. [SEP]")
['[CLS]', '한국어', '구', '##어', '##체', '특화', '개체', '##명', '인식', '##기', '##입니다', '.', '[SEP]']
>>> tokenizer.convert_tokens_to_ids(['[CLS]', '한국어', '구', '##어', '##체', '특화', '개체', '##명', '인식', '##기', '##입니다', '.', '[SEP]'])
[3, 10751, 1242, 4127, 4385, 19988, 21695, 4101, 7352, 4136, 6896, 1015, 4]
```

# ELECTRA Pretraining

## 1. Requirements

```python
torch==1.12.1
transformers==4.25.1
cudatoolkit == 10.0
sklearn
scipy
```

## 2. Make tfrecords

```bash
# `data` 디렉토리를 생성 후, corpus를 여러 개로 분리
$ mkdir data
$ split -a 4 -l {$NUM_LINES_PER_FILE} -d {$CORPUS_FILE} ./data/data_
```

```bash
python3 build_pretraining_dataset.py --corpus-dir data \
                                     --vocab-file vocab.txt \
                                     --output-dir pretrain_tfrecords \
                                     --max-seq-length 128 \
                                     --num-processes 4 \
                                     --no-lower-case
```

## 3. How to Run Pretraining

```bash
# Small model
$ python3 run_pretraining.py --data-dir {$BUCKET_NAME} --model-name {$SMALL_OUTPUT_DIR} --hparams config/small_config.json
```

# ELECTRA Finetuning

## 1. Requirements

```python
torch==1.10.2
transformers==4.18.0
cudatoolkit == 10.2
seqeval
fastprogress
attrdict
```

## 2. How to Run NER

```bash
$ python3 run_ner.py --task ner --config_file koelectra-small.json
```


## Reference
- [ELECTRA](https://github.com/google-research/electra)<br>
- [KoELECTRA](https://github.com/monologg/KoELECTRA)<br>
- [pytorch-bert-crf-ner](https://github.com/eagle705/pytorch-bert-crf-ner)<br>
- [LETR API](https://www.letr.ai/)<br>
- [AIHub](https://www.aihub.or.kr/)<br>
- [모두의 말뭉치](https://corpus.korean.go.kr/)<br>
- [트위그팜](https://www.twigfarm.net/)<br>
- [Label Studio](https://labelstud.io/)
