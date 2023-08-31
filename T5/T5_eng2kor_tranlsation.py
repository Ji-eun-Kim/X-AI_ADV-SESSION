# !pip install transformers==4.28.0
# !pip install sentencepiece
# !pip install evaluate
# !pip install sacrebleu

import pandas as pd
from datasets import load_dataset

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import DataCollatorForSeq2Seq
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer

import evaluate

import numpy as np
import torch
import multiprocessing

data_path = './Train Val Test_csv/'
save_path = './en_ko_data_tsv/'

# tsv 파일로 저장하기
en_ko_df_train = pd.read_csv(data_path + 'train.csv')
en_ko_df_train.to_csv('train.tsv', sep = '\t', index = False)

en_ko_df_valid = pd.read_csv(data_path + 'valid.csv')
en_ko_df_valid.to_csv('valid.tsv', sep = '\t', index = False)

en_ko_df_test = pd.read_csv(data_path + 'test.csv')
en_ko_df_test.to_csv('test.tsv', sep = '\t', index = False)

# 아래 필요한 데이터셋 형태로 변환
data_files = {"train": "train.tsv", "valid": "valid.tsv", "test": "test.tsv"}
dataset =  load_dataset("csv", data_files=data_files, delimiter="\t")

# cuda 활성화
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# tokenizer
model_ckpt = "KETI-AIR/ke-t5-base"
max_token_length = 64

tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

def convert_examples_to_features(examples):

    model_inputs = tokenizer(examples['english'],
                            text_target=examples['korean'],
                            max_length=max_token_length, truncation=True)

    return model_inputs

NUM_CPU = multiprocessing.cpu_count()

tokenized_datasets = dataset.map(convert_examples_to_features,
                                batched=True,
                                # 이걸 쓰지 않으면 원 데이터 'english', 'korean'가 남아서
                                # 아래서 콜레이터가 패딩을 못해서 에러남
                                remove_columns=dataset["train"].column_names,
                                num_proc=NUM_CPU)

# Model
model = AutoModelForSeq2SeqLM.from_pretrained(model_ckpt).to(device)

metric = evaluate.load("sacrebleu")

def compute_metrics(eval_preds):
    preds, labels = eval_preds

    if isinstance(preds, tuple):
        preds = preds[0]

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [[label.strip()] for label in decoded_labels]

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["score"]}

    return result

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

training_args = Seq2SeqTrainingArguments(
    output_dir="chkpt",
    learning_rate=0.0005,
    weight_decay=0.01,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=128,
    num_train_epochs=1,
    save_steps=500,
    save_total_limit=2,
    evaluation_strategy="epoch",
    logging_strategy="no",
    predict_with_generate=True,
    fp16=False,
    gradient_accumulation_steps=2,
    report_to="none" # Wandb 로그 끄기
)

trainer = Seq2SeqTrainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["valid"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# trainer.train() # 메모리 에러

# input_text = input("") # ocr output 값 사용

# inputs = tokenizer(input_text, return_tensors="pt",
#                    padding=True, max_length=max_token_length).to(device)

# result_text = model.generate(
#     **inputs,
#     max_length=max_token_length,
#     num_beams=5,
# )

# preds = tokenizer.batch_decode( result_text, skip_special_tokens=True )

# print(preds)