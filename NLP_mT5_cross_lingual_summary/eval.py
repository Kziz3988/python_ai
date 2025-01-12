import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import AdamW, get_scheduler
from tqdm.auto import tqdm
from rouge import Rouge
import random
import numpy as np
import os
import json

max_dataset_size = 200000
max_input_length = 512
max_target_length = 32
train_batch_size = 8
test_batch_size = 8
learning_rate = 2e-5
epoch_num = 3
beam_size = 4
no_repeat_ngram_size = 2

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')

class dataset(Dataset):
    def __init__(self, data_file):
        self.data = self.load_data(data_file)
    
    def load_data(self, data_file):
        Data = {}
        with open(data_file, 'rt', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                if idx >= max_dataset_size:
                    break
                data = json.loads(line)
                Data[idx] = {
                    'title': data["zh_summary"],
                    'content': "summarize: "+data["en_dialogue"]
                }
        return Data
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

test_data = dataset('./data/dialogsumx.test.jsonl')
rouge = Rouge()

def collate_fn(batch_samples):
    batch_inputs, batch_targets = [], []
    for sample in batch_samples:
        batch_inputs.append(sample['content'])
        batch_targets.append(sample['title'])
    batch_data = tokenizer(
        batch_inputs, 
        padding=True, 
        max_length=max_input_length,
        truncation=True, 
        return_tensors="pt"
    )
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            batch_targets, 
            padding=True, 
            max_length=max_target_length,
            truncation=True, 
            return_tensors="pt"
        )["input_ids"]
        batch_data['decoder_input_ids'] = model.prepare_decoder_input_ids_from_labels(labels)
        end_token_index = torch.where(labels == tokenizer.eos_token_id)[1]
        for idx, end_idx in enumerate(end_token_index):
            labels[idx][end_idx+1:] = -100
        batch_data['labels'] = labels
    return batch_data

model_checkpoint = "./mt5"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
model = model.to(device)
test_dataloader = DataLoader(test_data, batch_size=32, shuffle=False, collate_fn=collate_fn)
model.load_state_dict(torch.load('mt5_summarization_weights.bin'))
model.eval()

with torch.no_grad():
    print('evaluating on test set...')
    sources, preds, labels = [], [], []
    for batch_data in tqdm(test_dataloader):
        batch_data = batch_data.to(device)
        generated_tokens = model.generate(
            batch_data["input_ids"],
            attention_mask=batch_data["attention_mask"],
            max_length=max_target_length,
            num_beams=beam_size,
            no_repeat_ngram_size=no_repeat_ngram_size,
        ).cpu().numpy()
        if isinstance(generated_tokens, tuple):
            generated_tokens = generated_tokens[0]
        label_tokens = batch_data["labels"].cpu().numpy()

        decoded_sources = tokenizer.batch_decode(
            batch_data["input_ids"].cpu().numpy(), 
            skip_special_tokens=True, 
            use_source_tokenizer=True
        )
        decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        label_tokens = np.where(label_tokens != -100, label_tokens, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(label_tokens, skip_special_tokens=True)

        sources += [source.strip() for source in decoded_sources]
        preds += [pred.strip() for pred in decoded_preds]
        labels += [label.strip() for label in decoded_labels]

test_data = {}
for idx in range(len(preds)):
    if idx >= max_dataset_size:
        break
    test_data[idx] = {
        'title': labels[idx],
        'content': "translate English to Chinese: "+preds[idx]
    }
#英文摘要翻译为中文摘要
model_checkpoint = "./mt5"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
model = model.to(device)
test_dataloader = DataLoader(test_data, batch_size=32, shuffle=False, collate_fn=collate_fn)
model.load_state_dict(torch.load('mt5_translation_weights.bin'))
model.eval()

with torch.no_grad():
    print('evaluating on test set...')
    preds, labels = [], []
    for batch_data in tqdm(test_dataloader):
        batch_data = batch_data.to(device)
        generated_tokens = model.generate(
            batch_data["input_ids"],
            attention_mask=batch_data["attention_mask"],
            max_length=max_target_length,
            num_beams=beam_size,
            no_repeat_ngram_size=no_repeat_ngram_size,
        ).cpu().numpy()
        if isinstance(generated_tokens, tuple):
            generated_tokens = generated_tokens[0]
        label_tokens = batch_data["labels"].cpu().numpy()

        decoded_sources = tokenizer.batch_decode(
            batch_data["input_ids"].cpu().numpy(), 
            skip_special_tokens=True, 
            use_source_tokenizer=True
        )
        decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        label_tokens = np.where(label_tokens != -100, label_tokens, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(label_tokens, skip_special_tokens=True)

        preds += [pred.strip() for pred in decoded_preds]
        labels += [label.strip() for label in decoded_labels]
    scores = rouge.get_scores(
        hyps=[' '.join(pred) for pred in preds], 
        refs=[' '.join(label) for label in labels], 
        avg=True
    )
    rouges = {key: value['f'] * 100 for key, value in scores.items()}
    rouges['avg'] = np.mean(list(rouges.values()))
    print(f"Test Rouge1: {rouges['rouge-1']:>0.2f} Rouge2: {rouges['rouge-2']:>0.2f} RougeL: {rouges['rouge-l']:>0.2f}\n")
    results = []
    print('saving predicted results...')
    for source, pred, label in zip(sources, preds, labels):
        results.append({
            "document": source, 
            "prediction": pred, 
            "summarization": label
        })
        
    with open('two-stage-preds.json', 'wt', encoding='utf-8') as f:
        for exapmle_result in results:
            f.write(json.dumps(exapmle_result, ensure_ascii=False) + '\n')
