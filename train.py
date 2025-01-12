import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import AdamW, get_scheduler
from tqdm.auto import tqdm
from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu
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
sumry_epochs = 100
trans_epochs = 300
beam_size = 4
no_repeat_ngram_size = 2

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')

#数据预处理
class sumry_dataset(Dataset):
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
                    'title': data["en_summary"],
                    'content': "summarize: "+data["en_dialogue"]
                }
        return Data
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class trans_dataset(Dataset):
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
                    'content': "translate English to Chinese: "+data["en_summary"]
                }
        return Data
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

sumry_train_data = sumry_dataset('./data/dialogsumx.train.jsonl')
sumry_valid_data = sumry_dataset('./data/dialogsumx.dev.jsonl')
sumry_test_data = sumry_dataset('./data/dialogsumx.test.jsonl')

trans_train_data = trans_dataset('./data/dialogsumx.train.jsonl')
trans_valid_data = trans_dataset('./data/dialogsumx.dev.jsonl')
trans_test_data = trans_dataset('./data/dialogsumx.test.jsonl')

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

sumry_train_dataloader = DataLoader(sumry_train_data, batch_size=train_batch_size, shuffle=True, collate_fn=collate_fn)
sumry_valid_dataloader = DataLoader(sumry_valid_data, batch_size=test_batch_size, shuffle=False, collate_fn=collate_fn)
trans_train_dataloader = DataLoader(trans_train_data, batch_size=train_batch_size, shuffle=True, collate_fn=collate_fn)
trans_valid_dataloader = DataLoader(trans_valid_data, batch_size=test_batch_size, shuffle=False, collate_fn=collate_fn)

#训练
def train_loop(dataloader, model, optimizer, lr_scheduler, epoch, total_loss):
    progress_bar = tqdm(range(len(dataloader)))
    progress_bar.set_description(f'loss: {0:>7f}')
    finish_batch_num = (epoch-1) * len(dataloader)
    
    model.train()
    for batch, batch_data in enumerate(dataloader, start=1):
        batch_data = batch_data.to(device)
        outputs = model(**batch_data)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        total_loss += loss.item()
        progress_bar.set_description(f'loss: {total_loss/(finish_batch_num + batch):>7f}')
        progress_bar.update(1)
    return total_loss

rouge = Rouge()

#验证和测试
def test_loop(dataloader, model, mode='Test', eval_ind = ""):
    assert mode in ['Valid', 'Test']
    preds, labels = [], []
    
    model.eval()
    for batch_data in tqdm(dataloader):
        batch_data = batch_data.to(device)
        with torch.no_grad():
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

        decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        label_tokens = np.where(label_tokens != -100, label_tokens, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(label_tokens, skip_special_tokens=True)

        preds += [' '.join(pred.strip()) for pred in decoded_preds]
        labels += [' '.join(label.strip()) for label in decoded_labels]
    if eval_ind == "rouge":
        scores = rouge.get_scores(hyps=preds, refs=labels, avg=True)
        result = {key: value['f'] * 100 for key, value in scores.items()}
        result['avg'] = np.mean(list(result.values()))
        print(f"{mode} Rouge1: {result['rouge-1']:>0.2f} Rouge2: {result['rouge-2']:>0.2f} RougeL: {result['rouge-l']:>0.2f}\n")
    elif eval_ind == "bleu":
        bleu_1 = sentence_bleu(labels, preds, weights=(1, 0, 0, 0))
        bleu_2 = sentence_bleu(labels, preds, weights=(0.5, 0.5, 0, 0))
        bleu_3 = sentence_bleu(labels, preds, weights=(0.33, 0.33, 0.33, 0))
        bleu_4 = sentence_bleu(labels, preds, weights=(0.25, 0.25, 0.25, 0.25))
        print(f"{mode} Bleu1: {bleu_1:>0.2f} Bleu2: {bleu_2:>0.2f} Bleu3: {bleu_3:>0.2f} Bleu4: {bleu_4:>0.2f}\n")
        result['avg'] = (bleu_1 + bleu_2 + bleu_3 + bleu_4) / 4
    return result

#微调模型
total_loss = 0.
best_avg_rouge = 0.
model_checkpoint = "./mt5"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
model = model.to(device)

optimizer = AdamW(model.parameters(), lr=learning_rate)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=sumry_epochs*len(sumry_train_dataloader),
)

print("Summarization fine-tune starts")
for t in range(sumry_epochs):
    print(f"Epoch {t+1}/{sumry_epochs}\n-------------------------------")
    total_loss = train_loop(sumry_train_dataloader, model, optimizer, lr_scheduler, t+1, total_loss)
    valid_rouge = test_loop(sumry_valid_dataloader, model, mode='Valid', eval_ind = "rouge")
    rouge_avg = valid_rouge['avg']
    if rouge_avg > best_avg_rouge:
        best_avg_rouge = rouge_avg
        print('saving a new checkpoint...\n')
        torch.save(model.state_dict(), 'mt5_summarization_weights.bin')
print("Done!")

total_loss = 0.
best_avg_bleu = 0.
model_checkpoint = "./mt5"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
model = model.to(device)

optimizer = AdamW(model.parameters(), lr=learning_rate)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=trans_epochs*len(trans_train_dataloader),
)

print("Translation fine-tune starts")
for t in range(trans_epochs):
    print(f"Epoch {t+1}/{trans_epochs}\n-------------------------------")
    total_loss = train_loop(trans_train_dataloader, model, optimizer, lr_scheduler, t+1, total_loss)
    valid_bleu = test_loop(trans_valid_dataloader, model, mode='Valid', eval_ind = "rouge")
    bleu_avg = valid_bleu['avg']
    if bleu_avg > best_avg_bleu:
        best_avg_bleu = bleu_avg
        print('saving a new checkpoint...\n')
        torch.save(model.state_dict(), 'mt5_translation_weights.bin')
print("Done!")
