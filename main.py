import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from transformers import BertTokenizer
from preprocess_dataset import tokenize_text


class QADataset(Dataset):

    def __init__(self, tokenizer: BertTokenizer,
                 paragraph_tokens: list,
                 question_tokens: list,
                 answer_spans: list,
                 word2index: dict,
                 verbose=True,
                 max_seq_len=512,
                pad_token="[PAD]"):

        super().__init__()
        self.tokenizer = tokenizer
        self.word2index = word2index
        self.word2bert_tokens = {word: self.tokenizer.tokenize(word) for word
                                 in tqdm(list(self.word2index.keys())[1:])}
        self.word2bert_indices = {
            word: [self.tokenizer.vocab[bert_token] for bert_token in
                   self.word2bert_tokens[word]] for word in
            self.word2bert_tokens.keys()}

        self.sequence_length = max_seq_len
        self.pad_index = self.word2index[pad_token]

        self.x_data = []
        # self.load_x(paragraph_tokens, question_tokens, verbose=verbose)
        self.y_data = []
        # self.load_y(answer_spans)
        self.load_x_y(paragraph_tokens, question_tokens, answer_spans)

    def load_x_y(self, paragraphs, questions, spans, verbose=True):
        for par, quest, span in tqdm(zip(paragraphs, questions, spans),
                                     desc="Loading data", disable=not verbose):
            tokens = ["[CLS]"] + par + ["[SEP]"] + quest + ["[SEP]"]
            start, end = span.split(",")
            start, end = int(start), int(end)
            bert_tokens = [self.word2bert_indices[word] for word in tokens]
            bert_span_start = sum(len(x) for x in bert_tokens[:start + 1])
            bert_span_end = sum(len(x) for x in bert_tokens[:end + 1]) # прибавляем 1, т.к. у нас в начале есть еще токен CLS
            span = (bert_span_start, bert_span_end) 
            
            bert_tokens = sum(bert_tokens, [])
            if len(bert_tokens) > 512:
                par_tokens = [self.word2bert_indices[word] for word in ["[CLS]"] + par]
                quest_tokens = [self.word2bert_indices[word] for word in ["[SEP]"] + quest + ["[SEP]"]]
                if bert_span_start <= len(par_tokens)/2: # если спан в первой половине параграфа
                    slice_ = len(bert_tokens) - 512
                    bert_tokens = sum(par_tokens[:-slice_] + quest_tokens, [])
                elif bert_span_start > len(par_tokens)/2:
                    slice_ = len(bert_tokens) - 512
                    bert_tokens = sum(par_tokens[slice_:] + quest_tokens, [])
                    
            self.x_data.append(bert_tokens)
            target = [0] * self.sequence_length
            if bert_span_start < self.sequence_length:
                target[bert_span_start] = 1

            if bert_span_end < self.sequence_length:
                target[bert_span_end] = 2
                
            self.y_data.append(target)

    def padding(self, sequence):
        if len(sequence) > self.sequence_length:
            sequence = sequence[: self.sequence_length]
        elif len(sequence) < self.sequence_length:
            sequence += [self.pad_index for i in
                         range(self.sequence_length - len(sequence))]
        return sequence

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):

        x = self.x_data[idx]
        x = self.padding(x)
        x = torch.Tensor(x).long()
        y = self.y_data[idx]
        y = torch.Tensor(y).long()
        return x, y


def main():
    data = pd.read_csv("sberquad.csv")
    
    par_tokens = [i.split() for i in data.paragraph_tokens]
    que_tokens = [tokenize_text(i) for i in data.question]
    answer_spans = data.word_answer_span

    word2index = {"[PAD]":0, "[CLS]":1, "[SEP]":2}

    for sent in par_tokens:
        for token in sent:
            if token not in word2index:
                word2index[token] = len(word2index)

    for que in que_tokens:
        for token in que:
            if token not in word2index:
                word2index[token] = len(word2index)

    tokenizer = BertTokenizer.from_pretrained("lm", do_lower_case=False)
    
    dataset = QADataset(tokenizer=tokenizer,
                   paragraph_tokens=par_tokens,
                   question_tokens=que_tokens,
                   answer_spans=answer_spans,
                   word2index=word2index)

if __name__ == "__main__":
    main()