import pandas as pd
import torch
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
                 sequence_length=384,
                 pad_token='PAD'):

        super().__init__()
        self.tokenizer = tokenizer
        self.word2index = word2index
        self.word2bert_tokens = {word: self.tokenizer.tokenize(word) for word
                                 in tqdm(list(self.word2index.keys())[1:])}
        self.word2bert_indices = {
            word: [self.tokenizer.vocab[bert_token] for bert_token in
                   self.word2bert_tokens[word]] for word in
            self.word2bert_tokens.keys()}

        self.sequence_length = sequence_length
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
            self.x_data.append(sum(bert_tokens, []))
            self.y_data.append(span)

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

    def get_embeddings(self, text):
        tokenized_text = self.tokenizer.tokenize(text)
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        tokens_tensor = torch.tensor([indexed_tokens])
        segments_ids = [1] * len(tokenized_text)
        segments_tensors = torch.tensor([segments_ids])
        self.model.eval()
        with torch.no_grad():
            encoded_layers, _ = self.model.bert(tokens_tensor,
                                                segments_tensors)
        token_embeddings = torch.stack(encoded_layers, dim=0)
        token_embeddings = torch.squeeze(token_embeddings, dim=1)
        token_embeddings = token_embeddings.permute(1, 0, 2)
        token_vecs_cat = []
        for token in token_embeddings:
            cat_vec = torch.cat((token[-1], token[-2], token[-3], token[-4]),
                                dim=0)
            # cat_vec = torch.sum(token[-4:], dim=0)
            token_vecs_cat.append(cat_vec)
        return torch.stack(token_vecs_cat, dim=0)

    def embed_data(self, texts: list):
        entries = []
        data_iterator = tqdm(texts, desc='Loading embeddings')
        for entry in data_iterator:
            entries.append(self.get_embeddings(entry))
        return entries


def main():
    data = pd.read_csv("sberquad.csv")
    
    par_tokens = [i.split() for i in data.paragraph_tokens]
    que_tokens = [tokenize_text(i) for i in data.question]
    answer_spans = data.word_answer_span

    word2index = {"PAD":0, "[CLS]":1, "[SEP]":2}

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