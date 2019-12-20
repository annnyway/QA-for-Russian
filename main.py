import numpy as np
import pandas as pd
import torch
from pytorch_pretrained_bert import BertTokenizer, BertConfig, BertForMaskedLM
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from transformers import BertTokenizer
import joblib

from preprocess_dataset import tokenize_text

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


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
        self.y_data = []
        self.bert_spans = []
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
                    if slice_ > self.sequence_length:
                        bert_tokens = sum(par_tokens[slice_:-slice_] + quest_tokens, [])
                        bert_span_start = bert_span_start - slice_ 
                        bert_span_end = bert_span_end - slice_
                    else:
                        slice_ = len(bert_tokens) - 512
                        bert_span_start = bert_span_start - slice_ 
                        bert_span_end = bert_span_end - slice_
                        bert_tokens = sum(par_tokens[slice_:] + quest_tokens, [])
            
            bert_span = (bert_span_start, bert_span_end-1)
            
            target = [-1] * self.sequence_length
            if bert_span_start < self.sequence_length:
                target[bert_span_start] = 0

            # assert bert_span_end > 0
            if bert_span_end < self.sequence_length:
                target[bert_span_end-1] = 1
                
            self.x_data.append(bert_tokens)
            self.y_data.append(target)
            self.bert_spans.append(bert_span)

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
        bert_span = self.bert_spans[idx]
        bert_span = torch.Tensor(bert_span).long()


        return x, y, bert_span

class Classifier(torch.nn.Module):
    
    def __init__(self, 
               hidden_size=768,
               linear_out=2,
               batch_first=True):
  
        super(Classifier, self).__init__()
            
        self.output_model_file = "lm/pytorch_model.bin"
        self.output_config_file = "lm/config.json"
        self.tokenizer = BertTokenizer.from_pretrained("lm", do_lower_case=False)
        self.config = BertConfig.from_json_file(self.output_config_file)
        self.model = BertForMaskedLM(self.config)
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.state_dict = torch.load(self.output_model_file, map_location=device)
        self.model.load_state_dict(self.state_dict)
        self.lstm = torch.nn.LSTM(hidden_size, 300)
        self.linear = torch.nn.Linear(300, linear_out)
        
    
    def get_embeddings(self, x_instance):
        indexed_tokens = x_instance.tolist()
        tokens_tensor = torch.tensor([indexed_tokens])
        segments_ids = [1] * len(indexed_tokens)
        segments_tensors = torch.tensor([segments_ids])
        self.model.eval()
        with torch.no_grad():
            encoded_layers, _ = self.model.bert(tokens_tensor.to(device),
                                       segments_tensors.to(device))
        token_embeddings = torch.stack(encoded_layers, dim=0)
        token_embeddings = torch.squeeze(token_embeddings, dim=1)
        token_embeddings = token_embeddings.permute(1, 0, 2)
        token_vecs_cat = []
        for token in token_embeddings:
            cat_vec = torch.stack((token[-1], token[-2], token[-3], token[-4]))
            mean_vec = torch.mean(cat_vec, 0)
            token_vecs_cat.append(mean_vec)
        token_vecs_cat = torch.stack(token_vecs_cat, dim=0)
        return token_vecs_cat

    
    def embed_data(self, x): 
        entries = [] 
        for entry in x:
            emb = self.get_embeddings(entry.to(device)).to(device)
            entries.append(emb)
        return torch.stack(entries)
    
        
    def forward(self, x):

        h = self.embed_data(x)
        h = h.permute(1, 0, 2)
        output, _ = self.lstm(h)
        pred = self.linear(output)
        pred = pred.permute(1, 0, 2)
        return pred


def train_model(model, epochs, train_loader, optimizer, criterion):
    train_losses = []
    for n_epoch in range(epochs):
        progress_bar = tqdm(total=len(train_loader.dataset),
                            desc='Epoch {}'.format(n_epoch + 1))

        for x, y, bert_span in train_loader:
            optimizer.zero_grad()
            pred = model.forward(x.to(device))

            loss = criterion(pred.to(device).permute(0, 2, 1),
                             y.long().to(device))
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            progress_bar.set_postfix(loss=np.mean(train_losses[-500:]))
            progress_bar.update(x.shape[0])
        
        progress_bar.close()

        torch.save(model, "classifier-" + str(n_epoch+1) + ".pkl")
        joblib.dump(train_losses, "train_losses.pkl")

        # torch.save({
        #    'epoch': n_epoch+1,
        #     'model_state_dict': model.state_dict,
        #    'optimizer_state_dict': optimizer.state_dict,
        #    'loss': train_losses,
        #    }, 
        #    "/content/drive/My Drive/colab/classifier_state_dict" + str(n_epoch+1) + ".pkl")
        
    return train_losses


def main():
    data = pd.read_csv("sberquad.csv")

    data['span_len'] = data.apply(
        lambda row: int(row.word_answer_span.split(",")[1]) - int(
            row.word_answer_span.split(",")[0]), axis=1)
    data['span_avg'] = data.apply(lambda row: (int(
        row.word_answer_span.split(",")[1]) + int(
        row.word_answer_span.split(",")[0])) / 2, axis=1)

    data = data[(data.span_len <= 10) & (data.span_avg <= 150)]

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

    from sklearn.model_selection import train_test_split

    train, test = train_test_split(data, test_size=0.2, random_state=42)
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)

    # test = pd.read_csv("test.csv")
    # train = pd.read_csv("train.csv")

    par_tokens_test = [i.split() for i in test.paragraph_tokens]
    que_tokens_test = [tokenize_text(i) for i in test.question]
    answer_spans_test = test.word_answer_span

    par_tokens_train = [i.split() for i in train.paragraph_tokens]
    que_tokens_train = [tokenize_text(i) for i in train.question]
    answer_spans_train = train.word_answer_span

    train_data = QADataset(tokenizer=tokenizer,
                           paragraph_tokens=par_tokens_train,
                           question_tokens=que_tokens_train,
                           answer_spans=answer_spans_train,
                           word2index=word2index)

    test_data = QADataset(tokenizer=tokenizer,
                          paragraph_tokens=par_tokens_test,
                          question_tokens=que_tokens_test,
                          answer_spans=answer_spans_test,
                          word2index=word2index)

    train_loader = DataLoader(train_data, batch_size=32, drop_last=True)
    test_loader = DataLoader(test_data, batch_size=32, drop_last=True)

    epochs = 5

    device = torch.device('cuda') # if torch.cuda.is_available() else torch.device('cpu')

    model = Classifier().to(device)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=8e-6,
                                 weight_decay=0.01)

    print("Training the model...")
    train_losses = train_model(model=model, epochs=epochs, optimizer=optimizer,
                         criterion=criterion, train_loader=train_loader)


if __name__ == "__main__":
    main()
