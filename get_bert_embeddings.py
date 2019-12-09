# usage:
# 
# dataset = QADataset(tokenizer=tokenizer,
#                   paragraph_tokens=par_tokens,
#                   question_tokens=que_tokens,
#                   answer_spans=answer_spans,
#                   word2index=word2index)
# embeddings = embed_data(dataset.x_data)

import torch
from pytorch_pretrained_bert import BertTokenizer, BertConfig, BertForMaskedLM


def get_embeddings(x_data_instance:list):
    indexed_tokens = x_data_instance
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_ids = [1] * len(indexed_tokens)
    segments_tensors = torch.tensor([segments_ids])
    model.eval()
    with torch.no_grad():
        encoded_layers, _ = model.bert(tokens_tensor,
                                       segments_tensors)
    token_embeddings = torch.stack(encoded_layers, dim=0)
    token_embeddings = torch.squeeze(token_embeddings, dim=1)
    token_embeddings = token_embeddings.permute(1, 0, 2)
    token_vecs_cat = []
    for token in token_embeddings:
        cat_vec = torch.cat((token[-1], token[-2], token[-3], token[-4]),
                            dim=0)
        token_vecs_cat.append(cat_vec)
    return torch.stack(token_vecs_cat, dim=0)


def embed_data(x_data): 
    
    output_model_file = "lm/pytorch_model.bin"
    output_config_file = "lm/config.json"
    tokenizer = BertTokenizer.from_pretrained("lm", do_lower_case=False)
    config = BertConfig.from_json_file(output_config_file)
    model = BertForMaskedLM(config)
    state_dict = torch.load(output_model_file, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    
    entries = [] 
    data_iterator = tqdm(x_data, desc='Loading embeddings')
    
    for entry in data_iterator:
        entries.append(get_embeddings(entry))
        
    return entries  