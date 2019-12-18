from sklearn.metrics import accuracy_score, f1_score
from collections import Counter
from tqdm.auto import tqdm
import torch
import numpy as np



def compute_f1(gold_toks, pred_toks):
    common = Counter(gold_toks) & Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1



def test_model(model, test_loader, criterion, device):
    
    count = 0

    test_losses = []
    mean_test_losses = []
    all_test_answers = []
    all_real_answers = []
    all_test_answer_spans = []
    all_real_answer_spans = []
    all_f1_scores = []
    
    
    model.eval()
    
    for x, y, bert_span in tqdm(test_loader):
        
        with torch.no_grad():
            pred = model.forward(x.to(device))
            
        
        test_loss = criterion(pred.to(device).permute(0, 2, 1), y.long().to(device))
        mean_test_loss = np.mean(test_loss.item())
        mean_test_losses.append(mean_test_loss)
        print('Testing: test loss = {:.3f} '.format(mean_test_loss))
                
       
        # counting Exact Match
        preds_cat = pred.permute(0, 2, 1)
        test_preds_argmax = torch.argmax(preds_cat, dim=2).tolist()
        print(test_preds_argmax[0])
        test_answer_spans_str = [str(i[0])+":"+str(i[1]) for i in test_preds_argmax]
        real_answer_spans = [[item.index(word_indx) for word_indx in item if word_indx != -1] for item in y.tolist()]
        real_answer_spans_str = [str(i[0]) + ":" + str(i[1]) 
                             if len(i) == 2 
                             else str(i[0]) + ":" + str(i[0]) 
                             for i in real_answer_spans]
        accuracy = accuracy_score(real_answer_spans_str, test_answer_spans_str)
        print("Evaluating: test accuracy = {:.3f} ".format(accuracy))
        all_test_answer_spans.append(test_answer_spans_str)
        all_real_answer_spans.append(real_answer_spans_str)
        
        
        # soft F1 - overlap between word indices
        test_answer_indexes = [[i[0],i[1]] for i in test_preds_argmax]
        f1_scores = []
        for k,(i,j) in enumerate(zip(real_answer_spans, test_answer_indexes)):
            gold_tokens = x[k][i[0]:i[1]+1].tolist() if len(i)==2 else x[k][i[0]:i[0]+1].tolist()
            pred_tokens = x[k][j[0]:j[1]+1].tolist() if len(j)==2 else x[k][j[0]:j[0]+1].tolist()
            f1 = compute_f1(gold_tokens, pred_tokens)
            f1_scores.append(f1)
        
        print("Evaluating: test F1-score = {:.3f} ".format(np.mean(f1_scores)))
        all_f1_scores.append(f1_scores)
        
        
        # сохраним строки правильных и неправильных ответов (на всякий)
        test_answers = [model.tokenizer.decode(x[k][i[0]:i[1]+1].tolist()) if len(i) == 2 
                        else model.tokenizer.decode(x[k][i[0]:i[0]+1].tolist())
                        for k,i in enumerate(test_answer_indexes)]  
        real_answers = [model.tokenizer.decode(x[k][i[0]:i[1]+1].tolist()) if len(i) == 2 
                        else model.tokenizer.decode(x[k][i[0]:i[0]+1].tolist())
                        for k,i in enumerate(real_answer_spans)]
        all_test_answers.append(test_answers)
        all_real_answers.append(real_answers)
        
        
        
        # пока посчитаем 3 батча, чтобы долго не считалось
        count += 1
        if count == 3:
            break
                
    return (all_f1_scores, 
            all_test_answers, 
            all_real_answers, 
            all_test_answer_spans, 
            all_real_answer_spans, 
            mean_test_losses)