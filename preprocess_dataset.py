from razdel import sentenize, tokenize
from tqdm.auto import tqdm
import Levenshtein
from string import punctuation

punct = punctuation+'«»—…“”*–'

def tokenize_text(text):
    """Токенизация"""
    words = [_.text for _ in list(tokenize(text))]
    return words


def intersection(lst1, lst2):
    """Пересечение двух списков"""
    lst3 = [value for value in lst1 if value in lst2] 
    return lst3 


def lemmatize(sent,m):
    """Лемматизация"""
    return "".join([w.strip(punct) for w in m.lemmatize(sent)]).strip().split()

def get_leven_span_with_dist(answer, data):
    """Функция смотрит на ответ и параграф и выдает ([все спаны ответа], расстояние Левенштейна)"""
    answer = answer.lower()
    data = data.lower()
    anslen = len(answer)
    span_dict = {}
    span = []
    
    for i in range(0, len(data)-anslen):
        cur_data = data[i:i+anslen]
        if cur_data == answer:
            span.append((i,i+anslen)) 
            
    if span == []:
        for i in range(0, len(data)-anslen + 1):
            cur_data = data[i:i+anslen]
            dist = Levenshtein.distance(cur_data,answer)
            if (dist <= 5) and (dist not in span_dict):
                span_dict[dist] = [(i,i+anslen)]
            elif (dist <= 5) and (dist in span_dict):
                span_dict[dist].append((i,i+anslen))
                
    if span != []:
        span_dict[0] = span
    
    res_span = span_dict[min(span_dict)] if span_dict[min(span_dict)] != [] else span_dict[min(span_dict) + 1]
        
    return (res_span, min(span_dict))


def get_leven_span(answer, data):
    """Функция смотрит на ответ и параграф и выдает все спаны ответа (список)"""
    answer = answer.lower()
    data = data.lower()
    anslen = len(answer)
    span_dict = {}
    span = []
    
    for i in range(0, len(data)-anslen):
        cur_data = data[i:i+anslen]
        if cur_data == answer:
            span.append((i,i+anslen)) 
            
    if span == []:
        for i in range(0, len(data)-anslen + 1):
            cur_data = data[i:i+anslen]
            dist = Levenshtein.distance(cur_data,answer)
            if (dist <= 5) and (dist not in span_dict):
                span_dict[dist] = [(i,i+anslen)]
            elif (dist <= 5) and (dist in span_dict):
                span_dict[dist].append((i,i+anslen))
                
    if span != []:
        span_dict[0] = span
    
    res_span = span_dict[min(span_dict)] if span_dict[min(span_dict)] != [] else span_dict[min(span_dict) + 1]
        
    return res_span


def disambiguate_span(spans:list, paragraphs:list, questions:list, answers:list, m):
    """Функция возвращает список однозначных спанов ответов для всех вопросов"""
    result_spans = []
    
    for i,entry in tqdm(enumerate(spans)):
                
        if len(entry) > 1:
            
            spans_list = entry
            sent_chunks = {i.text:(i.start, i.stop) for i in list(sentenize(paragraphs[i]))}
            question = lemmatize(questions[i], m)
            max_overlap = 0
            optimal_span = ()

            for span in spans_list:
                for sent, chunk in sent_chunks.items():
                    if span[0] >= chunk[0] and span[1] <= chunk[1]:
                        cur_span = (span[0] - chunk[0], span[0] - chunk[0] + (span[1] - span[0]))
                        left_context, right_context = sent[:cur_span[0]], sent[cur_span[1]:]
                        left_context = lemmatize(left_context, m)
                        right_context = lemmatize(right_context, m)
                        overlap = len(intersection(left_context + right_context, question))
                        jaccar = overlap/(len(left_context + right_context) + len(question) - overlap)
                        if jaccar > max_overlap:
                            max_overlap = jaccar 
                            optimal_span = span

            result_spans.append(optimal_span)
            
        else:
            result_spans.append(entry[0])
    
    return result_spans 


def char_span_to_word_span(spans:list, paragraph:list):
    """Функция переводит спан на уровне символов в спан на уровне слов"""
    word_spans = []
    tokens = []
    for i, span in tqdm(enumerate(spans)):
        text = paragraph[i]
        if span[0] != 0:
            left_part = tokenize_text(text[:span[0]])
            left_part_len = len(left_part)
            answer_part = tokenize_text(text[span[0]:span[1]])
            answer_part_len = len(answer_part)
            word_spans.append([left_part_len, left_part_len + answer_part_len]) 
        else:
            answer_part_len = len(tokenize_text(text[span[0]:span[1]]))
            word_spans.append([0, answer_part_len]) 
        par_tokens = tokenize_text(text)
        tokens.append(par_tokens)
    return word_spans, tokens