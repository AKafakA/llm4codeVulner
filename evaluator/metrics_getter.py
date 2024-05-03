import code_bert_score
from evaluator.CodeBLEU.code_bleu import calculate_code_bleu, calculate_code_bleu_from_lists
from torch import mean


def get_code_bert(reference_file, prediction_file, lang='python'):
    references = [x.strip() for x in open(reference_file, 'r', encoding='utf-8').readlines()]
    predictions = [x.strip() for x in open(prediction_file, 'r', encoding='utf-8').readlines()]
    return get_code_bert_from_list(references, predictions, lang)


def get_code_bert_from_list(references, predictions, lang='python'):
    precision, recall, F1, F3 = code_bert_score.score(cands=predictions, refs=references, lang=lang)
    avg_pre = mean(precision).item()
    avg_rec = mean(recall).item()
    avg_f1 = mean(F1).item()
    avg_f3 = mean(F3).item()
    return avg_pre, avg_rec, avg_f1, avg_f3


def get_code_bleu(reference_file, prediction_file, lang='python'):
    return calculate_code_bleu(reference_file, prediction_file, lang)


def get_code_bleu_from_list(references, predictions, lang='python'):
    return calculate_code_bleu_from_lists(references, predictions, lang)
