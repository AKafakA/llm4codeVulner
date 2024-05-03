import code_bert_score
from bleu import _bleu
from evaluator.CodeBLEU.code_bleu import calculate_code_bleu
from torch import mean


def get_code_bert(reference_file, prediction_file, lang='python'):
    references = [x.strip() for x in open(reference_file, 'r', encoding='utf-8').readlines()]
    predictions = [x.strip() for x in open(prediction_file, 'r', encoding='utf-8').readlines()]
    precision, recall, F1, F3 = code_bert_score.score(cands=predictions, refs=references, lang=lang)
    avg_pre = mean(precision)
    avg_rec = mean(recall)
    avg_f1 = mean(F1)
    avg_f3 = mean(F3)
    return avg_pre, avg_rec, avg_f1, avg_f3


def get_bleu_score(ref_file, trans_file):
    return _bleu(ref_file, trans_file)


def get_code_bleu(reference_file, prediction_file, lang='python'):
    return calculate_code_bleu(reference_file, prediction_file, lang)


