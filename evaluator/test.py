from metrics_getter import get_bleu_score, get_code_bleu, get_code_bert

references_file_path = "references.txt"
prediction_file_path = "predictions.txt"

bleu_score = get_bleu_score(references_file_path, prediction_file_path)
code_bleu_score = get_code_bleu(references_file_path, prediction_file_path)
_, _, _, code_bert_score_f3 = get_code_bert(references_file_path, prediction_file_path)

print("BLEU Score: {}".format(bleu_score))
print("F3 Code BERT Score: {}".format(code_bert_score_f3))
print("Code BLEU Score: {}".format(code_bleu_score))
