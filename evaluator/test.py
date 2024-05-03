from metrics_getter import get_code_bleu, get_code_bert

references_file_path = "evaluator/references.txt"
prediction_file_path = "evaluator/predictions.txt"

code_bleu_score = get_code_bleu(references_file_path, prediction_file_path)
code_bert_score_precision, code_bert_score_recall, code_bert_score_F1, code_bert_score_f3 = get_code_bert(references_file_path, prediction_file_path)

print("Code Bleu :" + code_bleu_score)
print("Code Bert Precision :" + code_bert_score_precision)
print("Code Bert Recall : " + code_bert_score_recall)
print("Code Bert F1 :" + code_bert_score_f3)
print("Code Bert F3 :" + code_bert_score_f3)

