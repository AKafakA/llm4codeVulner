from evaluator.metrics_getter import get_bleu_score, get_code_bleu, get_code_bert

vulnerability = "plain_sql"
model_name = "Salesforce/codet5-small"

references_file_path = "../data/test/{}/references.json".format(vulnerability)
prediction_file_path = "../data/test/{}/references.json".format(vulnerability)

bleu_score = get_bleu_score(references_file_path, prediction_file_path)
code_bleu_score = get_code_bleu(references_file_path, prediction_file_path)
code_bert_score_precision, code_bert_score_recall, code_bert_score_f1, code_bert_score_f3 = get_code_bert(references_file_path, prediction_file_path)

print("BLEU Score: {}".format(bleu_score))
print("Code BLEU Score: {}".format(code_bleu_score))
print("Code BERT Score Precision: {}".format(code_bert_score_precision))
print("Code BERT Score Recall: {}".format(code_bert_score_recall))
print("Code BERT Score F1: {}".format(code_bert_score_f1))
print("Code BERT Score F3: {}".format(code_bert_score_f3))


