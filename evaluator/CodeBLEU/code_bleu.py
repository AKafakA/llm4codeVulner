# Copyright (c) Microsoft Corporation. 
# Licensed under the MIT license.

import evaluator
from evaluator.CodeBLEU import bleu, weighted_ngram_match, syntax_match, dataflow_match


# -*- coding:utf-8 -*-

def calculate_code_bleu_from_lists(pre_references, prediction, lang, alpha=0.25, beta=0.25, gamma=0.25, theta=0.25):

    references = []
    for i in range(len(prediction)):
        ref_for_instance = []
        for j in range(len(pre_references)):
            ref_for_instance.append(pre_references[j][i])
        references.append(ref_for_instance)
    assert len(references) == len(pre_references) * len(prediction)

    # calculate ngram match (BLEU)
    tokenized_prediction = [x.split() for x in prediction]
    tokenized_refs = [[x.split() for x in reference] for reference in references]

    ngram_match_score = bleu.corpus_bleu(tokenized_refs, tokenized_prediction)
    path = evaluator.CodeBLEU.__path__[0] + '/keywords/' + lang + ".txt"
    keywords = [x.strip() for x in open(path, 'r', encoding='utf-8').readlines()]

    def make_weights(reference_tokens, key_word_list):
        return {token: 1 if token in key_word_list else 0.2 \
                for token in reference_tokens}

    tokenized_refs_with_weights = [[[reference_tokens, make_weights(reference_tokens, keywords)] \
                                    for reference_tokens in reference] for reference in tokenized_refs]

    weighted_ngram_match_score = weighted_ngram_match.corpus_bleu(tokenized_refs_with_weights, tokenized_prediction)

    # calculate syntax match
    syntax_match_score = syntax_match.corpus_syntax_match(references, prediction, lang)

    # calculate dataflow match
    dataflow_match_score = dataflow_match.corpus_dataflow_match(references, prediction, lang)

    code_bleu_score = alpha * ngram_match_score \
                      + beta * weighted_ngram_match_score \
                      + gamma * syntax_match_score \
                      + theta * dataflow_match_score
    # print('Code BLEU score: ' + str(code_bleu_score))
    return code_bleu_score


def calculate_code_bleu(reference_file, prediction_file, lang, alpha=0.25, beta=0.25, gamma=0.25, theta=0.25):
    # preprocess inputs
    pre_references = [[x.strip() for x in open(reference_file, 'r', encoding='utf-8').readlines()]]
    prediction = [x.strip() for x in open(prediction_file, 'r', encoding='utf-8').readlines()]

    return calculate_code_bleu_from_lists(pre_references, prediction, lang, alpha, beta, gamma, theta)
