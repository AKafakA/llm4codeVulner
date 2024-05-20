import argparse

import torch
from transformers import AutoTokenizer
from utils import (convert_to_dataset, max_new_token_length,
                   text_column, label_column, ModelType, get_model,
                   print_metrics, get_prompt_prefix, inference_compare_code)
from data.process.utils import read_prompts


def main():
    parser = argparse.ArgumentParser(
        prog='Evaluation Scripts',
        description='Evaluation Scripts to print out semantic scores for trained and untrained models.',
        epilog=''
    )

    parser.add_argument('-v', '--vulnerability', type=str, required=True, default='sql_injection',
                        help='vulnerability type need to be repaired, default is sql_injection')
    parser.add_argument('-l', '--lang', type=str, default='python',
                        help='programming language need to be repaired, default is python')
    parser.add_argument('-m', '--model_name', required=True, type=str, default='Salesforce/codet5-small',
                        help='model need to be trained, default is Salesforce/codet5-small')
    parser.add_argument('-t', '--model_type', required=True, type=str, default='t5',
                        help='model type needed to be tested or trained. '
                             'It will used to initialized tokenizer from huggingface, default is t5. Use causal for '
                             'casualLM')
    parser.add_argument('--data_usage', type=float, default=1.0,
                        help='Indicate how the total data usage should be used, default is 1.0')
    parser.add_argument('--train_ratio', type=float, default=0.6,
                        help='Indicate the usage of selected data should be used for training, default is 0.6.')
    parser.add_argument('--val_ratio', type=float, default=0.2,
                        help='Indicate the usage of selected data should be used for validation, default is 0.2.')

    args = parser.parse_args()
    model_name = args.model_name
    model_type = ModelType[args.model_type]
    vulnerability = args.vulnerability
    lang = args.lang
    save_directory = "llm/models/{}".format(vulnerability + "-" + model_name)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    data_usage_ratio = args.data_usage

    trained_model = get_model(model_name, model_type, save_path=save_directory, device=device)
    untrained_model = get_model(model_name, model_type=model_type, device=device)

    prompt_prefix = get_prompt_prefix(vulnerability, lang)
    data_file = "data/{}.json".format(vulnerability)
    prompts, labels = read_prompts(data_file)
    train_dataset, validation_dataset, test_dataset = convert_to_dataset(prompts, labels,
                                                                         train_ratio=args.train_ratio,
                                                                         val_ratio=args.val_ratio,
                                                                         data_usage_ratio=data_usage_ratio)

    references = []
    predictions = []
    baseline_predictions = []
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    for example in test_dataset:
        inference_compare_code(target_tokenizer=tokenizer, baseline_tokenizer=tokenizer,
                               prompt_prefix=prompt_prefix,
                               baseline_predictions=baseline_predictions,
                               prompt=example[text_column], label=example[label_column],
                               target_model=trained_model, baseline_model=untrained_model, device=device,
                               references=references, predictions=predictions,
                               max_new_tokens=max_new_token_length)

    print("##################" + "Train model output metrics" + "##################")

    print_metrics(references, predictions, lang)

    print("##################" + "Raw model output metrics" + "##################")

    print_metrics(references, baseline_predictions, lang)
