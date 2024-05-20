import argparse

import torch
from transformers import AutoTokenizer
from utils import (convert_to_dataset, get_dataloader, inference_compare_code,
                   text_column, label_column, ModelType, get_model, get_pytorch_trainer,
                   print_metrics, get_prompt_prefix, max_new_token_length, get_model_type)
from data.process.utils import read_prompts
from pytorch_lightning.callbacks import LearningRateMonitor
from code_model import CodeModel
import os


def main():
    parser = argparse.ArgumentParser(
        prog='Training Models',
        description='Fine-tune Code Models using Transformers for certain vulnerabilities',
        epilog=''
    )

    parser.add_argument('-v', '--vulnerability', type=str, default='sql_injection',
                        help='Vulnerability type need to be repaired, default is sql_injection')
    parser.add_argument('-l', '--lang', type=str, default='python',
                        help='programming language need to be repaired, default is python')
    parser.add_argument('-e', '--training_epochs', type=int, default=20,
                        help='number of training epochs')
    parser.add_argument('-m', '--model_name', type=str, default='Salesforce/codet5-small',
                        help='model need to be trained, default is Salesforce/codet5-small')
    parser.add_argument('-t', '--model_type', type=str, default='t5',
                        help='model type needed to be tested or trained. '
                             'It will used to initialized tokenizer from huggingface, default is t5. Use causal for '
                             'casualLM')
    parser.add_argument('--use_lora', type=bool, default=False,
                        help='whether to use LoRA for training')
    parser.add_argument('--use_deepspeed', type=bool, default=False,
                        help='whether to use DeepSpeed Zero offloading for training')
    parser.add_argument('--data_usage', type=float, default=1.0,
                        help='Indicate how the total data usage should be used, default is 1.0')
    parser.add_argument('--train_ratio', type=float, default=0.6,
                        help='Indicate the usage of selected data should be used for training, default is 0.6.')
    parser.add_argument('--val_ratio', type=float, default=0.2,
                        help='Indicate the usage of selected data should be used for validation, default is 0.2.')
    parser.add_argument('--enable_evaluation', type=bool, default=True,
                        help='Also run the evaluation to print the metrics')

    args = parser.parse_args()
    vulnerability = args.vulnerability
    lang = args.lang
    training_epochs = args.training_epochs
    data_usage_ratio = args.data_usage
    accelerator = args.device
    warmup_steps = 10
    lr = 5e-5
    enable_parallelism_tokenizer = False
    enable_evaluation = args.enable_evaluation
    use_deepspeed = args.use_deepspeed
    use_lora = args.use_lora
    model_name = args.model_name
    model_type = get_model_type(args.model_type)
    device = "cuda:0" if torch.cuda.is_available() and accelerator == 'gpu' else "cpu"
    prompt_prefix = get_prompt_prefix(vulnerability, lang)

    if not enable_parallelism_tokenizer:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

    save_directory = "llm/models/{}".format(vulnerability + "-" + model_name)
    data_file = "data/{}.json".format(vulnerability)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    prompts, labels = read_prompts(data_file)
    train_dataset, validation_dataset, test_dataset = convert_to_dataset(prompts, labels,
                                                                         train_ratio=args.train_ratio,
                                                                         val_ratio=args.val_ratio,
                                                                         data_usage_ratio=data_usage_ratio)

    train_dataloader = get_dataloader(dataset=train_dataset, shuffle=True, batch_size=8,
                                      prompt_prefix=prompt_prefix, tokenizer=tokenizer)
    validation_dataloader = get_dataloader(dataset=validation_dataset, shuffle=False, batch_size=2,
                                           prompt_prefix=prompt_prefix, tokenizer=tokenizer)
    test_dataloader = get_dataloader(dataset=test_dataset, shuffle=False, batch_size=2,
                                     prompt_prefix=prompt_prefix, tokenizer=tokenizer)

    model = CodeModel(training_dataloader=train_dataloader, testing_dataloader=test_dataloader,
                      validating_dataloader=validation_dataloader, model_name=model_name, model_type=model_type,
                      num_train_epochs=training_epochs, lr=lr, warmup_steps=warmup_steps, use_lora=use_lora)

    lr_monitor = LearningRateMonitor(logging_interval='step')
    root_dir = "llm/models/{}".format(vulnerability + "-" + model_name)
    trainer = get_pytorch_trainer(vulnerability=vulnerability, training_epochs=training_epochs, model_name=model_name,
                                  lr_monitor=lr_monitor, use_deepspeed=use_deepspeed,
                                  accelerator=accelerator, root_dir=root_dir)

    trainer.fit(model)
    model.model.save_pretrained(save_directory)
    print("Training finished. Saved model to {}".format(save_directory))

    trained_model = get_model(model_name, model_type, save_path=save_directory, device=device)
    untrained_model = get_model(model_name, model_type=model_type, device=device)

    if enable_evaluation:
        references = []
        predictions = []
        baseline_predictions = []
        for test_example in test_dataset:
            inference_compare_code(target_tokenizer=tokenizer, baseline_tokenizer=tokenizer,
                                   prompt_prefix=prompt_prefix,
                                   prompt=test_example[text_column], label=test_example[label_column],
                                   target_model=trained_model, baseline_model=untrained_model, device=device,
                                   references=references, predictions=predictions,
                                   baseline_predictions=baseline_predictions, max_new_tokens=max_new_token_length)

        print("##################" + "Train model output metrics" + "##################")

        print_metrics(references, predictions, lang)

        print("##################" + "Raw model output metrics" + "##################")

        print_metrics(references, baseline_predictions, lang)


if __name__ == "__main__":
    main()
