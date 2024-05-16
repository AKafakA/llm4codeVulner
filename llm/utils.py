from enum import Enum

from datasets import Dataset
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader
from transformers import T5ForConditionalGeneration, AutoModelForCausalLM, AutoModel
from pytorch_lightning.strategies import deepspeed

from evaluator.metrics_getter import get_code_bleu_from_list, get_code_bert_from_list

max_input_length = 256
max_target_length = 256
max_new_token_length = 256

text_column = 'raw_code'
label_column = 'fixed_code'


class ModelType(Enum):
    T5_CONDITIONAL_GENERATION = "t5_conditional_generation"
    CAUSAL_LM = "CAUSAL_LM"
    AUTO = "auto"


def convert_to_dataset(prompts, labels, train_ratio=0.6, val_ratio=0.2, data_usage_ratio=1.0):
    total_prompts = len(prompts) * data_usage_ratio

    prompt_id = 0
    train_list, validation_list, test_list = [], [], []

    for prompt, label in zip(prompts, labels):
        record = {text_column: prompt, label_column: label, "ID": prompt_id}
        if prompt_id <= total_prompts * train_ratio:
            train_list.append(record)
        elif prompt_id <= total_prompts * (val_ratio + train_ratio):
            validation_list.append(record)
        elif prompt_id <= total_prompts:
            test_list.append(record)
        else:
            break
        prompt_id = prompt_id + 1

    train_dataset = Dataset.from_list(train_list)
    validation_dataset = Dataset.from_list(validation_list)
    test_dataset = Dataset.from_list(test_list)
    return train_dataset, validation_dataset, test_dataset


def preprocess_prompts(example, tokenizer, prompt_prefix):
    prefix = prompt_prefix
    codes = example['raw_code']
    fix = example['fixed_code']

    inputs = [prefix + code for code in codes]
    model_inputs = tokenizer(inputs, max_length=max_input_length, padding="max_length", truncation=True)

    labels = tokenizer(fix, max_length=max_target_length, padding="max_length", truncation=True).input_ids

    # important: we need to replace the index of the padding tokens by -100
    # such that they are not taken into account by the CrossEntropyLoss
    labels_with_ignore_index = []
    for labels_example in labels:
        labels_example = [label if label != 0 else -100 for label in labels_example]
        labels_with_ignore_index.append(labels_example)

    model_inputs["labels"] = labels_with_ignore_index

    return model_inputs


def get_dataloader(dataset, shuffle, batch_size, tokenizer, prompt_prefix,
                   preprocess_function=preprocess_prompts, num_workers=16):
    tokenizer = tokenizer

    def preprocess(example):
        return preprocess_function(example, tokenizer, prompt_prefix=prompt_prefix)

    processed_datasets = dataset.map(
        preprocess,
        batched=True
    )
    processed_datasets.set_format(type="torch", columns=['input_ids', 'attention_mask', 'labels'])
    return DataLoader(processed_datasets, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers)


def get_model(model_name, model_type, save_path=None, device='cpu'):
    model_context = model_name
    if save_path:
        model_context = save_path
    if model_type == ModelType.T5_CONDITIONAL_GENERATION:
        model = T5ForConditionalGeneration.from_pretrained(model_context)
    elif model_type == ModelType.CAUSAL_LM:
        model = AutoModelForCausalLM.from_pretrained(model_context)
    else:
        model = AutoModel.from_pretrained(model_context)
    return model.to(device)


def get_pytorch_trainer(vulnerability, model_name, lr_monitor, training_epochs, root_dir, use_deepspeed=False,
                        accelerator='gpu', log_every_n_steps=10, csv_logger=True):
    logger = CSVLogger(root_dir + "/logs", name=model_name)
    if use_deepspeed:
        trainer = Trainer(
            default_root_dir=root_dir,
            callbacks=[lr_monitor],
            max_epochs=training_epochs,
            accelerator=accelerator,
            strategy=deepspeed.DeepSpeedStrategy(
                stage=3,
                offload_optimizer=True,
                offload_parameters=True,
            ),
            precision=16,
            log_every_n_steps=log_every_n_steps,
            logger=logger
        )
        trainer.strategy.config["zero_force_ds_cpu_optimizer"] = False
        return trainer
    else:
        return Trainer(
            default_root_dir="./" + "models/{}".format(vulnerability + "-" + model_name),
            callbacks=[lr_monitor],
            max_epochs=training_epochs,
            accelerator=accelerator,
            log_every_n_steps=log_every_n_steps,
            logger=logger
        )


def print_metrics(references, predictions, lang):
    code_bleu_score = get_code_bleu_from_list([references], predictions, lang=lang)
    code_bert_score_precision, code_bert_score_recall, code_bert_score_f1, code_bert_score_f3 = (
        get_code_bert_from_list(references, predictions, lang=lang))
    print("Code bleu score : ", code_bleu_score)
    print("Average Code Bert score precision : ", code_bert_score_precision)
    print("Average Code Bert score recall : ", code_bert_score_recall)
    print("Average Code Bert score f1 : ", code_bert_score_f1)
    print("Average Code Bert score f3 : ", code_bert_score_f3)


def get_prompt_prefix(vulnerability, lang):
    prompt_prefix = "Please help to Fix this {}: ".format(lang)
    if vulnerability.endswith("sql"):
        prompt_prefix = "Please help to Fix this SQL code called in {}: ".format(lang)
    return prompt_prefix


def generate_and_write_fixed_code(model, source_code, tokenizer, prompt_prefix, prompts, device='cpu'):
    rep = {}
    fixed_code = source_code
    for prompt in prompts:
        input_ids = tokenizer(prompt_prefix + prompt, return_tensors='pt').input_ids.to(device)
        output = model.generate(input_ids, max_new_tokens=max_new_token_length)
        target_output_code = tokenizer.decode(output[0], skip_special_tokens=True)
        rep[prompt] = target_output_code
    for prompt, fix in rep.items():
        fixed_code = fixed_code.replace(prompt, fix)
    return fixed_code


def generate_and_write_fixed_code_without_prompts(model, source_code, tokenizer, prompt_prefix):
    input_ids = tokenizer(prompt_prefix + source_code, return_tensors='pt').input_ids
    output = model.generate(input_ids)
    target_output_code = tokenizer.decode(output[0], skip_special_tokens=True)
    return target_output_code


def apply_label(source_code, prompts, labels):
    rep = {}
    fixed_code = source_code
    for prompt, label in zip(prompts, labels):
        rep[prompt] = label
    for prompt, fix in rep.items():
        fixed_code = fixed_code.replace(prompt, fix)
    return fixed_code
