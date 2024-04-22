import json

from datasets import Dataset
from torch.utils.data import DataLoader

prompt_prefix = "Please help to Fix this SQL: "
max_input_length = 256
max_target_length = 256
max_new_token_length = 48

text_column = 'raw_code'
label_column = 'fixed_code'


def read_prompts(filename):
    with open(filename) as file:
        data_str = file.read()  # Read the content of the file as a string
        data = json.loads(data_str)  # Parse the JSON string
        prompts = []
        labels = []

        for repo_url, commits_info in data.items():
            for commit_hash, commit_info in commits_info.items():
                # Check if the 'files' key exists in the commit_info
                for file_info in commit_info["files"].values():
                    for change in file_info["changes"]:
                        prompt_lines = change["diff"].split("\n- ")[1:]
                        label_lines = change["diff"].split("\n+ ")[1:]

                        for prompt_line, label_line in zip(prompt_lines, label_lines):
                            prompt = prompt_line.split("\n")[0].strip()
                            label = label_line.split("\n")[0].strip()
                            prompts.append(prompt)
                            labels.append(label)
        return prompts, labels


def convert_to_dataset(prompts, labels, train_ratio=0.6, val_ratio=0.2):
    total_prompts = len(prompts)

    prompt_id = 0
    train_list, validation_list, test_list = [], [], []

    for prompt, label in zip(prompts, labels):
        record = {text_column: prompt, label_column: label, "ID": prompt_id}
        if prompt_id <= total_prompts * train_ratio:
            train_list.append(record)
        elif prompt_id <= total_prompts * (val_ratio + train_ratio):
            validation_list.append(record)
        else:
            test_list.append(record)
        prompt_id = prompt_id + 1

    train_dataset = Dataset.from_list(train_list)
    validation_dataset = Dataset.from_list(validation_list)
    test_dataset = Dataset.from_list(test_list)
    return train_dataset, validation_dataset, test_dataset


def preprocess_prompts(example, tokenizer):
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


def get_dataloader(dataset, shuffle, batch_size, tokenizer, preprocess_function=preprocess_prompts):
    tokenizer = tokenizer

    def preprocess(example):
        return preprocess_function(example, tokenizer)

    processed_datasets = dataset.map(
        preprocess,
        batched=True
    )
    processed_datasets.set_format(type="torch", columns=['input_ids', 'attention_mask', 'labels'])
    return DataLoader(processed_datasets, shuffle=shuffle, batch_size=batch_size)
