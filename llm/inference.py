from transformers import AutoTokenizer
from utils import (read_prompts, convert_to_dataset, get_dataloader, prompt_prefix, max_new_token_length,
                   text_column, label_column, ModelType, get_model, get_pytorch_trainer)

vulnerability = "plain_sql"
model_name = "Salesforce/codet5-small"
model_type = ModelType.T5_CONDITIONAL_GENERATION
tokenizer = AutoTokenizer.from_pretrained(model_name)


save_directory = "./models/{}".format(vulnerability + "-" + model_name)
data_file = "../data/{}.json".format(vulnerability)
data_usage_ratio = 1.0

input_file_path = "../data/test/{}/input.json".format(vulnerability)
references_file_path = "../data/test/{}/references.json".format(vulnerability)
prediction_file_path = "../data/test/{}/prediction.json".format(vulnerability)

train_ratio = 0.6
val_ratio = 0.2
prompts, labels = read_prompts(data_file)
train_dataset, validation_dataset, test_dataset = convert_to_dataset(prompts, labels, data_usage_ratio=data_usage_ratio)
test_dataloader = get_dataloader(dataset=test_dataset, shuffle=False, batch_size=2, tokenizer=tokenizer)

trained_model = get_model(model_name, model_type, save_path=save_directory)

for test_example in test_dataloader:
    with (open(input_file_path, 'w') as input_file,
          open(references_file_path, 'w') as references_file, open(prediction_file_path, 'w') as prediction_file):
        input_file.write(test_example[text_column])
        references_file.write(test_example[label_column])
        input_ids = tokenizer(prompt_prefix + test_example[text_column], return_tensors='pt').input_ids
        outputs = trained_model.generate(input_ids, max_new_tokens=max_new_token_length)
        prediction_file.write(tokenizer.decode(outputs[0], skip_special_tokens=True))



