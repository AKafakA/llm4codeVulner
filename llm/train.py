from transformers import AutoTokenizer
from utils import (read_prompts, convert_to_dataset, get_dataloader, max_new_token_length,
                   text_column, label_column, ModelType, get_model, get_pytorch_trainer, print_metrics)
from pytorch_lightning.callbacks import LearningRateMonitor
from code_model import CodeModel
import os

vulnerability = "plain_sql"
lang = 'python'
prompt_prefix = "Please help to Fix this Python: "
if vulnerability.endswith("sql"):
    prompt_prefix = "Please help to Fix this SQL code called in Python: "

training_epochs = 10
warmup_steps = 1000
lr = 5e-5
# test with small data for check the correctness
data_usage_ratio = 1.0
accelerator = 'gpu'
enable_parallelism_tokenizer = False
enable_evaluation = False
save_output = False
use_deepspeed = False
use_lora = False

model_name = "Salesforce/codet5-small"
model_type = ModelType.T5_CONDITIONAL_GENERATION
# Can test on cpu since the model is small
# accelerator = 'gpu'

# model_name = "google/codegemma-2b"
# model_type = ModelType.CAUSAL_LM

if not enable_parallelism_tokenizer:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

save_directory = "llm/models/{}".format(vulnerability + "-" + model_name)
data_file = "data/{}.json".format(vulnerability)
tokenizer = AutoTokenizer.from_pretrained(model_name)

prompts, labels = read_prompts(data_file)
train_dataset, validation_dataset, test_dataset = convert_to_dataset(prompts, labels, data_usage_ratio=data_usage_ratio)

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
trainer = get_pytorch_trainer(vulnerability=vulnerability, training_epochs=training_epochs, model_name=model_name,
                              lr_monitor=lr_monitor, use_deepspeed=use_deepspeed, accelerator=accelerator)

trainer.fit(model)
model.model.save_pretrained(save_directory)
print("Training finished. Saved model to {}".format(save_directory))

sanity_checking_example = train_dataset[0]
print("Test example : ")
print("Code to be fix:", sanity_checking_example[text_column])
print("Fixed code: ", sanity_checking_example[label_column])

trained_model = get_model(model_name, model_type, save_path=save_directory)
input_ids = tokenizer(prompt_prefix + sanity_checking_example[text_column], return_tensors='pt').input_ids
outputs = trained_model.generate(input_ids, max_new_tokens=max_new_token_length)
print("Train model output :", tokenizer.decode(outputs[0], skip_special_tokens=True))

untrained_model = get_model(model_name, model_type=model_type)
outputs = untrained_model.generate(input_ids, max_new_tokens=max_new_token_length)
print("Raw model output", tokenizer.decode(outputs[0], skip_special_tokens=True))

if enable_evaluation:

    input_file_path = "../data/test/{}/input.json".format(vulnerability)
    references_file_path = "../data/test/{}/references.json".format(vulnerability)
    prediction_file_path = "../data/test/{}/prediction.json".format(vulnerability)

    references = []
    predictions = []
    baseline_predictions = []
    for test_example in test_dataset:
        with (open(input_file_path, 'w') as input_file,
              open(references_file_path, 'w') as references_file, open(prediction_file_path, 'w') as prediction_file):

            input_ids = tokenizer(prompt_prefix + test_example[text_column], return_tensors='pt').input_ids
            output = trained_model.generate(input_ids, max_new_tokens=max_new_token_length)
            baseline_output = untrained_model.generate(input_ids, max_new_tokens=max_new_token_length)
            references.append(test_example[label_column])
            predictions.append(tokenizer.decode(output[0], skip_special_tokens=True))
            baseline_predictions.append(tokenizer.decode(baseline_output[0], skip_special_tokens=True))
            if save_output:
                references_file.write(test_example[label_column])
                input_file.write(test_example[text_column])
                prediction_file.write(tokenizer.decode(outputs[0], skip_special_tokens=True))

    print("##################" + "Train model output metrics" + "##################")

    print_metrics(references, predictions, lang)

    print("##################" + "Raw model output metrics" + "##################")

    print_metrics(references, baseline_predictions, lang)
