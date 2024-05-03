from transformers import AutoTokenizer
from utils import (read_prompts, convert_to_dataset, get_dataloader, prompt_prefix, max_new_token_length,
                   text_column, label_column, ModelType, get_model, get_pytorch_trainer)
from pytorch_lightning.callbacks import LearningRateMonitor
from code_model import CodeModel
import os
from evaluator.metrics_getter import get_code_bleu_from_list, get_code_bert_from_list


vulnerability = "plain_sql"
lang = 'python'

training_epochs = 10
warmup_steps = 1000
lr = 5e-5
# test with small data for check the correctness
data_usage_ratio = 1.0
accelerator = 'gpu'
enable_parallelism_tokenizer = False
enable_evaluation = False
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

save_directory = "./models/{}".format(vulnerability + "-" + model_name)
data_file = "../data/{}.json".format(vulnerability)
tokenizer = AutoTokenizer.from_pretrained(model_name)

prompts, labels = read_prompts(data_file)
train_dataset, validation_dataset, test_dataset = convert_to_dataset(prompts, labels, data_usage_ratio=data_usage_ratio)

train_dataloader = get_dataloader(dataset=train_dataset, shuffle=True, batch_size=4, tokenizer=tokenizer)
validation_dataloader = get_dataloader(dataset=validation_dataset, shuffle=False, batch_size=2, tokenizer=tokenizer)
test_dataloader = get_dataloader(dataset=test_dataset, shuffle=False, batch_size=2, tokenizer=tokenizer)

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
    references = []
    predictions = []
    for test_example in test_dataloader:
        references.append(test_example[label_column])
        input_ids = tokenizer(prompt_prefix + test_example[text_column], return_tensors='pt').input_ids
        outputs = trained_model.generate(input_ids, max_new_tokens=max_new_token_length)
        predictions.append(tokenizer.decode(outputs[0], skip_special_tokens=True))

    code_bleu_score = get_code_bleu_from_list(references, predictions, lang=lang)
    code_bert_score_precision, code_bert_score_recall, code_bert_score_f1, code_bert_score_f3 = (
        get_code_bert_from_list(references, predictions, lang=lang))
    print("Code bleu score : ", code_bleu_score)
    print("Code Bert score precision : ", code_bert_score_precision)
    print("Code Bert score recall : ", code_bert_score_recall)
    print("Code Bert score f1 : ", code_bert_score_f1)
    print("Code Bert score f3 : ", code_bert_score_f3)
