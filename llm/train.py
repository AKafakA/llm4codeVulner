from transformers import AutoTokenizer, T5ForConditionalGeneration, AutoModelForCausalLM
from utils import (read_prompts, convert_to_dataset, get_dataloader, prompt_prefix, max_new_token_length,
                   text_column, label_column, ModelType, get_model)
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor
from code_model import CodeModel

vulnerability = "plain_sql"
training_epochs = 1
warmup_steps = 1000
lr = 5e-5
model_name = "Salesforce/codet5-small"
model_type = ModelType.T5_CONDITIONAL_GENERATION

# model_name = "google/codegemma-2b"
# model_type = ModelType.CAUSAL_ML

save_directory = "./models/{}".format(vulnerability + "-" + model_name)
data_file = "../data/{}.json".format(vulnerability)
tokenizer = AutoTokenizer.from_pretrained(model_name)

prompts, labels = read_prompts(data_file)
train_dataset, validation_dataset, test_dataset = convert_to_dataset(prompts, labels)

train_dataloader = get_dataloader(dataset=train_dataset, shuffle=True, batch_size=8, tokenizer=tokenizer)
validation_dataloader = get_dataloader(dataset=validation_dataset, shuffle=False, batch_size=2, tokenizer=tokenizer)
test_dataloader = get_dataloader(dataset=test_dataset, shuffle=False, batch_size=2, tokenizer=tokenizer)

model = CodeModel(training_dataloader=train_dataloader, testing_dataloader=test_dataloader,
                  validating_dataloader=validation_dataloader, model_name=model_name, model_type=model_type,
                  num_train_epochs=training_epochs, lr=lr, warmup_steps=warmup_steps)

lr_monitor = LearningRateMonitor(logging_interval='step')
trainer = Trainer(
    default_root_dir="./" + "models/{}".format(vulnerability + "-" + model_name),
    callbacks=[lr_monitor],
    max_epochs=training_epochs)
trainer.fit(model)
model.model.save_pretrained(save_directory)
print("Training finished. Saved model to {}".format(save_directory))

test_example = train_dataset[0]
print("Test example : ")
print("Code to be fix:", test_example[text_column])
print("Fixed code: ", test_example[label_column])

trained_model = get_model(model_name, model_type, save_path=save_directory)
input_ids = tokenizer(prompt_prefix + test_example['raw_code'], return_tensors='pt').input_ids
outputs = trained_model.generate(input_ids, max_new_tokens=max_new_token_length)
print("Train model output :", tokenizer.decode(outputs[0], skip_special_tokens=True))


untrained_model = get_model(model_name, model_type=model_type)
outputs = untrained_model.generate(input_ids, max_new_tokens=max_new_token_length)
print("Raw model output", tokenizer.decode(outputs[0], skip_special_tokens=True))
