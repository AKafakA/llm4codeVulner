from transformers import AutoTokenizer
from utils import (max_new_token_length, ModelType, get_model)

vulnerability = "plain_sql"
model_name = "Salesforce/codet5-small"
model_type = ModelType.T5_CONDITIONAL_GENERATION
tokenizer = AutoTokenizer.from_pretrained(model_name)

lang = 'python'
prompt_prefix = "Please help to Fix this Python: "
if vulnerability.endswith("sql"):
    prompt_prefix = "Please help to Fix this SQL code called in Python: "


save_directory = "llm/models/{}".format(vulnerability + "-" + model_name)
model = get_model(model_name, model_type, save_path=save_directory)
baseline_model = get_model(model_name, model_type)

batch_input = False
prompt_path = "data/test/{}/prompts.txt".format(vulnerability + "-" + model_name)
prediction_path = "data/test/{}/prediction.txt".format(vulnerability)
baseline_prediction_path = "data/test/{}/baseline_prediction.txt".format(vulnerability)

test_example = ""


if batch_input:
    with (open(prompt_path, "r+") as f,
          open(prediction_path, "w+") as prediction_file,
          open(baseline_prediction_path, "w+") as baseline_prediction_file):
        prompts = f.read().splitlines()
        for prompt in prompts:
            input_ids = tokenizer(prompt_prefix + prompt, return_tensors='pt').input_ids
            output = model.generate(input_ids, max_new_tokens=max_new_token_length)
            prediction_file.write(tokenizer.decode(output[0], skip_special_tokens=True))

            baseline_output = baseline_model.generate(input_ids, return_tensors='pt')
            baseline_prediction_file.write(tokenizer.decode(baseline_output[0], skip_special_tokens=True))

else:
    input_ids = tokenizer(prompt_prefix + test_example, return_tensors='pt').input_ids
    output = model.generate(input_ids, max_new_tokens=max_new_token_length)
    print("Train model output :", tokenizer.decode(output[0], skip_special_tokens=True))
    output = baseline_model.generate(input_ids, max_new_tokens=max_new_token_length)
    print("Baseline model output :", tokenizer.decode(output[0], skip_special_tokens=True))


