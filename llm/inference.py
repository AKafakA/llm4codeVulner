import os
from enum import Enum

from transformers import AutoTokenizer
from utils import max_new_token_length, ModelType, get_model, get_prompt_prefix

vulnerability = "plain_sql"
lang = 'python'
prompt_prefix = get_prompt_prefix(vulnerability, lang)

target_model_name = "Salesforce/codet5-small"
model_type = ModelType.T5_CONDITIONAL_GENERATION
target_tokenizer = AutoTokenizer.from_pretrained(target_model_name)
save_directory = "llm/models/{}".format(vulnerability + "-" + target_model_name)
target_model = get_model(target_model_name, model_type, save_path=save_directory)

baseline_model_name = "google/codegemma-2b"
if not baseline_model_name == target_model_name:
    baseline_tokenizer = AutoTokenizer.from_pretrained(baseline_model_name)
else:
    baseline_tokenizer = target_tokenizer
baseline_model = get_model(baseline_model_name, model_type)
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class TestType(Enum):
    BATCH = "batch"
    SANITY = "sanity"


test_type = TestType.SANITY

################################################################
# input for batch check
prompt_path = "data/test/{}/prompts.txt".format(vulnerability + "-" + target_model_name)
prediction_path = "data/test/{}/prediction.txt".format(vulnerability)
baseline_prediction_path = "data/test/{}/baseline_prediction.txt".format(vulnerability)

###############################################################
# input for sanity check
test_examples = [""]
# test_examples = ["WHERE parent_id IN ({list_root_ids}"]
############################################################

if test_type == TestType.BATCH:
    with (open(prompt_path, "r+") as f,
          open(prediction_path, "w+") as prediction_file,
          open(baseline_prediction_path, "w+") as baseline_prediction_file):
        prompts = f.read().splitlines()
        for prompt in prompts:
            target_input_ids = target_tokenizer(prompt_prefix + prompt, return_tensors='pt').input_ids
            output = target_model.generate(target_input_ids, max_new_tokens=max_new_token_length)
            prediction_file.write(target_tokenizer.decode(output[0], skip_special_tokens=True))

            baseline_input_ids = baseline_tokenizer(prompt_prefix + prompt, return_tensors='pt').input_ids
            baseline_output = baseline_model.generate(baseline_input_ids, return_tensors='pt')
            baseline_prediction_file.write(baseline_tokenizer.decode(baseline_output[0], skip_special_tokens=True))

elif test_type == TestType.SANITY:
    for test_example in test_examples:
        input_ids = target_tokenizer(prompt_prefix + test_example, return_tensors='pt').input_ids
        output = target_model.generate(input_ids, max_new_tokens=max_new_token_length)
        print("Target model output :", target_tokenizer.decode(output[0], skip_special_tokens=True))
        baseline_input_ids = baseline_tokenizer(prompt_prefix + test_example, return_tensors='pt').input_ids
        baseline_output = baseline_model.generate(baseline_input_ids, return_tensors='pt')
        print("Baseline model output :", baseline_tokenizer.decode(output[0], skip_special_tokens=True))

