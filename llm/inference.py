import os
import subprocess
from enum import Enum
from transformers import AutoTokenizer
from utils import (max_new_token_length, ModelType, get_model, get_prompt_prefix)
from data.process.utils import read_patches, get_filename_from_patch

vulnerability = "plain_sql"
lang = 'python'
prompt_prefix = get_prompt_prefix(vulnerability, lang)

target_model_name = "Salesforce/codet5-small"
model_type = ModelType.T5_CONDITIONAL_GENERATION
target_tokenizer = AutoTokenizer.from_pretrained(target_model_name)
# save_directory = None
save_directory = "llm/models/{}".format(vulnerability + "-" + target_model_name)
target_model = get_model(target_model_name, model_type, save_path=save_directory)

# baseline_model_name = target_model_name
baseline_model_name = "google/codegemma-2b"
if not baseline_model_name == target_model_name:
    baseline_tokenizer = AutoTokenizer.from_pretrained(baseline_model_name)
else:
    baseline_tokenizer = target_tokenizer
baseline_model = get_model(baseline_model_name, model_type)


class TestType(Enum):
    BATCH = "batch"
    SANITY = "sanity"
    STATIC_ANALYSIS = "static_analysis"


test_type = TestType.STATIC_ANALYSIS

################################################################
# Setting for batch check
prompt_path = "data/test/{}/prompts.txt".format(vulnerability + "-" + target_model_name)
prediction_path = "data/test/{}/prediction.txt".format(vulnerability)
baseline_prediction_path = "data/test/{}/baseline_prediction.txt".format(vulnerability)

###############################################################
# Setting for sanity check
test_example = ""
# test_example = "WHERE parent_id IN ({list_root_ids}"

############################################################
# Setting for static check
prompt_data_file = "data/{}.json".format(vulnerability)
saved_buggy_file = "data/buggy_files/" + vulnerability + "/code/"
commits_file_name = "data/buggy_files/" + vulnerability + "/commits.txt"

fixed_directory = "data/test/static_check/" + vulnerability + "/"
target_fix_path = fixed_directory + "/target_fix/"
baseline_fix_path = fixed_directory + "/baseline_fix/"

train_and_valid_ratio = 0.8

if test_type == TestType.STATIC_ANALYSIS:
    if not os.path.exists(target_fix_path):
        os.makedirs(target_fix_path)
    if not os.path.exists(baseline_fix_path):
        os.makedirs(baseline_fix_path)

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
    input_ids = target_tokenizer(prompt_prefix + test_example, return_tensors='pt').input_ids
    output = target_model.generate(input_ids, max_new_tokens=max_new_token_length)
    print("Target model output :", target_tokenizer.decode(output[0], skip_special_tokens=True))
    baseline_input_ids = baseline_tokenizer(prompt_prefix + test_example, return_tensors='pt').input_ids
    baseline_output = baseline_model.generate(baseline_input_ids, return_tensors='pt')
    print("Baseline model output :", baseline_tokenizer.decode(output[0], skip_special_tokens=True))

elif test_type == TestType.STATIC_ANALYSIS:
    records = read_patches(prompt_data_file)
    total_patches = len(records)
    test_record = records[int(total_patches * train_and_valid_ratio):]
    available_commits = open(commits_file_name).readlines()
    for record in test_record:
        repo_name = record["repo"]
        for commit_record in record["commits"]:
            commit_hash = commit_record["commit_hash"]
            for file_record in commit_record["files"]:
                prompts = file_record["prompts"]
                file_name = file_record["file_name"]
                input_file_name = saved_buggy_file + get_filename_from_patch(repo_name, file_name, commit_hash)
                target_file_name = baseline_fix_path + get_filename_from_patch(repo_name, file_name, commit_hash)
                baseline_file_name = baseline_fix_path + get_filename_from_patch(repo_name, file_name, commit_hash)
                if not os.path.exists(input_file_name):
                    print("File {} not found".format(input_file_name))
                    continue

                with (open(input_file_name, "r") as f, open(target_file_name, "w+") as target,
                      open(baseline_file_name, "w+") as baseline):
                    source_codes = f.read()
                    baseline_codes = source_codes
                    target_codes = source_codes

                    for prompt in prompts:
                        input_ids = target_tokenizer(prompt_prefix + prompt, return_tensors='pt').input_ids
                        output = target_model.generate(input_ids, max_new_tokens=max_new_token_length)
                        target_output_code = target_tokenizer.decode(output[0], skip_special_tokens=True)

                        baseline_input_ids = baseline_tokenizer(prompt_prefix + prompt, return_tensors='pt').input_ids
                        baseline_output = baseline_model.generate(input_ids, max_new_tokens=max_new_token_length)
                        baseline_output_code = baseline_tokenizer.decode(baseline_output[0], skip_special_tokens=True)

                        baseline_code = baseline_code.replace(prompt, baseline_output_code)
                        target_codes = target_codes.replace(prompt, target_output_code)

                    baseline.write(baseline_codes)
                    target.write(target_codes)

    result = subprocess.run(["bandit", "-r", target_fix_path], capture_output=True, text=True)
    print("static analysis check for target mode: " + result.stdout.split("Run metrics:")[-1])

    result = subprocess.run(["bandit", "-r", baseline_fix_path], capture_output=True, text=True)
    print("static analysis check for baseline mode: " + result.stdout.split("Run metrics:")[-1])
