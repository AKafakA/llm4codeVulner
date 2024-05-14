import os
import shutil
import subprocess
import sys
from enum import Enum

from github import Auth, Github
from transformers import AutoTokenizer
from utils import (max_new_token_length, ModelType, get_model, get_prompt_prefix,
                   generate_and_write_fixed_code, apply_label, generate_and_write_fixed_code_without_prompts)
from data.process.utils import read_patches, get_filename_from_patch, download_vulnerable_file


#  Tunable Parameter
vulnerability = "plain_sql"
lang = 'python'
num_tests = 50
train_and_valid_ratio = 0.8
target_model_name = "Salesforce/codet5-small"
model_type = ModelType.T5_CONDITIONAL_GENERATION
# End of Parameters

prompt_prefix = get_prompt_prefix(vulnerability, lang)
target_tokenizer = AutoTokenizer.from_pretrained(target_model_name)
# save_directory = None
save_directory = "llm/models/{}".format(vulnerability + "-" + target_model_name)
target_model = get_model(target_model_name, model_type, save_path=save_directory)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
prompt_data_file = "data/{}.json".format(vulnerability)
saved_buggy_files_path = "data/buggy_files/" + vulnerability + '/'
saved_buggy_files_code_path = "data/buggy_files/" + vulnerability + '/code/'

if os.path.exists(saved_buggy_files_path):
    shutil.rmtree(saved_buggy_files_path)
    os.makedirs(saved_buggy_files_path)

fixed_directory = "data/test/static_check/" + vulnerability + '/'
target_fix_path = fixed_directory + "target_fix/"

if os.path.exists(target_fix_path):
    shutil.rmtree(target_fix_path)
    os.makedirs(target_fix_path)

github_token = sys.argv[1]
auth = Auth.Token(github_token)
github_client = Github(auth=auth)

records = read_patches(prompt_data_file)
total_patches = len(records)
start_patch_index = int(total_patches * train_and_valid_ratio)
test_records = records[start_patch_index:min(start_patch_index + num_tests, total_patches)]
num_processed_files = 0
for record in test_records:
    repo_name = record["repo"]
    file_downloaded = download_vulnerable_file(record, github_client, saved_buggy_files_path)
    if not file_downloaded:
        continue
    for commit_record in record["commits"]:
        commit_hash = commit_record["commit_hash"]
        for file_record in commit_record["files"]:
            prompts = file_record["prompts"]
            labels = file_record["labels"]
            file_name = file_record["file_name"]
            input_file_name = saved_buggy_files_code_path + get_filename_from_patch(repo_name, file_name,
                                                                                    commit_hash)
            target_file_name = target_fix_path + get_filename_from_patch(repo_name, file_name, commit_hash)
            if not os.path.exists(input_file_name):
                print("File {} not found".format(input_file_name))
                continue
            num_processed_files += 1
            with (open(input_file_name, "r") as f,
                  open(target_file_name, "w+") as target):
                source_code = f.read()
                target_code = generate_and_write_fixed_code(target_model, source_code, target_tokenizer,
                                                            prompt_prefix, prompts)
                target.write(target_code)

result = subprocess.run(["bandit", "-r", target_fix_path], capture_output=True, text=True)

# Organize the files
skipped_files = result.stdout.split("Files skipped")[1].strip().split("\n")[1:]
syntax_error_files_path = target_fix_path + "fix_with_syntax_error/"

original_syntax_error_files_path = saved_buggy_files_code_path + "fix_with_syntax_error/"
if not os.path.exists(syntax_error_files_path):
    os.mkdir(syntax_error_files_path)

if not os.path.exists(original_syntax_error_files_path):
    os.mkdir(original_syntax_error_files_path)

for skipped_file in skipped_files:
    skipped_file_path = skipped_file.strip('\t').split('(syntax error while parsing AST from file)')[0]
    file_name = skipped_file_path.split('/')[-1]
    suffix_index = file_name.find('.py')
    if suffix_index != -1:
        file_name = file_name[:suffix_index + len('.py')]
        shutil.move(target_fix_path + file_name, syntax_error_files_path + file_name)
        shutil.move(saved_buggy_files_code_path + file_name, original_syntax_error_files_path + file_name)

no_error_files_path = target_fix_path + "fix_with_no_error/"
if not os.path.exists(no_error_files_path):
    os.mkdir(no_error_files_path)

num_runnable_files = 0
for file in os.listdir(target_fix_path):
    if file.endswith(".py"):
        shutil.move(target_fix_path + file, no_error_files_path + file)
        num_runnable_files += 1

original_no_error_files_path = saved_buggy_files_code_path + "fix_with_no_error/"
if not os.path.exists(original_no_error_files_path):
    os.mkdir(original_no_error_files_path)

for file in os.listdir(saved_buggy_files_code_path):
    if file.endswith(".py"):
        shutil.move(saved_buggy_files_code_path + file, original_no_error_files_path + file)

print("************ Report for Model {}, Fine Tuned: {} ****************".
      format(target_model_name, save_directory is not None))

print("*****************************************************************")

result = subprocess.run(["bandit", "-r", original_no_error_files_path], capture_output=True, text=True)
print("static analysis check for runnable results without fix: " + result.stdout.split("Run metrics:")[-1])

print("******************************************************************")

result = subprocess.run(["bandit", "-r", no_error_files_path], capture_output=True, text=True)
print("static analysis check for runnable results with fix: " + result.stdout.split("Run metrics:")[-1])

print("*******************************************************************")
print("Processed {} files with {} is runnable".format(num_processed_files, num_runnable_files))

