import argparse
import os
import shutil
import subprocess

import torch
from transformers import AutoTokenizer
from utils import (ModelType, get_model, get_prompt_prefix,
                   generate_and_write_fixed_code, get_model_type)
from data.process.utils import read_patches, get_filename_from_patch, download_vulnerable_file, get_github_client


def main():
    parser = argparse.ArgumentParser(
        prog='Static Analysis Scripts',
        description='Static Analysis Scripts to automatically generate and write fixed code '
                    'along with tested by bandit tools.',
        epilog=''
    )

    parser.add_argument('-v', '--vulnerability', type=str, default='sql_injection',
                        help='Vulnerability type need to be repaired, default is sql_injection')
    parser.add_argument('-l', '--lang', type=str, default='python',
                        help='programming language need to be repaired, default is python')
    parser.add_argument('-n', '--num_test', type=int, default=50,
                        help='number of tested repo')
    parser.add_argument('-m', '--model_name', type=str, default='Salesforce/codet5-small',
                        help='model need to be trained, default is Salesforce/codet5-small')
    parser.add_argument('-t', '--model_type', type=str, default='t5',
                        help='model type needed to be tested or trained. '
                             'It will used to initialized tokenizer from huggingface, default is t5. Use causal for '
                             'casualLM')
    parser.add_argument('--model_tuned', type=bool, default=False,
                        help='Check if the model is tuned or using the raw models from huggingface')
    parser.add_argument('--model_path', type=str,
                        help='The path of the trained model. If not provided, it will assume stored '
                             'under the default output directory of the training scripts')
    parser.add_argument('--train_ratio', type=float, default=0.6,
                        help='Indicate the usage of selected data should be used for training, default is 0.6.')
    parser.add_argument('--val_ratio', type=float, default=0.2,
                        help='Indicate the usage of selected data should be used for validation, default is 0.2.')
    parser.add_argument('--token', type=str, required=True,
                        help='Personal github token to download files')

    args = parser.parse_args()

    #  Tunable Parameter
    vulnerability = args.vulnerability
    lang = args.lang
    num_tests = args.num_test
    train_and_valid_ratio = args.train_ratio + args.val_ratio
    target_model_name = args.model_name
    model_type = get_model_type(args.model_type)
    token = args.token
    github_client = get_github_client(token)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    prompt_prefix = get_prompt_prefix(vulnerability, lang)
    target_tokenizer = AutoTokenizer.from_pretrained(target_model_name)
    target_tokenizer.pad_token = target_tokenizer.eos_token
    save_directory = None
    if args.model_tuned:
        if args.model_path is None:
            save_directory = "llm/models/{}".format(vulnerability + "-" + target_model_name)
        else:
            save_directory = args.model_path
    target_model = get_model(target_model_name, model_type, save_path=save_directory, device=device)
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

    records = read_patches(prompt_data_file)
    total_patches = len(records)
    start_patch_index = int(total_patches * train_and_valid_ratio)
    test_records = records[start_patch_index:min(start_patch_index + num_tests, total_patches)]
    num_processed_files = 0
    for record in test_records:
        repo_name = record["repo"]
        file_downloaded = download_vulnerable_file(record, github_client, saved_buggy_files_path,
                                                   commit_truncated_number=5)
        if not file_downloaded:
            continue
        for commit_record in record["commits"]:
            commit_hash = commit_record["commit_hash"]
            for file_record in commit_record["files"]:
                prompts = file_record["prompts"]
                labels = file_record["labels"]
                file_name = file_record["file_name"]
                full_filename = get_filename_from_patch(repo_name, file_name, commit_hash, 5)
                input_file_name = saved_buggy_files_code_path + full_filename
                target_file_name = target_fix_path + full_filename
                if not os.path.exists(input_file_name):
                    print("File {} not found".format(input_file_name))
                    continue
                num_processed_files += 1
                with (open(input_file_name, "r") as f,
                      open(target_file_name, "w+") as target):
                    source_code = f.read()
                    target_code = generate_and_write_fixed_code(target_model, source_code, target_tokenizer,
                                                                prompt_prefix, prompts, device=device)
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
    runnable_commits_file = target_fix_path + "runnable_full_file_names.txt"
    runnable_files = set()
    with open(runnable_commits_file, "a+") as f:
        for file in os.listdir(target_fix_path):
            if not os.path.isdir(os.path.join(target_fix_path, file)):
                if file.endswith('.py') or file.endswith('.tpl'):
                    shutil.move(target_fix_path + file, no_error_files_path + file)
                    runnable_files.add(file)
                    num_runnable_files += 1
                else:
                    os.remove(target_fix_path + file)
        for runnable_file in runnable_files:
            f.write(runnable_file + "\n")

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


if __name__ == "__main__":
    main()