import ast
import base64
import json
import os

from github import UnknownObjectException


def read_patches(filename):
    with open(filename) as file:
        data_str = file.read()  # Read the content of the file as a string
        data = json.loads(data_str)  # Parse the JSON string
    collections = []
    for repo_url, commits_info in data.items():
        record = {"repo": repo_url.split("https://github.com/")[-1]}
        collections.append(record)
        commits = []
        record["commits"] = commits
        for commit_hash, commit_info in commits_info.items():
            commit_record = {"commit_hash": commit_hash}
            commits.append(commit_record)

            file_records = []
            commit_record["files"] = file_records

            # Check if the 'files' key exists in the commit_info
            for file_path, file_info in commit_info["files"].items():
                file_record = {"file_name": file_path}
                file_records.append(file_record)
                prompts = []
                labels = []
                file_record["prompts"] = prompts
                file_record["labels"] = labels
                for change in file_info["changes"]:
                    # Split the file content based on prompts and labels
                    prompt_start = change["diff"].find("\n-")  # Find the start of the first prompt
                    while prompt_start != -1:
                        prompt_end = change["diff"].find("\n+", prompt_start)  # Find the end of the current prompt
                        label_end = change["diff"].find("\n-", prompt_end)  # Find the start of the next prompt

                        # If no next prompt or if "\n-" occurs before "\n ", set label_end to prompt_end
                        if label_end == -1 or (
                                change["diff"].find("\n ", prompt_end) < label_end):
                            label_end = change["diff"].find("\n ", prompt_end)

                        prompt = change["diff"][prompt_start + len("\n-"):prompt_end].strip().replace("\n-",
                                                                                                      "\n")  # Extract and clean the prompt
                        label = change["diff"][prompt_end + len("\n+"):label_end].strip().replace("\n+",
                                                                                                  "\n")  # Extract and clean the label

                        prompts.append(prompt)
                        labels.append(label)

                        prompt_start = change["diff"].find("\n-", label_end)  # Find the start of the next prompt
    return collections


def read_prompts(filename):
    records = read_patches(filename)
    prompts = []
    labels = []
    for record in records:
        for commit_record in record["commits"]:
            for file_record in commit_record["files"]:
                prompts.extend(file_record["prompts"])
                labels.extend(file_record["labels"])
    return prompts, labels


def process_source_code(code):
    return code.replace("{{/*", "'''").replace("*/}}", "'''")


def get_filename_from_patch(repo_name, file_path, ref, ref_truncated_number=5):
    file_name = file_path.split("/")[-1]
    return ref[:ref_truncated_number] + "-" + repo_name.replace("/", "_") + file_name


def download_vulnerable_files(patch_records, output_path, github_client):
    available_commits = 0
    for record in patch_records:
        if download_vulnerable_file(record, github_client, output_path):
            available_commits += 1
    return available_commits


def download_vulnerable_file(patch_record, github_client, output_path, commit_filter=None, write_commits=True,
                             commit_truncated_number=5):
    code_path = output_path + "/code/"
    if not os.path.exists(code_path):
        os.makedirs(os.path.join(code_path))
    valid = False
    commits_file = None
    if write_commits:
        commits_file_path = os.path.join(output_path, "commits.txt")
        commits_file = open(commits_file_path, "a+")
    try:
        repo_name = patch_record["repo"]
        repo = github_client.get_repo(repo_name)
        for commit_record in patch_record["commits"]:
            commit_sha = commit_record["commit_hash"]
            if commit_filter and (commit_sha not in commit_filter):
                continue
            commits = repo.get_commits(commit_sha)
            for file_record in commit_record["files"]:
                file_path = file_record["file_name"]
                git_file = repo.get_contents(path=file_path, ref=commits[1].sha)
                file_data = base64.b64decode(git_file.content)
                processed_sourced = process_source_code(file_data.decode("utf-8"))
                try:
                    ast.parse(processed_sourced)
                    if len(processed_sourced) > 0:
                        output_file_name = code_path + get_filename_from_patch(repo_name, file_path, commit_sha,
                                                                               commit_truncated_number)
                        file = open(output_file_name, "w+")
                        file.write(processed_sourced)
                        file.close()
                        if write_commits:
                            commits_file.write(commit_sha + "\n")
                            valid = True
                except SyntaxError:
                    valid = False
    except UnknownObjectException:
        valid = False
        print("Could not find by repo {}".format(patch_record["repo"]))
    return valid
