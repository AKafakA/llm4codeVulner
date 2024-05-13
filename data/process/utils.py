import json


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

                        # If no next prompt or if "\n-" occurs before "\n \n", set label_end to prompt_end
                        if label_end == -1 or (
                                change["diff"].find("\n \n", prompt_end) < change["diff"].find("\n-", prompt_end)):
                            label_end = change["diff"].find("\n \n", prompt_end)

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


def get_filename_from_patch(repo_name, file_path, ref, truncated_in_hash=5):
    file_name = file_path.split("/")[-1]
    return repo_name.replace("/", "_") + "-" + ref[:truncated_in_hash] + "-" + file_name
