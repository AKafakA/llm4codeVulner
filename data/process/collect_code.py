import base64
import sys
import subprocess
import ast

from data.process.utils import read_patches, process_source_code, get_filename_from_patch
from github import Github, UnknownObjectException
from github import Auth


auth = Auth.Token(sys.argv[0])
g = Github(auth=auth)

vulnerability = "plain_sql"
data_file = "data/{}.json".format(vulnerability)
saved_buggy_file = "data/buggy_files/" + vulnerability + "/code/"
numb_patches = 10

commits_file_name = "data/buggy_files/" + vulnerability + "/commits.txt"

records = read_patches(data_file)
total_patches = len(records)
train_and_valid_ratio = 0.8
test_record = records[int(total_patches * train_and_valid_ratio):]

num_available_files = 0
for record in test_record:
    with open(commits_file_name, "a+") as commits_file:
        try:
            repo_name = record["repo"]
            repo = g.get_repo(repo_name)
            for commit_record in record["commits"]:
                commit_sha = commit_record["commit_hash"]
                commits = repo.get_commits(commit_sha)
                for file_record in commit_record["files"]:
                    file_path = file_record["file_name"]
                    file = repo.get_contents(path=file_path, ref=commits[1].sha)
                    file_data = base64.b64decode(file.content)
                    output_file_name = saved_buggy_file + get_filename_from_patch(repo_name, file_path, commit_sha)
                    file = open(output_file_name, "w+")
                    processed_sourced = process_source_code(file_data.decode("utf-8"))
                    try:
                        ast.parse(processed_sourced)
                        file.write(processed_sourced)
                        commits_file.write(commit_sha + "\n")
                        num_available_files += 1
                    except SyntaxError:
                        valid = False
                    finally:
                        file.close()
        except UnknownObjectException:
            print("Could not find by record {}".format(record["repo"]))


print("Number of available records: {}".format(num_available_files))
result = subprocess.run(["bandit", "-r", saved_buggy_file], capture_output=True, text=True)
print(result.stdout.split("Run metrics:")[-1])



