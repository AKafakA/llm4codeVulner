import os
import shutil
import subprocess
import sys

from github import Auth, Github

from data.process.utils import read_patches, download_vulnerable_files


vulnerability = "plain_sql"
data_file = "data/{}.json".format(vulnerability)
output_path = "data/buggy_files/" + vulnerability

github_token = sys.argv[1]
auth = Auth.Token(github_token)
g = Github(auth=auth)

records = read_patches(data_file)
total_patches = len(records)
train_and_valid_ratio = 0.8
numb_patches = 10
test_start_point = int(total_patches * train_and_valid_ratio)
test_records = records[test_start_point:test_start_point + numb_patches]

num_available_commits = download_vulnerable_files(test_records, output_path, g)

print("Number of available records: {}".format(num_available_commits))
result = subprocess.run(["bandit", "-r", output_path + "/code/"], capture_output=True, text=True)
print(result.stdout.split("Run metrics:")[-1])

shutil.rmtree(output_path)



