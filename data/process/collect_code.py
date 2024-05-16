import shutil
import subprocess

from data.process.utils import read_patches, download_vulnerable_files, get_github_client


vulnerability = "plain_sql"
data_file = "data/{}.json".format(vulnerability)
output_path = "data/buggy_files/" + vulnerability

token = ""
g = get_github_client(token)

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



