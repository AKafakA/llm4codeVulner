Forked from this project https://github.com/LauraWartschinski/VulnerabilityDetection which using the lstm to detect the potential Vulnerability

And plan to apply the llm to generate accordingly fix.

To start the llm prompt fine tuning with the given data sql buggy data

```
git clone https://github.com/AKafakA/llm4codeVulner.git
cd llm4codeVulner/
conda env create -f environment/nvidia/environment.yml
conda activate llm
python3 llm/train.py
```
GPU/TPU is not required for the testing examples

If want to test with model which require access, please apply it at hugging face and then login huggine face by

```
huggingface-cli login
```

In case if setting the python environment path is required, please also run 
```
export PYTHONPATH=$PYTHONPATH:/[home]/[username]/llm4codeVulner
```
