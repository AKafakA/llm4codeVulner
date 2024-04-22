Forked from this project https://github.com/LauraWartschinski/VulnerabilityDetection which using the lstm to detect the potential Vulnerability

And plan to apply the llm to generate accordingly fix.

To start the llm prompt fine tuning with the given data sql buggy data

```
conda env create -f environment/nvidia/environment.yml
cd llm
python3 train_example.py
```
GPU/TPU is not required for the testing examples

