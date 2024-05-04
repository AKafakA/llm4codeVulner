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

This train scripts will read the github patchs and train on the given llm models from hugging face, save the model and print the evalution metrics (code bert score and code bleu score) on test data.

And the inference.py allow user to load the trained model from training scripts and produce the repaired code 

If want to test with model which require access, please apply it at hugging face and then login huggine face by

```
huggingface-cli login
```

In case if setting the python environment path is required, please also run 
```
export PYTHONPATH=$PYTHONPATH:/[home]/[username]/llm4codeVulner
```
