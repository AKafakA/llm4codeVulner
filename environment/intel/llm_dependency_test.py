import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

############# code changes ###############
import intel_extension_for_pytorch as ipex
# verify Intel Arc GPU
print(torch.__version__)
print(ipex.__version__)
[print(f'[{i}]: {torch.xpu.get_device_properties(i)}') for i in range(torch.xpu.device_count())]
##########################################
model_id = "alpindale/WizardLM-2-8x22B"

model = AutoModelForCausalLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

model = model.eval().to("xpu")

with torch.inference_mode(), torch.no_grad(), torch.autocast(
        ############# code changes ###############
        device_type="xpu",
        ##########################################
        enabled=True
):
    text = "You may have heard of Schrodinger cat mentioned in a thought experiment in quantum physics. Briefly, according to the Copenhagen interpretation of quantum mechanics, the cat in a sealed box is simultaneously alive and dead until we open the box and observe the cat. The macrostate of cat (either alive or dead) is determined at the moment we observe the cat."
    input_ids = tokenizer(text, return_tensors="pt").input_ids
    ############# code changes ###############
    # move to Intel Arc GPU
    input_ids = input_ids.to("xpu")
    ##########################################
    generated_ids = model.generate(input_ids, max_new_tokens=128)[0]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

print(generated_text)
