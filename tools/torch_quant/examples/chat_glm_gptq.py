
import logging

from torch import nn
from torch_quant.experiment import GPTQuantizer
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, set_seed

logging.basicConfig(level=logging.DEBUG)

from transformers.dynamic_module_utils import get_class_from_dynamic_module


def do_inference_with_fixed_seed(model, tokenizer, prompt):
    set_seed(42)
    response, _ = model.chat(tokenizer, prompt, history=[])
    return response


prompt = "晚上睡不着应该怎么办"

# If use local mode, change it to the folder that contains model files
target_model = "chatglm_6b"

# The basic block within which all layers are calibrated together
target_block = get_class_from_dynamic_module(target_model, "modeling_chatglm.py", "GLMBlock")

tokenizer = AutoTokenizer.from_pretrained(target_model, trust_remote_code=True)
model = AutoModel.from_pretrained(target_model, trust_remote_code=True, resume_download=True).half().cuda()

# Get the output of the original model
response = do_inference_with_fixed_seed(model, tokenizer, prompt)
print(response)

# Get the GPTQuantizer, the block means that the GLMBlock is calibrated one-by-one
# and the last lm_head (of type nn.Linear) is calibrated alone
quantizer = GPTQuantizer(block=[target_block, nn.Linear])

# prepare the model for the quantization process
calib_model = quantizer.calib(model)

# Since we do not get the graph of the model (e.g. torchscript, fx graph), we must
# do inference on the model once and record the block order
with quantizer.record_order():
    do_inference_with_fixed_seed(model, tokenizer, prompt)

# Do calibration on the model. In each iter, one block will be quantized using GPTQ and
# you can use other prompts.
for i in tqdm(range(quantizer.calibration_iters)):
    with quantizer.start_calib_iter(i):
        response, history = model.chat(tokenizer, prompt, history=[])

# Get the result of the weight fake-quantized model
response = do_inference_with_fixed_seed(model, tokenizer, prompt)
print(response)