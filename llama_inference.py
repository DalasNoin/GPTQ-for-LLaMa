import argparse

import torch
import torch.nn as nn
import quant
import os

from gptq import GPTQ
from utils import find_layers, DEV, set_seed, get_wikitext2, get_ptb, get_c4, get_ptb_new, get_c4_new, get_loaders
import transformers
from transformers import AutoTokenizer


def get_llama(model):

    def skip(*args, **kwargs):
        pass

    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    from transformers import LlamaForCausalLM
    model = LlamaForCausalLM.from_pretrained(model, torch_dtype='auto')
    model.seqlen = 2048
    return model


def load_quant(model, checkpoint, wbits, groupsize=-1, fused_mlp=True, eval=True, warmup_autotune=True):
    from transformers import LlamaConfig, LlamaForCausalLM
    config = LlamaConfig.from_pretrained(model)

    def noop(*args, **kwargs):
        pass

    torch.nn.init.kaiming_uniform_ = noop
    torch.nn.init.uniform_ = noop
    torch.nn.init.normal_ = noop

    torch.set_default_dtype(torch.half)
    transformers.modeling_utils._init_weights = False
    torch.set_default_dtype(torch.half)
    model = LlamaForCausalLM(config)
    torch.set_default_dtype(torch.float)
    if eval:
        model = model.eval()
    layers = find_layers(model)
    for name in ['lm_head']:
        if name in layers:
            del layers[name]
    quant.make_quant_linear(model, layers, wbits, groupsize)

    del layers

    print('Loading model ...')
    if checkpoint.endswith('.safetensors'):
        from safetensors.torch import load_file as safe_load
        model.load_state_dict(safe_load(checkpoint), strict=False)
    else:
        model.load_state_dict(torch.load(checkpoint), strict=False)

    if eval:
        quant.make_quant_attn(model)
        quant.make_quant_norm(model)
        if fused_mlp:
            quant.make_fused_mlp(model)
    if warmup_autotune:
        quant.autotune_warmup_linear(model, transpose=not (eval))
        if eval and fused_mlp:
            quant.autotune_warmup_fused(model)
    model.seqlen = 2048
    print('Done.')

    return model

class LlamaWrapper:
    def __init__(self, model, checkpoint, wbits, groupsize=-1, fused_mlp=True):
        if type(checkpoint) is not str:
            checkpoint = checkpoint.as_posix()

        if checkpoint:
            self.model = load_quant(model=model, 
                                checkpoint=checkpoint, 
                                wbits=wbits, 
                                groupsize=groupsize,
                                fused_mlp=fused_mlp)
        else:
            self.model = get_llama(model)
            self.model.eval()
        model.to(DEV)
        self.tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)

    def __call__(self, text):
        input_ids = self.tokenizer.encode(text, return_tensors="pt").to(DEV)
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids,
                do_sample=True,
                min_length=10,
                max_length=50,
                top_p=0.95,
                temperature=0.8,
            )
        return self.tokenizer.decode([el.item() for el in generated_ids[0]])


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('model', type=str, help='llama model to load')
    parser.add_argument('--wbits', type=int, default=16, choices=[2, 3, 4, 8, 16], help='#bits to use for quantization; use 16 for evaluating base model.')
    parser.add_argument('--groupsize', type=int, default=-1, help='Groupsize to use for quantization; default uses full row.')
    parser.add_argument('--load', type=str, default='', help='Load quantized model checkpoint.')

    parser.add_argument('--text', type=str, help='input text', default='')
    # as an alternative take a textfile as input
    parser.add_argument('--textfile', type=str, help='input textfile')
    # offer an output file to write to
    parser.add_argument('--outputfile', type=str, help='output textfile')
    # offer a n parameter for the number of completions
    parser.add_argument('--n', type=int, default=1, help='number of completions to generate')

    parser.add_argument('--min_length', type=int, default=10, help='The minimum length of the sequence to be generated.')

    parser.add_argument('--max_length', type=int, default=50, help='The maximum length of the sequence to be generated.')

    parser.add_argument('--top_p',
                        type=float,
                        default=0.95,
                        help='If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.')

    parser.add_argument('--temperature', type=float, default=0.8, help='The value used to module the next token probabilities.')

    parser.add_argument('--device', type=int, default=-1, help='The device used to load the model when using safetensors. Default device is "cpu" or specify, 0,1,2,3,... for GPU device.')

    # fused mlp is sometimes not working with safetensors, no_fused_mlp is used to set fused_mlp to False, default is true
    parser.add_argument('--fused_mlp', action='store_true')
    parser.add_argument('--no_fused_mlp', dest='fused_mlp', action='store_false')
    parser.set_defaults(fused_mlp=True)

    args = parser.parse_args()



    # check if textfile is given or text
    if args.text:
        text = args.text
    elif args.textfile:
        # check if textfile exists
        if not os.path.isfile(args.textfile):
            raise ValueError('Textfile does not exist.')
        with open(args.textfile, 'r') as f:
            text = f.read()
    else:
        raise ValueError('No text or textfile given.')
    
    # check if outputfile is given
    if args.outputfile:
        outputfile = args.outputfile

    # create the llama wrapper
    llama_wrapper = LLamaWrapper(model=args.model,
                                 checkpoint=args.load,
                                 wbits=args.wbits,
                                 groupsize=args.groupsize,
                                 fused_mlp=args.fused_mlp)
    
    # generate the text
    for i in range(args.n):
        generated_text = llama_wrapper(text)
        # only use text after the input text
        generated_text = generated_text[len(text):]
        print(generated_text)
        if args.outputfile:
            with open(outputfile, 'a') as f:
                f.write(generated_text)
                f.write('\n')
    