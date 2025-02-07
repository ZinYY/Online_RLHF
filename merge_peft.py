# merge a PEFT model to a full model:

import argparse

from transformers import AutoTokenizer
from peft import AutoPeftModelForCausalLM


def merge_peft(peft_path, output_path):
    peft_model = AutoPeftModelForCausalLM.from_pretrained(peft_path)
    tokenizer = AutoTokenizer.from_pretrained(peft_path)
    print(type(peft_model))
    
    merged_model = peft_model.merge_and_unload()
    print(type(merged_model))
    # save:
    merged_model.save_pretrained(output_path)
    # save tokenizer
    tokenizer.save_pretrained(output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Merge a PEFT model to a full model')
    parser.add_argument('--peft_path', type=str, help='PEFT model file')
    args = parser.parse_args()
    # args.peft_path = './checkpoint/llama-3-8b-rlhf_lora16'
    
    output_path = args.peft_path + '_merged'
    merge_peft(args.peft_path, output_path)
