import argparse
import random
import torch

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path

from PIL import Image
from pathlib import Path

from datetime import datetime
import os
import json
from tqdm import tqdm

VISION_ENCODER_SIZE = (1024, 600)

PROMPT = """
INSERT YOUR PROMPT HERE. Refer to the appendix in the paper for example prompts.
"""
        
def load_image(image_file):
    image = Image.open(image_file).convert('RGB')
    return image.resize(VISION_ENCODER_SIZE)

def main(args):
    # Model
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, args.load_8bit, args.load_4bit, device=args.device)

    # conversation mode before the passed argument is analyzed
    if 'llama-2' in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        # if the passed conv_mode does not match the inferred conv_mode, print a warning & keep using the args.conv_mode later in the script
        print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, args.conv_mode, args.conv_mode))
    else:
        # if passed conv_mode is None, use the inferred conv_mode
        args.conv_mode = conv_mode

    # Get a list of all image file paths in the parent directory, 
    image_files_paths = []
    for root, dirs, files in os.walk(args.images_folder):
        for file in files:
            # make sure the file is an image
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')):
                image_files_paths.append(os.path.join(root, file))
    
    # shuffle the image file paths once
    if args.shuffle_image_paths == 'y':
        random.shuffle(image_files_paths)

    #    Prepare the output file
    # ------------------------------------------------------------------------ #
    
    # set the output file folder
    OUTPUT_FILE_FOLDER = '/home/username/Code/LLaVA/results/'
    # print the output file folder if trial
    print("Setting output file folder as:", OUTPUT_FILE_FOLDER) if args.trial == 'y' else None
    
    # create the output file name and the file's full path
    output_file_name = 'output_' + datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + '.jsonl'
    output_file_path = os.path.join(OUTPUT_FILE_FOLDER, output_file_name)
    
    #    Define and add to the log file
    # ------------------------------------------------------------------------ #
    log_file_path = os.path.join(OUTPUT_FILE_FOLDER, 'experimental_log.jsonl')
    # create the log file if it does not exist
    if not os.path.exists(log_file_path):
        with open(log_file_path, 'w') as f:
            f.write(json.dumps({"log_file_path": log_file_path}) + '\n')

    # write information to the log file
    with open(log_file_path, 'a') as f:
            f.write(json.dumps({
                "trial": args.trial,
                "experiment_description": args.experiment_description,
                "model_name": model_name,
                "images_folder": args.images_folder,
                "conv_mode": args.conv_mode,
                "temperature": str(args.temperature),
                "max_new_tokens": str(args.max_new_tokens),
                "VISION_ENCODER_SIZE": str(VISION_ENCODER_SIZE),
                "image_aspect_ratio": args.image_aspect_ratio,
                "shuffle_image_paths": str(args.shuffle_image_paths),
                "output_file_path": output_file_path,
                "log_file_path": log_file_path,
                "PROMPT": PROMPT
            }) + '\n')

    for image_path in tqdm(image_files_paths) if args.trial == 'n' else image_files_paths[:5]:

        conv = conv_templates[args.conv_mode].copy()

        # print the image path
        print("Reading file:", image_path) if args.trial == 'y' else None

        image = load_image(image_path)

        # even though the process_images function is for multiple images, we will use it for a single image here
        image_tensor = process_images([image], image_processor, args)
        if type(image_tensor) is list:
            image_tensor = [image.to(model.device, dtype=torch.float16) for image in image_tensor]
        else:
            image_tensor = image_tensor.to(model.device, dtype=torch.float16)

        if image is not None:
            # first message
            if model.config.mm_use_im_start_end:
                inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + PROMPT
            else:
                inp = DEFAULT_IMAGE_TOKEN + '\n' + PROMPT
            conv.append_message(conv.roles[0], inp)
            image = None
        else:
            # later messages
            conv.append_message(conv.roles[0], PROMPT)
        
        conv.append_message(conv.roles[1], None)
        
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                do_sample=True,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
                )

        outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()

        # write the outputs to the file
        with open(output_file_path, 'a') as f:
            f.write(json.dumps({"image_path": image_path,
                                # strip the </s> from the output when writing to the file
                                "output": outputs.strip("</s>")}) + '\n')

        if args.debug:
            print("\n", {"prompt": prompt, "outputs": outputs}, "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/home/username/Models/llava-v1.5-13b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--images-folder", type=str, required=True) # can provide
    parser.add_argument("--device", type=str, default="cuda") # can provide
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.2) # can provide
    parser.add_argument("--max-new-tokens", type=int, default=256) # can provide
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--image-aspect-ratio", type=str, default='resize')
    parser.add_argument("--shuffle-image-paths", type=str, default='y')
    parser.add_argument("--experiment-description", type=str, default="")
    parser.add_argument("--trial", type=str, default="n")
    args = parser.parse_args()
    main(args)