import torch
import argparse
import os
import json
import time
import random
from pathlib import Path
from tqdm.auto import tqdm
from diffusers import StableDiffusionPipeline, DDIMScheduler
from diffusers.utils import logging

from util import fix_seed, multidict
from model import SafeRedir, register_guided_redirector_hooks

logging.set_verbosity_error()  # Suppress warnings and progress output

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(args):
    # Load Stable Diffusion
    if args.model == "sd14":
        model_id = "CompVis/stable-diffusion-v1-4"
    elif args.model == "sd15":
        model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    else:
        raise ValueError
    sd_pipe = StableDiffusionPipeline.from_pretrained(model_id).to(device)
    sd_pipe.safety_checker = None

    # Replace default scheduler with DDIM
    sd_pipe.scheduler = DDIMScheduler.from_config(sd_pipe.scheduler.config)
    sd_pipe.scheduler.set_timesteps(args.ddim_steps)
    sd_pipe.set_progress_bar_config(disable=True)


    # Load redirector
    # if args.task =="Nudity":
    #     task="NSFW"
    # else:
    #     task = args.task

    redirector_ckpt = Path("ckpt") / args.task / "best_model.pt"
    redirector = SafeRedir().to(device)
    redirector.load_state_dict(torch.load(redirector_ckpt, map_location=device))
    redirector.eval()

    # Register hooks
    register_guided_redirector_hooks(sd_pipe, redirector,
                                     replace_steps=args.replace_step,
                                     alpha_scale=args.alpha_scale)

    return sd_pipe


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--task', type=str, default="Nudity", choices=["Nudity"],
                        help='task name, e.g., Nudity')
    parser.add_argument('--gen-type', type=str, required=True,
                        help='generation mode: retain | forgot')
    parser.add_argument('--save_dir', type=str, default="gen_imgs",
                        help='base directory to save generated images')
    parser.add_argument('--ddim_steps', type=int, default=50,
                        help='number of DDIM steps (e.g., 50 or 1000)')
    parser.add_argument('--gen_nums', type=int, default=5,
                        help='number of images to generate per prompt')
    parser.add_argument('--guidance_scale', type=float, default=7.5,
                        help='guidance scale for classifier-free guidance')
    parser.add_argument('--height', type=int, default=512,
                        help='image height')
    parser.add_argument('--width', type=int, default=512,
                        help='image width')
    parser.add_argument('--replace_step', type=int, default=2,
                        help='number of steps for redirector to intervene')
    parser.add_argument('--alpha_scale', type=float, default=1.5,
                        help='scaling factor for redirector delta strength')
    parser.add_argument('--model', type=float, default="sd14",choices=["sd14","sd15"],
                        help='base model')

    args = parser.parse_args()

    # Set random seed
    fix_seed(2024)
    args.seeds = random.sample(range(0, 65536), 1000)[:args.gen_nums]

    # Load model
    pipe = load_model(args)

    # Load prompts
    prompt_file = Path("data") / f"IGMU_{args.gen_type}.json"
    with open(prompt_file, 'r') as f:
        prompts = json.load(f)

    save_path = Path(args.save_dir) / args.gen_type / args.task
    save_path.mkdir(parents=True, exist_ok=True)

    print(f"Seeds: {args.seeds}")
    start_time = time.time()

    with torch.no_grad():
        img_gen_classes = list(prompts[args.task].keys())

        for class_idx, class_name in enumerate(img_gen_classes):
            class_prompts = prompts[args.task][class_name]

            for prompt_idx, prompt_text in enumerate(tqdm(class_prompts, desc=f"{args.gen_type} - {class_name}", disable=False)):
                prompt_list = [prompt_text]

                for img_idx in range(args.gen_nums):
                    seed = args.seeds[img_idx]
                    generator = torch.manual_seed(seed)
                    file_path = save_path / f"{class_idx}_{prompt_idx}_{img_idx}.png"

                    if file_path.exists():
                        continue

                    image = pipe(prompt=prompt_list,
                                 num_inference_steps=args.ddim_steps,
                                 generator=generator,
                                 guidance_scale=args.guidance_scale,
                                 height=args.height,
                                 width=args.width).images[0]

                    image.save(file_path)

    # Report time
    elapsed_time = time.time() - start_time
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)

    print(f"[{args.task}] Generation completed in {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}")


# python safe_generate.py --gen-type forgot --task Nudity --ddim_steps 50 --gen_nums 5 --guidance_scale 7.5 --replace_step 2 --alpha_scale 1.0 --save_dir gen_imgs