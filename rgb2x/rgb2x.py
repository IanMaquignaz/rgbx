import os
import argparse
from pathlib import Path
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import torch
import torchvision
from diffusers import DDIMScheduler
from load_image import load_exr_image, load_ldr_image
from pipeline_rgb2x import StableDiffusionAOVMatEstPipeline

current_directory = os.path.dirname(os.path.abspath(__file__))


def get_pipeline(disable_progress_bar=False):
    # Load pipeline
    pipe = StableDiffusionAOVMatEstPipeline.from_pretrained(
        "zheng95z/rgb-to-x",
        torch_dtype=torch.float16,
        cache_dir=os.path.join(current_directory, "model_cache"),
        device_map="balanced" if torch.cuda.is_available() else "cpu",
    )
    pipe.scheduler = DDIMScheduler.from_config(
        pipe.scheduler.config, rescale_betas_zero_snr=True, timestep_spacing="trailing"
    )
    pipe.set_progress_bar_config(disable=disable_progress_bar)
    return pipe


# Augmentation
def run_rgb2x(
    photo,
    pipe,
    aovs,
    seed=0,
    num_samples=1,
    inference_step=50,
):
    generator = torch.Generator(device="cuda").manual_seed(seed)

    print(f"Running rgb2x with photo: {photo}, seed: {seed}, num_samples: {num_samples}, inference_step: {inference_step}")
    if Path(photo).suffix == ".exr":
        photo = load_exr_image(photo, tonemapping=True, clamp=True).to("cuda")
    elif Path(photo).suffix in [".png", ".jpg", ".jpeg"]:
        photo = load_ldr_image(photo, from_srgb=True).to("cuda")

    # Check if the width and height are multiples of 8. If not, crop it using torchvision.transforms.CenterCrop
    old_height = photo.shape[1]
    old_width = photo.shape[2]
    new_height = old_height
    new_width = old_width
    radio = old_height / old_width
    max_size = 1000
    if old_height > old_width:
        new_height = max_size
        new_width = int(new_height / radio)
    else:
        new_width = max_size
        new_height = int(new_width * radio)

    if new_width % 8 != 0 or new_height % 8 != 0:
        new_width = new_width // 8 * 8
        new_height = new_height // 8 * 8

    photo = torchvision.transforms.Resize((new_height, new_width))(photo)

    return_list = []
    for i in range(num_samples):
        for aov_name in aovs.keys():
            generated_image = pipe(
                prompt=aovs[aov_name],
                photo=photo,
                num_inference_steps=inference_step,
                height=new_height,
                width=new_width,
                generator=generator,
                required_aovs=[aov_name],
            ).images[0][0]

            generated_image = torchvision.transforms.Resize(
                (old_height, old_width)
            )(generated_image)

            generated_image = (generated_image, f"Generated {aov_name} {i}")
            return_list.append(generated_image)

    return return_list


def trigger_save_all(save_dir, return_list):
    #generated_image = (generated_image, f"Generated {aov_name} {i}")
    status = f"Error? Nothing saved, check {save_dir}"
    save_dir = str(save_dir).strip()
    os.makedirs(save_dir, exist_ok=True)
    for image, label in return_list:
        label = label.split("Generated ")[-1].replace(" ", "_")+".png"
        image.save(os.path.join(save_dir, label))
    status = "Images saved"
    return status

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="rgb2x")

    # Target lighting – exactly one should be specified
    p.add_argument("--img",      default=None, help="Target image path (.exr, .hdr, .png, …)")
    p.add_argument("--save_dir",      default="./generated", help="Directory to save generated images")
    p.add_argument("--seed",                type=int,   default=0)
    p.add_argument("--inference_step",       type=int,   default=50)
    p.add_argument("-s","--num_samples",          type=int,   default=1)
    p.add_argument("--device",             type=str,   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument('-n','--normal', action='store_true', default=False, help='Whether to render normals.')
    p.add_argument('-a','--albedo', action='store_true', default=False, help='Whether to render albedo.')
    p.add_argument('-r','--roughness', action='store_true', default=False, help='Whether to render roughness.')
    p.add_argument('-m','--metallic', action='store_true', default=False, help='Whether to render metallic.')
    p.add_argument('-i','--irradiance', action='store_true', default=False, help='Whether to render irradiance.')

    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    args.device = torch.device(args.device)

    pipeline = get_pipeline()

    aovs = {}
    if args.normal:
        aovs["normal"] = "Camera-space Normal"
    if args.albedo:
        aovs["albedo"] = "Albedo (diffuse basecolor)"
    if args.roughness:
        aovs["roughness"] = "Roughness"
    if args.metallic:
        aovs["metallic"] = "Metallicness"
    if args.irradiance:
        aovs["irradiance"] = "Irradiance (diffuse lighting)"

    return_list = run_rgb2x(
        photo=args.img,
        pipe=pipeline,
        seed=args.seed,
        num_samples=args.num_samples,
        inference_step=args.inference_step,
        aovs=aovs,
    )

    trigger_save_all(
        save_dir=args.save_dir,
        return_list=return_list,
    )
