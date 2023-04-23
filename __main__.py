import os
from typing import Any, List

import torch
from diffusers import StableDiffusionPipeline

import chat

# Use the DPMSolverMultistepScheduler (DPM-Solver++) scheduler here instead
generator = StableDiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1", torch_dtype=torch.float16
)
assert isinstance(generator, StableDiffusionPipeline)
generator = generator.to("cuda")


theme = input("Story Theme: ")
story_dir = input("Directory ID: ")
output_dir = f"./output/{story_dir}"
os.makedirs(output_dir, exist_ok=True)
loop = 0
agent = chat.ChatAgent(theme)
chat_log = agent.chat_log
req = ""
while True:
    # Text response
    print(chat_log.entries[-1].message)

    # Image
    res: Any = generator(chat_log.entries[-1].message)
    image = res.images[0]
    image.save(f"{output_dir}/{loop}.png")

    # Dump story
    with open(f"{output_dir}/story.txt", "w") as f:
        f.write(str(chat_log))

    # Next
    req = input("What do you do?: ")
    if req.strip().lower() in ("quit", "exit"):
        break
    chat_log = agent.input(req)
    loop += 1
