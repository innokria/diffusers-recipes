import torch
from diffusers import ZImagePipeline


pipe = ZImagePipeline.from_pretrained(
    "Tongyi-MAI/Z-Image-Turbo",
    torch_dtype=torch.bfloat16,
    device_map="cuda",
)

pipe.load_lora_weights(
    "renderartist/Technically-Color-Z-Image-Turbo",
    weight_name="Technically_Color_Z_Image_Turbo_v1_renderartist_2000.safetensors",  # if the repo only has one lora weight file, this argument can be omitted
    adapter_name="color",  # if you're not planning to use set_adapters later, this argument can be omitted
)

pipe.set_adapters("color", 0.7)  # set the adapter to use with a specific weight if needed

prompt = "t3chnic4lly classic cinema film still, a close-up portrait of Santa Claus's face, his kind, crinkled eyes sparkling with mirth and wisdom. His magnificent white beard and mustache are neatly groomed, framing a gentle smile, and the rich texture of his red velvet hat, with its distinct, slightly pointed shape and single white pom-pom, is visible, pulled slightly over his brow. His spectacles are round and wire-rimmed, resting low on his nose. The lighting emphasizes the warmth and depth of his character, conveying a sense of 1950s magic and profound goodwill, depicted with a soft, diffused focus and rich, inviting tones. 1960s style film "

image = pipe(
    prompt=prompt,
    height=1024,
    width=1024,
    num_inference_steps=9,
    guidance_scale=0.0,
    generator=torch.Generator("cuda").manual_seed(42),
).images[0]

image.save("zimage_output.png")
