import os
import json
import torch
import numpy as np
from PIL import Image
import cv2
from diffusers import UniPCMultistepScheduler
from torchvision.transforms import ToPILImage
from torchvision.transforms import ToTensor

NODE_ROOT = os.path.dirname(os.path.abspath(__file__))
position = NODE_ROOT.find("custom_nodes")
STORYMAKER_ROOT = os.path.join(NODE_ROOT[:position], 'models/StoryMaker')

# 默认配置
DEFAULT_CONFIG = {
    "face_adapter_path": os.path.join(STORYMAKER_ROOT, "checkpoints/mask.bin"),
    "image_encoder_path": os.path.join(STORYMAKER_ROOT, "models/CLIP-ViT-H-14-laion2B-s32B-b79K"),
    "base_model_path": os.path.join(STORYMAKER_ROOT, "models/YamerMIX_v11")
}


def load_config():
    return DEFAULT_CONFIG


# 加载配置
CONFIG = load_config()


class StoryMakerSharedResources:
    def __init__(self):
        self.app = None
        self.pipe = None

    def initialize(self):
        from insightface.app import FaceAnalysis
        from .StoryMaker.pipeline_sdxl_storymaker import StableDiffusionXLStoryMakerPipeline
        if self.app is None:
            self.app = FaceAnalysis(name='buffalo_l', root=STORYMAKER_ROOT,
                                    providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
            self.app.prepare(ctx_id=0, det_size=(640, 640))

        if self.pipe is None:
            self.pipe = StableDiffusionXLStoryMakerPipeline.from_pretrained(
                CONFIG['base_model_path'],
                torch_dtype=torch.float16,
            )
            self.pipe.cuda()
            self.pipe.load_storymaker_adapter(CONFIG['image_encoder_path'], CONFIG['face_adapter_path'], scale=0.8,
                                              lora_scale=0.8)
            self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)

    def deinitialize(self):
        self.pipe = None
        self.app = None


shared_resources = StoryMakerSharedResources()


class StoryMakerBaseNode:
    def __init__(self):
        self.shared = shared_resources

    def preprocess_image(self, image):
        if isinstance(image, torch.Tensor):
            # 确保我们处理的是 4D 张量 (batch, height, width, channels)
            if image.dim() != 4:
                raise ValueError(f"预期 4D 张量，但得到 {image.dim()}D 张量")

            image = image.squeeze(0).permute(2, 0, 1)  # Remove batch dimension and permute

            pil_image = ToPILImage()(image)  # Convert the tensor to a PIL Image
        elif isinstance(image, str):
            pil_image = Image.open(image)
        elif isinstance(image, Image.Image):
            pil_image = image
        else:
            raise ValueError("不支持的图像类型。期望 torch.Tensor、str 或 PIL.Image.Image")

        # 打印最终的图像尺寸
        print(f"Final image size: {pil_image}")

        # 确保返回的是 RGB 模式的 PIL 图像
        # pil_image.save('test111.jpg')
        # pil_image.convert('RGB').save('test222.jpg')
        return pil_image.convert('RGB')

    def get_face_info(self, image):
        face_info = self.shared.app.get(cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR))
        return sorted(face_info, key=lambda x: (x['bbox'][2] - x['bbox'][0]) * (x['bbox'][3] - x['bbox'][1]))[-1]


class SinglePortraitNode(StoryMakerBaseNode):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask_image": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True}),
                "negative_prompt": ("STRING", {"multiline": True}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "height": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "width": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "latent_image": ("LATENT",)
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate"
    CATEGORY = "StoryMaker"

    def generate(self, image, mask_image, prompt, negative_prompt, seed, height, width, latent_image):
        self.shared.initialize()
        image = self.preprocess_image(image)
        mask_image = self.preprocess_image(mask_image)
        face_info = self.get_face_info(image)

        generator = torch.Generator(device='cuda').manual_seed(seed)
        output = self.shared.pipe(
            image=image, mask_image=mask_image, face_info=face_info,
            prompt=prompt,
            negative_prompt=negative_prompt,
            ip_adapter_scale=0.8, lora_scale=0.8,
            num_inference_steps=25,
            guidance_scale=7.5,
            height=height, width=width,
            generator=generator,
            latents=latent_image
        ).images[0]
        processed_tensor = ToTensor()(output)  # Convert the PIL Image back to a tensor
        processed_tensor = processed_tensor.unsqueeze(0).permute(0, 2, 3,
                                                                 1)  # Put back the batch dimension and permute back to [batch_size, height, width, channels]
        self.shared.deinitialize()
        return processed_tensor,


class TwoPortraitNode(StoryMakerBaseNode):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image1": ("IMAGE",),
                "mask_image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "mask_image2": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True}),
                "negative_prompt": ("STRING", {"multiline": True}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate"
    CATEGORY = "StoryMaker"

    def generate(self, image1, mask_image1, image2, mask_image2, prompt, negative_prompt, seed):
        self.shared.initialize()
        image1 = self.preprocess_image(image1)
        mask_image1 = self.preprocess_image(mask_image1)
        image2 = self.preprocess_image(image2)
        mask_image2 = self.preprocess_image(mask_image2)
        face_info1 = self.get_face_info(image1)
        face_info2 = self.get_face_info(image2)

        generator = torch.Generator(device='cuda').manual_seed(seed)
        output = self.shared.pipe(
            image=image1, mask_image=mask_image1, face_info=face_info1,
            image_2=image2, mask_image_2=mask_image2, face_info_2=face_info2,
            prompt=prompt,
            negative_prompt=negative_prompt,
            ip_adapter_scale=0.8, lora_scale=0.8,
            num_inference_steps=25,
            guidance_scale=7.5,
            height=1280, width=960,
            generator=generator,
        ).images[0]
        processed_tensor = ToTensor()(output)  # Convert the PIL Image back to a tensor
        processed_tensor = processed_tensor.unsqueeze(0).permute(0, 2, 3,
                                                                 1)  # Put back the batch dimension and permute back to [batch_size, height, width, channels]
        self.shared.deinitialize()
        return processed_tensor,


class SwapClothNode(StoryMakerBaseNode):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask_image": ("IMAGE",),
                "cloth": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True}),
                "negative_prompt": ("STRING", {"multiline": True}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate"
    CATEGORY = "StoryMaker"

    def generate(self, image, mask_image, cloth, prompt, negative_prompt, seed):
        self.shared.initialize()
        image = self.preprocess_image(image)
        mask_image = self.preprocess_image(mask_image)
        cloth = self.preprocess_image(cloth)
        face_info = self.get_face_info(image)

        generator = torch.Generator(device='cuda').manual_seed(seed)
        output = self.shared.pipe(
            image=image, mask_image=mask_image, face_info=face_info, cloth=cloth,
            prompt=prompt,
            negative_prompt=negative_prompt,
            ip_adapter_scale=0.8, lora_scale=0.8,
            num_inference_steps=25,
            guidance_scale=7.5,
            height=1280, width=960,
            generator=generator,
        ).images[0]
        self.shared.deinitialize()
        processed_tensor = ToTensor()(output)  # Convert the PIL Image back to a tensor
        processed_tensor = processed_tensor.unsqueeze(0).permute(0, 2, 3,
                                                                 1)  # Put back the batch dimension and permute back to [batch_size, height, width, channels]
        return processed_tensor,


NODE_CLASS_MAPPINGS = {
    "StoryMakerSinglePortraitNode": SinglePortraitNode,
    "StoryMakerTwoPortraitNode": TwoPortraitNode,
    "StoryMakerSwapClothNode": SwapClothNode
}
