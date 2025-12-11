# --- START OF FILE generate_vicos_real_final_workflow.py ---

import torch
import argparse
from pathlib import Path
from PIL import Image
import numpy as np
import cv2
from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel, UniPCMultistepScheduler, AutoencoderKL
import os
from tqdm import tqdm
import random
from typing import Optional

# --- [无变化] Prompt生成引擎和图像处理辅助函数 ---
STYLES = ["photorealistic", "hyperrealistic", "impressionist", "expressionist", "surrealist", "cubist", "abstract", "minimalist", "brutalist", "art nouveau", "art deco", "pop art", "steampunk", "cyberpunk", "synthwave", "vaporwave", "gothic", "renaissance", "baroque", "ukiyo-e", "chinese ink wash (shanshui)", "bauhaus", "brutalism", "psychedelic art", "fantasy art", "sci-fi concept art"]
MATERIALS = ["brushed aluminum", "polished chrome", "anodized titanium", "rusty iron", "weathered copper", "raw concrete", "cracked plaster", "exposed brick", "terrazzo", "marble", "granite", "dark oak wood", "bamboo", "corkboard", "carbon fiber", "plexiclass", "holographic foil", "velvet fabric", "raw silk", "denim texture", "knitted wool", "leather", "handmade paper", "old parchment", "recycled cardboard", "liquid metal", "molten lava", "glowing energy field", "nebula clouds", "crystal formations", "geode pattern"]
ENVIRONMENTS = ["a professional photography studio", "a minimalist art gallery", "an industrial loft", "a futuristic sci-fi laboratory", "a cozy wooden cabin", "a serene zen garden", "a bustling tokyo street at night", "a sun-drenched mediterranean patio", "the surface of Mars", "an underwater coral reef", "a grand library with old books", "an abandoned warehouse", "inside a spaceship cockpit", "a vibrant brazilian favela", "a foggy victorian street", "a clean, modern data center"]
LIGHTING = ["soft studio lighting", "dramatic cinematic lighting", "volumetric light rays", "caustic reflections", "warm morning sunlight", "cool twilight glow", "neon signs reflection", "backlit", "hard shadows", "soft, diffused light", "glowing from within", "lens flare", "god rays filtering through clouds", "moonlight", "candlelight"]
COLORS = ["a monochromatic palette", "a complementary color scheme", "an analogous color scheme", "a triadic color scheme", "a pastel color palette", "vibrant and saturated colors", "iridescent and pearlescent colors", "a gradient from deep blue to purple", "earthy tones", "neon cyberpunk colors", "a simple black and white", "a warm color palette of reds and oranges", "a cool color palette of blues and greens", "sepia toned"]
ARTISTS = ["greg rutkowski", "alphonse mucha", "vincent van gogh", "salvador dali", "pablo picasso", "hokusai", "moebius", "zaha hadid", "h.r. giger", "james gurney", "ilya kuvshinov", "loish"]

def generate_diverse_prompt():
    prompt_templates = ["template_style_material", "template_environment_lighting", "template_abstract_color", "template_artist_style"]
    chosen_template = random.choice(prompt_templates)
    if chosen_template == "template_style_material": return f"{random.choice(STYLES)} background, {random.choice(MATERIALS)} surface, with {random.choice(LIGHTING)}. high resolution, sharp focus, 8k"
    if chosen_template == "template_environment_lighting": return f"A background of {random.choice(ENVIRONMENTS)}, focusing on the environment, {random.choice(LIGHTING)}. photorealistic, depth of field, high detail"
    if chosen_template == "template_abstract_color": return f"{random.choice(['abstract', 'minimalist', 'geometric'])} background, {random.choice(COLORS)}, with a {random.choice(MATERIALS)} texture. modern art, clean lines"
    if chosen_template == "template_artist_style": return f"{random.choice(STYLES)} background heavily inspired by the works of {random.choice(ARTISTS)}, focusing on texture and light, masterpiece quality."
    return f"{random.choice(STYLES)}, {random.choice(MATERIALS)}, {random.choice(LIGHTING)}"

def feather_mask_edge(mask_pil, pixels):
    if pixels <= 0: return mask_pil
    radius = pixels if pixels % 2 != 0 else pixels + 1
    blurred_mask_np = cv2.GaussianBlur(np.array(mask_pil), (radius, radius), 0)
    return Image.fromarray(blurred_mask_np)


def find_matching_file(directory: Path, base_name: str, exts=(".png", ".jpg", ".jpeg", ".webp")) -> Optional[Path]:
    for ext in exts:
        candidate = directory / f"{base_name}{ext}"
        if candidate.exists():
            return candidate
    return None

def make_divisible_size(width: int, height: int, divisor: int = 8) -> tuple[int, int]:
    """Return width/height rounded up to the nearest multiple of `divisor`."""
    div = divisor
    divisible_width = ((width + div - 1) // div) * div
    divisible_height = ((height + div - 1) // div) * div
    return divisible_width, divisible_height

def main(args):
    """最终工作流：1. 加载预处理资产 -> 2. 风格化背景 -> 3. 合成最终图像"""
    # --- 1. 路径配置 ---
    base_model_path, controlnet_tile_path, vae_path = (
        args.base_model_path,
        args.controlnet_tile_path,
        args.vae_path,
    )
    input_rgb_dir = Path(args.input_rgb_dir)
    input_mask_dir = Path(args.input_mask_dir) if args.input_mask_dir else None
    input_clean_bg_dir = Path(args.input_clean_bg_dir)
    input_foreground_dir = Path(args.input_foreground_dir)
    output_dir = Path(args.output_dir)
    
    required_paths = [Path(p) for p in [base_model_path, controlnet_tile_path, vae_path]]
    required_paths += [input_rgb_dir, input_clean_bg_dir, input_foreground_dir]
    if input_mask_dir:
        required_paths.append(input_mask_dir)

    for p in required_paths:
        if not p.exists(): raise FileNotFoundError(f"关键路径不存在: {p}")
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.batch_seed is not None: random.seed(args.batch_seed)
    
    # --- 在循环外加载模型 ---
    print("--- 正在加载 ControlNet Pipeline 模型... ---")
    controlnet_tile = ControlNetModel.from_pretrained(controlnet_tile_path, torch_dtype=torch.float16)
    pipe_tile = StableDiffusionControlNetInpaintPipeline.from_pretrained(
        base_model_path, 
        controlnet=controlnet_tile, 
        vae=AutoencoderKL.from_pretrained(vae_path, torch_dtype=torch.float16),
        torch_dtype=torch.float16, 
        safety_checker=None
    ).to("cuda")
    pipe_tile.scheduler = UniPCMultistepScheduler.from_config(pipe_tile.scheduler.config)

    # --- 批量处理 ---
    rgb_files = sorted([f for f in os.listdir(input_rgb_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    processed = 0
    skipped_existing = 0

    for filename in tqdm(rgb_files, desc="图像生成 (最终工作流)"):
        # --- 准备路径和文件 ---
        base_name = os.path.splitext(filename)[0]
        input_rgb_path = input_rgb_dir / filename

        output_final_path = output_dir / f"{base_name}_general.jpg"
        if output_final_path.exists():
            skipped_existing += 1
            tqdm.write(f"[Skip] 已存在最终输出: {output_final_path.name}")
            continue

        clean_bg_path = find_matching_file(input_clean_bg_dir, base_name)
        if not clean_bg_path:
            tqdm.write(f"跳过 {base_name}: 未找到对应的干净背景文件。")
            continue

        foreground_path = find_matching_file(input_foreground_dir, base_name, exts=(".png",))
        if not foreground_path:
            tqdm.write(f"跳过 {base_name}: 未找到对应的前景RGBA文件。")
            continue

        # --- 加载和缩放图像 ---
        clean_background_image = Image.open(clean_bg_path).convert("RGB")
        W, H = clean_background_image.size
        foreground_rgba = Image.open(foreground_path).convert("RGBA").resize((W, H), Image.LANCZOS)
        pipe_W, pipe_H = make_divisible_size(W, H)
        if (pipe_W, pipe_H) != (W, H):
            tqdm.write(f"  调整尺寸满足8倍数: {W}x{H} -> {pipe_W}x{pipe_H}")
        clean_background_for_pipe = (
            clean_background_image
            if (pipe_W, pipe_H) == (W, H)
            else clean_background_image.resize((pipe_W, pipe_H), Image.LANCZOS)
        )

        style_seed = random.randint(0, 2**32 - 1)

        # ====================================================================
        # ==            步骤 2: 风格化完整的背景                         ==
        # ====================================================================
        tqdm.write(f"\n  处理 '{filename}' | 步骤 2: 背景风格化 | 种子: {style_seed}")
        random_bg_prompt = generate_diverse_prompt()
        generator_style = torch.Generator(device="cuda").manual_seed(style_seed)
        
        # 使用全白蒙版来重绘整个图像
        full_white_mask = Image.new("L", (pipe_W, pipe_H), 255)

        stylized_background = pipe_tile(
            prompt=random_bg_prompt,
            negative_prompt=args.negative_prompt,
            image=clean_background_for_pipe,        # 使用干净背景作为起始图
            mask_image=full_white_mask,
            control_image=clean_background_for_pipe,# 控制输入: 干净背景，用于保持构图
            controlnet_conditioning_scale=args.controlnet_scale,
            num_inference_steps=args.steps,
            strength=args.strength,
            generator=generator_style,
            height=pipe_H,
            width=pipe_W,
        ).images[0]
        
        # =========================================================
        # ==            步骤 3: 合成最终图像                        ==
        # =========================================================
        tqdm.write(f"  处理 '{filename}' | 步骤 3: 最终合成")
        alpha_channel = foreground_rgba.split()[-1]
        alpha_channel = feather_mask_edge(alpha_channel, args.edge_feather_pixels)
        foreground_rgba = foreground_rgba.copy()
        foreground_rgba.putalpha(alpha_channel)

        stylized_background_rgb = stylized_background.resize((W, H), Image.LANCZOS) if stylized_background.size != (W, H) else stylized_background
        final_rgba = stylized_background_rgb.convert("RGBA")
        final_rgba.alpha_composite(foreground_rgba)
        final_image = final_rgba.convert("RGB")

        final_image.save(output_final_path, quality=97)
        processed += 1

    del pipe_tile, controlnet_tile
    torch.cuda.empty_cache()
    print(
        f"\n🎉 批处理完成！生成 {processed} 张，跳过已有 {skipped_existing} 张。输出目录: {output_dir}"
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="最终工作流：通过创建无缝的完美图像输入来进行背景风格化。")
    
    # --- 核心路径参数 ---
    parser.add_argument("--base_model_path", type=str, required=True, help="基础模型的本地路径。")
    parser.add_argument("--controlnet_tile_path", type=str, required=True, help="ControlNet Tile模型的本地路径。")
    parser.add_argument("--vae_path", type=str, required=True, help="VAE的本地路径。")
    parser.add_argument("--input_rgb_dir", type=str, required=True, help="原始RGB图像文件夹。")
    parser.add_argument("--input_mask_dir", type=str, default=None, help="前景Mask文件夹（可选，如提供则用于日志与辅助羽化）。")
    parser.add_argument("--input_foreground_dir", type=str, required=True, help="预处理得到的前景RGBA文件夹。")
    parser.add_argument("--input_clean_bg_dir", type=str, required=True, help="干净背景图文件夹。")
    parser.add_argument("--output_dir", type=str, required=True, help="生成图像输出文件夹。")

    # --- 核心控制参数 ---
    parser.add_argument("--edge_feather_pixels", type=int, default=7, help="[步骤3] 最终合成时蒙版的羽化半径。")
    parser.add_argument("--controlnet_scale", type=float, default=1.0, help="[步骤2] ControlNet-Tile的权重。")
    parser.add_argument("--strength", type=float, default=1.0, help="[步骤2] 去噪强度，1.0为完全重绘。")
    
    # --- 其他质量控制参数 ---
    parser.add_argument("--steps", type=int, default=30, help="[步骤2] 推理步数。")
    parser.add_argument("--batch_seed", type=int, default=None, help="全局主种子，用于复现结果。")
    parser.add_argument("--negative_prompt", type=str, default="ugly, deformed, disfigured, poor details, bad anatomy, worst quality, low quality, blurry, noisy, text, watermark, signature, cloth, fabric", help="通用负向提示词。")
    
    args = parser.parse_args()
    main(args)
