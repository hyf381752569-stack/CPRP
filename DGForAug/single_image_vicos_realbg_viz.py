#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
单图片版本的“生成式前景重组”工作流，基于 generate_vicos_realbg.py 的三步逻辑：
  1) 加载预处理得到的干净背景 & 前景资产
  2) 用 ControlNet Tile 对干净背景进行风格化重绘
  3) 将风格化背景与预处理前景进行羽化合成

并额外生成一个带有流程图的可视化，把每一步的输入/输出图片缩略图放到流程图节点上。

依赖：torch, diffusers, PIL, numpy, cv2, matplotlib
示例运行：
  python single_image_vicos_realbg_viz.py \
    --base_model_path /path/to/base_model \
    --controlnet_tile_path /path/to/controlnet_tile \
    --vae_path /path/to/vae \
    --input_rgb /path/to/rgb.jpg \
    --input_mask /path/to/mask.png \
    --input_clean_bg /path/to/clean_bg.jpg \
    --output_dir /tmp/out_single \
    --steps 30 --strength 1.0 --controlnet_scale 1.0 --seed 123

注意：本脚本默认使用 CUDA。如果没有 GPU，可把 --device cpu，但耗时会明显更长。
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import cv2
from PIL import Image
import torch
from diffusers import (
    StableDiffusionControlNetInpaintPipeline,
    ControlNetModel,
    UniPCMultistepScheduler,
    AutoencoderKL,
)

# 可视化流程图使用 matplotlib 来画箭头和放置图片
import matplotlib
matplotlib.use("Agg")  # 服务器/脚本环境下避免需要 GUI
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib import font_manager as fm

# 优先设置支持中文的字体，避免中文乱码
def _setup_chinese_font():
    candidates = [
        # Noto / Source Han 系列
        "Noto Sans CJK SC", "Noto Sans CJK", "Noto Serif CJK SC",
        "Source Han Sans SC", "Source Han Serif SC",
        # 常见中文字体（Linux/Win）
        "WenQuanYi Zen Hei", "AR PL UKai CN", "AR PL UMing CN",
        "SimHei", "Microsoft YaHei",
        # 兜底
        "DejaVu Sans",
    ]
    # 允许环境变量覆盖字体名
    env_font = os.environ.get("MPL_CHINESE_FONT", "").strip()
    if env_font:
        candidates.insert(0, env_font)

    # 在系统字体中查找第一个可用字体
    available = {f.name for f in fm.fontManager.ttflist}
    chosen = None
    for name in candidates:
        if name in available:
            chosen = name
            break

    if chosen:
        matplotlib.rcParams['font.family'] = 'sans-serif'
        matplotlib.rcParams['font.sans-serif'] = [chosen]
        matplotlib.rcParams['axes.unicode_minus'] = False
    else:
        # 找不到中文字体时，依然禁用负号乱码；提示一次
        matplotlib.rcParams['axes.unicode_minus'] = False
        print("[Warn] 未找到可用中文字体，可能出现中文显示不完整。可设置环境变量 MPL_CHINESE_FONT 或安装 Noto Sans CJK。")

_setup_chinese_font()


# --------- 与 generate_vicos_realbg.py 中一致/相近的提示词组件 ---------
STYLES = [
    "photorealistic", "hyperrealistic", "impressionist", "expressionist",
    "surrealist", "cubist", "abstract", "minimalist", "brutalist",
    "art nouveau", "art deco", "pop art", "steampunk", "cyberpunk",
    "synthwave", "vaporwave", "gothic", "renaissance", "baroque",
    "ukiyo-e", "chinese ink wash (shanshui)", "bauhaus", "brutalism",
    "psychedelic art", "fantasy art", "sci-fi concept art"
]
MATERIALS = [
    "brushed aluminum", "polished chrome", "anodized titanium", "rusty iron",
    "weathered copper", "raw concrete", "cracked plaster", "exposed brick",
    "terrazzo", "marble", "granite", "dark oak wood", "bamboo",
    "corkboard", "carbon fiber", "plexiclass", "holographic foil",
    "velvet fabric", "raw silk", "denim texture", "knitted wool",
    "leather", "handmade paper", "old parchment", "recycled cardboard",
    "liquid metal", "molten lava", "glowing energy field", "nebula clouds",
    "crystal formations", "geode pattern"
]
ENVIRONMENTS = [
    "a professional photography studio", "a minimalist art gallery",
    "an industrial loft", "a futuristic sci-fi laboratory",
    "a cozy wooden cabin", "a serene zen garden",
    "a bustling tokyo street at night", "a sun-drenched mediterranean patio",
    "the surface of Mars", "an underwater coral reef",
    "a grand library with old books", "an abandoned warehouse",
    "inside a spaceship cockpit", "a vibrant brazilian favela",
    "a foggy victorian street", "a clean, modern data center"
]
LIGHTING = [
    "soft studio lighting", "dramatic cinematic lighting", "volumetric light rays",
    "caustic reflections", "warm morning sunlight", "cool twilight glow",
    "neon signs reflection", "backlit", "hard shadows", "soft, diffused light",
    "glowing from within", "lens flare", "god rays filtering through clouds",
    "moonlight", "candlelight"
]
COLORS = [
    "a monochromatic palette", "a complementary color scheme",
    "an analogous color scheme", "a triadic color scheme",
    "a pastel color palette", "vibrant and saturated colors",
    "iridescent and pearlescent colors", "a gradient from deep blue to purple",
    "earthy tones", "neon cyberpunk colors", "a simple black and white",
    "a warm color palette of reds and oranges", "a cool color palette of blues and greens",
    "sepia toned"
]
ARTISTS = [
    "greg rutkowski", "alphonse mucha", "vincent van gogh", "salvador dali",
    "pablo picasso", "hokusai", "moebius", "zaha hadid", "h.r. giger",
    "james gurney", "ilya kuvshinov", "loish"
]


def generate_diverse_prompt(seed: Optional[int] = None) -> str:
    # 与原始文件一致的多样提示词生成器
    import random
    if seed is not None:
        random.seed(seed)
    prompt_templates = [
        "template_style_material",
        "template_environment_lighting",
        "template_abstract_color",
        "template_artist_style",
    ]
    chosen = random.choice(prompt_templates)
    if chosen == "template_style_material":
        return f"{random.choice(STYLES)} background, {random.choice(MATERIALS)} surface, with {random.choice(LIGHTING)}. high resolution, sharp focus, 8k"
    if chosen == "template_environment_lighting":
        return f"A background of {random.choice(ENVIRONMENTS)}, focusing on the environment, {random.choice(LIGHTING)}. photorealistic, depth of field, high detail"
    if chosen == "template_abstract_color":
        return f"{random.choice(['abstract', 'minimalist', 'geometric'])} background, {random.choice(COLORS)}, with a {random.choice(MATERIALS)} texture. modern art, clean lines"
    if chosen == "template_artist_style":
        return f"{random.choice(STYLES)} background heavily inspired by the works of {random.choice(ARTISTS)}, focusing on texture and light, masterpiece quality."
    return f"{random.choice(STYLES)}, {random.choice(MATERIALS)}, {random.choice(LIGHTING)}"


def feather_mask_edge(mask_pil: Image.Image, pixels: int) -> Image.Image:
    # 使用高斯模糊对边缘羽化，以避免合成边界的硬边
    if pixels <= 0:
        return mask_pil
    radius = pixels if pixels % 2 != 0 else pixels + 1
    blurred = cv2.GaussianBlur(np.array(mask_pil), (radius, radius), 0)
    return Image.fromarray(blurred)


def ensure_size(img: Image.Image, size: Tuple[int, int], resample=Image.LANCZOS) -> Image.Image:
    if img.size == size:
        return img
    return img.resize(size, resample)


def save_image(img: Image.Image, path: Path, quality: int = 95) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # PNG/JPG 根据后缀保存
    if path.suffix.lower() in {".png"}:
        img.save(path)
    else:
        img.convert("RGB").save(path, quality=quality)


def build_pipeline(base_model_path: str, controlnet_tile_path: str, vae_path: str, device: str):
    controlnet_tile = ControlNetModel.from_pretrained(controlnet_tile_path, torch_dtype=torch.float16 if device == "cuda" else torch.float32)
    pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
        base_model_path,
        controlnet=controlnet_tile,
        vae=AutoencoderKL.from_pretrained(vae_path, torch_dtype=torch.float16 if device == "cuda" else torch.float32),
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        safety_checker=None,
    )
    pipe = pipe.to(device)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    return pipe, controlnet_tile


def draw_flowchart(nodes, arrows, out_path: Path, title: str = "Single-Image Workflow"):
    """
    使用 matplotlib 画一个简单流程图，把缩略图放在节点上。
    nodes: list of dict with keys: id, xy(center), w, h, title, img (PIL or np.ndarray)
    arrows: list of (from_id, to_id) tuples
    """
    fig_w, fig_h = 16, 9
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title(title)

    # 预先构建 renderer，用于像素尺度换算
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    ax_bbox_px = ax.get_window_extent(renderer=renderer)
    ax_px_w, ax_px_h = ax_bbox_px.width, ax_bbox_px.height

    # 画节点框和图片
    id2node = {n['id']: n for n in nodes}
    for n in nodes:
        x, y = n['xy']
        w, h = n['w'], n['h']
        # 背景框
        patch = FancyBboxPatch(
            (x - w/2, y - h/2), w, h,
            boxstyle="round,pad=0.004,rounding_size=0.008",
            fc=(0.96, 0.96, 0.98), ec=(0.3, 0.3, 0.4), lw=1.2, alpha=0.95,
            zorder=2,
        )
        ax.add_patch(patch)

        # 标题文本
        ax.text(x, y + h/2 + 0.02, n.get('title', ''), ha='center', va='bottom', fontsize=10, zorder=3)

        # 图片缩略图
        img = n.get('img')
        if img is not None:
            if isinstance(img, Image.Image):
                arr = np.asarray(img.convert('RGB'))
            else:
                arr = img
            # 目标节点框对应的像素宽高
            node_px_w = w * ax_px_w
            node_px_h = h * ax_px_h
            # 计算缩放，使图片落在框内（留一点边距）
            zoom = min(node_px_w / arr.shape[1], node_px_h / arr.shape[0]) * 0.95
            imagebox = OffsetImage(arr, zoom=zoom)
            ab = AnnotationBbox(imagebox, (x, y), frameon=False)
            ab.set_zorder(3)
            ax.add_artist(ab)

    # 计算从矩形边缘到矩形边缘的锚点，避免箭头穿过节点
    def rect_edge_point(cx, cy, w, h, tx, ty):
        dx, dy = tx - cx, ty - cy
        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            return (cx, cy)
        # 计算到矩形边界的比例 t，取较小者
        t_vals = []
        if abs(dx) > 1e-6:
            t_vals.append((w/2) / abs(dx))
        if abs(dy) > 1e-6:
            t_vals.append((h/2) / abs(dy))
        t = min(t_vals) if t_vals else 1.0
        return (cx + dx * t, cy + dy * t)

    # 画箭头
    for u, v in arrows:
        nu, nv = id2node[u], id2node[v]
        x1, y1 = nu['xy']
        x2, y2 = nv['xy']
        # 计算边缘锚点
        p1 = rect_edge_point(x1, y1, nu['w'], nu['h'], x2, y2)
        p2 = rect_edge_point(x2, y2, nv['w'], nv['h'], x1, y1)
        # 稍微向外偏移终点，避免覆盖边框
        ex, ey = p2
        dx, dy = ex - x2, ey - y2
        if abs(dx) + abs(dy) > 1e-6:
            p2 = (ex - dx * 0.02, ey - dy * 0.02)

        arrow = FancyArrowPatch(
            p1, p2, arrowstyle='->',
            connectionstyle='arc3,rad=0.0',  # 直线连接，减少弯曲重叠
            mutation_scale=12, lw=1.4, color=(0.2, 0.2, 0.25), zorder=1
        )
        ax.add_patch(arrow)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="单图片工作流（含流程图可视化），基于 generate_vicos_realbg.py")
    # 模型与资源
    parser.add_argument("--base_model_path", type=str, required=True)
    parser.add_argument("--controlnet_tile_path", type=str, required=True)
    parser.add_argument("--vae_path", type=str, required=True)
    # 单张图输入
    parser.add_argument("--input_rgb", type=str, required=True)
    parser.add_argument("--input_foreground", type=str, required=True, help="预处理的前景RGBA文件")
    parser.add_argument("--input_mask", type=str, default=None, help="原始二值前景Mask，可选")
    parser.add_argument("--input_clean_bg", type=str, required=True)
    # 输出
    parser.add_argument("--output_dir", type=str, required=True)

    # 推理/控制参数（与批处理版对齐）
    parser.add_argument("--edge_feather_pixels", type=int, default=7)
    parser.add_argument("--controlnet_scale", type=float, default=1.0)
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--negative_prompt", type=str, default="ugly, deformed, disfigured, poor details, bad anatomy, worst quality, low quality, blurry, noisy, text, watermark, signature, cloth, fabric")
    parser.add_argument("--seed", type=int, default=None, help="风格随机种子（影响 prompt 和生成器）。")
    parser.add_argument("--prompt", type=str, default=None, help="可选：手动指定背景风格化 prompt；不填则随机。")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="推理设备。默认 cuda。")

    args = parser.parse_args()

    device = args.device
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------- 读取输入图像并对齐尺寸 -------
    rgb = Image.open(args.input_rgb).convert("RGB")
    W, H = rgb.size
    foreground_rgba = Image.open(args.input_foreground).convert("RGBA").resize((W, H), Image.LANCZOS)
    if args.input_mask:
        mask = Image.open(args.input_mask).convert("L").resize((W, H), Image.NEAREST)
    else:
        mask = foreground_rgba.split()[-1]
    clean_bg = Image.open(args.input_clean_bg).convert("RGB").resize((W, H), Image.LANCZOS)

    # 保存原始输入，便于流程图引用
    save_image(rgb, out_dir / "step1_input_rgb.jpg")
    save_image(mask.convert("L"), out_dir / "step1_input_mask.png")
    save_image(clean_bg, out_dir / "step1_input_clean_bg.jpg")
    save_image(foreground_rgba, out_dir / "step1_input_foreground.png")
    rgb_rgba = rgb.convert("RGBA")
    alpha = np.array(rgb_rgba.split()[-1], dtype=np.uint8)
    mask_np = np.array(mask)
    alpha[mask_np > 0] = 0
    background_removed = rgb_rgba.copy()
    background_removed.putalpha(Image.fromarray(alpha, mode="L"))
    save_image(background_removed, out_dir / "step1_input_background_removed.png")

    # ------- 步骤 1：资产准备（干净背景 + 前景） -------

    # ------- 步骤 2：风格化背景（ControlNet Tile + Inpaint 全局蒙版） -------
    # 准备 prompt/种子
    if args.prompt is not None:
        prompt = args.prompt
    else:
        prompt = generate_diverse_prompt(seed=args.seed)
    generator = None
    if args.seed is not None:
        generator = torch.Generator(device=device).manual_seed(int(args.seed))

    # 全白蒙版：让管线重绘整幅图
    full_white_mask = Image.new("L", (W, H), 255)
    save_image(full_white_mask, out_dir / "step2_full_white_mask.png")

    # 构建/加载 pipeline
    pipe, _ = build_pipeline(
        base_model_path=args.base_model_path,
        controlnet_tile_path=args.controlnet_tile_path,
        vae_path=args.vae_path,
        device=device,
    )

    # 生成
    result = pipe(
        prompt=prompt,
        negative_prompt=args.negative_prompt,
        image=clean_bg,
        mask_image=full_white_mask,
        control_image=clean_bg,
        controlnet_conditioning_scale=float(args.controlnet_scale),
        num_inference_steps=int(args.steps),
        strength=float(args.strength),
        generator=generator,
        height=H,
        width=W,
    )
    stylized_bg = result.images[0]
    save_image(stylized_bg, out_dir / "step2_stylized_background.jpg")

    # ------- 步骤 3：最终合成（较小羽化） -------
    final_mask = feather_mask_edge(mask, args.edge_feather_pixels)
    save_image(final_mask.convert("L"), out_dir / "step3_final_mask.png")
    fg_alpha = foreground_rgba.split()[-1]
    fg_alpha = feather_mask_edge(fg_alpha, args.edge_feather_pixels)
    foreground_rgba.putalpha(fg_alpha)
    stylized_bg_rgba = stylized_bg if stylized_bg.size == (W, H) else stylized_bg.resize((W, H), Image.LANCZOS)
    stylized_bg_rgba = stylized_bg_rgba.convert("RGBA")
    stylized_bg_rgba.alpha_composite(foreground_rgba)
    final_image = stylized_bg_rgba.convert("RGB")
    save_image(final_image, out_dir / "step3_final_image.jpg")

    # ------- 绘制流程图（把每步输入/输出放到节点上） -------
    # 缩略图片读取
    def load_thumb(p: Path, max_side: int = 512) -> Image.Image:
        im = Image.open(p).convert("RGB")
        w, h = im.size
        s = max(w, h)
        if s > max_side:
            scale = max_side / s
            im = im.resize((int(w*scale), int(h*scale)), Image.LANCZOS)
        return im

    n_rgb = load_thumb(out_dir / "step1_input_rgb.jpg")
    n_mask = Image.open(out_dir / "step1_input_mask.png").convert("RGB")
    n_clean = load_thumb(out_dir / "step1_input_clean_bg.jpg")
    n_fg = Image.open(out_dir / "step1_input_foreground.png").convert("RGB")
    n_fullmask = Image.open(out_dir / "step2_full_white_mask.png").convert("RGB")
    n_styled = load_thumb(out_dir / "step2_stylized_background.jpg")
    n_finalmask = Image.open(out_dir / "step3_final_mask.png").convert("RGB")
    n_final = load_thumb(out_dir / "step3_final_image.jpg")

    # 节点布局（0-1 归一化坐标）
    nodes = [
        {"id": "rgb",    "xy": (0.10, 0.80), "w": 0.20, "h": 0.22, "title": "原图 RGB", "img": n_rgb},
        {"id": "mask",   "xy": (0.10, 0.50), "w": 0.20, "h": 0.22, "title": "前景 Mask/Alpha", "img": n_mask},
        {"id": "clean",  "xy": (0.10, 0.20), "w": 0.20, "h": 0.22, "title": "干净背景", "img": n_clean},
        {"id": "fg",     "xy": (0.38, 0.80), "w": 0.20, "h": 0.22, "title": "预处理前景", "img": n_fg},

        {"id": "fullm",  "xy": (0.64, 0.50), "w": 0.20, "h": 0.22, "title": "全白蒙版（步骤2）", "img": n_fullmask},
        {"id": "styled",  "xy": (0.64, 0.80), "w": 0.20, "h": 0.22, "title": "步骤2：风格化背景", "img": n_styled},

        {"id": "fmask",  "xy": (0.64, 0.20), "w": 0.20, "h": 0.22, "title": "最终羽化蒙版（步骤3）", "img": n_finalmask},
        {"id": "final",  "xy": (0.88, 0.75), "w": 0.20, "h": 0.22, "title": "步骤3：最终合成", "img": n_final},
    ]

    arrows = [
        ("rgb", "styled"), ("clean", "styled"), ("fullm", "styled"),
        ("fg", "final"), ("styled", "final"), ("fmask", "final"),
    ]

    flow_path = out_dir / "workflow_flowchart.png"
    draw_flowchart(
        nodes,
        arrows,
        flow_path,
        title=f"Single-Image Workflow | steps={args.steps}, scale={args.controlnet_scale}, strength={args.strength}\nPrompt: {prompt}",
    )

    print("\nDone. Outputs saved to:", out_dir)
    print("- step1_input_rgb.jpg, step1_input_mask.png, step1_input_clean_bg.jpg, step1_input_foreground.png, step1_input_background_removed.png")
    print("- step2_full_white_mask.png, step2_stylized_background.jpg")
    print("- step3_final_mask.png, step3_final_image.jpg")
    print("- workflow_flowchart.png (流程图)")


if __name__ == "__main__":
    main()
