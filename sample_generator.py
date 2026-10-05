"""
Generates synthetic realistic retinal fundus images for immediate demonstration and testing.
- Normal Retina (Non-DR)
- Diabetic Retinopathy Retina (with microaneurysms, hemorrhages, and exudates)
"""

import os
import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def create_retinal_samples(output_dir="samples"):
    os.makedirs(output_dir, exist_ok=True)
    width, height = 512, 512
    center_x, center_y = width // 2, height // 2
    radius = int(width * 0.44)

    # ----------------------------------------------------
    # 1. Base Fundus Disc Generator
    # ----------------------------------------------------
    def make_base_fundus():
        img = Image.new("RGB", (width, height), (10, 10, 15))
        draw = ImageDraw.Draw(img)

        # Create radial gradient for the retinal fundus
        y, x = np.ogrid[:height, :width]
        dist_from_center = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)

        # Retinal orange/red coloration
        r = np.clip(220 - dist_from_center * 0.25 + np.random.normal(0, 3, (height, width)), 0, 255)
        g = np.clip(110 - dist_from_center * 0.28 + np.random.normal(0, 2, (height, width)), 0, 255)
        b = np.clip(45 - dist_from_center * 0.15 + np.random.normal(0, 2, (height, width)), 0, 255)

        base_arr = np.stack([r, g, b], axis=-1).astype(np.uint8)
        base_img = Image.fromarray(base_arr)

        # Create circular mask for fundus aperture
        mask = Image.new("L", (width, height), 0)
        mask_draw = ImageDraw.Draw(mask)
        mask_draw.ellipse(
            (center_x - radius, center_y - radius, center_x + radius, center_y + radius),
            fill=255
        )
        mask = mask.filter(ImageFilter.GaussianBlur(radius=3))

        final_img = Image.composite(base_img, img, mask)
        draw = ImageDraw.Draw(final_img)

        # Draw Optic Disc (bright yellowish oval)
        optic_x = center_x - int(radius * 0.45)
        optic_y = center_y - int(radius * 0.1)
        draw.ellipse(
            (optic_x - 30, optic_y - 36, optic_x + 30, optic_y + 36),
            fill=(255, 235, 170)
        )
        draw.ellipse(
            (optic_x - 18, optic_y - 22, optic_x + 18, optic_y + 22),
            fill=(255, 248, 210)
        )

        # Draw Macula / Fovea (dark reddish subtle depression)
        macula_x = center_x + int(radius * 0.22)
        macula_y = center_y + 5
        draw.ellipse(
            (macula_x - 28, macula_y - 28, macula_x + 28, macula_y + 28),
            fill=(140, 50, 20)
        )

        # Draw Retinal Blood Vessels radiating from Optic Disc
        vessel_color = (120, 25, 15)
        vessel_branches = [
            [(optic_x, optic_y), (optic_x + 50, optic_y - 80), (optic_x + 130, optic_y - 140), (optic_x + 200, optic_y - 170)],
            [(optic_x, optic_y), (optic_x + 40, optic_y + 80), (optic_x + 120, optic_y + 140), (optic_x + 210, optic_y + 160)],
            [(optic_x, optic_y), (optic_x - 40, optic_y - 70), (optic_x - 90, optic_y - 130)],
            [(optic_x, optic_y), (optic_x - 35, optic_y + 70), (optic_x - 85, optic_y + 130)],
            [(optic_x + 50, optic_y - 80), (optic_x + 90, optic_y - 60), (optic_x + 160, optic_y - 40)],
            [(optic_x + 40, optic_y + 80), (optic_x + 85, optic_y + 60), (optic_x + 150, optic_y + 45)],
        ]
        for branch in vessel_branches:
            for w_line in [5, 3, 2]:
                draw.line(branch, fill=vessel_color, width=w_line, joint="curve")

        return final_img

    # ----------------------------------------------------
    # Generate Normal Retina
    # ----------------------------------------------------
    normal_retina = make_base_fundus()
    normal_retina = normal_retina.filter(ImageFilter.GaussianBlur(radius=0.7))
    normal_path = os.path.join(output_dir, "sample_normal_retina.png")
    normal_retina.save(normal_path)
    print(f"Generated: {normal_path}")

    # ----------------------------------------------------
    # Generate Diabetic Retinopathy Retina
    # (Adds microaneurysms, flame hemorrhages, hard exudates)
    # ----------------------------------------------------
    dr_retina = make_base_fundus()
    dr_draw = ImageDraw.Draw(dr_retina)

    np.random.seed(99)
    # Microaneurysms & dot hemorrhages (deep red / crimson spots)
    for _ in range(45):
        spot_x = np.random.randint(center_x - radius + 40, center_x + radius - 40)
        spot_y = np.random.randint(center_y - radius + 40, center_y + radius - 40)
        if np.sqrt((spot_x - center_x) ** 2 + (spot_y - center_y) ** 2) < radius - 20:
            sz = np.random.randint(2, 7)
            dr_draw.ellipse(
                (spot_x - sz, spot_y - sz, spot_x + sz, spot_y + sz),
                fill=(90, 8, 8)
            )

    # Larger blot hemorrhages
    for _ in range(12):
        spot_x = np.random.randint(center_x - int(radius*0.6), center_x + int(radius*0.6))
        spot_y = np.random.randint(center_y - int(radius*0.6), center_y + int(radius*0.6))
        sz_w = np.random.randint(6, 14)
        sz_h = np.random.randint(4, 10)
        dr_draw.ellipse(
            (spot_x - sz_w, spot_y - sz_h, spot_x + sz_w, spot_y + sz_h),
            fill=(100, 10, 10)
        )

    # Hard Exudates (bright yellowish lipid deposits)
    for _ in range(25):
        ex_x = np.random.randint(center_x, center_x + int(radius*0.7))
        ex_y = np.random.randint(center_y - int(radius*0.5), center_y + int(radius*0.5))
        sz = np.random.randint(2, 6)
        dr_draw.ellipse(
            (ex_x - sz, ex_y - sz, ex_x + sz, ex_y + sz),
            fill=(255, 240, 120)
        )

    # Soft cotton-wool spots (fluffy white/pale patches)
    for _ in range(4):
        cw_x = np.random.randint(center_x - 80, center_x + 120)
        cw_y = np.random.randint(center_y - 100, center_y + 100)
        dr_draw.ellipse(
            (cw_x - 12, cw_y - 10, cw_x + 12, cw_y + 10),
            fill=(240, 230, 210)
        )

    dr_retina = dr_retina.filter(ImageFilter.GaussianBlur(radius=0.6))
    dr_path = os.path.join(output_dir, "sample_dr_retina.png")
    dr_retina.save(dr_path)
    print(f"Generated: {dr_path}")

    return normal_path, dr_path


if __name__ == "__main__":
    create_retinal_samples()
