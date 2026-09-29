from PIL import Image, ImageDraw, ImageFont
import os, math

def images_to_pdf_with_titles(
    folder_path,
    output_pdf="figures_grid.pdf",
    font_size=18,
    include_str=None,
    columns=2,
    margin=50,
    spacing=20
):
    """
    Creates a PDF with images arranged in a grid on A4 pages.
    Each image gets a title (its filename). Only images containing
    `include_str` are included (case-insensitive).
    """
    # A4 size in pixels at 300 DPI
    DPI = 300
    A4_WIDTH_PX = int(8.27 * DPI)
    A4_HEIGHT_PX = int(11.69 * DPI)

    # Collect matching images
    image_files = sorted([
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".tiff"))
        and (include_str.lower() in f.lower() if include_str else True)
    ])
    if not image_files:
        raise ValueError(f"No images found matching '{include_str}' in folder {folder_path}")

    # Load font
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        font = ImageFont.load_default()

    processed_images = []
    for path in image_files:
        img = Image.open(path).convert("RGB")
        title = os.path.basename(path)

        # Compute title height
        dummy_draw = ImageDraw.Draw(img)
        bbox = dummy_draw.textbbox((0, 0), title, font=font)
        text_h = bbox[3] - bbox[1]
        title_height = text_h + 20

        # Add title area
        new_img = Image.new("RGB", (img.width, img.height + title_height), color="white")
        new_img.paste(img, (0, title_height))

        draw = ImageDraw.Draw(new_img)
        bbox = draw.textbbox((0, 0), title, font=font)
        text_w = bbox[2] - bbox[0]
        draw.text(
            ((new_img.width - text_w) // 2, (title_height - text_h) // 2),
            title, fill="black", font=font
        )

        processed_images.append(new_img)

    # Determine max image size that fits the grid on A4
    usable_width = A4_WIDTH_PX - 2 * margin - (columns - 1) * spacing
    cell_width = usable_width // columns
    scale_ratio = cell_width / max(img.width for img in processed_images)
    scaled_height = int(max(img.height for img in processed_images) * scale_ratio)

    rows_per_page = max(1, (A4_HEIGHT_PX - 2 * margin + spacing) // (scaled_height + spacing))

    # Prepare pages
    pages = []
    page = Image.new("RGB", (A4_WIDTH_PX, A4_HEIGHT_PX), color="white")
    x, y = margin, margin
    count = 0

    for idx, img in enumerate(processed_images):
        resized = img.resize((cell_width, scaled_height))

        page.paste(resized, (x, y))
        count += 1

        if (count % columns) == 0:
            x = margin
            y += scaled_height + spacing
        else:
            x += cell_width + spacing

        # Move to next page if out of vertical space or end reached
        if (count % (columns * rows_per_page) == 0) or (idx == len(processed_images) - 1):
            pages.append(page)
            page = Image.new("RGB", (A4_WIDTH_PX, A4_HEIGHT_PX), color="white")
            x, y = margin, margin

    # Save to PDF
    pages[0].save(output_pdf, save_all=True, append_images=pages[1:])
    print(f"PDF saved as: {output_pdf} ({len(processed_images)} images, {len(pages)} pages, A4 layout)")


# Example usage:
# Only include files that contain 'utility_' in their filename
images_to_pdf_with_titles("experiments/forecast_only", "learning_curves.pdf", include_str="learning_curves", columns=2)