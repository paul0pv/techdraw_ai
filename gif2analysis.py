import os
from PIL import Image, ImageDraw
import imageio

# Directorios
output_dir = "outputs"
comparison_dir = os.path.join(output_dir, "comparisons_full")
os.makedirs(comparison_dir, exist_ok=True)

# Detectar épocas disponibles
epochs = sorted(
    {
        int(f.split("epoch")[1].split(".")[0])
        for f in os.listdir(output_dir)
        if f.startswith("fakeA_epoch") or f.startswith("fakeB_epoch")
    }
)

# Lista para el GIF
comparison_images = []

# Crear comparaciones por época
for epoch in epochs:
    images = []

    # Cargar imágenes si existen
    for prefix in ["realA", "fakeB", "recovA"]:
        path = os.path.join(output_dir, f"{prefix}_epoch{epoch}.png")
        if os.path.exists(path):
            img = Image.open(path).resize((256, 256))
            images.append(img)

    if not images:
        continue

    # Imagen combinada
    total_width = 256 * len(images)
    combined = Image.new("RGB", (total_width, 256))
    for i, img in enumerate(images):
        combined.paste(img, (i * 256, 0))

    # Añadir texto
    draw = ImageDraw.Draw(combined)
    draw.text((10, 10), f"Epoch {epoch}", fill=(255, 255, 255))

    # Guardar imagen
    comparison_path = os.path.join(comparison_dir, f"comparison_epoch{epoch}.png")
    combined.save(comparison_path)
    comparison_images.append(combined)

# Crear GIF
gif_path = os.path.join(output_dir, "cycleGAN_comparison.gif")
if comparison_images:
    comparison_images[0].save(
        gif_path,
        save_all=True,
        append_images=comparison_images[1:],
        duration=800,
        loop=0,
    )
    print(f"GIF generado exitosamente: {gif_path}")
else:
    print("No se encontraron imágenes para generar el GIF.")
