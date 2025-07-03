import os
from PIL import Image, ImageDraw
import imageio

# Ruta de salida del entrenamiento
output_dir = "outputs"

# Detectar las épocas disponibles
epochs = sorted(
    [
        int(f.split("epoch")[1].split(".")[0])
        for f in os.listdir(output_dir)
        if f.startswith("fakeA_epoch") and f.endswith(".png")
    ]
)

# Crear carpeta para comparaciones
comparison_dir = os.path.join(output_dir, "comparisons")
os.makedirs(comparison_dir, exist_ok=True)

# Lista para almacenar imágenes del GIF
comparison_images = []

# Crear imágenes comparativas lado a lado
for epoch in epochs:
    fakeA_path = os.path.join(output_dir, f"fakeA_epoch{epoch}.png")
    fakeB_path = os.path.join(output_dir, f"fakeB_epoch{epoch}.png")

    if os.path.exists(fakeA_path) and os.path.exists(fakeB_path):
        fakeA = Image.open(fakeA_path).resize((256, 256))
        fakeB = Image.open(fakeB_path).resize((256, 256))

        # Imagen combinada
        combined = Image.new("RGB", (512, 256))
        combined.paste(fakeA, (0, 0))
        combined.paste(fakeB, (256, 0))

        # Añadir texto de época
        draw = ImageDraw.Draw(combined)
        draw.text((10, 10), f"Epoch {epoch}", fill=(255, 255, 255))

        # Guardar imagen comparativa
        comparison_path = os.path.join(comparison_dir, f"comparison_epoch{epoch}.png")
        combined.save(comparison_path)
        comparison_images.append(combined)

# Crear GIF animado
gif_path = os.path.join(output_dir, "cycleGAN_progress.gif")
comparison_images[0].save(
    gif_path, save_all=True, append_images=comparison_images[1:], duration=800, loop=0
)

print(f"GIF generado exitosamente: {gif_path}")
