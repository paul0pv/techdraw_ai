import os
import itertools
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms

# from torchvision.io import read_image
# from torchvision.datasets import DatasetFolder
from PIL import Image  # , ImageDraw
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torchvision.utils import make_grid


# Configuración del dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transformaciones de imagen
transform = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]
)

# Carga de datos
data_root = "cycleGAN_dataset"


class FlatImageFolder(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = [
            os.path.join(root_dir, fname)
            for fname in os.listdir(root_dir)
            if fname.lower().endswith((".png", ".jpg", ".jpeg"))
        ]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, 0  # dummy label


trainA = FlatImageFolder(os.path.join(data_root, "trainA"), transform=transform)
trainB = FlatImageFolder(os.path.join(data_root, "trainB"), transform=transform)

train_loader_A = DataLoader(trainA, batch_size=1, shuffle=True)
train_loader_B = DataLoader(trainB, batch_size=1, shuffle=True)


# Bloque residual para el generador
class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.InstanceNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.InstanceNorm2d(dim),
        )

    def forward(self, x):
        return x + self.block(x)


# Generador
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        model = [
            nn.Conv2d(3, 64, kernel_size=7, padding=3),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
        ]
        for _ in range(6):
            model += [ResidualBlock(64)]
        model += [nn.Conv2d(64, 3, kernel_size=7, padding=3), nn.Tanh()]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


# Discriminador
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        model = [
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 1, kernel_size=4, padding=1),
        ]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


# Inicialización de modelos
G_AB = Generator().to(device)
G_BA = Generator().to(device)
D_A = Discriminator().to(device)
D_B = Discriminator().to(device)

# Funciones de pérdida
criterion_GAN = nn.MSELoss()
criterion_cycle = nn.L1Loss()

# Optimizadores
optimizer_G = optim.Adam(
    itertools.chain(G_AB.parameters(), G_BA.parameters()), lr=0.0002, betas=(0.5, 0.999)
)
optimizer_D_A = optim.Adam(D_A.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D_B = optim.Adam(D_B.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Entrenamiento
epochs = 5
for epoch in range(epochs):
    for (real_A, _), (real_B, _) in zip(train_loader_A, train_loader_B):
        real_A = real_A.to(device)
        real_B = real_B.to(device)

        # valid = torch.ones((real_A.size(0), 1, 30, 30), device=device)
        # fake = torch.zeros((real_A.size(0), 1, 30, 30), device=device)

        # ------------------
        #  Train Generators
        # ------------------
        optimizer_G.zero_grad()

        fake_B = G_AB(real_A)
        fake_A = G_BA(real_B)

        # Obtener predicciones de los discriminadores
        pred_fake_B = D_B(fake_B)
        pred_fake_A = D_A(fake_A)

        # Crear etiquetas válidas y falsas con el mismo tamaño que las predicciones
        valid_B = torch.ones_like(pred_fake_B, device=device)
        valid_A = torch.ones_like(pred_fake_A, device=device)

        # Pérdidas adversariales
        loss_GAN_AB = criterion_GAN(pred_fake_B, valid_B)
        loss_GAN_BA = criterion_GAN(pred_fake_A, valid_A)

        # Pérdidas de ciclo
        recov_A = G_BA(fake_B)
        recov_B = G_AB(fake_A)
        loss_cycle_A = criterion_cycle(recov_A, real_A)
        loss_cycle_B = criterion_cycle(recov_B, real_B)

        # Pérdida total del generador
        loss_G = loss_GAN_AB + loss_GAN_BA + 10.0 * (loss_cycle_A + loss_cycle_B)
        loss_G.backward()
        optimizer_G.step()

        # -----------------------
        #  Train Discriminator A
        # -----------------------
        optimizer_D_A.zero_grad()
        pred_real_A = D_A(real_A)
        pred_fake_A = D_A(fake_A.detach())
        valid = torch.ones_like(pred_real_A, device=device)
        fake = torch.zeros_like(pred_fake_A, device=device)
        loss_real = criterion_GAN(pred_real_A, valid)
        loss_fake = criterion_GAN(pred_fake_A, fake)
        loss_D_A = (loss_real + loss_fake) * 0.5
        loss_D_A.backward()
        optimizer_D_A.step()

        # -----------------------
        #  Train Discriminator B
        # -----------------------
        optimizer_D_B.zero_grad()
        pred_real_B = D_B(real_B)
        pred_fake_B = D_B(fake_B.detach())
        valid = torch.ones_like(pred_real_B, device=device)
        fake = torch.zeros_like(pred_fake_B, device=device)
        loss_real = criterion_GAN(pred_real_B, valid)
        loss_fake = criterion_GAN(pred_fake_B, fake)
        loss_D_B = (loss_real + loss_fake) * 0.5
        loss_D_B.backward()
        optimizer_D_B.step()

    print(
        f"Epoch [{epoch + 1}/{epochs}] - Loss_G: {loss_G.item():.4f} - Loss_D_A: {loss_D_A.item():.4f} - Loss_D_B: {loss_D_B.item():.4f}"
    )

    # Guardar imágenes generadas
    os.makedirs("outputs", exist_ok=True)
    save_image((fake_B + 1) / 2.0, f"outputs/fakeB_epoch{epoch + 1}.png")
    save_image((fake_A + 1) / 2.0, f"outputs/fakeA_epoch{epoch + 1}.png")
    grid = make_grid([(real_A[0] + 1) / 2, (fake_B[0] + 1) / 2], nrow=2)
    save_image(grid, f"outputs/comparison_epoch{epoch + 1}.png")


# Guardar modelos
torch.save(G_AB.state_dict(), "G_AB.pth")
torch.save(G_BA.state_dict(), "G_BA.pth")
torch.save(D_A.state_dict(), "D_A.pth")
torch.save(D_B.state_dict(), "D_B.pth")
print("Entrenamiento completo y modelos guardados.")
