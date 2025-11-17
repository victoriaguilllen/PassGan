# passgan.py
import random

import torch
import torch.nn as nn
import torch.optim as optim

from utils import MAX_LEN, VOCAB, decode_password, passwords_to_tensor

LATENT_DIM = 64
EMBED_DIM  = 64
HIDDEN_DIM = 128


# ============================
# 1. MODELOS
# ============================

class Generator(nn.Module):
    """
    Toma un vector latente (ruido) y produce una secuencia de caracteres.
    Implementación simple con MLP + proyección.
    """
    def __init__(self, latent_dim, seq_len, vocab_size):
        super().__init__()
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, seq_len * vocab_size)
        )

    def forward(self, z):
        out = self.fc(z)  # (batch, seq_len * vocab_size)
        out = out.view(-1, self.seq_len, self.vocab_size)
        return out

    def sample(self, batch_size, device):
        z = torch.randn(batch_size, LATENT_DIM, device=device)
        logits = self.forward(z)
        probs = torch.softmax(logits, dim=-1)
        idxs = torch.argmax(probs, dim=-1)
        return idxs  # (batch, seq_len)


class Discriminator(nn.Module):
    """
    Clasifica secuencias como reales (1) o falsas (0).
    """
    def __init__(self, vocab_size, embed_dim, hidden_dim, seq_len):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        emb = self.embed(x)        # (batch, seq_len, embed_dim)
        _, (h, _) = self.lstm(emb)
        h = h[-1]
        out = self.fc(h)
        return out.squeeze(1)


# ============================
# 2. ENTRENAMIENTO GAN
# ============================

def train_gan(generator, discriminator, train_passwords,
              device, epochs=40, batch_size=128):
    """
    Entrena la GAN sobre el conjunto de contraseñas de entrenamiento.
    """
    criterion = nn.BCELoss()
    opt_D = optim.Adam(discriminator.parameters(), lr=2e-4)
    opt_G = optim.Adam(generator.parameters(), lr=2e-4)

    train_tensor = passwords_to_tensor(train_passwords, device)

    def get_real_batch(bs):
        idxs = random.sample(range(len(train_tensor)), bs)
        return train_tensor[idxs]

    for epoch in range(1, epochs + 1):
        generator.train()
        discriminator.train()

        # --- 1) Actualizar Discriminador ---
        real_data = get_real_batch(batch_size)
        real_labels = torch.ones(batch_size, device=device)

        fake_data = generator.sample(batch_size, device=device).detach()
        fake_labels = torch.zeros(batch_size, device=device)

        opt_D.zero_grad()
        pred_real = discriminator(real_data)
        pred_fake = discriminator(fake_data)

        loss_real = criterion(pred_real, real_labels)
        loss_fake = criterion(pred_fake, fake_labels)
        loss_D = (loss_real + loss_fake) / 2
        loss_D.backward()
        opt_D.step()

        # --- 2) Actualizar Generador ---
        opt_G.zero_grad()
        fake_data = generator.sample(batch_size, device=device)
        fake_target = torch.ones(batch_size, device=device)
        pred_fake_for_G = discriminator(fake_data)
        loss_G = criterion(pred_fake_for_G, fake_target)
        loss_G.backward()
        opt_G.step()

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch}/{epochs} | Loss_D: {loss_D.item():.4f} | Loss_G: {loss_G.item():.4f}")


# ============================
# 3. GENERACIÓN DE CONTRASEÑAS
# ============================

def generate_passwords_gan(generator, n, device):
    """
    Genera n contraseñas usando el generador entrenado.
    """
    generator.eval()
    all_pw = []
    batch_size = 128
    while len(all_pw) < n:
        bs = min(batch_size, n - len(all_pw))
        idxs = generator.sample(bs, device=device)
        for seq in idxs:
            pw = decode_password(seq)
            if len(pw) > 0:
                all_pw.append(pw)
    return all_pw
