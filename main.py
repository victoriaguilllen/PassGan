# main.py
import random
import os

import torch
import matplotlib.pyplot as plt

from utils import (
    generate_synthetic_passwords,
    generate_passwords_random,
    hit_rate,
    char_entropy,
    MAX_LEN,
    VOCAB,
)
from PassGan import (
    Generator,
    Discriminator,
    LATENT_DIM,
    EMBED_DIM,
    HIDDEN_DIM,
    train_gan,
    generate_passwords_gan,
)

# ============================
# 0. CONFIGURACIÓN GLOBAL
# ============================
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("Usando dispositivo:", device)


random.seed(42)
torch.manual_seed(42)

# Carpeta donde se guardarán las figuras
FIG_DIR = "figuras"
os.makedirs(FIG_DIR, exist_ok=True)

# ============================
# 1. GENERAR DATASET SINTÉTICO
# ============================

dataset = generate_synthetic_passwords(6000)
random.shuffle(dataset)

split = int(0.8 * len(dataset))
train_passwords = dataset[:split]
test_passwords  = dataset[split:]
test_set = set(test_passwords)

print("Tamaño dataset:", len(dataset))
print("Ejemplos de contraseñas sintéticas:")
for pw in dataset[:10]:
    print("  ", pw)

# ============================
# 2. CREAR Y ENTRENAR GAN
# ============================

vocab_size = len(VOCAB)

generator = Generator(LATENT_DIM, MAX_LEN, vocab_size).to(device)
discriminator = Discriminator(vocab_size, EMBED_DIM, HIDDEN_DIM, MAX_LEN).to(device)

train_gan(generator, discriminator, train_passwords,
          device=device, epochs=40, batch_size=128)

# ============================
# 3. GENERAR CONTRASEÑAS
# ============================

generated_passwords = generate_passwords_gan(generator, n=2000, device=device)

print("\nEjemplos de contraseñas generadas por la GAN:")
for pw in generated_passwords[:20]:
    print("  ", pw)

random_passwords = generate_passwords_random(2000)

# ============================
# 4. MÉTRICAS DE EFICACIA
# ============================

ks = [100, 500, 1000, 2000]
hr_gan_list = []
hr_rand_list = []

for k in ks:
    hr_gan = hit_rate(generated_passwords, test_set, top_k=k)
    hr_rand = hit_rate(random_passwords, test_set, top_k=k)
    hr_gan_list.append(hr_gan)
    hr_rand_list.append(hr_rand)

    print(f"\nTOP-{k}:")
    print(f"  Hit-rate GAN      : {hr_gan:.2f}%")
    print(f"  Hit-rate Aleatorio: {hr_rand:.2f}%")

# ============================
# 5. ANÁLISIS DE RIESGOS (ENTROPÍA)
# ============================

H_train = char_entropy(train_passwords)
H_gan   = char_entropy(generated_passwords)
H_rand  = char_entropy(random_passwords)

print("\nEntropía de caracteres (Shannon, bits):")
print(f"  Dataset sintético real : {H_train:.3f}")
print(f"  GAN (contraseñas gen.) : {H_gan:.3f}")
print(f"  Aleatorio puro         : {H_rand:.3f}")

# ============================
# 6. GRÁFICAS Y GUARDADO
# ============================

# --- 6.1 Gráfica de Hit-rate vs TOP-K (GAN vs Aleatorio) ---
plt.figure()
plt.plot(ks, hr_gan_list, marker="o", label="GAN")
plt.plot(ks, hr_rand_list, marker="o", label="Aleatorio")
plt.xlabel("Top-K contraseñas generadas")
plt.ylabel("Hit-rate sobre test (%)")
plt.title("Comparación de eficacia: GAN vs generador aleatorio")
plt.legend()
plt.grid(True)
hitrate_path = os.path.join(FIG_DIR, "hitrate_gan_vs_aleatorio.png")
plt.savefig(hitrate_path, bbox_inches="tight", dpi=200)
plt.close()
print(f"Gráfica guardada en: {hitrate_path}")

# --- 6.2 Gráfica de entropía (barras) ---
plt.figure()
labels = ["Real (sintético)", "GAN", "Aleatorio"]
values = [H_train, H_gan, H_rand]
plt.bar(labels, values)
plt.ylabel("Entropía de caracteres (bits)")
plt.title("Comparación de entropía de contraseñas")
entropy_path = os.path.join(FIG_DIR, "entropia_comparacion.png")
plt.savefig(entropy_path, bbox_inches="tight", dpi=200)
plt.close()
print(f"Gráfica guardada en: {entropy_path}")

# --- 6.3 Distribución de longitudes de contraseñas ---
def lengths(passwords):
    return [len(pw) for pw in passwords]

len_train = lengths(train_passwords)
len_gan   = lengths(generated_passwords)
len_rand  = lengths(random_passwords)

plt.figure()
bins = range(1, MAX_LEN + 2)  # de 1 a MAX_LEN
plt.hist(len_train, bins=bins, alpha=0.5, label="Real (sintético)")
plt.hist(len_gan,   bins=bins, alpha=0.5, label="GAN")
plt.hist(len_rand,  bins=bins, alpha=0.5, label="Aleatorio")
plt.xlabel("Longitud de la contraseña")
plt.ylabel("Frecuencia")
plt.title("Distribución de longitudes de contraseñas")
plt.legend()
lengths_path = os.path.join(FIG_DIR, "distribucion_longitudes.png")
plt.savefig(lengths_path, bbox_inches="tight", dpi=200)
plt.close()
print(f"Gráfica guardada en: {lengths_path}")

# ============================
# 7. RESUMEN PARA PRESENTACIÓN
# ============================

print("\n=== RESUMEN PARA PRESENTACIÓN ===")
print(f"Nº contraseñas entrenamiento : {len(train_passwords)}")
print(f"Nº contraseñas test          : {len(test_passwords)}")
print(f"Nº contraseñas GAN generadas : {len(generated_passwords)}")
print(f"Nº contraseñas aleatorias    : {len(random_passwords)}")

for k, hr_g, hr_r in zip(ks, hr_gan_list, hr_rand_list):
    print(f"\nTOP-{k}: Hit-rate sobre test")
    print(f"  GAN     : {hr_g:5.2f}%")
    print(f"  Random  : {hr_r:5.2f}%")

print("\nEntropía aproximada de caracteres:")
print(f"  Real (sintético): {H_train:.3f} bits")
print(f"  GAN             : {H_gan:.3f} bits")
print(f"  Random          : {H_rand:.3f} bits")

print("\nFiguras generadas en la carpeta 'figuras':")
print("  - hitrate_gan_vs_aleatorio.png")
print("  - entropia_comparacion.png")
print("  - distribucion_longitudes.png")