# utils.py
import random
import string
import math
from collections import Counter

import torch

# ============================
# 1. VOCABULARIO Y CONSTANTES
# ============================

LOWER = string.ascii_lowercase
UPPER = string.ascii_uppercase
DIGITS = string.digits
SYMS  = "!@#$%&*"

VOCAB = list(LOWER + UPPER + DIGITS + SYMS)
MAX_LEN = 12
PAD_TOKEN = "<PAD>"

VOCAB = [PAD_TOKEN] + VOCAB
vocab2idx = {ch: i for i, ch in enumerate(VOCAB)}
idx2vocab = {i: ch for ch, i in vocab2idx.items()}


# ============================
# 2. GENERACIÓN DE PASSWORDS
# ============================

def random_common_pattern():
    """Simula contraseñas típicas débiles."""
    bases = ["password", "qwerty", "abc123", "admin", "welcome", "dragon", "iloveyou"]
    base = random.choice(bases)
    suf = str(random.randint(0, 999))
    return (base + suf)[:MAX_LEN]


def random_mixed():
    """Contraseñas algo más complejas, pero realistas."""
    length = random.randint(6, MAX_LEN)
    chars = []
    for _ in range(length):
        pool = random.choice([LOWER, UPPER, DIGITS, SYMS])
        chars.append(random.choice(pool))
    return "".join(chars)


def random_phrase_style():
    """Tipo 'Frase123!' simplificada."""
    words = ["Sun", "Moon", "Cat", "Dog", "Blue", "Red", "Star", "Cloud"]
    pw = random.choice(words) + random.choice(words) + str(random.randint(0, 99))
    if random.random() < 0.5:
        pw += random.choice(SYMS)
    return pw[:MAX_LEN]


def generate_synthetic_passwords(n=5000):
    """Genera un dataset sintético tipo 'leak'."""
    passwords = []
    for _ in range(n):
        kind = random.random()
        if kind < 0.4:
            pw = random_common_pattern()
        elif kind < 0.8:
            pw = random_mixed()
        else:
            pw = random_phrase_style()
        passwords.append(pw)
    return passwords


# ============================
# 3. ENCODING / DECODING
# ============================

def encode_password(pw):
    """Codifica una contraseña como vector de índices (con padding)."""
    pw = pw[:MAX_LEN]
    idxs = [vocab2idx[ch] for ch in pw if ch in vocab2idx]
    while len(idxs) < MAX_LEN:
        idxs.append(vocab2idx[PAD_TOKEN])
    return idxs


def decode_password(idxs):
    """Reconstruye la contraseña a partir de índices."""
    chars = []
    for i in idxs:
        ch = idx2vocab[int(i)]
        if ch == PAD_TOKEN:
            break
        chars.append(ch)
    return "".join(chars)


def passwords_to_tensor(passwords, device):
    encoded = [encode_password(pw) for pw in passwords]
    return torch.tensor(encoded, dtype=torch.long, device=device)


# ============================
# 4. GENERADOR ALEATORIO
# ============================

def generate_passwords_random(n=1000):
    """Generador baseline totalmente aleatorio."""
    pws = []
    for _ in range(n):
        length = random.randint(4, MAX_LEN)
        chars = [random.choice(VOCAB[1:]) for _ in range(length)]  # sin PAD
        pws.append("".join(chars))
    return pws


# ============================
# 5. MÉTRICAS
# ============================

def hit_rate(generated, test_set, top_k=None):
    """
    Porcentaje de contraseñas del test que aparecen en la lista generada.
    """
    if top_k is not None:
        generated = generated[:top_k]
    gen_set = set(generated)
    hits = len(test_set & gen_set)
    return hits / len(test_set) * 100


def char_entropy(passwords):
    """Entropía de Shannon (aprox.) sobre distribución de caracteres."""
    all_chars = "".join(passwords)
    counts = Counter(all_chars)
    total = sum(counts.values())
    H = 0.0
    for c, cnt in counts.items():
        p = cnt / total
        H -= p * math.log2(p)
    return H
