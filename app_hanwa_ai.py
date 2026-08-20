import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
from dotenv import load_dotenv
from groq import Groq

# Memuat file .env jika dijalankan secara lokal
load_dotenv()

# --- 1. KONFIGURASI HALAMAN WEB STREAMLIT ---
st.set_page_config(page_title="Hanwa AI Hybrid Chat", page_icon="🤖")
st.title("🤖 Hanwa AI - Hybrid System")
st.write("Sistem chat otonom dengan opsi pemilihan mesin model AI (Local Neural Network vs Groq API).")

# --- 2. SIDEBAR: PENGATURAN & PEMILIHAN MODEL ---
st.sidebar.header("⚙️ Pengaturan Model")
selected_model = st.sidebar.selectbox(
    "Pilih Mesin AI:",
    ["Hanwa AI (Local Model)", "API Eksternal (Groq API)"]
)

st.sidebar.markdown("---")
if selected_model == "API Eksternal (Groq API)":
    st.sidebar.success("API Key Groq terhubung secara aman melalui Secrets/Environment.")
else:
    st.sidebar.success("Menggunakan model neural network buatan sendiri (Hanwa AI PyTorch).")

# --- 3. ARSITEKTUR HANWA AI (LOCAL MODEL) ---
BLOCK_SIZE = 64
EMBED_DIM = 128
NUM_HEADS = 8
NUM_LAYERS = 4
DROPOUT = 0.1
DEVICE = 'cpu'

corpus = (
    "hanwa ai adalah model bahasa otonom tingkat lanjut yang mengimplementasikan arsitektur transformer modern. "
    "rotary positional embeddings atau rope memungkinkan representasi posisi relatif yang superior pada inferensi teks panjang. "
    "swiglu activation function menggantikan relu tradisional untuk meningkatkan kapasitas non-linier jaringan syaraf tiruan."
)

chars = sorted(list(set(corpus)))
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}
vocab_size = len(chars)

def precompute_rope_frequencies(dim, end, theta=10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(end, device=DEVICE).type_as(freqs)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs).contiguous()

def apply_rope(xq, xk, freqs_cis):
    xq_ = torch.view_as_complex(xq.float().contiguous().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().contiguous().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.contiguous().reshape(1, xq_.shape[1], 1, xq_.shape[-1])
    xq_out = torch.view_as_real(xq_ * freqs_cis).reshape(*xq.shape)
    xk_out = torch.view_as_real(xk_ * freqs_cis).reshape(*xk.shape)
    return xq_out.type_as(xq), xk_out.type_as(xk)

class SwiGLUMLP(nn.Module):
    def __init__(self):
        super().__init__()
        hidden_dim = int(8 * EMBED_DIM / 3)
        self.w1 = nn.Linear(EMBED_DIM, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, EMBED_DIM, bias=False)
        self.w3 = nn.Linear(EMBED_DIM, hidden_dim, bias=False)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))

class UltimateCausalAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.head_dim = EMBED_DIM // NUM_HEADS
        self.q_proj = nn.Linear(EMBED_DIM, EMBED_DIM, bias=False)
        self.k_proj = nn.Linear(EMBED_DIM, EMBED_DIM, bias=False)
        self.v_proj = nn.Linear(EMBED_DIM, EMBED_DIM, bias=False)
        self.out_proj = nn.Linear(EMBED_DIM, EMBED_DIM, bias=False)
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)).view(1, 1, BLOCK_SIZE, BLOCK_SIZE)
        )
        self.freqs_cis = precompute_rope_frequencies(self.head_dim, BLOCK_SIZE)

    def forward(self, x):
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, NUM_HEADS, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, NUM_HEADS, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, NUM_HEADS, self.head_dim).transpose(1, 2)

        q, k = apply_rope(q, k, self.freqs_cis[:T])

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(y)

class UltimateTransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln_1 = nn.LayerNorm(EMBED_DIM)
        self.attn = UltimateCausalAttention()
        self.ln_2 = nn.LayerNorm(EMBED_DIM)
        self.mlp = SwiGLUMLP()

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class UltimateNanoGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, EMBED_DIM)
        self.drop = nn.Dropout(DROPOUT)
        self.blocks = nn.Sequential(*[UltimateTransformerBlock() for _ in range(NUM_LAYERS)])
        self.ln_f = nn.LayerNorm(EMBED_DIM)
        self.lm_head = nn.Linear(EMBED_DIM, vocab_size, bias=False)

    def forward(self, idx):
        x = self.drop(self.token_embedding(idx))
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits

@st.cache_resource
def load_local_model():
    m = UltimateNanoGPT().to(DEVICE)
    m.eval()
    return m

local_model = load_local_model()

# --- 4. MANAJEMEN RIWAYAT CHAT & LOGIKA PEMILIHAN MODEL ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ketik pesan Anda di sini..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if selected_model == "Hanwa AI (Local Model)":
            with st.spinner("Hanwa AI (Local) sedang memproses teks..."):
                idx = torch.tensor([[char_to_idx.get(c, 0) for c in prompt]], dtype=torch.long, device=DEVICE)

                with torch.no_grad():
                    for _ in range(100):
                        idx_cond = idx[:, -BLOCK_SIZE:]
                        logits = local_model(idx_cond)
                        logits = logits[:, -1, :] / 0.6

                        top_k = 3
                        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                        logits[logits < v[:, [-1]]] = -float('Inf')

                        probs = F.softmax(logits, dim=-1)
                        next_idx = torch.multinomial(probs, num_samples=1)
                        idx = torch.cat((idx, next_idx), dim=1)

                response_text = "".join([idx_to_char[i.item()] for i in idx[0]])
                st.markdown(f"🧠 **[Hanwa AI Local]**: {response_text}")

        else:
            # Bagian koneksi ke Groq API (Streamlit Secrets / Environment Variable)
            with st.spinner("Menghubungkan ke Groq API..."):
                try:
                    # Mencoba mengambil kunci dari Streamlit Secrets atau Environment Variable (.env)
                    try:
                        api_key = st.secrets["GROQ_API_KEY"]
                    except Exception:
                        api_key = os.environ.get("GROQ_API_KEY")

                    if not api_key:
                        raise ValueError("GROQ_API_KEY tidak ditemukan. Cek file .env atau Streamlit Secrets.")

                    client = Groq(api_key=api_key)

                    completion = client.chat.completions.create(
                        model="openai/gpt-oss-120b",  # Model gratis Groq (pengganti llama-3.3-70b-versatile yang sudah deprecated)
                        messages=[
                            {"role": "user", "content": prompt}
                        ],
                    )
                    reply = completion.choices[0].message.content
                    response_text = f"🌐 **[Groq API]**: {reply}"
                except Exception as e:
                    response_text = f"❌ Terjadi kesalahan saat menghubungi API: {e}"

                st.markdown(response_text)

    st.session_state.messages.append({"role": "assistant", "content": response_text})
