import torch
import os

def generate_private_key(channel, size, save_name, m_bits=50):

    print(f"Generating {m_bits}-bit Orthogonal Key...")
    D_latent = channel * size * size
    R = torch.randn(m_bits, D_latent, dtype=torch.float32)
    Q, _ = torch.linalg.qr(R.T)
    P = Q.T 

    save_path="/root/autodl-tmp/watermark_project/Multi_bit"
    path=os.path.join(save_path, f"{save_name}_key_{m_bits}.pt")
    torch.save(P, path)
    print(f"Global Key successfully saved to {path}")
    return P

if __name__ == "__main__":
    generate_private_key(channel=4, size=64, save_name="global_watermark", m_bits=30)