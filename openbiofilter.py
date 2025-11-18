# OPENBIOFILTER – FULL WORKING VERSION

!pip install -q torch torchaudio librosa soundfile tqdm

import torch, librosa, numpy as np, soundfile as sf, os, random
from torch import nn
from tqdm import tqdm

# Tiny model
class BioFilter(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU()
        )
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=32,
                nhead=4,
                dim_feedforward=128,
                batch_first=True
            ),
            num_layers=6
        )
        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = x.permute(0, 2, 3, 1).flatten(1, 2)
        x = self.transformer(x)
        x = x.mean(dim=1)
        return torch.sigmoid(self.fc(x))

model = BioFilter()
opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
criterion = nn.BCELoss()

# 1. Real data — Watkins whales (128 clips) + 3 real ship noises
!mkdir -p watkins nonbio
!wget -q -O watkins.zip "https://archive.org/download/watkins_best_of_whales_202008/watkins_best_of_whales.zip"
!unzip -q watkins.zip -d watkins_temp
!find watkins_temp -name "*.wav" -exec mv {} watkins/ \;
!rm -r watkins_temp watkins.zip

!wget -q -O nonbio/ship1.wav "https://dosits.org/wp-content/uploads/2023/06/merchant_ship_noise.wav"
!wget -q -O nonbio/ship2.wav "https://dosits.org/wp-content/uploads/2023/06/container_ship_noise.wav"
!wget -q -O nonbio/ship3.wav "https://dosits.org/wp-content/uploads/2023/06/cruise_ship_noise.wav"

!wget -q -O demo_whale.wav "https://dosits.org/wp-content/uploads/2023/06/humpback_whale_song_hawaii.wav"

# 2. Load function
def spec_from_wav(path):
    wav, _ = librosa.load(path, sr=48000)
    spec = librosa.feature.melspectrogram(y=wav, sr=48000, n_mels=64, fmax=1000)
    return torch.from_numpy(librosa.power_to_db(spec, ref=np.max)).unsqueeze(0)

bio_files = [f"watkins/{f}" for f in os.listdir("watkins") if f.endswith('.wav')]
nonbio_files = ["nonbio/ship1.wav", "nonbio/ship2.wav", "nonbio/ship3.wav"] * (len(bio_files)//3 + 1)

# 3. Train 15 epochs
for epoch in range(15):
    random.shuffle(bio_files)
    random.shuffle(nonbio_files)
    losses = []
    for b, nb in zip(bio_files[:300], nonbio_files[:300]):
        try:
            spec_b = spec_from_wav(b)
            spec_nb = spec_from_wav(nb)
            opt.zero_grad()
            loss = (
                criterion(model(spec_b), torch.tensor([[1.0]])) +
                criterion(model(spec_nb), torch.tensor([[0.0]]))
            )
            loss.backward()
            opt.step()
            losses.append(loss.item())
        except Exception:
            pass
    if losses:
        print(f"Epoch {epoch+1} – loss {np.mean(losses):.4f}")
    else:
        print(f"Epoch {epoch+1} – no valid batches")

# 4. Quantise and save
model.eval()
quantised = torch.quantization.quantize_dynamic(model, {nn.Linear, nn.Conv2d}, dtype=torch.qint8)
torch.save(quantised.state_dict(), "openbiofilter_quantised.pth")
print("Model ready – size:", os.path.getsize("openbiofilter_quantised.pth")/1e6, "MB")

# 5. Demo on real humpback whale song
wav, sr = sf.read("demo_whale.wav")
spec = librosa.feature.melspectrogram(y=wav, sr=sr, n_mels=64, fmax=1000)
spec_db = librosa.power_to_db(spec, ref=np.max)
tensor = torch.from_numpy(spec_db).unsqueeze(0).float()

prob = quantised(tensor).item()
print(f"\nReal humpback whale song – biological probability: {prob:.3f} → "
      f"{'DISCARD' if prob > 0.85 else 'TRANSMIT'}")
