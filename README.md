# 🌿 CauliScan — Deteksi Penyakit Daun Kembang Kol

**Stack:** Flask · EfficientNetV2S · Tailwind CSS · Vanilla JS

## Kelas Deteksi
| Kelas | Keterangan | Severity |
|-------|-----------|----------|
| 🦠 Black Rot | Busuk hitam (bakteri Xanthomonas) | 🔴 Kritis |
| 🌿 Healthy | Daun sehat | 🟢 Sehat |
| 🐛 Insect Hole | Kerusakan hama serangga | 🟡 Waspada |

## Setup

```bash
# 1. Masuk ke folder
cd cauliscan

# 2. Install dependencies
pip install -r requirements.txt

# 3. Letakkan model (download dari Google Drive)
#    https://drive.google.com/file/d/1Chtef-gEz9IM51RcHBKEsHyF1sGnP7tn
mkdir -p model
# simpan sebagai: model/cauliflower_efficientnetv2s_model.keras

# 4. Jalankan
python app.py
# Buka: http://localhost:5000
```

## Struktur
```
cauliscan/
├── app.py
├── requirements.txt
├── model/
│   └── cauliflower_efficientnetv2s_model.keras   ← taruh model di sini
├── templates/
│   └── index.html
├── static/
│   └── js/app.js
└── uploads/
```

## Fitur UI
- Upload drag & drop + preview gambar
- Kamera live (capture & scan)
- Animasi scan line + confidence ring
- Probability bars animasi per kelas
- Detail: gejala, penanganan, pencegahan
- Custom cursor animated
- Dark botanical aesthetic
- Demo mode otomatis jika model belum ada
