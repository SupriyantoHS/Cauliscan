import logging
from flask import Flask, render_template, request, jsonify
import numpy as np
import os, base64, io, time, traceback
from PIL import Image, UnidentifiedImageError

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Endpoint debug hanya aktif kalau env var ini di-set eksplisit (mis. saat development lokal)
ENABLE_DEBUG_ENDPOINTS = os.environ.get('ENABLE_DEBUG_ENDPOINTS', '0') == '1'

# Logger internal — traceback lengkap masuk sini, TIDAK pernah dikirim ke client
logger = logging.getLogger('cauliscan')
logging.basicConfig(level=logging.INFO)

# ── Konstanta ─────────────────────────────────────────────────────────────────
CLASS_NAMES = ['Black Rot', 'Healthy', 'Insect Hole']

CLASS_META = {
    'Black Rot': {
        'icon': '🦠', 'severity': 'Kritis', 'severity_color': 'red',
        'cause': 'Bakteri Xanthomonas campestris pv. campestris',
        'description': 'Penyakit busuk hitam yang menyerang sistem vaskular tanaman. Menyebabkan kerusakan serius dan dapat menyebar ke seluruh tanaman jika tidak segera ditangani.',
        'symptoms': ['Tepi daun menguning berbentuk huruf V', 'Pembuluh daun berwarna hitam', 'Jaringan daun layu dan busuk', 'Bau tidak sedap pada jaringan'],
        'treatment': ['Semprot bakterisida berbasis tembaga', 'Cabut dan bakar tanaman parah', 'Hindari penyiraman dari atas', 'Rotasi tanaman minimal 2 tahun'],
        'prevention': ['Gunakan benih bersertifikat bebas penyakit', 'Pastikan drainase lahan baik', 'Hindari luka mekanis pada tanaman'],
    },
    'Healthy': {
        'icon': '🌿', 'severity': 'Sehat', 'severity_color': 'green',
        'cause': 'Tidak ada patogen terdeteksi',
        'description': 'Daun kembang kol dalam kondisi prima. Pertumbuhan optimal dengan tidak ada tanda-tanda infeksi penyakit maupun serangan hama.',
        'symptoms': ['Warna hijau merata dan segar', 'Permukaan daun mulus dan utuh', 'Tepian daun tidak bergerigi abnormal', 'Struktur tulang daun normal'],
        'treatment': ['Lanjutkan perawatan rutin', 'Pemupukan NPK sesuai jadwal', 'Penyiraman teratur di pangkal', 'Monitoring mingguan'],
        'prevention': ['Jaga kelembaban tanah stabil', 'Pengendalian gulma berkala', 'Pemangkasan daun tua secara rutin'],
    },
    'Insect Hole': {
        'icon': '🐛', 'severity': 'Waspada', 'severity_color': 'amber',
        'cause': 'Serangan serangga hama (Plutella xylostella, Spodoptera)',
        'description': 'Kerusakan akibat serangan serangga pemakan daun. Lubang-lubang tidak beraturan pada helaian daun menandakan aktivitas larva atau dewasa hama.',
        'symptoms': ['Lubang tidak beraturan pada daun', 'Tepi daun sobek atau bergerigi', 'Kotoran serangga berwarna hitam', 'Bekas gigitan di permukaan'],
        'treatment': ['Aplikasi insektisida berbahan aktif Klorpirifos', 'Semprot Bacillus thuringiensis (Bt)', 'Gunakan perangkap feromon', 'Pengendalian hayati dengan parasitoid'],
        'prevention': ['Pasang jaring anti serangga', 'Tanam tanaman pengusir hama', 'Inspeksi rutin di bawah daun'],
    },
}

# ── Global state ──────────────────────────────────────────────────────────────
model = None
IMG_SIZE = 384
RESCALE = False  # False → piksel mentah [0-255], sesuai cara model dilatih

# Kompatibilitas Pillow lama (<9.1) dan baru (>=9.1)
_LANCZOS = Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.LANCZOS

# ── Konfigurasi model ─────────────────────────────────────────────────────────
MODEL_PATH = os.environ.get('MODEL_PATH', 'model/efficientnetv2s_model.keras')
# ID file Google Drive untuk model — set via env var MODEL_GDRIVE_ID di Railway.
# Ambil dari link https://drive.google.com/file/d/<ID_INI>/view
MODEL_GDRIVE_ID = os.environ.get('MODEL_GDRIVE_ID', '')


def download_model_if_missing():
    """Download model dari Google Drive kalau belum ada di disk (sekali per deploy/restart)."""
    if os.path.exists(MODEL_PATH):
        return
    if not MODEL_GDRIVE_ID:
        print("WARNING: MODEL_GDRIVE_ID tidak di-set, tidak bisa auto-download model")
        return
    try:
        import gdown
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        print(f"Model tidak ditemukan, mendownload dari Google Drive (ID: {MODEL_GDRIVE_ID}) ...")
        gdown.download(id=MODEL_GDRIVE_ID, output=MODEL_PATH, quiet=False)
        print("Download model selesai ✔")
    except Exception as e:
        print(f"ERROR download model: {e}")


# ── Model loader ──────────────────────────────────────────────────────────────
def load_model():
    global model, IMG_SIZE
    try:
        download_model_if_missing()

        import tensorflow as tf

        if not os.path.exists(MODEL_PATH):
            print("WARNING: Model tidak ditemukan - DEMO mode aktif")
            return

        print(f"Loading model: {MODEL_PATH} ...")
        model = tf.keras.models.load_model(MODEL_PATH)

        # Ambil IMG_SIZE langsung dari model agar tidak mismatch
        IMG_SIZE = model.input_shape[1]
        print(f"Model loaded! Input: {model.input_shape}, IMG_SIZE: {IMG_SIZE}")
        print(f"Class order: {CLASS_NAMES}")

        # Warmup: hilangkan cold-start latency pada request pertama
        print("Warming up model ...")
        dummy = np.zeros((1, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
        model.predict(dummy, verbose=0)
        print("Warmup selesai ✔")
    except Exception as e:
        print(f"ERROR load model: {e}")
        model = None


# ── Core predict function ─────────────────────────────────────────────────────
def predict(mdl, image: Image.Image, img_size: int, rescale: bool) -> tuple:
    """Resize → array → (opsional rescale /255) → predict."""
    img = image.convert('RGB').resize((img_size, img_size), _LANCZOS)
    arr = np.array(img, dtype=np.float32)
    if rescale:
        arr /= 255.0
    arr = np.expand_dims(arr, axis=0)
    probs = mdl.predict(arr, verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    return CLASS_NAMES[pred_idx], float(probs[pred_idx]), probs


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict_route():
    # ── Tahap 1: baca & validasi gambar ──
    # Error di tahap ini (file bukan gambar, base64 rusak, dll) adalah kesalahan
    # input pengguna, jadi pesannya boleh cukup spesifik dan aman ditampilkan.
    try:
        if 'file' in request.files and request.files['file'].filename:
            img = Image.open(request.files['file'].stream)
            img.load()
        elif 'image_data' in request.form:
            raw = request.form['image_data']
            if ',' in raw:
                raw = raw.split(',')[1]
            img = Image.open(io.BytesIO(base64.b64decode(raw)))
            img.load()
        else:
            return jsonify({'error': 'Tidak ada gambar yang diterima'}), 400
    except (UnidentifiedImageError, ValueError, OSError, base64.binascii.Error):
        logger.warning('Gagal membaca gambar dari request', exc_info=True)
        return jsonify({'error': 'File yang dikirim bukan gambar yang valid'}), 400

    # ── Tahap 2: inference ──
    # Error di tahap ini adalah masalah internal (model, memori, dll) — jangan
    # pernah dikirim ke client, cukup log server-side dan balas pesan generik.
    try:
        t0 = time.time()
        if model is not None:
            top_class, top_prob, probs = predict(model, img, IMG_SIZE, RESCALE)
            preds = probs.tolist()
            mode = 'live'
        else:
            # Demo mode — random probabilitas
            np.random.seed(int(time.time() * 1000) % 9999)
            preds = np.random.dirichlet([2, 2, 2]).tolist()
            top_i = int(np.argmax(preds))
            top_class = CLASS_NAMES[top_i]
            top_prob = float(preds[top_i])
            mode = 'demo'

        ms = round((time.time() - t0) * 1000, 1)
        confidence = round(top_prob * 100, 2)

        all_results = sorted([
            {
                'class': CLASS_NAMES[i],
                'probability': round(float(p) * 100, 2),
                'is_top': CLASS_NAMES[i] == top_class,
            }
            for i, p in enumerate(preds)
        ], key=lambda x: -x['probability'])

        # Debug log di terminal
        print(f"\n>>> Prediksi ({mode}):")
        for r in all_results:
            mark = " << TOP" if r['is_top'] else ""
            print(f"  {r['class']:<20} {r['probability']:>6.2f}%{mark}")
        print(f"  Inference: {ms}ms")

        return jsonify({
            'success': True,
            'predicted_class': top_class,
            'confidence': confidence,
            'results': all_results,
            'meta': CLASS_META.get(top_class, CLASS_META['Healthy']),
            'inference_ms': ms,
            'mode': mode,
        })
    except Exception:
        # Traceback lengkap tetap dicatat di server untuk debugging,
        # tapi client hanya menerima pesan generik (tidak ada detail internal).
        logger.error('ERROR saat inference', exc_info=True)
        return jsonify({'error': 'Terjadi kesalahan saat memproses gambar. Coba lagi.'}), 500


@app.route('/health')
def health():
    return jsonify({
        'model_loaded': model is not None,
        'classes': CLASS_NAMES,
        'img_size': IMG_SIZE,
        'rescale': RESCALE,
        'mode': 'live' if model is not None else 'demo',
    })


@app.route('/debug/class-order')
def debug_class_order():
    # Nonaktif secara default — membocorkan info internal model.
    # Aktifkan hanya saat development lokal: set ENABLE_DEBUG_ENDPOINTS=1
    if not ENABLE_DEBUG_ENDPOINTS:
        return jsonify({'error': 'Not found'}), 404
    return jsonify({
        'class_names': CLASS_NAMES,
        'img_size': IMG_SIZE,
        'rescale': RESCALE,
        'model_loaded': model is not None,
        'model_input_shape': str(model.input_shape) if model else None,
        'model_output_shape': str(model.output_shape) if model else None,
    })


# Load model sekali saat modul di-import — ini yang dijalankan gunicorn di Railway
load_model()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', debug=os.environ.get('FLASK_DEBUG', '0') == '1', port=port)
