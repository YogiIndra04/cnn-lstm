# Kolok Sentence Recognition (CNN-LSTM)

Prototype pengenalan kalimat sederhana Bahasa Kolok berbasis gesture tangan dari webcam.

Mode landmark saat ini: otomatis 1 atau 2 tangan per frame.
- Format fitur selalu tetap 84 (`2 tangan x 21 titik x (x,y)`).
- Jika hanya 1 tangan terdeteksi, slot tangan yang tidak ada akan diisi nol.

## Struktur Project

- `capture_data.py` -> rekam video gesture per label kalimat.
- `extract_landmarks.py` -> ekstrak landmark tangan (x, y) hingga 2 tangan, total fitur 84/frame.
- `preprocess_dataset.py` -> normalisasi MinMax, bentuk sequence tetap, split train/val.
- `train_model.py` -> training model CNN + LSTM.
- `realtime_inference.py` -> inferensi real-time webcam.
- `reset_project.py` -> reset artefak agar eksperimen bisa diulang dari awal.
- `kolok_utils.py` -> utilitas bersama.
- `data/raw_videos/` -> video mentah per label.
- `data/landmarks/` -> sequence landmark `.npy`.
- `data/processed/` -> dataset `.npz`, scaler, label encoder.
- `models/` -> model hasil training.

## Instalasi

```bash
pip install -r requirements.txt
```

## 1) Capture Data Gesture

Contoh rekam 20 sample label `saya makan`:

```bash
python capture_data.py --label "saya makan" --num-samples 20 --duration 3
```

Ulangi untuk label lain:
- `saya pergi`
- `kamu datang`
- `terima kasih`
- `apa kabar`

Folder otomatis dibuat menjadi format aman, misalnya `saya makan` -> `saya_makan`.

## 2) Ekstraksi Landmark MediaPipe

```bash
python extract_landmarks.py --input-dir data/raw_videos --output-dir data/landmarks
```

Output tiap video: array shape `(jumlah_frame, 84)` untuk mode 2 tangan.
Anda bisa paksa 1 tangan jika perlu:
```bash
python extract_landmarks.py --max-num-hands 1
```

## 3) Preprocessing Sequence Dataset

Contoh fixed sequence 30 frame:

```bash
python preprocess_dataset.py --input-dir data/landmarks --output-dir data/processed --seq-len 30
```

Secara default preprocessing akan menambah augmentasi tukar slot tangan kiri/kanan
agar model tidak terlalu sensitif jika posisi tangan saat inferensi berubah.
Jika ingin mematikan augmentasi:
```bash
python preprocess_dataset.py --no-augment-swap-hands
```

Output utama:
- `data/processed/dataset_sequences.npz`
- `data/processed/minmax_scaler.pkl`
- `data/processed/label_encoder.pkl`

## 4) Training CNN-LSTM

```bash
python train_model.py --dataset data/processed/dataset_sequences.npz --epochs 50 --batch-size 16
```

Model terbaik disimpan di:
- `models/cnn_lstm_kolok.keras`

## 5) Real-time Inference Webcam

```bash
python realtime_inference.py --model models/cnn_lstm_kolok.keras --seq-len 30
```

Tekan `Q` untuk keluar.

Opsional paksa inferensi 1 tangan:
```bash
python realtime_inference.py --max-num-hands 1
```

Untuk mengurangi hasil ambigu, realtime inference sekarang memakai:
- smoothing probabilitas antar frame (`--smoothing-window`, default 5)
- batas confidence minimal (`--confidence-threshold`, default 0.6)
- batas selisih top-1 vs top-2 (`--min-margin`, default 0.08)

Contoh set lebih longgar saat dataset masih kecil:
```bash
python realtime_inference.py --confidence-threshold 0.5 --min-margin 0.05
```

## Menambah Kalimat Baru

1. Rekam data dengan label baru:
   ```bash
   python capture_data.py --label "label baru" --num-samples 20
   ```
2. Jalankan ulang `extract_landmarks.py`.
3. Jalankan ulang `preprocess_dataset.py`.
4. Jalankan ulang `train_model.py`.

Pipeline akan otomatis membaca folder label baru dan memperbarui kelas klasifikasi.

## Reset Dari Awal

Reset aman (video mentah tetap disimpan):
```bash
python reset_project.py --yes
```

Full reset (termasuk `data/raw_videos`):
```bash
python reset_project.py --full --yes
```

Setelah reset, ulang pipeline dari awal:
1. `python capture_data.py --label "..." --num-samples 20`
2. `python extract_landmarks.py`
3. `python preprocess_dataset.py --seq-len 30`
4. `python train_model.py`
5. `python realtime_inference.py`
