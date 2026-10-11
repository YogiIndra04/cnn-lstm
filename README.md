# Pengenalan Kata Bahasa Kolok (CNN2D-LSTM Video RGB)

Sistem pengenalan kata Bahasa Kolok menggunakan arsitektur deep learning CNN 2D dan LSTM berbasis video RGB. Input model berupa frame video RGB murni tanpa ekstraksi landmark MediaPipe.

---

## Spesifikasi Metodologi Penelitian

- **Input Model**: Sequence video RGB `(batch, 30, 300, 300, 3)` dengan nilai ternormalisasi `float32 [0, 1]`.
- **Ekstraksi Spasial (CNN 2D)**:
  - Blok 1: `Conv2D(32, (3, 3), padding="same", activation="relu")` + `MaxPooling2D((2, 2))`
  - Blok 2: `Conv2D(64, (3, 3), padding="same", activation="relu")` + `MaxPooling2D((2, 2))`
  - Blok 3: `Conv2D(128, (3, 3), padding="same", activation="relu")` + `MaxPooling2D((2, 2))`
  - Reduksi Spasial: `TimeDistributed(GlobalAveragePooling2D())` → output sequence `(batch, 30, 128)`
- **Ekstraksi Temporal (LSTM)**:
  - `LSTM(128 unit, return_sequences=False)` → representasi temporal `(batch, 128)`
- **Klasifikasi**:
  - `Dropout(0.5)`
  - `Dense(64, activation="relu")`
  - `Dense(n_classes, activation="softmax")`
- **Loss & Optimizer**: `SparseCategoricalCrossentropy` & `Adam(learning_rate=0.001)`.

---

## Struktur Folder Proyek

```
bahasa-kolok/
├── configs/
│   └── model_config.py          # Konfigurasi terpusat parameter penelitian
├── data/
│   ├── raw_videos/              # Video RGB mentah per folder kelas
│   ├── processed/               # dataset_sequences.npz & label_map.json
│   └── splits/                  # split_manifest.json (train/val/test)
├── models/                      # Checkpoint model (.keras), history (.npz), logs
├── reports/                     # Hasil evaluasi, confusion matrix, report JSON/CSV
├── tests/                       # Unit & integration test (pytest)
├── capture_data.py              # Perekaman video RGB via webcam
├── preprocess_dataset.py        # Preprocessing video RGB, validasi tensor, data split
├── model_architecture.py       # Definisi arsitektur CNN2D-LSTM Keras
├── train_model.py               # Pelatihan model dengan callbacks & tf.data
├── evaluate_model.py            # Evaluasi model pada test set independen
├── predict_video.py             # Inferensi prediksi kata dari satu video
├── realtime_inference.py        # Inferensi real-time kamera webcam (RGB)
├── reset_project.py             # Reset artefak hasil generate
├── requirements.txt             # Dependensi proyek
└── pytest.ini                   # Konfigurasi pengujian pytest
```

---

## Instalasi

```bash
pip install -r requirements.txt
```

---

## Alur Penggunaan Pipeline

### 1. Perekaman Data Baru (Opsional)
Untuk merekam video gesture kata baru menggunakan webcam:
```bash
python capture_data.py --label "nama_kata" --num-samples 20 --duration 3
```
Video akan disimpan di `data/raw_videos/<nama_kata>/sample_XXX.mp4`.

### 2. Preprocessing Dataset
Memproses video mentah menjadi sequence 30 frame RGB 300×300 piksel dan membaginya ke train/val/test (70/15/15):
```bash
# Cek deteksi dataset tanpa memproses:
python preprocess_dataset.py --dry-run

# Jalankan preprocessing penuh:
python preprocess_dataset.py
```
Output tersimpan di:
- `data/processed/dataset_sequences.npz`
- `data/processed/label_map.json`
- `data/splits/split_manifest.json`
- `reports/preprocessing_report.json`

### 3. Pelatihan Model (Training)
```bash
# Pelatihan default (batch-size disesuaikan dengan kapasitas RAM/VRAM):
python train_model.py --batch-size 8

# Jika menggunakan CPU laptop dengan memori terbatas:
python train_model.py --batch-size 2
```
Model terbaik akan disimpan di `models/best_model.keras` dipantau berdasarkan `val_loss`.

### 4. Evaluasi Model pada Test Set Independen
Evaluasi metrik (Accuracy, Precision, Recall, F1-Score, Confusion Matrix):
```bash
python evaluate_model.py --model models/best_model.keras
```
Laporan evaluasi tersimpan di folder `reports/`:
- `reports/evaluation_summary.json`
- `reports/classification_report.csv`
- `reports/confusion_matrix.csv`
- `reports/confusion_matrix.png`
- `reports/test_predictions.csv`

### 5. Prediksi Video Tunggal
Memprediksi kata dari file video baru:
```bash
python predict_video.py --video path/ke/video.mp4
```

### 6. Inferensi Real-time Webcam
Menjalankan inferensi interaktif melalui kamera webcam:
```bash
python realtime_inference.py --model models/best_model.keras
```
Tekan tombol `Q` untuk keluar.

### 7. Menjalankan Automated Test
```bash
python -m pytest
```

### 8. Reset Artefak
Untuk membersihkan hasil training/preprocessing tanpa menghapus video mentah:
```bash
python reset_project.py --yes
```
