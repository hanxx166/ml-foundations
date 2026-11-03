<div align="center">

# Machine Learning Portfolio

**Kumpulan eksplorasi, implementasi, dan analisis algoritma pembelajaran mesin**

Dokumentasi tugas mandiri dan proyek kelompok Mata Kuliah Pembelajaran Mesin, Semester 5.

[![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![pandas](https://img.shields.io/badge/pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)

</div>

---

## Daftar Isi

- [Tentang Repositori](#tentang-repositori)
- [Struktur Repositori](#struktur-repositori)
- [Ringkasan Modul](#ringkasan-modul)
- [Detail Eksplorasi](#detail-eksplorasi)
  - [Tugas 1 - Regresi Linear dan ANN](#tugas-1---regresi-linear-dan-ann)
  - [Tugas 2 - KNN, Decision Tree, dan Naive Bayes](#tugas-2---knn-decision-tree-dan-naive-bayes)
  - [Tugas 3 - SVM dan ANN](#tugas-3---svm-dan-ann)
  - [Tugas 4 - PCA dan SVD](#tugas-4---pca-dan-svd)
  - [Tugas 5 - K-Means dan Agglomerative Clustering](#tugas-5---k-means-dan-agglomerative-clustering)
- [Proyek Kelompok 01](#proyek-kelompok-01)
- [Teknologi yang Digunakan](#teknologi-yang-digunakan)
- [Cara Menjalankan](#cara-menjalankan)
- [Catatan Dataset](#catatan-dataset)
- [Author](#author)
- [Status dan Pembaruan](#status-dan-pembaruan)

---

## Tentang Repositori

Repositori ini berisi kumpulan notebook Jupyter yang mendokumentasikan proses pembelajaran mesin secara menyeluruh, mulai dari eksplorasi data, praproses, pembangunan model, evaluasi, hingga penarikan kesimpulan.

Materi disusun secara bertahap:

- **Supervised learning** untuk regresi dan klasifikasi.
- **Unsupervised learning** untuk reduksi dimensi dan clustering.
- **Implementasi manual dan berbasis library** menggunakan NumPy, scikit-learn, dan PyTorch.
- **Studi kasus kelompok** pada dataset asuransi medis.

Setiap notebook tidak hanya berisi kode, tetapi juga analisis hasil, perbandingan model, dan interpretasi metrik evaluasi.

---

## Struktur Repositori

```text
Machine_Learning/
├── Code/
│   ├── EksplorasiMandiri/
│   │   ├── Tugas1/
│   │   │   └── modul_01_supervised_learning.ipynb
│   │   ├── Tugas2/
│   │   │   └── modul_02_supervised_learning.ipynb
│   │   ├── Tugas3/
│   │   │   ├── data/
│   │   │   │   └── heart.csv
│   │   │   └── modul_03_supervised_learning.ipynb
│   │   ├── Tugas4/
│   │   │   └── modul_04_unsupervised_learning.ipynb
│   │   └── Tugas5/
│   │       ├── data/
│   │       │   └── Mall_Customers.csv
│   │       └── modul_05_unsupervised_learning.ipynb
│   └── TugasKelompok01/
│       ├── data/
│       │   └── insurance.csv
│       └── TugasKelompokBerbasisKasus01_Coba-coba.ipynb
├── PDF/
│   └── laporan proyek kelompok
├── .gitignore
└── README.md
```

---

## Ringkasan Modul

| Modul | Topik | Dataset | Pendekatan | Notebook |
|---|---|---|---|---|
| Tugas 1 | Regresi linear | Data sintetis dan California Housing | Least Square Estimator, ANN PyTorch | `modul_01_supervised_learning.ipynb` |
| Tugas 2 | Klasifikasi | Wine | KNN, Decision Tree, Gaussian Naive Bayes | `modul_02_supervised_learning.ipynb` |
| Tugas 3 | Klasifikasi biner | Heart Disease | SVM dan ANN PyTorch | `modul_03_supervised_learning.ipynb` |
| Tugas 4 | Reduksi dimensi | Iris | PCA dan Truncated SVD | `modul_04_unsupervised_learning.ipynb` |
| Tugas 5 | Clustering | Mall Customers | K-Means dan Agglomerative Clustering | `modul_05_unsupervised_learning.ipynb` |
| Kelompok 01 | Regresi dan klasifikasi | Medical Insurance Cost | Linear/SVR/MLP dan Logistic/SVM/MLP | `TugasKelompokBerbasisKasus01_Coba-coba.ipynb` |

---

## Detail Eksplorasi

### Tugas 1 - Regresi Linear dan ANN

Eksplorasi regresi linear dilakukan melalui dua pendekatan yang berbeda.

**Regresi linear univariat**

- Data sintetis berjumlah 500 sampel dengan pola linear dan noise terkontrol.
- Parameter model diestimasi menggunakan **Least Square Estimator (LSE)** secara closed-form.
- Evaluasi dilakukan dengan **R²** dan **MSE** pada data latih dan data uji.
- Pembagian data menggunakan rasio 70:30.

**Regresi multivariat dengan ANN**

- Dataset **California Housing** digunakan untuk memprediksi `MedHouseVal`.
- Empat fitur dengan korelasi tertinggi dipilih: `MedInc`, `AveRooms`, `Latitude`, dan `HouseAge`.
- Data dinormalisasi menggunakan **MinMaxScaler**.
- Model ANN dibangun dengan PyTorch menggunakan optimizer **SGD**, learning rate `0.001`, 100 epoch, dan batch size `256`.
- Hasil prediksi dibandingkan dengan nilai aktual untuk melihat kualitas generalisasi model.

**Poin pembelajaran:** LSE cocok untuk regresi linear sederhana karena memberikan solusi eksak, sedangkan ANN lebih fleksibel untuk regresi multivariat dan data berdimensi lebih kompleks.

---

### Tugas 2 - KNN, Decision Tree, dan Naive Bayes

Notebook ini membandingkan tiga algoritma klasifikasi pada **Wine Dataset** yang terdiri dari 178 sampel, 13 fitur kimia, dan 3 kelas kultivar anggur.

**Konfigurasi model**

- **K-Nearest Neighbor** dengan `n_neighbors=10`.
- **Decision Tree** dengan `splitter='best'`.
- **Gaussian Naive Bayes** sebagai baseline probabilistik.
- Data dinormalisasi menggunakan **MinMaxScaler** dan dibagi dengan rasio 70:30.

**Hasil utama**

| Model | Akurasi Uji | Catatan |
|---|---:|---|
| KNN | 98% | Akurasi tertinggi dan distribusi prediksi paling seimbang |
| Naive Bayes | 94% | Cepat, stabil, dan kompetitif sebagai baseline |
| Decision Tree | 87% | Mudah diinterpretasikan, tetapi menunjukkan gejala overfitting |

**Poin pembelajaran:** KNN sangat terbantu oleh normalisasi fitur, sedangkan Decision Tree unggul dalam interpretabilitas. Naive Bayes tetap kompetitif meskipun asumsi independensi antarfitur tidak sepenuhnya terpenuhi.

---

### Tugas 3 - SVM dan ANN

Notebook ini mengeksplorasi klasifikasi biner pada **Heart Disease Dataset** menggunakan dua pendekatan: Support Vector Machine dan Artificial Neural Network.

**Praproses data**

- Nilai `0` yang tidak logis pada `RestingBP`, `Cholesterol`, dan `MaxHR` diubah menjadi `NaN`, lalu diimputasi dengan nilai rata-rata.
- Kolom kategorikal dikonversi menjadi numerik.
- Fitur dinormalisasi menggunakan **MinMaxScaler**.
- Data dibagi menjadi data latih dan data uji dengan rasio 70:30.

**Model**

- **SVM** dengan kernel polinomial, `degree=3`, dan `C=1.0`.
- **ANN** berupa Multilayer Perceptron dengan dua hidden layer, aktivasi ReLU, output Sigmoid, optimizer SGD, 100 epoch, dan learning rate `0.1`.

**Poin pembelajaran:** Praproses yang tepat, terutama imputasi dan normalisasi, berperan besar dalam kualitas model SVM maupun ANN. Evaluasi dilakukan menggunakan `classification_report` untuk melihat precision, recall, dan F1-score.

---

### Tugas 4 - PCA dan SVD

Notebook ini membandingkan dua teknik reduksi dimensi pada **Iris Dataset**: **Principal Component Analysis (PCA)** dan **Truncated Singular Value Decomposition (SVD)**.

**Alur eksplorasi**

- Data distandarisasi menggunakan **StandardScaler**.
- Korelasi antarfitur dianalisis melalui heatmap.
- Empat fitur direduksi menjadi tiga komponen.
- Hasil PCA dan SVD dibandingkan berdasarkan explained variance dan ortogonalitas komponen.
- Visualisasi 3D digunakan untuk melihat sebaran spesies setelah reduksi dimensi.

**Hasil utama**

- Tiga komponen mempertahankan sekitar **99,48% varians** data asli.
- PCA dan TruncatedSVD menghasilkan nilai explained variance yang identik karena data telah di-centering.
- Principal components bersifat ortogonal dengan korelasi mendekati nol.
- Spesies *setosa* paling mudah dipisahkan, sedangkan *versicolor* dan *virginica* masih sedikit tumpang tindih.

**Poin pembelajaran:** PCA cocok untuk data dense yang memerlukan centering, sedangkan TruncatedSVD lebih fleksibel untuk data sparse seperti TF-IDF.

---

### Tugas 5 - K-Means dan Agglomerative Clustering

Notebook ini menerapkan clustering pada **Mall Customers Dataset** untuk mengelompokkan pelanggan berdasarkan karakteristik demografis dan perilaku belanja.

**Alur eksplorasi**

- Kolom `Gender` dikonversi menjadi numerik.
- Fitur yang digunakan meliputi `Gender`, `Age`, `Annual Income`, dan `Spending Score`.
- Data dinormalisasi menggunakan **MinMaxScaler**.
- **PCA** dengan tiga komponen digunakan untuk kebutuhan visualisasi 3D.
- Jumlah cluster diuji pada rentang 2 sampai 15.

**Evaluasi model**

- **K-Means** dievaluasi menggunakan **Silhouette Score**.
- **Agglomerative Clustering** dengan linkage `complete` dievaluasi menggunakan **Davies-Bouldin Score**.
- Model final menggunakan `n_clusters=2` untuk kedua pendekatan.
- Hasil clustering divisualisasikan dalam ruang PCA 3D.

**Poin pembelajaran:** Pemilihan jumlah cluster sebaiknya didukung metrik evaluasi dan interpretasi domain. K-Means dan Agglomerative dapat menghasilkan struktur kelompok yang berbeda meskipun menggunakan jumlah cluster yang sama.

---

## Proyek Kelompok 01

**Tugas Kelompok Berbasis Kasus 01 - Medical Insurance Cost**

Proyek ini menggunakan dataset asuransi medis untuk menyelesaikan dua tugas sekaligus:

1. **Regresi** untuk memprediksi biaya asuransi (`charges`).
2. **Klasifikasi** untuk memprediksi status perokok (`smoker`).

**Tahapan utama**

- Exploratory Data Analysis (EDA) untuk melihat distribusi target, keseimbangan kelas, dan hubungan antarfitur.
- Praproses data: label encoding, one-hot encoding untuk `region`, rekayasa fitur `obese` dan `smoker_bmi`, serta penghapusan data duplikat.
- Pencegahan data leakage dengan melatih scaler hanya pada data latih.
- Pemisahan data 70:30 untuk train+validation dan test, dilanjutkan pembagian 80:20 untuk train dan validation.

**Model regresi**

- Linear Regression sebagai baseline.
- Support Vector Regression (SVR).
- ANN/MLP Regresi.

**Model klasifikasi**

- Logistic Regression sebagai baseline.
- Support Vector Machine (SVM).
- ANN/MLP Klasifikasi.

**Hasil utama**

- Model regresi terbaik adalah **SVR polynomial** dengan **R² = 0,8550**.
- Model klasifikasi terbaik mencapai **F1-score = 0,9153** dengan recall `0,9878`.
- Eksperimen tanpa fitur `charges` menunjukkan bahwa fitur tersebut sangat prediktif, tetapi fitur demografis masih menyimpan sinyal lemah untuk mengidentifikasi perokok.
- Laporan lengkap proyek tersedia pada folder `PDF/`.

**Poin pembelajaran:** Rekayasa fitur, pencegahan data leakage, pemilihan metrik yang sesuai, dan validasi model yang konsisten sangat menentukan kualitas hasil akhir.

---

## Teknologi yang Digunakan

| Kategori | Teknologi |
|---|---|
| Bahasa | Python |
| Lingkungan | Jupyter Notebook, Anaconda |
| Komputasi numerik | NumPy, pandas |
| Machine learning | scikit-learn |
| Deep learning | PyTorch |
| Visualisasi | Matplotlib, Seaborn |
| Metrik evaluasi | Accuracy, Precision, Recall, F1-score, R², MSE, MAE, RMSE, Silhouette Score, Davies-Bouldin Score |

---

## Cara Menjalankan

### 1. Clone repositori

```bash
git clone https://github.com/hanxx166/ml-foundations.git
cd ml-foundations
```

### 2. Buat environment Python

```bash
python -m venv .venv
```

Aktivasi environment:

**Windows (PowerShell)**

```powershell
.venv\Scripts\Activate.ps1
```

**Linux/macOS**

```bash
source .venv/bin/activate
```

### 3. Install dependensi

```bash
pip install numpy pandas scikit-learn matplotlib seaborn jupyter
pip install torch
```

Untuk instalasi PyTorch dengan dukungan GPU, sesuaikan perintah dengan dokumentasi resmi PyTorch.

### 4. Jalankan Jupyter Notebook

```bash
jupyter notebook
```

Buka notebook yang diinginkan, lalu jalankan seluruh cell secara berurutan menggunakan **Run All**.

> Catatan: notebook Tugas 3, Tugas 5, dan proyek kelompok membaca dataset dari folder `data/` relatif terhadap lokasi notebook. Pastikan file CSV tersedia sebelum menjalankan notebook.

---

## Catatan Dataset

| Dataset | Sumber | Digunakan pada |
|---|---|---|
| California Housing | scikit-learn built-in | Tugas 1 |
| Wine | scikit-learn built-in | Tugas 2 |
| Heart Disease | `heart.csv` | Tugas 3 |
| Iris | scikit-learn built-in | Tugas 4 |
| Mall Customers | `Mall_Customers.csv` | Tugas 5 |
| Medical Insurance Cost | `insurance.csv` | Proyek Kelompok 01 |

Dataset lokal disimpan di dalam folder `data/` pada masing-masing direktori tugas. Laporan proyek kelompok dalam format PDF tersedia di folder `PDF/`.

---

## Author

**Erhan Kurniawan**

GitHub: [@hanxx166](https://github.com/hanxx166)

---

## Status dan Pembaruan

> **Repositori ini masih dalam proses pengembangan.**

README dan seluruh notebook akan terus diperbarui seiring dengan:

- Penambahan tugas dan modul baru.
- Perbaikan kode serta peningkatan kualitas analisis.
- Penambahan eksperimen, visualisasi, dan perbandingan model.
- Penyempurnaan dokumentasi agar lebih mudah dipelajari dan direproduksi.

Untuk melihat pembaruan terbaru, silakan periksa riwayat commit pada branch `main`.

---

<div align="center">

<sub>Dibuat untuk keperluan pembelajaran dan dokumentasi Mata Kuliah Pembelajaran Mesin.</sub>

</div>
