# Proyek Klasifikasi Gambar: Intel Image Classification (Computer Vision)

Repositori ini memuat implementasi *end-to-end* proyek **Computer Vision** untuk mengklasifikasikan pemandangan alam ke dalam 6 kategori berbeda menggunakan dataset populer **Intel Image Classification**. Proyek ini diselesaikan untuk memenuhi kriteria kelulusan bersertifikasi dari Dicoding.

## Ringkasan Dataset & Analisis Awal (EDA)
Dataset yang digunakan diambil langsung dari Kaggle (`puneet6060/intel-image-classification`) menggunakan utilitas `kagglehub`. Dataset ini terdiri dari ribuan gambar pemandangan alam yang dibagi menjadi 6 kelas:
* `mountain` (Pegunungan)
* `glacier` (Gletser/Es)
* `forest` (Hutan)
* `sea` (Laut)
* `buildings` (Bangunan/Gedung)
* `street` (Jalanan kota)

### Karakteristik Eksplorasi Data:
* **Resolusi Gambar Dinamis:** Analisis sampel menunjukkan adanya variasi ukuran piksel (resolusi unik) pada dataset asli, sehingga diperlukan proses standarisasi dimensi gambar sebelum masuk ke arsitektur model.
* **Distribusi Kelas:** Distribusi jumlah data divisualisasikan menggunakan diagram batang (*bar plot*) untuk memastikan kestabilan persebaran data latih antar-kategori objek.

---

## Alur Rekayasa Data (Data Preprocessing Pipeline)

Untuk menjamin objektivitas evaluasi dan keandalan performa model, dilakukan perancangan ulang pembagian dataset melalui jalur berikut:

1.  **Penggabungan Dataset (*Data Merging*):** Direktori bawaan `seg_train` dan `seg_test` dilebur menjadi satu basis data tunggal di folder `combined_dataset` untuk penataan ulang proporsi data yang lebih ketat.
2.  **Stratified Data Splitting:** Dataset dipisahkan secara berjenjang menggunakan fungsi `train_test_split` dengan parameter `stratify` guna memastikan rasio distribusi kelas tetap seimbang di setiap sub-sampel:
    * **Training Set (70%)** $\rightarrow$ Untuk proses pembelajaran model.
    * **Validation Set (15%)** $\rightarrow$ Untuk pemantauan *fitting* selama pelatihan.
    * **Testing Set (15%)** $\rightarrow$ Untuk pengujian independen akhir.
3.  **Augmentasi Gambar (Image Augmentation):** Menggunakan `ImageDataGenerator` pada data training untuk meminimalkan risiko overfitting dan memperkaya variasi sampel buatan secara *real-time* lewat teknik:
    * Penskalaan nilai piksel (`rescale=1./255`)
    * Rotasi acak (`rotation_range=20`)
    * Pergeseran sudut pandang (`shear_range=0.2` & `zoom_range=0.2`)
    * Pembalikan horizontal (`horizontal_flip=True`)
4.  **Data Stream Ingestion:** Gambar dialirkan dari DataFrame menuju memori dengan target dimensi seragam **150x150 piksel** dan ukuran *batch* sebesar 32.

---

## Arsitektur Model Deep Learning (CNN)

Model dibangun dari awal (*from scratch*) menggunakan paradigma **Sequential CNN** berlapis tinggi dengan struktur sebagai berikut:

* **Blok Konvolusi (4 Layer Convolutional):**
    * Menggunakan filter bertingkat: **32, 64, 128, dan 256** dengan ukuran *kernel* `(3,3)`.
    * Fungsi aktivasi `relu` untuk menangkap pola non-linearitas visual (tepi, warna, tekstur).
    * Setiap layer konvolusi dilengkapi dengan **BatchNormalization** untuk menstabilkan dan mempercepat proses konvergensi latihan.
    * **MaxPooling2D (2,2)** dipasang di setiap akhir blok untuk mereduksi dimensi spasial gambar.
* **Fully Connected Layer (Dense):**
    * Lapisan `Flatten` untuk mengubah matriks 2D menjadi vektor 1D.
    * Dua unit *Hidden Layer* masing-masing sebesar **512 dan 256 neuron**.
    * Lapisan **Dropout (0.5 & 0.3)** disisipkan untuk mencegah model menghafal detail data latih secara berlebihan (*regularisasi*).
* **Output Layer:** Lapisan akhir dengan **6 neuron** beraktivasi **Softmax** untuk menghasilkan distribusi probabilitas multi-kelas.
