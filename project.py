import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import glob
import os
import re

# =============================================================================
# AYARLAR (DOSYA YOLLARINI BURAYA GİRİNİZ)
# =============================================================================
# Windows yolları için r"..." kullanıyoruz.
IMAGE_FOLDER = r"C:\Users\emino\Desktop\İYTE\2025-2026\TERM-1\CENG391\ASSIGNMENTS\ASSIGNMENT-5\Automated-People-Counting\VideoDataset\ximagepixels"
GT_FILE = r"C:\Users\emino\Desktop\İYTE\2025-2026\TERM-1\CENG391\ASSIGNMENTS\ASSIGNMENT-5\Automated-People-Counting\VideoDataset\groundtruth.txt"

# Sonuçların kaydedileceği klasör (Otomatik oluşturulur)
OUTPUT_DIR = "Odev_Sonuclari"


class PeopleCounterFinal:
    def __init__(self, img_folder, gt_file):
        self.img_folder = img_folder
        self.gt_file = gt_file
        self.features = {}
        self.ground_truth = []
        self.img_paths = []

        # Çıktı klasörünü oluştur
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)

        print("\n" + "=" * 60)
        print("CENG391 - AUTOMATED PEOPLE COUNTING (NİHAİ VERSİYON)")
        print("=" * 60)

        self.load_ground_truth()
        self.load_image_paths()
        self.sync_data()

    def natural_sort_key(self, s):
        """Dosyaları 1, 2, 10, 100 sırasında dizer."""
        return [
            int(text) if text.isdigit() else text.lower()
            for text in re.split("([0-9]+)", s)
        ]

    def read_image_safe(self, path):
        """Türkçe karakterli dosya yollarını okumak için özel fonksiyon."""
        try:
            with open(path, "rb") as f:
                bytes_data = bytearray(f.read())
                numpy_array = np.asarray(bytes_data, dtype=np.uint8)
                img = cv2.imdecode(numpy_array, cv2.IMREAD_GRAYSCALE)
                return img
        except Exception as e:
            return None

    def load_ground_truth(self):
        print(f"-> Ground Truth okunuyor...")
        try:
            with open(self.gt_file, "r") as f:
                lines = f.readlines()
                self.ground_truth = []
                for line in lines:
                    nums = re.findall(r"\d+", line)
                    if nums:
                        self.ground_truth.append(int(nums[-1]))
            self.ground_truth = np.array(self.ground_truth)
            print(f"   [OK] {len(self.ground_truth)} etiket okundu.")
        except Exception as e:
            print(f"   [HATA] Dosya okunamadı: {e}")

    def load_image_paths(self):
        print(f"-> Resimler taranıyor...")
        exts = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif"]
        paths = []
        for ext in exts:
            paths.extend(glob.glob(os.path.join(self.img_folder, ext)))

        self.img_paths = sorted(paths, key=self.natural_sort_key)
        print(f"   [OK] {len(self.img_paths)} resim bulundu.")

    def sync_data(self):
        """Veri sayısını makaledeki gibi 153'e veya minimuma eşitler."""
        if len(self.img_paths) == 0 or len(self.ground_truth) == 0:
            print("[KRİTİK HATA] Veri yok!")
            return

        # Makalede 153 görüntü kullanıldığı belirtilmiş.
        TARGET_COUNT = 153
        min_len = min(len(self.img_paths), len(self.ground_truth))

        if min_len > TARGET_COUNT:
            print(
                f"   [BİLGİ] Veri seti makaleye uygun olarak ilk {TARGET_COUNT} kareye kesiliyor."
            )
            min_len = TARGET_COUNT

        self.img_paths = self.img_paths[:min_len]
        self.ground_truth = self.ground_truth[:min_len]
        print(f"   [HAZIR] Analiz için {min_len} örnek kullanılacak.")

    def get_struct_element(self, shape, radius):
        """Makaleye uygun morfolojik yapı elemanları."""
        if shape == "disk":
            size = 2 * radius + 1
            return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        elif shape == "square":
            # Erozyon için genellikle 3x3 kullanılır (radius=1 -> size=3)
            # Makalede boyut belirtilmemişse 3x3 standarttır.
            size = 2 * radius + 1
            return cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))

    def extract_features(self):
        print("\n[İŞLEM] Özellik çıkarımı başladı...")
        f_m1, f_m2_d4, f_m2_d5, f_m3 = [], [], [], []

        # Yapı Elemanları
        se_disk4 = self.get_struct_element("disk", 4)
        se_disk5 = self.get_struct_element("disk", 5)
        se_square = self.get_struct_element("square", 1)  # 3x3 Square Kernel

        valid_idx = []

        for i, path in enumerate(self.img_paths):
            if i % 25 == 0:
                print(f"   ... {i}/{len(self.img_paths)} işlendi")

            img = self.read_image_safe(path)
            if img is None:
                continue

            # Threshold (Makale: 30)
            _, bin_img = cv2.threshold(img, 30, 255, cv2.THRESH_BINARY)

            # --- Method 1: Foreground Pixels (X) ---
            X = np.count_nonzero(bin_img)
            f_m1.append([X])

            # --- Method 2: Closed Pixels (C) ---
            # Disk-4
            closed_d4 = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, se_disk4)
            C_d4 = np.count_nonzero(closed_d4)
            f_m2_d4.append([C_d4])

            # Disk-5
            closed_d5 = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, se_disk5)
            C_d5 = np.count_nonzero(closed_d5)
            f_m2_d5.append([C_d5])

            # --- Method 3: C (from disk4) and Ratio (S/X) ---
            # Erozyon (2 kez tekrar - iterations=2)
            eroded = cv2.erode(bin_img, se_square, iterations=2)
            S = np.count_nonzero(eroded)

            ratio = (S / X) if X > 0 else 0
            f_m3.append([C_d4, ratio])

            valid_idx.append(i)

        # Listeleri numpy array'e çevir
        self.features["m1"] = np.array(f_m1)
        self.features["m2_d4"] = np.array(f_m2_d4)
        self.features["m2_d5"] = np.array(f_m2_d5)
        self.features["m3"] = np.array(f_m3)

        # Eğer okunamayan resim olduysa Ground Truth'u da ona göre güncelle
        if len(valid_idx) != len(self.ground_truth):
            self.ground_truth = self.ground_truth[valid_idx]

        print("[TAMAMLANDI] Özellikler hafızaya alındı.")

    def run_experiments(self):
        if not self.features or len(self.features["m1"]) == 0:
            return

        total = len(self.ground_truth)
        indices = np.arange(total)
        # Makale kuralı: Her 3'lü bloğun ilk 2'si TRAIN, 3.'sü TEST
        train_mask = indices % 3 != 2

        y_train = self.ground_truth[train_mask]
        y_test = self.ground_truth[~train_mask]
        test_indices = indices[~train_mask]  # X ekseni için (Sample Number)

        # Deney Listesi: (Başlık, Özellik Key, Nöron Sayısı, Figür No Başlangıcı)
        experiments = [
            ("Method 1 (Pixels)", "m1", (10,), 5),
            ("Method 2 (Disk-4)", "m2_d4", (10,), 7),
            ("Method 2 (Disk-5)", "m2_d5", (10,), 9),
            ("Method 3 (Ratio)", "m3", (50,), 11),
        ]

        results_txt = []
        results_txt.append(
            f"{'METHOD':<25} {'AVG ERROR (%)':<15} {'ACC <10%':<15} {'ACC <15%':<15}"
        )
        results_txt.append("-" * 75)

        print("\n--- YAPAY SİNİR AĞI EĞİTİLİYOR ---")

        for name, key, hidden_nodes, fig_num in experiments:
            X = self.features[key]
            X_train = X[train_mask]
            X_test = X[~train_mask]

            # Veriyi Ölçekle (NN için şart)
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)

            # Modeli Eğit
            model = MLPRegressor(
                hidden_layer_sizes=hidden_nodes,
                activation="relu",
                solver="lbfgs",
                max_iter=5000,
                random_state=42,  # Tekrarlanabilirlik için sabit
            )
            model.fit(X_train_s, y_train)
            preds = model.predict(X_test_s)

            # Sonuçları Hesapla
            errors = np.abs(preds - y_test)
            err_pct = (errors / y_test) * 100

            avg_err = np.mean(err_pct)
            acc_10 = np.mean(err_pct <= 10) * 100
            acc_15 = np.mean(err_pct <= 15) * 100

            # Tabloya ekle
            row_str = (
                f"{name:<25} {avg_err:<15.2f} {acc_10:<15.1f} {acc_15:<15.1f}"
            )
            results_txt.append(row_str)
            print(f"-> {name} tamamlandı. (Ort. Hata: %{avg_err:.2f})")

            # Grafikleri Kaydet ve Göster
            self.save_and_show_plots(
                name, fig_num, y_test, preds, err_pct, test_indices
            )

        # Sonuçları Yazdır ve Kaydet
        print("\n" + "=" * 75)
        for line in results_txt:
            print(line)
        print("=" * 75)

        # Tabloyu dosyaya yaz
        with open(os.path.join(OUTPUT_DIR, "Table_I_Results.txt"), "w") as f:
            for line in results_txt:
                f.write(line + "\n")

        print(
            f"\n[BİLGİ] Tüm grafikler ve sonuç tablosu '{OUTPUT_DIR}' klasörüne kaydedildi."
        )

    def save_and_show_plots(
        self, method_name, fig_start, gt, pred, err, indices
    ):
        # Dosya ismi için temizlik
        clean_name = (
            method_name.replace(" ", "_").replace("(", "").replace(")", "")
        )

        # --- Figür A (Tahmin vs Gerçek) ---
        plt.figure(figsize=(10, 5))
        plt.plot(
            indices, gt, "b-", linewidth=1.5, label="Ground Truth"
        )  # Mavi çizgi
        plt.plot(
            indices, pred, "r*", markersize=8, label="Estimated Count"
        )  # Kırmızı yıldız
        plt.title(f"{method_name} - Estimation Results")
        plt.xlabel("Sample Sequence")
        plt.ylabel("Number of People")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Kaydet
        save_path_est = os.path.join(
            OUTPUT_DIR, f"Fig{fig_start}_{clean_name}_Estimation.png"
        )
        plt.savefig(save_path_est)
        plt.close()  # Hafızadan sil

        # --- Figür B (Hata Yüzdesi) ---
        plt.figure(figsize=(10, 5))
        plt.plot(indices, err, "b*", markersize=8)  # Mavi yıldız
        plt.title(f"{method_name} - Error Percentage")
        plt.xlabel("Sample Sequence")
        plt.ylabel("Error (%)")
        plt.ylim(0, 80)  # Makaledeki ölçeğe yakın olması için
        plt.grid(True, alpha=0.3)

        # Kaydet
        save_path_err = os.path.join(
            OUTPUT_DIR, f"Fig{fig_start+1}_{clean_name}_Error.png"
        )
        plt.savefig(save_path_err)
        plt.close()


# --- ÇALIŞTIRMA ---
if __name__ == "__main__":
    app = PeopleCounterFinal(IMAGE_FOLDER, GT_FILE)
    if len(app.img_paths) > 0 and len(app.ground_truth) > 0:
        app.extract_features()
        app.run_experiments()
