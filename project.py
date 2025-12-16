import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import glob
import os
import re

# --- KONFİGÜRASYON ---
IMAGE_FOLDER = r"C:\Users\emino\Desktop\İYTE\2025-2026\TERM-1\CENG391\ASSIGNMENTS\ASSIGNMENT-5\Automated-People-Counting\VideoDataset\ximagepixels"
GT_FILE = r"C:\Users\emino\Desktop\İYTE\2025-2026\TERM-1\CENG391\ASSIGNMENTS\ASSIGNMENT-5\Automated-People-Counting\VideoDataset\groundtruth.txt"


class PeopleCounterAssignment:
    def __init__(self, img_folder, gt_file):
        self.img_folder = img_folder
        self.gt_file = gt_file
        self.features = {}
        self.ground_truth = []
        self.img_paths = []

        print("\n" + "=" * 50)
        print("PROGRAM BAŞLATILIYOR (Türkçe Karakter Yamalı)")
        print("=" * 50)

        # İşlemleri Başlat
        self.load_ground_truth()
        self.load_image_paths()
        self.sync_data()

    def natural_sort_key(self, s):
        return [
            int(text) if text.isdigit() else text.lower()
            for text in re.split("([0-9]+)", s)
        ]

    def load_ground_truth(self):
        try:
            print("\n-> Ground Truth okunuyor...")
            with open(self.gt_file, "r") as f:
                lines = f.readlines()
                self.ground_truth = []
                for line in lines:
                    nums = re.findall(r"\d+", line)
                    if nums:
                        self.ground_truth.append(int(nums[-1]))

            self.ground_truth = np.array(self.ground_truth)
            print(f"   [BAŞARILI] {len(self.ground_truth)} satır veri okundu.")
        except Exception as e:
            print(f"   [HATA] Dosya okunurken hata: {e}")
            self.ground_truth = np.array([])

    def load_image_paths(self):
        print("\n-> Resimler taranıyor...")
        if not os.path.exists(self.img_folder):
            print(f"   [KRİTİK HATA] Klasör bulunamadı: {self.img_folder}")
            return

        exts = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif"]
        paths = []
        for ext in exts:
            # glob da Türkçe karakter sevmeyebilir ama genellikle çalışır,
            # asıl sorun cv2.imread'deydi.
            paths.extend(glob.glob(os.path.join(self.img_folder, ext)))

        self.img_paths = sorted(paths, key=self.natural_sort_key)

        if len(self.img_paths) > 0:
            print(
                f"   [BAŞARILI] Klasörde {len(self.img_paths)} adet resim bulundu."
            )
        else:
            print(
                f"   [UYARI] Klasör bulundu ama içi BOŞ! Aranan yol: {self.img_folder}"
            )

    def sync_data(self):
        if len(self.img_paths) == 0 or len(self.ground_truth) == 0:
            print("\n[DURDURULDU] Veri eksik olduğu için devam edilemiyor.")
            return

        min_len = min(len(self.img_paths), len(self.ground_truth))
        self.img_paths = self.img_paths[:min_len]
        self.ground_truth = self.ground_truth[:min_len]
        print(
            f"\n[HAZIR] {min_len} adet resim ve etiket eşleştirildi. Analiz başlıyor..."
        )

    def get_struct_element(self, shape, radius):
        if shape == "disk":
            size = 2 * radius + 1
            return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        elif shape == "square":
            return cv2.getStructuringElement(cv2.MORPH_RECT, (radius, radius))

    def read_image_safe(self, path):
        """
        Türkçe karakter içeren yolları okumak için güvenli yöntem.
        cv2.imread yerine numpy+imdecode kullanılır.
        """
        try:
            # Dosyayı binary olarak oku
            with open(path, "rb") as f:
                bytes_data = bytearray(f.read())
                numpy_array = np.asarray(bytes_data, dtype=np.uint8)
                # OpenCV ile decode et
                img = cv2.imdecode(numpy_array, cv2.IMREAD_GRAYSCALE)
                return img
        except Exception as e:
            print(f"Okuma hatası: {path} -> {e}")
            return None

    def extract_features(self):
        print("\n[İŞLEM] Özellikler çıkarılıyor (Lütfen bekleyin)...")
        f_m1, f_m2_d4, f_m2_d5, f_m3 = [], [], [], []

        se_disk4 = self.get_struct_element("disk", 4)
        se_disk5 = self.get_struct_element("disk", 5)
        se_square2 = self.get_struct_element("square", 2)

        valid_count = 0

        for i, path in enumerate(self.img_paths):
            if i % 20 == 0:
                print(f"   ... {i}/{len(self.img_paths)} resim işlendi")

            # YENİ GÜVENLİ OKUMA FONKSİYONU
            img = self.read_image_safe(path)

            if img is None:
                print(f"   [UYARI] Resim okunamadı (atlandı): {path}")
                continue

            valid_count += 1
            _, bin_img = cv2.threshold(img, 30, 255, cv2.THRESH_BINARY)

            # Method 1
            X = np.count_nonzero(bin_img)
            f_m1.append([X])

            # Method 2
            closed_d4 = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, se_disk4)
            C_d4 = np.count_nonzero(closed_d4)
            f_m2_d4.append([C_d4])

            closed_d5 = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, se_disk5)
            C_d5 = np.count_nonzero(closed_d5)
            f_m2_d5.append([C_d5])

            # Method 3
            eroded = cv2.erode(bin_img, se_square2, iterations=2)
            S = np.count_nonzero(eroded)
            ratio = (S / X) if X > 0 else 0
            f_m3.append([C_d4, ratio])

        self.features["m1"] = np.array(f_m1)
        self.features["m2_d4"] = np.array(f_m2_d4)
        self.features["m2_d5"] = np.array(f_m2_d5)
        self.features["m3"] = np.array(f_m3)

        print(f"[TAMAMLANDI] {valid_count} adet resimden özellik çıkarıldı.")

        # Eğer hiç resim okunamazsa listeler boş kalır, burada önlem alıyoruz.
        if valid_count == 0:
            print("[KRİTİK] Hiçbir resim okunamadı! Program durduruluyor.")
            self.features = {}  # Boşalt

    def plot_results(self, title, gt, pred, err):
        plt.figure(figsize=(10, 8))

        # Üst Grafik
        plt.subplot(2, 1, 1)
        plt.plot(gt, "b-", label="Ground Truth")
        plt.plot(pred, "r*", label="Estimated Count")
        plt.title(f"{title} - Estimation Results")
        plt.legend()
        plt.ylabel("Number of People")
        plt.grid(True, alpha=0.3)

        # Alt Grafik
        plt.subplot(2, 1, 2)
        plt.plot(err, "b*")
        plt.title("Error Percentage (%)")
        plt.xlabel("Sample Sequence")
        plt.ylabel("Error (%)")
        plt.ylim(0, 100)
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def run_experiments(self):
        if not self.features or len(self.features["m1"]) == 0:
            print("[HATA] Özellik verisi yok. Eğitim yapılamaz.")
            return

        # Okunan geçerli resim sayısına göre etiketleri de güncelle
        # (Eğer bazı resimler okunamayıp atlandıysa, ground_truth ile eşleşmez, program çöker)
        current_data_len = len(self.features["m1"])
        if current_data_len != len(self.ground_truth):
            print(
                f"[UYARI] Okunan resim ({current_data_len}) ile Ground Truth ({len(self.ground_truth)}) eşitleniyor..."
            )
            self.ground_truth = self.ground_truth[:current_data_len]

        total = len(self.ground_truth)
        indices = np.arange(total)
        # Her 3 resimden ilk 2'si TRAIN (Mod 0 ve 1), 3.'sü TEST (Mod 2)
        train_mask = indices % 3 != 2

        y_train = self.ground_truth[train_mask]
        y_test = self.ground_truth[~train_mask]

        experiments = [
            ("Method 1 (Pixels)", "m1", (10,)),
            ("Method 2 (Disk-4)", "m2_d4", (10,)),
            ("Method 2 (Disk-5)", "m2_d5", (10,)),
            ("Method 3 (Ratio)", "m3", (50,)),
        ]

        table_rows = []
        print("\n--- YAPAY SİNİR AĞI EĞİTİLİYOR ---")

        for name, key, hidden_nodes in experiments:
            X = self.features[key]
            X_train = X[train_mask]
            X_test = X[~train_mask]

            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)

            model = MLPRegressor(
                hidden_layer_sizes=hidden_nodes,
                activation="relu",
                solver="lbfgs",
                max_iter=5000,
                random_state=42,
            )
            model.fit(X_train_s, y_train)
            preds = model.predict(X_test_s)

            errors = np.abs(preds - y_test)
            err_pct = (errors / y_test) * 100

            avg_err = np.mean(err_pct)
            acc_10 = np.mean(err_pct <= 10) * 100
            acc_15 = np.mean(err_pct <= 15) * 100

            table_rows.append([name, avg_err, acc_10, acc_15])

            print(f"-> {name} tamamlandı. Grafik açılıyor...")
            self.plot_results(name, y_test, preds, err_pct)

        print("\n" + "=" * 75)
        print(
            f"{'METHOD':<25} {'AVG ERROR (%)':<15} {'ACC <10%':<15} {'ACC <15%':<15}"
        )
        print("-" * 75)
        for row in table_rows:
            print(
                f"{row[0]:<25} {row[1]:<15.2f} {row[2]:<15.1f} {row[3]:<15.1f}"
            )
        print("=" * 75)


if __name__ == "__main__":
    app = PeopleCounterAssignment(IMAGE_FOLDER, GT_FILE)
    if len(app.img_paths) > 0 and len(app.ground_truth) > 0:
        app.extract_features()
        app.run_experiments()
