import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import glob
import os
import re
import warnings
import shutil

# Terminal kirliliğini önlemek için uyarıları kapatıyoruz
warnings.filterwarnings("ignore")

# =============================================================================
# KONFİGÜRASYON (DOSYA YOLLARI)
# =============================================================================
IMAGE_FOLDER = r"C:\Users\emino\Desktop\İYTE\2025-2026\TERM-1\CENG391\ASSIGNMENTS\ASSIGNMENT-5\Automated-People-Counting\VideoDataset\ximagepixels"
GT_FILE = r"C:\Users\emino\Desktop\İYTE\2025-2026\TERM-1\CENG391\ASSIGNMENTS\ASSIGNMENT-5\Automated-People-Counting\VideoDataset\groundtruth.txt"
OUTPUT_DIR = "Odev_Sonuclari"


class PeopleCounterFinalMerged:
    def __init__(self, img_folder, gt_file):
        self.img_folder = img_folder
        self.gt_file = gt_file
        self.features, self.ground_truth, self.img_paths = {}, [], []

        # Klasörü temizle ve yeniden oluştur (Overwrite - Eski dosyalar kopyalanmaz, silinip yenisi gelir)
        if os.path.exists(OUTPUT_DIR):
            shutil.rmtree(OUTPUT_DIR)
        os.makedirs(OUTPUT_DIR)

        print("\n" + "=" * 65)
        print("CENG391 - AUTOMATED PEOPLE COUNTING (NİHAİ BİRLEŞTİRİLMİŞ KOD)")
        print("=" * 65)
        self.load_data()

    def load_data(self):
        # Ground Truth okuma
        with open(self.gt_file, "r") as f:
            self.ground_truth = [
                int(line.strip()) for line in f if line.strip().isdigit()
            ]

        # .bmp dosyalarını doğal sıralı oku
        self.img_paths = glob.glob(os.path.join(self.img_folder, "*.bmp"))
        self.img_paths.sort(
            key=lambda f: [int(n) for n in re.findall(r"\d+", f)]
        )

        # Makale kuralı: 153 kare
        limit = min(len(self.img_paths), len(self.ground_truth), 153)
        self.img_paths, self.ground_truth = self.img_paths[:limit], np.array(
            self.ground_truth[:limit]
        )
        print(f"-> {limit} adet veri eşleşti.")

    def extract_features(self):
        print("[İŞLEM] Makale özellik çıkarımı (X, C, S/X) yapılıyor...")
        f_m1, f_m2_d4, f_m2_d5, f_m3 = [], [], [], []

        # Yapı Elemanları (Kernels)
        disk4 = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (9, 9)
        )  # Yarıçap 4
        disk5 = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (11, 11)
        )  # Yarıçap 5
        sq2x2 = cv2.getStructuringElement(
            cv2.MORPH_RECT, (2, 2)
        )  # Metod 3 erozyonu için

        for path in self.img_paths:
            with open(path, "rb") as f:
                img = cv2.imdecode(
                    np.frombuffer(f.read(), np.uint8), cv2.IMREAD_GRAYSCALE
                )

            # Threshold: 30
            _, bin_img = cv2.threshold(img, 30, 255, cv2.THRESH_BINARY)

            X = np.count_nonzero(bin_img)  # Metod 1
            c_d4 = np.count_nonzero(
                cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, disk4)
            )  # Metod 2 (disk-4)
            c_d5 = np.count_nonzero(
                cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, disk5)
            )  # Metod 2 (disk-5)

            # Method 3: Erosion for two times with 2x2 kernel
            eroded = cv2.erode(bin_img, sq2x2, iterations=2)
            ratio = (np.count_nonzero(eroded) / X) if X > 0 else 0

            f_m1.append([X])
            f_m2_d4.append([c_d4])
            f_m2_d5.append([c_d5])
            f_m3.append([c_d4, ratio])

        self.features = {
            "m1": np.array(f_m1),
            "m2_d4": np.array(f_m2_d4),
            "m2_d5": np.array(f_m2_d5),
            "m3": np.array(f_m3),
        }
        print("[OK] Özellikler hazırlandı.")

    def run_experiments(self):
        indices = np.arange(len(self.ground_truth))
        # Eğitim/Test Ayrımı (102 Eğitim / 51 Test)
        train_mask = indices % 3 != 2
        y_train, y_test = (
            self.ground_truth[train_mask],
            self.ground_truth[~train_mask],
        )

        # Deneyler [İsim, Anahtar, Nöron Sayısı, İlk Figür No]
        exps = [
            ("Method 1", "m1", 10, 5),
            ("Method 2 (disk-4)", "m2_d4", 10, 7),
            ("Method 2 (disk-5)", "m2_d5", 10, 9),
            ("Method 3", "m3", 50, 11),
        ]

        table_results = []
        print("\n--- ANALİZ (10 Run Average - Tablo I Üretiliyor) ---")

        for name, key, neurons, fig_start in exps:
            X_all = self.features[key]
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_all[train_mask])
            X_test_s = scaler.transform(X_all[~train_mask])

            err_accum, acc10_accum, acc15_accum = [], [], []
            best_tr_p, best_ts_p, best_ts_err = None, None, None

            # Metod 3 başarısı için 'logistic' aktivasyonu ve 'alpha' kullanımı birleştirildi
            for i in range(10):
                model = MLPRegressor(
                    hidden_layer_sizes=(neurons,),
                    activation="logistic",  # Sigmoid aktivasyonu (Metod 3 başarısı için kritik)
                    solver="lbfgs",
                    max_iter=80000,
                    alpha=0.01,  # Overfitting engelleme
                    random_state=i,
                )
                model.fit(X_train_s, y_train)
                tr_preds = model.predict(X_train_s)
                ts_preds = model.predict(X_test_s)

                pct = (np.abs(ts_preds - y_test) / y_test) * 100
                err_accum.append(np.mean(pct))
                acc10_accum.append(np.mean(pct <= 10) * 100)
                acc15_accum.append(np.mean(pct <= 15) * 100)

                if i == 0:
                    best_tr_p, best_ts_p, best_ts_err = tr_preds, ts_preds, pct

            # ŞEKİLLERİ ÜRET
            self.plot_training(
                name, fig_start, X_all[train_mask], y_train, best_tr_p
            )
            self.plot_test(name, fig_start + 1, y_test, best_ts_p, best_ts_err)

            res_str = f"{name:<20} | AvgErr: {np.mean(err_accum):.2f}% | Acc<10%: {np.mean(acc10_accum):.2f}% | Acc<15%: {np.mean(acc15_accum):.2f}%"
            print(res_str)
            table_results.append(res_str)

        # Tablo I Sonuçlarını Kaydet
        with open(os.path.join(OUTPUT_DIR, "Table_I_Results.txt"), "w") as f:
            f.write(
                "TABLE I - COMPARISON OF RESULTS (PAPER REPRODUCTION)\n"
                + "=" * 75
                + "\n"
            )
            f.write("\n".join(table_results))

    def plot_training(self, name, f_num, X, y_true, y_pred):
        """Eğitim İlişkisi: Mavi Çizgi (GT), Kırmızı Yıldız (EST). X-ekseni 10^4 formatı."""
        plt.figure(figsize=(9, 6))
        if name == "Method 3":
            # Fig 11 özel: İki panelli, artan sırada düzenlenmiş grafik
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            s1 = np.argsort(X[:, 0])  # C özelliğine göre sırala
            s2 = np.argsort(X[:, 1])  # S/X oranına göre sırala

            # Panel 1: C
            ax1.plot(X[s1, 0], y_true[s1], "b-", label="ground truth")
            ax1.plot(
                X[s1, 0],
                y_pred[s1],
                "r*",
                label="estimated count",
                markersize=4,
            )
            ax1.set_xlabel("Foreground Pixels after Closing (C)")
            ax1.set_ylabel("Number of People")
            ax1.xaxis.set_major_formatter(
                ticker.ScalarFormatter(useMathText=True)
            )
            ax1.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # Panel 2: S/X
            ax2.plot(X[s2, 1], y_true[s2], "b-", label="ground truth")
            ax2.plot(
                X[s2, 1],
                y_pred[s2],
                "r*",
                label="estimated count",
                markersize=4,
            )
            ax2.set_xlabel("S/X (Erosion Ratio)")
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            fig.suptitle(f"Figure {f_num}: {name} - Training Relationship")
            plt.savefig(
                f"{OUTPUT_DIR}/Fig{f_num}_{name.replace(' ', '_')}_Train.png"
            )
        else:
            # Fig 5, 7, 9
            s_idx = np.argsort(X[:, 0])
            plt.plot(X[s_idx, 0], y_true[s_idx], "b-", label="ground truth")
            plt.plot(
                X[s_idx, 0],
                y_pred[s_idx],
                "r*",
                label="estimated count",
                markersize=6,
            )
            plt.xlabel("Number of Foreground Pixels")
            plt.ylabel("Number of People")
            plt.title(f"Figure {f_num}: {name} - Training Relationship")
            plt.legend()
            plt.grid(True, alpha=0.3)
            # Bilimsel gösterim ekle (x10^4)
            plt.gca().xaxis.set_major_formatter(
                ticker.ScalarFormatter(useMathText=True)
            )
            plt.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
            plt.savefig(
                f"{OUTPUT_DIR}/Fig{f_num}_{name.replace(' ', '_')}_Train.png"
            )
        plt.close()

    def plot_test(self, name, f_num, gt, pred, err):
        """Test Performansı: 1-51 Sample Sequence."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 9))
        seq = np.arange(1, 52)

        # Üst Panel: Tahmin vs Gerçek
        ax1.plot(seq, gt, "b-", label="ground truth")
        ax1.plot(seq, pred, "r*", label="estimated count", markersize=7)
        ax1.set_ylabel("Number of People")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_title(f"Figure {f_num}: {name} - Test Results")

        # Alt Panel: Hata Yüzdesi
        ax2.plot(seq, err, "b*", markersize=7)
        ax2.set_ylabel("Error Percentage (%)")
        ax2.set_xlabel("Sample Sequence")
        ax2.set_ylim(0, 80)
        ax2.set_xlim(0, 60)  # Makale sınırı
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            f"{OUTPUT_DIR}/Fig{f_num}_{name.replace(' ', '_')}_Test.png"
        )
        plt.close()


if __name__ == "__main__":
    app = PeopleCounterFinalMerged(IMAGE_FOLDER, GT_FILE)
    app.extract_features()
    app.run_experiments()
    print(
        f"\n[TAMAMLANDI] Tüm grafikler (Fig 5-12) ve Tablo I sonuçları '{OUTPUT_DIR}' klasöründe üretildi."
    )
