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

# Gereksiz uyarıları terminalde gizle
warnings.filterwarnings("ignore")

# =============================================================================
# AYARLAR (SETTINGS)
# =============================================================================
IMAGE_FOLDER = r"C:\Users\emino\Desktop\İYTE\2025-2026\TERM-1\CENG391\ASSIGNMENTS\ASSIGNMENT-5\Automated-People-Counting\VideoDataset\ximagepixels"
GT_FILE = r"C:\Users\emino\Desktop\İYTE\2025-2026\TERM-1\CENG391\ASSIGNMENTS\ASSIGNMENT-5\Automated-People-Counting\VideoDataset\groundtruth.txt"
OUTPUT_DIR = "Odev_Sonuclari"


class PeopleCounterFinalReproduction:
    def __init__(self, img_folder, gt_file):
        self.img_folder, self.gt_file = img_folder, gt_file
        self.features, self.ground_truth, self.img_paths = {}, [], []

        # Klasör yönetimi: Kopyalamak yerine temizle ve üzerine yaz
        if os.path.exists(OUTPUT_DIR):
            shutil.rmtree(OUTPUT_DIR)
        os.makedirs(OUTPUT_DIR)

        print("\nCENG391 - AUTOMATED PEOPLE COUNTING (NİHAİ RAPOR ÇIKTISI)")
        print("-" * 65)
        self.load_data()

    def load_data(self):
        # Ground Truth okuma
        with open(self.gt_file, "r") as f:
            self.ground_truth = [
                int(line.strip()) for line in f if line.strip().isdigit()
            ]

        # Sadece .bmp dosyalarını doğal sıralı oku
        self.img_paths = glob.glob(os.path.join(self.img_folder, "*.bmp"))
        self.img_paths.sort(
            key=lambda f: [int(n) for n in re.findall(r"\d+", f)]
        )

        # Makale kuralı: 153 kare [cite: 121]
        limit = min(len(self.img_paths), len(self.ground_truth), 153)
        self.img_paths, self.ground_truth = self.img_paths[:limit], np.array(
            self.ground_truth[:limit]
        )
        print(f"-> {limit} adet veri eşleşti.")

    def extract_features(self):
        print("[İŞLEM] Makale özellik çıkarımı (X, C, S/X) yapılıyor...")
        f_m1, f_m2_d4, f_m2_d5, f_m3 = [], [], [], []
        d4 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        d5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        sq = cv2.getStructuringElement(
            cv2.MORPH_RECT, (2, 2)
        )  # Makale: 2x2 square [cite: 327]

        for path in self.img_paths:
            with open(path, "rb") as f:
                img = cv2.imdecode(
                    np.frombuffer(f.read(), np.uint8), cv2.IMREAD_GRAYSCALE
                )
            _, bin_img = cv2.threshold(
                img, 30, 255, cv2.THRESH_BINARY
            )  # Threshold: 30 [cite: 132]

            X = np.count_nonzero(bin_img)
            c_d4 = np.count_nonzero(
                cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, d4)
            )
            c_d5 = np.count_nonzero(
                cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, d5)
            )
            eroded = cv2.erode(
                bin_img, sq, iterations=2
            )  # Makale: 2 times erosion [cite: 111]
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
        print("[OK] Özellikler çıkarıldı.")

    def run_experiments(self):
        indices = np.arange(len(self.ground_truth))
        train_mask = (
            indices % 3 != 2
        )  # Her 3 kareden 2'si train (102 train, 51 test) [cite: 123]
        y_train, y_test = (
            self.ground_truth[train_mask],
            self.ground_truth[~train_mask],
        )

        # Deneyler [İsim, Key, Nöron, FigNo Başlangıcı]
        exps = [
            ("Method 1", "m1", 10, 5),
            ("Method 2 (disk-4)", "m2_d4", 10, 7),
            ("Method 2 (disk-5)", "m2_d5", 10, 9),
            ("Method 3", "m3", 50, 11),
        ]  # [cite: 136, 144, 328]

        table_results = []
        print("\n--- ANALİZ (10 Run Average) ---")

        for name, key, neurons, fig_start in exps:
            X_all = self.features[key]
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_all[train_mask])
            X_test_s = scaler.transform(X_all[~train_mask])

            err_accum, acc10_accum, acc15_accum = [], [], []
            best_tr_p, best_ts_p, best_ts_err = None, None, None

            for i in range(
                10
            ):  # Matlab rastgeleliğini önlemek için 10 test ortalaması [cite: 352]
                model = MLPRegressor(
                    hidden_layer_sizes=(neurons,),
                    activation="tanh",
                    solver="lbfgs",
                    max_iter=30000,
                    random_state=i,
                )
                model.fit(X_train_s, y_train)
                tr_preds, ts_preds = model.predict(X_train_s), model.predict(
                    X_test_s
                )

                pct = (np.abs(ts_preds - y_test) / y_test) * 100
                err_accum.append(np.mean(pct))
                acc10_accum.append(np.mean(pct <= 10) * 100)
                acc15_accum.append(np.mean(pct <= 15) * 100)

                if i == 0:
                    best_tr_p, best_ts_p, best_ts_err = tr_preds, ts_preds, pct

            # Grafikleri Üret
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
                + "=" * 65
                + "\n"
            )
            f.write("\n".join(table_results))

    def plot_training(self, name, f_num, X, y_true, y_pred):
        """Fig 5, 7, 9, 11: Eğitim ilişkisi (Scientific Notation ve Renkler)."""
        plt.figure(figsize=(9, 6))
        if name == "Method 3":
            # Fig 11 özel: İki panelli görünüm [cite: 330]
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            s1 = np.argsort(X[:, 0])
            s2 = np.argsort(X[:, 1])

            # Panel 1: C
            ax1.plot(X[s1, 0], y_true[s1], "b-", label="ground truth")
            ax1.plot(
                X[s1, 0],
                y_pred[s1],
                "r*",
                label="estimated count",
                markersize=4,
            )
            ax1.set_xlabel("Number of Closed Pixels (C)")
            ax1.set_ylabel("Number of People")
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.xaxis.set_major_formatter(
                ticker.ScalarFormatter(useMathText=True)
            )

            # Panel 2: S/X
            ax2.plot(X[s2, 1], y_true[s2], "b-", label="ground truth")
            ax2.plot(
                X[s2, 1],
                y_pred[s2],
                "r*",
                label="estimated count",
                markersize=4,
            )
            ax2.set_xlabel("Erosion Ratio (S/X)")
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            fig.suptitle(f"Figure {f_num}: {name} - Training Relationship")
            plt.savefig(
                f"{OUTPUT_DIR}/Fig{f_num}_{name.replace(' ', '_')}_Train.png"
            )
        else:
            s_idx = np.argsort(X[:, 0])
            plt.plot(X[s_idx, 0], y_true[s_idx], "b-", label="ground truth")
            plt.plot(
                X[s_idx, 0],
                y_pred[s_idx],
                "r*",
                label="estimated count",
                markersize=6,
            )
            plt.xlabel("Number of Pixels")
            plt.ylabel("Number of People")
            plt.title(f"Figure {f_num}: {name} - Training Relationship")
            plt.legend()
            plt.grid(True, alpha=0.3)
            # x10^4 formatı ekle
            plt.gca().xaxis.set_major_formatter(
                ticker.ScalarFormatter(useMathText=True)
            )
            plt.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
            plt.savefig(
                f"{OUTPUT_DIR}/Fig{f_num}_{name.replace(' ', '_')}_Train.png"
            )
        plt.close()

    def plot_test(self, name, f_num, gt, pred, err):
        """Fig 6, 8, 10, 12: Test performansı (1-51 Eksen)."""
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
        ax2.set_xlim(0, 60)  # [cite: 202, 235]
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            f"{OUTPUT_DIR}/Fig{f_num}_{name.replace(' ', '_')}_Test.png"
        )
        plt.close()


if __name__ == "__main__":
    app = PeopleCounterFinalReproduction(IMAGE_FOLDER, GT_FILE)
    app.extract_features()
    app.run_experiments()
    print(
        f"\n[TAMAMLANDI] Dosyalar '{OUTPUT_DIR}' klasöründe makale formatına uygun şekilde güncellendi."
    )
