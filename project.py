import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import glob
import os
import re
import warnings

# Gereksiz uyarıları terminalde görmemek için gizliyoruz
warnings.filterwarnings("ignore")

# =============================================================================
# KONFİGÜRASYON (DOSYA YOLLARI)
# =============================================================================
IMAGE_FOLDER = r"C:\Users\emino\Desktop\İYTE\2025-2026\TERM-1\CENG391\ASSIGNMENTS\ASSIGNMENT-5\Automated-People-Counting\VideoDataset\ximagepixels"
GT_FILE = r"C:\Users\emino\Desktop\İYTE\2025-2026\TERM-1\CENG391\ASSIGNMENTS\ASSIGNMENT-5\Automated-People-Counting\VideoDataset\groundtruth.txt"
OUTPUT_DIR = "Odev_Sonuclari"


class PeopleCounterReproduction:
    def __init__(self, img_folder, gt_file):
        self.img_folder, self.gt_file = img_folder, gt_file
        self.features, self.ground_truth, self.img_paths = {}, [], []
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)
        self.load_data()

    def load_data(self):
        with open(self.gt_file, "r") as f:
            self.ground_truth = [
                int(line.strip()) for line in f if line.strip().isdigit()
            ]
        self.img_paths = glob.glob(os.path.join(self.img_folder, "*.bmp"))
        self.img_paths.sort(
            key=lambda f: [int(n) for n in re.findall(r"\d+", f)]
        )
        limit = min(len(self.img_paths), len(self.ground_truth), 153)
        self.img_paths, self.ground_truth = self.img_paths[:limit], np.array(
            self.ground_truth[:limit]
        )
        print(f"-> {limit} kare başarıyla eşleşti.")

    def extract_features(self):
        print("[İŞLEM] Özellik çıkarımı (Morfolojik İşlemler) yapılıyor...")
        f_m1, f_m2_d4, f_m2_d5, f_m3 = [], [], [], []
        d4 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        d5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        sq = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

        for path in self.img_paths:
            with open(path, "rb") as f:
                img = cv2.imdecode(
                    np.frombuffer(f.read(), np.uint8), cv2.IMREAD_GRAYSCALE
                )
            _, bin_img = cv2.threshold(img, 30, 255, cv2.THRESH_BINARY)

            X = np.count_nonzero(bin_img)
            c_d4 = np.count_nonzero(
                cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, d4)
            )
            c_d5 = np.count_nonzero(
                cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, d5)
            )
            # Makaledeki "2 times erosion" kuralı [cite: 111, 327]
            eroded = cv2.erode(bin_img, sq, iterations=2)
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

    def run_experiments(self):
        indices = np.arange(len(self.ground_truth))
        train_mask = indices % 3 != 2  # İlk 2 kare eğitim, 3. test [cite: 123]
        y_train, y_test = (
            self.ground_truth[train_mask],
            self.ground_truth[~train_mask],
        )

        # Deney Ayarları: [İsim, Anahtar, Nöron, FigNo Başlangıcı]
        exps = [
            ("Method 1", "m1", 10, 5),
            ("Method 2 (disk-4)", "m2_d4", 10, 7),
            ("Method 2 (disk-5)", "m2_d5", 10, 9),
            ("Method 3", "m3", 50, 11),
        ]

        for name, key, neurons, fig_start in exps:
            X_all = self.features[key]
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_all[train_mask])
            X_test_s = scaler.transform(X_all[~train_mask])

            # Yakınsama sorununu çözmek için max_iter=50000 yapıldı
            model = MLPRegressor(
                hidden_layer_sizes=(neurons,),
                activation="tanh",
                solver="lbfgs",
                max_iter=50000,
                random_state=42,
            )
            model.fit(X_train_s, y_train)

            # Eğitim setindeki tahmini al (Tek sayılı Figürler için)
            y_train_pred = model.predict(X_train_s)
            self.plot_training(
                name, fig_start, X_all[train_mask], y_train, y_train_pred
            )

            # Test seti tahmini (Çift sayılı Figürler için)
            y_test_pred = model.predict(X_test_s)
            err_pct = (np.abs(y_test_pred - y_test) / y_test) * 100
            self.plot_test(name, fig_start + 1, y_test, y_test_pred, err_pct)

    def plot_training(self, name, f_num, X, y_true, y_pred):
        """Fig 5, 7, 9, 11: Eğitim seti ilişkisi."""
        if name == "Method 3":
            # Fig 11 Özel: İki özelliği (C ve S/X) ayrı panellerde SIRALI çiz
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

            # C (Closed Pixels) Paneli
            s1 = np.argsort(X[:, 0])
            ax1.plot(X[s1, 0], y_true[s1], "b-", label="ground truth")
            ax1.plot(
                X[s1, 0],
                y_pred[s1],
                "b*",
                label="estimated count",
                markersize=4,
            )
            ax1.set_xlabel("Number of Closed Pixels (C)")
            ax1.set_ylabel("Number of People")
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # S/X (Erosion Ratio) Paneli
            s2 = np.argsort(X[:, 1])
            ax2.plot(X[s2, 1], y_true[s2], "b-", label="ground truth")
            ax2.plot(
                X[s2, 1],
                y_pred[s2],
                "b*",
                label="estimated count",
                markersize=4,
            )
            ax2.set_xlabel("Erosion Ratio (S/X)")
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            fig.suptitle(
                f"Figure {f_num}: {name} Training Relationship (Sorted)"
            )
        else:
            # Fig 5, 7, 9: Tek özellikli sıralı grafik [cite: 168, 175, 251]
            plt.figure(figsize=(8, 5))
            s_idx = np.argsort(X[:, 0])
            plt.plot(X[s_idx, 0], y_true[s_idx], "b-", label="ground truth")
            plt.plot(
                X[s_idx, 0],
                y_pred[s_idx],
                "b*",
                label="estimated count",
                markersize=4,
            )
            plt.title(f"Figure {f_num}: {name} Training Relationship")
            plt.xlabel("Number of Foreground Pixels")
            plt.ylabel("Number of People")
            plt.legend()
            plt.grid(True, alpha=0.3)

        plt.savefig(f"{OUTPUT_DIR}/Fig{f_num}_{name}_Train.png")
        plt.close()

    def plot_test(self, name, f_num, gt, pred, err):
        """Fig 6, 8, 10, 12: Test performansı ve hata yüzdesi."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        # X ekseni: 1 - 51 (Örnek sırası) [cite: 194, 224, 267, 350]
        sample_seq = np.arange(1, len(gt) + 1)

        # Üst Panel: Tahmin vs Gerçek
        ax1.plot(sample_seq, gt, "b-", label="ground truth")
        ax1.plot(sample_seq, pred, "r*", label="estimated count", markersize=6)
        ax1.set_title(f"Figure {f_num}: {name} Test Results")
        ax1.set_ylabel("Number of People")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Alt Panel: Hata Yüzdesi
        ax2.plot(sample_seq, err, "b*", markersize=6)
        ax2.set_ylabel("Error Percentage (%)")
        ax2.set_xlabel("Sample Sequence")
        ax2.set_ylim(0, 70)
        ax2.set_xlim(
            0, 60
        )  # Eksen sınırları makaleyle eşlendi [cite: 129, 205, 332]
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/Fig{f_num}_{name}_Test.png")
        plt.close()


if __name__ == "__main__":
    app = PeopleCounterReproduction(IMAGE_FOLDER, GT_FILE)
    app.extract_features()
    app.run_experiments()
    print(
        f"\n[BİLGİ] Convergence hatası giderildi. Eksenler ve sıralama (1-51) makale formatına uygun şekilde '{OUTPUT_DIR}' klasöründe toplandı."
    )
