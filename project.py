import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
import glob
import os
import re
import warnings
import shutil

# hide warnings for cleaner output from terminal
warnings.filterwarnings("ignore")

# file paths
IMAGE_FOLDER = r"C:\Users\emino\Desktop\İYTE\2025-2026\TERM-1\CENG391\ASSIGNMENTS\ASSIGNMENT-5\Automated-People-Counting\VideoDataset\ximagepixels"
GT_FILE = r"C:\Users\emino\Desktop\İYTE\2025-2026\TERM-1\CENG391\ASSIGNMENTS\ASSIGNMENT-5\Automated-People-Counting\VideoDataset\groundtruth.txt"
OUTPUT_DIR = "Final_Output"


class PeopleCounterPaperReproduction:
    # Executes when the class is instantiated
    def __init__(self, img_folder, gt_file):
        self.img_folder, self.gt_file = img_folder, gt_file
        self.features, self.ground_truth, self.img_paths = {}, [], []

        # File management: Clear and recreate the output directory
        if os.path.exists(OUTPUT_DIR):
            shutil.rmtree(OUTPUT_DIR)
        os.makedirs(OUTPUT_DIR)

        print("\n" + "=" * 65)
        print("CENG391 - AUTOMATED PEOPLE COUNTING")
        print("=" * 65)
        self.load_data()

    # Loads images and ground truth data
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
        print(f"-> {limit} data points matched (102 Train, 51 Test).")

    # Extracts features from images
    def extract_features(self):
        print("[PROCESS] Extracting features (X, C, S/X) as per the paper...")
        f_m1, f_m2_d4, f_m2_d5, f_m3 = [], [], [], []

        # 2*radius+1(center pixel) for cv2 getStructuringElement
        d4 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        d5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))

        # 2x2 square kernel
        sq2 = np.ones((2, 2), np.uint8)

        for path in self.img_paths:
            # Load image in grayscale
            with open(path, "rb") as f:
                img = cv2.imdecode(
                    np.frombuffer(f.read(), np.uint8), cv2.IMREAD_GRAYSCALE
                )

            # Threshold at 30
            _, bin_img = cv2.threshold(img, 30, 255, cv2.THRESH_BINARY)

            # method 1: just count non-zero pixels
            X = np.count_nonzero(bin_img)
            # method 2: count closed pixels with disc structuring elements
            c_d4 = np.count_nonzero(
                cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, d4)
            )
            c_d5 = np.count_nonzero(
                cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, d5)
            )
            # method 3: erosion ratio
            eroded = cv2.erode(bin_img, sq2, iterations=2)
            # S/X Ratio
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
        print("[OK] Features extracted.")

    # Runs experiments as per the paper
    def run_experiments(self):
        # Split data into training and testing sets
        indices = np.arange(len(self.ground_truth))
        # First 2 out of every 3 frames are Training
        train_mask = indices % 3 != 2
        y_train, y_test = (
            self.ground_truth[train_mask],
            self.ground_truth[~train_mask],
        )
        # Experiment configurations: (Name, Feature Key, Neurons, Figure Start Number)
        exps = [
            ("Method 1", "m1", 10, 5),
            ("Method 2 (disc-4)", "m2_d4", 10, 7),
            ("Method 2 (disc-5)", "m2_d5", 10, 9),
            ("Method 3", "m3", 50, 11),
        ]

        table_results = []
        print("\n--- ANALYSIS (10 Run Average - Generating Table I) ---")

        # Run experiments for each method
        for name, key, neurons, fig_start in exps:
            # Prepare data
            X_all = self.features[key]
            # Scaling to 0-1 for Matlab standardization
            # Neural networks perform better with scaled data (pixels>>(0,1))
            scaler = MinMaxScaler()

            X_train_s = scaler.fit_transform(X_all[train_mask])
            X_test_s = scaler.transform(X_all[~train_mask])

            err_accum, acc10_accum, acc15_accum = [], [], []
            best_tr_p, best_ts_p, best_ts_err = None, None, None

            # 10 runs with different seeds
            # Because of random initialization in MLPRegressor
            # Averaging results
            for i in range(10):
                model = MLPRegressor(
                    hidden_layer_sizes=(
                        neurons,
                    ),  # more layer means more complex relations can be learned
                    activation=(
                        "logistic"
                    ),  # sigmoid: makes outputs between 0-1, when pixels are doubled number of people not doubled so this makes these kind of complex relations better to learn
                    solver=(
                        "lbfgs"
                    ),  # an algorithm to make error near 0, lbfgs is better for smaller datasets
                    max_iter=100000,  # work until convergence
                    alpha=(
                        0.1 if name == "Method 3" else 0.01
                    ),  # bigger alpha means more regularization, dont memorize data; learn and generalize better
                    random_state=i,
                )
                # Train and predict
                model.fit(X_train_s, y_train)
                tr_preds, ts_preds = model.predict(X_train_s), model.predict(
                    X_test_s
                )
                # Calculate error percentages
                pct = (np.abs(ts_preds - y_test) / y_test) * 100
                err_accum.append(np.mean(pct))
                acc10_accum.append(np.mean(pct <= 10) * 100)
                acc15_accum.append(np.mean(pct <= 15) * 100)

                if i == 0:
                    best_tr_p, best_ts_p, best_ts_err = tr_preds, ts_preds, pct

            # GENERATE FIGURES
            self.plot_training(
                name, fig_start, X_all[train_mask], y_train, best_tr_p
            )
            self.plot_test(name, fig_start + 1, y_test, best_ts_p, best_ts_err)

            res_str = f"{name:<20} | AvgErr: {np.mean(err_accum):.2f}% | Acc<10%: {np.mean(acc10_accum):.2f}% | Acc<15%: {np.mean(acc15_accum):.2f}%"
            print(res_str)
            table_results.append(res_str)

        with open(os.path.join(OUTPUT_DIR, "Table_I_Results.txt"), "w") as f:
            f.write(
                "TABLE I - COMPARISON OF RESULTS (10 Runs Average)\n"
                + "=" * 70
                + "\n"
            )
            f.write("\n".join(table_results))

    def plot_training(self, name, f_num, X, y_true, y_pred):
        """Eğitim İlişkisi: GT Mavi Çizgi, EST Kırmızı Nokta (Continuous Look)."""
        plt.figure(figsize=(9, 6))
        formatter = ticker.ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((4, 4))

        if name == "Method 3":
            # Fig 11 special: Two-panel plot arranged in ascending order
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            s1 = np.argsort(X[:, 0])
            s2 = np.argsort(X[:, 1])

            ax1.plot(
                X[s1, 0], y_true[s1], "b-", label="ground truth", alpha=0.8
            )
            ax1.plot(
                X[s1, 0],
                y_pred[s1],
                "r.",
                markersize=4,
                label="estimated count",
            )
            ax1.set_xlabel("Number of Closed Pixels (C)")
            ax1.set_ylabel("Number of People")
            ax1.xaxis.set_major_formatter(formatter)
            ax1.legend()
            ax1.grid(True, alpha=0.2)

            ax2.plot(
                X[s2, 1], y_true[s2], "b-", label="ground truth", alpha=0.8
            )
            ax2.plot(
                X[s2, 1],
                y_pred[s2],
                "r.",
                markersize=4,
                label="estimated count",
            )
            ax2.set_xlabel("Erosion Ratio (S/X)")
            ax2.legend()
            ax2.grid(True, alpha=0.2)
            fig.suptitle(
                f"Figure {f_num}: {name} - Training Relationship (Ascending Order)"
            )
            plt.savefig(
                f"{OUTPUT_DIR}/Fig{f_num}_{name.replace(' ', '_')}_Train.png"
            )
        else:
            s = np.argsort(X[:, 0])
            plt.plot(X[s, 0], y_true[s], "b-", label="ground truth", alpha=0.8)
            plt.plot(
                X[s, 0], y_pred[s], "r.", markersize=5, label="estimated count"
            )
            plt.xlabel("Number of Pixels")
            plt.ylabel("Number of People")
            plt.title(f"Figure {f_num}: {name} - Training Relationship")
            plt.legend()
            plt.grid(True, alpha=0.2)
            plt.gca().xaxis.set_major_formatter(formatter)
            plt.savefig(
                f"{OUTPUT_DIR}/Fig{f_num}_{name.replace(' ', '_')}_Train.png"
            )
        plt.close()

    def plot_test(self, name, f_num, gt, pred, err):
        """Fig 6, 8, 10, 12: Test performansı (1-51 Eksen)."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 9))
        seq = np.arange(1, 52)
        ax1.plot(seq, gt, "b-", label="ground truth", linewidth=1.2)
        ax1.plot(seq, pred, "r*", label="estimated count", markersize=6)
        ax1.set_ylabel("Number of People")
        ax1.legend()
        ax1.grid(True, alpha=0.2)
        ax1.set_title(f"Figure {f_num}: {name} - Test Results")
        ax2.plot(seq, err, "b*", markersize=6)
        ax2.set_ylabel("Error Percentage (%)")
        ax2.set_xlabel("Sample Sequence")
        ax2.set_ylim(0, 80)
        ax2.set_xlim(0, 60)
        ax2.grid(True, alpha=0.2)
        plt.tight_layout()
        plt.savefig(
            f"{OUTPUT_DIR}/Fig{f_num}_{name.replace(' ', '_')}_Test.png"
        )
        plt.close()


if __name__ == "__main__":
    app = PeopleCounterPaperReproduction(IMAGE_FOLDER, GT_FILE)
    app.extract_features()
    app.run_experiments()
    print(
        f"\n[COMPLETED] Results and figures are saved in the '{OUTPUT_DIR}' directory."
    )
