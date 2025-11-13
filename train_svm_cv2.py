import os
import cv2
import numpy as np

# =====================
# CẤU HÌNH
# =====================
DATA_DIR = "data"              # thư mục chứa train/val
IMG_SIZE = (64, 64)
MODEL_PATH = "svm_hog_cv2.xml"

# Tạo HOGDescriptor giống kích thước ảnh
hog = cv2.HOGDescriptor(
    _winSize=IMG_SIZE,
    _blockSize=(16, 16),
    _blockStride=(8, 8),
    _cellSize=(8, 8),
    _nbins=9
)

def extract_hog(gray_img):
    """gray_img: ảnh xám 64x64"""
    return hog.compute(gray_img).flatten()

def load_split(split="train"):
    split_dir = os.path.join(DATA_DIR, split)
    X, y = [], []

    class_names = sorted(os.listdir(split_dir))
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    print(f"[{split}] class mapping:", class_to_idx)

    for class_name in class_names:
        class_folder = os.path.join(split_dir, class_name)
        if not os.path.isdir(class_folder):
            continue

        label = class_to_idx[class_name]
        for fname in os.listdir(class_folder):
            if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
                continue

            fpath = os.path.join(class_folder, fname)
            img = cv2.imread(fpath)
            if img is None:
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, IMG_SIZE)

            feat = extract_hog(gray)
            X.append(feat)
            y.append(label)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    return X, y, class_to_idx

def train_svm_cv2():
    print("==> Load train data ...")
    X_train, y_train, class_to_idx = load_split("train")
    print("Train:", X_train.shape, y_train.shape)

    print("==> Load val data ...")
    X_val, y_val, _ = load_split("val")
    print("Val:", X_val.shape, y_val.shape)

    # Tạo SVM của OpenCV
    svm = cv2.ml.SVM_create()
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setKernel(cv2.ml.SVM_RBF)
    svm.setC(2.5)
    svm.setGamma(0.02)

    print("==> Training SVM ...")
    svm.train(X_train, cv2.ml.ROW_SAMPLE, y_train)

    # Đánh giá đơn giản trên val
    ret, y_pred = svm.predict(X_val)
    y_pred = y_pred.flatten().astype(np.int32)

    acc = (y_pred == y_val).sum() / len(y_val)
    print(f"Val accuracy: {acc:.4f}")

    # Lưu model và mapping class
    svm.save(MODEL_PATH)
    print(f"Đã lưu SVM vào {MODEL_PATH}")

    import json
    with open("class_mapping.json", "w", encoding="utf-8") as f:
        json.dump(class_to_idx, f, ensure_ascii=False, indent=2)
    print("Đã lưu class_mapping.json")

if __name__ == "__main__":
    train_svm_cv2()
