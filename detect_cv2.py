import cv2
import numpy as np
import json

IMG_SIZE = (64, 64)
MODEL_PATH = "svm_hog_cv2.xml"

hog = cv2.HOGDescriptor(
    _winSize=IMG_SIZE,
    _blockSize=(16, 16),
    _blockStride=(8, 8),
    _cellSize=(8, 8),
    _nbins=9
)

def extract_hog(gray_img):
    return hog.compute(gray_img).flatten().astype(np.float32)

def load_model():
    svm = cv2.ml.SVM_load(MODEL_PATH)
    with open("class_mapping.json", "r", encoding="utf-8") as f:
        class_to_idx = json.load(f)
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    return svm, idx_to_class

def get_color_masks(hsv):
    # giống code trước của mình
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 70, 50])
    upper_red2 = np.array([180, 255, 255])
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)

    lower_blue = np.array([100, 70, 50])
    upper_blue = np.array([130, 255, 255])
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

    lower_yellow = np.array([20, 70, 50])
    upper_yellow = np.array([35, 255, 255])
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

    mask = cv2.bitwise_or(mask_red, mask_blue)
    mask = cv2.bitwise_or(mask, mask_yellow)
    return mask

def detect_image(image_path):
    svm, idx_to_class = load_model()

    img = cv2.imread(image_path)
    orig = img.copy()

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = get_color_masks(hsv)

    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 300:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        roi = orig[y:y+h, x:x+w]
        if roi.size == 0:
            continue

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, IMG_SIZE)

        feat = extract_hog(gray).reshape(1, -1)
        _, result = svm.predict(feat)
        label = int(result[0][0])
        class_name = idx_to_class[label]

        cv2.rectangle(orig, (x,y), (x+w,y+h), (0,255,0), 2)
        cv2.putText(orig, class_name, (x,y-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)

    cv2.imshow("Detection", orig)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_image("test.jpg")
