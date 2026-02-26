import cv2
import numpy as np
import matplotlib.pyplot as plt


# =========================
# 1. Власна конвертація RGB → Gray
# =========================
def rgb_to_gray_manual(bgr_img):
    b = bgr_img[:, :, 0].astype(np.float32)
    g = bgr_img[:, :, 1].astype(np.float32)
    r = bgr_img[:, :, 2].astype(np.float32)

    gray = 0.114*b + 0.587*g + 0.299*r
    return np.clip(gray, 0, 255).astype(np.uint8)


# =========================
# 2. Квантування
# =========================
def quantize_gray(gray, levels=16):
    q = (gray.astype(np.int32) * levels) // 256
    q[q == levels] = levels - 1
    return q


# =========================
# 3. Марковська ентропія 1-го порядку
# =========================
def markov_entropy_first_order(gray_levels, levels):

    h, w = gray_levels.shape
    C = np.zeros((levels, levels), dtype=np.int64)

    # Горизонтальні переходи
    left = gray_levels[:, :-1].ravel()
    right = gray_levels[:, 1:].ravel()
    np.add.at(C, (left, right), 1)

    # Вертикальні переходи
    up = gray_levels[:-1, :].ravel()
    down = gray_levels[1:, :].ravel()
    np.add.at(C, (up, down), 1)

    total = C.sum()
    if total == 0:
        return 0.0

    row_sums = C.sum(axis=1)
    pi = row_sums / total

    H = 0.0
    for i in range(levels):
        if row_sums[i] == 0:
            continue
        p_row = C[i] / row_sums[i]
        nz = p_row > 0
        H_i = -np.sum(p_row[nz] * np.log2(p_row[nz]))
        H += pi[i] * H_i

    return float(H)


# =========================
# 4. Сегментація
# =========================
def segment_image(gray, segment_size):

    h, w = gray.shape
    segments = []

    for i in range(0, h - segment_size + 1, segment_size):
        for j in range(0, w - segment_size + 1, segment_size):
            seg = gray[i:i+segment_size, j:j+segment_size]
            segments.append((seg, i, j))

    return segments


# =========================
# 5. Аналіз для одного розміру сегмента
# =========================
def analyze_for_segment_size(gray_q, segment_size, levels):

    segments = segment_image(gray_q, segment_size)

    h, w = gray_q.shape
    grid_h = h // segment_size
    grid_w = w // segment_size

    H_map = np.zeros((grid_h, grid_w))
    H_values = []

    for seg, i, j in segments:
        H_seg = markov_entropy_first_order(seg, levels)
        gi = i // segment_size
        gj = j // segment_size
        H_map[gi, gj] = H_seg
        H_values.append(H_seg)

    return np.array(H_values), H_map


# =========================
# 6. Основний запуск
# =========================
def run_analysis(image_path):

    bgr = cv2.imread(image_path)
    gray = rgb_to_gray_manual(bgr)
    gray_q = quantize_gray(gray, levels=16)

    H_total = markov_entropy_first_order(gray_q, levels=16)
    print("Марковська ентропія всього зображення:", H_total)

    segment_sizes = [8, 16, 32]

    for seg_size in segment_sizes:

        H_values, H_map = analyze_for_segment_size(gray_q, seg_size, levels=16)

        print(f"\n=== Розмір сегмента {seg_size}×{seg_size} ===")
        print("Середнє значення:", np.mean(H_values))
        print("Мінімум:", np.min(H_values))
        print("Максимум:", np.max(H_values))

        # Карта
        plt.figure(figsize=(10,4))
        plt.subplot(1,2,1)
        plt.title(f"Карта H, сегмент {seg_size}")
        plt.imshow(H_map, cmap="gray")
        plt.colorbar()
        plt.axis("off")

        # Гістограма
        plt.subplot(1,2,2)
        plt.title(f"Гістограма H, сегмент {seg_size}")
        plt.hist(H_values, bins=30)
        plt.grid(True)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    run_analysis("I04.BMP")
