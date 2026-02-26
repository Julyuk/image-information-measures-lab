# Завантаження зображення
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from google.colab import files

uploaded_files = files.upload()
file_name = list(uploaded_files.keys())[0]

# Конвертація в градації сірого
src_img = cv2.imread(file_name)
img_rgb = cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB)
img_gray = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)

# Функція розрахунку ентропії Шеннона
def calculate_shannon_entropy(matrix):
    hist = np.bincount(matrix.flatten(), minlength=256)
    p = hist / np.sum(hist)
    p_non_zero = p[p > 0]
    return -np.sum(p_non_zero * np.log2(p_non_zero))

# Функція сегментації та обчислення
def analyze_for_segment_size(gray, segment_size):
    h, w = gray.shape
    grid_h = h // segment_size
    grid_w = w // segment_size
    H_map = np.zeros((grid_h, grid_w))
    H_values = []

    for i in range(0, grid_h * segment_size, segment_size):
        for j in range(0, grid_w * segment_size, segment_size):
            seg = gray[i:i+segment_size, j:j+segment_size]
            H_seg = calculate_shannon_entropy(seg)
            gi = i // segment_size
            gj = j // segment_size
            H_map[gi, gj] = H_seg
            H_values.append(H_seg)

    return np.array(H_values), H_map

# Розрахунок для цілого зображення
H_total = calculate_shannon_entropy(img_gray)
print(f"Ентропія Шеннона всього зображення: {H_total}")

# Аналіз для різних розмірів сегмента
segment_sizes = [8, 16, 32]
stats = []

for seg_size in segment_sizes:
    H_values, H_map = analyze_for_segment_size(img_gray, seg_size)

    mean_val = np.mean(H_values)
    min_val = np.min(H_values)
    max_val = np.max(H_values)
    stats.append((min_val, mean_val, max_val))

    # Візуалізація карти та гістограми
    fig = plt.figure(figsize=(10, 4))

    ax1 = fig.add_subplot(1, 2, 1)
    ax1.set_title(f"Карта H, сегмент {seg_size}")
    im = ax1.imshow(H_map, cmap="gray")
    fig.colorbar(im, ax=ax1)
    ax1.axis("off")

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.set_title(f"Гістограма H, сегмент {seg_size}")
    ax2.hist(H_values, bins=30)
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

    print(f"=== Розмір сегмента {seg_size}x{seg_size} ===")
    print(f"Середнє значення: {mean_val}")
    print(f"Мінімум: {min_val}")
    print(f"Максимум: {max_val}\n")

# Побудова 3D діаграми порівняння
fig_3d = plt.figure(figsize=(11, 7))
ax_3d = fig_3d.add_subplot(111, projection='3d')

colors = ['#8E44AD', '#2980B9', '#27AE60']
x_labels = ['8x8', '16x16', '32x32']
y_labels = ['Мінімум', 'Середнє', 'Максимум']

for j in range(3):
    x_pos = np.arange(3)
    y_pos = np.ones(3) * j
    z_pos = np.zeros(3)

    dx = np.ones(3) * 0.45
    dy = np.ones(3) * 0.45
    dz = [stats[0][j], stats[1][j], stats[2][j]]

    ax_3d.bar3d(x_pos, y_pos, z_pos, dx, dy, dz, color=colors[j], alpha=0.85, shade=True)

ax_3d.view_init(elev=20, azim=-50)

ax_3d.set_xticks(np.arange(3) + 0.25)
ax_3d.set_xticklabels(x_labels)
ax_3d.set_yticks(np.arange(3) + 0.25)
ax_3d.set_yticklabels(y_labels)
ax_3d.set_zlabel('Значення')
ax_3d.set_title('Розподіл Ентропії Шеннона')

legend_patches = [mpatches.Patch(color=colors[i], label=y_labels[i]) for i in range(3)]
plt.legend(handles=legend_patches, loc='center left', bbox_to_anchor=(1.05, 0.5))

plt.show()
