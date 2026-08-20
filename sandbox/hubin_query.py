"""hubin 语义查询: 点云特征 x 文本提示 -> top-1 标签 + 可视化"""
import os, glob, numpy as np, json
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import torch
import open3d as o3d
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

import sys
sys.path.insert(0, "preprocessing")
from preprocess import SigLipNetwork

CLASSES = [
    # 用 'aerial view / seen from above' 措辞 + 每个类别多个 prompt 平均
    ("building", [
        "aerial view of a building rooftop, building seen from above",
        "top-down view of urban buildings, gray rooftop",
        "aerial photo of city buildings, rooftops of buildings",
    ]),
    ("road", [
        "aerial view of a road, asphalt street seen from above",
        "top-down view of roads and streets in a city",
        "aerial photo of parking lot or street, dark asphalt",
    ]),
    ("vegetation", [
        "aerial view of trees and vegetation, green park seen from above",
        "top-down view of trees and plants",
        "satellite view of green vegetation, forests and parks",
        "aerial photo of a green park with trees",
    ]),
    ("water", [
        "aerial view of water, swimming pool or lake seen from above",
        "top-down view of water surface, blue pool",
        "aerial photo of a pond or swimming pool",
    ]),
    ("ground", [
        "aerial view of bare ground, plaza or empty lot",
        "top-down view of dirt, soil, sand, empty plaza",
        "aerial photo of open bare ground",
    ]),
    ("empty", [
        "aerial view of empty open space, no buildings",
        "top-down view of unused open land",
        "aerial photo of open sky",
    ]),
]

CLASS_COLORS = {
    0: [0.85, 0.25, 0.25], 1: [0.45, 0.45, 0.45], 2: [0.10, 0.65, 0.25],
    3: [0.15, 0.45, 0.90], 4: [0.72, 0.55, 0.35], 5: [0.90, 0.88, 0.85],
}

def main():
    out = "eval/hubin"
    feats = np.load(os.path.join(out, "point_features_highlight.npy"))  # (4, N, 1152)
    n_levels, N, dim = feats.shape
    pts = np.load(os.path.join(out, "mesh_points.npy"))
    n_obs = np.load(os.path.join(out, "n_observed.npy"))

    # 有效点: 任意 level 有特征 (非全 zero)
    f = feats.astype(np.float32)
    norms = np.linalg.norm(f, axis=-1)  # (4, N)
    valid = (norms > 1e-4)  # (4, N)
    any_valid = valid.any(0)
    print(f"总点数 {N}, 有特征覆盖 {any_valid.mean()*100:.1f}%")
    for L in range(n_levels):
        print(f"  level {L}: 覆盖 {valid[L].mean()*100:.1f}%")

    # SigLIP 文本编码 (每类多个 prompt 平均)
    txt = SigLipNetwork("cuda")
    text_embeds = []
    for name, descs in CLASSES:
        with torch.no_grad():
            e = txt.encode_text(list(descs)).float().cpu().numpy()
            e = e / (np.linalg.norm(e, axis=-1, keepdims=True) + 1e-8)
            e_mean = e.mean(0)
            e_mean = e_mean / (np.linalg.norm(e_mean) + 1e-8)
        text_embeds.append(e_mean)
    text_embeds = np.stack(text_embeds).astype(np.float16)
    print("文本编码完成, shape:", text_embeds.shape)

    # 相似度: 对所有有效点、全部 level 取最大 (多层级聚合)
    scores = np.zeros((N, len(CLASSES)), dtype=np.float32)
    for L in range(n_levels):
        fL = f[L]  # (N, 1152)
        simL = fL @ text_embeds.T  # (N, ncls)
        scores = np.maximum(scores, simL)
    labels = scores.argmax(1).astype(np.int16)
    conf = scores.max(1)

    # 统计
    print("\n=== 语义分布 (覆盖点) ===")
    cv = any_valid
    for i, (name, _) in enumerate(CLASSES):
        m = cv & (labels == i)
        if m.sum() == 0:
            print(f"  {name:12s}: 0 个点")
            continue
        print(f"  {name:12s}: {m.sum():7d} 点 ({m.sum()/cv.sum()*100:5.1f}%)  平均conf={conf[m].mean():.4f}  平均sim={scores[m, i].mean():.4f}")
    print(f"\n平均 conf: {conf[cv].mean():.4f}")

    # 全部点(含无特征) 的 top1 分布
    print("\n=== 全部点 top1 分布(无特征点算 empty) ===")
    labels_all = labels.copy()
    labels_all[~cv] = 5  # empty
    for i, (name, _) in enumerate(CLASSES):
        m = (labels_all == i)
        print(f"  {name:12s}: {m.sum()/N*100:6.1f}%")

    np.save(os.path.join(out, "semantic_labels.npy"), labels_all)
    np.save(os.path.join(out, "semantic_conf.npy"), conf)

    # ============ 可视化 ============
    label_colors = CLASS_COLORS
    c_arr = np.zeros((N, 3))
    for i in label_colors:
        m = (labels_all == i)
        c_arr[m] = label_colors[i]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(c_arr)
    o3d.io.write_point_cloud(os.path.join(out, "semantic_visual.ply"), pcd)

    # 俯视散点图 (xy 平面, z 为高程)
    bb = pts.min(0), pts.max(0)
    print("\n点云 bounds:", bb[0], "->", bb[1])
    xs, ys, zs = pts[:, 0], pts[:, 1], pts[:, 2]
    fig, ax = plt.subplots(1, 3, figsize=(21, 7))
    # 采样展示 (5万点)
    idx = np.random.choice(N, min(50000, N), replace=False)
    for a, (x, y), ttl, cmap_, norm_ in [
        (ax[0], (xs[idx], ys[idx]), "Semantic top view (z=height)", None, None),
        (ax[1], (zs[idx], ys[idx]), "Semantic side view (x-z)", None, None),
        (ax[2], (xs[idx], zs[idx]), "Semantic side view (y-z)", None, None),
    ]:
        cc = c_arr[idx]
        a.scatter(x, y, c=cc, s=1.5, linewidths=0)
        a.set_title(ttl)
        a.set_aspect("equal", adjustable="box")
    plt.tight_layout()
    plt.savefig(os.path.join(out, "semantic_views.png"), dpi=110)
    plt.close()
    print("可视化保存:", os.path.join(out, "semantic_views.png"))

    # 建筑/植被/道路 覆盖热力图 (8x8 grid)
    gx = np.linspace(bb[0][0], bb[1][0], 9)
    gy = np.linspace(bb[0][1], bb[1][1], 9)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ai, (ci, ttl) in enumerate([(0, "Building coverage"), (2, "Vegetation coverage"), (1, "Road coverage")]):
        grid = np.zeros((8, 8))
        npts_grid = np.zeros((8, 8))
        for i in range(8):
            for j in range(8):
                m = (pts[:, 0] >= gx[i]) & (pts[:, 0] < gx[i+1]) & (pts[:, 1] >= gy[j]) & (pts[:, 1] < gy[j+1])
                if m.sum() > 0:
                    npts_grid[j, i] = m.sum()
                    grid[j, i] = (labels_all[m] == ci).mean()
        im = axes[ai].imshow(grid, cmap="Reds", origin="lower", extent=[gx[0], gx[-1], gy[0], gy[-1]], vmin=0, vmax=1)
        axes[ai].set_title(ttl + " (per 8x8 block)")
        axes[ai].set_xlabel("x (m)"); axes[ai].set_ylabel("y (m)")
        plt.colorbar(im, ax=axes[ai], fraction=0.046)
    plt.tight_layout()
    plt.savefig(os.path.join(out, "coverage_heatmaps.png"), dpi=110)
    plt.close()
    print("热力图保存:", os.path.join(out, "coverage_heatmaps.png"))

if __name__ == "__main__":
    main()