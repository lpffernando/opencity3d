"""虹桥: 用已生成的 per-level 点特征 → 开放词汇语义查询 (SigLIP)
每点: 4 levels 特征 → 各类多prompt相似度 → 平均 → 类别
"""
import os, sys
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import numpy as np, torch, time
import open3d as o3d
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, "preprocessing")
from preprocess import SigLipNetwork

CLASS_PROMPTS = {
    "building": [
        "aerial view of building rooftop", "buildings seen from above",
        "concrete and steel building, urban structure",
    ],
    "road": [
        "aerial view of road", "asphalt street",
        "parking lot or road surface seen from above",
    ],
    "vegetation": [
        "green trees and vegetation from above",
        "aerial view of park with trees and grass",
    ],
    "water": [
        "aerial view of blue water", "blue swimming pool, lake from above",
    ],
    "ground": [
        "bare ground, gravel, empty lot from above",
        "dirt and soil, plaza",
    ],
    "empty": [
        "empty open space from above", "open sky no features",
    ],
}
CLASS_COLORS = {
    "building": [0.85, 0.25, 0.25], "road": [0.45, 0.45, 0.45],
    "vegetation": [0.10, 0.65, 0.25], "water": [0.15, 0.45, 0.90],
    "ground": [0.72, 0.55, 0.35], "empty": [0.90, 0.88, 0.85],
}
CLASS_NAMES = list(CLASS_PROMPTS.keys())
N_CLASS = len(CLASS_NAMES)


def main():
    out = "eval/hongqiao"
    pts = np.load(os.path.join(out, "mesh_points.npy"))
    feats = np.load(os.path.join(out, "point_features_highlight.npy"))  # (4,N,1152)
    n_obs = np.load(os.path.join(out, "n_observed.npy"))
    N = len(pts)

    txt = SigLipNetwork("cuda")
    # 各类 prompt 平均编码
    embeds = []
    for name, plist in CLASS_PROMPTS.items():
        with torch.no_grad():
            e = txt.encode_text(plist).float()
        e = e / e.norm(dim=-1, keepdim=True)
        m = e.mean(0); m = m / (m.norm() + 1e-8)
        embeds.append(m.cpu().numpy())
    emb = np.stack(embeds).astype(np.float32)  # (N_CLASS, 1152)

    # 每点: 各 level 特征 -> 相似度 -> 取所有 level 有特征的最大/平均
    sim_all = np.zeros((N, N_CLASS), dtype=np.float32)
    cnt = np.zeros(N, dtype=np.float32)
    cov = n_obs > 0
    for L in range(4):
        fl = feats[L].astype(np.float32)
        valid = cov & (np.linalg.norm(fl, axis=-1) > 1e-4)
        if not valid.any():
            continue
        sim = fl[valid] @ emb.T  # (P, N_CLASS)
        sim_all[valid] += sim
        cnt[valid] += 1
    avg = sim_all / np.maximum(1, cnt[:, None])
    labels = avg.argmax(1).astype(np.int16)
    conf = avg.max(1)
    labels[~cov] = CLASS_NAMES.index("empty")

    print("\n=== 语义分布 (覆盖点) ===")
    cv = cov
    for i, name in enumerate(CLASS_NAMES):
        m = cv & (labels == i)
        if m.sum() == 0: print(f"  {name:12s}: 0"); continue
        print(f"  {name:12s}: {m.sum():6d} ({100*m.sum()/cv.sum():.1f}%) conf={avg[m,i].mean():.3f}")
    print(f"\n平均 conf: {avg[cv].mean():.3f}")

    np.save(os.path.join(out, "semantic_labels.npy"), labels)
    np.save(os.path.join(out, "semantic_scores.npy"), avg)
    np.save(os.path.join(out, "semantic_conf.npy"), conf)

    c_arr = np.array([CLASS_COLORS[CLASS_NAMES[i]] for i in labels])
    # 保存 PLY
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(c_arr)
    o3d.io.write_point_cloud(os.path.join(out, "semantic_hongqiao.ply"), pcd)

    # 可视化
    idx = np.arange(N)
    fig, axes = plt.subplots(1, 3, figsize=(21, 7))
    for a, (xx, yy), ttl in [
        (axes[0], (pts[idx,0], pts[idx,1]), "Top view (x-y)"),
        (axes[1], (pts[idx,0], pts[idx,2]), "Side view (x-z)"),
        (axes[2], (pts[idx,1], pts[idx,2]), "Side view (y-z)"),
    ]:
        a.scatter(xx, yy, c=c_arr[idx], s=2, linewidths=0)
        a.set_title(ttl); a.set_aspect("equal", adjustable="box")
    plt.tight_layout(); plt.savefig(os.path.join(out, "semantic_views.png"), dpi=120); plt.close()
    print("保存可视化:", out)


if __name__ == "__main__":
    main()