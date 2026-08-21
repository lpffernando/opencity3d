"""虹桥: mesh 顶点 + ortho top-down 可见性 → 语义点云
对每 tile: 顶点投影到像素, 该像素列最高z顶点可见; 从 SAM mask 聚合 SigLIP 特征
绕开 pyrender depth 标定 (用几何最高z判断遮挡)
"""
import os, sys, json, glob
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import numpy as np, trimesh, time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    out = "eval/hongqiao"
    os.makedirs(out, exist_ok=True)

    scene = trimesh.load("data/hongqiao-e2e/area.glb", force='scene', process=False)
    geos = [g for g in scene.geometry.values() if isinstance(g, trimesh.Trimesh) and len(g.faces) > 0]
    pts = np.concatenate([np.asarray(g.vertices) for g in geos]).astype(np.float64)
    print("mesh 顶点:", len(pts), "量级:", pts.min(0), pts.max(0))

    # tile 元数据
    meta = json.load(open("data/hongqiao-e2e/render/tiles.json"))
    render_dir = "data/hongqiao-e2e/render"
    feat_dir = "data/hongqiao-e2e/scene/language_features_highlight"
    W = H = 512

    # 从任一 _s.npy 得到 dims
    probe = np.load(os.path.join(feat_dir, "tile_0_0_s.npy"))
    n_levels = probe.shape[0]
    probe_f = np.load(os.path.join(feat_dir, "tile_0_0_f.npy"))
    feat_dim = probe_f.shape[1]
    print(f"levels={n_levels} feat_dim={feat_dim}")

    N = len(pts)
    point_feats = np.zeros((n_levels, N, feat_dim), dtype=np.float16)
    n_obs = np.zeros(N, dtype=np.int16)

    for ti, t in enumerate(meta):
        tag = t["tag"]
        if not tag.startswith("tile_"):
            continue
        cx, cy, hx, hy = t["cx"], t["cy"], t["half_x"], t["half_y"]
        # 顶点 -> 像素
        u = (pts[:, 0] - cx) / (2 * hx) * W + (W - 1) / 2
        v = (H - 1) / 2 - (pts[:, 1] - cy) / (2 * hy) * H
        ui = np.round(u).astype(np.int32)
        vi = np.round(v).astype(np.int32)
        inb = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
        if not inb.any():
            continue
        idx = np.where(inb)[0]
        uii, vii = ui[inb], vi[inb]
        # 该像素列的最高 z (top-down 可见: 顶点z >= 该列max_z - tol 视为可见)
        # 用唯一(u,v)键
        key = vii * W + uii
        # 求每像素 max z
        pix_max = {}
        # 为了速度, 排序
        order = np.argsort(key, kind='stable')
        keys_s = key[order]; z_s = pts[idx, 2][order]
        # 分组 max
        uniq, starts, counts = np.unique(keys_s, return_index=True, return_counts=True)
        # maxz per uniq using np.maximum.reduceat
        zm = np.maximum.reduceat(z_s, starts)
        maxz_map = dict(zip(uniq, zm))
        maxz = np.array([maxz_map[k] for k in key])
        tol = 3.0  # m 容差
        vis = -pts[idx, 2] <= -maxz + tol  # idx 点 z >= maxz - tol
        sel = idx[vis]
        if len(sel) == 0:
            continue
        # 读 mask
        mask = np.load(os.path.join(feat_dir, f"{tag}_s.npy"))  # (levels, H, W) int32
        masks_feat = np.load(os.path.join(feat_dir, f"{tag}_f.npy"))  # (n_masks, feat_dim)
        pv, pu = vii[vis], uii[vis]
        point_count = np.add.at(n_obs, sel, 1)
        for L in range(n_levels):
            uid_arr = mask[L, pv, pu]  # per selected point
            valid = uid_arr >= 0
            if not valid.any():
                continue
            mask_ids = uid_arr[valid]
            sel_pts = sel[valid]
            # 对每个 mask id, 加入对应特征
            for uid in np.unique(mask_ids):
                sel_in_uid = sel_pts[mask_ids == uid]
                point_feats[L, sel_in_uid, :] = masks_feat[uid]
    # 统计
    covered = n_obs > 0
    print(f"\n特征覆盖点: {covered.sum()} / {N} = {100*covered.mean():.1f}%")
    print(f"平均每点观测 tile 数: {n_obs[covered].mean():.2f}")

    np.save(os.path.join(out, "mesh_points.npy"), pts)
    np.save(os.path.join(out, "point_features_highlight.npy"), point_feats)
    np.save(os.path.join(out, "n_observed.npy"), n_obs)

    # 写带 RGB 的点云 (用 mesh 顶点无需纹理, 简单灰度或用渲染色)
    pcd = trimesh.PointCloud(vertices=pts)
    pcd.export(os.path.join(out, "hongqiao_point_cloud.ply"))
    print("保存:", out)


if __name__ == "__main__":
    main()