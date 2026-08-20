"""ortho 投影特征聚合: mesh 表面采样点 -> ortho 像素 -> mask 特征 -> 多视图平均
输出: point_cloud.ply (RGB) + point_features.npy (4, N, 1152)
"""
import os, sys, glob, json, time
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import numpy as np
import trimesh
import open3d as o3d
from tqdm import tqdm

GLB_DIR = "/media/fernando/1282-0785/ZgyHZupdate/data/3dtiles/330102007/hubin"
RENDER_DIR = "data/hubin-e2e/render"
FEAT_DIR = "data/hubin-e2e/scene/language_features_highlight"
OUT_DIR = "eval/hubin"


def load_merged_mesh():
    geos, names = [], []
    glbs = sorted(glob.glob(os.path.join(GLB_DIR, "NoLod_*.glb")))
    for g_path in glbs:
        sc = trimesh.load(g_path, force='scene', process=False)
        for n, g_ in sc.geometry.items():
            if isinstance(g_, trimesh.Trimesh) and len(g_.faces) > 0:
                geos.append(g_)
    # concat 成单个 Trimesh
    offsets = np.cumsum([0] + [len(g.vertices) for g in geos[:-1]])
    verts = np.concatenate([g.vertices for g in geos])
    faces = np.concatenate([g.faces + int(offsets[i]) for i, g in enumerate(geos)])
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    return mesh


def sample_surface_points(mesh, n_points=300000):
    pts, tri_ix = trimesh.sample.sample_surface(mesh, n_points)
    return pts.astype(np.float32)


def load_calib():
    ps = np.load("data/hubin-e2e/depth_calib.npy")  # (z, depth) 标定点
    return ps[:, 1].copy(), ps[:, 0].copy()  # depth_sorted(降序?), z

def depth_to_z(depth_vals, calib):
    ds, zs = calib
    z = np.interp(depth_vals, ds[::-1], zs[::-1])  # ds 单调递减 -> 反转成递增
    return z


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    t0 = time.time()
    print("[1/4] 加载合并 mesh...")
    mesh = load_merged_mesh()
    print(f"  mesh: {len(mesh.vertices)} verts, {len(mesh.faces)} faces, "
          f"bounds {mesh.bounds[0]} -> {mesh.bounds[1]}")

    print("[2/4] 表面采样...")
    pts = sample_surface_points(mesh, 300000)
    print(f"  采样点: {len(pts)}")

    print("[3/4] 加载 tiles 元数据 + masks + features...")
    with open(os.path.join(RENDER_DIR, "tiles.json")) as f:
        tiles = json.load(f)
    # 只用 tile_ 图 (去掉 full, 分辨率过低)
    tiles = [t for t in tiles if t["tag"].startswith("tile_")]
    print(f"  使用 {len(tiles)} 张 tile 图")

    calib = load_calib()
    ds_, zs_ = calib
    W = 512; H = 512

    # 预加载全部 masks/features
    masks, feats = [], []
    np_imgs = []
    for t in tiles:
        tag = t["tag"]
        m = np.load(os.path.join(FEAT_DIR, f"{tag}_s.npy"))
        f = np.load(os.path.join(FEAT_DIR, f"{tag}_f.npy"))
        masks.append(m); feats.append(f)
        img = np.load(os.path.join(RENDER_DIR, f"{tag}_depth.npy"))
        np_imgs.append(img)
    n_levels, dim = feats[0].shape[1], feats[0].shape[1]
    # n_levels 来自 mask 第一维 (4), dim 来自 feature 第二维 (1152)
    n_levels = masks[0].shape[0]
    dim = feats[0].shape[1]
    print(f"  n_levels={n_levels}, dim={dim}")

    # 逐 tile 投影
    point_features_sum = np.zeros((n_levels, len(pts), dim), dtype=np.float32)
    n_observed = np.zeros((n_levels, len(pts)), dtype=np.int16)
    point_colors_sum = np.zeros((len(pts), 3), dtype=np.float32)

    pts3 = pts
    for ti, t in enumerate(tqdm(tiles, desc="tiles")):
        cx, cy, hx, hy, cam_z = t["cx"], t["cy"], t["half_x"], t["half_y"], t["cam_z"]
        depth_img = np_imgs[ti].astype(np.float32)
        mask = masks[ti]; feat = feats[ti]
        # 像素 -> world (ortho 线性)
        u = (pts3[:, 0] - cx) / (2 * hx) * W + (W - 1) / 2
        v = (H - 1) / 2 - (pts3[:, 1] - cy) / (2 * hy) * H
        ui = np.round(u).astype(np.int32)
        vi = np.round(v).astype(np.int32)
        inb = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
        if not inb.any():
            continue
        # 深度可见性
        ui_ = ui[inb]; vi_ = vi[inb]
        dvals = depth_img[vi_, ui_]
        vis = dvals > 0
        if not vis.any():
            continue
        # 逆深度 -> world z
        z_from_depth = depth_to_z(dvals[vis], calib)
        pt_z = pts3[inb][vis][:, 2]
        # 可见: 点 z 与表面 z 接近 (容差)
        z_tol = 6.0
        ok = np.abs(z_from_depth - pt_z) < z_tol
        if not ok.any():
            continue
        pix_u = ui_[vis][ok]; pix_v = vi_[vis][ok]
        idx = np.where(inb)[0][vis][ok]
        # 聚合特征: (4 levels) mask 索引
        m_ix = mask[:, pix_v, pix_u]           # (4, nok)
        for L in range(n_levels):
            msel = m_ix[L]
            valid = msel >= 0
            if not valid.any():
                continue
            sel = msel[valid]
            idx_sel = idx[valid]
            # features[sel] (nok, dim)
            fvals = feat[sel].astype(np.float32)
            # 避免重复 idx 多次相加 (同一像素多 level) -> 用 np.add.at 安全
            np.add.at(point_features_sum[L], idx_sel, fvals)
            np.add.at(n_observed[L], idx_sel, 1)
        # 颜色 (RGB tile 图)
        rgb = np.load_rgb(t) if False else None
    # 颜色采样单独一次 (加载 RGB jpg)
    img_rgb_all = []
    for t in tiles:
        from PIL import Image
        im = Image.open(os.path.join(RENDER_DIR, f"{t['tag']}.jpg")).convert("RGB")
        img_rgb_all.append(np.array(im))
    for ti, t in enumerate(tqdm(tiles, desc="color")):
        cx, cy, hx, hy = t["cx"], t["cy"], t["half_x"], t["half_y"]
        img = img_rgb_all[ti].astype(np.float32)
        u = (pts3[:, 0] - cx) / (2 * hx) * W + (W - 1) / 2
        v = (H - 1) / 2 - (pts3[:, 1] - cy) / (2 * hy) * H
        ui = np.round(u).astype(np.int32); vi = np.round(v).astype(np.int32)
        inb = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
        depth_img = np_imgs[ti].astype(np.float32)
        if inb.any():
            ui_ = ui[inb]; vi_ = vi[inb]
            dvals = depth_img[vi_, ui_]
            vis = dvals > 0
            if vis.any():
                idxs = np.where(inb)[0][vis]
                cols = img[vi_[vis], ui_[vis]]
                point_colors_sum[idxs] += cols
    n_col = (point_colors_sum.sum(1) > 0)
    point_colors_sum[n_col] = point_colors_sum[n_col] / 255.0

    # 平均特征
    point_features = point_features_sum / np.maximum(1, n_observed)[:, :, None]
    point_features = point_features.astype(np.float16)

    print("[4/4] 保存...")
    # 颜色: 未观察到的给灰色
    colors = point_colors_sum.copy()
    colors[~n_col] = [0.6, 0.6, 0.6]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(os.path.join(OUT_DIR, "hubin_point_cloud.ply"), pcd)
    np.save(os.path.join(OUT_DIR, "point_features_highlight.npy"), point_features)
    np.save(os.path.join(OUT_DIR, "mesh_points.npy"), pts)
    np.save(os.path.join(OUT_DIR, "n_observed.npy"), n_observed)

    covered = (n_observed.sum(0) > 0).mean()
    print(f"  点数: {len(pts)}, 有特征覆盖: {covered*100:.1f}%")
    print(f"  保存到 {OUT_DIR}")
    print(f"耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()