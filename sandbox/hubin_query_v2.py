"""hubin 改进查询: per-mask (per-level) 独立 SigLIP 分类 + 拒绝过小 mask + 多视图 + per-tile 兜底"""
import os, sys, glob, json
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import numpy as np, torch, time
from PIL import Image
from tqdm import tqdm
import open3d as o3d
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "preprocessing")
from preprocess import SigLipNetwork

CLASS_PROMPTS = {
    "building": [
        "aerial view of a building rooftop, building seen from above",
        "top-down view of urban buildings, gray rooftop",
        "aerial photograph of city buildings",
    ],
    "road": [
        "aerial view of a road, asphalt street seen from above",
        "top-down view of roads and streets in a city",
        "aerial photo of parking lot or street, dark asphalt",
    ],
    "vegetation": [
        "aerial view of trees and vegetation, green park seen from above",
        "top-down view of trees and plants",
        "green vegetation from above, satellite image of green trees",
        "aerial photograph of a green park with trees",
    ],
    "water": [
        "aerial view of water, blue swimming pool seen from above",
        "top-down view of water surface, blue pool",
        "aerial photo of a pond or swimming pool, blue water",
    ],
    "ground": [
        "aerial view of bare ground, plaza or empty lot",
        "top-down view of dirt, soil, sand, empty plaza",
        "aerial photo of open bare ground, dirt plaza",
    ],
    "empty": [
        "aerial view of empty open space, no buildings",
        "top-down view of unused open land",
        "aerial photo of open sky, no features",
    ],
}
CLASS_COLORS = {
    "building":  [0.85, 0.25, 0.25],
    "road":      [0.45, 0.45, 0.45],
    "vegetation":[0.10, 0.65, 0.25],
    "water":     [0.15, 0.45, 0.90],
    "ground":    [0.72, 0.55, 0.35],
    "empty":    [0.90, 0.88, 0.85],
}
CLASS_NAMES = list(CLASS_PROMPTS.keys())
N_CLASS = len(CLASS_NAMES)


def encode_texts(txt, model_dim):
    """对每类 prompt 列表平均, 归一化"""
    embeds = []
    for name, plist in CLASS_PROMPTS.items():
        with torch.no_grad():
            e = txt.encode_text(plist).float()
        e = e / e.norm(dim=-1, keepdim=True)
        m = e.mean(0)
        m = m / (m.norm() + 1e-8)
        embeds.append(m.cpu().numpy())
    return np.stack(embeds).astype(np.float16)


def per_mask_scores(img_arr, mask_arr, txt, text_embeds_np,
                    min_mask_pixels=300, max_mask_pixels=None):
    """对每个 level 独立: 每个 mask crop -> siglip -> per-mask 类别分数
    返回: dict[level] -> dict[mask_id] -> (n_class,) score"""
    img_pil = Image.fromarray(img_arr)
    n_levels = mask_arr.shape[0]
    out = {}
    for L in range(n_levels):
        out[L] = {}
        for uid in np.unique(mask_arr[L]):
            if uid < 0:
                continue
            m = (mask_arr[L] == uid)
            if m.sum() < min_mask_pixels:
                continue
            if max_mask_pixels is not None and m.sum() > max_mask_pixels:
                continue  # 跳过全图大 mask（避免信号稀释）
            ys, xs = np.where(m)
            y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
            crop = img_arr[y0:y1+1, x0:x1+1]
            crop_pil = Image.fromarray(crop).resize((384, 384))
            arr_t = torch.from_numpy((np.array(crop_pil).astype(float)/255).transpose(2,0,1)[None]).half().to("cuda")
            with torch.no_grad(), torch.cuda.amp.autocast():
                e = txt.model.encode_image(arr_t).float()
            e = e / e.norm(dim=-1, keepdim=True)
            e_np = e.cpu().numpy()[0].astype(np.float16)
            score = e_np @ text_embeds_np.T  # (n_class,)
            out[L][uid] = score
    return out


def per_tile_score(img_arr, txt, text_embeds_np):
    """整图 -> siglip -> 类别分数"""
    img_pil = Image.fromarray(img_arr).resize((384, 384))
    arr_t = torch.from_numpy((np.array(img_pil).astype(float)/255).transpose(2,0,1)[None]).half().to("cuda")
    with torch.no_grad(), torch.cuda.amp.autocast():
        e = txt.model.encode_image(arr_t).float()
    e = e / e.norm(dim=-1, keepdim=True)
    e_np = e.cpu().numpy()[0].astype(np.float16)
    return e_np @ text_embeds_np.T


def main():
    out = "eval/hubin"
    pts = np.load(os.path.join(out, "mesh_points.npy"))
    feats_path = os.path.join(out, "point_features_highlight.npy")
    feats = np.load(feats_path)  # (4, N, 1152) 备用
    N = len(pts)

    # 加载 tile 元数据
    with open("data/hubin-e2e/render/tiles.json") as f:
        tiles_meta = json.load(f)
    tiles_meta = [t for t in tiles_meta if t["tag"].startswith("tile_")]
    render_dir = "data/hubin-e2e/render"
    feat_dir = "data/hubin-e2e/scene/language_features_highlight"

    # SigLIP 编码
    print("[1/4] SigLIP 文本编码...")
    txt = SigLipNetwork("cuda")
    text_embeds = encode_texts(txt, 1152)
    print("  text embeds:", text_embeds.shape)

    # 预计算: 每张 tile 的 (per-mask score, per-tile score)
    print("[2/4] 计算 per-mask + per-tile SigLIP 分数 (每张 tile)...")
    tile_mask_scores = []   # list of dict[level][uid] -> (n_class,)
    tile_full_score = []    # list of (n_class,)
    for t in tqdm(tiles_meta, desc="tiles"):
        tag = t["tag"]
        mask = np.load(os.path.join(feat_dir, f"{tag}_s.npy"))
        img = np.array(Image.open(os.path.join(render_dir, f"{tag}.jpg")).convert("RGB"))
        m_scores = per_mask_scores(img, mask, txt, text_embeds,
                                    min_mask_pixels=500,
                                    max_mask_pixels=200000)
        f_score = per_tile_score(img, txt, text_embeds)
        tile_mask_scores.append(m_scores)
        tile_full_score.append(f_score)
    print("  完成")

    # 投影: 每个 point -> 每张 tile -> 取分数
    print("[3/4] 点云分类 (per-mask 优先, per-tile 兜底)...")
    W = H = 512
    point_score_sum = np.zeros((N, N_CLASS), dtype=np.float32)
    n_obs = np.zeros(N, dtype=np.int16)
    n_calib = np.load(os.path.join(out, "n_observed.npy"))
    calib = (np.load("data/hubin-e2e/depth_calib.npy")[:, 1][::-1].copy(),
             np.load("data/hubin-e2e/depth_calib.npy")[:, 0][::-1].copy())
    def depth_to_z(d):
        return np.interp(d, calib[0], calib[1])

    for ti, t in enumerate(tqdm(tiles_meta, desc="classify")):
        cx, cy, hx, hy, cam_z = t["cx"], t["cy"], t["half_x"], t["half_y"], t["cam_z"]
        depth_img = np.load(os.path.join(render_dir, f"{t['tag']}_depth.npy")).astype(np.float32)
        u = (pts[:, 0] - cx) / (2 * hx) * W + (W - 1) / 2
        v = (H - 1) / 2 - (pts[:, 1] - cy) / (2 * hy) * H
        ui = np.round(u).astype(np.int32)
        vi = np.round(v).astype(np.int32)
        inb = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
        if not inb.any():
            continue
        ui_ = ui[inb]; vi_ = vi[inb]
        dvals = depth_img[vi_, ui_]
        vis = dvals > 0
        if not vis.any():
            continue
        z_from_depth = depth_to_z(dvals[vis])
        z_tol = 8.0
        ok = np.abs(z_from_depth - pts[inb][vis][:, 2]) < z_tol
        if not ok.any():
            continue
        pix_v = vi_[vis][ok]; pix_u = ui_[vis][ok]
        idxs = np.where(inb)[0][vis][ok]

        # mask lookup
        mask = np.load(os.path.join(feat_dir, f"{t['tag']}_s.npy"))
        m_scores = tile_mask_scores[ti]
        n_assigned = 0
        # 收集每个 point 在每个 level 的 mask 分数, 后面平均
        per_pt_scores = np.zeros((len(idxs), N_CLASS), dtype=np.float32)
        per_pt_weights = np.zeros(len(idxs), dtype=np.float32)
        for L in range(4):
            uid_arr = mask[L, pix_v, pix_u]
            for k in range(len(idxs)):
                u_id = int(uid_arr[k])
                if u_id in m_scores[L]:
                    s = m_scores[L][u_id]
                    per_pt_scores[k] += s
                    per_pt_weights[k] += 1.0
        # 没 mask 时用 per-tile 兜底
        ft = tile_full_score[ti]
        miss = per_pt_weights == 0
        if miss.any():
            per_pt_scores[miss] = ft
            per_pt_weights[miss] = 0.5  # 兜底权小
        per_pt_scores = per_pt_scores / np.maximum(1.0, per_pt_weights[:, None])
        # 累加
        np.add.at(point_score_sum, idxs, per_pt_scores)
        np.add.at(n_obs, idxs, 1)
    print("  完成")

    # 最终 label
    avg_scores = point_score_sum / np.maximum(1, n_obs[:, None])
    avg_scores[n_obs == 0] = 0
    labels = avg_scores.argmax(1).astype(np.int16)
    conf = avg_scores.max(1)
    labels[n_obs == 0] = CLASS_NAMES.index("empty")  # 无观察点 -> empty
    labels_str = np.array([CLASS_NAMES[i] for i in labels])

    # 统计
    print("[4/4] 统计 + 可视化...")
    print("\n=== 语义分布 (覆盖点) ===")
    cv = n_obs > 0
    for i, name in enumerate(CLASS_NAMES):
        m = cv & (labels == i)
        if m.sum() == 0:
            print(f"  {name:12s}: 0"); continue
        print(f"  {name:12s}: {m.sum():7d} ({m.sum()/cv.sum()*100:5.1f}%) avg sim={avg_scores[m, i].mean():.4f}")
    print(f"\n平均 conf: {conf[cv].mean():.4f}")
    print("\n=== 全点 (无特征点=empty) ===")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {name:12s}: {(labels==i).sum()/N*100:6.1f}%")

    np.save(os.path.join(out, "semantic_labels_v2.npy"), labels)
    np.save(os.path.join(out, "semantic_scores.npy"), avg_scores)
    np.save(os.path.join(out, "semantic_conf_v2.npy"), conf)

    # 可视化
    c_arr = np.array([CLASS_COLORS[CLASS_NAMES[i]] for i in labels])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(c_arr)
    o3d.io.write_point_cloud(os.path.join(out, "semantic_visual_v2.ply"), pcd)

    # 俯视图 + 侧视图
    bb = pts.min(0), pts.max(0)
    fig, ax = plt.subplots(1, 3, figsize=(21, 7))
    idx = np.random.choice(N, min(80000, N), replace=False)
    xs, ys, zs = pts[:, 0], pts[:, 1], pts[:, 2]
    for a, (xx, yy), ttl in [
        (ax[0], (xs[idx], ys[idx]), "Top view (x-y)"),
        (ax[1], (xs[idx], zs[idx]), "Side view (x-z)"),
        (ax[2], (ys[idx], zs[idx]), "Side view (y-z)"),
    ]:
        a.scatter(xx, yy, c=c_arr[idx], s=1.5, linewidths=0)
        a.set_title(ttl); a.set_aspect("equal", adjustable="box")
    plt.tight_layout()
    plt.savefig(os.path.join(out, "semantic_views_v2.png"), dpi=120); plt.close()

    # 类别热力图
    gx = np.linspace(bb[0][0], bb[1][0], 9)
    gy = np.linspace(bb[0][1], bb[1][1], 9)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    for ai, name in enumerate(CLASS_NAMES):
        ax = axes[ai//3, ai%3]
        grid = np.zeros((8, 8))
        for i in range(8):
            for j in range(8):
                m = (pts[:, 0] >= gx[i]) & (pts[:, 0] < gx[i+1]) & (pts[:, 1] >= gy[j]) & (pts[:, 1] < gy[j+1])
                if m.sum() > 50:
                    grid[j, i] = (labels[m] == ai).mean()
        im = ax.imshow(grid, cmap="Reds", origin="lower", extent=[gx[0], gx[-1], gy[0], gy[-1]], vmin=0, vmax=max(grid.max(), 0.1))
        ax.set_title(f"{name} coverage (per 8x8 block)")
        ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
        plt.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    plt.savefig(os.path.join(out, "coverage_heatmaps_v2.png"), dpi=120); plt.close()

    # 类别图例图
    fig, ax = plt.subplots(1, 1, figsize=(10, 1))
    handles = [plt.Rectangle((0, 0), 1, 1, color=CLASS_COLORS[n]) for n in CLASS_NAMES]
    ax.legend(handles, CLASS_NAMES, loc="center", ncol=6, frameon=False, fontsize=11)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(out, "class_legend.png"), dpi=120); plt.close()

    # 每张 tile 的语义图
    fig, axes = plt.subplots(4, 4, figsize=(20, 20))
    for ti, t in enumerate(tiles_meta):
        ax = axes[ti//4, ti%4]
        cx, cy, hx, hy = t["cx"], t["cy"], t["half_x"], t["half_y"]
        m = ((pts[:, 0] >= cx-hx) & (pts[:, 0] < cx+hx) & (pts[:, 1] >= cy-hy) & (pts[:, 1] < cy+hy))
        if m.sum() < 10:
            ax.set_title(t["tag"]+" (empty)"); continue
        sub = pts[m]
        sub_c = c_arr[m]
        sub_lab = labels[m]
        # 只画覆盖点
        sub_obs = n_obs[m] > 0
        ax.scatter(sub[sub_obs, 0], sub[sub_obs, 1], c=sub_c[sub_obs], s=2, linewidths=0)
        ax.set_title(t["tag"] + f" ({sub.sum()} pts)")
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(os.path.join(out, "semantic_tiles.png"), dpi=110); plt.close()

    print("保存:", os.path.join(out, "semantic_visual_v2.ply"),
          "\n     ", os.path.join(out, "semantic_views_v2.png"),
          "\n     ", os.path.join(out, "coverage_heatmaps_v2.png"),
          "\n     ", os.path.join(out, "semantic_tiles.png"))


if __name__ == "__main__":
    main()