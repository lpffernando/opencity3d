"""合并全部 GLB + ortho 俯视分块渲染（RGB-D），输出供 preprocess/反投影"""
import os
if os.name != "nt":
    os.environ["PYOPENGL_PLATFORM"] = "egl"
import argparse, glob, time
import numpy as np, trimesh, pyrender, imageio


def load_all_geometry(glb_dir, pattern="NoLod_*.glb"):
    glbs = sorted(glob.glob(os.path.join(glb_dir, pattern)))
    print(f"找到 {len(glbs)} 个 GLB")
    geos = []
    names = []
    for g_path in glbs:
        t0 = time.time()
        sc = trimesh.load(g_path, force='scene', process=False)
        for n, g_ in sc.geometry.items():
            if isinstance(g_, trimesh.Trimesh) and len(g_.faces) > 0:
                geos.append(g_)
                names.append(f"{os.path.basename(g_path)}::{n}")
        print(f"  {os.path.basename(g_path)}: {len(sc.geometry)} geoms, {time.time()-t0:.1f}s")
    return geos, names


def render_ortho_tile(scene_meshes, cx, cy, cam_z, half_x, half_y, width=512, height=512,
                      bg=(0.6, 0.68, 0.82), ambient=0.85):
    """渲染一张 ortho 俯视图: 相机在 (cx, cy, cam_z) 看 (cx, cy, 0)"""
    scene = pyrender.Scene(bg_color=np.array(bg), ambient_light=np.ones(3)*ambient)
    for m in scene_meshes:
        scene.add(m)
    cam = pyrender.OrthographicCamera(xmag=half_x, ymag=half_y, znear=1, zfar=6000)
    pose = np.eye(4)
    pose[:3, 3] = [cx, cy, cam_z]
    scene.add(cam, pose=pose)
    ren = pyrender.OffscreenRenderer(width, height)
    try:
        color, depth = ren.render(scene)
    finally:
        del ren
    return color, depth


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glb-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--n", type=int, default=4, help="分块网格大小 n×n")
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--cam-z", type=float, default=3000)
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    geos, names = load_all_geometry(args.glb_dir)
    nv = sum(len(g.vertices) for g in geos)
    nf = sum(len(g.faces) for g in geos)
    print(f"总顶点: {nv}, 总面: {nf}, 几何数: {len(geos)}")

    # 全局 bounds
    all_v = np.concatenate([g.vertices for g in geos])
    bb_min, bb_max = all_v.min(0), all_v.max(0)
    B = bb_max - bb_min
    print(f"全局 bounds: min={bb_min}  max={bb_max}  size={B}")
    np.save(os.path.join(args.output_dir, "bounds.npy"),
            np.array([bb_min, bb_max]))

    # pyrender mesh 只建一次（纹理保留）
    pm = [pyrender.Mesh.from_trimesh(g) for g in geos]
    print(f"pyrender meshes 建立完成")

    n = args.n
    cam_z = args.cam_z
    # 全图一张
    cx0, cy0 = (bb_min[0]+bb_max[0])/2, (bb_min[1]+bb_max[1])/2
    tasks = [("full", cx0, cy0, B[0]/2*1.02, B[1]/2*1.02)]
    # n×n 分块（15% overlap）
    for i in range(n):
        for j in range(n):
            cx = bb_min[0] + B[0]*(i+0.5)/n
            cy = bb_min[1] + B[1]*(j+0.5)/n
            hx = B[0]/(2*n)*1.15
            hy = B[1]/(2*n)*1.15
            tasks.append((f"tile_{i}_{j}", cx, cy, hx, hy))

    meta = []
    for tag, cx, cy, hx, hy in tasks:
        t0 = time.time()
        color, depth = render_ortho_tile(pm, cx, cy, cam_z, hx, hy,
                                         args.width, args.height)
        dt = time.time()-t0
        col_fn = os.path.join(args.output_dir, f"{tag}.jpg")
        dep_np = os.path.join(args.output_dir, f"{tag}_depth.npy")
        imageio.imwrite(col_fn, color)
        np.save(dep_np, depth)
        d_pct = float((depth > 0).mean())
        depth_min = float(depth[depth > 0].min()) if (depth > 0).any() else 0
        depth_max = float(depth.max())
        print(f"  {tag}: depth%={d_pct:.3f} depth[{depth_min:.0f}~{depth_max:.0f}] "
              f"meanRGB={tuple(round(c,1) for c in np.array(color).mean((0,1)))} {dt:.1f}s")
        meta.append(dict(tag=tag, cx=cx, cy=cy, half_x=hx, half_y=hy,
                         cam_z=cam_z, d_pct=d_pct))
    with open(os.path.join(args.output_dir, "tiles.json"), "w") as f:
        import json
        json.dump(meta, f, indent=1)
    print("完成，共", len(tasks), "张")


if __name__ == "__main__":
    main()