"""正交投影俯视渲染器（专为 CesiumLab3 单面薄片 GLB，按区域分块扫描）"""
import os
if os.name != "nt":
    os.environ["PYOPENGL_PLATFORM"] = "egl"
import argparse, numpy as np, trimesh, pyrender, imageio


def render_ortho_view(mesh, look_at, fov_y_deg=60, width=512, height=512, bg=(0.55,0.65,0.85)):
    """从 mesh 上方俯视（z 是高度）用 ortho 相机"""
    mesh = mesh.copy()
    cen = mesh.bounds.mean(0)
    mesh.apply_translation(-cen)
    # mesh 现以 (0,0,0) 为中心
    scene = pyrender.Scene(bg_color=np.array(bg), ambient_light=np.ones(3)*0.8)
    scene.add(pyrender.Mesh.from_trimesh(mesh))
    # 相机位置: 上方 z 高处, 看向原点
    size = mesh.bounds[1] - mesh.bounds[0]
    pos = np.array([look_at[0], look_at[1], 1500])  # +z 1500m 高
    target = np.array([look_at[0], look_at[1], 0])
    fwd = target - pos; fwd /= np.linalg.norm(fwd)
    up = np.array([0,1,0])
    x_ax = np.cross(up, fwd); x_ax /= np.linalg.norm(x_ax)
    y_ax = np.cross(fwd, x_ax)
    pose = np.eye(4); pose[:3,0]=x_ax; pose[:3,1]=y_ax; pose[:3,2]=fwd; pose[:3,3]=pos
    # 用 ortho 相机: 取 mesh 水平维度（x, y）为半边
    # 取场景较大水平维的一半
    half = max(size[0], size[1]) * 0.6
    cam = pyrender.OrthographicCamera(xmag=half, ymag=half, znear=1, zfar=5000)
    scene.add(cam, pose=pose)
    ren = pyrender.OffscreenRenderer(width, height)
    try:
        color, depth = ren.render(scene)
    finally:
        del ren
    return color, depth


def render_grid(input_glb, output_prefix, n_tiles=3, width=512, height=512):
    """网格扫描渲染: 把 mesh 分成 n×n 块, 每块一张 ortho 俯视图"""
    sc = trimesh.load(input_glb, force='scene', process=False)
    big = max(sc.geometry.values(), key=lambda g: len(g.vertices))
    bb = big.bounds; size = bb[1]-bb[0]
    print(f"主 mesh: {len(big.vertices)} verts, size: {size}")
    cen = bb.mean(0)
    # 计算各 tile 的 look_at (mesh 的局部坐标)
    xy = size[:2]  # x, y 维 (水平)
    # tile 在 [-xy/2, xy/2] 之间均匀分布
    coords = np.linspace(-float(xy[0])/2*0.7, float(xy[0])/2*0.7, n_tiles).tolist()
    coords2 = np.linspace(-float(xy[1])/2*0.7, float(xy[1])/2*0.7, n_tiles).tolist()
    saved = []
    idx = 0
    for i, x in enumerate(coords):
        for j, y in enumerate(coords2):
            local_look = [float(x), float(y), 0.0]  # 局部坐标
            try:
                color, depth = render_ortho_view(big, local_look, width=width, height=height)
                fn = f"{output_prefix}_tile{i*n_tiles+j:02d}.jpg"
                imageio.imwrite(fn, color)
                d_pct = float((depth > 0).mean())
                print(f"  tile({i},{j}) look=({x:.0f},{y:.0f}): depth%={d_pct:.3f} meanRGB={tuple(round(c,1) for c in np.array(color).mean((0,1)))}")
                saved.append(fn)
                idx += 1
            except Exception as e:
                print(f"  tile({i},{j}): FAIL {e}")
    return saved


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output-prefix", required=True)
    ap.add_argument("--n-tiles", type=int, default=3)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--height", type=int, default=512)
    args = ap.parse_args()
    render_grid(args.input, args.output_prefix, args.n_tiles, args.width, args.height)