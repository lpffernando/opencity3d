"""俯视多视角渲染器（专为 CesiumLab3 单面薄片 GLB：正上方俯视 z=+高度）"""
import os
if os.name != "nt":
    os.environ["PYOPENGL_PLATFORM"] = "egl"
import argparse, numpy as np, trimesh, pyrender, imageio


def safe_render_top(mesh, view_pos, look_at, up, fov=55, width=512, height=512, bg=(0.55,0.65,0.85), ambient=0.8):
    mesh = mesh.copy()
    bb = mesh.bounds
    cen = bb.mean(0)
    mesh.apply_translation(-cen)
    scene = pyrender.Scene(bg_color=np.array(bg), ambient_light=np.ones(3)*ambient)
    scene.add(pyrender.Mesh.from_trimesh(mesh))
    cam = pyrender.PerspectiveCamera(yfov=np.deg2rad(fov), aspectRatio=width/height)
    pos = np.array(view_pos, dtype=float)
    target = np.array(look_at, dtype=float)
    fwd = target - pos; fwd /= (np.linalg.norm(fwd) + 1e-8)
    up = np.array(up, dtype=float); up /= np.linalg.norm(up)
    x_ax = np.cross(up, fwd); x_ax /= (np.linalg.norm(x_ax) + 1e-8)
    y_ax = np.cross(fwd, x_ax)
    pose = np.eye(4); pose[:3,0]=x_ax; pose[:3,1]=y_ax; pose[:3,2]=fwd; pose[:3,3]=pos
    scene.add(cam, pose=pose)
    ren = pyrender.OffscreenRenderer(width, height)
    try:
        color, depth = ren.render(scene)
    finally:
        del ren
    return color, depth


def render_topdown(input_glb, output_prefix, n_views=12, width=512, height=512, height_offset=2000):
    """从上方俯视 z=+高位置看, 各种 yaw"""
    sc = trimesh.load(input_glb, force='scene', process=False)
    big = max(sc.geometry.values(), key=lambda g: len(g.vertices))
    bb = big.bounds; size = bb[1]-bb[0]
    print(f"主 mesh: {len(big.vertices)} verts, size: {size}")

    # size.max() 是最宽方向. camera 距离 = size.max() * 1.4 (鸟瞰)
    # 因为 mesh z 维度是高度 (薄), 水平 x/y 占主导
    saved = []
    # 6 个 yaw x 2 个 pitch (倾斜度) = 12 视角
    for i, (yaw, pitch) in enumerate([
        (0, 30), (60, 30), (120, 30), (180, 30), (240, 30), (300, 30),
        (0, 60), (60, 60), (120, 60), (180, 60), (240, 60), (300, 60),
    ][:n_views]):
        yaw_r = np.deg2rad(yaw); pitch_r = np.deg2rad(pitch)
        # camera 水平距离: 大, 让 mesh 充满视野
        d_horiz = size.max() * 1.4 / np.cos(pitch_r)
        pos = np.array([
            d_horiz*np.cos(yaw_r),
            d_horiz*np.sin(yaw_r),
            size.max()*1.4*np.tan(pitch_r) + 200,  # z = 高度
        ])
        look_at = np.array([0,0,0])
        up = np.array([0,0,1])  # z up
        try:
            color, depth = safe_render_top(big, pos, look_at, up, fov=55, width=width, height=height)
            fn = f"{output_prefix}_v{i:02d}_y{yaw:03d}_p{pitch:02d}.jpg"
            imageio.imwrite(fn, color)
            d_pct = float((depth > 0).mean())
            print(f"  v{i:02d} yaw={yaw:3d} pitch={pitch:3d}: depth%={d_pct:.3f} meanRGB={tuple(round(c,1) for c in np.array(color).mean((0,1)))}")
            saved.append(fn)
        except Exception as e:
            print(f"  v{i:02d}: FAIL {e}")
    return saved


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output-prefix", required=True)
    ap.add_argument("--n-views", type=int, default=12)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--height", type=int, default=512)
    args = ap.parse_args()
    render_topdown(args.input, args.output_prefix, args.n_views, args.width, args.height)