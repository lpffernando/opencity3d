"""俯视/低斜角多视角渲染器（适配 CesiumLab3 单面薄片 GLB）"""
import os
if os.name != "nt":
    os.environ["PYOPENGL_PLATFORM"] = "egl"
import argparse, numpy as np, trimesh, pyrender, imageio


def safe_render_top(mesh, view_pos, fov=55, width=512, height=512, bg=(0.55,0.65,0.85), ambient=0.8):
    bb = mesh.bounds
    size = bb[1] - bb[0]
    cen = bb.mean(0).copy()
    mesh = mesh.copy()
    mesh.apply_translation(-cen)
    scene = pyrender.Scene(bg_color=np.array(bg), ambient_light=np.ones(3)*ambient)
    scene.add(pyrender.Mesh.from_trimesh(mesh))
    cam = pyrender.PerspectiveCamera(yfov=np.deg2rad(fov), aspectRatio=width/height)
    pos = np.array(view_pos, dtype=float)
    fwd = -pos.copy(); fwd /= (np.linalg.norm(fwd) + 1e-8)
    # 选 up：俯视时 up 应为水平方向
    if abs(abs(fwd[2]) - 1.0) < 0.05:  # 几乎纯垂直看
        up = np.array([0,1,0])
    else:
        up = np.array([0,0,1])  # 默认 z-up（因为 mesh 高度方向是 z）
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


def render_glb_top_views(input_glb, output_prefix, n_views=12, width=512, height=512):
    sc = trimesh.load(input_glb, force='scene', process=False)
    big = max(sc.geometry.values(), key=lambda g: len(g.vertices))
    bb = big.bounds; size = bb[1]-bb[0]
    print(f"主 mesh: {len(big.vertices)} verts, size: {size}")

    # 低斜视角为主（30-70°），从 -z 看（即从下方往上看也行，但我们的 mesh 顶面朝 +z）
    # 实际上从正上方看 (-z 方向)看到顶面（屋顶纹理）
    # 但 low oblique 应该从 +z 高位置看 (相机在 +z 高，往 -z 看)
    # 这是 "鸟瞰"
    views = []
    for i, (yaw, pitch, d_scale) in enumerate([
        (0,   65, 1.3), (60,  65, 1.3), (120, 65, 1.3), (180, 65, 1.3),
        (240, 65, 1.3), (300, 65, 1.3),
        (30,  50, 1.2), (90,  50, 1.2), (150, 50, 1.2), (210, 50, 1.2),
        (270, 50, 1.2), (330, 50, 1.2),
    ][:n_views]):
        yaw_r = np.deg2rad(yaw); pitch_r = np.deg2rad(pitch)
        d = size.max() * d_scale
        # 相机在 +z 高位置，yaw 旋转绕 z 轴
        pos = np.array([
            d*np.cos(pitch_r)*np.cos(yaw_r),
            d*np.cos(pitch_r)*np.sin(yaw_r),
            d*np.sin(pitch_r) + 5,  # +5 防止相机在 mesh 中
        ])
        views.append((f"view_{i:02d}_y{yaw:03d}_p{pitch:02d}", pos))

    saved = []
    for tag, pos in views:
        try:
            color, depth = safe_render_top(big, pos, fov=55, width=width, height=height)
            fn = f"{output_prefix}_{tag}.jpg"
            imageio.imwrite(fn, color)
            d_pct = float((depth > 0).mean())
            print(f"  {tag}: depth%={d_pct:.3f} meanRGB={tuple(round(c,1) for c in np.array(color).mean((0,1)))}")
            saved.append(fn)
        except Exception as e:
            print(f"  {tag}: FAIL {e}")
    return saved


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output-prefix", required=True)
    ap.add_argument("--n-views", type=int, default=12)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--height", type=int, default=512)
    args = ap.parse_args()
    render_glb_top_views(args.input, args.output_prefix, args.n_views, args.width, args.height)