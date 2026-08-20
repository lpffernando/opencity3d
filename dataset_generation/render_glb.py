"""GLB 多视角稳健渲染器（每次新建 renderer 避免 EGL context 冲突）"""
import os
if os.name != "nt":
    os.environ["PYOPENGL_PLATFORM"] = "egl"
import argparse, sys, numpy as np, trimesh, pyrender, imageio


def safe_render(mesh, view_pos, fov=55, width=512, height=512, bg=(0.55,0.65,0.85), ambient=0.8):
    """单次渲染: 新建 scene/camera/renderer, 渲染后立即销毁"""
    bb = mesh.bounds
    size = bb[1] - bb[0]
    cen = bb.mean(0).copy()
    mesh = mesh.copy()
    mesh.apply_translation(-cen)
    scene = pyrender.Scene(bg_color=np.array(bg), ambient_light=np.ones(3)*ambient)
    scene.add(pyrender.Mesh.from_trimesh(mesh))
    cam = pyrender.PerspectiveCamera(yfov=np.deg2rad(fov), aspectRatio=width/height)
    pos = np.array(view_pos, dtype=float)
    fwd = -pos.copy(); fwd /= np.linalg.norm(fwd)
    if abs(abs(np.dot(fwd, [0,1,0])) - 1.0) < 0.05:
        up = np.array([0,0,1.0]) if abs(fwd[1])>0.95 else np.array([1,0,0])
    else:
        up = np.array([0,1,0])
    x_ax = np.cross(up, fwd); x_ax /= (np.linalg.norm(x_ax)+1e-8)
    y_ax = np.cross(fwd, x_ax)
    pose = np.eye(4); pose[:3,0]=x_ax; pose[:3,1]=y_ax; pose[:3,2]=fwd; pose[:3,3]=pos
    scene.add(cam, pose=pose)
    ren = pyrender.OffscreenRenderer(width, height)
    try:
        color, depth = ren.render(scene)
    finally:
        del ren
    return color, depth


def render_glb(input_glb, output_prefix, n_views=12, width=512, height=512):
    sc = trimesh.load(input_glb, force='scene', process=False)
    big = max(sc.geometry.values(), key=lambda g: len(g.vertices))
    bb = big.bounds; size = bb[1]-bb[0]
    print(f"主 mesh: {len(big.vertices)} verts, size: {size}")

    views = []
    # 12 个相机: 球面均匀采样
    import itertools
    for i, (yaw, pitch) in enumerate([
        (0, 25), (90, 25), (180, 25), (270, 25),
        (45, 55), (135, 55), (225, 55), (315, 55),
        (0, 75), (90, 75), (180, 75), (270, 75),
    ][:n_views]):
        yaw_r = np.deg2rad(yaw); pitch_r = np.deg2rad(pitch)
        d = size.max()*1.5
        pos = np.array([
            d*np.cos(pitch_r)*np.cos(yaw_r),
            d*np.cos(pitch_r)*np.sin(yaw_r),
            d*np.sin(pitch_r)
        ])
        views.append((f"view_{i:02d}_y{yaw:03d}_p{pitch:02d}", pos))

    saved = []
    for tag, pos in views:
        try:
            color, depth = safe_render(big, pos, fov=55, width=width, height=height)
            fn = f"{output_prefix}_{tag}.jpg"
            imageio.imwrite(fn, color)
            d_pct = float((depth>0).mean())
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
    render_glb(args.input, args.output_prefix, args.n_views, args.width, args.height)