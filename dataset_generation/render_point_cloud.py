"""点云 → 多视角 RGB-D 渲染（OpenCity3D 的点云输入路线）。

对带 RGB 颜色的城市点云采样相机位姿，用 pyrender 渲染 color/depth/pose/intrinsic，
输出与 `generate_dataset.py` 完全一致的目录结构，供 preprocess.py / convert_to_point_cloud.py 使用。

用法:
    python render_point_cloud.py --input-points Area11.txt --output-dir scene-output \
        --n-samples 200 --point-size 3
"""
import os
import argparse
import numpy as np
import pyrender
import imageio
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

if os.name != "nt":
    os.environ["PYOPENGL_PLATFORM"] = "egl"


def load_points(path, max_points=int(4e6)):
    """读取点云 txt (x y z r g b [labels...]) 或 ply/npy。返回 (points, colors)。"""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".txt":
        arr = np.loadtxt(path)
        pts = arr[:, :3].astype(np.float32)
        col = (arr[:, 3:6].astype(np.float32) / 255.0)
    elif ext in (".ply", ".pcd", ".xyz"):
        import open3d as o3d
        pcd = o3d.io.read_point_cloud(path)
        pts = np.asarray(pcd.points).astype(np.float32)
        col = np.asarray(pcd.colors).astype(np.float32)
    elif ext == ".npy":
        arr = np.load(path)
        pts = arr[:, :3].astype(np.float32)
        col = (arr[:, 3:6].astype(np.float32) / 255.0) if arr.shape[1] >= 6 else None
    else:
        raise ValueError(f"unsupported: {path}")
    if len(pts) > max_points:
        idx = np.random.choice(len(pts), max_points, replace=False)
        pts, col = pts[idx], col[idx]
    return pts, col


def render(pts, colors, output_path, width=384, height=384, n_samples=100, point_size=3.0):
    fx = fy = 1200 // 2
    cx, cy = 600 // 2, 600 // 2
    z_far, z_near = 2000, 10
    intrinsic = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    pm = pyrender.Mesh.from_points(pts, colors=colors)
    scene = pyrender.Scene(ambient_light=np.ones(3) * 0.9, bg_color=[1, 1, 1])
    scene.add(pm)
    renderer = pyrender.OffscreenRenderer(width, height, point_size=point_size)

    for d in ("depth", "color", "pose", "intrinsic"):
        os.makedirs(f"{output_path}/{d}", exist_ok=True)

    lo, hi = pts.min(0), pts.max(0)
    # 在水平范围网格采样相机位姿
    border = 10
    n_grid = max(3, int(np.sqrt(n_samples)))
    xs = np.linspace(lo[0] + border, hi[0] - border, n_grid)
    zs = np.linspace(lo[2] + border, hi[2] - border, n_grid)
    cam = pyrender.PerspectiveCamera(yfov=np.deg2rad(60), aspectRatio=width / height, znear=z_near, zfar=z_far)
    projection = cam.get_projection_matrix(width, height)

    counter = 0
    for x in tqdm(xs, desc="rendering"):
        for z in zs:
            for attempt in range(5):
                height_cam = np.random.uniform(lo[1] + 25, lo[1] + 120)
                yaw = np.random.uniform(0, 360)
                pitch = 90 if np.random.rand() < 0.3 else np.random.uniform(40, 70)
                rot = R.from_euler("yxz", [yaw, pitch, 0], degrees=True).as_matrix()
                pose = np.eye(4)
                pose[:3, :3] = rot
                pose[:3, 3] = [x, height_cam, z]
                node = scene.add(cam, pose=pose)
                scene.main_camera_node = node
                color, depth = renderer.render(scene)
                valid = (depth > z_near).sum()
                if valid < 0.5 * depth.size:
                    continue
                np.save(f"{output_path}/depth/{counter}.npy", depth.astype(np.float16))
                imageio.imwrite(f"{output_path}/color/{counter}.jpg", color)
                np.savetxt(f"{output_path}/pose/{counter}.txt", pose, fmt="%f")
                counter += 1
                break
    np.savetxt(f"{output_path}/intrinsic/intrinsic_color.txt", intrinsic, fmt="%f")
    np.savetxt(f"{output_path}/intrinsic/projection_matrix.txt", projection, fmt="%f")
    print(f"rendered {counter} valid views -> {output_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-points", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--n-samples", type=int, default=200)
    ap.add_argument("--point-size", type=float, default=3.0)
    ap.add_argument("--width", type=int, default=384)
    ap.add_argument("--height", type=int, default=384)
    ap.add_argument("--max-points", type=int, default=4_000_000)
    args = ap.parse_args()
    pts, col = load_points(args.input_points, args.max_points)
    print(f"loaded {len(pts)} points, bounds {pts.min(0)} -> {pts.max(0)}")
    render(pts, col, args.output_dir, args.width, args.height, args.n_samples, args.point_size)
