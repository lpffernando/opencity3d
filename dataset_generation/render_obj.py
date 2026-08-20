"""OBJ/GLB 多视角渲染器（高产出 + 屋顶/墙面/地面几何上色）"""
import os, argparse
import numpy as np, trimesh, pyrender, imageio
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

if os.name != "nt":
    os.environ["PYOPENGL_PLATFORM"] = "egl"


def colorize_geometry(mesh, roof_color=(70,70,80), wall_color=(220,225,230),
                       ground_color=(110,130,90), default=(180,180,180)):
    """基于法向量 + 高度着色"""
    v = mesh.vertices; n = mesh.vertex_normals; z = v[:,2]
    nz = n[:,2]
    roof = nz > 0.6
    ground = (z < np.percentile(z, 5)) | (nz < -0.6)
    wall = ~(roof | ground)
    col = np.tile(np.array(default, dtype=np.uint8), (len(v),1))
    col[roof] = roof_color
    col[wall] = wall_color
    col[ground] = ground_color
    mesh.visual.vertex_colors = col
    return mesh


def render_mesh(input_path, output_dir, width=384, height=384, n_views=80,
                 point_size=3.0, colorize=True):
    raw = trimesh.load(input_path, force='scene')
    if colorize:
        for name, g in raw.geometry.items():
            colorize_geometry(g)
    scene = pyrender.Scene.from_trimesh_scene(raw, ambient_light=np.ones(3)*0.7,
                                              bg_color=np.ones(3))
    for d in ("depth","color","pose","intrinsic"):
        os.makedirs(f"{output_dir}/{d}", exist_ok=True)

    bb = scene.bounds
    cen = bb.mean(0); size = bb[1]-bb[0]
    print(f"场景 bounds: {bb}, 尺寸: {size}")

    cam = pyrender.PerspectiveCamera(yfov=np.deg2rad(60), aspectRatio=width/height,
                                      znear=10, zfar=size.max()*8)
    r = pyrender.OffscreenRenderer(width, height, point_size=point_size)
    R_world = np.diag([1,-1,1]) if abs(size[0])>abs(size[2]) else np.eye(3)
    n_grid = max(3, int(np.sqrt(n_views)))
    xs = np.linspace(bb[0,0]+size[0]*0.1, bb[1,0]-size[0]*0.1, n_grid)
    zs = np.linspace(bb[0,2]+size[2]*0.1, bb[1,2]-size[2]*0.1, n_grid)
    y_min, y_max = cen[1]+5, cen[1]+max(size[1], size.max()*0.2)+10
    cnt = 0; attempts = 0
    for x in tqdm(xs, desc="rendering"):
        for z in zs:
            for h in np.linspace(y_min, y_max, 3):
                for _ in range(2):
                    attempts += 1
                    yaw = np.random.uniform(0, 360)
                    pitch = 60 if np.random.rand()<0.4 else np.random.uniform(35, 75)
                    rot = R.from_euler('yxz', [yaw, pitch, 0], degrees=True).as_matrix()
                    pose = np.eye(4); pose[:3,:3] = rot @ R_world.T; pose[:3,3] = [x,h,z]
                    try:
                        n = scene.add(cam, pose=pose)
                        scene.main_camera_node = n
                        c, d = r.render(scene)
                        scene.remove_node(n)
                    except Exception:
                        continue
                    if (d > 1).mean() > 0.15:
                        np.save(f"{output_dir}/depth/{cnt}.npy", d.astype(np.float16))
                        imageio.imwrite(f"{output_dir}/color/{cnt}.jpg", c)
                        np.savetxt(f"{output_dir}/pose/{cnt}.txt", pose, fmt="%f")
                        cnt += 1
                        break
                if cnt >= n_views: break
            if cnt >= n_views: break
        if cnt >= n_views: break
    intrinsic = np.array([[600,0,192],[0,600,192],[0,0,1]], dtype=float)
    np.savetxt(f"{output_dir}/intrinsic/intrinsic_color.txt", intrinsic)
    print(f"rendered {cnt}/{attempts} -> {output_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-mesh", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--width", type=int, default=384)
    ap.add_argument("--height", type=int, default=384)
    ap.add_argument("--n-views", type=int, default=80)
    ap.add_argument("--no-colorize", action="store_true")
    args = ap.parse_args()
    render_mesh(args.input_mesh, args.output_dir, args.width, args.height,
                 args.n_views, colorize=not args.no_colorize)