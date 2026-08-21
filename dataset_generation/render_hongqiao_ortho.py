"""虹桥: 合并 mesh (area.glb, ENU平面) → ortho 俯视分块渲染 RGB-D"""
import os
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import numpy as np, trimesh, pyrender, imageio, json, time


def render_ortho_tile(pm_list, cx, cy, cam_z, half_x, half_y, width=512, height=512,
                      bg=(0.6, 0.68, 0.82), ambient=0.9, znear=1, zfar=5000):
    scene = pyrender.Scene(bg_color=np.array(bg), ambient_light=np.ones(3) * ambient)
    for m in pm_list:
        scene.add(m)
    cam = pyrender.OrthographicCamera(xmag=half_x, ymag=half_y, znear=znear, zfar=zfar)
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
    os.makedirs("data/hongqiao-e2e/render", exist_ok=True)
    scene = trimesh.load("data/hongqiao-e2e/area.glb", force='scene', process=False)
    geos = [g for g in scene.geometry.values() if isinstance(g, trimesh.Trimesh) and len(g.faces) > 0]
    print(f"几何数: {len(geos)}")
    all_v = np.concatenate([g.vertices for g in geos])
    bb_min, bb_max = all_v.min(0), all_v.max(0)
    B = bb_max - bb_min
    print(f"bounds: min={bb_min} max={bb_max} size={B}")
    cx0, cy0 = (bb_min[0]+bb_max[0])/2, (bb_min[1]+bb_max[1])/2

    pm = [pyrender.Mesh.from_trimesh(g) for g in geos]
    print("pyrender mesh 就绪")

    cam_z = 2000.0
    n = 4
    tasks = [("full", cx0, cy0, B[0]/2*1.02, B[1]/2*1.02)]
    for i in range(n):
        for j in range(n):
            cx = bb_min[0] + B[0]*(i+0.5)/n
            cy = bb_min[1] + B[1]*(j+0.5)/n
            hx = B[0]/(2*n)*1.18
            hy = B[1]/(2*n)*1.18
            tasks.append((f"tile_{i}_{j}", cx, cy, hx, hy))

    meta = []
    for tag, cx, cy, hx, hy in tasks:
        t0 = time.time()
        color, depth = render_ortho_tile(pm, cx, cy, cam_z, hx, hy)
        # 亮度/对比度增强 (虹桥 ContextCapture 纹理偏暗): stretch + gamma + 提亮
        cimg = color.astype(np.float32)
        lo, hi = np.percentile(cimg, [1, 99])
        if hi - lo > 1:
            cimg = np.clip((cimg - lo) / (hi - lo), 0, 1)
        cimg = cimg ** 0.7          # gamma 提亮
        cimg = np.clip(cimg * 1.35, 0, 1)   # 增益
        cimg = (cimg * 255).astype(np.uint8)
        dt = time.time() - t0
        imageio.imwrite(f"data/hongqiao-e2e/render/{tag}.jpg", cimg)
        np.save(f"data/hongqiao-e2e/render/{tag}_depth.npy", depth)
        d_pct = float((depth > 0).mean())
        dmin = float(depth[depth > 0].min()) if (depth > 0).any() else 0
        print(f"  {tag}: depth%={d_pct:.3f} depth[{dmin:.0f}~{depth.max():.0f}] "
              f"RGB={tuple(round(c) for c in np.array(color).mean((0,1)))} {time.time()-t0:.1f}s")
        meta.append(dict(tag=tag, cx=cx, cy=cy, half_x=hx, half_y=hy, cam_z=cam_z, d_pct=d_pct))
    json.dump(meta, open("data/hongqiao-e2e/render/tiles.json", "w"), indent=1)
    print("完成", len(tasks), "张")


if __name__ == "__main__":
    main()