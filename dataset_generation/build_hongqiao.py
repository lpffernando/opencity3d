"""虹桥 b3dm → 局部 ENU 平面 mesh (用于 ortho 渲染)
选中心 4x4 连续格网的顶层 b3dm, ECEF -> ENU(以区域中心为原点) -> 合并 Trimesh
"""
import sys, os, glob, json
import numpy as np, trimesh
sys.path.insert(0, "dataset_generation")
from parse_b3dm import parse_b3dm

DATA = "/media/fernando/Elements SE/倾斜摄影数据/上海虹桥商务区/hongqiao-3DTiles/Scene/Data"


def ecef_to_enu_matrix(orig):
    ox, oy, oz = orig; a = 6378137.0; f = 1 / 298.257223563; e2 = f * (2 - f)
    lon = np.arctan2(oy, ox); sl = np.sin(lon); cl = np.cos(lon)
    lat0 = np.arctan2(oz, np.hypot(ox, oy))
    for _ in range(5):
        nz = a / np.sqrt(1 - e2 * np.sin(lat0) ** 2)
        lat0 = np.arctan2(oz + e2 * nz * np.sin(lat0), np.hypot(ox, oy))
    sa = np.sin(lat0); ca = np.cos(lat0)
    return np.array([[-sl, cl, 0], [-sa * cl, -sa * sl, ca], [ca * cl, ca * sl, sa]])


def enu(pts_ecef, R, origin):
    return (pts_ecef - origin) @ R.T


def main():
    # 区域2: 围绕商业核心 Tile_p006_p021, 较亮区域, 4列(p005-008) x 4行(p020-023)
    tiles = [
        "Tile_p005_p020", "Tile_p005_p021", "Tile_p005_p022", "Tile_p005_p023",
        "Tile_p006_p020", "Tile_p006_p021", "Tile_p006_p022", "Tile_p006_p023",
        "Tile_p007_p020", "Tile_p007_p021", "Tile_p007_p022", "Tile_p007_p023",
        "Tile_p008_p020", "Tile_p008_p021", "Tile_p008_p022", "Tile_p008_p023",
    ]
    paths = [os.path.join(DATA, t, t + ".b3dm") for t in tiles]
    # 用第一个 tile 的 RTC center 作为 ENU 原点
    _ms = parse_b3dm(paths[0])
    origin = np.array(_ms[0].metadata["rtc_center"], float)
    R = ecef_to_enu_matrix(origin)

    parts = []
    for p in paths:
        if not os.path.exists(p):
            print(f"  [缺失] {p}"); continue
        try:
            ms = parse_b3dm(p)
            for m in ms:
                verts = np.asarray(m.vertices, dtype=np.float64)
                e = enu(verts, R, origin)
                nm = trimesh.Trimesh(vertices=e, faces=m.faces,
                                      visual=m.visual, process=False, validate=False)
                parts.append(nm)
                print(f"  {os.path.basename(p):30s} verts={len(e)} tex={'Y' if m.visual.material.baseColorTexture is not None else 'N'}")
        except Exception as ex:
            print(f"  [失败] {p}: {ex}")

    mesh = trimesh.util.concatenate(parts)
    bb = mesh.bounds
    print(f"\n总顶点: {len(mesh.vertices)}, 面: {len(mesh.faces)}")
    print(f"ENU 范围 x:{bb[0][0]:.1f}..{bb[1][0]:.1f} y:{bb[0][1]:.1f}..{bb[1][1]:.1f} z:{bb[0][2]:.1f}..{bb[1][2]:.1f}")

    os.makedirs("data/hongqiao-e2e", exist_ok=True)
    out_obj = "data/hongqiao-e2e/area.obj"
    out_glb = "data/hongqiao-e2e/area.glb"
    mesh.export(out_obj)
    mesh.export(out_glb)
    print("保存:", out_obj, out_glb)
    # 保存 ENU 原点/旋转供后续 ortho 反投影
    np.save("data/hongqiao-e2e/enu_origin.npy", origin)
    np.save("data/hongqiao-e2e/enu_R.npy", R)


if __name__ == "__main__":
    main()