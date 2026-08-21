"""解析虹桥 CesiumLab/ContextCapture b3dm (glTF1.0 + KHR_binary_glTF + CESIUM_RTC) → Trimesh
每个 b3dm: 单 mesh, 单纹理 JPEG, RTC center 提供局部->真实偏移
"""
import struct, json, io
import numpy as np

def parse_b3dm(path):
    with open(path, "rb") as f:
        raw = f.read()
    if raw[0:4] != b"b3dm":
        raise ValueError(f"not b3dm: {path}")
    ftj, ftb, btj, btb = struct.unpack_from("<IIII", raw, 12)
    off = 28 + ftj + ftb + btj + btb
    glb = raw[off:]
    # glTF 1.0 glb: header12 + json_len4 + json + binary
    jlen = struct.unpack_from("<I", glb, 12)[0]
    if glb[0:4] != b"glTF":
        raise ValueError(f"not glTF: {path[:40]}")
    j = json.loads(glb[20:20 + jlen])
    bin_start = 20 + jlen
    try:
        # 默认无对齐; 若 binary 开头不是 shader 声明则尝试 4 字节对齐
        data = glb[bin_start:]
    except Exception:
        pass
    data = glb[bin_start:]

    # RTC center
    center = np.array(j.get("extensions", {}).get("CESIUM_RTC", {}).get("center"),
                      dtype=np.float64)
    if center is None:
        center = np.zeros(3)

    buffers = j.get("buffers", {})
    if isinstance(buffers, dict):
        bname = list(buffers.keys())[0]
    else:
        bname = buffers[0]["id"]

    def read_bv(bvid):
        bv = j["bufferViews"][bvid]
        return data[bv["byteOffset"]: bv["byteOffset"] + bv["byteLength"]]

    def decode_accessor(aid):
        acc = j["accessors"][aid]
        bv = j["bufferViews"][acc["bufferView"]]
        body = data[bv["byteOffset"] + acc["byteOffset"]:
                    bv["byteOffset"] + bv["byteLength"] if (bv["byteOffset"] + bv["byteLength"]) <= len(data) else len(data)]
        comp_size = {5120: 1, 5121: 1, 5122: 2, 5123: 2, 5125: 4, 5126: 4}[acc["componentType"]]
        type_n = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}[acc["type"]]
        if acc["componentType"] == 5126:
            arr = np.frombuffer(body, '<f4', count=acc["count"] * type_n)
        elif acc["componentType"] == 5123:
            arr = np.frombuffer(body, '<u2', count=acc["count"] * type_n)
        elif acc["componentType"] == 5121:
            arr = np.frombuffer(body, '<u1', count=acc["count"] * type_n)
        elif acc["componentType"] == 5122:
            arr = np.frombuffer(body, '<i2', count=acc["count"] * type_n)
        else:
            arr = np.frombuffer(body, '<i4', count=acc["count"] * type_n)
        return arr.reshape(acc["count"], type_n)

    import trimesh
    meshes = []
    for mesh_key, mesh in j.get("meshes", {}).items():
        for prim in mesh["primitives"]:
            pos_aid = prim["attributes"]["POSITION"]
            pos = decode_accessor(pos_aid).astype(np.float64)   # (N,3) 局部
            pos = pos + center                                    # + RTC
            uv = None
            if "TEXCOORD_0" in prim["attributes"]:
                uv = decode_accessor(prim["attributes"]["TEXCOORD_0"]).astype(np.float32)
            ind_aid = prim.get("indices")
            ind = decode_accessor(ind_aid).astype(np.int64)  # (M,)
            faces = ind.reshape(-1, 3)
            # material / texture (img)
            tex_img = None
            if prim.get("material"):
                mat = j["materials"][prim["material"]]
                tex_id = mat.get("values", {}).get("tex")
                if tex_id:
                    # glTF1: values.tex -> textures 字典键 -> source (images 键)
                    tex_entry = j.get("textures", {})
                    if isinstance(tex_entry, dict) and tex_id in tex_entry:
                        src_id = tex_entry[tex_id].get("source")
                    img_info = j.get("images", {})
                    if isinstance(img_info, dict):
                        img_info = img_info.get(src_id)
                    if img_info and img_info.get("extensions") and "KHR_binary_glTF" in img_info["extensions"]:
                        bvi = img_info["extensions"]["KHR_binary_glTF"]["bufferView"]
                        mime = img_info["extensions"]["KHR_binary_glTF"]["mimeType"]
                        img_bytes = read_bv(bvi)
                    import PIL.Image as Image
                    img = Image.open(io.BytesIO(img_bytes))
                    if img.mode != "RGB":
                        img = img.convert("RGB")
                    tex_img = img
            m = trimesh.Trimesh(
                vertices=pos, faces=faces,
                process=False, validate=False)
            if uv is not None and tex_img is not None:
                import PIL.Image as PILIm
                uv_img = PILIm.fromarray(np.array(tex_img))
                mat_ = trimesh.visual.material.PBRMaterial(
                    baseColorTexture=tex_img)
                uv_vis = trimesh.visual.texture.TextureVisuals(uv=uv, material=mat_)
                m.visual = uv_vis
            # 记录 RTC center 和 b3dm 元信息
            m.metadata["rtc_center"] = tuple(center)
            m.metadata["b3dm"] = path
            meshes.append(m)
    return meshes


def parse_b3dm_list(paths):
    """遍历多个 b3dm, 返回合并 Trimesh (顶点坐标保留真实世界 RTC)"""
    parts = []
    for p in paths:
        try:
            parts.extend(parse_b3dm(p))
        except Exception as e:
            print(f"  [skip] {p}: {e}")
    return trimesh.util.concatenate(parts) if parts else None


if __name__ == "__main__":
    import os, sys, glob, trimesh
    src = sys.argv[1]
    paths = sorted(glob.glob(src))
    print(f"共 {len(paths)} 个 b3dm")
    for i, p in enumerate(paths[:3]):
        ms = parse_b3dm(p)
        for m in ms:
            bb = m.bounds
            print(f"[{p.split('/')[-1]}] verts={len(m.vertices)} faces={len(m.faces)} "
                  f"tex={m.visual.material.baseColorTexture.size if isinstance(m.visual, trimesh.visual.TextureVisuals) and m.visual.material.baseColorTexture is not None else 'N'} "
                  f"bounds=({bb[0][0]:.1f},{bb[0][1]:.1f},{bb[0][2]:.1f})-({bb[1][0]:.1f},{bb[1][1]:.1f},{bb[1][2]:.1f}) "
                  f"rtc={m.metadata['rtc_center']}")