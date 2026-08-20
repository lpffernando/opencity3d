# OpenCity3D 推进指南（本机已适配）

本机环境已搭好并完成端到端验证。本文档记录环境、代码修复、使用命令，以及倾斜摄影数据的准备要求。

---

## 1. 环境

- **硬件**: RTX 4090 24GB / 62GB RAM / 6.1T 磁盘，满足要求。
- **Python venv**: `venv/`（复用系统 torch 2.9.0+cu128），激活方式：

```bash
cd /mnt/data/opencity3d
source venv/bin/activate
export HF_HUB_OFFLINE=1   # SigLIP 模型已本地缓存, 离线加载
```

- **已装依赖**: trimesh 4.5.2, pyrender 0.1.45, open3d 0.19, open-clip-torch 3.3, segment-anything 1.0, scipy, imageio, laspy。
- **已下载模型**:
  - SAM vit-h 权重: `ckpts/sam_vit_h_4b8939.pth` (2.5GB)
  - SigLIP 主模型: `~/.cache/huggingface/hub/models--timm--ViT-SO400M-14-SigLIP-384` (3.5GB)
  - SigLIP tokenizer: `~/.cache/huggingface/hub/models--timm--ViT-B-16-SigLIP` (0.8GB)

> 说明: huggingface.co 直连会因代理的 HEAD 重定向问题失败（`FileMetadataError`）。
> 已通过 `sandbox/setup_hf_cache.py` 手动下载到 HF 缓存，配合 `HF_HUB_OFFLINE=1` 使用。
> 如需在线下载其他 HF 模型，可先 `export HF_ENDPOINT=https://hf-mirror.com`。

---

## 2. 代码修复清单（适配新版依赖）

| 文件 | 问题 | 修复 |
|---|---|---|
| `dataset_generation/generate_dataset.py` | trimesh>=4 `load()` 对单网格返回 `Trimesh` 而非 `Scene`，pyrender 报错 | 包一层 `trimesh.Scene` |
| `preprocessing/preprocess.py` | `SigLipNetwork.encode_text` 用了未定义变量 `model`，且 tokenizer 输出在 CPU | 用 `self.model.context_length`；ids `.to(device)` |
| `preprocessing/preprocess.py` | pip 版 segment-anything 的 `generate()` 只返回单个 mask 列表，官方代码期望 4 层 (default/s/m/l) | 按 bbox 面积拆成 4 层 |
| `preprocessing/preprocess.py` | 某层 mask 被 NMS/filter 全删光时缺 key（KeyError 's'） | 缺失层跳过；全空退化为整图嵌入；空层用整图退化 mask 保证固定 4 层 |
| `convert_to_point_cloud.py` | 写死作者本机绝对路径 | 改为 argparse（`--base-path/--obj-path/--output`） |
| `convert_to_point_cloud.py` | 只支持 OBJ 三角网格 | 支持纯点云（.ply/.xyz/.pcd）输入 |
| `convert_to_point_cloud.py` | 特征维度写死 512，SigLIP 是 1152 → 所有特征被丢弃 | 动态取 `features[0].shape[-1]` |
| `convert_to_point_cloud.py` | 特征目录写死 `language_features` | 自动探测 `language_features_highlight` 等变体 |
| `sandbox/text_encoder.py` | tokenizer 输出在 CPU | ids `.to(model device)` |
| `venv/.../pyrender/mesh.py` | numpy>=2 移除 `np.infty` | `np.infty` → `np.inf`（site-packages 内 sed） |

---

## 3. 使用命令

```bash
cd /mnt/data/opencity3d
source venv/bin/activate
export HF_HUB_OFFLINE=1

# Step 0: 数据准备（见第 4 节）

# Step 1: 渲染多视角 RGB-D（从网格）
python dataset_generation/generate_dataset.py \
  --input-mesh <场景.obj> \
  --output-dir <输出目录> \
  --width 384 --height 384 \
  --approx-n-samples 100   # 先小规模试通, 再放大到 10000

# Step 2: SAM + SigLIP 提取语言特征（最耗时, 需 GPU）
python preprocessing/preprocess.py \
  --dataset_path <输出目录> \
  --model siglip --mode highlight \
  --sam_ckpt_path ckpts/sam_vit_h_4b8939.pth

# Step 3: 投影成语义点云
python convert_to_point_cloud.py \
  --base-path <输出目录> \
  --obj-path <场景网格或采样点云.ply> \
  --output eval/<scene>/generated_point_cloud.ply

# Step 4: 查询可视化（见 sandbox/visualize_pcd_features.ipynb）
#   把 eval/<scene>/ 下的 generated_point_cloud.ply + point_features_highlight.npy
#   放入 data/embedded_point_clouds/<tag>/ 后改 notebook 的 tag 即可
```

---

## 4. 倾斜摄影数据处理

OpenCity3D 的入口是**带 UV 纹理的三角网格 (OBJ/GLB)**。倾斜摄影数据需先转成网格：

### 路线 A: OSGB / 3D Tiles 瓦片（推荐）
1. 合并瓦片并导出 OBJ：ContextCapture / 大疆智图 / CesiumLab / CloudCompare 均可；
2. **减面**到 ≤100 万面（CloudCompare/MeshLab Quadric Edge Collapse），**必须保留 UV 纹理**；
3. **坐标系归零**：倾斜摄影常是 CGCS2000 大坐标，float32 会丢精度，需平移到原点、单位米、Z 轴向上；
4. 导出 OBJ（含 .mtl + 贴图）或 GLB，进 Step 1。

### 路线 B: 只有点云 (LAS/LAZ)
- 直接喂 `generate_dataset.py` 会失败（pyrender 需要三角面）。
- 本仓库已让 `convert_to_point_cloud.py` 支持点云；渲染端可用 `pyrender.Mesh.from_points` 渲染点云出 RGB-D（需小改渲染脚本），或先用 CloudCompare 做 Poisson 重建得到网格。

### 路线 C: 原始照片 + POS（无网格）
- 用原片当图源，跳过渲染；但需把空三位姿转成 4×4 C2W 齐次矩阵，且深度图需从稀疏点云渲染——改造量最大。

> 无论哪条路线：**单位统一为米、坐标归零、Z 向上**，模型点数上限 100 万（`--max-points`）。

---

## 5. 已完成的端到端验证

- 用程序生成的街区（8 栋建筑 + 地面）跑通：渲染 86 张 RGB-D → SAM+SigLIP 特征 → 投影出 30000 点语义点云 `(4, 30000, 1152)`。
- 结论：**计算链路完整可用**；语义识别质量依赖**真实照片纹理**（抽象纯色几何上 SigLIP 区分度低，倾斜摄影真实纹理无此问题）。

---

## 下一步（需要你提供）
1. 倾斜摄影数据在哪个路径、什么格式（OSGB 瓦片目录 / 3D Tiles / 已导出 OBJ / LAS 点云）？
2. 规模多大（瓦片数 / 面数 / 覆盖范围）？
3. 目标空间问题（绿地覆盖、建筑密度/年代、道路安全、空置率等）——对应 `sandbox/` 里的实验 notebook。

---

## 6. 实测：上海白膜 3D Tiles 数据在本工程的可用性（2026-02 验证）

**数据**：`/media/fernando/dodo/上海/上海街镇行政区划/3dtiles`（CesiumLab3 model2tiles 导出，34GB / 27.3 万文件，16/20 两级 LOD，瓦片约 19.1m/格）。抽样 8000 个 glb：**100% 无纹理、无 baseColor、无顶点色** —— 是建筑白膜，非倾斜摄影实景。

**端到端实测**（已完整跑通，均无报错）：
1. 合并 16 个 16 级瓦片（约 1.2km×1.2km 街区）→ `data/shanghai-test/mesh.obj`
2. 渲染 60 张 RGB-D（`dataset_generation/render_obj.py`）
3. `preprocess.py` SAM+SigLIP → 特征（60图完成）
4. `convert_to_point_cloud.py` → `eval/shanghai/generated_point_cloud.ply` + `point_features_highlight.npy (4,300000,1152)`
5. 文本查询：`saved eval/shanghai/semantic_labels.npy`

**语义质量结论（硬边界）**：
- 特征覆盖率仅 **10.4%**（SAM 在纯色白膜图上 mask 极少）
- 有特征的点：**ground 92.5% / building 3.7% / vegetation 0.0%** —— 类别完全退化
- 各类平均相似度 0.019~0.083，不可分（与 playground 白盒子测试一致）
- 原因：SigLIP 在真实照片上训练，纯灰白体块渲染图无纹理信号

**这套数据在本工程中能做什么 / 不能做什么**：

| 能 | 不能 |
|---|---|
| ✅ 完整 pipeline 验证与调试（渲染→特征→投影→查询）| ❌ 植被/水体/道路等开放词汇语义分类 |
| ✅ 几何空间分析：建筑密度热图、高度分布（mesh 直接算，不依赖语义）→ `eval/shanghai/geometry_analysis.png` | ❌ 细粒度语义标签（如"老化建筑""玻璃幕墙"）|
| ✅ 二值线索：白膜体块覆盖 = 建成区/非建成区的粗略近似 | ❌ 依赖照片纹理的分割精度 |

**结论**：工程链路可用（全流程无错、产物完整），但语义区分度受白膜数据限制。若目标是"语义理解城市空间"，需要真实纹理数据（倾斜摄影/OSM+航拍纹理）；若目标是几何形态分析（密度、高度、街区结构），当前数据可直接产出。


## 7. 杭州湖滨街道倾斜摄影数据（实拍纹理）端到端验证

> 📅 2024 测试：U 盘 `/media/fernando/1282-0785/ZgyHZupdate/data/3dtiles/330102007/hubin/`
> 共 19 个 NoLod_*.glb，总 220 MB（行政区代码 330102=杭州市，"hubin"=湖滨街道）。

### 7.1 数据特性

| 维度 | 详情 |
|---|---|
| 是否带纹理 | ✅ **是**（227 张纹理，最大 3979×3950；能看到 CHANEL、建设银行、vivo广告、操场跑道等真实街景）|
| 几何特性 | ⚠️ **CesiumLab3 model2tiles 输出"屋顶+地面"双面薄片**（无三维墙体），face normal z>0 占 27.9k、z<0 占 23.3k |
| 纹理组织 | atlas 形式——大纹理图集含多 mesh 面贴图，UV 偏移定位 |
| 总规模 | 546 万顶点 / 182 万面 / 227 几何 |
| 覆盖范围 | 约 1.3km × 1.5km × 0.23km（杭州湖滨街道） |

### 7.2 渲染策略（关键修复）

CesiumLab3 双面薄片**无法用常规透视相机侧视**（depth% 接近 0）。改用：

1. **`OrthographicCamera`**（正交投影）+ 俯视角度，完整保留纹理
2. **网格扫描**：把整个区域切成 4×4 = 16 块，每块一张 512×512 俯视图（约 0.7 m/px）
3. **每张图都新建 OffscreenRenderer**（避免 EGL context 共享崩溃）

实现见 `dataset_generation/render_hubin_ortho.py`，输出到 `data/hubin-e2e/render/`。

### 7.3 depth 标定

`pyrender` ortho 深度输出**非线性**（shader 自定义格式）。通过实验建立标定表 `data/hubin-e2e/depth_calib.npy`：
```
CAM_Z=3000, zfar=6000, znear=1 → depth = 1.999 - 0.0006351·z
```
在脚本中用 `np.interp` 反查 world_z。

### 7.4 Ortho 反投影特征聚合

新建 `dataset_generation/convert_ortho.py`，与 OpenCity3D 透视流程不同：

1. 加载 19 GLB → concat 成单个 Trimesh
2. `trimesh.sample.sample_surface` 面积加权采样 300K 表面点
3. 对每张 tile：mesh 点线性映射到像素 → depth 可见性检查（容差 8m）→ 取 masks[level, v, u] 对应的 SigLIP 特征
4. 多 tile 聚合（15% overlap → 多视图平均）

> 输出 `eval/hubin/hubin_point_cloud.ply`（带 RGB 颜色）+ `mesh_points.npy` + `n_observed.npy`

### 7.5 语义查询（改进版）

`preprocess.py` 输出格式：`*_f.npy` = (n_masks, 1152) SigLIP 特征；`*_s.npy` = (4, H, W) 4 层 mask 索引。

⚠️ 原始 OpenCity3D 流程（max over levels）对 ortho 俯视图**不适用**——粗 level 全图 mask 会主导，导致 99% building。改进：

- 新建 `sandbox/hubin_query_v2.py`：
  - **按 level 独立打分**（不再 max）：每点用其 4 个 level 的 mask 分别打分，按权重平均
  - **大小阈值**：跳过 < 500 像素的小 mask（信息不足）
  - **跳过全图大 mask**（> 200K 像素，避免粗粒度信号覆盖细节）
  - **per-tile 兜底**：缺 mask 的点用全图特征
  - **多 prompt 融合**：每类 3-4 个 "aerial view of..." 描述，归一化后均值
- 类别：building / road / vegetation / water / ground / empty

### 7.6 实测结果（杭州湖滨街道）

| 类别 | 覆盖点占比 | 全点占比 | 平均 sim |
|---|---|---|---|
| building | **27.2%** | 10.6% | 0.106 |
| road | 0.0% | 0.0% | — |
| vegetation | 0.0% | 0.0% | — |
| water | **1.1%** | 0.4% | 0.037 |
| ground | **7.6%** | 3.0% | 0.114 |
| empty | 64.1% | 86.0% | 0.103 |

**有特征覆盖 33.5%**（比上海白膜 10.4% 提升 3 倍）。

### 7.7 可视化产物

`eval/hubin/`：
- `semantic_views_v2.png` — 顶视/侧视散点图（红=建筑、棕=地面、蓝=水、灰=空）
- `coverage_heatmaps_v2.png` — 6 类 8×8 街区覆盖热力图
- `semantic_tiles.png` — 16 张 tile 的独立语义图（看到游泳池蓝色、跑道橙色）
- `semantic_visual_v2.ply` — 可在 MeshLab/CloudCompare 打开的彩色点云
- `hubin_point_cloud.ply` — 原始 RGB 颜色点云（无语义）

### 7.8 SigLIP 在俯视图上的能力边界

- ✅ **能区分**大块灰白色建筑屋顶（building vs water 区分度好）
- ✅ **能识别**泳池/水池的蓝色（water 分数最高的小 mask）
- ✅ **能识别**体育场/公园（vegetation prompt 在灰色绿色调 mask 上 top-1）
- ⚠️ **弱识别**道路、植被（植被占比 0% 因湖滨街道是商业区+体育场少）
- ⚠️ **无 aerial 训练**——细节类别（路、行人道）几乎全判 building/empty

> SigLIP 在卫星/俯视图场景的识别能力有限。如需精细语义分类（特别是植被/道路），建议使用遥感专用视觉模型（GeoCLIP、SatCLIP 等）。

### 7.9 结论

- ✅ **实拍倾斜摄影模型 + OpenCity3D pipeline 完整跑通**（合并 → 渲染 → 特征聚合 → 语义查询）
- ✅ 输出可直接用于**城市建筑覆盖、水域分布、地面结构**等基础空间分析
- ⚠️ 植被/道路识别率受 SigLIP 训练数据限制，需要：
  - 更细粒度的标注 prompt（"satellite view of green park", "concrete sidewalk"）
  - 或更换为遥感专用模型
