"""Nb / 任意单晶的交互式 CBED + 衍射/菊池带模拟 GUI。

左侧：衍射花样（g⊥ 线性投影）+ 菊池带（Kossel 锥 gnomonic 直线近似），
      鼠标左键拖动旋转晶体，带轴同步更新；
左上：参数面板（导入 CIF、带轴设置、面内旋转、厚度、加速电压等）。
右侧：CBED 参数面板 + "CBED 模拟"按钮，按下后后台多片法计算并显示。

运行：  conda run -n temsim python cbed_gui.py
依赖：  numpy, scipy, matplotlib, tkinter, ase, abtem,
        diffsims, orix, diffpy.structure（衍射斑/菊池带用专业库计算）
"""

import os
import queue
import threading
import time

import numpy as np

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter

# 衍射斑物理灰度色标：浅灰 → 黑（弱斑柔和、强斑近黑）。
# 不用内置 Greys：其低端为纯白，弱斑在浅色背景上会显成刺眼白点。
GREY_PHYS = LinearSegmentedColormap.from_list(
    "grey_phys", ["#d9d9d9", "#000000"])

# 让 matplotlib 正确显示中文（Windows 上优先微软雅黑/黑体）
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "PingFang SC",
                                   "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# =============================================================================
# 默认参数（启动时载入，界面里可再改）
# =============================================================================
DEFAULTS = {
    "energy_kev": 200.0,       # 加速电压 (keV)
    "thickness_nm": 20.0,      # 样品厚度 (nm)
    "gmax": 1.2,               # 倒易矢量截断 |g| (1/Å)，决定所画斑/带的数量
    "spot_gmax_ref": None,     # 衍射斑灰度/消光的固定绝对参考口径 (1/Å)。
                               # 默认取 gmax 初始值(1.2)：消光阈值与灰度基准据此计算，
                               # 不随用户调节探测口径漂移，避免弱斑随口径调小而突然显现。
    "brightness": 1.0,         # 衍射斑/菊池带显示亮度倍率（0~10，默认 1）
    "max_hkl": 6,              # 生成倒易矢量的最大 Miller 指数（用于菊池带列表）
    "proj_rmax": 0.92,         # (已由 cam_len 计算取代) 屏幕最大半径（gnomonic 投影，g⊥=gmax 时映射到该半径）
    "cam_len": 30.56,          # 相机长度(L)：参考圆半径 rmax = L·tan(2θ_max)，
                               # θ_max=arcsin(λ·gmax/2) 为 gmax 探测口径对应的最大布拉格半角。
                               # 默认 L 使 gmax=1.2、200keV(λ=0.02508Å) 时 rmax≈0.92，与旧观感一致。
    "exc_spot": 0.04,          # 衍射斑最大激发误差 s_g (1/Å)：|s|≥此值的反射直接剔除（硬门限）。
                               # 越小→远角/离轴弱斑随倾转更快消失；太大(如0.10)会有大量
                               # 远处的弱斑残存、难以用倾斜消除。
    "exc_width": 0.02,         # 激发包络半宽 σ (1/Å)：Gaussian env=exp(-(s/σ)²)。
                               # 越小→斑点随激发误差衰减越快(比 Lorentzian 尾部更陡)，
                               # 弱激发斑快速淡出、数量显著减少；σ=0.05 的尾部太宽会显示大量弱斑。
                               # 物理上 s·t 相位累积大（t≈200Å 时 s=0.05 已累积 31rad），
                               # 离轴反射应近零强度；σ=0.05 的尾部太宽会显示大量弱斑。
    "band_max": 0.85,          # 绘制菊池带的最大 |g| (1/Å)，避免带过密
    # CBED 晶体取向取整：构造超胞用的整数带轴上限（最大分量放大到该值再取整约分）。
    # 越大带轴读数越精细/越连续，12000 时角误差≈0；代价是远离低指数轴时超胞原子数可能增大。
    "zone_max_index": 12000,
    # CBED 参数
    "semiangle_mrad": 4.32,    # 会聚半角 (mrad)
    "max_angle_mrad": 30.0,    # CBED 显示最大角 (mrad)
    "ncells": 32,              # 等价横向超胞数（用于估算 box_width_nm 默认），已不再直接使用
    "box_width_nm": 10.8,      # 束框横向边长 (nm)，构造长方体样品，横向≈32×晶胞长
    "sampling": 0.2,           # 实空间采样 (Å/px)
    "scalebar_nm": 5.0,        # 导出 CBED 图底部标尺长度 (nm^-1)，默认 5
    "dpi": 150,                # CBED 出图分辨率
    # STEM 原子像
    "stem_row_chunk": 8,       # 手动分块扫描时每批的扫描行数。STEM 扫描必须分批，
                               # 否则 abTEM lazy=False 会一次性物化"全部扫描位置 × 势网格"
                               # (如 128×128×1080×1080 complex64) 而内存爆炸。
                               # 分批后单批内存 ≈ row_chunk×gx×(势网格)，可线性控内存。
}

Z_AXIS = np.array([0.0, 0.0, 1.0])

# 电镜双倾样品杆的旋转轴方向选项（实验室/绘图区坐标，屏幕 +X 向右、+Y 向上）
TEM_TILT_AXES = {
    "向下 (−Y)": (0.0, -1.0, 0.0),
    "向上 (+Y)": (0.0, 1.0, 0.0),
    "向右 (+X)": (1.0, 0.0, 0.0),
    "向左 (−X)": (-1.0, 0.0, 0.0),
}


# =============================================================================
# 晶体学 / 数学工具（纯 numpy，无 GUI 依赖）
# =============================================================================
def rotation_align(a, b):
    """返回把单位向量 a 旋转到单位向量 b 的 3x3 正交矩阵（Rodrigues）。"""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    s = np.linalg.norm(v)
    c = np.dot(a, b)
    if s < 1e-12:
        if c > 0:
            return np.eye(3)
        # 反向共线：MUST be a proper rotation (det=+1). -np.eye(3) has det=-1
        # (improper = mirror) and would FLIP reciprocal handedness — exactly the
        # bug that makes zone axis [00-1] (v ∥ −z) display 200/020 as a mirror
        # instead of a proper rotation. Use a 180° rotation about an axis ⊥ a
        # instead. For a ≈ ±z pick the x-axis (Rx(π)): consistent with the
        # align_c_to_z convention "a* → +x", so after reversing the zone axis
        # 200 stays to the right (+x) and 020 flips to −y.
        if abs(a[2]) > 0.9:
            return np.array([[1.0, 0.0, 0.0],
                             [0.0, -1.0, 0.0],
                             [0.0, 0.0, -1.0]])          # Rx(π), det=+1
        e = np.eye(3)[int(np.argmin(np.abs(a)))]
        t = np.cross(a, e)
        t = t / np.linalg.norm(t)
        return 2.0 * np.outer(t, t) - np.eye(3)          # 180° about t, det=+1
    k = np.array([[0, -v[2], v[1]],
                  [v[2], 0, -v[0]],
                  [-v[1], v[0], 0]])
    return np.eye(3) + k + k @ k * ((1.0 - c) / (s * s))


# 旋转矩阵（Rx/Ry/Rz）为“left-multiply on column vectors”的主动旋转约定：
# 左乘列向量 v' = Rx(θ)@v 即对 v 做主动旋转；若要在行向量侧使用（如
# g_lab = G @ R），可先转置 v_row @ Rx(θ).T，或利用正交性写 Rx(θ).T=Rx(-θ)
# 即 v_row @ Rx(-θ)。
def Rx(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def Ry(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def Rz(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def rotation_about_axis(d, angle):
    """绕任意单位轴 d 的主动旋转矩阵（左乘列向量）：v' = R @ v。

    与 Rx/Ry/Rz 同族；用于绕 XY 平面内旋转轴 d 旋转 angle 弧度。
    作用方式同 _on_motion 拖拽：左乘 self.R，即绕实验室固定轴转晶体（束不动）。
    """
    d = np.asarray(d, float)
    n = np.linalg.norm(d)
    if n < 1e-12:
        return np.eye(3)
    d = d / n
    c, s = np.cos(angle), np.sin(angle)
    k = np.array([[0, -d[2], d[1]],
                  [d[2], 0, -d[0]],
                  [-d[1], d[0], 0]])
    return np.eye(3) + s * k + (1.0 - c) * (k @ k)


def reciprocal_basis(cell):
    """结晶学倒易基（不含 2π），|g| = 1/d，单位 1/Å。"""
    a1, a2, a3 = (np.asarray(v, float) for v in cell)
    vol = np.dot(a1, np.cross(a2, a3))
    b1 = np.cross(a2, a3) / vol
    b2 = np.cross(a3, a1) / vol
    b3 = np.cross(a1, a2) / vol
    return np.array([b1, b2, b3])


def miller_to_cartesian(hkl, cell):
    A = np.asarray(cell, float).T
    return A @ np.asarray(hkl, float)


def _gcd(a, b):
    a, b = abs(int(a)), abs(int(b))
    while b:
        a, b = b, a % b
    return a


def _reduce_int(vec):
    vals = [int(round(x)) for x in vec]
    g = 0
    for x in vals:
        g = _gcd(g, x)
    if g == 0:
        return tuple(vals)
    return tuple(x // g for x in vals)


def reduce_hkl(c, max_idx=24):
    """把浮点比例 (h,k,l) 归约成整数元组。

    扫描 1..max_idx，找与方向最接近的精确原始整数轴（各分量取整后约分）。
    好处：输入本身就是小整数轴（如 [0 1 10]）时能精确还原，不因"把最大
    分量强放大到 max_idx"而四舍五入出错（旧算法 2.4->2 把 [0 1 10] 讹成
    [0 1 12]）；小角度拖拽时仍能连续给出最近的整数近似。
    """
    c = np.asarray(c, float)
    c[np.abs(c) < 1e-11] = 0.0
    base = np.abs(c).max()
    if base < 1e-11:
        return (0, 0, 0)
    cu = c / base                      # 最大分量归一为 ±1
    best, besterr = None, np.inf
    for k in range(1, max_idx + 1):
        t = np.round(cu * k)
        if np.max(np.abs(t)) < 0.5:
            continue
        p = np.asarray(_reduce_int(t), int)
        if np.max(np.abs(p)) > max_idx:
            continue
        pn = p / (np.linalg.norm(p) + 1e-12)
        err = 1.0 - abs(np.dot(cu, pn))
        if err < besterr:
            best, besterr = p, err
    if best is None:
        return _reduce_int(cu * max_idx)
    return (int(best[0]), int(best[1]), int(best[2]))


def cartesian_to_miller(v, cell):
    A = np.asarray(cell, float).T
    coef = np.linalg.solve(A, np.asarray(v, float))
    return reduce_hkl(coef)


def closest_zone_plane(hkl, uvw, max_idx=24):
    """把晶面指数 (h,k,l) 规范成"垂直于带轴 [uvw]"的最近整数解。

    带轴 [u v w] 与晶面 (h k l) 正交（倒易矢 g∥法向 ⊥ 正空间 [uvw]）的
    充要条件是它们满足晶带定律（zone law）：u·h + v·k + w·l = 0。
    若输入 hkl 不满足该式，则在 2·R+1 的整数格内枚举满足晶带定律的整数
    三元组，取与输入方向最接近（夹角尽可能小、保留约分后的最小指数）者
    作为合法替代；找不到则返回 None。
    """
    u = np.asarray(uvw, float)
    infl = np.asarray(hkl, float)
    if np.linalg.norm(infl) < 1e-12:
        return None
    if abs(u[0] * infl[0] + u[1] * infl[1] + u[2] * infl[2]) < 1e-9:
        return tuple(_reduce_int(infl))     # 已满足晶带定律：约分还原成最小整数
    R = int(max(6.0, float(np.abs(infl).max()) * 1.5 + 2.0))
    R = min(R, max_idx)
    best, best_hit = None, -1.0
    rng = range(-R, R + 1)
    for h in rng:
        for k in rng:
            for l in rng:
                if abs(u[0] * h + u[1] * k + u[2] * l) > 1e-9:
                    continue
                if h == 0 and k == 0 and l == 0:
                    continue
                t = np.array([h, k, l], float)
                hit = abs(np.dot(infl, t)) / (
                    np.linalg.norm(infl) * np.linalg.norm(t))
                if hit > best_hit:
                    best_hit, best = hit, (h, k, l)
    if best is None:
        return None
    return tuple(_reduce_int(best))


def snap_zone_axis(v, cell, max_idx):
    """把晶格(笛卡尔)方向 v 取整到最近的整数带轴 [u v w]。

    复用 reduce_hkl 的"放大-取整-约分"连分式式方法：把最大分量放大到
    max_idx 再取整并约分，得到原方向的最佳整数近似。无需暴力枚举，
    指数上限再大也只是该 max_idx 的数值，不增加计算量。
    """
    A = np.asarray(cell, float).T
    coef = np.linalg.solve(A, np.asarray(v, float))
    return reduce_hkl(coef, max_idx=max_idx)


def align_c_to_z(atoms):
    """把晶胞 c 轴对齐到 z，并把倒易矢 a*（(100) 方向）对准 +x 正方向。

    步骤：先用最短旋转把 c 转到 z；c∥z 后 a*=b×c/V 与 c 正交、z 分量为 0，
    完全落在 xy 面内，再绕 z 旋转即可让 (100) 严格指向 +x——对任意晶系
    （立方/四方/六方/单斜/三斜）都精确成立。返回总对齐矩阵 A。
    """
    cell = np.asarray(atoms.get_cell()[:], float).copy()
    pos = np.asarray(atoms.get_positions(), float).copy()
    c = cell[2]
    n = np.linalg.norm(c)
    if n < 1e-9:
        return np.eye(3)
    A = rotation_align(c / n, Z_AXIS)
    cell = (A @ cell.T).T
    pos = (A @ pos.T).T
    # 第二自由度：绕 z 旋转，把倒易矢 a*=(100) 对准 +x（a*_x>0，a*_y=0）。
    a_star = reciprocal_basis(cell)[0]
    phi = -np.arctan2(a_star[1], a_star[0])
    B = Rz(phi)
    cell = (B @ cell.T).T
    pos = (B @ pos.T).T
    atoms.set_cell(cell)
    atoms.set_positions(pos)
    atoms.wrap()
    return B @ A


_LOBATO = None                 # Lobato–Van Dyck 参数化单例（延迟加载）
_FF_CACHE = {}                 # 元素符号 -> 散射因子函数缓存


def _electron_ff(symbol):
    """返回元素 symbol 的电子散射因子函数 f(k²)，k²=s²=(sinθ/λ)² (Å⁻²)。

    采用 Lobato–Van Dyck (2014) 参数化（abtem 内置，覆盖 1–103 号元素，
    即 diffsims `xtables` 用到的同一套表），f 单位 Å。
    """
    global _LOBATO
    fn = _FF_CACHE.get(symbol)
    if fn is None:
        if _LOBATO is None:
            from abtem.parametrizations import LobatoParametrization
            _LOBATO = LobatoParametrization()
        fn = _LOBATO.scattering_factor(symbol)
        _FF_CACHE[symbol] = fn
    return fn


def structure_factor(hkl, g, atoms):
    """结构因子模 |F|（Lobato–Van Dyck 参数化电子散射因子，中性原子、0 K）。

    与旧的 f≈Z·exp(-0.3 s²) 相比，散射因子随 s=sinθ/λ 的衰减真实得多、
    且原子序数/壳层结构正确，故 |F| 与消光距离显著更准确。
    """
    fracs = atoms.get_scaled_positions()
    symbols = atoms.get_chemical_symbols()
    s2 = (np.linalg.norm(g) / 2.0) ** 2
    hkl = np.asarray(hkl, float)
    F = 0.0 + 0.0j
    for frac, sym in zip(fracs, symbols):
        f = float(_electron_ff(sym)(np.array([s2]))[0])
        F += f * np.exp(2j * np.pi * np.dot(hkl, frac))
    return abs(F)


def build_g_list(cell, gmax, max_hkl):
    """生成 |g|<=gmax 的 (hkl, g) 列表（g 单位 1/Å）。"""
    b = reciprocal_basis(cell)
    out = []
    for h in range(-max_hkl, max_hkl + 1):
        for k in range(-max_hkl, max_hkl + 1):
            for l in range(-max_hkl, max_hkl + 1):
                if h == k == l == 0:
                    continue
                g = np.array([h, k, l]) @ b
                if np.linalg.norm(g) <= gmax:
                    out.append(((h, k, l), g))
    return out


def _friedel_key(hkl):
    """把 (h,k,l) 与 (-h,-k,-l) 归到同一规范键，用于菊池带去重。"""
    h, k, l = (int(x) for x in hkl)
    if h > 0:
        return (h, k, l)
    if h < 0:
        return (-h, -k, -l)
    if k > 0:
        return (h, k, l)
    if k < 0:
        return (-h, -k, -l)
    if l > 0:
        return (h, k, l)
    return (-h, -k, -l)


def _circle_line(a, b, c, r):
    """圆 x²+y²=r² 与直线 a·x+b·y+c=0 的两个交点；无两交点时返回 None。"""
    d = a * a + b * b
    if d < 1e-12:
        return None
    den = np.sqrt(d)
    dist = abs(c) / den
    if dist >= r - 1e-9:
        return None
    x0, y0 = -a * c / d, -b * c / d
    ux, uy = -b / den, a / den
    s = np.sqrt(max(r * r - dist * dist, 0.0))
    return (x0 + s * ux, y0 + s * uy), (x0 - s * ux, y0 - s * uy)


# =============================================================================
# GUI
# =============================================================================
class CBEDApp:
    def __init__(self, root):
        self.root = root
        self.root.title("交互式 CBED / 衍射·菊池带模拟")
        self.root.geometry("1280x760")

        self.cfg = dict(DEFAULTS)
        self.atoms = None          # 对齐后的 ASE Atoms（c 轴平行 z）
        self.cell = None           # 对齐后的晶胞 (3x3)
        # 晶体取向矩阵：**左乘列向量 = 晶体→实验室**（self.R @ v_col）。
        # 行向量右乘（旧约定）等价需用 self.R.T。相机/衍射显示统一用新约定：
        #   · 衍射位置 g_lab = G @ self.R.T
        #   · 带轴/倾角（实验室 z → 晶体方向）= self.R @ Z
        # CBED 提供原子位置的功能沿用"束框=实验室坐标"，其内部用旧(行)约定
        # 计算，经边界传 self.R.T 接入。
        self.R = np.eye(3)
        # 电镜倾转：“设置当前倾转角”只记录读数、不改变下方花样，并反推电镜 0
        # 倾转状态 R_0 = [Rα(α) Rβ(β)]ᵀ @ R；目标倾转面板再由 R_0 起算
        # R = Rα(α_tgt) @ Rβ(β_tgt) @ R_0。
        self._tem_tilt_R0 = None

        self.g_list = []           # [(hkl, g)]
        self.F2 = {}               # hkl -> F^2
        self.phase = None          # orix Phase（用于 diffsims 衍射斑）
        self.gen = None            # diffsims SimulationGenerator
        self._drag = None          # 拖拽状态 (last_x, last_y)
        self._spot_mx = 0.0        # 当前普适灰度满标（参考球内最强反射 × 亮度）
        self._press_pt = None      # 左键按下点 (px, py)，用于区分点击 / 拖拽
        self._spot_data = []       # 点击衍射斑所需的物理量（坐标/指数/|F|/|g|/强度）
        self._spot_bubble = None   # 当前衍射斑气泡标注（annotate 对象）
        self._wavelength = None    # 当前加速电压对应的电子波长 (Å)
        self._rmax = 0.0           # 左面板参考圆显示半径
        self._redraw_pending = False   # 节流重绘标志（合并高频 motion 事件）
        self._result_q = queue.Queue()
        self._running = False
        self._cancel_event = threading.Event()

        # ---- STEM 原子像独立运行状态/队列/取消 ----
        self._stem_result_q = queue.Queue()
        self._stem_running = False
        self._stem_cancel = threading.Event()
        self._stem_start = None

        # ---- 旋转操控盘（绕 XY 平面内直径方向旋转）----
        self._rot_phi = 0.0            # 指针角（弧度，0=+X，π/2=+Y，页面上为向右/向上）
        self._rot_dial = None           # 磁盘 tk.Canvas
        self._rot_arrow = None          # 左图旋转方向箭头 artist
        self._show_rot_arrow = False    # 是否在左图中显示旋转方向箭头

        self._build_layout()
        # 初始不加载任何晶体，等待用户通过"导入 CIF"自行导入

    # ------------------------------------------------------------------ 布局
    def _build_layout(self):
        paned = ttk.PanedWindow(self.root, orient="horizontal")
        paned.pack(fill="both", expand=True)

        # ---------------- 左 ----------------
        self.left = ttk.Frame(paned)
        paned.add(self.left, weight=3)

        self._build_left_panel(self.left)
        # 带轴/束倾状态（置于面板下方、图像上方，仿 CBED 面板的"就绪"状态）
        ttk.Label(self.left, textvariable=self.zone_label, foreground="#3366cc",
                  font=("", 9)).pack(fill="x", anchor="w", padx=6, pady=(2, 0))
        self.fig_left = Figure(figsize=(5.0, 5.2), dpi=100)
        self.ax_left = self.fig_left.add_subplot(111)
        self.canvas_left = FigureCanvasTkAgg(self.fig_left, master=self.left)
        self.canvas_left.get_tk_widget().pack(fill="both", expand=True)
        self._bind_left_mouse()

        # ---------------- 右 ----------------
        self.right = ttk.Frame(paned)
        paned.add(self.right, weight=3)

        self._build_right_panel(self.right)
        self.fig_right = Figure(figsize=(4.6, 4.9), dpi=100)
        self.ax_right = self.fig_right.add_subplot(111)
        # CBED 画布放进"CBED 控制"选项卡内（fill both expand），而非右侧整体
        self.canvas_right = FigureCanvasTkAgg(self.fig_right, master=self.cbed_tab)
        self.canvas_right.get_tk_widget().pack(fill="both", expand=True)
        # 右侧 CBED：滚轮缩放（围绕光标，约束在全幅内）
        self.canvas_right.mpl_connect("scroll_event", self._on_cbed_scroll)

    def _build_left_panel(self, parent):
        nb = ttk.Notebook(parent)
        nb.pack(fill="x", padx=4, pady=4)

        # ============================ 选项卡：晶体设置 ============================
        tab_crystal = ttk.Frame(nb)
        nb.add(tab_crystal, text="晶体设置")

        crystal_row = ttk.Frame(tab_crystal)
        crystal_row.pack(fill="both", expand=True, padx=2, pady=2)

        # ---------- 左列：晶体设置控件 ----------
        left_c = ttk.Frame(crystal_row)
        left_c.pack(side="left", fill="y", anchor="n", padx=(4, 10))

        row0 = ttk.Frame(left_c)
        row0.pack(fill="x", padx=4, pady=2)
        self.cif_label = tk.StringVar(value="未加载 CIF")
        ttk.Button(row0, text="导入 CIF", command=self._on_import_cif).pack(side="left")
        # 导入 CIF 后此处显示 化学式(_chemical_formula_structural) + 空间群符号
        self.crystal_info_label = tk.StringVar(value="")
        ttk.Label(row0, textvariable=self.crystal_info_label,
                  foreground="#3366cc", font=("", 9)).pack(
            side="left", padx=(6, 0))

        row1 = ttk.Frame(left_c)
        row1.pack(fill="x", padx=4, pady=2)
        ttk.Label(row1, text="带轴 [u v w]").pack(side="left")
        self.uvw_vars = [tk.StringVar(value="0"), tk.StringVar(value="0"),
                         tk.StringVar(value="1")]
        for i, v in enumerate(self.uvw_vars):
            ttk.Entry(row1, textvariable=v, width=4).pack(side="left", padx=2)
        ttk.Button(row1, text="设置带轴", command=self._on_set_zone).pack(side="left", padx=4)
        self.zone_label = tk.StringVar(value="当前带轴: [0 0 1]")

        row2 = ttk.Frame(left_c)
        row2.pack(fill="x", padx=4, pady=2)
        ttk.Label(row2, text="x轴晶面 (h k l)").pack(side="left")
        self.hkl_vars = [tk.StringVar(value="1"), tk.StringVar(value="0"),
                         tk.StringVar(value="0")]
        for i, v in enumerate(self.hkl_vars):
            ttk.Entry(row2, textvariable=v, width=4).pack(side="left", padx=2)
        ttk.Label(row2, text="(设带轴时对准+ X)", foreground="#888").pack(
            side="left", padx=2)

        row2b = ttk.Frame(left_c)
        row2b.pack(fill="x", padx=4, pady=2)
        ttk.Label(row2b, text="面内旋转角度").pack(side="left")
        self.inplane_var = tk.StringVar(value="0.0")
        ttk.Entry(row2b, textvariable=self.inplane_var, width=6).pack(side="left", padx=2)
        ttk.Label(row2b, text="°").pack(side="left")
        ttk.Button(row2b, text="应用", command=self._on_inplane).pack(side="left", padx=4)
        ttk.Button(row2b, text="重置方向", command=self._on_reset).pack(side="left", padx=6)

        # ---------- 右列：旋转操控盘（绕 XY 平面内任意直径方向旋转）----------
        # 指针方向确定旋转轴 d，圆下输入旋转角（mrad），确认后按
        # self.R ← rotation_about_axis(d, θ) @ self.R（左乘，绕实验室固定轴转晶体），
        # 与拖拽旋转（Rx/Ry）同族。
        right_c = ttk.Frame(crystal_row)
        right_c.pack(side="left", fill="y", anchor="n", padx=(0, 4))

        ttk.Label(right_c, text="旋转操控（绕 XY 平面内方向）",
                  foreground="#666").pack(anchor="w", padx=2)

        dial_size = 150
        dial = tk.Canvas(right_c, width=dial_size, height=dial_size,
                         bg="white", highlightthickness=1,
                         highlightbackground="#bbb")
        dial.pack(padx=2, pady=2)
        self._rot_dial = dial
        self._dial_c = dial_size / 2.0          # 圆心
        self._dial_r = dial_size / 2.0 - 16     # 圆半径（留边给刻度文字）

        # 指针方向角（与 x 轴夹角，单位度，默认 0° 指向 +X），回车/失焦生效
        # （放旋转角行之上；与旋转角行在 Y 轴上位置对调）
        rowp = ttk.Frame(right_c)
        rowp.pack(fill="x", padx=2, pady=(0, 2))
        ttk.Label(rowp, text="指针角 (X 轴, °)").pack(side="left")
        self._rot_dir_var = tk.StringVar(value="0")
        re = ttk.Entry(rowp, textvariable=self._rot_dir_var, width=6)
        re.pack(side="left", padx=2)
        re.bind("<Return>", self._on_rot_dir_entry)
        re.bind("<FocusOut>", self._on_rot_dir_entry)

        rowr = ttk.Frame(right_c)
        rowr.pack(fill="x", padx=2, pady=(0, 2))
        ttk.Label(rowr, text="旋转角 (mrad)").pack(side="left")
        self._rot_angle_var = tk.StringVar(value="1")
        ttk.Entry(rowr, textvariable=self._rot_angle_var, width=6).pack(
            side="left", padx=2)
        ttk.Button(rowr, text="旋转", command=self._on_rot_apply).pack(
            side="left", padx=2)
        self._rot_arrow_lbl = tk.StringVar(value="显示箭头")
        ttk.Button(rowr, textvariable=self._rot_arrow_lbl,
                   command=self._on_toggle_rot_arrow).pack(side="left", padx=2)

        dial.bind("<Button-1>", self._on_dial_drag)
        dial.bind("<B1-Motion>", self._on_dial_drag)
        self._draw_dial()

        # ============================ 选项卡：电镜倾转 ============================
        # 双倾样品杆：α、β 两个倾转轴（实验室/绘图区坐标）。α 轴默认向下(−Y)，
        # β 轴默认向右(+X)。目标状态 R = Rα(α) @ Rβ(β) @ R_0（先 β 后 α）：
        # β 轴随样品杆 α 角一起转动，故必须在 α 未动时先绕 β 轴转，再绕 α 轴转。
        # “设置当前倾转角”只是记录当前花样的读数（不旋转），供目标倾转起算。
        tab_tem_tilt = ttk.Frame(nb)
        nb.add(tab_tem_tilt, text="电镜倾转")

        frame_tem_tilt_cur = ttk.LabelFrame(tab_tem_tilt, text="设置当前倾转角")
        frame_tem_tilt_cur.pack(fill="x", padx=4, pady=(4, 2))

        row_tem_tilt_alpha = ttk.Frame(frame_tem_tilt_cur)
        row_tem_tilt_alpha.pack(fill="x", padx=4, pady=2)
        ttk.Label(row_tem_tilt_alpha, text="α 倾转角").pack(side="left")
        self.tem_tilt_alpha_var = tk.StringVar(value="0.0")
        ttk.Entry(row_tem_tilt_alpha, textvariable=self.tem_tilt_alpha_var, width=7).pack(
            side="left", padx=2)
        ttk.Label(row_tem_tilt_alpha, text="°").pack(side="left")
        ttk.Label(row_tem_tilt_alpha, text="α 旋转轴").pack(side="left", padx=(12, 2))
        self.tem_tilt_alpha_axis_var = tk.StringVar(value="向下 (−Y)")
        ttk.Combobox(row_tem_tilt_alpha, textvariable=self.tem_tilt_alpha_axis_var, width=10,
                     state="readonly",
                     values=list(TEM_TILT_AXES.keys())).pack(side="left")

        row_tem_tilt_beta = ttk.Frame(frame_tem_tilt_cur)
        row_tem_tilt_beta.pack(fill="x", padx=4, pady=2)
        ttk.Label(row_tem_tilt_beta, text="β 倾转角").pack(side="left")
        self.tem_tilt_beta_var = tk.StringVar(value="0.0")
        ttk.Entry(row_tem_tilt_beta, textvariable=self.tem_tilt_beta_var, width=7).pack(
            side="left", padx=2)
        ttk.Label(row_tem_tilt_beta, text="°").pack(side="left")
        ttk.Label(row_tem_tilt_beta, text="β 旋转轴").pack(side="left", padx=(12, 2))
        self.tem_tilt_beta_axis_var = tk.StringVar(value="向右 (+X)")
        ttk.Combobox(row_tem_tilt_beta, textvariable=self.tem_tilt_beta_axis_var, width=10,
                     state="readonly",
                     values=list(TEM_TILT_AXES.keys())).pack(side="left")

        row_tem_tilt_apply = ttk.Frame(frame_tem_tilt_cur)
        row_tem_tilt_apply.pack(fill="x", padx=4, pady=(0, 2))
        ttk.Button(row_tem_tilt_apply, text="设置当前倾转角",
                   command=self._on_set_current_tem_tilt).pack(side="left")
        self.tem_tilt_info_var = tk.StringVar(
            value="尚未记录当前倾转（按 α = β = 0° 处理）")
        ttk.Label(frame_tem_tilt_cur, textvariable=self.tem_tilt_info_var,
                  foreground="#3366cc", font=("", 9)).pack(
            fill="x", padx=8, pady=(0, 2))

        ttk.Label(tab_tem_tilt,
                  text="注：β 轴实际随样品杆的 α 角一起转动，故模拟时须在 α 未动时"
                       "先旋转 β，再旋转 α（即先 β 后 α），倾斜结果才正确。",
                  foreground="#888", font=("", 9), justify="left",
                  wraplength=560).pack(fill="x", padx=8, pady=(0, 4))

        frame_tem_tilt_target = ttk.LabelFrame(tab_tem_tilt, text="设置目标倾转角")
        frame_tem_tilt_target.pack(fill="x", padx=4, pady=(2, 4))

        row_tem_tilt_target = ttk.Frame(frame_tem_tilt_target)
        row_tem_tilt_target.pack(fill="x", padx=4, pady=2)
        ttk.Label(row_tem_tilt_target, text="α 倾转角").pack(side="left")
        self.tem_tilt_target_alpha_var = tk.StringVar(value="0.0")
        ttk.Entry(row_tem_tilt_target, textvariable=self.tem_tilt_target_alpha_var, width=7).pack(
            side="left", padx=2)
        ttk.Label(row_tem_tilt_target, text="°").pack(side="left")
        ttk.Label(row_tem_tilt_target, text="β 倾转角").pack(side="left", padx=(12, 2))
        self.tem_tilt_target_beta_var = tk.StringVar(value="0.0")
        ttk.Entry(row_tem_tilt_target, textvariable=self.tem_tilt_target_beta_var, width=7).pack(
            side="left", padx=2)
        ttk.Label(row_tem_tilt_target, text="°").pack(side="left")
        ttk.Button(row_tem_tilt_target, text="确定",
                   command=self._on_set_target_tem_tilt).pack(side="left", padx=(16, 0))

        # ============================ 选项卡：样品信息 ============================
        tab_sample = ttk.Frame(nb)
        nb.add(tab_sample, text="样品信息")

        row3 = ttk.Frame(tab_sample)
        row3.pack(fill="x", padx=4, pady=2)
        ttk.Label(row3, text="厚度 (nm)").pack(side="left")
        self.thick_var = tk.StringVar(value=str(self.cfg["thickness_nm"]))
        ttk.Entry(row3, textvariable=self.thick_var, width=6).pack(side="left", padx=2)
        ttk.Label(row3, text="加速电压 (keV)").pack(side="left", padx=8)
        self.energy_var = tk.StringVar(value=str(self.cfg["energy_kev"]))
        ttk.Entry(row3, textvariable=self.energy_var, width=6).pack(side="left", padx=2)

        row3b = ttk.Frame(tab_sample)
        row3b.pack(fill="x", padx=4, pady=2)
        ttk.Label(row3b, text="横向边长 (nm)").pack(side="left")
        self.width_var = tk.StringVar(value=str(self.cfg["box_width_nm"]))
        ttk.Entry(row3b, textvariable=self.width_var, width=6).pack(
            side="left", padx=2)
        ttk.Label(row3b, text="采样 (Å/px)").pack(side="left", padx=8)
        self.sampling_var = tk.StringVar(value=str(self.cfg["sampling"]))
        ttk.Entry(row3b, textvariable=self.sampling_var, width=6).pack(
            side="left", padx=2)
        ttk.Label(row3b, text="(势场/波实空间采样)", foreground="#888").pack(
            side="left", padx=4)
        # 实空间采样决定倒空间奈奎斯特截止，即 CBED 可见的散射半角上限
        self._nyq_half_var = tk.StringVar()
        ttk.Label(row3b, textvariable=self._nyq_half_var, foreground="#888").pack(
            side="left", padx=4)
        self.sampling_var.trace_add("write", lambda *_: self._update_nyq_half())
        self.energy_var.trace_add("write", lambda *_: self._update_nyq_half())
        self._update_nyq_half()

        # ============================ 选项卡：显示 ============================
        tab_disp = ttk.Frame(nb)
        nb.add(tab_disp, text="显示")

        row4 = ttk.Frame(tab_disp)
        row4.pack(fill="x", padx=4, pady=2)
        self.show_spots = tk.BooleanVar(value=True)
        self.show_bands = tk.BooleanVar(value=True)
        self.show_index = tk.BooleanVar(value=True)
        ttk.Checkbutton(row4, text="衍射斑", variable=self.show_spots,
                        command=self._redraw_left).pack(side="left")
        ttk.Checkbutton(row4, text="菊池带", variable=self.show_bands,
                        command=self._redraw_left).pack(side="left")
        ttk.Checkbutton(row4, text="标注指数", variable=self.show_index,
                        command=self._redraw_left).pack(side="left", padx=6)

        # 探测角口径 rmax_msr：以 mrad 指定，对应的 gmax = 2·sin(半角)/λ。
        # 改动后重建倒易列表并重绘。默认探测口径 rmax = 30 mrad。
        row4b = ttk.Frame(tab_disp)
        row4b.pack(fill="x", padx=4, pady=2)
        ttk.Label(row4b, text="探测口径 rmax (mrad)").pack(side="left")
        self.aperture_var = tk.StringVar(value="30.00")
        ttk.Entry(row4b, textvariable=self.aperture_var, width=6).pack(
            side="left", padx=2)
        ttk.Button(row4b, text="应用", command=self._on_aperture).pack(side="left", padx=4)

        # 菊池带绘制数量限制：band_max = 绘制的最大 |g| (1/Å)，越大带越密。
        row4d = ttk.Frame(tab_disp)
        row4d.pack(fill="x", padx=4, pady=2)
        ttk.Label(row4d, text="菊池带限 |g|max (1/Å)").pack(side="left")
        self.bandlim_var = tk.StringVar(value=str(
            self.cfg.get("band_max", 0.85)))
        ttk.Entry(row4d, textvariable=self.bandlim_var, width=6).pack(
            side="left", padx=2)
        ttk.Button(row4d, text="应用", command=self._on_bandlim).pack(
            side="left", padx=4)

        # 亮度倍率滑块：0~10 倍，调节衍射斑/菊池带显示的明暗。
        row4c = ttk.Frame(tab_disp)
        row4c.pack(fill="x", padx=4, pady=2)
        ttk.Label(row4c, text="亮度倍率").pack(side="left")
        self.bright_var = tk.DoubleVar(value=float(
            self.cfg.get("brightness", 1.0)))
        self.bright_scale = ttk.Scale(
            row4c, from_=0.0, to=10.0, orient="horizontal",
            variable=self.bright_var,
            command=lambda _: self._redraw_left())
        self.bright_scale.pack(side="left", fill="x", expand=True, padx=4)
        self.bright_lbl = tk.StringVar(value=f"{self.bright_var.get():.2f}")
        ttk.Label(row4c, textvariable=self.bright_lbl, width=5).pack(
            side="left", anchor="e")
        # 实时更新数值显示
        def _bright_label(*_):
            self.bright_lbl.set(f"{self.bright_var.get():.2f}")
        self.bright_var.trace_add("write", _bright_label)

    def _build_right_panel(self, parent):
        # 右侧整体改为外层 Notebook：两个选项卡——"CBED 控制"（原全部内容）
        # 与"STEM 原子像"。各自含参数面板 + 独立的 matplotlib 画布。
        self.right_nb = ttk.Notebook(parent)
        self.right_nb.pack(fill="both", expand=True, padx=4, pady=4)

        # ---- 选项卡：CBED 控制（含内部"CBED 参数/导出设置/显示"子选项卡）----
        self.cbed_tab = ttk.Frame(self.right_nb)
        self.right_nb.add(self.cbed_tab, text="CBED 控制")

        f = ttk.LabelFrame(self.cbed_tab, text="CBED 控制")
        f.pack(fill="x", padx=4, pady=4)
        nb = ttk.Notebook(f)
        nb.pack(fill="x", padx=4, pady=4)

        def entryed(frame, label, key, width=6, unit="", default=None):
            r = ttk.Frame(frame)
            r.pack(fill="x", padx=4, pady=2)
            ttk.Label(r, text=label, width=14).pack(side="left")
            val = self.cfg[key] if default is None else default
            var = tk.StringVar(value=str(val))
            setattr(self, key + "_var", var)
            ttk.Entry(r, textvariable=var, width=width).pack(side="left", padx=2)
            if unit:
                ttk.Label(r, text=unit).pack(side="left")
            return var

        # ============================ 选项卡：CBED 参数 ============================
        tab_cbed = ttk.Frame(nb)
        nb.add(tab_cbed, text="CBED 参数")

        self.semiangle_var = entryed(tab_cbed, "会聚半角", "semiangle_mrad",
                                     unit="mrad")
        self.max_angle_var = entryed(tab_cbed, "最大显示角", "max_angle_mrad",
                                     unit="mrad")

        bar = ttk.Frame(tab_cbed)
        bar.pack(fill="x", padx=4, pady=4)
        self.cbed_btn = ttk.Button(bar, text="CBED 模拟", command=self._on_cbed)
        self.cbed_btn.pack(side="left")
        self.cbed_cancel_btn = ttk.Button(bar, text="取消模拟",
                                          command=self._cancel_cbed, state="disabled")
        self.cbed_cancel_btn.pack(side="left", padx=(6, 0))

        # ============================ 选项卡：导出设置 ============================
        tab_export = ttk.Frame(nb)
        nb.add(tab_export, text="导出设置")

        self.sclen_var = entryed(tab_export, "标尺长度", "scalebar_nm",
                                 default="5", unit="nm^-1")

        bar2 = ttk.Frame(tab_export)
        bar2.pack(fill="x", padx=4, pady=4)
        ttk.Button(bar2, text="导出图像", command=self._export_cbed_tif,
                   state="normal").pack(side="left")
        ttk.Button(bar2, text="导出原子坐标", command=self._export_atoms).pack(
            side="left", padx=4)
        ttk.Label(bar2, text="(电子沿 -Z 方向入射)", foreground="#888").pack(
            side="left", padx=(0, 4))

        # ============================ 选项卡：显示 ============================
        # 亮度/对比度通过仿射变换作用于归一化到 [0, M] 的对数强度数组：
        #   disp = clip(对比度*(log/M) + 亮度, 0, 1)，
        # 再以固定 clim=(0,1) 的 inferno 呈像。默认(亮度=0, 对比度=1)与
        # 改动前 auto-clim 的效果逐像素一致。
        tab_display = ttk.Frame(nb)
        nb.add(tab_display, text="显示")

        def display_scale_row(frame, var, label, frm, to, res, fmt, reset_val):
            r = ttk.Frame(frame)
            r.pack(fill="x", padx=4, pady=3)
            ttk.Label(r, text=label, width=14).pack(side="left")
            scl = ttk.Scale(r, from_=frm, to=to, variable=var,
                            orient="horizontal")
            scl.pack(side="left", fill="x", expand=True, padx=2)
            vtxt = tk.StringVar()

            def _on(*_):
                vtxt.set(fmt.format(var.get()))
                self._update_cbed_display()
            scl.configure(command=_on)
            var.trace("w", lambda *_: _on())
            _on()
            ttk.Label(r, textvariable=vtxt, width=6).pack(side="left")
            ttk.Button(r, text="重置", width=5,
                       command=lambda: var.set(reset_val)).pack(side="left", padx=3)
            return scl

        self.cbed_bright_var = tk.DoubleVar(value=0.0)
        self.cbed_contrast_var = tk.DoubleVar(value=1.0)
        self.cbed_gamma_var = tk.DoubleVar(value=1.0)
        display_scale_row(tab_display, self.cbed_bright_var, "亮度",
                          -1.0, 1.0, 0.01, "{:+.2f}", 0.0)
        display_scale_row(tab_display, self.cbed_contrast_var, "对比度",
                          0.0, 5.0, 0.05, "{:.2f}", 1.0)
        display_scale_row(tab_display, self.cbed_gamma_var, "伽马",
                          0.2, 4.0, 0.05, "{:.2f}", 1.0)
        ttk.Label(tab_display, text="(亮度=偏移, 对比度=增益, 伽马=幂次)",
                  foreground="#888").pack(anchor="w", padx=6, pady=(0, 4))

        # 状态与耗时显示（公共，置底）
        self.cbed_status = tk.StringVar(value="就绪")
        ttk.Label(f, textvariable=self.cbed_status, foreground="#888").pack(
            side="left", anchor="w", padx=6)
        self.cbed_time_var = tk.StringVar(value="")
        ttk.Label(f, textvariable=self.cbed_time_var, foreground="#3366cc").pack(
            side="bottom", anchor="w", padx=6, pady=(0, 3))

        # ---- 选项卡：STEM 原子像 ----
        self.stem_tab = ttk.Frame(self.right_nb)
        self.right_nb.add(self.stem_tab, text="STEM 原子像")
        self._build_stem_panel(self.stem_tab)

    # ------------------------------------------------------------------ STEM 原子像
    def _build_stem_panel(self, parent):
        """构建"STEM 原子像"选项卡：abTEM 所需参数 + 开始/取消按钮 + 画布 + 状态。"""
        f = ttk.LabelFrame(parent, text="STEM 原子像控制")
        f.pack(fill="x", padx=4, pady=4)

        def field(frame, label, key, defv, unit="", width=6):
            r = ttk.Frame(frame)
            r.pack(fill="x", padx=4, pady=2)
            ttk.Label(r, text=label, width=13).pack(side="left")
            var = tk.StringVar(value=str(defv))
            setattr(self, key, var)
            ttk.Entry(r, textvariable=var, width=width).pack(side="left", padx=2)
            if unit:
                ttk.Label(r, text=unit, foreground="#888").pack(side="left", padx=2)

        # 第 1 组：能量 / 厚度
        row = ttk.Frame(f); row.pack(fill="x")
        field(row, "加速电压", "stem_energy_var", self.cfg.get("energy_kev", 200.0), "keV")
        field(row, "厚度", "stem_thick_var", self.cfg.get("thickness_nm", 20.0), "nm")
        # 第 2 组：横向边长 / 采样
        row = ttk.Frame(f); row.pack(fill="x")
        field(row, "横向边长", "stem_width_var", self.cfg.get("box_width_nm", 10.8), "nm")
        field(row, "采样", "stem_sampling_var", 0.1, "Å/px")
        # 第 3 组：会聚半角 / 离焦 (C1)
        row = ttk.Frame(f); row.pack(fill="x")
        field(row, "会聚半角", "stem_semi_var", 25.0, "mrad")
        field(row, "离焦 (C1)", "stem_defocus_var", 0.0, "Å")
        # 第 4 组：环形探测器内/外角
        row = ttk.Frame(f); row.pack(fill="x")
        field(row, "探测器内角", "stem_inner_var", 60.0, "mrad")
        field(row, "探测器外角", "stem_outer_var", 200.0, "mrad")
        # 第 5 组：扫描点数（每边）
        row = ttk.Frame(f); row.pack(fill="x")
        field(row, "扫描点数/边", "stem_gpts_var", 128, "")

        bar = ttk.Frame(f); bar.pack(fill="x", padx=4, pady=4)
        self.stem_btn = ttk.Button(bar, text="STEM 模拟", command=self._on_stem)
        self.stem_btn.pack(side="left")
        self.stem_cancel_btn = ttk.Button(bar, text="取消模拟",
                                          command=self._cancel_stem,
                                          state="disabled")
        self.stem_cancel_btn.pack(side="left", padx=(6, 0))

        # 底部独立画布显示 STEM 结果（HAADF 原子像）
        self.fig_stem = Figure(figsize=(4.6, 4.9), dpi=100)
        self.ax_stem = self.fig_stem.add_subplot(111)
        self.canvas_stem = FigureCanvasTkAgg(self.fig_stem, master=parent)
        self.canvas_stem.get_tk_widget().pack(fill="both", expand=True)

        self.stem_status = tk.StringVar(value="就绪")
        ttk.Label(parent, textvariable=self.stem_status, foreground="#888").pack(
            anchor="w", padx=6)
        self.stem_time_var = tk.StringVar(value="")
        ttk.Label(parent, textvariable=self.stem_time_var, foreground="#3366cc").pack(
            anchor="w", padx=6, pady=(0, 3))

    def _on_stem(self):
        """读取 STEM 参数并以当前取向 self.R 启动后台多片法 STEM 模拟。"""
        if self.atoms is None:
            messagebox.showinfo("提示", "请先导入 CIF 文件")
            return
        if self._stem_running:
            return

        def rd(var, name):
            v = self._read_float(var, name)
            if v is None:
                raise ValueError(name)
            return v
        try:
            params = {}
            params["energy_kev"] = rd(self.stem_energy_var, "加速电压")
            params["thickness_nm"] = max(0.0, rd(self.stem_thick_var, "厚度"))
            params["width_nm"] = max(0.1, rd(self.stem_width_var, "横向边长"))
            params["sampling"] = max(0.01, rd(self.stem_sampling_var, "采样"))
            params["semiangle_mrad"] = max(0.01, rd(self.stem_semi_var, "会聚半角"))
            params["defocus_ang"] = rd(self.stem_defocus_var, "离焦")
            params["inner_mrad"] = max(0.0, rd(self.stem_inner_var, "探测器内角"))
            params["outer_mrad"] = max(0.0, rd(self.stem_outer_var, "探测器外角"))
            params["gpts"] = int(max(8, rd(self.stem_gpts_var, "扫描点数")))
        except ValueError:
            return

        if params["thickness_nm"] <= 0:
            messagebox.showinfo("提示", "STEM 原子像需要正样品厚度 (>0)")
            return
        if params["inner_mrad"] <= params["semiangle_mrad"]:
            self.stem_status.set("提示:内角未大于会聚半角,HAADF 可能无信号")
        if params["outer_mrad"] <= params["inner_mrad"]:
            self.stem_status.set("提示:外角需大于内角")

        # 快照当前取向，保证与左面板后续拖拽无关。
        params["_snap_R"] = np.array(self.R, float)

        self._stem_running = True
        self._stem_cancel.clear()
        self.stem_btn.config(state="disabled")
        self.stem_cancel_btn.config(state="normal")
        self.stem_status.set("计算中…（大超胞/大扫描网格可能需数分钟）")
        self.stem_time_var.set("")
        self._stem_start = time.time()

        t = threading.Thread(target=self._stem_worker, args=(params,), daemon=True)
        t.start()
        self.root.after(150, self._poll_stem_result)

    def _cancel_stem(self):
        """请求取消本次 STEM 模拟。worker 在关键步骤检查标志后尽快返回。"""
        self._stem_cancel.set()
        self.stem_btn.config(state="disabled")
        self.stem_cancel_btn.config(state="disabled")
        self.stem_status.set("正在取消…")

    def _stem_worker(self, params):
        """后台多片法 STEM（HAADF）：构造束框原子→翻转→势→Probe。

        扫描必须**手动按行分批**（lazy=False 每批），否则 abTEM 一次性物化
        "全部扫描位置 × 势网格" 的探针波（如 128×128×1080×1080 complex64）
        → 内存爆炸（实测 Ti 001 默认参数下报 142 GiB 分配失败）。abTEM 的
        lazy 分块路径在此版本有 bug（"too many indices"），故用手动分批。
        abTEM scan 数组轴序为 (x, y)（已验证与一次性结果逐点一致，diff=0）。
        """
        try:
            import abtem
            R = params["_snap_R"]
            wx = params["width_nm"] * 10.0     # nm -> Å
            wy = wx
            t = params["thickness_nm"] * 10.0

            atoms = self._build_box_fill(R, wx, wy, t)
            atoms = self._flip_to_plus_z(atoms)     # 转第二类模型（束 +z）
            if self._stem_cancel.is_set():
                return

            potential = abtem.Potential(atoms, sampling=params["sampling"])
            probe = abtem.Probe(energy=params["energy_kev"] * 1e3,
                                semiangle_cutoff=params["semiangle_mrad"],
                                aberrations={"defocus": params["defocus_ang"]},
                                sampling=params["sampling"])

            # 探测器外角不得超出"势/探针网格"能表示的最大衍射角（奈奎斯特限），
            # 否则 abTEM 的 radial 积分直接抛 RuntimeError。此处按采样换算自动截断。
            #   格点最大倒易矢 k_max = 1/(2·sampling)；散射半角 sinθ = λ·k_max/2
            lam = self._wavelength  # Å
            th_max = np.degrees(np.arcsin(lam / (2.0 * 2.0 * params["sampling"]))) * 1000.0
            inner = params["inner_mrad"]
            outer = params["outer_mrad"]
            outer = max(inner + params["sampling"], min(outer, th_max * 0.98))

            det = abtem.AnnularDetector(inner=inner, outer=outer)

            g = params["gpts"]
            gx = gy = g
            dy = wy / gy
            row_chunk = int(self.cfg.get("stem_row_chunk", 8))
            out = np.zeros((gx, gy), dtype=np.float32)

            for k in range(0, gy, row_chunk):
                if self._stem_cancel.is_set():
                    return
                n = min(row_chunk, gy - k)
                sub = abtem.GridScan(start=(0.0, dy * k),
                                     end=(wx, dy * (k + n)),
                                     gpts=(gx, n))
                arr = np.asarray(probe.scan(potential, scan=sub,
                                            detectors=[det],
                                            lazy=False).array)
                arr = np.squeeze(arr).astype(np.float32)
                if arr.ndim == 2 and arr.shape != (gx, n):
                    arr = arr.T          # 防御：确保轴序为 (x, y)
                out[:, k:k+n] = arr

            if self._stem_cancel.is_set():
                return
            self._stem_result_q.put(("ok", out, params))
        except Exception as e:
            import traceback
            self._stem_result_q.put(("err", traceback.format_exc()))

    def _poll_stem_result(self):
        try:
            msg = self._stem_result_q.get_nowait()
        except queue.Empty:
            if self._stem_running:
                self.root.after(150, self._poll_stem_result)
            return
        self._stem_running = False
        self.stem_btn.config(state="normal")
        self.stem_cancel_btn.config(state="disabled")
        if self._stem_cancel.is_set():
            self._show_stem_elapsed()
            self.stem_status.set("已取消")
            return
        if msg[0] == "err":
            self.stem_status.set("失败")
            self._show_stem_elapsed()
            messagebox.showerror("STEM 计算失败", msg[1])
            return
        _, arr, params = msg
        self._draw_stem(arr, params)
        self.stem_status.set("完成")
        self._show_stem_elapsed()

    def _draw_stem(self, arr, params):
        """绘制 HAADF-STEM 原子像（明原子亮），并给颜色条与标题。"""
        ax = self.ax_stem
        ax.clear()
        d = np.asarray(arr, float)
        d[np.isnan(d)] = 0.0
        vmax = float(np.nanmax(d)) if d.size else 1.0
        vmin = float(np.nanmin(d)) if d.size else 0.0
        if vmax <= vmin:
            vmax = vmin + 1.0
        im = ax.imshow(d, cmap="gray", origin="lower", aspect="equal",
                       vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_title(f"HAADF-STEM  t={params['thickness_nm']:g} nm, "
                     f"{params['inner_mrad']:g}-{params['outer_mrad']:g} mrad")
        cb = self.fig_stem.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label("强度 (arb. u.)")
        self.canvas_stem.draw_idle()

    def _show_stem_elapsed(self):
        if getattr(self, "_stem_start", None) is None:
            return
        dt = time.time() - self._stem_start
        self.stem_time_var.set(f"本次模拟耗时: {dt:.1f} s")

    # ------------------------------------------------------------------ 鼠标拖拽
    def _bind_left_mouse(self):
        self.canvas_left.mpl_connect("button_press_event", self._on_press)
        self.canvas_left.mpl_connect("button_release_event", self._on_release)
        self.canvas_left.mpl_connect("motion_notify_event", self._on_motion)

    def _on_press(self, event):
        if event.button == 1 and event.inaxes is self.ax_left:
            self._drag = (event.x, event.y)
            self._press_pt = (event.x, event.y)

    def _on_release(self, event):
        # 判为“点击”（移动很小且落在轴内）时显示对应衍射斑的物理信息，
        # 否则是拖拽旋转，不弹窗。
        if (self._drag is not None and self._press_pt is not None
                and event.button == 1):
            px, py = self._press_pt
            if max(abs(event.x - px), abs(event.y - py)) <= 4:
                try:
                    self._show_spot_info(event.xdata, event.ydata)
                except Exception as e:
                    print("spot info error:", e)
        self._drag = None
        self._press_pt = None

    def _show_spot_info(self, xdata, ydata):
        """点击某衍射斑时，以气泡标注显示其晶面间距、结构因子模、强度与
        预估消光距离；点击无斑空白处则移除气泡。"""
        # 先清除已有气泡（点击空白处也走这里达到“消失”）
        self._clear_spot_bubble()
        spots = getattr(self, "_spot_data", None)
        if not spots:
            return
        if xdata is None or ydata is None:
            return
        # 找最近的点
        best, bestd = None, np.inf
        for s in spots:
            d = (s["x"] - xdata) ** 2 + (s["y"] - ydata) ** 2
            if d < bestd:
                best, bestd = s, d
        if bestd > (0.08 * getattr(self, "_rmax", 1.0)) ** 2:
            self.cbed_status.set("点击邻近空白，已清除斑点信息")
            return
        hkl = "".join(str(int(v)) for v in best["hkl"])
        g = best["g"]
        dsp = 1.0 / g if g > 1e-9 else float("inf")    # 晶面间距 Å
        F = best["F"]
        I = best["I"]
        # 预估消光距离 ξ_g = π·V_c/(λ·|F_g|)（两束近似，Å）
        Vc = abs(np.linalg.det(self.cell)) if self.cell is not None else None
        lam_wl = getattr(self, "_wavelength", None)
        if F and F > 1e-12 and Vc and lam_wl:
            xi = np.pi * Vc / (lam_wl * F)             # Å
        else:
            xi = None
        dspn = dsp * 0.1 if np.isfinite(dsp) else float("inf")   # 转 nm
        lines = [
            f"斑点 ({hkl})",
            f"晶面间距 d = {dsp:.4f} Å" if np.isfinite(dspn)
            else "晶面间距 d = ∞",
            f"结构因子 |F| = {F:.3f}" if F is not None else "结构因子 |F| = N/A",
        ]
        if xi is not None:
            lines.append(f"预估消光距离 ξ = {xi*0.1:.3f} nm")
        else:
            lines.append("预估消光距离 = N/A")
        lines.append(f"强度(相对) I = {I:.4g}")
        txt = "\n".join(lines)

        # 气泡标注：锚点在斑点正上方，带方框与箭头；离开一定偏移避免遮住斑
        ax = self.ax_left
        off = 0.06 * getattr(self, "_rmax", 1.0)
        self._spot_bubble = ax.annotate(
            txt, xy=(best["x"], best["y"]), xytext=(best["x"], best["y"] + off),
            fontsize=7, ha="left", va="bottom", color="#111",
            bbox=dict(boxstyle="round,pad=0.35", fc="#fffbe8",
                      ec="#c8a84e", lw=0.9),
            arrowprops=dict(arrowstyle="->", color="#c8a84e", lw=0.9),
            zorder=50)
        self.canvas_left.draw_idle()
        self.cbed_status.set(f"斑点 ({hkl})：d={dspn:.4f} nm, |F|={F}")

    def _clear_spot_bubble(self):
        """移除当前衍射斑气泡（若存在）。"""
        bubble = getattr(self, "_spot_bubble", None)
        if bubble is not None:
            try:
                if hasattr(bubble, "remove"):
                    bubble.remove()
                elif hasattr(bubble, "set_visible"):
                    bubble.set_visible(False)
            except Exception:
                pass
        self._spot_bubble = None
        # 若已有画布，触发重绘（让被移除的标注消失）
        getattr(self, "canvas_left", None) and self.canvas_left.draw_idle()

    def _sync_zone_readout(self):
        """把当前带轴（束方向在晶体中）换算成 [u v w] 并同步到界面读数，保留一位小数。"""
        if self.atoms is None or self.cell is None:
            return
        A = np.asarray(self.cell, float).T          # miller -> Cartesian
        n = self.R.T @ Z_AXIS   # 实验室 z → 晶体方向 = R.T@Z（带轴沿束的方向）
        uvw = np.linalg.lstsq(A, n, rcond=None)[0]
        m = max(np.abs(uvw))
        if m < 1e-12:
            return
        uvw = uvw / m                               # 归一，令最大分量为 ±1
        self.zone_label.set(
            f"当前带轴: [{uvw[0]:.1f} {uvw[1]:.1f} {uvw[2]:.1f}]")
        for i, val in enumerate(uvw):
            self.uvw_vars[i].set(f"{val:.1f}")

    def _on_motion(self, event):
        if self._drag is None or event.x is None or event.y is None:
            return
        x0, y0 = self._drag
        dx, dy = event.x - x0, event.y - y0
        self._drag = (event.x, event.y)
        # 灵敏度：弧度/像素。调小使拖拽更平滑、单次旋转幅度更小。
        s = 0.0005
        # 束沿 -z 入射：Ewald 球心在倒易原点 +z 侧 (0,0,+k)。
        # 新约定 self.R 左乘列向量 = 晶体→实验室。拖拽旋转作用在**固定实验室/屏幕
        # 坐标轴**（真实载物台 gnometer 式）：晶体取向被主动旋转 LEFT 乘，
        # F = Rx(-dy·s) @ Ry(dx·s)，累加为 self.R_new = F @ self.R（而非旧的右乘
        # self.R @ F 的晶体系旋转——在面内 z 旋转非零时晶体系轴已被扭转，
        # 右乘会令屏幕上方向错误，见 _rotcheck.py 数值验证）。
        # 拖拽 = 绕实验室固定轴主动旋转晶体（束不动、晶体动，真实测角台式）：
        #   · 向下拖（dy<0）：Rx(-dy·s) 绕 x 正向，上方 +y 侧（020）强激发；
        #   · 向左拖（dx<0）：Ry(dx·s) 绕 y 负向，右方 +x 侧（200）强激发。
        self.R = (Rx(-dy * s) @ Ry(dx * s)) @ self.R
        self._sync_zone_readout()
        # 节流：合并高频 motion 事件，约 60 fps 重绘一次，
        # 避免 diffsims 逐事件重算堆积导致卡顿、画面"一格一格"跳变。
        if not self._redraw_pending:
            self._redraw_pending = True
            self.root.after(16, self._do_redraw)

    def _do_redraw(self):
        self._redraw_pending = False
        if self.atoms is not None:
            self._redraw_left()

    # ------------------------------------------------------------------ 行为
    def _load_cif(self, path):
        if not path:
            return
        self._cif_path = path
        try:
            with open(path, encoding="utf-8", errors="replace") as f:
                cif_text = f.read()
        except Exception as e:
            messagebox.showerror("读取 CIF 失败", str(e))
            return
        self._load_cif_text(cif_text, os.path.basename(path))

    def _load_cif_text(self, cif_text, name):
        """从 CIF 文本加载晶体（路径或内嵌常量皆可），并重建衍射/菊池量。"""
        import io
        from ase.io import read
        try:
            atoms = read(io.BytesIO(cif_text.encode("utf-8")), format="cif")
            if len(atoms) == 1:
                # 单等效原子 CIF（如 bcc Nb/α-Fe 只有 (0,0,0) 位）：补面心位置 (½,½,½) 成双原子基元
                from ase.atom import Atom
                frac0 = np.asarray(atoms.get_scaled_positions()[0])
                atoms.append(Atom(atoms[0].symbol,
                                  scaled_position=frac0 + np.array([0.5, 0.5, 0.5])))
                atoms.wrap()
        except Exception as e:
            messagebox.showerror("读取 CIF 失败", str(e))
            return

        align_c_to_z(atoms)
        self.atoms = atoms
        self.cell = np.asarray(atoms.get_cell()[:], float)
        self.R = np.eye(3)
        self._reset_tem_tilt_fields()
        self.g_list = build_g_list(self.cell, self.cfg["gmax"], self.cfg["max_hkl"])
        self.F2 = {hkl: structure_factor(hkl, g, atoms) ** 2
                   for hkl, g in self.g_list}
        fmax = max(self.F2.values()) if self.F2 else 1.0
        self.F2 = {hkl: v / fmax for hkl, v in self.F2.items()}
        self._build_phase(cif_text)
        self.cif_label.set(name + f"  (a≈{np.linalg.norm(self.cell[0]):.3f} Å)")
        fml, sg = self._read_formula_and_sg_symbol(cif_text)
        if fml:
            fml = fml.replace("_", "₂").replace("'", "")
        if sg:
            sg = sg.replace("'", "").strip()
        info = ", ".join(x for x in (fml, sg) if x)
        self.crystal_info_label.set(info)
        self._redraw_left()

    def _read_spacegroup_number(self, txt):
        """从 CIF 文本里读空间群编号，读不到返回 1（P1，退回按原子求和）。"""
        import re
        for key in ("_symmetry_Int_Tables_number", "_space_group_IT_number"):
            m = re.search(key + r"\s+(\d+)", txt)
            if m:
                return int(m.group(1))
        return 1

    def _read_formula_and_sg_symbol(self, txt):
        """从 CIF 文本读化学式(优先 _chemical_formula_structural)与空间群符号(H-M)。
        任一项缺失则返回空串。"""
        import re
        formula = ""
        for key in ("_chemical_formula_structural",
                    "_chemical_formula_sum",
                    "_chemical_formula_moiety"):
            m = re.search(key + r"\s+[\"']?([^\"'\n]+)[\"']?", txt)
            if m:
                formula = m.group(1).strip()
                if formula:
                    break
        sg = ""
        for key in ("_space_group_name_H-M_alt",
                    "_symmetry_space_group_name_H-M",
                    "_space_group_name_H-M"):
            m = re.search(key + r"\s+[\"']?([^\"'\n]+)[\"']?", txt)
            if m:
                sg = m.group(1).strip()
                if sg:
                    break
        return formula, sg

    def _build_phase(self, cif_text):
        """用 diffpy/orix 从对齐后的晶胞构建 Phase，并新建 SimulationGenerator。"""
        try:
            from diffpy.structure import Atom, Lattice, Structure
            from orix.crystal_map import Phase
            from diffsims.generators.simulation_generator import SimulationGenerator
        except Exception as e:
            self.phase = None
            self.gen = None
            messagebox.showwarning(
                "缺少衍射库", "需安装 diffsims / orix / diffpy.structure：\n" + str(e))
            return

        cell = np.asarray(self.atoms.get_cell()[:], float)
        fracs = self.atoms.get_scaled_positions()
        symbols = self.atoms.get_chemical_symbols()

        lat = Lattice(base=cell)
        structure = Structure(lattice=lat)
        for sym, fr in zip(symbols, fracs):
            structure.append(Atom(sym, fr, lattice=lat))

        sg = self._read_spacegroup_number(cif_text)
        self.phase = Phase("crystal", space_group=sg, structure=structure)
        self.gen = SimulationGenerator(accelerating_voltage=self.cfg["energy_kev"],
                                       scattering_params="xtables")

    def _on_import_cif(self):
        path = filedialog.askopenfilename(
            title="选择 CIF 文件",
            filetypes=[("CIF 文件", "*.cif"), ("所有文件", "*.*")])
        if path:
            self._load_cif(path)

    def _read_miller_fields(self, vars_):
        """逐字段解析 Miller 指数：空串或非法值一律按 0 处理。"""
        out = []
        for v in vars_:
            try:
                out.append(float(v.get()))
            except (ValueError, AttributeError):
                out.append(0.0)
        return tuple(out)

    def _on_set_zone(self):
        uvw_f = self._read_miller_fields(self.uvw_vars)
        if np.all(np.abs(uvw_f) < 1e-9):
            messagebox.showwarning("输入错误", "带轴不能为 [0 0 0]")
            return
        # 允许非整数输入：按浮点方向取最近的整数带轴（整数化），再回写显示。
        v_f = miller_to_cartesian(uvw_f, self.cell)
        uvw = snap_zone_axis(v_f, self.cell, self.cfg.get("zone_max_index", 12000))
        if uvw == (0, 0, 0):
            messagebox.showwarning("输入错误", "带轴不能为 [0 0 0]")
            return
        for i, val in enumerate(uvw):
            self.uvw_vars[i].set(str(val))
        v = miller_to_cartesian(uvw, self.cell)      # 正空间带轴方向
        nv = np.linalg.norm(v)
        if nv < 1e-12:
            return
        v = v / nv

        # 设带轴后倾转输入归零、基准清空：下次点电镜倾转时以新显示取向起算
        self._reset_tem_tilt_fields()

        # 晶面指数 hkl：决定带轴取正后，哪个倒易方向 g 对准实验室 +x。
        hkl_base = tuple(0.0 for _ in self.hkl_vars)
        if getattr(self, "hkl_vars", None) is not None:
            hkl_base = self._read_miller_fields(self.hkl_vars)
        if np.all(np.abs(hkl_base) < 1e-9):
            # 无有效 hkl：退化为只对齐带轴（不指定面内方向，同旧行为）。
            # 用 rotation_align(v, Z)（v→+z）而非 rotation_align(Z, v)，
            # 以便带轴读回 R.T@Z = v（R 沿束方向，见 _sync_zone_readout）。
            self.R = rotation_align(v, Z_AXIS)
            self._redraw_left()
            return

        # 若不满足晶带定律 uvw·hkl≠0，把 hkl 规范为最接近且垂直于 uvw 的整数解
        hkl = closest_zone_plane(hkl_base, uvw,
                                 self.cfg.get("zone_max_index", 12000))
        if hkl is None or np.all(np.asarray(hkl) == 0):
            messagebox.showwarning(
                "输入错误",
                f"找不到垂直于带轴 [{uvw[0]} {uvw[1]} {uvw[2]}] 的晶面指数")
            return
        for i, val in enumerate(hkl):
            self.hkl_vars[i].set(str(val))

        # 晶面法向的倒易矢 g = h·b1 + k·b2 + l·b3（将对准 +x 的方向）
        g = np.asarray(hkl, float) @ reciprocal_basis(self.cell)
        ng = np.linalg.norm(g)
        if ng < 1e-12:                 # 防御：hkl 有效时不应出现
            self.R = rotation_align(v, Z_AXIS)
            self._redraw_left()
            return
        g = g / ng
        # 重新正交化（float 容差兜底），组成右手正交基 [g, v×g, v]；
        # R = S.T 满足 R@g=+x、R@v=+z、R@(v×g)=+y（proper，det=+1），
        # 且带轴读回 R.T@Z = v（R 沿束方向，见 _sync_zone_readout）。
        g = g - (g @ v) * v
        g = g / (np.linalg.norm(g) + 1e-30)
        t = np.cross(v, g)
        S = np.column_stack([g, t, v])            # 列 = [êx' êy' êz']
        self.R = S.T                               # 正交阵转置=逆：晶体→实验室
        self._redraw_left()

    def _on_inplane(self):
        try:
            deg = float(self.inplane_var.get())
        except ValueError:
            return
        # 面内旋转 = 绕电子束(实验室 z)旋转，作用在固定实验室轴（左乘），
        # 与拖拽/操控盘一致，避免晶体系右乘在面内角下的方向偏差
        self.R = Rz(np.deg2rad(-deg)) @ self.R
        self._redraw_left()

    def _on_reset(self):
        self.R = np.eye(3)
        self._reset_tem_tilt_fields()
        self.uvw_vars[0].set("0")
        self.uvw_vars[1].set("0")
        self.uvw_vars[2].set("1")
        for i, h in enumerate((1, 0, 0)):
            self.hkl_vars[i].set(str(h))
        self.inplane_var.set("0.0")
        self._redraw_left()

    # ------------------------------------------------------------- 电镜倾转
    def _read_tem_tilt_angle(self, var, name):
        try:
            return float(var.get())
        except (ValueError, AttributeError):
            messagebox.showwarning("输入错误", f"{name} 须为数字 (°)")
            return None

    def _tem_tilt_rot(self, alpha_deg, beta_deg):
        """双倾样品杆的倾转旋转：R_tilt = Rα(α) @ Rβ(β)（先 β 后 α）。

        先绕 β 轴转（此时 α 未动，β 轴处于其真实方位），再绕 α 轴转（β 轴
        随样品杆一起转过 α）——与样品杆的实际运动次序一致。两轴均取
        实验室/绘图区固定方向（左乘列向量）。
        """
        d_a = np.asarray(TEM_TILT_AXES[self.tem_tilt_alpha_axis_var.get()], float)
        d_b = np.asarray(TEM_TILT_AXES[self.tem_tilt_beta_axis_var.get()], float)
        R_beta = rotation_about_axis(d_b, np.radians(beta_deg))
        R_alpha = rotation_about_axis(d_a, np.radians(alpha_deg))
        return R_alpha @ R_beta

    def _on_set_current_tem_tilt(self):
        """记录“当前花样对应的 α/β 倾转读数”——只记录，不旋转、不重绘。

        把当前显示的取向解释成电镜 0 状态先 β 后 α 倾转后的状态，由此反推
        0 状态：R_0 = [Rα(α) @ Rβ(β)]ᵀ @ R，供目标倾转面板起算。
        """
        a = self._read_tem_tilt_angle(self.tem_tilt_alpha_var, "α 倾转角")
        b = self._read_tem_tilt_angle(self.tem_tilt_beta_var, "β 倾转角")
        if a is None or b is None:
            return
        if self.atoms is None:
            messagebox.showinfo("提示", "请先导入 CIF 文件")
            return
        self._tem_tilt_R0 = self._tem_tilt_rot(a, b).T @ self.R
        self.tem_tilt_info_var.set(f"当前倾转：α = {a:g}°，β = {b:g}°")

    def _on_set_target_tem_tilt(self):
        """按目标 α/β 改变下方花样（从电镜 0 状态起算）。

        未记录当前倾转时，把当前显示取向当作 0 状态（即 α = β = 0°）。
        """
        a = self._read_tem_tilt_angle(self.tem_tilt_target_alpha_var, "目标 α 倾转角")
        b = self._read_tem_tilt_angle(self.tem_tilt_target_beta_var, "目标 β 倾转角")
        if a is None or b is None:
            return
        if self.atoms is None:
            messagebox.showinfo("提示", "请先导入 CIF 文件")
            return
        R0 = self.R.copy() if self._tem_tilt_R0 is None else self._tem_tilt_R0
        self.R = self._tem_tilt_rot(a, b) @ R0
        self._sync_zone_readout()
        self._redraw_left()

    def _reset_tem_tilt_fields(self):
        """倾转基准改变（设带轴/重置/导入）后，倾转输入归零并清空记录。"""
        for var in (self.tem_tilt_alpha_var, self.tem_tilt_beta_var,
                    self.tem_tilt_target_alpha_var, self.tem_tilt_target_beta_var):
            var.set("0.0")
        self._tem_tilt_R0 = None
        self.tem_tilt_info_var.set("尚未记录当前倾转（按 α = β = 0° 处理）")

    # ------------------------------------------------------------- 旋转操控盘
    def _draw_dial(self):
        """重绘旋转操控盘：圆、刻度参考、指针。"""
        if self._rot_dial is None:
            return
        c = self._rot_dial
        c.delete("all")
        cx, cy, r = self._dial_c, self._dial_c, self._dial_r
        c.create_oval(cx - r, cy - r, cx + r, cy + r, outline="#999",
                      width=1.5)
        # 刻度参考线：+X 右（canvas 向右），+Y 上（canvas y 向下 → 向上为负 y）
        c.create_line(cx - r, cy, cx + r, cy, fill="#e0e0e0")
        c.create_line(cx, cy - r, cx, cy + r, fill="#e0e0e0")
        # 指针：屏幕方向（右=x, 上=y）即 3D 实验室 XY 平面方向
        px = cx + r * float(np.cos(self._rot_phi))
        py = cy - r * float(np.sin(self._rot_phi))
        c.create_line(cx, cy, px, py, fill="#d04040", width=2.5,
                      arrow="last", arrowshape=(8, 10, 4))
        c.create_oval(cx - 3, cy - 3, cx + 3, cy + 3, fill="#d04040",
                      outline="")

    def _on_dial_drag(self, event):
        """拖动/点按操控盘：更新指针方向并同步左图旋转方向箭头。"""
        if self._rot_dial is None:
            return
        dx = event.x - self._dial_c
        dy = event.y - self._dial_c
        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            return
        # canvas y 向下 → 用 -dy 使"上=+Y"；方向 = (cosφ, sinφ, 0)
        self._rot_phi = float(np.arctan2(-dy, dx))
        self._rot_dir_var.set(f"{np.degrees(self._rot_phi):.1f}")
        self._set_rot_arrow_visible(True)
        self._draw_dial()
        self._update_rot_arrow()

    def _on_rot_dir_entry(self, event=None):
        """由"指针角 (X轴, °)"输入框更新指针方向（回车/失焦生效）。"""
        try:
            deg = float(self._rot_dir_var.get())
        except ValueError:
            return
        self._rot_phi = float(np.radians(deg))
        self._draw_dial()
        self._update_rot_arrow()

    def _set_rot_arrow_visible(self, show):
        """切换下方图中旋转方向箭头的显示，并同步按钮文字。"""
        self._show_rot_arrow = bool(show)
        lbl = getattr(self, "_rot_arrow_lbl", None)
        if lbl is not None:
            lbl.set("隐藏箭头" if self._show_rot_arrow else "显示箭头")

    def _on_toggle_rot_arrow(self):
        """按钮：显示/隐藏下方图中的旋转方向箭头。"""
        self._show_rot_arrow = not self._show_rot_arrow
        self._set_rot_arrow_visible(self._show_rot_arrow)
        self._update_rot_arrow()

    def _rot_axis(self):
        """当前指针方向对应的 3D 旋转轴（XY 平面单位向量）。"""
        return np.array([float(np.cos(self._rot_phi)),
                         float(np.sin(self._rot_phi)), 0.0])

    def _on_rot_apply(self):
        """确认旋转：绕当前指针方向轴旋转所输角度（mrad）。"""
        try:
            mrad = float(self._rot_angle_var.get())
        except ValueError:
            messagebox.showwarning("输入错误", "旋转角须为数字 (mrad)")
            return
        if self.atoms is None:
            return
        # mrad → 弧度：1 mrad = 0.001 rad。勿再套 np.radians（那会把已是的 rad
        # 再乘 π/180≈0.01745，令实际转角小 57.3 倍）。
        ang = mrad / 1000.0
        # 绕实验室/屏幕 XY 平面内的方向 d 旋转（左乘 self.R，同拖拽：
        # 施加面内 z 旋转后右乘会令晶体系轴扭转、方向错误，故用左乘）
        self.R = rotation_about_axis(self._rot_axis(), ang) @ self.R
        self._set_rot_arrow_visible(True)
        self._redraw_left()

    def _update_rot_arrow(self):
        """在左图当前状态上原位更新"旋转方向"箭头（绕着图中心）。"""
        ax = getattr(self, "ax_left", None)
        canvas = getattr(self, "canvas_left", None)
        arrow = getattr(self, "_rot_arrow", None)
        if arrow is not None:
            try:
                arrow.remove()
            except Exception:
                pass
            self._rot_arrow = None
        if not self._show_rot_arrow or getattr(self, "atoms", None) is None \
                or ax is None:
            if canvas is not None:
                canvas.draw_idle()
            return
        rmax = getattr(self, "_rmax", 1.0) or 1.0
        L = rmax   # 旋转方向箭头拉长到整个参考圆半径
        dx = L * float(np.cos(self._rot_phi))
        dy = L * float(np.sin(self._rot_phi))
        self._rot_arrow = ax.annotate(
            "", xy=(dx, dy), xytext=(0.0, 0.0),
            arrowprops=dict(arrowstyle="-|>", color="#d04040", lw=2.0,
                            shrinkA=0, shrinkB=0),
            zorder=100)
        if canvas is not None:
            canvas.draw_idle()

    def _tilt_mrad(self):
        """当前带轴在实验室中相对 +z 的倾角 (mrad)，仅作显示参考（不再用于 CBED，
        CBED 改用旋转原子、束固定 +z，见 _cbed_worker）。"""
        u = self.R.T @ Z_AXIS   # 实验室 z → 晶体方向 = R.T@Z（带轴沿束的方向）
        u /= np.linalg.norm(u)
        # 弧度 → mrad：直接 ×1000。切勿用 np.degrees()*1000（那是毫度，比 mrad 大 57.3 倍）。
        ax = np.arctan2(u[0], u[2]) * 1000.0
        ay = np.arctan2(u[1], u[2]) * 1000.0
        return ax, ay

    def _on_aperture(self):
        """按用户输入的探测口径 rmax(mrad) 反推 gmax，重建倒易列表并重绘。"""
        try:
            apert = float(self.aperture_var.get())
        except ValueError:
            messagebox.showwarning("输入错误", "探测口径需为数字 (mrad)")
            self._sync_aperture_default()
            return
        if apert < 1e-6:
            messagebox.showwarning("输入错误", "探测口径需为正数 (mrad)")
            self._sync_aperture_default()
            return
        lam = 12.26 / np.sqrt(float(self.energy_var.get()) * 1e3 *
                              (1 + 9.78e-7 * float(self.energy_var.get()) * 1e3))
        # 口径 mrad -> 全散射角 rad = apert/1000；半角 = apert/2000 rad；g=2 sin(半角)/λ
        self.cfg["gmax"] = 2.0 * np.sin(apert / 2000.0) / lam
        if self.cell is not None:
            self.g_list = build_g_list(self.cell, self.cfg["gmax"],
                                       self.cfg["max_hkl"])
            self.F2 = {hkl: structure_factor(hkl, g, self.atoms) ** 2
                       for hkl, g in self.g_list}
            fmax = max(self.F2.values()) if self.F2 else 1.0
            self.F2 = {hkl: val / fmax for hkl, val in self.F2.items()}
        self._redraw_left()

    def _sync_aperture_default(self):
        """把输入框恢复为当前 cfg gmax 对应的 mrad 口径。"""
        try:
            lam = 12.26 / np.sqrt(float(self.energy_var.get()) * 1e3 *
                                  (1 + 9.78e-7 * float(self.energy_var.get()) * 1e3))
            apert = 1000.0 * 2.0 * np.arcsin(np.clip(
                0.5 * lam * self.cfg["gmax"], -1.0, 1.0))
            self.aperture_var.set(f"{apert:.2f}")
        except Exception:
            self.aperture_var.set("30.00")

    def _on_bandlim(self):
        """按用户输入更新菊池带绘制限 |g|max (1/Å)，并重绘。"""
        try:
            lb = float(self.bandlim_var.get())
        except ValueError:
            messagebox.showwarning("输入错误", "菊池带限需为数字 (1/Å)")
            return
        if lb < 1e-6:
            messagebox.showwarning("输入错误", "菊池带限需为正数 (1/Å)")
            return
        self.cfg["band_max"] = lb
        self._redraw_left()

    def _redraw_left(self):
        if self.atoms is None:
            return
        ax = self.ax_left
        ax.clear()
        band_max = self.cfg.get("band_max", 0.85)
        # 探测口径 mrad：优先读用户输入，无效则按 cfg gmax 推算。
        apert_msr = float("nan")
        try:
            if getattr(self, "aperture_var", None) is not None:
                ap = float(self.aperture_var.get())
                if ap > 1e-6:
                    apert_msr = ap
        except ValueError:
            pass

        # 同步加速电压 -> 波长（Å）
        try:
            energy = float(self.energy_var.get())
        except (ValueError, AttributeError):
            energy = self.cfg["energy_kev"]
        if self.gen is not None:
            self.gen.accelerating_voltage = energy
            wavelength = self.gen.wavelength
        else:
            wavelength = 12.26 / np.sqrt(energy * 1e3 * (1 + 9.78e-7 * energy * 1e3))
        self._wavelength = float(wavelength)

        # 探测口径 mrad → 真实 gmax = 2·sin(半角)/λ，真实 rmax = cam_len·tan(2θ_max)。
        # 参考圆半径 → cam_len 做显示缩放，rmax 显示即散射角全角 2θ_max 等于用户输入 mrad。
        if not np.isfinite(apert_msr):
            # 回退：从 cfg gmax 算 mrad
            apert_msr = 1000.0 * 2.0 * np.arcsin(np.clip(
                0.5 * wavelength * self.cfg["gmax"], -1.0, 1.0))
            self._sync_aperture_default()
        gmax = 2.0 * np.sin(apert_msr / 2000.0) / wavelength
        theta_max = apert_msr / 2000.0   # 半角 rad
        rmax = self.cfg.get("cam_len", 30.56) * np.tan(2.0 * theta_max)
        self._rmax = float(rmax)

        # gnomonic 投影比例：方向 (dx,dy,dz) -> 屏幕 (f*dx/dz, f*dy/dz)。
        # 用于菊池带：Kossel 锥（小圆 n·r̂=±sinθ_B）在小视场下线性化为直线。
        f = rmax / (wavelength * gmax)

        # ---- 菊池带：每条带 = 面法线 n 对应的一对 Kikuchi 边。
        # 边是 Kossel 锥的 gnomonic 投影，小视场下退化为直线（束沿 -z，k̂=-ẑ）
        #   nx·X + ny·Y + f·(-nz ∓ sinθ_B) = 0
        # 只有 n_z≈0（面法线⊥束）时带边才过中心；其余交于不同菊池极，倾转后迅速移出视场。
        if self.show_bands.get():
            seen = set()
            for hkl, g in self.g_list:
                F2 = self.F2.get(hkl, 0.0)
                if F2 < 0.01:
                    continue
                gn = float(np.linalg.norm(g))
                if gn <= 1e-9 or gn > band_max:
                    continue
                key = _friedel_key(hkl)
                if key in seen:
                    continue
                seen.add(key)

                n_lab = self.R @ (g / gn)   # 面法线（倒易方向）在实验室系：新约定 self.R 直接左乘列向量 = 晶体→实验室
                nx, ny, nz = n_lab[0], n_lab[1], n_lab[2]
                rho = np.hypot(nx, ny)
                if rho < 1e-6:
                    continue   # 面法线 ∥ 束：带占满整个视场，略去

                sB = 0.5 * wavelength * gn   # sinθ_B = λ|g|/2
                alpha_band = 0.25 + 0.40 * min(1.0, F2)
                for sgn in (-1.0, +1.0):
                    # 束沿 -z（k̂=-ẑ）时 Kossel 锥条件 n·k̂=-nz=±sB，
                    # 带边 nx·X+ny·Y = f·(±sB + nz) ⇒ C = f·(-nz - sgn·sB)。
                    C = f * (-nz - sgn * sB)
                    ip = _circle_line(nx, ny, C, rmax)
                    if ip is None:
                        continue
                    (x1, y1), (x2, y2) = ip
                    ax.plot([x1, x2], [y1, y2], color="tab:blue",
                            lw=1.2, alpha=alpha_band)

        # ---- 衍射斑：运动学自算，不再依赖 diffsims/orix 的输出坐标。
        # 原因：orix 对非立方晶胞存在定向偏差（六方 {100} 被整体旋转 30°，
        # 使 (100) 反射从 +x 方向消失，且 hkl 标签错乱为 (0,0,0) 等）。
        # 位置：X=scale·(self.R.T·g)_x, Y=scale·(self.R.T·g)_y（g 为晶体倒易矢，
        # 新约定行形式 g_lab = G @ self.R.T 即列形式 self.R @ g）。
        # 强度：运动学 |F|² × Lorentzian 激发包络 × 亮度倍率，绝对定标到 f2ref。
        spots_x, spots_y, spots_I, spots_h = [], [], [], []
        self._spot_data = []
        if self.show_spots.get() and self.cell is not None and self.atoms is not None:
            gmax_ref = (self.cfg.get("spot_gmax_ref") or
                        DEFAULTS.get("spot_gmax_ref") or
                        DEFAULTS["gmax"])
            glist = build_g_list(self.cell, max(gmax, gmax_ref),
                                 self.cfg["max_hkl"])
            if glist:
                scale = rmax / gmax
                HKL = np.array([h for h, _ in glist], float)
                G = np.array([g for _, g in glist], float)   # 晶体倒易（行向量）
                g_lab = G @ self.R.T                          # 实验室倒易（新约定行形式 = self.R.T）
                X = scale * g_lab[:, 0]
                Y = scale * g_lab[:, 1]
                inside = (X * X + Y * Y) <= (rmax * rmax) * 1.02
                gmag = np.linalg.norm(G, axis=1)
                F2s = np.array([structure_factor(tuple(h), gi, self.atoms) ** 2
                                for h, gi in zip(HKL, gmag)])
                # 绝对 F² 基准：gmax_ref 内最强反射（与口径、带轴、视场无关）
                f2ref = 0.0
                for _h, _g in build_g_list(self.cell, gmax_ref,
                                           self.cfg["max_hkl"]):
                    fv = structure_factor(_h, _g, self.atoms) ** 2
                    if fv > f2ref:
                        f2ref = fv
                # 显示规则：仅当 F=0（空间群系统消光，如螺旋轴/滑移产生的规则缺反射）
                # 才不显示；其余一律保留。弱但允许的反射(如超结构(001))不再被 1% 强度
                # 阈值删掉，改用 Gaussian 包络 env 让离带轴者平滑淡出。
                nonzero = F2s > 1e-9
                # 激发包络：Gaussian env=exp(-(s/σ)²)，s 为 Ewald 球激发误差。
                # 束沿 -z 入射（k 矢量 = -k·ẑ）：s = |k+g| - |k|
                # = sqrt(gperp² + (k-gz)²) - k，即对 gz 的符号与 +z 约定相反。
                k = 1.0 / wavelength
                s = np.sqrt(g_lab[:, 0]**2 + g_lab[:, 1]**2 +
                            (k - g_lab[:, 2])**2) - k
                sigma = self.cfg.get("exc_width",
                                     0.5 * self.cfg.get("exc_spot", 0.10))
                env = np.exp(-((s / sigma) ** 2))
                # 激发误差硬门限：|s| ≥ exc_spot 的反射直接剔除（与 diffsims
                # max_excitation_error 行为一致）。必须用 |s|：s 为负（Ewald 球
                # 另一侧，如 [001] 带轴下 l>0 的反射 gz>0 在束 -z 约定下 s≈-gz）
                # 同样远离 Bragg 条件，若只判 s<exc_spot 则负值恒通过，
                # 导致大量弱斑涌入。
                # 否则 Gaussian 包络永不为 0，强反射但强离带轴的斑（如 [001]
                # 下投影到中心的 (002)，F² 大但 |s|≈0.43）会以弱强度显示成白色
                # 小点，叠在强斑/直射束上。
                keep = inside & nonzero & (np.abs(s) < self.cfg["exc_spot"])
                kept = np.where(keep)[0]
                X, Y, hkls = X[keep], Y[keep], HKL[keep]
                F2k = F2s[keep]
                envk = env[keep]
                # 亮度倍率（0~10）：整体调亮/调暗，不改变相对强度分布。
                brightness = float(self.bright_var.get())
                brightness = max(0.0, min(10.0, brightness))
                if brightness <= 0:
                    brightness = 1e-6   # 避免除零；允许调到极暗
                rel = F2k * envk * brightness   # 携带亮度倍率的物理相对强度
                # 绝对满标 = 参考最强反射 f2ref（不含亮度倍率，否则滑块无效）。
                mx = f2ref if f2ref > 0 else \
                    (float(np.max(F2k * envk)) if len(F2k) else 1.0)
                mx = float(mx) if mx > 0 else 1.0
                self._spot_mx = mx
                intens = rel

                # 记录点击所需的物理量（|F|、|g|、强度等）
                for j2, i in enumerate(kept):
                    self._spot_data.append({
                        "x": float(X[j2]), "y": float(Y[j2]),
                        "hkl": tuple(int(round(v)) for v in hkls[j2]),
                        "g": float(gmag[i]),
                        "F": float(np.sqrt(F2s[i])),
                        "I": float(intens[j2]),
                    })

            # 000 直接束：中心直射束。设为整图最强斑（黑），与灰度色标"强斑近黑"一致，
            # 不参与归一以免占满动态范围。
            if self.show_spots.get() and self.cell is not None:
                ax.scatter([0.0], [0.0], s=75, marker="o",
                           facecolors="black", edgecolors="none",
                           zorder=5)
            if len(X):
                diffuse = np.ones(len(X), bool)
                if np.any(diffuse):
                    # 普适绝对灰度：满标 = 参考球内最强反射强度（f2ref·亮度），
                    # 与探测口径、带轴、视场无关 → 调小口径/旋转时弱斑明暗稳定。
                    mx = getattr(self, "_spot_mx", 0.0)
                    if mx <= 0:
                        mx = float(np.max(intens[diffuse]))
                    mx = float(mx) if mx > 0 else 1.0
                    # 物理灰度斑：强斑近黑、弱斑浅灰，模拟乳胶/底片式衍射斑明暗，
                    # 而非纯色圆。用 GREY_PHYS（浅灰→黑）而非 Greys：Greys 低端
                    # 是纯白，弱斑在白背景上会显成刺眼白点；浅灰起点让弱斑柔和。
                    # 相对强度 ratio（含亮度倍率）超出满标时颜色饱和到最黑，
                    # 点大小封顶避免无限膨胀。线性比例：弱斑偏浅灰、强斑近黑。
                    ratio = intens[diffuse] / mx
                    ax.scatter(X[diffuse], Y[diffuse], c=ratio,
                               s=8 + 70 * np.minimum(ratio, 1.0),
                               cmap=GREY_PHYS, vmin=0, vmax=1.0,
                               edgecolors="none", alpha=0.92)
                    spots_x, spots_y, spots_I, spots_h = \
                        X[diffuse], Y[diffuse], intens[diffuse], hkls[diffuse]

            if self.show_index.get() and len(spots_x):
                mx = getattr(self, "_spot_mx", 0.0)
                if mx <= 0:
                    mx = float(np.max(spots_I)) if len(spots_I) else 1.0
                mx = float(mx) if mx > 0 else 1.0
                for x, y, I, h in zip(spots_x, spots_y, spots_I, spots_h):
                    if I < 0.12 * mx:
                        continue
                    lbl = "".join(str(int(round(v))) for v in h)
                    # 标签置于斑点正上方（上下各空一点间距），避免与斑重叠
                    dy = 0.06 * rmax
                    ax.text(x, y + dy, lbl, fontsize=6, color="#a33",
                            ha="center", va="bottom")

        # 参考圆与中心十字
        th = np.linspace(0, 2 * np.pi, 200)
        ax.plot(rmax * np.cos(th), rmax * np.sin(th), color="gray",
                lw=0.7, alpha=0.5)
        ax.axhline(0, color="gray", lw=0.5, alpha=0.3)
        ax.axvline(0, color="gray", lw=0.5, alpha=0.3)

        # 显示长度与倒易量换算常量，供下方坐标刻度使用
        scale_s = rmax / gmax

        # 带轴同步显示
        u = self.R.T @ Z_AXIS   # 实验室 z → 晶体方向 = R.T@Z（带轴沿束的方向）
        uvw = cartesian_to_miller(u, self.cell)
        tx, ty = self._tilt_mrad()
        self.zone_label.set(f"当前带轴: {' '.join(str(x) for x in uvw)}    "
                            f"束倾: ({tx:+.2f}, {ty:+.2f}) mrad")

        ax.set_xlim(-rmax, rmax)
        ax.set_ylim(-rmax, rmax)
        ax.set_aspect("equal")
        ax.set_title("衍射花样 + 菊池带（运动学 · 拖动左键旋转）")

        # ---- 坐标刻度：仿照右侧 CBED——主刻度 mrad、副刻度(顶/右) nm^-1。
        # 显示长度 X=scale_s·g(Å⁻¹)，g≈2θ/λ ⇒ mrad=1000·λ·X/scale_s；
        # nm^-1 = 10·Å⁻¹ ⇒ 显示长度 → nm^-1 换算为 10·X/scale_s。
        k_scale = scale_s / (1000.0 * wavelength)     # 显示长度 / mrad
        _l2m = lambda v: np.asarray(v) / k_scale      # 显示长度 -> mrad
        _m2l = lambda v: np.asarray(v) * k_scale      # mrad -> 显示长度
        _m2n = lambda v: np.asarray(v) * 10.0 / scale_s    # 显示长度 -> nm^-1
        _n2m = lambda v: np.asarray(v) * scale_s / 10.0    # nm^-1 -> 显示长度
        if np.isfinite(apert_msr) and apert_msr > 0:
            step = 5 if apert_msr <= 20 else (10 if apert_msr <= 60 else 20)
            t_mrad = np.arange(-apert_msr, apert_msr + 1e-9, step)
            ax.set_xticks(_m2l(t_mrad))
            ax.set_yticks(_m2l(t_mrad))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{_l2m(v):g}"))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{_l2m(v):g}"))
        ax.set_xlabel("X [mrad]")
        ax.set_ylabel("Y [mrad]")
        ax.secondary_xaxis('top', functions=(_m2n, _n2m)).set_xlabel(
            'nm^-1', fontsize=8)
        ax.secondary_yaxis('right', functions=(_m2n, _n2m)).set_ylabel(
            'nm^-1', fontsize=8)
        # 旋转操控方向箭头：ax.clear() 会清除已有 artist，故每次重绘后重建
        self._update_rot_arrow()

    # ------------------------------------------------------------------ CBED
    def _read_float(self, var, name):
        try:
            return float(var.get())
        except ValueError:
            messagebox.showwarning("输入错误", f"{name} 需为数字")
            return None

    def _update_nyq_half(self):
        """实空间采样决定的奈奎斯特散射半角(mrad)，即 CBED 可见角上限。"""
        try:
            e_kev = float(self.energy_var.get())
            s = float(self.sampling_var.get())
            if e_kev <= 0 or s <= 0:
                raise ValueError
            lam = 12.26 / np.sqrt(e_kev * 1e3 * (1 + 9.78e-7 * e_kev * 1e3))
            half = np.arcsin(lam / (4.0 * s)) * 1000.0  # 弧度直接×1000 → mrad
            self._nyq_half_var.set(f"半角限 ≈{half:.0f} mrad")
        except (ValueError, ZeroDivisionError):
            self._nyq_half_var.set("")

    def _on_cbed(self):
        if self.atoms is None:
            messagebox.showinfo("提示", "请先导入 CIF 文件")
            return
        if self._running:
            return
        params = {}
        for name, var in [("semiangle_mrad", self.semiangle_var),
                          ("max_angle_mrad", self.max_angle_var),
                          ("box_width_nm", self.width_var),
                          ("sampling", self.sampling_var)]:
            v = self._read_float(var, name)
            if v is None:
                return
            params[name] = v
        params["box_width_nm"] = max(0.1, params["box_width_nm"])
        params["energy_kev"] = self._read_float(self.energy_var, "加速电压") or 200.0
        _t = self._read_float(self.thick_var, "厚度")
        # 0 表示零/空厚度（自由空间，无散射）；仅非数字输入才回退默认 20nm
        params["thickness_nm"] = 20.0 if _t is None else max(0.0, _t)

        # 快照本次模拟的取向与带轴读数：以后拖拽左侧面板只改变实时 self.R，
        # 不改变已提交的 CBED 结果（右侧标题也冻结为本时刻带轴）。
        # _build_box_fill 内部即 p = posC @ R.T，需新约定 self.R，故直接存 self.R
        # （原存 self.R.T 会再转置一次，使 CBED 原子方向与屏幕相反）。
        params["_snap_R"] = np.array(self.R, float)
        params["_snap_zone"] = self.zone_label.get().split("    ")[0].replace(
            "当前带轴:", "").strip()

        self._running = True
        self._cancel_event.clear()   # 本次新模拟开始时复位取消标志
        self.cbed_btn.config(state="disabled")
        self.cbed_cancel_btn.config(state="normal")
        self.cbed_status.set("计算中…（大超胞/厚样品可能需数分钟）")
        self.cbed_time_var.set("")
        self._cbed_start = time.time()

        t = threading.Thread(target=self._cbed_worker, args=(params,), daemon=True)
        t.start()
        self.root.after(150, self._poll_result)

    def _cancel_cbed(self):
        """请求取消本次 CBED 模拟。worker 在关键步骤检查标志后尽快返回。"""
        if not self._running:
            return
        self._cancel_event.set()
        self.cbed_status.set("正在取消…")
        self.cbed_btn.config(state="disabled")
        self.cbed_cancel_btn.config(state="disabled")
        # worker 取消时直接 return 不向队列写结果 → poll 永远不会复位。
        # 这里轮询等线程结束（daemon 线程），再手动复位 UI。
        self.root.after(60, self._finish_cancel)

    def _finish_cancel(self):
        # 没有好办法在 tk 线程里拿到 worker 线程对象本身已句柄，等它结束靠
        # running 仍为 True + 已置取消标志即可复位。为简单起见取消即复位：
        # worker 是 daemon 且会在下次检查点退出，让出主线程时间片后复位。
        self._running = False
        self.cbed_btn.config(state="normal")
        self.cbed_cancel_btn.config(state="disabled")
        self.cbed_status.set("已取消")
        self._show_elapsed()

    def _cbed_worker(self, params):
        try:
            import abtem

            # 束框(beam-frame)长方体样品：横向边长 box_width_nm、厚度 thickness_nm。
            # 用"枚举晶胞平移→旋转→剪切→横向回绕"填充原子，见 _build_box_fill。
            R = params.get("_snap_R", self.R)   # 新约定：snap 存 self.R；兜底同样用 self.R
            zone_snap = params.get("_snap_zone", "")

            # 把当前连续取向取整到高精度整数带轴，再据此重建旋转矩阵。
            # 这样超胞沿束方向的周期严格整数、可被 abTEM 干净延拓，同时因
            # zone_max_index 很大(12000)角误差≈0，与左面板真取向几乎一致。
            cell0 = np.asarray(self.atoms.get_cell()[:], float)
            n_orig = R.T @ Z_AXIS
            n_orig = n_orig / (np.linalg.norm(n_orig) or 1.0)
            uvw0 = snap_zone_axis(n_orig, cell0, self.cfg.get("zone_max_index", 12000))
            n_snap = np.asarray(uvw0, float) @ cell0
            ns = np.linalg.norm(n_snap)
            if ns > 1e-9:
                n_snap /= ns
                Align = rotation_align(n_orig, Z_AXIS)
                M = R @ Align.T
                phi = np.arctan2(M[1, 0], M[0, 0])
                AlignS = rotation_align(n_snap, Z_AXIS)
                R = Rz(phi) @ AlignS
                ang_uvw = float(np.degrees(np.arccos(np.clip(np.dot(n_orig, n_snap), -1, 1))))
                # 内部用高精度(12000)保证周期性；状态栏只显示小指数归约，便于阅读。
                uvw_show = snap_zone_axis(n_orig, cell0, 24)
            else:
                uvw0, ang_uvw, uvw_show = None, 0.0, None

            wx_a = params["box_width_nm"] * 10.0     # nm -> Å (横向 wx)
            wy_a = params["box_width_nm"] * 10.0     # nm -> Å (横向 wy)
            t_a = params["thickness_nm"] * 10.0      # nm -> Å (厚度 t)
            if params["thickness_nm"] <= 0:
                # 零/空厚度：不砌原子，直接自由空间传播，结果只有直接束(入射探针
                # 盘)、无衍射。cell 只需一个非零 z 供 abtem 网格对齐。
                from ase import Atoms
                atoms = Atoms(positions=np.zeros((0, 3)),
                              cell=np.diag([max(wx_a, 1.0), max(wy_a, 1.0), 1.0]),
                              pbc=[True, True, False])
                n_atoms = 0
            else:
                atoms = self._build_box_fill(R, wx_a, wy_a, t_a)
                n_atoms = len(atoms)
                # _build_box_fill 产出的是第一类模型（束沿 -z，衍射坐标系）。
                # abtem 束固定 +z，故绕 +x 轴真旋转 180° 转为第二类模型再送入多片法。
                atoms = self._flip_to_plus_z(atoms)

            energy = params["energy_kev"] * 1e3
            sampling = params["sampling"]

            potential = abtem.Potential(atoms, sampling=sampling)
            probe = abtem.Probe(energy=energy,
                                semiangle_cutoff=params["semiangle_mrad"],
                                sampling=sampling)   # 束沿 +z（＝带轴），无需 tilt
            # lazy=False：即时计算，避免斜晶胞大网格触发 dask 自动分块失败
            # (RuntimeError: Object cannot be automatically chunked)
            if self._cancel_event.is_set():
                return
            exit_waves = probe.multislice(potential, lazy=False)
            if self._cancel_event.is_set():
                return
            patterns = exit_waves.diffraction_patterns(
                max_angle=params["max_angle_mrad"])
            arr = np.asarray(patterns.array)
            # abtem 约定：数组轴序 (…, kx, ky)，angular_coordinates 返回
            # (alpha_x, alpha_y) = (kx, ky)（第一项是 kx）。照官方 show() 处理：
            # 绘制时把 arr 转置，使 横向= kx、纵向= ky。
            kx, ky = patterns.angular_coordinates

            self._result_q.put(("ok", arr, kx, ky, (0.0, 0.0), params,
                                uvw_show, ang_uvw, n_atoms, zone_snap))
        except Exception as e:
            import traceback
            self._result_q.put(("err", traceback.format_exc()))

    def _build_box_fill(self, R, wx, wy, t):
        """沿束框长方体 (0..wx, 0..wy, 0..t，单位 Å) 填充旋转后的晶体原子。

        做法：把晶体按 R 旋转后，枚举晶胞平移 → 旋转 → 只保留落在盒内的原子，
        再整体平移到盒子正象限。关键点：
        - 只裁剪、不折叠，且不留出盒松量 ⇒ 坐标全落在 (0,0,0)..(wx,wy,t) 内，
          写 CIF 时 ase 不会再对负坐标取模回绕（否则 x=-10.13 会被折到 +9.87，
          与盒边真实原子撞出 ~0.27Å 的伪近对）。
        返回 ase.Atoms（**第一类模型**：入射束沿 -z、带轴 +z，对应衍射斑点坐标系），
        周期盒 = 束框正交盒。供 CBED 前需经 `_flip_to_plus_z` 绕 +x 轴真旋转 180°，
        转成第二类模型（束沿 +z）后再交给 abtem。
        """
        if self.atoms is None:
            raise RuntimeError("未加载晶体")
        cell0 = np.asarray(self.atoms.get_cell()[:], float)
        pos0 = np.asarray(self.atoms.get_positions(), float)
        sym0 = np.asarray(self.atoms.get_chemical_symbols())

        corners = np.array([[sx*wx/2, sy*wy/2, sz*t]
                            for sx in (-1, 1) for sy in (-1, 1) for sz in (0, 1)])
        cry = corners @ R
        frac = cry @ np.linalg.inv(cell0)
        lo = np.floor(frac.min(axis=0) - 1).astype(int)
        hi = np.ceil(frac.max(axis=0) + 1).astype(int)
        nxyz = hi - lo + 1
        total = int(nxyz.prod())
        if total > 100_000_000:
            raise RuntimeError(f"横向/厚度过大，晶胞平移数 {total} 超安全上限 1e8")
        idx = (np.indices(tuple(nxyz)).reshape(3, -1).T + lo).astype(float)
        trans = idx @ cell0
        posC = (trans[:, None, :] + pos0[None, :, :]).reshape(-1, 3)
        sym = np.tile(sym0, len(trans))
        p = posC @ R.T                                   # 旋转到束框

        # 束框在构造坐标系里以原点为中心 (x,y ∈ [-wx/2, wx/2], z ∈ [0,t])，
        # 因此裁剪区间必须与之匹配。之前误用 [0, wx]：只留 x≥0 半边、还放进盒子外的
        # 晶胞余量原子；平移 +wx/2 后这些原子越出 [0,20]，写 CIF 时被 ase 取模回绕到
        # 盒子中心另一侧，造出 0.268Å 镜像伪近对。改成对称区间后平移量=wx/2，
        # 坐标必然落在 [0,wx]×[0,wy]×[0,t]，无需再担心回绕。
        keep = (p[:, 0] >= -wx/2 - 1e-3) & (p[:, 0] <= wx/2 + 1e-3) \
               & (p[:, 1] >= -wy/2 - 1e-3) & (p[:, 1] <= wy/2 + 1e-3) \
               & (p[:, 2] >= -1e-3) & (p[:, 2] <= t + 1e-3)
        p, sym = p[keep], sym[keep]

        # 束框以原点为中心；平移到盒正象限，所有坐标保在 [0,wx][0,wy][0,t] 内
        p = p + np.array([wx/2, wy/2, 0.0])
        # 钳制回盒内：裁剪容差 ±1e-3 会放行落在面内侧 -1e-6 量级的边界原子，
        # 平移到盒正象限后其坐标仍微负/超界，写成轴向小数会出现 <0 或 >1，
        # 再被 ase/VESTA 读取时折叠到对向面，与对侧表面原子撞出 ~2Å 伪近对
        # （如原顶面 atom66/78 与底面临界原子）。nudge ≤1e-3 Å，物理可忽略。
        p = np.clip(p, [0.0, 0.0, 0.0], [wx, wy, t])

        cell = np.array([[wx, 0, 0], [0, wy, 0], [0, 0, t]])
        from ase import Atoms
        return Atoms(sym.tolist(), positions=p, cell=cell, pbc=[True, True, False])

    def _flip_to_plus_z(self, atoms):
        """把**第一类模型**（入射束沿 -z、带轴 +z，对应衍射斑点坐标系）绕 +x 轴
        真旋转 180°，转成**第二类模型**（入射束沿 +z、带轴 -z），供 abtem
        （束固定 +z）做 CBED 多片法使用。

        绕 +x 轴 180° 是真旋转（proper rotation，det=+1）：(x, y, z) -> (x, -y, -z)，
        把带轴 +z 翻成 -z；再整体平移 (0, wy, t) 把坐标移回正象限
        [0,wx]×[0,wy]×[0,t]。束框盒子绕自身中心对称，平移是刚体平移，不改变晶体结构，
        故 cell 保持对角盒 [wx,wy,t] 不变。物理上等价于把整个散射实验绕水平轴翻转，
        使"束 -z 射向带轴 +z"变为"束 +z 射向带轴 -z"。
        """
        import numpy as np
        from ase import Atoms
        pos = np.asarray(atoms.get_positions(), float)
        cell = np.asarray(atoms.get_cell()[:], float)
        sym = atoms.get_chemical_symbols()
        M = np.diag([1.0, -1.0, -1.0])      # 绕 +x 轴真旋转 180°
        p = pos @ M.T                       # (x, -y, -z)
        p = p + np.array([0.0, float(cell[1, 1]), float(cell[2, 2])])
        return Atoms(sym, positions=p, cell=cell, pbc=[True, True, False])

    def _poll_result(self):
        try:
            msg = self._result_q.get_nowait()
        except queue.Empty:
            if self._running:
                self.root.after(150, self._poll_result)
            return
        self._running = False
        self.cbed_btn.config(state="normal")
        self.cbed_cancel_btn.config(state="disabled")
        # 竞态防护：worker 若在取消前越过检查点仍可能写入结果；取消标志已置
        # 则丢弃，保持"已取消"状态不变。
        if self._cancel_event.is_set():
            return

        if msg[0] == "err":
            self.cbed_status.set("失败")
            self._show_elapsed()
            messagebox.showerror("CBED 计算失败", msg[1])
            return

        _, arr, kx, ky, tilt, params, uvw0, ang_uvw, n_atoms, zone_snap = msg
        self._draw_cbed(arr, kx, ky, tilt, params, zone_snap)
        if uvw0 is not None:
            self.cbed_status.set(
                f"完成  带轴 {uvw0[0]} {uvw0[1]} {uvw0[2]}  (角偏差 {ang_uvw:.2f}°)  "
                f"盒内 {n_atoms} 原子")
        else:
            self.cbed_status.set("完成")
        self._show_elapsed()

    def _export_atoms(self):
        """按当前面板参数（横向边长、厚度）与当前取向 self.R，直接构造束框原子并导出。

        无需先跑 CBED 模拟：构建是向量化的（~毫秒级）。用 ase.io.write 按扩展名选格式。
        """
        if self.atoms is None:
            messagebox.showinfo("导出原子坐标", "请先导入晶体 (CIF)。")
            return

        def _rv(name, var):
            v = self._read_float(var, name)
            return v if v is not None else self.cfg.get("box_width_nm", 10.8)

        w_nm = max(0.1, _rv("横向边长", self.width_var))   # nm，横向 w
        t_nm = _rv("厚度", self.thick_var) or DEFAULTS["thickness_nm"]
        if t_nm <= 0:
            messagebox.showinfo("导出原子坐标", "厚度为 0，样品为空，无可导出的原子。")
            return
        try:
            # _build_box_fill 内部已用 p = posC @ R.T，故其 R 参数即为新约定 self.R
            #（晶体→实验室）。直接传 self.R；若误传 self.R.T 会再转置一次得到反向模型。
            atoms = self._build_box_fill(self.R, w_nm*10.0, w_nm*10.0, t_nm*10.0)
        except Exception as exc:
            messagebox.showerror("导出原子坐标", f"构造原子失败:\n{exc}")
            return

        from tkinter import filedialog
        path = filedialog.asksaveasfilename(
            title="导出原子坐标",
            defaultextension=".xyz",
            initialfile="cbed_atoms.xyz",
            filetypes=[("XYZ 文本", "*.xyz"),
                       ("CIF 晶体结构", "*.cif"),
                       ("原子坐标列表 文本", "*.txt"),
                       ("VASP POSCAR", "*.vasp"),
                       ("PDB", "*.pdb")])
        if not path:
            return
        try:
            if path.lower().endswith(".txt"):
                self._export_coord_list(path, atoms)
            else:
                import ase.io
                ase.io.write(path, atoms)
            self.cbed_status.set(f"已导出 {path}（{len(atoms)} 原子，横向{w_nm:.2f}nm，厚{t_nm:.2f}nm）")
        except Exception as exc:  # 写出失败时给出可读提示，不打断主线程
            messagebox.showerror("导出原子坐标", f"写入失败:\n{exc}")

    def _export_coord_list(self, path, atoms):
        """纯文本原子坐标列表：每个原子一行 `元素  x  y  z`（笛卡尔坐标，单位 Å）。
        数字直接取原子的实际数学坐标（四舍五入到 6 位），不写 P1/CIF 式头部，
        方便直接粘贴进脚本或表格。"""
        cell = atoms.get_cell()
        w, h, t = [float(cell[i, i]) for i in range(3)]
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"# atom coords list  (Cartesian, Angstrom).  box {w:.4f} x {h:.4f} x {t:.4f} A^3\n")
            f.write("# symbol\tx\ty\tz\n")
            for sym, pos in zip(atoms.get_chemical_symbols(), atoms.get_positions()):
                f.write(f"{sym}\t{pos[0]:.6f}\t{pos[1]:.6f}\t{pos[2]:.6f}\n")

    def _show_elapsed(self):
        if getattr(self, "_cbed_start", None) is None:
            return
        dt = time.time() - self._cbed_start
        self.cbed_time_var.set(f"本次模拟耗时: {dt:.1f} s")

    def _export_cbed_tif(self):
        """把右侧当前 CBED 图（含标尺/刻度/标题/颜色轴）导出为带标尺的 TIFF。"""
        # 若无 CBED 结果，先提示但不阻塞
        if getattr(self, "_cbed_start", None) is None:
            # _cbed_start 仅在第一次模拟时设置；用右侧画布是否有内容粗判
            has_content = bool(self.ax_right.images) or bool(self.ax_right.texts) \
                or bool(self.ax_right.lines)
            if not has_content:
                messagebox.showinfo("导出 TIFF",
                                    "尚未进行 CBED 模拟，请先点击”CBED 模拟“。")
                return
        from tkinter import filedialog
        path = filedialog.asksaveasfilename(
            parent=self.root, title="导出 CBED 图 (TIFF)",
            defaultextension=".tif",
            initialfile="cbed_pattern.tif",
            filetypes=[("TIFF 图像", "*.tif *.tiff"), ("PNG 图像", "*.png"),
                       ("所有文件", "*.*")])
        if not path:
            return
        try:
            self._write_cbed_tif(path)
        except Exception as exc:
            messagebox.showerror("导出失败", f"导出 CBED 图时出错：\n{exc}")
            return
        self.cbed_status.set(f"已导出 {os.path.basename(path)}")

    def _write_cbed_tif(self, path):
        """把真实 CBED 强度归一化到 [0,1] 后以 float32（单通道、非 RGB）写成 TIFF。

        在图像数组之外（下方）直接扩充一块黑色背景条（高=原图高的 20%），并在这条
        黑带上画白色标尺条（左起点距图左边缘 10% 图宽，长度由"标尺长度 (nm^-1)"
        控制，默认 5）与白色标记（位于标尺右侧，数值+单位随长度）。全部在 numpy 数组
        层面完成，不做 matplotlib 绘图，保证数值图可被 DM3/ImageJ 直接读取真实强度。
        """
        c = getattr(self, "_cbed_cached", None)
        if c is None or not len(c.get("arr", ())):
            raise RuntimeError("尚无 CBED 结果，请先点击“CBED 模拟”。")
        from PIL import Image, ImageDraw, ImageFont

        # 轴序 (…, kx, ky)，转置成 (ky, kx) → 行=ky、列=kx。
        # 方向约定分两次 Y 翻转，导出时恰好抵消：
        #  · 第1次(模型镜像)：第二类模型(束沿+z)相对第一类(束沿-z)翻转，
        #    已在显示侧 _draw_cbed 完成(其 np.flipud(arr.T))；
        #  · 第2次(图像存储)：CBED 面板 Y 朝上，而 DM3/看图器 Y 朝下，
        #    导出需再翻转 Y 才能与面板画面一致。
        #  两次翻转抵消 → 导出数组直接取 arr.T（不再额外 flipud），
        #  载入 Y 朝下的看图器后，画面即与右侧 CBED 面板完全一致（+kx 右、+ky 上）。
        arr = np.asarray(c["arr"], np.float32).T
        mx = arr.max()
        arr = arr / mx if mx > 0 else arr

        H, W = arr.shape
        bar_h = max(3, int(round(0.20 * H)))        # 黑条高 = 原图高 20%

        # 标尺长度：px/nm^-1 换算（dkx mrad/px；nm^-1 = mrad/(100·λ)）
        lam = c["lam"]
        kx = np.asarray(c["kx"], float)
        dkx = (kx[-1] - kx[0]) / max(1, len(kx) - 1)
        dnm = dkx / (100.0 * lam)                   # nm^-1 / 像素
        try:
            scl = float(self.sclen_var.get())
        except (AttributeError, ValueError):
            scl = float(DEFAULTS.get("scalebar_nm", 5.0))
        scl = max(1e-3, scl)
        scale_px = int(round(scl / dnm))
        scale_px = max(2, min(scale_px, int(0.6 * W)))   # 限 60% 图宽

        # 直接扩充数组：图像下方追加黑色条（全 0）
        out = np.zeros((H + bar_h, W), dtype=np.float32)
        out[:H, :] = arr

        # 白色标尺条：左起点距图左边缘 10% 图宽；垂直居中于黑条；厚度=黑条高的 45%×0.25
        x0 = int(round(0.10 * W))
        sh = max(2, int(round(0.1125 * bar_h)))      # 标尺条厚度（原 45% 的 1/4）
        yc = H + bar_h // 2
        x1 = min(W, x0 + scale_px)
        out[yc - sh // 2: yc - sh // 2 + sh, x0:x1] = 1.0

        # 白色标记：PIL 渲染文字位图，放到标尺右侧（亮度=1.0，与标尺一致）
        text_h = max(4, int(round(0.5 * bar_h)))
        font = None
        for fp in (r"C:\Windows\Fonts\arial.ttf", "arial.ttf", "DejaVuSans.ttf"):
            try:
                font = ImageFont.truetype(fp, size=text_h)
                break
            except Exception:
                continue
        if font is None:
            font = ImageFont.load_default()
        txt = f"{scl:g} nm-1"
        dr = ImageDraw.Draw(Image.new("L", (8, 8), 0))
        bb = dr.textbbox((0, 0), txt, font=font)
        tw, th = bb[2] - bb[0], bb[3] - bb[1]
        glyph = Image.new("L", (tw, th), 0)
        dg = ImageDraw.Draw(glyph)
        dg.text((-bb[0], -bb[1]), txt, fill=255, font=font)
        garr = np.asarray(glyph, np.float32) / 255.0

        tx0 = x1 + max(2, int(round(0.02 * W)))
        ty0 = yc - th // 2
        w_put = min(tw, max(0, W - tx0))
        h_put = min(th, max(0, H + bar_h - ty0))
        if w_put > 0 and h_put > 0 and tx0 < W and ty0 >= H:
            out[ty0:ty0 + h_put, tx0:tx0 + w_put] = np.maximum(
                out[ty0:ty0 + h_put, tx0:tx0 + w_put], garr[:h_put, :w_put])

        Image.fromarray(out, mode="F").save(path, format="TIFF")

    def _draw_cbed(self, arr, kx, ky, tilt, params, zone_snap=""):
        ax = self.ax_right
        ax.clear()
        # 数组轴序 (…, kx, ky)；转置成 (ky, kx) 后: 行=ky(纵向)、列=kx(横向)。
        # 采用与 abtem 官方 show() 一致的 origin='lower'：这样 +kx 朝右、+ky 朝上，
        # 与左侧衍射/菊池面板(gx 右、gy 上)完全对齐（origin='lower' 而非 upper，
        # 否则纵向会多一个 180° 镜像）。
        # CBED 使用第二类模型（由第一类绕 +x 轴真旋转 180° 得到），故其花样已
        # 相对左侧第一类坐标系翻转；显示前需再对 Y(ky) 轴翻转一次（np.flipud）
        # 才与左侧坐标一致，即"绕 x 轴转 180°"补回这一镜像。
        log_arr = np.log1p(np.flipud(arr.T))
        # 缓存对数强度供"显示"选项卡拉条实时重绘；显示数组经亮度/对比度变换。
        self._cbed_log_arr = log_arr
        disp = self._cbed_display_transform(log_arr)
        # 把 000 盘（束倾位置）移到中心，便于对齐当前带轴
        tx, ty = tilt
        ext = [kx[0] - tx, kx[-1] - tx, ky[0] - ty, ky[-1] - ty]
        im = ax.imshow(disp, extent=ext, cmap="inferno", origin="lower",
                       vmin=0.0, vmax=1.0)
        self._cbed_im = im
        ax.set_xlabel("散射角 [mrad] (kx)")
        ax.set_ylabel("散射角 [mrad] (ky)")
        # 用按下按钮时冻结的带轴，拖拽左面板不再改写右侧标题
        zone_str = zone_snap or (self.zone_label.get().split("    ")[0].replace(
            "当前带轴:", "").strip())
        ax.set_title(f"CBED  t={params['thickness_nm']:g} nm, "
                     f"{params['energy_kev']:g} keV, "
                     f"α={params['semiangle_mrad']:g} mrad\n"
                     f"带轴 {zone_str}")
        ax.set_aspect("equal")
        # 记录全幅显示范围，供滚轮缩放约束回界
        self._cbed_full_ext = ext
        self.ax_right.set_xlim(ext[0], ext[1])
        self.ax_right.set_ylim(ext[2], ext[3])
        # ---- 次级刻度：右侧/顶部用 nm^-1。小角 δθ(mrad)=1000·λ·|q|(Å⁻¹)，
        # 故 q(nm^-1)=θ(mrad)/(100·λ)。λ 由加速电压的电子波长给出。
        lam = 12.26 / np.sqrt(params['energy_kev'] * 1e3 *
                              (1 + 9.78e-7 * params['energy_kev'] * 1e3))
        _m2n = lambda v: np.asarray(v) / (100.0 * lam)      # mrad -> nm^-1
        _n2m = lambda v: np.asarray(v) * (100.0 * lam)      # nm^-1 -> mrad
        ax.secondary_xaxis('top', functions=(_m2n, _n2m)).set_xlabel('nm^-1',
                                                                     fontsize=8)
        ax.secondary_yaxis('right', functions=(_m2n, _n2m)).set_ylabel('nm^-1',
                                                                      fontsize=8)
        self.canvas_right.draw_idle()
        # 保存真实强度与坐标信息，供"导出图像"写出 16 位灰度 TIFF（保留真实强度值）。
        self._cbed_cached = {
            "arr": np.asarray(arr, float),   # 原始真实强度，(…, kx, ky)
            "kx": np.asarray(kx, float),
            "ky": np.asarray(ky, float),
            "tilt": tilt,
            "params": dict(params),
            "zone": zone_str,
            "lam": lam,                      # Å
        }

    def _cbed_display_transform(self, log_arr):
        """‘显示’选项卡的亮度/对比度/伽马变换，返回 [0,1] 显示数组。

        以 000 盘的对数峰值 M 归一到 [0,1]：
          disp = clip(对比度*(log/M)^伽马 + 亮度, 0, 1)。
        默认(对比度=1, 亮度=0, 伽马=1)恒等于改动前的 auto-clim 呈现。
        """
        c = float(self.cbed_contrast_var.get())
        b = float(self.cbed_bright_var.get())
        g = float(self.cbed_gamma_var.get())
        M = float(log_arr.max()) if getattr(log_arr, "size", 0) else 1.0
        if not np.isfinite(M) or M <= 0:
            M = 1.0
        n = np.clip(log_arr / M, 0.0, 1.0)
        return np.clip(c * np.power(n, g) + b, 0.0, 1.0)

    def _on_cbed_scroll(self, event):
        """右侧 CBED：滚轮围绕光标缩放，约束在数据全幅内。"""
        if event.inaxes is not self.ax_right:
            return
        if not getattr(self, "_cbed_im", None):
            return
        ax = self.ax_right
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        xmin, xmax, ymin, ymax = self._cbed_full_ext
        factor = 1.2 if event.step < 0 else 1.0 / 1.2   # 上滚放大，下滚缩小（Windows 上滚 step<0）
        # 保持窗口为正方形，避免 aspect=equal 使坐标轴框高度变化、视场尺寸漂移
        full_side = min(xmax - xmin, ymax - ymin)
        cur_side = min(x1 - x0, y1 - y0)
        side = min(full_side, max(0.1 * full_side, cur_side * factor))
        xc = event.xdata if event.xdata is not None else 0.5 * (xmin + xmax)
        yc = event.ydata if event.ydata is not None else 0.5 * (ymin + ymax)
        hx, hy = side / 2.0, side / 2.0
        cx = min(max(xc, xmin + hx), xmax - hx)
        cy = min(max(yc, ymin + hy), ymax - hy)
        ax.set_xlim(cx - hx, cx + hx)
        ax.set_ylim(cy - hy, cy + hy)
        self.canvas_right.draw_idle()

    def _update_cbed_display(self):
        """拉条联动：用缓存的对数强度实时重绘（不重跑仿真）。"""
        if getattr(self, "_cbed_log_arr", None) is None:
            return
        im = getattr(self, "_cbed_im", None)
        if im is None:
            return
        im.set_data(self._cbed_display_transform(self._cbed_log_arr))
        self.canvas_right.draw_idle()


def main():
    root = tk.Tk()
    CBEDApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()