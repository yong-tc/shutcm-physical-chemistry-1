import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.interpolate import CubicSpline, PchipInterpolator
from scipy.optimize import minimize_scalar, brentq

st.set_page_config(page_title="乙醇-环己烷相图（平滑曲线）", layout="wide")
st.title("🧪 乙醇-环己烷系统沸点-组成图（平滑曲线）")
st.markdown("根据沸点及气、液相折光率数据，计算环己烷摩尔分数，绘制平滑的沸点-组成相图，并确定最低恒沸点。")

# ==================== 初始化数据 ====================
if "sample_data" not in st.session_state:
    default_data = {
        "编号": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "沸点 (℃)": [80.6, 73.3, 66.4, 65.5, 65.1, 65.5, 66.4, 70.2, 74.5, 78.2],
        "液相折光率 n": [1.4229, 1.4220, 1.4184, 1.4155, 1.3990, 1.3791, 1.3685, 1.3641, 1.3608, 1.3586],
        "气相折光率 n": [1.4227, 1.4058, 1.4012, 1.4002, 1.3991, 1.3968, 1.3933, 1.3851, 1.3732, 1.3587]
    }
    st.session_state.sample_data = pd.DataFrame(default_data)

# 组成计算函数（内置经验公式）
def calc_composition(n):
    """内置公式：环己烷摩尔分数 = -46.8505*n^2 + 145.819*n - 111.634"""
    return -46.8505 * n**2 + 145.819 * n - 111.634

# 平滑曲线绘制辅助函数
def smooth_curve(x, y, num=300):
    """返回插值后的 x_smooth, y_smooth（单调保留）"""
    if len(x) < 3:
        return x, y
    # 按 x 排序（确保 x 单调递增，对于液相线需要）
    idx = np.argsort(x)
    x_sorted = np.array(x)[idx]
    y_sorted = np.array(y)[idx]
    try:
        # 使用 PCHIP 插值保留单调性（更适合相图）
        cs = PchipInterpolator(x_sorted, y_sorted)
        x_smooth = np.linspace(x_sorted.min(), x_sorted.max(), num)
        y_smooth = cs(x_smooth)
        return x_smooth, y_smooth
    except:
        # 降级使用三次样条
        cs = CubicSpline(x_sorted, y_sorted, extrapolate=False)
        x_smooth = np.linspace(x_sorted.min(), x_sorted.max(), num)
        y_smooth = cs(x_smooth)
        return x_smooth, y_smooth

def find_azeotrope_from_curves(x_liq, y_liq, x_vap, y_vap):
    """
    基于插值后的曲线寻找液相线和气相线的交点
    将两条曲线均插值为组成->沸点的函数，然后在组成轴上寻找沸点差零点
    """
    # 按组成排序（液相线）
    idx_liq = np.argsort(x_liq)
    x_liq_sorted = np.array(x_liq)[idx_liq]
    T_liq_sorted = np.array(y_liq)[idx_liq]
    # 按组成排序（气相线）
    idx_vap = np.argsort(x_vap)
    x_vap_sorted = np.array(x_vap)[idx_vap]
    T_vap_sorted = np.array(y_vap)[idx_vap]

    # 插值器：给定组成 -> 沸点
    try:
        f_liq = PchipInterpolator(x_liq_sorted, T_liq_sorted)
        f_vap = PchipInterpolator(x_vap_sorted, T_vap_sorted)
    except:
        f_liq = CubicSpline(x_liq_sorted, T_liq_sorted)
        f_vap = CubicSpline(x_vap_sorted, T_vap_sorted)

    # 定义组成范围（两条曲线组成范围的交集）
    x_min = max(x_liq_sorted.min(), x_vap_sorted.min())
    x_max = min(x_liq_sorted.max(), x_vap_sorted.max())
    if x_min >= x_max:
        return None, None

    # 定义沸点差函数
    def delta_T(x):
        return f_liq(x) - f_vap(x)

    # 寻找零点
    try:
        x_azeo = brentq(delta_T, x_min, x_max)
        T_azeo = f_liq(x_azeo)
        return x_azeo, T_azeo
    except:
        # 若无法找到精确零点，取差值绝对值最小的点
        x_test = np.linspace(x_min, x_max, 500)
        diff = np.abs(delta_T(x_test))
        idx_min = np.argmin(diff)
        return x_test[idx_min], (f_liq(x_test[idx_min]) + f_vap(x_test[idx_min])) / 2

# ==================== 侧边栏 ====================
st.sidebar.header("数据输入")
method = st.sidebar.radio("组成计算方法", ["使用内置经验公式", "上传标准曲线（二次多项式）"])
if method == "上传标准曲线（二次多项式）":
    uploaded_std = st.sidebar.file_uploader("标准曲线CSV (列：折光率, 环己烷摩尔分数)", type="csv")
    if uploaded_std:
        df_std = pd.read_csv(uploaded_std)
        if df_std.shape[0] >= 3:
            coeffs = np.polyfit(df_std.iloc[:,0], df_std.iloc[:,1], 2)
            st.sidebar.success(f"拟合多项式: {coeffs[0]:.4f} n² + {coeffs[1]:.4f} n + {coeffs[2]:.4f}")
            def calc_composition(n):
                return coeffs[0]*n**2 + coeffs[1]*n + coeffs[2]
        else:
            st.sidebar.warning("标准曲线点数不足，继续使用内置公式")
    else:
        st.sidebar.info("未上传文件，使用内置经验公式")

# ==================== 主界面数据编辑 ====================
st.header("📝 实验数据编辑")
data_source = st.radio("数据来源", ["使用内置示例数据", "手动编辑表格", "上传CSV文件"])
if data_source == "使用内置示例数据":
    df = st.session_state.sample_data.copy()
elif data_source == "手动编辑表格":
    df = st.data_editor(st.session_state.sample_data, num_rows="dynamic")
else:
    uploaded = st.file_uploader("CSV (列：编号, 沸点(℃), 液相折光率 n, 气相折光率 n)", type="csv")
    if uploaded:
        df = pd.read_csv(uploaded)
    else:
        st.stop()
st.session_state.sample_data = df

if st.button("📈 生成平滑相图"):
    if df.empty:
        st.error("无数据")
    else:
        # 计算组成
        df["环己烷液相组成 (x)"] = df["液相折光率 n"].apply(calc_composition)
        df["环己烷气相组成 (y)"] = df["气相折光率 n"].apply(calc_composition)

        # 剔除异常值（超出[0,1]的提醒）
        if ((df["环己烷液相组成 (x)"] < 0) | (df["环己烷液相组成 (x)"] > 1)).any():
            st.warning("部分液相组成超出 [0,1]，请检查折光率数据")
        if ((df["环己烷气相组成 (y)"] < 0) | (df["环己烷气相组成 (y)"] > 1)).any():
            st.warning("部分气相组成超出 [0,1]，请检查折光率数据")

        # 平滑插值
        x_liq = df["环己烷液相组成 (x)"].values
        T_liq = df["沸点 (℃)"].values
        x_vap = df["环己烷气相组成 (y)"].values
        T_vap = df["沸点 (℃)"].values

        x_liq_smooth, T_liq_smooth = smooth_curve(x_liq, T_liq)
        x_vap_smooth, T_vap_smooth = smooth_curve(x_vap, T_vap)

        # 寻找最低恒沸点（基于平滑曲线）
        x_azeo, T_azeo = find_azeotrope_from_curves(x_liq_smooth, T_liq_smooth, x_vap_smooth, T_vap_smooth)

        # 绘图
        fig = go.Figure()

        # 液相线（平滑）
        fig.add_trace(go.Scatter(
            x=x_liq_smooth, y=T_liq_smooth,
            mode='lines', name='液相线 (平滑)',
            line=dict(color='blue', width=3)
        ))
        # 液相原始数据点
        fig.add_trace(go.Scatter(
            x=x_liq, y=T_liq,
            mode='markers', name='液相实验点',
            marker=dict(color='blue', size=8, symbol='circle')
        ))

        # 气相线（平滑）
        fig.add_trace(go.Scatter(
            x=x_vap_smooth, y=T_vap_smooth,
            mode='lines', name='气相线 (平滑)',
            line=dict(color='red', width=3, dash='dash')
        ))
        # 气相原始数据点
        fig.add_trace(go.Scatter(
            x=x_vap, y=T_vap,
            mode='markers', name='气相实验点',
            marker=dict(color='red', size=8, symbol='x')
        ))

        # 恒沸点标注
        if x_azeo is not None and T_azeo is not None:
            fig.add_trace(go.Scatter(
                x=[x_azeo], y=[T_azeo],
                mode='markers', name=f'最低恒沸点 ({T_azeo:.1f}℃, {x_azeo:.3f})',
                marker=dict(color='green', size=14, symbol='star')
            ))
            st.success(f"**最低恒沸点温度：{T_azeo:.2f} ℃**\n\n**恒沸物组成（环己烷摩尔分数）：{x_azeo:.4f}**")
        else:
            st.warning("未能计算出精确的恒沸点，可能数据范围不足。")

        fig.update_layout(
            title="乙醇-环己烷系统沸点-组成图（平滑曲线）",
            xaxis_title="环己烷摩尔分数",
            yaxis_title="沸点 (℃)",
            legend=dict(x=0.05, y=0.95),
            width=800, height=600
        )
        st.plotly_chart(fig, use_container_width=True)

        # 显示计算结果表格
        st.subheader("📊 环己烷摩尔分数计算结果")
        result_cols = ["编号", "沸点 (℃)", "液相折光率 n", "环己烷液相组成 (x)", "气相折光率 n", "环己烷气相组成 (y)"]
        st.dataframe(df[result_cols], use_container_width=True)

        # 导出
        csv = df[result_cols].to_csv(index=False).encode('utf-8')
        st.download_button("下载计算结果 CSV", csv, "phase_diagram_smooth.csv", "text/csv")

# 实验原理说明
with st.expander("ℹ️ 实验原理及平滑曲线说明"):
    st.markdown("""
    - **平滑曲线的生成**：使用 PCHIP（分段三次 Hermite 插值）对实验点进行插值，保留了数据的单调性，避免了过冲，使相图曲线更符合热力学规律。
    - **最低恒沸点**：通过求解液相线与气相线的交点（沸点相等）得到，插值后使用数值求根（brentq）获得精确值。
    - **组成计算**：默认采用参考数据拟合的二次多项式 `y = -46.8505 n² + 145.819 n - 111.634`；也可上传自己的标准曲线进行多项式拟合。
    """)
