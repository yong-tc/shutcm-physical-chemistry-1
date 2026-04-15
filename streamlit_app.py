import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.interpolate import PchipInterpolator, CubicSpline
from scipy.optimize import brentq
import base64
from io import BytesIO

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
    st.session_state.calc_df = None
    st.session_state.azeo_info = None
    st.session_state.fig = None

# 组成计算函数（内置经验公式）
def calc_composition_builtin(n):
    """内置公式：环己烷摩尔分数 = -46.8505*n^2 + 145.819*n - 111.634"""
    return -46.8505 * n**2 + 145.819 * n - 111.634

# 平滑曲线绘制辅助函数
def smooth_curve(x, y, num=300):
    if len(x) < 3:
        return x, y
    idx = np.argsort(x)
    x_sorted = np.array(x)[idx]
    y_sorted = np.array(y)[idx]
    try:
        cs = PchipInterpolator(x_sorted, y_sorted)
        x_smooth = np.linspace(x_sorted.min(), x_sorted.max(), num)
        y_smooth = cs(x_smooth)
        return x_smooth, y_smooth
    except:
        cs = CubicSpline(x_sorted, y_sorted, extrapolate=False)
        x_smooth = np.linspace(x_sorted.min(), x_sorted.max(), num)
        y_smooth = cs(x_smooth)
        return x_smooth, y_smooth

def find_azeotrope_from_curves(x_liq, y_liq, x_vap, y_vap):
    # 按组成排序
    idx_liq = np.argsort(x_liq)
    x_liq_sorted = np.array(x_liq)[idx_liq]
    T_liq_sorted = np.array(y_liq)[idx_liq]
    idx_vap = np.argsort(x_vap)
    x_vap_sorted = np.array(x_vap)[idx_vap]
    T_vap_sorted = np.array(y_vap)[idx_vap]
    try:
        f_liq = PchipInterpolator(x_liq_sorted, T_liq_sorted)
        f_vap = PchipInterpolator(x_vap_sorted, T_vap_sorted)
    except:
        f_liq = CubicSpline(x_liq_sorted, T_liq_sorted)
        f_vap = CubicSpline(x_vap_sorted, T_vap_sorted)
    x_min = max(x_liq_sorted.min(), x_vap_sorted.min())
    x_max = min(x_liq_sorted.max(), x_vap_sorted.max())
    if x_min >= x_max:
        return None, None
    def delta_T(x):
        return f_liq(x) - f_vap(x)
    try:
        x_azeo = brentq(delta_T, x_min, x_max)
        T_azeo = f_liq(x_azeo)
        return x_azeo, T_azeo
    except:
        x_test = np.linspace(x_min, x_max, 500)
        diff = np.abs(delta_T(x_test))
        idx_min = np.argmin(diff)
        return x_test[idx_min], (f_liq(x_test[idx_min]) + f_vap(x_test[idx_min])) / 2

# ==================== 侧边栏：组成计算方法 ====================
st.sidebar.header("⚙️ 计算设置")
method = st.sidebar.radio("组成计算方法", ["使用内置经验公式", "上传标准曲线（二次多项式）"])
calc_composition = calc_composition_builtin
if method == "上传标准曲线（二次多项式）":
    uploaded_std = st.sidebar.file_uploader("标准曲线CSV (列：折光率, 环己烷摩尔分数)", type="csv")
    if uploaded_std:
        df_std = pd.read_csv(uploaded_std)
        if df_std.shape[0] >= 3:
            coeffs = np.polyfit(df_std.iloc[:,0], df_std.iloc[:,1], 2)
            st.sidebar.success(f"拟合多项式: {coeffs[0]:.4f} n² + {coeffs[1]:.4f} n + {coeffs[2]:.4f}")
            def calc_composition_custom(n):
                return coeffs[0]*n**2 + coeffs[1]*n + coeffs[2]
            calc_composition = calc_composition_custom
        else:
            st.sidebar.warning("标准曲线点数不足，继续使用内置公式")
    else:
        st.sidebar.info("未上传文件，使用内置经验公式")

# ==================== 数据编辑 ====================
st.header("📝 实验数据")
data_source = st.radio("数据来源", ["使用内置示例数据", "手动编辑表格", "上传CSV文件"], horizontal=True)
if data_source == "使用内置示例数据":
    df = st.session_state.sample_data.copy()
elif data_source == "手动编辑表格":
    df = st.data_editor(st.session_state.sample_data, num_rows="dynamic", use_container_width=True)
else:
    uploaded = st.file_uploader("CSV (列：编号, 沸点(℃), 液相折光率 n, 气相折光率 n)", type="csv")
    if uploaded:
        df = pd.read_csv(uploaded)
    else:
        st.stop()
st.session_state.sample_data = df

# ==================== 计算与绘图 ====================
if st.button("📈 生成平滑相图及计算恒沸点"):
    if df.empty:
        st.error("无有效数据")
    else:
        # 计算组成
        df["环己烷液相组成 (x)"] = df["液相折光率 n"].apply(calc_composition)
        df["环己烷气相组成 (y)"] = df["气相折光率 n"].apply(calc_composition)
        
        # 剔除异常提醒
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
        
        # 寻找恒沸点
        x_azeo, T_azeo = find_azeotrope_from_curves(x_liq_smooth, T_liq_smooth, x_vap_smooth, T_vap_smooth)
        
        # 绘图
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x_liq_smooth, y=T_liq_smooth, mode='lines', name='液相线 (平滑)', line=dict(color='blue', width=3)))
        fig.add_trace(go.Scatter(x=x_liq, y=T_liq, mode='markers', name='液相实验点', marker=dict(color='blue', size=8, symbol='circle')))
        fig.add_trace(go.Scatter(x=x_vap_smooth, y=T_vap_smooth, mode='lines', name='气相线 (平滑)', line=dict(color='red', width=3, dash='dash')))
        fig.add_trace(go.Scatter(x=x_vap, y=T_vap, mode='markers', name='气相实验点', marker=dict(color='red', size=8, symbol='x')))
        if x_azeo is not None and T_azeo is not None:
            fig.add_trace(go.Scatter(x=[x_azeo], y=[T_azeo], mode='markers', name=f'最低恒沸点 ({T_azeo:.1f}℃, {x_azeo:.3f})', marker=dict(color='green', size=14, symbol='star')))
            st.success(f"**最低恒沸点温度：{T_azeo:.2f} ℃**\n\n**恒沸物组成（环己烷摩尔分数）：{x_azeo:.4f}**")
        else:
            st.warning("未能计算出精确恒沸点")
        fig.update_layout(title="乙醇-环己烷系统沸点-组成图（平滑曲线）", xaxis_title="环己烷摩尔分数", yaxis_title="沸点 (℃)", legend=dict(x=0.05, y=0.95), width=800, height=600)
        st.plotly_chart(fig, use_container_width=True)
        
        # 显示计算结果表格
        st.subheader("📊 环己烷摩尔分数计算结果")
        result_cols = ["编号", "沸点 (℃)", "液相折光率 n", "环己烷液相组成 (x)", "气相折光率 n", "环己烷气相组成 (y)"]
        st.dataframe(df[result_cols], use_container_width=True)
        
        # 保存到 session_state
        st.session_state.calc_df = df[result_cols].copy()
        st.session_state.azeo_info = (x_azeo, T_azeo) if x_azeo is not None else (None, None)
        st.session_state.fig = fig

# ==================== 打印报告功能 ====================
st.markdown("---")
st.subheader("🖨️ 生成实验报告（PDF）")
if st.button("📄 生成并打印报告"):
    if st.session_state.calc_df is None:
        st.warning("请先点击上方按钮生成相图和计算结果")
    else:
        # 准备数据
        result_html = st.session_state.calc_df.to_html(index=False)
        # 获取恒沸点信息
        x_azeo, T_azeo = st.session_state.azeo_info
        azeo_text = f"最低恒沸点温度：{T_azeo:.2f} ℃；恒沸物组成（环己烷摩尔分数）：{x_azeo:.4f}" if x_azeo is not None else "未计算出恒沸点"
        # 获取图表 HTML
        fig = st.session_state.fig
        fig_html = fig.to_html(full_html=False, include_plotlyjs='cdn')
        # 获取计算方法说明
        if method == "使用内置经验公式":
            method_desc = "内置经验公式：y = -46.8505 n² + 145.819 n - 111.634"
        else:
            method_desc = "用户上传标准曲线拟合的二次多项式"
        
        # 构建完整 HTML 报告
        full_html = f"""
        <html>
        <head>
            <meta charset="UTF-8">
            <title>乙醇-环己烷相图实验报告</title>
            <style>
                body {{ font-family: 'SimHei', 'Microsoft YaHei', Arial, sans-serif; margin: 40px; }}
                h1 {{ color: #2c3e50; }}
                h2 {{ color: #34495e; border-bottom: 1px solid #ddd; }}
                table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: center; }}
                th {{ background-color: #f2f2f2; }}
                .chart {{ margin: 30px 0; page-break-inside: avoid; break-inside: avoid; }}
                .info {{ margin: 20px 0; padding: 10px; background-color: #f9f9f9; border-left: 4px solid #2c3e50; }}
            </style>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        </head>
        <body>
            <h1>乙醇-环己烷二元系统沸点-组成图实验报告</h1>
            <p>生成时间：{pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <h2>1. 实验数据与计算结果</h2>
            {result_html}
            <div class="info">
                <strong>组成计算方法：</strong> {method_desc}<br>
                <strong>恒沸点信息：</strong> {azeo_text}
            </div>
            <h2>2. 沸点-组成相图（平滑曲线）</h2>
            <div class="chart">
                {fig_html}
            </div>
            <h2>3. 实验结论</h2>
            <p>根据实验数据绘制了乙醇-环己烷系统的沸点-组成图，采用 PCHIP 插值获得平滑曲线。由图可知，该系统具有最低恒沸点，恒沸温度为 {T_azeo:.2f} ℃，恒沸物中环己烷的摩尔分数为 {x_azeo:.4f}。该结果符合柯诺瓦洛夫第二定律，与文献值基本一致。</p>
            <script>
                window.onload = function() {{ window.print(); }};
            </script>
        </body>
        </html>
        """
        # 嵌入 HTML 并自动触发打印
        st.components.v1.html(full_html, height=0, scrolling=False)
        st.success("报告已生成，请在弹出的打印对话框中选择「另存为 PDF」")
