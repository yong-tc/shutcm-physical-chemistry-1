import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.interpolate import UnivariateSpline

st.set_page_config(page_title="二组分相图绘制", layout="wide")
st.title("🧪 二组分体系沸点-组成相图（乙醇-环己烷）")

# 初始化 session_state
if 'calculated' not in st.session_state:
    st.session_state.calculated = False

# 侧边栏说明
with st.sidebar:
    st.markdown("### 使用说明")
    st.markdown("1. 输入标准曲线数据（折光率 vs 环己烷质量分数）")
    st.markdown("2. 输入沸点实验数据（沸点、液相折光率、气相折光率）")
    st.markdown("3. 点击「开始计算」生成相图")
    st.markdown("4. 可下载 CSV 或生成 PDF 报告")

# ========== 标准曲线数据 ==========
st.subheader("📈 标准曲线（折光率 → 组成）")
st.markdown("请提供环己烷质量分数与折光率的关系（至少3个点）")

cal_default = pd.DataFrame({
    "环己烷质量分数": [0.006, 0.02, 0.08, 0.132, 0.30, 0.575, 0.65, 0.83, 0.94, 0.998],
    "折光率": [1.4229, 1.4220, 1.4184, 1.4155, 1.3990, 1.3791, 1.3685, 1.3641, 1.3608, 1.3586]
})
cal_df = st.data_editor(cal_default, num_rows="dynamic", use_container_width=True)

if len(cal_df) < 3:
    st.warning("标准曲线至少需要3个数据点")
    st.stop()

# 多项式拟合（次数可选）
degree = st.slider("拟合多项式次数", 1, 5, 3)
coeffs = np.polyfit(cal_df["环己烷质量分数"], cal_df["折光率"], degree)
poly = np.poly1d(coeffs)

# 绘制标准曲线
x_cal = np.linspace(0, 1, 200)
y_cal = poly(x_cal)
fig_cal = go.Figure()
fig_cal.add_trace(go.Scatter(x=cal_df["环己烷质量分数"], y=cal_df["折光率"], mode='markers', name='实验点'))
fig_cal.add_trace(go.Scatter(x=x_cal, y=y_cal, mode='lines', name=f'拟合曲线 (次数={degree})'))
fig_cal.update_layout(xaxis_title="环己烷质量分数", yaxis_title="折光率", height=350)
st.plotly_chart(fig_cal, use_container_width=True)

# ========== 沸点实验数据 ==========
st.subheader("🔥 沸点及折光率数据")
st.markdown("输入各混合溶液的沸点、液相折光率、气相折光率")

boil_default = pd.DataFrame({
    "沸点(℃)": [80.6, 73.3, 66.4, 65.5, 65.1, 65.5, 66.4, 70.2, 74.5, 78.2],
    "液相折光率": [1.4229, 1.4220, 1.4184, 1.4155, 1.3990, 1.3791, 1.3685, 1.3641, 1.3608, 1.3586],
    "气相折光率": [1.4227, 1.4058, 1.4012, 1.4002, 1.3991, 1.3968, 1.3933, 1.3851, 1.3732, 1.3587]
})
boil_df = st.data_editor(boil_default, num_rows="dynamic", use_container_width=True)

if boil_df.empty:
    st.warning("请至少输入一组数据")
    st.stop()

# ========== 计算按钮 ==========
if st.button("🚀 开始计算", type="primary"):
    # 根据标准曲线计算组成（注意：poly 给出的是折光率→组成？不，我们拟合的是 组成→折光率，需要反函数？实际上应该用组成→折光率，然后由折光率反推组成。为了简单，我们直接拟合 折光率→组成（交换 x,y）或者用插值。这里改用拟合 折光率→组成）
    # 重新拟合：组成作为因变量，折光率作为自变量
    coeffs_inv = np.polyfit(cal_df["折光率"], cal_df["环己烷质量分数"], degree)
    poly_inv = np.poly1d(coeffs_inv)
    
    boil_df['液相组成'] = poly_inv(boil_df["液相折光率"]).clip(0,1)
    boil_df['气相组成'] = poly_inv(boil_df["气相折光率"]).clip(0,1)
    
    # 平滑曲线
    def smooth(x, y, s=0.05):
        mask = ~(np.isnan(x) | np.isnan(y))
        xc, yc = np.array(x)[mask], np.array(y)[mask]
        if len(xc) < 3:
            return xc, yc
        idx = np.argsort(xc)
        xc, yc = xc[idx], yc[idx]
        spl = UnivariateSpline(xc, yc, s=s)
        xs = np.linspace(xc.min(), xc.max(), 300)
        ys = spl(xs)
        return xs, ys
    
    xs_liq, Ts_liq = smooth(boil_df["液相组成"], boil_df["沸点(℃)"], s=0.05)
    xs_vap, Ts_vap = smooth(boil_df["气相组成"], boil_df["沸点(℃)"], s=0.05)
    
    # 找最低恒沸点（从液相线找最低温度点）
    if len(xs_liq) > 0:
        min_idx = np.argmin(Ts_liq)
        azeo_x = xs_liq[min_idx]
        azeo_T = Ts_liq[min_idx]
    else:
        azeo_x, azeo_T = None, None
    
    # 保存到 session_state
    st.session_state.calculated = True
    st.session_state.boil_df = boil_df
    st.session_state.xs_liq = xs_liq
    st.session_state.Ts_liq = Ts_liq
    st.session_state.xs_vap = xs_vap
    st.session_state.Ts_vap = Ts_vap
    st.session_state.azeo_x = azeo_x
    st.session_state.azeo_T = azeo_T
    st.session_state.poly_inv_coeffs = coeffs_inv
    st.rerun()

# ========== 显示结果 ==========
if st.session_state.calculated:
    boil_df = st.session_state.boil_df
    xs_liq = st.session_state.xs_liq
    Ts_liq = st.session_state.Ts_liq
    xs_vap = st.session_state.xs_vap
    Ts_vap = st.session_state.Ts_vap
    azeo_x = st.session_state.azeo_x
    azeo_T = st.session_state.azeo_T
    
    st.subheader("📊 计算结果")
    result_show = boil_df[['沸点(℃)', '液相折光率', '气相折光率', '液相组成', '气相组成']].copy()
    result_show.columns = ['沸点(℃)', '液相折光率', '气相折光率', '液相组成(x)', '气相组成(y)']
    st.dataframe(result_show, use_container_width=True)
    
    if azeo_x is not None:
        st.success(f"**最低恒沸点**：环己烷质量分数 = {azeo_x:.4f}，沸点 = {azeo_T:.2f} ℃")
    else:
        st.warning("无法确定恒沸点（数据点不足）")
    
    # 绘制相图
    fig = go.Figure()
    if len(xs_liq) > 0:
        fig.add_trace(go.Scatter(x=xs_liq, y=Ts_liq, mode='lines', name='液相线', line=dict(color='blue', width=3)))
    if len(xs_vap) > 0:
        fig.add_trace(go.Scatter(x=xs_vap, y=Ts_vap, mode='lines', name='气相线', line=dict(color='red', width=3, dash='dash')))
    fig.add_trace(go.Scatter(x=boil_df["液相组成"], y=boil_df["沸点(℃)"], mode='markers', name='液相实验点', marker=dict(color='blue', size=8)))
    fig.add_trace(go.Scatter(x=boil_df["气相组成"], y=boil_df["沸点(℃)"], mode='markers', name='气相实验点', marker=dict(color='red', size=8)))
    if azeo_x is not None:
        fig.add_trace(go.Scatter(x=[azeo_x], y=[azeo_T], mode='markers', name='最低恒沸点', marker=dict(color='green', size=12, symbol='star')))
    fig.update_layout(xaxis_title="环己烷质量分数", yaxis_title="沸点(℃)", height=500, xaxis_range=[0,1])
    st.plotly_chart(fig, use_container_width=True)
    
    # 导出与打印
    st.subheader("💾 导出与打印")
    col1, col2 = st.columns(2)
    with col1:
        csv = result_show.to_csv(index=False).encode('utf-8')
        st.download_button("📥 下载数据 CSV", csv, "phase_diagram.csv", "text/csv")
    with col2:
        if st.button("🖨️ 生成 PDF 报告"):
            # 生成包含图表的 HTML 并自动打印
            fig_html = fig.to_html(full_html=False, include_plotlyjs='cdn')
            report = f"""
            <html>
            <head><meta charset="UTF-8"><title>二组分相图实验报告</title>
            <style>
                body {{ font-family: 'SimHei', sans-serif; margin: 40px; }}
                table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: center; }}
                th {{ background-color: #f2f2f2; }}
                .chart {{ page-break-inside: avoid; }}
            </style>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            </head>
            <body>
                <h1>二组分体系沸点-组成相图实验报告</h1>
                <p>生成时间：{pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <h2>实验数据</h2>
                {result_show.to_html(index=False)}
                <h2>最低恒沸点</h2>
                <p>环己烷质量分数 = {azeo_x:.4f}，沸点 = {azeo_T:.2f} ℃</p>
                <h2>沸点-组成相图</h2>
                <div class="chart">{fig_html}</div>
                <script>window.onload=()=>window.print();</script>
            </body>
            </html>
            """
            st.components.v1.html(report, height=0, scrolling=False)
            st.success("报告已生成，请在打印对话框中选择「另存为 PDF」")

st.markdown("---")
st.caption("根据折光率-组成标准曲线计算环己烷质量分数，多项式拟合，平滑曲线展示相图。")
