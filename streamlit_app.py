import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.interpolate import UnivariateSpline

# ==================== 页面配置 ====================
st.set_page_config(page_title="二组分体系相图绘制", layout="wide")
st.title("🧪 二组分体系相图的绘制（乙醇-环己烷）")
st.markdown("根据各混合溶液的沸点及气液相折光率，计算环己烷组成，绘制平滑的沸点-组成相图，确定最低恒沸点。")

# ==================== 初始化 session_state ====================
if 'calculated' not in st.session_state:
    st.session_state.calculated = False
if 'result_df' not in st.session_state:
    st.session_state.result_df = None
if 'azeo_x' not in st.session_state:
    st.session_state.azeo_x = None
if 'azeo_T' not in st.session_state:
    st.session_state.azeo_T = None
if 'fig_html' not in st.session_state:
    st.session_state.fig_html = None

# ==================== 辅助函数 ====================
def calc_composition(refractive_index):
    """根据折光率 n 计算环己烷的摩尔分数"""
    return -46.8505 * refractive_index**2 + 145.819 * refractive_index - 111.634

def smooth_curve(x, y, smoothing=0.05, num_points=500):
    """样条插值生成平滑曲线"""
    mask = ~(np.isnan(x) | np.isnan(y))
    x_clean = np.array(x)[mask]
    y_clean = np.array(y)[mask]
    if len(x_clean) < 3:
        return x_clean, y_clean
    sort_idx = np.argsort(x_clean)
    x_sorted = x_clean[sort_idx]
    y_sorted = y_clean[sort_idx]
    spl = UnivariateSpline(x_sorted, y_sorted, s=smoothing)
    x_smooth = np.linspace(x_sorted.min(), x_sorted.max(), num_points)
    y_smooth = spl(x_smooth)
    return x_smooth, y_smooth

def find_azeotrope(x_smooth, y_smooth):
    """从平滑曲线中找最低点"""
    min_idx = np.argmin(y_smooth)
    return x_smooth[min_idx], y_smooth[min_idx]

# ==================== 数据输入 ====================
st.subheader("📥 实验数据输入")
data_source = st.radio("数据来源", ["使用示例数据", "手动编辑表格", "上传 CSV 文件"])

# 示例数据
example_data = pd.DataFrame({
    "组别": [1,2,3,4,5,6,7,8,9,10],
    "环己烷:乙醇": ["99.4:0.6", "98:2", "92:8", "86.8:13.2", "70:30", "42.5:57.5", "35:65", "17:83", "6:94", "0.2:99.8"],
    "沸点(℃)": [80.6, 73.3, 66.4, 65.5, 65.1, 65.5, 66.4, 70.2, 74.5, 78.2],
    "液相折光率": [1.4229, 1.4220, 1.4184, 1.4155, 1.3990, 1.3791, 1.3685, 1.3641, 1.3608, 1.3586],
    "气相折光率": [1.4227, 1.4058, 1.4012, 1.4002, 1.3991, 1.3968, 1.3933, 1.3851, 1.3732, 1.3587]
})

if data_source == "使用示例数据":
    df = example_data.copy()
    st.dataframe(df, use_container_width=True)
elif data_source == "手动编辑表格":
    df = st.data_editor(example_data, num_rows="dynamic", use_container_width=True)
else:
    uploaded = st.file_uploader("上传 CSV 文件", type="csv")
    if uploaded:
        df = pd.read_csv(uploaded)
        st.dataframe(df, use_container_width=True)
    else:
        st.stop()

# 检查必要列
required_cols = ['沸点(℃)', '液相折光率', '气相折光率']
if not all(col in df.columns for col in required_cols):
    st.error("数据表缺少必要列：沸点(℃), 液相折光率, 气相折光率")
    st.stop()

# 平滑参数调节
st.sidebar.subheader("📈 曲线平滑参数")
smoothing_factor = st.sidebar.slider("平滑因子 (s)", 0.0, 1.0, 0.05, 0.01,
                                     help="值越小曲线越接近原始点，越大越平滑")

# ==================== 计算按钮 ====================
if st.button("🚀 开始计算", type="primary"):
    # 计算组成
    df['x'] = df['液相折光率'].apply(calc_composition)
    df['y'] = df['气相折光率'].apply(calc_composition)
    
    # 生成平滑曲线
    x_smooth_liq, T_smooth_liq = smooth_curve(df['x'], df['沸点(℃)'], smoothing=smoothing_factor)
    x_smooth_vap, T_smooth_vap = smooth_curve(df['y'], df['沸点(℃)'], smoothing=smoothing_factor)
    
    # 寻找恒沸点
    if len(x_smooth_liq) > 0:
        azeo_x, azeo_T = find_azeotrope(x_smooth_liq, T_smooth_liq)
    else:
        azeo_x, azeo_T = None, None
    
    # 存储计算结果
    st.session_state.result_df = df.copy()
    st.session_state.azeo_x = azeo_x
    st.session_state.azeo_T = azeo_T
    st.session_state.x_smooth_liq = x_smooth_liq
    st.session_state.T_smooth_liq = T_smooth_liq
    st.session_state.x_smooth_vap = x_smooth_vap
    st.session_state.T_smooth_vap = T_smooth_vap
    st.session_state.calculated = True
    
    st.success("计算完成！")
    st.rerun()

# ==================== 显示结果（如果已计算）====================
if st.session_state.calculated:
    df = st.session_state.result_df
    azeo_x = st.session_state.azeo_x
    azeo_T = st.session_state.azeo_T
    x_smooth_liq = st.session_state.x_smooth_liq
    T_smooth_liq = st.session_state.T_smooth_liq
    x_smooth_vap = st.session_state.x_smooth_vap
    T_smooth_vap = st.session_state.T_smooth_vap
    
    st.subheader("📊 详细计算结果")
    # 显示计算表格
    display_df = df[['组别', '环己烷:乙醇', '沸点(℃)', '液相折光率', '气相折光率', 'x', 'y']].copy()
    display_df.columns = ['组别', '配比', '沸点(℃)', '液相折光率', '气相折光率', '液相组成 x (环己烷)', '气相组成 y (环己烷)']
    st.dataframe(display_df, use_container_width=True)
    
    # 统计信息
    st.subheader("📈 拟合结果与恒沸点")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("最低恒沸点温度", f"{azeo_T:.2f} ℃" if azeo_T else "无法计算")
    with col2:
        st.metric("恒沸点组成 (环己烷摩尔分数)", f"{azeo_x:.4f}" if azeo_x else "无法计算")
    
    # 绘图
    fig = go.Figure()
    # 液相平滑曲线
    fig.add_trace(go.Scatter(
        x=x_smooth_liq, y=T_smooth_liq,
        mode='lines', name='液相线 (拟合曲线)',
        line=dict(color='blue', width=3)
    ))
    # 气相平滑曲线
    fig.add_trace(go.Scatter(
        x=x_smooth_vap, y=T_smooth_vap,
        mode='lines', name='气相线 (拟合曲线)',
        line=dict(color='red', width=3, dash='dash')
    ))
    # 原始实验点
    fig.add_trace(go.Scatter(
        x=df['x'], y=df['沸点(℃)'],
        mode='markers', name='液相实验点',
        marker=dict(color='blue', size=8, symbol='circle')
    ))
    fig.add_trace(go.Scatter(
        x=df['y'], y=df['沸点(℃)'],
        mode='markers', name='气相实验点',
        marker=dict(color='red', size=8, symbol='square')
    ))
    # 恒沸点标记
    if azeo_x is not None:
        fig.add_trace(go.Scatter(
            x=[azeo_x], y=[azeo_T],
            mode='markers', name=f'最低恒沸点 ({azeo_x:.3f}, {azeo_T:.1f}℃)',
            marker=dict(color='green', size=14, symbol='star')
        ))
    fig.update_layout(
        xaxis_title="环己烷摩尔分数",
        yaxis_title="沸点 (℃)",
        legend=dict(x=0.05, y=0.95),
        height=600
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # 保存图表 HTML 供打印
    st.session_state.fig_html = fig.to_html(full_html=False, include_plotlyjs='cdn')
    
    # 导出与打印
    st.subheader("💾 导出与打印")
    col1, col2 = st.columns(2)
    with col1:
        csv_data = display_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 下载数据 CSV", csv_data, "phase_diagram_data.csv", "text/csv")
    with col2:
        if st.button("🖨️ 生成 PDF 报告"):
            report_html = f"""
            <html>
            <head>
                <meta charset="UTF-8">
                <title>乙醇-环己烷二组分体系相图报告</title>
                <style>
                    body {{ font-family: 'SimHei', 'Microsoft YaHei', Arial, sans-serif; margin: 40px; }}
                    h1 {{ color: #2c3e50; }}
                    h2 {{ color: #34495e; border-bottom: 1px solid #ddd; }}
                    table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: center; }}
                    th {{ background-color: #f2f2f2; }}
                    .chart {{ margin: 30px 0; page-break-inside: avoid; break-inside: avoid; }}
                </style>
                <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            </head>
            <body>
                <h1>乙醇-环己烷二组分体系相图实验报告</h1>
                <p>生成时间：{pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <h2>1. 实验数据与计算结果</h2>
                {display_df.to_html(index=False)}
                <h2>2. 最低恒沸点</h2>
                <p>环己烷摩尔分数 = {azeo_x:.4f}，沸点 = {azeo_T:.2f} ℃</p>
                <h2>3. 沸点-组成相图（平滑曲线）</h2>
                <div class="chart">{st.session_state.fig_html}</div>
                <script>
                    window.onload = function() {{ window.print(); }};
                </script>
            </body>
            </html>
            """
            st.components.v1.html(report_html, height=0, scrolling=False)
            st.success("报告已生成，请在弹出的打印对话框中选择「另存为 PDF」")
else:
    st.info("👆 请输入或上传数据后，点击「开始计算」按钮查看结果")

st.markdown("---")
st.caption("组成计算公式：Y_环 = -46.8505·n² + 145.819·n - 111.634 (n为折光率)")
