import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.optimize import curve_fit
from scipy.interpolate import UnivariateSpline

# ==================== 页面配置 ====================
st.set_page_config(page_title="二组分体系相图绘制", layout="wide")
st.title("🧪 二组分体系相图的绘制（乙醇-环己烷）")
st.markdown("根据各混合溶液的沸点及气液相折光率，计算环己烷组成，绘制沸点-组成相图，确定最低恒沸点。")

# ==================== 辅助函数 ====================
def calc_composition(refractive_index):
    """
    根据折光率 n 计算环己烷的摩尔分数
    公式: Y_环 = -46.8505 * n^2 + 145.819 * n - 111.634
    """
    return -46.8505 * refractive_index**2 + 145.819 * refractive_index - 111.634

def find_azeotrope(df, poly_deg=5):
    """
    从数据中找出最低恒沸点（温度最低点）
    df: 包含 'x' (液相组成) 和 'boiling_point' 的 DataFrame
    返回: (azeo_composition, azeo_temperature)
    """
    # 使用多项式拟合液相线
    x_vals = df['x'].values
    y_vals = df['boiling_point'].values
    # 按组成排序
    sort_idx = np.argsort(x_vals)
    x_sorted = x_vals[sort_idx]
    y_sorted = y_vals[sort_idx]
    # 多项式拟合
    coeffs = np.polyfit(x_sorted, y_sorted, deg=poly_deg)
    poly = np.poly1d(coeffs)
    # 在更密的点上求最小值
    x_fine = np.linspace(x_sorted.min(), x_sorted.max(), 500)
    y_fine = poly(x_fine)
    min_idx = np.argmin(y_fine)
    azeo_x = x_fine[min_idx]
    azeo_y = y_fine[min_idx]
    return azeo_x, azeo_y

# ==================== 数据输入 ====================
st.subheader("📥 实验数据输入")
data_source = st.radio("数据来源", ["使用示例数据", "手动编辑表格", "上传 CSV 文件"])

# 示例数据（从PDF中整理）
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

# ==================== 计算环己烷组成 ====================
df['x'] = df['液相折光率'].apply(calc_composition)
df['y'] = df['气相折光率'].apply(calc_composition)

# 显示计算结果
st.subheader("📊 组成计算结果")
result_display = df[['组别', '环己烷:乙醇', '沸点(℃)', '液相折光率', '气相折光率', 'x', 'y']].copy()
result_display.columns = ['组别', '配比', '沸点(℃)', '液相折光率', '气相折光率', '液相组成 x (环己烷摩尔分数)', '气相组成 y (环己烷摩尔分数)']
st.dataframe(result_display, use_container_width=True)

# ==================== 绘图 ====================
st.subheader("📈 乙醇-环己烷体系沸点-组成相图")

fig = go.Figure()

# 添加液相线（散点+连线）
fig.add_trace(go.Scatter(
    x=df['x'], y=df['沸点(℃)'],
    mode='markers+lines',
    name='液相线 (沸点-液相组成)',
    marker=dict(color='blue', size=8),
    line=dict(color='blue', width=2)
))

# 添加气相线（散点+连线）
fig.add_trace(go.Scatter(
    x=df['y'], y=df['沸点(℃)'],
    mode='markers+lines',
    name='气相线 (沸点-气相组成)',
    marker=dict(color='red', size=8),
    line=dict(color='red', width=2, dash='dash')
))

# 找出最低恒沸点
if len(df) >= 3:
    azeo_x, azeo_T = find_azeotrope(df)
    fig.add_trace(go.Scatter(
        x=[azeo_x], y=[azeo_T],
        mode='markers',
        name=f'最低恒沸点 ({azeo_x:.3f}, {azeo_T:.1f}℃)',
        marker=dict(color='green', size=12, symbol='star')
    ))
    st.success(f"🔍 最低恒沸点：环己烷摩尔分数 = {azeo_x:.4f}，沸点 = {azeo_T:.2f} ℃")
else:
    st.warning("数据点不足，无法自动计算恒沸点")

# 图表布局
fig.update_layout(
    xaxis_title="环己烷摩尔分数",
    yaxis_title="沸点 (℃)",
    legend=dict(x=0.05, y=0.95),
    width=None,
    height=600
)
st.plotly_chart(fig, use_container_width=True)

# 可选：显示原始数据点表格
with st.expander("查看原始数据及计算结果明细"):
    st.dataframe(result_display, use_container_width=True)

# ==================== 导出与报告 ====================
st.subheader("💾 导出与打印")
col1, col2 = st.columns(2)
with col1:
    csv_data = result_display.to_csv(index=False).encode('utf-8')
    st.download_button("📥 下载数据 CSV", csv_data, "phase_diagram_data.csv", "text/csv")
with col2:
    if st.button("🖨️ 生成 PDF 报告"):
        # 生成报告 HTML
        report_html = f"""
        <html>
        <head>
            <meta charset="UTF-8">
            <title>二组分体系相图实验报告</title>
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
            {result_display.to_html(index=False)}
            <h2>2. 沸点-组成相图</h2>
            <div class="chart">{fig.to_html(full_html=False, include_plotlyjs='cdn')}</div>
            <h2>3. 最低恒沸点</h2>
            <p>环己烷摩尔分数 = {azeo_x:.4f}，沸点 = {azeo_T:.2f} ℃</p>
            <script>
                window.onload = function() {{ window.print(); }};
            </script>
        </body>
        </html>
        """
        st.components.v1.html(report_html, height=0, scrolling=False)
        st.success("报告已生成，请在弹出的打印对话框中选择「另存为 PDF」")

st.markdown("---")
st.caption("组成计算公式：Y_环 = -46.8505·n² + 145.819·n - 111.634 (n为折光率)")
