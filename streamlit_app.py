import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.interpolate import UnivariateSpline
from scipy.optimize import curve_fit

# ==================== 页面配置 ====================
st.set_page_config(page_title="乙醇-环己烷沸点组成图绘制", layout="wide")
st.title("🧪 具有最低恒沸点二元系统的沸点组成图绘制")
st.markdown("**参考教材：** 《物理化学实验 新世纪第四版》实验四 · 乙醇-环己烷体系")

# ==================== 初始化 session_state ====================
if 'calculated' not in st.session_state:
    st.session_state.calculated = False
if 'result_df' not in st.session_state:
    st.session_state.result_df = None
if 'calibration_df' not in st.session_state:
    st.session_state.calibration_df = None
if 'azeo_x' not in st.session_state:
    st.session_state.azeo_x = None
if 'azeo_T' not in st.session_state:
    st.session_state.azeo_T = None
if 'fig_html' not in st.session_state:
    st.session_state.fig_html = None

# ==================== 辅助函数 ====================
def fit_calibration_curve(x_data, y_data, degree=3):
    """多项式拟合折光率-组成标准曲线"""
    coeffs = np.polyfit(x_data, y_data, degree)
    poly = np.poly1d(coeffs)
    return poly, coeffs

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

# ==================== 侧边栏：实验说明 ====================
with st.sidebar:
    st.header("📖 实验说明")
    with st.expander("实验原理", expanded=True):
        st.markdown("""
        乙醇-环己烷二元系统对拉乌尔定律有较大正偏差，其沸点-组成曲线图上出现最低恒沸点。
        
        **柯诺瓦洛夫第二定律：** 二元系统处于恒沸点时，其气相组成与液相组成相同。
        
        本实验通过测定不同组成溶液的气相和液相折光率，从折光率-组成标准曲线上找出相应的组成，绘制沸点-组成曲线图，确定最低恒沸点。
        """)
    with st.expander("实验步骤", expanded=False):
        st.markdown("""
        1. 配制10个乙醇-环己烷标准溶液（纯乙醇、纯环己烷及8个不同比例）
        2. 用阿贝折光仪测定25℃时各标准溶液的折光率
        3. 在蒸馏瓶中加入标准溶液，加热至沸腾
        4. 温度恒定后记录沸点温度
        5. 分别取气相冷凝液和液相混合液测定折光率（各测三次取平均值）
        6. 按上述步骤测定各组混合液
        """)
    with st.expander("数据处理方法", expanded=False):
        st.markdown("""
        1. 绘制折光率-组成标准曲线
        2. 从标准曲线中查找各次蒸馏中气相与液相的组成
        3. 绘制乙醇-环己烷沸点-组成图
        4. 找出最低恒沸点的温度及组成
        """)

# ==================== Tab 1: 标准曲线（折光率-组成） ====================
st.subheader("📊 第一步：折光率-组成标准曲线")

st.markdown("**表2-5 乙醇-环己烷标准曲线测定**")

calibration_data = {
    "样品编号": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "乙醇:环己烷": ["99.4:0.6", "98:2", "92:8", "86.8:13.2", "70:30", "42.5:57.5", "35:65", "17:83", "6:94", "0.2:99.8"],
    "环己烷质量分数": [0.006, 0.02, 0.08, 0.132, 0.30, 0.575, 0.65, 0.83, 0.94, 0.998],
    "折光率 (25℃)": [1.4229, 1.4220, 1.4184, 1.4155, 1.3990, 1.3791, 1.3685, 1.3641, 1.3608, 1.3586]
}
calibration_df_default = pd.DataFrame(calibration_data)

# 数据输入方式选择
cal_data_source = st.radio("标准曲线数据来源", ["使用示例数据", "手动编辑表格", "上传 CSV 文件"], key="cal")
if cal_data_source == "使用示例数据":
    calibration_df = calibration_df_default.copy()
    st.dataframe(calibration_df, use_container_width=True)
elif cal_data_source == "手动编辑表格":
    calibration_df = st.data_editor(calibration_df_default, num_rows="dynamic", use_container_width=True)
else:
    cal_uploaded = st.file_uploader("上传 CSV (列名: 环己烷质量分数, 折光率)", type="csv")
    if cal_uploaded:
        calibration_df = pd.read_csv(cal_uploaded)
        st.dataframe(calibration_df, use_container_width=True)
    else:
        st.stop()

# 多项式拟合标准曲线
st.markdown("**折光率-组成标准曲线拟合**")
fit_degree = st.slider("拟合多项式次数", min_value=1, max_value=5, value=3, key="fit_deg")

# 拟合
x_cal = calibration_df["环己烷质量分数"].values
y_cal = calibration_df["折光率 (25℃)"].values
poly, coeffs = fit_calibration_curve(x_cal, y_cal, degree=fit_degree)

# 显示拟合公式
coeff_str = " + ".join([f"{coeff:.6f}·x^{len(coeffs)-1-i}" for i, coeff in enumerate(coeffs)])
st.info(f"拟合公式: **n = {coeff_str.replace('x^1', 'x').replace('x^0', '')}**")

# 绘制标准曲线
x_cal_fine = np.linspace(0, 1, 200)
y_cal_fine = poly(x_cal_fine)

fig_cal = go.Figure()
fig_cal.add_trace(go.Scatter(
    x=calibration_df["环己烷质量分数"], y=calibration_df["折光率 (25℃)"],
    mode='markers', name='实验数据点',
    marker=dict(color='blue', size=8, symbol='circle')
))
fig_cal.add_trace(go.Scatter(
    x=x_cal_fine, y=y_cal_fine,
    mode='lines', name=f'拟合曲线 (次数={fit_degree})',
    line=dict(color='red', width=2)
))
fig_cal.update_layout(
    xaxis_title="环己烷质量分数",
    yaxis_title="折光率 (25℃)",
    legend=dict(x=0.05, y=0.95),
    height=400
)
st.plotly_chart(fig_cal, use_container_width=True)

# ==================== Tab 2: 沸点与组成数据 ====================
st.subheader("📊 第二步：沸点及气液相组成测定")

st.markdown("**表2-6 乙醇-环己烷系统的沸点及气、液相组成测定**")

# 示例数据（来源于教材典型数据）
boiling_data = {
    "组别": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "乙醇:环己烷": ["99.4:0.6", "98:2", "92:8", "86.8:13.2", "70:30", "42.5:57.5", "35:65", "17:83", "6:94", "0.2:99.8"],
    "沸点(℃)": [80.6, 73.3, 66.4, 65.5, 65.1, 65.5, 66.4, 70.2, 74.5, 78.2],
    "液相折光率": [1.4229, 1.4220, 1.4184, 1.4155, 1.3990, 1.3791, 1.3685, 1.3641, 1.3608, 1.3586],
    "气相折光率": [1.4227, 1.4058, 1.4012, 1.4002, 1.3991, 1.3968, 1.3933, 1.3851, 1.3732, 1.3587]
}
boiling_df_default = pd.DataFrame(boiling_data)

boil_data_source = st.radio("沸点组成数据来源", ["使用示例数据", "手动编辑表格", "上传 CSV 文件"], key="boil")
if boil_data_source == "使用示例数据":
    boiling_df = boiling_df_default.copy()
    st.dataframe(boiling_df, use_container_width=True)
elif boil_data_source == "手动编辑表格":
    boiling_df = st.data_editor(boiling_df_default, num_rows="dynamic", use_container_width=True)
else:
    boil_uploaded = st.file_uploader("上传 CSV (列名: 沸点, 液相折光率, 气相折光率)", type="csv")
    if boil_uploaded:
        boiling_df = pd.read_csv(boil_uploaded)
        st.dataframe(boiling_df, use_container_width=True)
    else:
        st.stop()

# ==================== 计算按钮 ====================
if st.button("🚀 开始计算并绘制相图", type="primary"):
    # 使用标准曲线拟合多项式计算液相和气相组成
    boiling_df['液相组成_x'] = poly(boiling_df["液相折光率"].values)
    boiling_df['气相组成_y'] = poly(boiling_df["气相折光率"].values)
    
    # 确保组成在有效范围内
    boiling_df['液相组成_x'] = boiling_df['液相组成_x'].clip(0, 1)
    boiling_df['气相组成_y'] = boiling_df['气相组成_y'].clip(0, 1)
    
    # 生成平滑曲线
    x_smooth_liq, T_smooth_liq = smooth_curve(boiling_df['液相组成_x'], boiling_df['沸点(℃)'], smoothing=0.05)
    x_smooth_vap, T_smooth_vap = smooth_curve(boiling_df['气相组成_y'], boiling_df['沸点(℃)'], smoothing=0.05)
    
    # 寻找最低恒沸点
    if len(x_smooth_liq) > 0:
        azeo_x, azeo_T = find_azeotrope(x_smooth_liq, T_smooth_liq)
    else:
        azeo_x, azeo_T = None, None
    
    # 准备结果表格（教材表2-6格式）
    result_df = boiling_df[['组别', '乙醇:环己烷', '沸点(℃)', '液相折光率', '气相折光率', '液相组成_x', '气相组成_y']].copy()
    result_df.columns = ['组别', '配比', '沸点(℃)', '液相折光率', '气相折光率', '液相组成 x (环己烷)', '气相组成 y (环己烷)']
    
    # 存储到 session_state
    st.session_state.calculated = True
    st.session_state.result_df = result_df
    st.session_state.calibration_df = calibration_df
    st.session_state.azeo_x = azeo_x
    st.session_state.azeo_T = azeo_T
    st.session_state.x_smooth_liq = x_smooth_liq
    st.session_state.T_smooth_liq = T_smooth_liq
    st.session_state.x_smooth_vap = x_smooth_vap
    st.session_state.T_smooth_vap = T_smooth_vap
    st.session_state.poly_coeffs = coeffs
    
    st.success("计算完成！")
    st.rerun()

# ==================== 显示计算结果（如果已计算）====================
if st.session_state.calculated:
    result_df = st.session_state.result_df
    azeo_x = st.session_state.azeo_x
    azeo_T = st.session_state.azeo_T
    x_smooth_liq = st.session_state.x_smooth_liq
    T_smooth_liq = st.session_state.T_smooth_liq
    x_smooth_vap = st.session_state.x_smooth_vap
    T_smooth_vap = st.session_state.T_smooth_vap
    
    # 显示计算结果表格
    st.subheader("📊 计算结果")
    st.markdown("**表2-6 乙醇-环己烷系统的沸点及气、液相组成测定（含计算组成）**")
    st.dataframe(result_df, use_container_width=True)
    
    # 统计信息
    st.subheader("📈 最低恒沸点")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("最低恒沸点温度", f"{azeo_T:.2f} ℃" if azeo_T else "无法计算")
    with col2:
        st.metric("恒沸点组成 (环己烷质量分数)", f"{azeo_x:.4f}" if azeo_x else "无法计算")
    
    # 绘制沸点-组成相图
    st.subheader("📈 乙醇-环己烷体系沸点-组成图")
    
    fig = go.Figure()
    
    # 液相线（平滑曲线）
    fig.add_trace(go.Scatter(
        x=x_smooth_liq, y=T_smooth_liq,
        mode='lines', name='液相线',
        line=dict(color='blue', width=3)
    ))
    
    # 气相线（平滑曲线）
    fig.add_trace(go.Scatter(
        x=x_smooth_vap, y=T_smooth_vap,
        mode='lines', name='气相线',
        line=dict(color='red', width=3, dash='dash')
    ))
    
    # 液相实验点
    fig.add_trace(go.Scatter(
        x=result_df['液相组成 x (环己烷)'], y=result_df['沸点(℃)'],
        mode='markers', name='液相实验点',
        marker=dict(color='blue', size=8, symbol='circle')
    ))
    
    # 气相实验点
    fig.add_trace(go.Scatter(
        x=result_df['气相组成 y (环己烷)'], y=result_df['沸点(℃)'],
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
        # 添加恒沸点垂线
        fig.add_vline(x=azeo_x, line_dash="dot", line_color="gray", opacity=0.5)
        fig.add_hline(y=azeo_T, line_dash="dot", line_color="gray", opacity=0.5)
    
    fig.update_layout(
        xaxis_title="环己烷质量分数",
        yaxis_title="沸点 (℃)",
        legend=dict(x=0.05, y=0.95),
        height=500,
        xaxis_range=[0, 1]
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # 保存图表 HTML 供打印
    st.session_state.fig_html = fig.to_html(full_html=False, include_plotlyjs='cdn')
    
    # 折光率-组成标准曲线再次显示
    st.subheader("📈 折光率-组成标准曲线")
    fig_cal = go.Figure()
    fig_cal.add_trace(go.Scatter(
        x=st.session_state.calibration_df["环己烷质量分数"], 
        y=st.session_state.calibration_df["折光率 (25℃)"],
        mode='markers', name='实验数据点',
        marker=dict(color='blue', size=8)
    ))
    x_cal_fine = np.linspace(0, 1, 200)
    y_cal_fine = np.poly1d(st.session_state.poly_coeffs)(x_cal_fine)
    fig_cal.add_trace(go.Scatter(
        x=x_cal_fine, y=y_cal_fine,
        mode='lines', name='拟合曲线',
        line=dict(color='red', width=2)
    ))
    fig_cal.update_layout(
        xaxis_title="环己烷质量分数",
        yaxis_title="折光率 (25℃)",
        height=400
    )
    st.plotly_chart(fig_cal, use_container_width=True)
    
    # ==================== 导出与打印 ====================
    st.subheader("💾 导出与打印")
    col1, col2, col3 = st.columns(3)
    with col1:
        csv_data = result_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 下载数据 CSV", csv_data, "phase_diagram_data.csv", "text/csv")
    with col2:
        # 生成完整实验报告 HTML
        if st.button("🖨️ 生成 PDF 报告"):
            report_html = f"""
            <html>
            <head>
                <meta charset="UTF-8">
                <title>乙醇-环己烷沸点组成图实验报告</title>
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
                <h1>乙醇-环己烷二组分体系沸点组成图实验报告</h1>
                <p><strong>教材：</strong> 物理化学实验 新世纪第四版 实验四</p>
                <p><strong>生成时间：</strong> {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                
                <h2>1. 折光率-组成标准曲线</h2>
                <p>拟合公式: n = {coeff_str.replace('x^', 'x<sup>').replace(' ', '')}</p>
                <div class="chart">{fig_cal.to_html(full_html=False, include_plotlyjs='cdn')}</div>
                
                <h2>2. 沸点及气液相组成测定数据</h2>
                {result_df.to_html(index=False)}
                
                <h2>3. 最低恒沸点</h2>
                <p>环己烷质量分数 = {azeo_x:.4f}，沸点 = {azeo_T:.2f} ℃</p>
                
                <h2>4. 乙醇-环己烷体系沸点-组成图</h2>
                <div class="chart">{st.session_state.fig_html}</div>
                
                <script>
                    window.onload = function() {{ window.print(); }};
                </script>
            </body>
            </html>
            """
            st.components.v1.html(report_html, height=0, scrolling=False)
            st.success("报告已生成，请在弹出的打印对话框中选择「另存为 PDF」")
    with col3:
        # 显示组成计算公式
        with st.expander("🔍 查看组成计算公式"):
            st.markdown(f"""
            **折光率 n → 环己烷质量分数 x 的计算公式：**
            x = {coeffs[0]:.6f}·n^{len(coeffs)-1} {'+' if coeffs[1]>=0 else '-'} {abs(coeffs[1]):.6f}·n^{len(coeffs)-2} {'+' if coeffs[2]>=0 else '-'} {abs(coeffs[2]):.6f}·n^{len(coeffs)-3} ...
            
**使用方法：** 将实验测得的液相或气相折光率 n 代入上述多项式，即可计算对应的环己烷质量分数组成。
""")

st.markdown("---")
st.caption("📚 数据处理依据：《物理化学实验 新世纪第四版》实验四 · 具有最低恒沸点二元系统的沸点组成图绘制")
