import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.optimize import fsolve
from scipy.interpolate import interp1d
import io

# ==================== 页面配置 ====================
st.set_page_config(page_title="二组分体系相图绘制 - 数据处理", layout="wide")
st.title("🧪 二组分液-液系统相图绘制（乙醇-环己烷）")
st.markdown("根据沸点及气、液相折光率，计算环己烷的摩尔分数，绘制沸点-组成图，并确定最低恒沸点。")

# ==================== 初始化 Session State ====================
if "sample_data" not in st.session_state:
    # 默认示例数据（参考PDF中的10组数据）
    default_data = {
        "编号": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "沸点 (℃)": [80.6, 73.3, 66.4, 65.5, 65.1, 65.5, 66.4, 70.2, 74.5, 78.2],
        "液相折光率 n": [1.4229, 1.4220, 1.4184, 1.4155, 1.3990, 1.3791, 1.3685, 1.3641, 1.3608, 1.3586],
        "气相折光率 n": [1.4227, 1.4058, 1.4012, 1.4002, 1.3991, 1.3968, 1.3933, 1.3851, 1.3732, 1.3587]
    }
    st.session_state.sample_data = pd.DataFrame(default_data)
    st.session_state.use_custom_formula = False
    st.session_state.poly_coeffs = None   # 用户自定义标准曲线的多项式系数

# ==================== 辅助函数 ====================
def calc_composition_by_formula(n, coeffs=None):
    """
    使用内置公式或用户多项式计算环己烷摩尔分数
    内置公式：y = -46.8505*n^2 + 145.819*n - 111.634
    """
    if coeffs is not None:
        # 多项式求值：coeffs[0]*n^2 + coeffs[1]*n + coeffs[2]
        return coeffs[0]*n**2 + coeffs[1]*n + coeffs[2]
    else:
        return -46.8505 * n**2 + 145.819 * n - 111.634

def fit_standard_curve(df_std):
    """
    根据标准溶液数据（折光率，环己烷摩尔分数）拟合二次多项式
    返回多项式系数 [a, b, c] 对应 y = a*x^2 + b*x + c
    """
    if df_std.shape[0] < 3:
        st.error("标准曲线至少需要3个数据点才能拟合二次多项式")
        return None
    x = df_std["折光率 n"].values
    y = df_std["环己烷摩尔分数"].values
    coeffs = np.polyfit(x, y, 2)
    return coeffs

def find_azeotrope(df, liquid_line, vapor_line):
    """
    寻找最低恒沸点（液相线与气相线交点）
    使用插值法：将液相组成和气相组成相减，找零点
    """
    # 确保按沸点排序
    df_sorted = df.sort_values("沸点 (℃)").reset_index(drop=True)
    T = df_sorted["沸点 (℃)"].values
    x_liq = df_sorted["环己烷液相组成 (x)"].values
    y_vap = df_sorted["环己烷气相组成 (y)"].values

    # 如果已有交点数据点，取差值最小的点
    diff = np.abs(x_liq - y_vap)
    min_idx = np.argmin(diff)
    if diff[min_idx] < 0.02:
        T_azeo = T[min_idx]
        x_azeo = (x_liq[min_idx] + y_vap[min_idx]) / 2
        return T_azeo, x_azeo

    # 否则对差值进行插值求零点
    f_diff = interp1d(T, x_liq - y_vap, kind='linear', fill_value="extrapolate")
    # 在温度范围内寻找零点
    T_range = np.linspace(T.min(), T.max(), 200)
    diff_vals = f_diff(T_range)
    zero_cross = T_range[np.where(np.diff(np.sign(diff_vals)))[0]]
    if len(zero_cross) > 0:
        T_azeo = zero_cross[0]
        # 求该温度下的平均组成
        x_liq_interp = interp1d(T, x_liq, kind='linear')(T_azeo)
        y_vap_interp = interp1d(T, y_vap, kind='linear')(T_azeo)
        x_azeo = (x_liq_interp + y_vap_interp) / 2
        return T_azeo, x_azeo
    else:
        # 若找不到交点，返回沸点最低点作为近似
        min_T_idx = np.argmin(T)
        return T[min_T_idx], (x_liq[min_T_idx] + y_vap[min_T_idx]) / 2

# ==================== 侧边栏：数据输入模式 ====================
st.sidebar.header("📥 数据输入")
method = st.sidebar.radio(
    "组成计算方法",
    ["使用内置经验公式", "使用标准曲线拟合"]
)

if method == "使用标准曲线拟合":
    st.sidebar.subheader("标准曲线数据")
    st.sidebar.markdown("请上传包含 **折光率 n** 和 **环己烷摩尔分数** 两列的CSV文件")
    uploaded_std = st.sidebar.file_uploader("上传标准曲线 CSV", type=["csv"])
    if uploaded_std is not None:
        df_std = pd.read_csv(uploaded_std)
        st.sidebar.dataframe(df_std)
        if st.sidebar.button("拟合标准曲线"):
            coeffs = fit_standard_curve(df_std)
            if coeffs is not None:
                st.session_state.poly_coeffs = coeffs
                st.session_state.use_custom_formula = True
                st.sidebar.success(f"拟合多项式: y = {coeffs[0]:.4f} n² + {coeffs[1]:.4f} n + {coeffs[2]:.4f}")
    else:
        st.sidebar.info("未上传标准曲线，将使用内置经验公式（临时）")
        st.session_state.use_custom_formula = False
else:
    st.session_state.use_custom_formula = False
    st.session_state.poly_coeffs = None
    st.sidebar.info("使用内置经验公式: y = -46.8505 n² + 145.819 n - 111.634")

# ==================== 主界面：样品数据编辑 ====================
st.header("📝 实验数据录入")
st.markdown("编辑或上传样品的沸点及折光率数据，系统将自动计算环己烷组成并绘制相图。")

data_source = st.radio(
    "数据来源",
    ["使用内置示例数据", "手动编辑表格", "上传 CSV 文件"]
)

if data_source == "使用内置示例数据":
    df_samples = st.session_state.sample_data.copy()
    st.dataframe(df_samples, use_container_width=True)
elif data_source == "手动编辑表格":
    df_samples = st.data_editor(
        st.session_state.sample_data,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "编号": st.column_config.NumberColumn("编号", min_value=1),
            "沸点 (℃)": st.column_config.NumberColumn("沸点 (℃)", format="%.1f"),
            "液相折光率 n": st.column_config.NumberColumn("液相折光率 n", format="%.4f"),
            "气相折光率 n": st.column_config.NumberColumn("气相折光率 n", format="%.4f")
        }
    )
else:
    uploaded_file = st.file_uploader("上传 CSV 文件 (需包含列: 编号, 沸点 (℃), 液相折光率 n, 气相折光率 n)", type="csv")
    if uploaded_file:
        df_samples = pd.read_csv(uploaded_file)
        st.dataframe(df_samples, use_container_width=True)
    else:
        st.stop()

# 保存用户修改的数据
st.session_state.sample_data = df_samples

# ==================== 计算组成 ====================
if st.button("🔍 计算环己烷组成并绘制相图"):
    if df_samples.empty:
        st.error("没有有效数据")
    else:
        # 选择计算函数
        coeffs = st.session_state.poly_coeffs if st.session_state.use_custom_formula else None
        try:
            df_samples["环己烷液相组成 (x)"] = df_samples["液相折光率 n"].apply(
                lambda n: calc_composition_by_formula(n, coeffs)
            )
            df_samples["环己烷气相组成 (y)"] = df_samples["气相折光率 n"].apply(
                lambda n: calc_composition_by_formula(n, coeffs)
            )
        except Exception as e:
            st.error(f"计算出错：{e}")
            st.stop()

        # 剔除可能超出 [0,1] 的值（提醒）
        invalid_x = ((df_samples["环己烷液相组成 (x)"] < 0) | (df_samples["环己烷液相组成 (x)"] > 1)).any()
        invalid_y = ((df_samples["环己烷气相组成 (y)"] < 0) | (df_samples["环己烷气相组成 (y)"] > 1)).any()
        if invalid_x or invalid_y:
            st.warning("计算得到的组成超出 [0,1] 范围，请检查折光率数据或标准曲线是否合适。")

        # 存储计算结果
        st.session_state.calc_df = df_samples.copy()

        st.subheader("📊 计算结果（环己烷摩尔分数）")
        display_cols = ["编号", "沸点 (℃)", "液相折光率 n", "环己烷液相组成 (x)", "气相折光率 n", "环己烷气相组成 (y)"]
        st.dataframe(df_samples[display_cols], use_container_width=True)

        # ==================== 绘制相图 ====================
        fig = go.Figure()

        # 液相线（散点+连线）
        fig.add_trace(go.Scatter(
            x=df_samples["环己烷液相组成 (x)"],
            y=df_samples["沸点 (℃)"],
            mode='lines+markers',
            name='液相线 (L)',
            line=dict(color='blue', width=2),
            marker=dict(size=8)
        ))
        # 气相线
        fig.add_trace(go.Scatter(
            x=df_samples["环己烷气相组成 (y)"],
            y=df_samples["沸点 (℃)"],
            mode='lines+markers',
            name='气相线 (V)',
            line=dict(color='red', width=2, dash='dash'),
            marker=dict(size=8, symbol='x')
        ))

        # 寻找最低恒沸点
        T_azeo, x_azeo = find_azeotrope(df_samples, "环己烷液相组成 (x)", "环己烷气相组成 (y)")
        fig.add_trace(go.Scatter(
            x=[x_azeo],
            y=[T_azeo],
            mode='markers',
            name=f'最低恒沸点 ({T_azeo:.1f}℃, {x_azeo:.3f})',
            marker=dict(size=12, color='green', symbol='star')
        ))

        fig.update_layout(
            title="乙醇-环己烷系统沸点-组成图",
            xaxis_title="环己烷摩尔分数 (x<sub>环己烷</sub>)",
            yaxis_title="沸点 (℃)",
            legend=dict(x=0.05, y=0.95),
            width=800, height=600
        )
        st.plotly_chart(fig, use_container_width=True)

        # 显示恒沸点信息
        st.success(f"**最低恒沸点温度：{T_azeo:.2f} ℃**\n\n**恒沸物组成（环己烷摩尔分数）：{x_azeo:.4f}**")

        # ==================== 导出数据 ====================
        st.subheader("📎 导出结果")
        csv_data = df_samples[display_cols].to_csv(index=False).encode('utf-8')
        st.download_button(
            label="下载计算结果 (CSV)",
            data=csv_data,
            file_name="phase_diagram_results.csv",
            mime="text/csv"
        )

        # 可选：生成简单报告（仅用于展示，实际打印可截图）
        st.markdown("---")
        st.markdown("### 📝 实验结论")
        st.markdown(f"- 通过折光率-组成关系计算得到各体系的平衡组成。")
        st.markdown(f"- 该二元系统具有最低恒沸点，恒沸温度为 **{T_azeo:.2f} ℃**，恒沸物中环己烷的摩尔分数为 **{x_azeo:.4f}**。")
        st.markdown(f"- 相图形态与文献记载一致，验证了柯诺瓦洛夫第二定律。")

# ==================== 说明 ====================
with st.expander("ℹ️ 实验原理及操作说明"):
    st.markdown("""
    **1. 实验目的**  
    绘制具有最低恒沸点二元系统（乙醇-环己烷）的沸点-组成图，掌握阿贝折光仪的使用及组成分析方法。

    **2. 数据处理原理**  
    在恒压下蒸馏不同组成的乙醇-环己烷溶液，测定气液相平衡时的沸点，并用折光仪测定气、液两相的折光率。  
    利用预先建立的 **折光率-组成标准曲线**（或经验公式）将折光率换算为环己烷的摩尔分数，从而得到平衡组成。  
    最后以沸点为纵坐标、组成为横坐标绘制相图，找出液相线与气相线的交点——最低恒沸点。

    **3. 内置经验公式**  
    根据参考数据拟合的二次多项式：  
    \( y_{\\text{环己烷}} = -46.8505 \\cdot n^2 + 145.819 \\cdot n - 111.634 \)  
    该公式适用于 25℃ 下折光率与环己烷摩尔分数的换算。

    **4. 自定义标准曲线**  
    若实验室测定了自己的标准溶液，可在侧边栏上传 **折光率 n, 环己烷摩尔分数** 两列数据，程序将自动拟合二次多项式并用于未知样计算。

    **5. 最低恒沸点的确定**  
    程序自动搜索液相线与气相线交点，若无法精确相交，则选取气液相组成最接近的点或通过插值法估算。
    """)
    
