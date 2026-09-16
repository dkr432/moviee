import math
import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ------------------------------------------------------------
# 기본 설정
# ------------------------------------------------------------
st.set_page_config(page_title="영화 유형 나누기", page_icon="🎬")
st.title("🎬 영화 유형 나누기")

DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/kobis_movies.csv"

# ------------------------------------------------------------
# 데이터 불러오기
# ------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8")
    return df

raw_df = load_data()

# ------------------------------------------------------------
# 파생 속성 만들기
# ------------------------------------------------------------
def prepare_features(df):
    df = df.copy()

    # 첫 주 관객이 0이거나 필요한 값이 비어 있는 행 제거를 위해
    # 우선 필요한 원본 열을 숫자형으로 변환
    needed_cols = ["first_scrn", "total_audi", "days_in_top10", "first_week_audi"]
    for col in needed_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 결측치 제거 + 첫 주 관객 0인 영화 제거
    df = df.dropna(subset=needed_cols)
    df = df[df["first_week_audi"] != 0]

    # 로그 변환 (상용로그, math.log10 사용)
    # 0 이하 값이 있으면 로그를 계산할 수 없으므로 안전하게 제거
    df = df[(df["first_scrn"] > 0) & (df["total_audi"] > 0)]

    df["log_first_scrn"] = df["first_scrn"].apply(lambda x: math.log10(x))
    df["log_total_audi"] = df["total_audi"].apply(lambda x: math.log10(x))

    # 롱런 지수: 누적 관객 / 첫 주 관객, 20 초과 시 20으로 자르기
    df["longrun_index"] = df["total_audi"] / df["first_week_audi"]
    df["longrun_index"] = df["longrun_index"].clip(upper=20)

    # days_in_top10 은 그대로 사용
    df["top10_days"] = df["days_in_top10"]

    return df

feature_df = prepare_features(raw_df)

# ------------------------------------------------------------
# 전체 편수 / 사용 편수 안내
# ------------------------------------------------------------
st.write(f"전체 영화 편수: {len(raw_df)}편 · 분석에 사용한 편수: {len(feature_df)}편")

# ------------------------------------------------------------
# 속성 선택 (군집화용)
# ------------------------------------------------------------
FEATURE_LABELS = {
    "log_first_scrn": "스크린 수(로그)",
    "log_total_audi": "누적 관객(로그)",
    "top10_days": "10위권 일수",
    "longrun_index": "롱런 지수",
}

st.subheader("군집화에 사용할 속성 선택")
selected_features = st.multiselect(
    "두 개 이상 선택해 주세요.",
    options=list(FEATURE_LABELS.keys()),
    default=list(FEATURE_LABELS.keys()),
    format_func=lambda x: FEATURE_LABELS[x],
)

if len(selected_features) < 2:
    st.warning("속성을 두 개 이상 선택해야 군집화를 진행할 수 있습니다.")
    st.stop()

# ------------------------------------------------------------
# 표준화 + K-평균 (군집 3개, 난수 고정)
# ------------------------------------------------------------
X = feature_df[selected_features].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
raw_labels = kmeans.fit_predict(X_scaled)

feature_df = feature_df.copy()
feature_df["raw_cluster"] = raw_labels

# ------------------------------------------------------------
# 군집 번호를 누적 관객 평균 기준으로 재배열 -> ㉮㉯㉰
# ------------------------------------------------------------
cluster_order = (
    feature_df.groupby("raw_cluster")["total_audi"]
    .mean()
    .sort_values(ascending=False)
    .index.tolist()
)

symbol_map = {}
symbols = ["㉮", "㉯", "㉰"]
for rank, raw_c in enumerate(cluster_order):
    symbol_map[raw_c] = symbols[rank]

feature_df["cluster"] = feature_df["raw_cluster"].map(symbol_map)

# ------------------------------------------------------------
# 2차원 산점도
# ------------------------------------------------------------
st.subheader("2차원 산점도")

col1, col2 = st.columns(2)
with col1:
    x_axis_2d = st.selectbox(
        "가로축", options=selected_features, format_func=lambda x: FEATURE_LABELS[x], key="x2d"
    )
with col2:
    y_axis_2d = st.selectbox(
        "세로축",
        options=selected_features,
        index=min(1, len(selected_features) - 1),
        format_func=lambda x: FEATURE_LABELS[x],
        key="y2d",
    )

fig_2d = px.scatter(
    feature_df,
    x=x_axis_2d,
    y=y_axis_2d,
    color="cluster",
    hover_name="movieNm",
    labels={x_axis_2d: FEATURE_LABELS[x_axis_2d], y_axis_2d: FEATURE_LABELS[y_axis_2d]},
    category_orders={"cluster": symbols},
)
st.plotly_chart(fig_2d, use_container_width=True)

# ------------------------------------------------------------
# 3차원 산점도
# ------------------------------------------------------------
st.subheader("3차원 산점도")

if len(selected_features) < 3:
    st.info("3차원 산점도를 보려면 속성을 세 개 이상 선택해 주세요.")
else:
    col3, col4, col5 = st.columns(3)
    with col3:
        x_axis_3d = st.selectbox(
            "X축", options=selected_features, format_func=lambda x: FEATURE_LABELS[x], key="x3d"
        )
    with col4:
        y_axis_3d = st.selectbox(
            "Y축",
            options=selected_features,
            index=min(1, len(selected_features) - 1),
            format_func=lambda x: FEATURE_LABELS[x],
            key="y3d",
        )
    with col5:
        z_axis_3d = st.selectbox(
            "Z축",
            options=selected_features,
            index=min(2, len(selected_features) - 1),
            format_func=lambda x: FEATURE_LABELS[x],
            key="z3d",
        )

    fig_3d = px.scatter_3d(
        feature_df,
        x=x_axis_3d,
        y=y_axis_3d,
        z=z_axis_3d,
        color="cluster",
        hover_name="movieNm",
        labels={
            x_axis_3d: FEATURE_LABELS[x_axis_3d],
            y_axis_3d: FEATURE_LABELS[y_axis_3d],
            z_axis_3d: FEATURE_LABELS[z_axis_3d],
        },
        category_orders={"cluster": symbols},
    )
    fig_3d.update_traces(marker=dict(size=3))
    st.plotly_chart(fig_3d, use_container_width=True)

# ------------------------------------------------------------
# 묶음별 편수 및 평균 표 (원래 단위)
# ------------------------------------------------------------
st.subheader("묶음별 편수와 평균값 (원래 단위)")

summary_rows = []
for sym in symbols:
    sub = feature_df[feature_df["cluster"] == sym]
    row = {
        "묶음": sym,
        "편수": len(sub),
        "스크린 수 평균": sub["first_scrn"].mean(),
        "누적 관객 평균": sub["total_audi"].mean(),
        "10위권 일수 평균": sub["top10_days"].mean(),
        "롱런 지수 평균": sub["longrun_index"].mean(),
    }
    summary_rows.append(row)

summary_df = pd.DataFrame(summary_rows)
st.dataframe(summary_df, use_container_width=True)

# ------------------------------------------------------------
# 묶음별 누적 관객 상위 5편
# ------------------------------------------------------------
st.subheader("묶음별 누적 관객 상위 5편")

for sym in symbols:
    sub = feature_df[feature_df["cluster"] == sym].sort_values(
        "total_audi", ascending=False
    ).head(5)
    st.markdown(f"**{sym} 묶음**")
    st.write(", ".join(sub["movieNm"].tolist()))
