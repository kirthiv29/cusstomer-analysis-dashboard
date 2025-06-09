import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Page setup
st.set_page_config(page_title="📊 Customer Analysis Dashboard", layout="wide")
st.title("📊 Fully Customizable Customer Analysis Dashboard")

# File uploader
uploaded_file = st.file_uploader("📂 Upload your customer CSV or TSV file", type=["csv", "txt"])

if uploaded_file:
    # Try reading as CSV, fallback to TSV
    try:
        df = pd.read_csv(uploaded_file)
    except:
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, sep="\t")

    # Clean column names
    df.columns = df.columns.str.strip()

    st.subheader("📄 Raw Data Preview")
    st.dataframe(df.head())

    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    non_numeric_cols = df.select_dtypes(exclude='number').columns.tolist()

    st.sidebar.header("🔧 Configuration Panel")

    # Cluster columns
    cluster_cols = st.sidebar.multiselect(
        "🧠 Select columns for clustering (numeric only)",
        options=numeric_cols,
        default=numeric_cols[:3]
    )

    # Handling missing values
    missing_strategy = st.sidebar.radio(
        "🩹 How to handle missing values in clustering?",
        options=["Drop rows", "Fill with mean"],
        index=0
    )

    # Number of clusters
    cluster_k = st.sidebar.slider("🔢 Number of Clusters (KMeans)", min_value=2, max_value=10, value=3)

    # Visualization columns
    vis_cols = st.sidebar.multiselect(
        "📊 Select columns for visualization",
        options=df.columns,
        default=numeric_cols[:2]
    )

    # Clustering block
    if cluster_cols:
        st.subheader("🧠 KMeans Clustering")

        try:
            # Handle missing values
            if missing_strategy == "Drop rows":
                df_cluster = df[cluster_cols].dropna()
            else:
                df_cluster = df[cluster_cols].fillna(df[cluster_cols].mean())

            # Standardize
            scaler = StandardScaler()
            scaled = scaler.fit_transform(df_cluster)

            # Fit KMeans
            kmeans = KMeans(n_clusters=cluster_k, random_state=42, n_init='auto')
            clusters = kmeans.fit_predict(scaled)

            # Add cluster results back to original DataFrame
            df.loc[df_cluster.index, 'Cluster'] = clusters

            st.success(f"✅ Clustering done on: {', '.join(cluster_cols)}")
            st.write("🔢 Cluster Distribution:")
            st.bar_chart(df['Cluster'].value_counts())
            st.dataframe(df.head())

        except Exception as e:
            st.error(f"❌ Error during clustering: {e}")

    # Visualizations
    st.markdown("---")
    st.subheader("📊 Visualizations")

    selected_charts = st.sidebar.multiselect(
        "Choose visualizations to display (max 5)",
        ["Histogram", "Box Plot", "Scatter Plot", "Pair Plot", "Heatmap"],
        default=["Histogram", "Scatter Plot"]
    )

    # Histogram
    if "Histogram" in selected_charts and numeric_cols:
        col = st.selectbox("📌 Select column for Histogram", numeric_cols, key='hist_col')
        if col in df.columns:
            fig, ax = plt.subplots()
            sns.histplot(df[col].dropna(), kde=True, ax=ax)
            st.pyplot(fig)

    # Box Plot
    if "Box Plot" in selected_charts and numeric_cols and non_numeric_cols:
        y = st.selectbox("📌 Y-axis (numeric) for Box Plot", numeric_cols, key='box_y')
        x = st.selectbox("📌 X-axis (category) for Box Plot", non_numeric_cols, key='box_x')
        fig, ax = plt.subplots()
        sns.boxplot(data=df, x=x, y=y, ax=ax)
        st.pyplot(fig)

    # Scatter Plot
    if "Scatter Plot" in selected_charts and len(numeric_cols) >= 2:
        x_scatter = st.selectbox("📌 X-axis (Scatter)", numeric_cols, key='scatter_x')
        y_scatter = st.selectbox("📌 Y-axis (Scatter)", numeric_cols, key='scatter_y')
        hue = st.selectbox("📌 Hue (optional)", ["None"] + non_numeric_cols + (['Cluster'] if 'Cluster' in df.columns else []), key='scatter_hue')
        fig, ax = plt.subplots()
        if hue != "None" and hue in df.columns:
            sns.scatterplot(data=df, x=x_scatter, y=y_scatter, hue=df[hue], ax=ax)
        else:
            sns.scatterplot(data=df, x=x_scatter, y=y_scatter, ax=ax)
        st.pyplot(fig)

    # Pair Plot
    if "Pair Plot" in selected_charts and len(numeric_cols) >= 2:
        st.markdown("🔀 Pair Plot (subset of numeric columns)")
        try:
            fig = sns.pairplot(df[numeric_cols[:5]].dropna(), hue="Cluster" if 'Cluster' in df.columns else None)
            st.pyplot(fig)
        except Exception as e:
            st.warning(f"Pair plot skipped due to error: {e}")

    # Heatmap
    if "Heatmap" in selected_charts and len(numeric_cols) >= 2:
        st.markdown("🧮 Correlation Heatmap")
        fig, ax = plt.subplots()
        sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

else:
    st.info("📁 Please upload a CSV or TSV file to begin.")
