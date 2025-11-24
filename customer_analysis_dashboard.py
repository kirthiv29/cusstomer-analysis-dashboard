import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from fpdf import FPDF
import base64


# ============================
# Page setup
# ============================
st.set_page_config(page_title="Customer Analysis Dashboard", layout="wide")
st.title("Fully Customizable Customer Analysis Dashboard")


# ============================
# File uploader
# ============================
uploaded_file = st.file_uploader(
    "Upload your customer data file",
    type=["csv", "txt", "xlsx", "xls"]
)


# ============================
# Tutorials Section
# ============================
with st.expander("📘 Tutorials / Explanations"):
    st.markdown("""
    **🧠 KMeans Clustering (Simple Explanation)**  
    - It groups similar customers together based on numeric features.  
    - You choose K (number of clusters).  
    - The algorithm finds K centers and assigns each customer to the closest one.  
    - Useful for marketing, segmentation, personalization.

    **📊 Visualizations Meaning**
    - Histogram → Value distribution  
    - Box Plot → Outliers  
    - Scatter → Relationship between variables  
    - Pair Plot → Multi-feature visualization  
    - Heatmap → Correlation strength  
    - Bar Chart → Count comparison  
    - Line Chart → Trends  
    - Area Chart → Volume changes  
    - Violin Plot → Distribution shape  
    - Count Plot → Frequency of categories  
    """)


if uploaded_file:
    # ============================
    # Read file
    # ============================
    try:
        if uploaded_file.name.endswith(("xlsx", "xls")):
            df = pd.read_excel(uploaded_file)
        else:
            df = pd.read_csv(uploaded_file)
    except:
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, sep="\t")

    df.columns = df.columns.str.strip()

    st.subheader("📄 Raw Data Preview")
    st.dataframe(df.head())

    # ============================
    # Full Data Analysis
    # ============================
    st.subheader("📊 Full Data Analysis")

    st.write("### Head:")
    st.write(df.head())
    
    # FIXED df.info()
    st.write("### Info:")
    import io
    buf = io.StringIO()
    df.info(buf=buf)
    info_output = buf.getvalue()
    st.text(info_output)
    
    st.write("### Describe:")
    st.write(df.describe(include="all"))



    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    non_numeric_cols = df.select_dtypes(exclude='number').columns.tolist()


    # ============================
    # Sidebar Config
    # ============================
    st.sidebar.header("🔧 Configuration Panel")

    cluster_cols = st.sidebar.multiselect(
        "Select numeric columns for clustering",
        options=numeric_cols,
        default=numeric_cols[:3]
    )

    missing_strategy = st.sidebar.radio(
        "Missing Value Strategy",
        ["Drop rows", "Fill with mean"]
    )

    cluster_k = st.sidebar.slider(
        "Number of Clusters",
        min_value=2,
        max_value=10,
        value=3
    )

    vis_cols = st.sidebar.multiselect(
        "Select columns for visualization",
        options=df.columns,
        default=numeric_cols[:2]
    )


    # ============================
    # KMeans Clustering
    # ============================
    if cluster_cols:
        st.subheader("🧠 KMeans Clustering")

        try:
            if missing_strategy == "Drop rows":
                df_cluster = df[cluster_cols].dropna()
            else:
                df_cluster = df[cluster_cols].fillna(df[cluster_cols].mean())

            scaler = StandardScaler()
            scaled = scaler.fit_transform(df_cluster)

            kmeans = KMeans(n_clusters=cluster_k, random_state=42, n_init='auto')
            clusters = kmeans.fit_predict(scaled)

            df.loc[df_cluster.index, 'Cluster'] = clusters

            st.success(f"KMeans clustering completed using: {', '.join(cluster_cols)}")

            st.write("### Cluster Distribution:")
            st.bar_chart(df['Cluster'].value_counts())

            st.write("### Data with Cluster Labels:")
            st.dataframe(df.head())

        except Exception as e:
            st.error(f"Error in clustering: {e}")


    # ============================
    # Customer Segmentation Summary
    # ============================
    if "Cluster" in df.columns:
        st.subheader("🧩 Customer Segmentation Summary")

        summary_text = ""
        for cid in sorted(df['Cluster'].unique()):
            cluster_data = df[df['Cluster'] == cid]
            summary_text += f"""
Cluster {cid} Overview:
- Total Customers: {len(cluster_data)}
- Mean values:
{cluster_data[numeric_cols].mean().to_string()}
"""

        st.text(summary_text)


    # ============================
    # AI-GENERATED RECOMMENDATIONS
    # ============================
    if "Cluster" in df.columns:
        st.subheader("🤖 AI-Generated Recommendations")

        def generate_recommendations(cluster_data, cluster_id):
            avg = cluster_data[numeric_cols].mean()
            rec = f"Cluster {cluster_id} Insights:\n"

            if avg.mean() > df[numeric_cols].mean().mean():
                rec += "- 🔥 High-value customers — push premium plans.\n"
            else:
                rec += "- 🧊 Low-value group — offer onboarding, discounts.\n"

            if (cluster_data[numeric_cols] > cluster_data[numeric_cols].quantile(0.75)).sum().sum() > 0:
                rec += "- 📈 Outliers detected — potential VIP customers.\n"

            if cluster_data[numeric_cols].std().mean() < df[numeric_cols].std().mean():
                rec += "- 📉 Stable spending behavior.\n"
            else:
                rec += "- 🔄 Highly variable — personal recommendations needed.\n"

            return rec

        final_recs = ""
        for cid in sorted(df['Cluster'].unique()):
            final_recs += generate_recommendations(df[df['Cluster'] == cid], cid) + "\n"

        st.text(final_recs)


    # ============================
    # Visualizations
    # ============================
    st.subheader("🎨 Visualizations")

    selected_charts = st.sidebar.multiselect(
        "Choose up to 10 visualizations",
        ["Histogram", "Box Plot", "Scatter Plot", "Pair Plot", "Heatmap",
         "Bar Chart", "Line Chart", "Area Chart", "Violin Plot", "Count Plot"],
        default=["Histogram", "Scatter Plot"]
    )

    # Histogram
    if "Histogram" in selected_charts and numeric_cols:
        col = st.selectbox("Histogram Column", numeric_cols, key="hist")
        fig, ax = plt.subplots()
        sns.histplot(df[col].dropna(), kde=True, ax=ax)
        st.pyplot(fig)

    # Box Plot
    if "Box Plot" in selected_charts and numeric_cols:
        y = st.selectbox("Box Plot Y", numeric_cols, key="box_y")
        x = st.selectbox("Box Plot X", non_numeric_cols or ["None"], key="box_x")
        fig, ax = plt.subplots()
        sns.boxplot(data=df, x=x, y=y, ax=ax)
        st.pyplot(fig)

    # Scatter Plot
    if "Scatter Plot" in selected_charts and len(numeric_cols) >= 2:
        x_sc = st.selectbox("Scatter X", numeric_cols, key="sc_x")
        y_sc = st.selectbox("Scatter Y", numeric_cols, key="sc_y")
        hue = st.selectbox("Hue", ["None"] + non_numeric_cols + (["Cluster"] if "Cluster" in df.columns else []))
        fig, ax = plt.subplots()
        if hue != "None":
            sns.scatterplot(data=df, x=x_sc, y=y_sc, hue=hue, ax=ax)
        else:
            sns.scatterplot(data=df, x=x_sc, y=y_sc, ax=ax)
        st.pyplot(fig)

    # Pair Plot
    if "Pair Plot" in selected_charts and len(numeric_cols) >= 2:
        fig = sns.pairplot(df[numeric_cols[:5]].dropna(),
                           hue="Cluster" if "Cluster" in df.columns else None)
        st.pyplot(fig)

    # Heatmap
    if "Heatmap" in selected_charts:
        fig, ax = plt.subplots()
        sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

    # Bar Chart
    if "Bar Chart" in selected_charts and non_numeric_cols:
        col = st.selectbox("Bar Chart Column", non_numeric_cols, key="bar")
        st.bar_chart(df[col].value_counts())

    # Line Chart
    if "Line Chart" in selected_charts and numeric_cols:
        col = st.selectbox("Line Chart Column", numeric_cols, key="line")
        st.line_chart(df[col])

    # Area Chart
    if "Area Chart" in selected_charts and numeric_cols:
        col = st.selectbox("Area Chart Column", numeric_cols, key="area")
        st.area_chart(df[col])

    # Violin Plot
    if "Violin Plot" in selected_charts:
        y = st.selectbox("Violin Column", numeric_cols, key="violin")
        fig, ax = plt.subplots()
        sns.violinplot(y=df[y], ax=ax)
        st.pyplot(fig)

    # Count Plot
    if "Count Plot" in selected_charts and non_numeric_cols:
        col = st.selectbox("Count Plot Column", non_numeric_cols, key="count")
        fig, ax = plt.subplots()
        sns.countplot(data=df, x=col, ax=ax)
        st.pyplot(fig)


    # ============================
    # Insights
    # ============================
    st.subheader("🧾 Insights & Suggestions")
    st.markdown("""
    - Check cluster size to understand customer segmentation strength.  
    - High correlation may indicate redundant features.  
    - Large number of outliers → consider cleansing.  
    - Scatter plots help verify cluster separation.  
    """)


    # ============================
    # PDF REPORT GENERATION
    # ============================
    st.subheader("📄 Download Full PDF Report")

    def generate_pdf():
        pdf = FPDF()
        pdf.add_page()
        pdf.add_font("DejaVu", "", "DejaVuSans.ttf", uni=True)
        pdf.set_font("DejaVu", size=12)


        pdf.cell(200, 10, txt="Customer Analysis Report", ln=True, align='C')

        # Summary
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, "Dataset Summary", ln=True)
        pdf.add_font("DejaVu", "", "DejaVuSans.ttf", uni=True)
        pdf.set_font("DejaVu", size=12)

        pdf.multi_cell(0, 8, df.describe(include='all').to_string())
        pdf.ln(5)

        if "Cluster" in df.columns:
            pdf.set_font("Arial", 'B', 12)
            pdf.cell(0, 10, "Cluster Summary", ln=True)
            pdf.set_font("Arial", size=11)

            for cid in sorted(df['Cluster'].unique()):
                cluster_data = df[df['Cluster'] == cid]
                text = f"""
Cluster {cid}:
Size: {len(cluster_data)}
Mean values:
{cluster_data[numeric_cols].mean().to_string()}
"""
                pdf.multi_cell(0, 8, text)

        pdf.ln(5)
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, "AI Recommendations", ln=True)
        pdf.set_font("Arial", size=11)
        pdf.multi_cell(0, 8, final_recs)

        return pdf.output(dest='S').encode('latin1')

    if st.button("⬇️ Download PDF Report"):
        pdf_bytes = generate_pdf()
        b64 = base64.b64encode(pdf_bytes).decode()
        href = f'<a href="data:application/pdf;base64,{b64}" download="Customer_Report.pdf">Click to download PDF</a>'
        st.markdown(href, unsafe_allow_html=True)


    # ============================
    # DOWNLOAD CSV WITH CLUSTER LABELS
    # ============================
    if "Cluster" in df.columns:
        st.subheader("📥 Download Processed CSV")

        csv_data = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Download CSV with Cluster Labels",
            data=csv_data,
            file_name="clustered_customers.csv",
            mime="text/csv"
        )


else:
    st.info("📁 Upload a file to begin.")

