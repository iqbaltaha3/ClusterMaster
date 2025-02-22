import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

# --- Custom CSS for a modern look ---
st.set_page_config(page_title="ClusterMaster", page_icon="", layout="wide")
custom_css = """
<style>
body { font-family: 'Segoe UI', sans-serif; background-color: #f4f6f9; color: #333333; }
.stContainer { background-color: #ffffff; padding: 2rem; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); margin-bottom: 2rem; }
h1, h2, h3, h4 { color: #1f77b4; }
.sidebar .sidebar-content { background-color: #ffffff; border-radius: 10px; padding: 1rem; }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# --- Header & Landing Page ---
with st.container():
    st.title("ClusterMaster: Segment any Audience")
    st.markdown("""
    Welcome to **ClusterMaster** – an interactive tool to segment your audience.
    
    **Features:**
    - Upload your dataset (CSV format)
    - Configure preprocessing, clustering, and visualization parameters
    - View interactive 2D/3D charts, maps, and detailed insights
    - Download your clustered data and summaries
    
    Use the tabs below for navigation.
    """)
    with st.expander("User Guide & Instructions"):
        st.markdown("""
        **How to Use:**
        1. **Data Upload:** Upload your CSV file using the sidebar.
        2. **Parameter Settings:** Adjust which columns to use, choose scaling methods, and configure clustering options. Then click **Update File** to process.
        3. **Clustering & Visualizations:** View interactive charts, maps, and evaluation metrics.
        4. **Download:** Export the results for further analysis.
        """)

# --- Sidebar: File Upload & Parameter Settings ---
with st.sidebar.form("Parameter Settings", clear_on_submit=False):
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file:
        # Read file once
        df = pd.read_csv(uploaded_file)
        primary_key = st.selectbox("Primary Key Column (Optional)", [None] + list(df.columns))
        if primary_key:
            df = df.set_index(primary_key)

        default_numeric = df.select_dtypes(include=np.number).columns.tolist()
        selected_columns = st.multiselect("Select Columns for Clustering", options=df.columns.tolist(), default=default_numeric)
        scaling_method = st.radio("Scaling Method", options=["Standard", "MinMax"], index=0)
        missing_method = st.radio("Missing Value Handling", options=["Fill with Mean", "Fill with Median", "Drop Missing"], index=0)
        dr_method = st.radio("Dimensionality Reduction", options=["PCA", "t-SNE"], index=0)
        algorithm = st.selectbox("Clustering Algorithm", ["KMeans", "DBSCAN", "Agglomerative", "Gaussian Mixture"])
        if algorithm == "KMeans":
            num_clusters = st.slider("Number of Clusters", 2, 10, 3)
        elif algorithm == "DBSCAN":
            eps = st.slider("Epsilon (eps)", 0.1, 2.0, 0.5, step=0.1)
            min_samples = st.slider("Minimum Samples", 2, 10, 5)
        elif algorithm == "Agglomerative":
            num_clusters = st.slider("Number of Clusters", 2, 10, 3)
        else:
            num_clusters = st.slider("Number of Clusters", 2, 10, 3)
    submitted = st.form_submit_button("Update File")

# Only proceed if a file has been uploaded and the form submitted
if uploaded_file and submitted:
    with st.spinner("Processing data and clustering..."):
        # --- Data Preprocessing ---
        st.header("Data & Preprocessing")
        st.subheader("Raw Data Preview")
        st.dataframe(df.head())

        # Preserve original categorical data for later insights
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        original_categorical_data = df[categorical_cols].copy() if categorical_cols else pd.DataFrame()

        # Encode categorical columns
        label_encoders = {}
        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le

        # Handle missing values
        if missing_method == "Fill with Mean":
            df.fillna(df.mean(), inplace=True)
        elif missing_method == "Fill with Median":
            df.fillna(df.median(), inplace=True)
        else:
            df.dropna(inplace=True)

        # Filter selected columns and keep only numeric features
        if selected_columns:
            df_selected = df[selected_columns]
        else:
            st.error("No columns selected for clustering.")
            st.stop()

        df_selected = df_selected.select_dtypes(include=np.number)
        if df_selected.empty:
            st.error("No numerical features available in the selected columns.")
            st.stop()

        selected_features = list(df_selected.columns)
        st.info(f"Selected Numeric Features: {', '.join(selected_features)}")

        # Caching the scaling computation
        @st.cache_data(show_spinner=False)
        def scale_data(data, method):
            if method == "Standard":
                scaler = StandardScaler()
            else:
                scaler = MinMaxScaler()
            return scaler, scaler.fit_transform(data)
        scaler, scaled_data = scale_data(df_selected, scaling_method)

        st.success("Data preprocessing completed.")

        # --- Clustering & Visualizations ---
        st.header("Clustering & Visualizations")

        # Dimensionality reduction (cached)
        @st.cache_data(show_spinner=False)
        def reduce_dimensions(data, method):
            if method == "PCA":
                reducer = PCA(n_components=3)
            else:
                reducer = TSNE(n_components=3, random_state=42)
            dr_result = reducer.fit_transform(data)
            return pd.DataFrame(dr_result, columns=["Component 1", "Component 2", "Component 3"])
        dr_df = reduce_dimensions(scaled_data, dr_method)

        # Clustering (cached for performance)
        @st.cache_data(show_spinner=False)
        def perform_clustering(data, algo, **kwargs):
            if algo == "KMeans":
                model = KMeans(n_clusters=kwargs.get("num_clusters"), random_state=42)
            elif algo == "DBSCAN":
                model = DBSCAN(eps=kwargs.get("eps"), min_samples=kwargs.get("min_samples"))
            elif algo == "Agglomerative":
                model = AgglomerativeClustering(n_clusters=kwargs.get("num_clusters"))
            else:
                model = GaussianMixture(n_components=kwargs.get("num_clusters"), random_state=42)
            clusters = model.fit_predict(data)
            return model, clusters
        if algorithm in ["KMeans", "Agglomerative", "Gaussian Mixture"]:
            model, clusters = perform_clustering(scaled_data, algorithm, num_clusters=num_clusters)
        else:
            model, clusters = perform_clustering(scaled_data, algorithm, eps=eps, min_samples=min_samples)

        df["Cluster"] = clusters
        dr_df["Cluster"] = clusters

        # --- Cluster Evaluation Section ---
        st.subheader("Cluster Evaluation Metrics")
        # If using DBSCAN and noise (-1) is present, filter noise points for evaluation metrics.
        if algorithm == "DBSCAN" and (-1 in clusters):
            valid_idx = np.where(clusters != -1)[0]
        else:
            valid_idx = np.arange(len(clusters))
        if len(np.unique(clusters[valid_idx])) > 1:
            silhouette_avg = silhouette_score(scaled_data[valid_idx], clusters[valid_idx])
            calinski_score = calinski_harabasz_score(scaled_data[valid_idx], clusters[valid_idx])
            davies_score = davies_bouldin_score(scaled_data[valid_idx], clusters[valid_idx])
            st.markdown(f"**Silhouette Score:** {silhouette_avg:.3f}\n\n"
                        "This metric measures how similar an object is to its own cluster compared to other clusters. A higher value means that the clusters are well-separated.")
            st.markdown(f"**Calinski-Harabasz Score:** {calinski_score:.3f}\n\n"
                        "This score reflects the ratio of between-cluster dispersion and within-cluster dispersion. Higher values indicate better-defined clusters.")
            st.markdown(f"**Davies-Bouldin Score:** {davies_score:.3f}\n\n"
                        "This metric assesses the average similarity between each cluster and its most similar one. Lower values indicate better separation between clusters.")
        else:
            st.warning("Not enough clusters for evaluation metrics. Consider adjusting your clustering parameters.")

        # 3D Scatter Plot
        st.subheader("3D Cluster Visualization")
        fig_3d = px.scatter_3d(dr_df, x="Component 1", y="Component 2", z="Component 3", 
                               color=dr_df["Cluster"].astype(str),
                               title="3D Cluster Visualization")
        fig_3d.update_traces(marker=dict(size=5, opacity=0.8))
        st.plotly_chart(fig_3d, use_container_width=True)

        # 2D Scatter Plot
        st.subheader("2D Cluster Visualization")
        fig_2d = px.scatter(dr_df, x="Component 1", y="Component 2", 
                            color=dr_df["Cluster"].astype(str),
                            title="2D Cluster Visualization")
        st.plotly_chart(fig_2d, use_container_width=True)

        # Cluster Distribution Bar Chart
        st.subheader("Cluster Distribution")
        cluster_counts = df["Cluster"].value_counts().reset_index()
        cluster_counts.columns = ["Cluster", "Count"]
        fig_bar = px.bar(cluster_counts, x="Cluster", y="Count", title="Cluster Distribution")
        st.plotly_chart(fig_bar, use_container_width=True)

        # Optional Map Visualization (if location data exists)
        if "latitude" in df.columns and "longitude" in df.columns:
            st.subheader("Geographical Cluster Map")
            map_fig = px.scatter_mapbox(df, lat="latitude", lon="longitude", color=df["Cluster"].astype(str),
                                        hover_data=selected_features, zoom=3, height=400,
                                        title="Cluster Map")
            map_fig.update_layout(mapbox_style="open-street-map")
            st.plotly_chart(map_fig, use_container_width=True)

        # For KMeans: Radar Chart of Cluster Centroids
        if algorithm == "KMeans":
            st.subheader("Cluster Centroid Analysis")
            centroids = model.cluster_centers_
            centroids_orig = scaler.inverse_transform(centroids)
            centroid_df = pd.DataFrame(centroids_orig, columns=selected_features)
            categories = selected_features
            fig_radar = go.Figure()
            for i, row in centroid_df.iterrows():
                fig_radar.add_trace(go.Scatterpolar(
                    r=row.values,
                    theta=categories,
                    fill='toself',
                    name=f"Cluster {i}"
                ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True)),
                showlegend=True,
                title="Average Feature Values per Cluster"
            )
            st.plotly_chart(fig_radar, use_container_width=True)

        # --- Insights & Detailed Analysis ---
        st.header("Insights & Detailed Analysis")
        st.subheader("Cluster Summary Statistics")
        summary = df.groupby("Cluster")[selected_features].agg(["mean", "std", "min", "max"])
        st.dataframe(summary)

        st.subheader("Detailed Cluster Insights")
        for cluster in sorted(df["Cluster"].unique()):
            st.markdown(f"#### Cluster {cluster}")
            cluster_data = df[df["Cluster"] == cluster]
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Size:** {len(cluster_data)}")
                mean_vals = cluster_data[selected_features].mean().to_dict()
                st.table(pd.DataFrame(mean_vals, index=["Mean"]).T)
            with col2:
                if not original_categorical_data.empty:
                    cat_insights = {}
                    for col in categorical_cols:
                        if col in original_categorical_data.columns:
                            mode_val = original_categorical_data.loc[cluster_data.index, col].mode()
                            cat_insights[col] = mode_val[0] if not mode_val.empty else "N/A"
                    st.table(pd.DataFrame(list(cat_insights.items()), columns=["Category", "Most Frequent Value"]))
            # Optional Demographic Analysis
            demo_cols = [col for col in df.columns if col.lower() in ["age", "gender", "income"]]
            if demo_cols:
                st.markdown("**Demographic Distributions:**")
                for col in demo_cols:
                    if col in df.columns:
                        fig_demo = px.histogram(cluster_data, x=col, title=f"Distribution of {col} in Cluster {cluster}")
                        st.plotly_chart(fig_demo, use_container_width=True)

        # --- Download Section ---
        st.header("Download Your Results")
        csv_data = df.to_csv(index=False).encode('utf-8')
        st.download_button(label="Download Clustered Data CSV", data=csv_data, file_name="clustered_data.csv", mime="text/csv")
        summary_csv = summary.to_csv().encode('utf-8')
        st.download_button(label="Download Cluster Summary CSV", data=summary_csv, file_name="cluster_summary.csv", mime="text/csv")

