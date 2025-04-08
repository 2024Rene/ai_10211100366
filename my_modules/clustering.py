# my_modules/clustering.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def create_2d_plot(df, features, kmeans, scaler):
    # Create a copy of the dataframe to avoid modifying the original
    plot_df = df.copy()
    plot_df['Cluster'] = kmeans.labels_
    
    viz_col1, viz_col2 = st.columns(2)
    with viz_col1:
        feature1 = st.selectbox("X-axis Feature", features, index=0)
    with viz_col2:
        feature2 = st.selectbox("Y-axis Feature", features, index=1)
    
    fig = px.scatter(
        plot_df,
        x=feature1,
        y=feature2,
        color='Cluster',
        title=f'Clusters by {feature1} vs {feature2}',
        template='plotly_white',
        color_continuous_scale='viridis',
        labels={'Cluster': 'Cluster Assignment'}
    )
    
    # Add cluster centers
    centers = kmeans.cluster_centers_
    centers_df = pd.DataFrame(
        scaler.inverse_transform(centers),
        columns=features
    )
    
    fig.add_trace(
        go.Scatter(
            x=centers_df[feature1],
            y=centers_df[feature2],
            mode='markers',
            marker=dict(
                color='black',
                symbol='x',
                size=12,
                line=dict(width=2)
            ),
            name='Cluster Centers'
        )
    )
    
    # Update layout for better visualization
    fig.update_layout(
        height=600,
        xaxis_title=feature1,
        yaxis_title=feature2,
        legend_title="Clusters"
    )
    
    return fig

def create_3d_plot(df, features, kmeans, scaler):
    viz_col1, viz_col2, viz_col3 = st.columns(3)
    with viz_col1:
        feature1 = st.selectbox("X-axis Feature", features, index=0)
    with viz_col2:
        feature2 = st.selectbox("Y-axis Feature", features, index=1)
    with viz_col3:
        feature3 = st.selectbox("Z-axis Feature", features, index=2)
    
    fig = px.scatter_3d(
        df,
        x=feature1,
        y=feature2,
        z=feature3,
        color='Cluster',
        title='3D Cluster Visualization',
        template='plotly_white',
        color_continuous_scale='viridis'
    )
    
    # Add cluster centers
    centers = kmeans.cluster_centers_
    centers_df = pd.DataFrame(
        scaler.inverse_transform(centers),
        columns=features
    )
    
    fig.add_trace(
        go.Scatter3d(
            x=centers_df[feature1],
            y=centers_df[feature2],
            z=centers_df[feature3],
            mode='markers',
            marker=dict(
                color='black',
                symbol='x',
                size=8,
                line=dict(width=2)
            ),
            name='Cluster Centers'
        )
    )
    return fig

def create_pair_plot(df, features):
    fig = px.scatter_matrix(
        df,
        dimensions=features,
        color='Cluster',
        title='Feature Pair Plot',
        template='plotly_white',
        color_continuous_scale='viridis'
    )
    fig.update_layout(height=800)
    return fig

def plot_elbow_curve(X_scaled, max_clusters=10):
    inertias = []
    silhouette_scores = []
    K = range(2, max_clusters + 1)
    
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X_scaled)
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))
    
    fig = make_subplots(rows=1, cols=2,
                       subplot_titles=('Elbow Curve', 'Silhouette Score'))
    
    # Elbow curve
    fig.add_trace(
        go.Scatter(x=K, y=inertias, mode='lines+markers',
                  name='Inertia'),
        row=1, col=1
    )
    
    # Silhouette score
    fig.add_trace(
        go.Scatter(x=K, y=silhouette_scores, mode='lines+markers',
                  name='Silhouette Score'),
        row=1, col=2
    )
    
    fig.update_layout(height=400, showlegend=True,
                     title_text="Optimal Number of Clusters Analysis")
    return fig

def clustering_app():
    st.header("🎯 Clustering Module")
    
    # File upload
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        
        # Dataset Preview with Statistics
        st.subheader("📊 Dataset Overview")
        col1, col2 = st.columns(2)
        with col1:
            st.write("Data Preview:")
            st.dataframe(df.head())
        with col2:
            st.write("Dataset Statistics:")
            st.dataframe(df.describe())
        
        # Data Preprocessing
        st.subheader("🔧 Data Preprocessing")
        preprocessing = st.expander("Preprocessing Options")
        
        with preprocessing:
            # Handle missing values
            handle_missing = st.checkbox("Handle Missing Values")
            if handle_missing:
                missing_strategy = st.selectbox(
                    "Missing Values Strategy",
                    ["Drop", "Mean", "Median", "Zero"]
                )
                if missing_strategy == "Drop":
                    df = df.dropna()
                elif missing_strategy == "Mean":
                    df = df.fillna(df.mean())
                elif missing_strategy == "Median":
                    df = df.fillna(df.median())
                elif missing_strategy == "Zero":
                    df = df.fillna(0)
        
        # Feature Selection
        st.subheader("📋 Feature Selection")
        features = st.multiselect(
            "Select Features for Clustering",
            df.columns,
            help="Select at least two features for clustering"
        )
        
        if len(features) >= 2:
            X = df[features]
            
            # Standardize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            X_scaled = pd.DataFrame(X_scaled, columns=features)
            
            # Clustering Parameters
            st.subheader("🎛️ Clustering Parameters")
            
            col1, col2 = st.columns(2)
            with col1:
                # Interactive cluster selection with explanation
                st.write("### Number of Clusters Selection")
                st.info("""
                Use the elbow curve and silhouette score to help determine 
                the optimal number of clusters for your data.
                """)
                
                # Add Elbow curve analysis
                max_clusters = st.slider(
                    "Maximum number of clusters to analyze",
                    min_value=3,
                    max_value=15,
                    value=10,
                    help="Maximum number of clusters to consider in the analysis"
                )
                
                if st.button("Analyze Optimal Clusters"):
                    with st.spinner("Analyzing optimal number of clusters..."):
                        elbow_fig = plot_elbow_curve(X_scaled, max_clusters)
                        st.plotly_chart(elbow_fig, use_container_width=True)
            
            with col2:
                # Interactive cluster selection
                n_clusters = st.slider(
                    "Select Number of Clusters",
                    min_value=2,
                    max_value=max_clusters,
                    value=3,
                    help="Choose the number of clusters based on the analysis"
                )
                
                # Advanced parameters in expander
                with st.expander("Advanced Clustering Parameters"):
                    n_init = st.slider(
                        "Number of Initializations",
                        min_value=1,
                        max_value=20,
                        value=10,
                        help="Higher values increase chance of finding optimal clusters"
                    )
                    max_iter = st.slider(
                        "Maximum Iterations",
                        min_value=100,
                        max_value=1000,
                        value=300,
                        help="Maximum number of iterations for each initialization"
                    )
            
            # Perform clustering
            kmeans = KMeans(
                n_clusters=n_clusters,
                n_init=n_init,
                max_iter=max_iter,
                random_state=42
            )
            
            # Add progress bar for clustering
            with st.spinner("Performing clustering..."):
                clusters = kmeans.fit_predict(X_scaled)
                df['Cluster'] = clusters  # Add cluster labels to main dataframe
            
            # Display cluster quality metrics
            metrics_col1, metrics_col2 = st.columns(2)
            with metrics_col1:
                silhouette_avg = silhouette_score(X_scaled, clusters)
                st.metric(
                    "Silhouette Score",
                    f"{silhouette_avg:.3f}",
                    help="Ranges from -1 to 1. Higher values indicate better-defined clusters."
                )
            
            with metrics_col2:
                inertia = kmeans.inertia_
                st.metric(
                    "Inertia (Within-cluster Sum of Squares)",
                    f"{inertia:.2f}",
                    help="Lower values indicate more compact clusters."
                )
            
            # Add cluster information table
            st.subheader("📊 Cluster Summary")
            cluster_summary = pd.DataFrame({
                'Cluster': range(n_clusters),
                'Size': [sum(clusters == i) for i in range(n_clusters)],
                'Percentage': [(sum(clusters == i) / len(df) * 100).round(2) for i in range(n_clusters)]
            })
            st.dataframe(cluster_summary)
            
            # Add download section with multiple options
            st.subheader("💾 Download Options")
            col1, col2 = st.columns(2)
            
            with col1:
                # Download full dataset with cluster labels
                csv_full = df.to_csv(index=False)
                st.download_button(
                    "Download Full Dataset with Clusters",
                    csv_full,
                    "clustered_data_full.csv",
                    "text/csv",
                    help="Download the complete dataset with cluster assignments"
                )
            
            with col2:
                # Download cluster centroids
                centroids_df = pd.DataFrame(
                    scaler.inverse_transform(kmeans.cluster_centers_),
                    columns=features
                )
                centroids_df['Cluster'] = range(n_clusters)
                csv_centroids = centroids_df.to_csv(index=False)
                st.download_button(
                    "Download Cluster Centroids",
                    csv_centroids,
                    "cluster_centroids.csv",
                    "text/csv",
                    help="Download the coordinates of cluster centers"
                )
            
            # Visualizations
            st.subheader("📈 Clustering Visualizations")
            
            viz_type = st.radio(
                "Select Visualization Type",
                ["2D Scatter Plot", "3D Scatter Plot", "Pair Plot"],
                help="Choose visualization type based on number of features"
            )
            
            # Initialize fig as None
            fig = None
            
            if viz_type == "2D Scatter Plot" and len(features) >= 2:
                try:
                    fig = create_2d_plot(df, features, kmeans, scaler)
                except Exception as e:
                    st.error(f"Error creating 2D plot: {str(e)}")
                    fig = None
                
            elif viz_type == "3D Scatter Plot" and len(features) >= 3:
                fig = create_3d_plot(df, features, kmeans, scaler)
                
            elif viz_type == "Pair Plot" and len(features) >= 2:
                fig = create_pair_plot(df, features)
            
            # Only update and display plot if fig exists
            if fig is not None:
                fig.update_layout(
                    showlegend=True,
                    legend=dict(
                        yanchor="top",
                        y=0.99,
                        xanchor="left",
                        x=0.01
                    )
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Add plot controls
                if st.checkbox("Show Plot Controls"):
                    plot_height = st.slider("Plot Height", 400, 1000, 600)
                    fig.update_layout(height=plot_height)
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning(f"Cannot create {viz_type}. Please ensure you have selected enough features:\n"
                          "- 2D Scatter Plot: needs at least 2 features\n"
                          "- 3D Scatter Plot: needs at least 3 features\n"
                          "- Pair Plot: needs at least 2 features")
            
            # Cluster Information
            st.subheader("📊 Cluster Analysis")
            cluster_info = pd.DataFrame({
                'Cluster': range(n_clusters),
                'Size': [sum(clusters == i) for i in range(n_clusters)]
            })
            
            # Calculate cluster statistics
            cluster_stats = []
            for i in range(n_clusters):
                cluster_data = df[clusters == i][features]
                stats = cluster_data.mean()
                cluster_stats.append(stats)
            
            cluster_stats_df = pd.DataFrame(cluster_stats)
            cluster_stats_df['Size'] = cluster_info['Size']
            cluster_stats_df['Percentage'] = (cluster_stats_df['Size'] / len(df) * 100).round(2)
            
            st.write("Cluster Statistics:")
            st.dataframe(cluster_stats_df)
            
            # Download clustered data
            st.subheader("💾 Download Results")
            csv = df.to_csv(index=False)
            st.download_button(
                "Download Clustered Data",
                csv,
                "clustered_data.csv",
                "text/csv",
                key='download-csv'
            )
