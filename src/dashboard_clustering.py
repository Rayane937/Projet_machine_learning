import streamlit as st
import pandas as pd
import plotly.express as px

# Chargement des données
st.title("Dashboard de Clustering")

df_metric = pd.read_excel("output/save_metric.xlsx")

descriptor = st.sidebar.selectbox('Sélectionner un descripteur', 
    ["HISTOGRAM", "HOG", "HOG_DBSCAN", "HISTOGRAM_DBSCAN", "COLOR_HIST", "COLOR_HIST_DBSCAN", "CNN", "CNN_DBSCAN"])

if descriptor == "HISTOGRAM":
    df = pd.read_excel("output/save_clustering_hist_kmeans.xlsx")
elif descriptor == "HOG":
    df = pd.read_excel("output/save_clustering_hog_kmeans.xlsx")
elif descriptor == "HISTOGRAM_DBSCAN":
    df = pd.read_excel("output/save_clustering_hist_dbscan.xlsx")
elif descriptor == "HOG_DBSCAN":
    df = pd.read_excel("output/save_clustering_hog_dbscan.xlsx")
elif descriptor == "COLOR_HIST":
    df = pd.read_excel("output/save_clustering_color_kmeans.xlsx")
elif descriptor == "COLOR_HIST_DBSCAN":
    df = pd.read_excel("output/save_clustering_color_dbscan.xlsx")
elif descriptor == "CNN":
    df = pd.read_excel("output/save_clustering_cnn_kmeans.xlsx")
elif descriptor == "CNN_DBSCAN":
    df = pd.read_excel("output/save_clustering_cnn_dbscan.xlsx")

selected_cluster = st.sidebar.selectbox('Sélectionner un Cluster', range(20))
filtered_data = df[df['cluster'] == selected_cluster]

fig = px.scatter_3d(df, x='x', y='y', z='z', color='cluster', title=f"Clustering 3D avec {descriptor}")
fig.add_scatter3d(x=filtered_data['x'], y=filtered_data['y'], z=filtered_data['z'],
                  mode='markers', marker=dict(color='red', size=10),
                  name=f'Cluster {selected_cluster}')
st.plotly_chart(fig)

st.write("## Analyse Globale des Descripteurs")
for metric in ['ami', 'ari', 'silhouette', 'homogeneity', 'completeness', 'v_measure']:
    fig = px.bar(df_metric, x='descriptor', y=metric, title=f"Score {metric} par descripteur", color='descriptor')
    st.plotly_chart(fig)
st.dataframe(df_metric)
