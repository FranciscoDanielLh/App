import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics import silhouette_score
import scipy.stats as stats
from io import BytesIO
import base64
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Configuración de la página
st.set_page_config(
    page_title="Plataforma de Análisis - Proyecto Final",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 0.75rem;
        border-radius: 0.25rem;
        border: 1px solid #c3e6cb;
    }
</style>
""", unsafe_allow_html=True)

# Funciones auxiliares
@st.cache_data
def load_data(uploaded_file):
    """Carga y procesa el archivo CSV"""
    try:
        df = pd.read_csv(uploaded_file)
        return df
    except Exception as e:
        st.error(f"Error al cargar el archivo: {str(e)}")
        return None

def calculate_correlation_matrix(df, columns):
    """Calcula la matriz de correlación"""
    try:
        # Verificar que las columnas existen en el DataFrame
        existing_cols = [col for col in columns if col in df.columns]
        
        if len(existing_cols) < 2:
            return pd.DataFrame()
        
        # Seleccionar solo columnas numéricas
        numeric_df = df[existing_cols].select_dtypes(include=[np.number])
        
        if numeric_df.empty or len(numeric_df.columns) < 2:
            return pd.DataFrame()
        
        # Eliminar filas con valores nulos
        numeric_df = numeric_df.dropna()
        
        if len(numeric_df) < 2:
            return pd.DataFrame()
        
        # Calcular correlación
        corr_matrix = numeric_df.corr()
        return corr_matrix
        
    except Exception as e:
        st.error(f"Error en cálculo de correlaciones: {str(e)}")
        return pd.DataFrame()

def create_time_series_plot(df, date_col, value_cols):
    """Crea gráfico de series temporales"""
    fig = go.Figure()
    
    # Verificar que las columnas existen y son numéricas
    valid_cols = [col for col in value_cols[:5] if col in df.columns and df[col].dtype in ['int64', 'float64']]
    
    if not valid_cols:
        return fig
    
    # Usar índice si no hay columna de fecha válida
    x_axis = df[date_col] if date_col and date_col in df.columns else df.index
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for i, col in enumerate(valid_cols):
        fig.add_trace(go.Scatter(
            x=x_axis,
            y=df[col],
            mode='lines',
            name=col,
            line=dict(width=2, color=colors[i % len(colors)])
        ))
    
    fig.update_layout(
        title="Series Temporales por Medio",
        xaxis_title="Fecha" if date_col else "Período",
        yaxis_title="Valor",
        hovermode='x unified',
        height=400,
        showlegend=True
    )
    
    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(showgrid=True)
    
    return fig

def create_correlation_heatmap(corr_matrix):
    """Crea mapa de calor de correlaciones"""
    if corr_matrix.empty:
        return go.Figure()
    
    try:
        fig = ff.create_annotated_heatmap(
            z=corr_matrix.values,
            x=list(corr_matrix.columns),
            y=list(corr_matrix.index),
            annotation_text=corr_matrix.round(3).values,
            colorscale='RdBu',
            zmid=0,
            showscale=True
        )
        fig.update_layout(
            title="Matriz de Correlaciones",
            height=500,
            xaxis_title="Variables",
            yaxis_title="Variables"
        )
        return fig
    except Exception as e:
        st.error(f"Error al crear heatmap: {str(e)}")
        return go.Figure()

def simulate_mmm_model(df, target_col, media_cols, config):
    """Simula el modelado MMM (versión simplificada)"""
    np.random.seed(42)
    
    # Verificar que las columnas existen y son numéricas
    if target_col not in df.columns or df[target_col].dtype not in ['int64', 'float64']:
        st.error(f"Error: La columna objetivo '{target_col}' no existe o no es numérica.")
        return None
    
    # Filtrar medios que existen y son numéricos
    valid_media_cols = [col for col in media_cols 
                       if col in df.columns and df[col].dtype in ['int64', 'float64']]
    
    if not valid_media_cols:
        st.error("Error: No hay variables de medios válidas (numéricas) para modelar.")
        return None
    
    # Crear datos de contribución simulados
    results = {
        'contributions': df.copy(),
        'roi': {},
        'saturation_curves': {},
        'summary_stats': {}
    }
    
    # Simular contribuciones por medio
    total_target = df[target_col].sum()
    base_contribution = total_target * 0.4  # 40% base orgánica
    
    for i, media in enumerate(valid_media_cols):
        # Aplicar transformación de adstock
        adstock = config.get('adstock_params', {}).get(media, 0.5)
        raw_spend = df[media].values
        
        # Aplicar adstock (simplificado)
        adstocked = np.zeros_like(raw_spend, dtype=float)
        adstocked[0] = raw_spend[0]
        for t in range(1, len(raw_spend)):
            adstocked[t] = raw_spend[t] + adstock * adstocked[t-1]
        
        # Simular saturación y contribución
        saturation_param = np.random.uniform(0.5, 2.0)
        max_contribution = total_target * 0.15  # Máximo 15% por medio
        contribution = (adstocked ** saturation_param) / ((adstocked ** saturation_param) + np.random.uniform(1000, 5000))
        contribution = contribution * np.random.uniform(0.8, 1.2) * max_contribution
        
        results['contributions'][f'{media}_contribution'] = contribution
        
        # Calcular ROI
        total_spend = df[media].sum()
        total_contrib = contribution.sum()
        results['roi'][media] = total_contrib / total_spend if total_spend > 0 else 0
    
    # Base orgánica
    results['contributions']['base_contribution'] = base_contribution / len(df)
    
    return results

def create_contribution_chart(df, media_cols):
    """Crea gráfico de contribuciones apiladas"""
    fig = go.Figure()
    
    # Verificar qué columnas de contribución existen realmente
    contribution_cols = [f'{media}_contribution' for media in media_cols 
                        if f'{media}_contribution' in df.columns]
    
    if 'base_contribution' in df.columns:
        contribution_cols.append('base_contribution')
    
    if not contribution_cols:
        return fig
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    
    for i, col in enumerate(contribution_cols):
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df[col],
            mode='none',
            fill='tonexty' if i > 0 else 'tozeroy',
            name=col.replace('_contribution', '').title(),
            stackgroup='one',
            line=dict(color=colors[i % len(colors)])
        ))
    
    fig.update_layout(
        title="Contribuciones por Medio (Stacked)",
        xaxis_title="Período",
        yaxis_title="Contribución",
        height=400,
        showlegend=True
    )
    return fig

def create_roi_pie_chart(roi_data):
    """Crea gráfico de pastel de ROI"""
    labels = list(roi_data.keys())
    values = list(roi_data.values())
    
    fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.3)])
    fig.update_layout(
        title="ROI por Medio",
        height=400
    )
    return fig

def perform_clustering(df, selected_features, n_clusters_range):
    """Realiza análisis de clustering"""
    # Preparar datos
    X = df[selected_features].select_dtypes(include=[np.number])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Calcular métricas para diferentes números de clusters
    inertias = []
    silhouette_scores = []
    aic_scores = []
    bic_scores = []
    
    for k in range(n_clusters_range[0], n_clusters_range[1] + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        
        inertias.append(kmeans.inertia_)
        if k > 1:
            silhouette_scores.append(silhouette_score(X_scaled, labels))
        else:
            silhouette_scores.append(0)
        
        # AIC y BIC aproximados para K-means
        n_samples, n_features = X_scaled.shape
        aic = kmeans.inertia_ + 2 * k * n_features
        bic = kmeans.inertia_ + np.log(n_samples) * k * n_features
        aic_scores.append(aic)
        bic_scores.append(bic)
    
    # Encontrar número óptimo de clusters (usando método del codo)
    optimal_k = find_elbow_point(inertias) + n_clusters_range[0]
    
    # Clustering final
    final_kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    final_labels = final_kmeans.fit_predict(X_scaled)
    
    results = {
        'elbow_data': pd.DataFrame({
            'n_clusters': range(n_clusters_range[0], n_clusters_range[1] + 1),
            'inertia': inertias,
            'silhouette': silhouette_scores,
            'aic': aic_scores,
            'bic': bic_scores
        }),
        'optimal_k': optimal_k,
        'labels': final_labels,
        'cluster_centers': scaler.inverse_transform(final_kmeans.cluster_centers_),
        'feature_names': selected_features
    }
    
    return results

def find_elbow_point(inertias):
    """Encuentra el punto del codo en la curva de inercia"""
    if len(inertias) < 3:
        return 0
    
    # Método de la segunda derivada
    diffs = np.diff(inertias)
    second_diffs = np.diff(diffs)
    elbow_idx = np.argmax(second_diffs) + 1
    return elbow_idx

def create_elbow_plot(elbow_data):
    """Crea gráfico del codo"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Inercia (Método del Codo)', 'Silhouette Score', 'AIC', 'BIC'),
        vertical_spacing=0.1
    )
    
    fig.add_trace(
        go.Scatter(x=elbow_data['n_clusters'], y=elbow_data['inertia'], 
                  mode='lines+markers', name='Inercia'),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=elbow_data['n_clusters'], y=elbow_data['silhouette'], 
                  mode='lines+markers', name='Silhouette'),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Scatter(x=elbow_data['n_clusters'], y=elbow_data['aic'], 
                  mode='lines+markers', name='AIC'),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=elbow_data['n_clusters'], y=elbow_data['bic'], 
                  mode='lines+markers', name='BIC'),
        row=2, col=2
    )
    
    fig.update_layout(height=600, showlegend=False, title="Análisis de Clustering - Métricas de Evaluación")
    fig.update_xaxes(title_text="Número de Clusters")
    fig.update_yaxes(title_text="Valor de Métrica")
    return fig

def create_cluster_distribution_plot(labels):
    """Crea gráfico de distribución de clusters"""
    cluster_counts = pd.Series(labels).value_counts().sort_index()
    
    fig = go.Figure(data=[
        go.Bar(x=[f'Cluster {i}' for i in cluster_counts.index], 
               y=cluster_counts.values)
    ])
    
    fig.update_layout(
        title="Distribución de Clusters",
        xaxis_title="Cluster",
        yaxis_title="Número de Observaciones",
        height=400
    )
    return fig

def download_excel(data, filename):
    """Crea enlace de descarga para Excel"""
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        if isinstance(data, dict):
            for sheet_name, df in data.items():
                df.to_excel(writer, sheet_name=sheet_name, index=False)
        else:
            data.to_excel(writer, index=False)
    
    excel_data = output.getvalue()
    b64 = base64.b64encode(excel_data).decode()
    href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{filename}">📥 Descargar {filename}</a>'
    return href

# Interfaz principal
def main():
    st.markdown('<h1 class="main-header">📊 Plataforma de Análisis - Proyecto Final</h1>', 
                unsafe_allow_html=True)
    
    # Sidebar para navegación
    st.sidebar.title("🎯 Navegación")
    app_mode = st.sidebar.selectbox(
        "Selecciona la aplicación:",
        ["📈 MMM (Media Mix Modeling)", "👥 Segmentación de Clientes"]
    )
    
    if "📈 MMM" in app_mode:
        mmm_app()
    else:
        segmentation_app()

def mmm_app():
    """Aplicación de Media Mix Modeling"""
    st.header("📈 Media Mix Modeling (MMM)")
    
    # Tabs
    tab1, tab2 = st.tabs(["📊 Input y Configuración", "🔧 Modelado y Resultados"])
    
    with tab1:
        st.subheader("📁 Carga de Datos")
        
        uploaded_file = st.file_uploader(
            "Cargar archivo CSV",
            type=['csv'],
            help="El archivo debe incluir: Fecha, Ventas (unidades/revenue), Inversiones por medio, Actividades por medio"
        )
        
        if uploaded_file is not None:
            df = load_data(uploaded_file)
            
            if df is not None:
                st.success("✅ Archivo cargado exitosamente!")
                
                # Mostrar vista previa
                st.subheader("👀 Vista Previa de Datos")
                st.dataframe(df.head())
                
                # Información básica
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("📊 Filas", df.shape[0])
                with col2:
                    st.metric("📈 Columnas", df.shape[1])
                with col3:
                    st.metric("📅 Período", f"{df.shape[0]} semanas")
                
                # Análisis Exploratorio Básico
                st.subheader("📊 Análisis Exploratorio")
                st.info("💡 Explora tus datos aquí. La configuración de variables y modelado se hace en la siguiente pestaña.")
                
                # Guardar datos en session state para la siguiente pestaña
                st.session_state['mmm_data'] = df
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Series temporales (todas las columnas numéricas)
                    st.subheader("📈 Series Temporales")
                    
                    # Buscar columna de fecha
                    date_col = None
                    for col in df.columns:
                        if any(word in col.lower() for word in ['fecha', 'date', 'time']):
                            date_col = col
                            break
                    
                    # Obtener columnas numéricas
                    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                    
                    if numeric_cols:
                        try:
                            fig_ts = create_time_series_plot(df, date_col, numeric_cols[:5])
                            if fig_ts.data:
                                st.plotly_chart(fig_ts, use_container_width=True)
                            else:
                                st.info("📊 No hay columnas numéricas para mostrar series temporales.")
                        except Exception as e:
                            st.error(f"❌ Error al crear gráfico: {str(e)}")
                    else:
                        st.warning("⚠️ No se encontraron columnas numéricas para visualizar.")
                
                with col2:
                    # Información general de correlaciones
                    st.subheader("🔗 Vista General de Correlaciones")
                    
                    if len(numeric_cols) > 1:
                        try:
                            corr_matrix = calculate_correlation_matrix(df, numeric_cols)
                            if not corr_matrix.empty:
                                # Mostrar solo estadísticas generales
                                correlations_flat = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)]
                                
                                st.metric("📊 Variables Numéricas", len(numeric_cols))
                                st.metric("🔗 Correlación Promedio", f"{np.mean(np.abs(correlations_flat)):.3f}")
                                st.metric("📈 Correlación Máxima", f"{np.max(np.abs(correlations_flat)):.3f}")
                                
                                st.info("🎯 Configura variables específicas en la siguiente pestaña para análisis detallado.")
                            else:
                                st.warning("⚠️ No se pudo calcular matriz de correlación.")
                        except Exception as e:
                            st.error(f"❌ Error en análisis de correlaciones: {str(e)}")
                    else:
                        st.warning("⚠️ Se necesitan al menos 2 variables numéricas para calcular correlaciones.")
                
                # Mensaje de siguiente paso
                st.success("✅ Datos cargados correctamente. Ve a la pestaña '🔧 Modelado y Resultados' para configurar variables y ejecutar el modelo.")
    
    with tab2:
        st.subheader("🔧 Configuración del Modelo y Modelado")
        
        if 'mmm_data' in st.session_state:
            df = st.session_state['mmm_data']
            
            # Configuración de Variables (movido desde primera pestaña)
            st.subheader("⚙️ Configuración de Variables")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Selección de variable objetivo (solo columnas numéricas)
                numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
                
                if not numeric_columns:
                    st.error("❌ No se encontraron columnas numéricas en el archivo.")
                    st.stop()
                
                # Obtener valor previo si existe
                current_target = st.session_state.get('mmm_config', {}).get('target_col', numeric_columns[0])
                if current_target not in numeric_columns:
                    current_target = numeric_columns[0]
                
                target_col = st.selectbox(
                    "🎯 Variable Objetivo (Target)",
                    numeric_columns,
                    index=numeric_columns.index(current_target) if current_target in numeric_columns else 0,
                    help="Selecciona la variable de ventas que quieres modelar (solo variables numéricas)"
                )
                
                # Frecuencia del modelo
                current_frequency = st.session_state.get('mmm_config', {}).get('frequency', 'weekly')
                frequency = st.selectbox(
                    "📅 Frecuencia del Modelo",
                    ["weekly", "monthly"],
                    index=0 if current_frequency == 'weekly' else 1,
                    help="Frecuencia de agregación de los datos"
                )
            
            with col2:
                # Selección de medios (excluir fechas y target, solo numéricas)
                all_numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                
                # Excluir el target y columnas que parecen fechas
                available_cols = [col for col in all_numeric_cols 
                                if col != target_col 
                                and not any(word in col.lower() for word in ['fecha', 'date', 'time', 'id'])]
                
                if not available_cols:
                    st.warning("⚠️ No hay variables de medios disponibles después de excluir el target.")
                    media_cols = []
                else:
                    # Obtener selección previa si existe
                    current_media = st.session_state.get('mmm_config', {}).get('media_cols', [])
                    # Filtrar solo las que siguen siendo válidas
                    valid_current_media = [col for col in current_media if col in available_cols]
                    # Si no hay selección previa válida, usar default
                    if not valid_current_media:
                        valid_current_media = available_cols[:4] if len(available_cols) >= 4 else available_cols
                    
                    media_cols = st.multiselect(
                        "📺 Variables de Medios",
                        available_cols,
                        default=valid_current_media,
                        help="Selecciona las variables de medios a incluir en el modelo"
                    )
            
            # Validar y guardar configuración
            if media_cols:
                # Validar que las columnas de medios son numéricas
                valid_media_cols = [col for col in media_cols if col in df.columns and df[col].dtype in ['int64', 'float64']]
                
                if valid_media_cols:
                    # Actualizar configuración
                    st.session_state['mmm_config'] = {
                        'target_col': target_col,
                        'media_cols': valid_media_cols,
                        'frequency': frequency,
                        'adstock_params': st.session_state.get('mmm_config', {}).get('adstock_params', {})
                    }
                    
                    # Mostrar correlaciones específicas con target
                    st.subheader("🔗 Correlaciones con Target Seleccionado")
                    corr_cols = valid_media_cols + [target_col]
                    
                    try:
                        corr_matrix = calculate_correlation_matrix(df, corr_cols)
                        if not corr_matrix.empty and target_col in corr_matrix.columns:
                            target_corr = corr_matrix[target_col].drop(target_col, errors='ignore')
                            
                            if not target_corr.empty:
                                # Mostrar correlaciones como tabla
                                corr_df = pd.DataFrame({
                                    'Variable': target_corr.index,
                                    'Correlación': target_corr.values.round(3)
                                }).sort_values('Correlación', key=abs, ascending=False)
                                
                                # Mostrar en columnas para mejor presentación
                                col1, col2 = st.columns([2, 1])
                                with col1:
                                    st.dataframe(corr_df, use_container_width=True)
                                with col2:
                                    # Interpretación rápida
                                    strong_corr = corr_df[abs(corr_df['Correlación']) > 0.5]
                                    st.metric("💪 Correlaciones Fuertes (>0.5)", len(strong_corr))
                                    avg_corr = corr_df['Correlación'].abs().mean()
                                    st.metric("📊 Correlación Promedio", f"{avg_corr:.3f}")
                    except Exception as e:
                        st.warning(f"⚠️ No se pudieron calcular correlaciones: {str(e)}")
                    
                else:
                    st.warning("⚠️ No hay variables de medios numéricas válidas seleccionadas.")
                    media_cols = []
            else:
                st.info("💡 Selecciona variables de medios para continuar con la configuración.")
            
            # Solo continuar si hay configuración válida
            if 'mmm_config' in st.session_state and st.session_state['mmm_config'].get('media_cols'):
                config = st.session_state['mmm_config']
            
                # Configuración de parámetros del modelo DLM
                st.subheader("⚙️ Parámetros del Modelo DLM")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    discount_factor_base = st.number_input(
                        "🎯 Discount Factor Base",
                        min_value=0.0,
                        max_value=1.0,
                        value=0.9,
                        step=0.1,
                        help="Factor de descuento para la base orgánica"
                    )
                
                with col2:
                    initial_point_base = st.number_input(
                        "📍 Punto Inicial Base",
                        min_value=0,
                        value=100,
                        help="Punto inicial para la base orgánica"
                    )
                
                with col3:
                    seasonality = st.checkbox(
                        "🌊 Estacionalidad Fourier",
                        value=True,
                        help="Incluir componente de estacionalidad"
                    )
                
                # Configuración de Adstock (movido desde primera pestaña)
                st.subheader("🔄 Parámetros de Adstock por Medio")
                st.info("💡 Los parámetros de adstock controlan cuánto 'carry-over' tiene cada medio publicitario.")
                
                adstock_params = {}
                if config['media_cols']:
                    cols = st.columns(min(len(config['media_cols']), 3))
                    for i, media in enumerate(config['media_cols']):
                        with cols[i % 3]:
                            # Mantener valor previo si existe
                            current_value = config.get('adstock_params', {}).get(media, 0.5)
                            adstock_params[media] = st.slider(
                                f"📺 {media}",
                                min_value=0.0,
                                max_value=1.0,
                                value=current_value,
                                step=0.1,
                                help=f"Adstock para {media}: 0=sin carry-over, 1=máximo carry-over"
                            )
                    
                    # Actualizar configuración con nuevos parámetros de adstock
                    st.session_state['mmm_config']['adstock_params'] = adstock_params
                
                else:
                    st.warning("⚠️ No hay variables de medios configuradas.")
                
                # Botón para ejecutar modelo
                st.subheader("🚀 Ejecutar Modelo")
                
                # Verificar que todo esté configurado
                if config['media_cols'] and config.get('adstock_params'):
                    if st.button("🚀 Ejecutar Modelo MMM", type="primary", use_container_width=True):
                        with st.spinner("🔄 Ejecutando modelo..."):
                            # Usar configuración actualizada
                            model_config = {
                                'adstock_params': st.session_state['mmm_config']['adstock_params'],
                                'discount_factor_base': discount_factor_base,
                                'initial_point_base': initial_point_base,
                                'seasonality': seasonality
                            }
                            
                            results = simulate_mmm_model(
                                df, 
                                config['target_col'], 
                                config['media_cols'], 
                                model_config
                            )
                            
                            if results is not None:
                                st.session_state['mmm_results'] = results
                                st.success("✅ Modelo ejecutado exitosamente!")
                            else:
                                st.error("❌ Error al ejecutar el modelo. Verifica los datos y configuración.")
                else:
                    st.warning("⚠️ Configura las variables y parámetros de adstock antes de ejecutar el modelo.")
                
                # Mostrar resultados si existen
                if 'mmm_results' in st.session_state:
                    results = st.session_state['mmm_results']
                    
                    st.subheader("📊 Resultados del Modelo")
                    
                    # Verificar que hay resultados válidos
                    if results and 'contributions' in results and 'roi' in results:
                        # Gráfico de contribuciones apiladas
                        st.subheader("📈 Contribuciones por Medio")
                        fig_contrib = create_contribution_chart(results['contributions'], config['media_cols'])
                        if fig_contrib.data:  # Solo mostrar si hay datos
                            st.plotly_chart(fig_contrib, use_container_width=True)
                        else:
                            st.warning("⚠️ No se pudieron generar las contribuciones.")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Gráfico de pastel de ROI
                            st.subheader("🥧 ROI por Medio")
                            if results['roi']:
                                fig_pie = create_roi_pie_chart(results['roi'])
                                st.plotly_chart(fig_pie, use_container_width=True)
                            else:
                                st.warning("⚠️ No hay datos de ROI disponibles.")
                        
                        with col2:
                            # Tabla de ROI
                            st.subheader("💰 ROI Detallado")
                            if results['roi']:
                                roi_df = pd.DataFrame(list(results['roi'].items()), 
                                                    columns=['Medio', 'ROI'])
                                roi_df['ROI'] = roi_df['ROI'].round(2)
                                st.dataframe(roi_df, use_container_width=True)
                            else:
                                st.warning("⚠️ No hay datos de ROI disponibles.")
                        
                        # Opciones de descarga
                        st.subheader("💾 Descargar Resultados")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            if results['contributions'] is not None and results['roi']:
                                # Preparar datos para Excel
                                excel_data = {
                                    'Contribuciones': results['contributions'],
                                    'ROI': pd.DataFrame(list(results['roi'].items()), 
                                                      columns=['Medio', 'ROI'])
                                }
                                
                                excel_link = download_excel(excel_data, "resultados_mmm.xlsx")
                                st.markdown(excel_link, unsafe_allow_html=True)
                        
                        with col2:
                            if st.button("💾 Guardar Modelo"):
                                st.session_state['saved_mmm_model'] = results
                                st.success("✅ Modelo guardado como 'modelo1.mmm'")
                    else:
                        st.error("❌ Los resultados del modelo no son válidos.")
            
            else:
                st.info("📋 **Pasos para configurar el modelo:**\n1. Selecciona variable objetivo (target)\n2. Selecciona variables de medios\n3. Configura parámetros del modelo\n4. Ejecuta el modelo")
        
        else:
            st.warning("⚠️ Por favor, carga los datos en la pestaña 'Input y Configuración' primero.")
            st.info("📋 **Pasos necesarios:**\n1. Ve a la primera pestaña\n2. Carga archivo CSV\n3. Regresa aquí para configurar y ejecutar modelo")

def segmentation_app():
    """Aplicación de Segmentación"""
    st.header("👥 Segmentación de Clientes")
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📁 Carga de Datos", 
        "🔍 Clustering", 
        "📊 Probabilidades", 
        "👨‍👩‍👧‍👦 Demografía"
    ])
    
    with tab1:
        st.subheader("📁 Carga de Datos para Segmentación")
        
        uploaded_file = st.file_uploader(
            "Cargar archivo CSV",
            type=['csv'],
            help="El archivo debe contener variables numéricas y continuas",
            key="seg_upload"
        )
        
        if uploaded_file is not None:
            df = load_data(uploaded_file)
            
            if df is not None:
                st.success("✅ Archivo cargado exitosamente!")
                
                # Vista previa
                st.subheader("👀 Vista Previa de Datos")
                st.dataframe(df.head())
                
                # Información básica
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("👥 Respondientes", df.shape[0])
                with col2:
                    st.metric("📊 Variables", df.shape[1])
                with col3:
                    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                    st.metric("🔢 Variables Numéricas", len(numeric_cols))
                
                # Selección de variables
                st.subheader("🎯 Selección de Variables para Clustering")
                
                all_numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                
                selected_vars = st.multiselect(
                    "Selecciona las variables para el análisis:",
                    all_numeric_cols,
                    default=all_numeric_cols,
                    help="Selecciona solo variables numéricas y continuas"
                )
                
                if selected_vars:
                    st.session_state['seg_data'] = df
                    st.session_state['seg_selected_vars'] = selected_vars
                    
                    # Estadísticas descriptivas
                    st.subheader("📈 Estadísticas Descriptivas")
                    desc_stats = df[selected_vars].describe()
                    st.dataframe(desc_stats)
    
    with tab2:
        st.subheader("🔍 Configuración de Clustering")
        
        if 'seg_data' in st.session_state and 'seg_selected_vars' in st.session_state:
            df = st.session_state['seg_data']
            selected_vars = st.session_state['seg_selected_vars']
            
            col1, col2 = st.columns(2)
            
            with col1:
                min_clusters = st.number_input(
                    "🔢 Número Mínimo de Clusters",
                    min_value=2,
                    max_value=10,
                    value=2,
                    help="Rango mínimo para el análisis de clusters"
                )
            
            with col2:
                max_clusters = st.number_input(
                    "🔢 Número Máximo de Clusters",
                    min_value=min_clusters + 1,
                    max_value=15,
                    value=8,
                    help="Rango máximo para el análisis de clusters"
                )
            
            if st.button("🚀 Ejecutar Análisis de Clustering", type="primary"):
                with st.spinner("🔄 Ejecutando clustering..."):
                    clustering_results = perform_clustering(
                        df, 
                        selected_vars, 
                        (min_clusters, max_clusters)
                    )
                    
                    st.session_state['clustering_results'] = clustering_results
                
                st.success("✅ Clustering ejecutado exitosamente!")
            
            # Mostrar resultados si existen
            if 'clustering_results' in st.session_state:
                results = st.session_state['clustering_results']
                
                st.subheader("📊 Análisis del Codo y Métricas")
                
                # Gráfico del codo
                fig_elbow = create_elbow_plot(results['elbow_data'])
                st.plotly_chart(fig_elbow, use_container_width=True)
                
                # Número óptimo de clusters
                st.info(f"🎯 Número óptimo de clusters sugerido: **{results['optimal_k']}**")
                
                # Distribución de clusters
                st.subheader("📊 Distribución de Clusters")
                fig_dist = create_cluster_distribution_plot(results['labels'])
                st.plotly_chart(fig_dist, use_container_width=True)
                
                # Tabla de métricas
                st.subheader("📈 Métricas por Número de Clusters")
                metrics_df = results['elbow_data'].round(3)
                st.dataframe(metrics_df, use_container_width=True)
        
        else:
            st.warning("⚠️ Por favor, carga y selecciona las variables en la pestaña 'Carga de Datos' primero.")
    
    with tab3:
        st.subheader("📊 Análisis de Probabilidades y Variables")
        
        if 'clustering_results' in st.session_state:
            results = st.session_state['clustering_results']
            df = st.session_state['seg_data']
            selected_vars = st.session_state['seg_selected_vars']
            
            # Crear DataFrame con clusters asignados
            df_with_clusters = df.copy()
            df_with_clusters['Cluster'] = results['labels']
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🎯 Centros de Clusters")
                
                # Crear tabla de centros de clusters
                centers_df = pd.DataFrame(
                    results['cluster_centers'],
                    columns=results['feature_names'],
                    index=[f'Cluster {i}' for i in range(results['optimal_k'])]
                )
                
                st.dataframe(centers_df.round(2))
            
            with col2:
                st.subheader("📊 Análisis por Variable")
                
                # Boxplot por cluster para cada variable
                selected_var_for_plot = st.selectbox(
                    "Selecciona variable para visualizar:",
                    selected_vars
                )
                
                if selected_var_for_plot:
                    fig_box = px.box(
                        df_with_clusters,
                        x='Cluster',
                        y=selected_var_for_plot,
                        title=f'Distribución de {selected_var_for_plot} por Cluster'
                    )
                    st.plotly_chart(fig_box, use_container_width=True)
            
            # Análisis de importancia de variables (simulado)
            st.subheader("🔍 Relevancia de Variables")
            
            # Calcular variabilidad entre clusters para cada variable
            variable_importance = {}
            for var in selected_vars:
                cluster_means = df_with_clusters.groupby('Cluster')[var].mean()
                overall_mean = df_with_clusters[var].mean()
                importance = np.var(cluster_means) / np.var(df_with_clusters[var])
                variable_importance[var] = importance
            
            # Crear DataFrame de importancia
            importance_df = pd.DataFrame(
                list(variable_importance.items()),
                columns=['Variable', 'Importancia']
            ).sort_values('Importancia', ascending=False)
            
            importance_df['Importancia_Norm'] = (
                importance_df['Importancia'] / importance_df['Importancia'].max()
            )
            
            # Gráfico de barras de importancia
            fig_importance = px.bar(
                importance_df,
                x='Variable',
                y='Importancia_Norm',
                title='Relevancia de Variables para Distinguir Clusters',
                labels={'Importancia_Norm': 'Importancia Normalizada'}
            )
            fig_importance.update_xaxes(tickangle=45)
            st.plotly_chart(fig_importance, use_container_width=True)
            
            # Guardar resultados para exportación
            st.session_state['cluster_assignments'] = df_with_clusters
        
        else:
            st.warning("⚠️ Por favor, ejecuta el clustering en la pestaña anterior primero.")
    
    with tab4:
        st.subheader("👨‍👩‍👧‍👦 Variables Demográficas y Exportación")
        
        # Opción para cargar variables demográficas adicionales
        st.subheader("📁 Cargar Variables Demográficas Adicionales")
        
        demo_file = st.file_uploader(
            "Cargar archivo con variables demográficas",
            type=['csv'],
            help="Archivo adicional para hacer merge con los clusters",
            key="demo_upload"
        )
        
        if demo_file is not None:
            demo_df = load_data(demo_file)
            if demo_df is not None:
                st.success("✅ Variables demográficas cargadas!")
                st.dataframe(demo_df.head())
                st.session_state['demo_data'] = demo_df
        
        # Análisis de clusters con demografía
        if 'cluster_assignments' in st.session_state:
            df_clusters = st.session_state['cluster_assignments']
            
            st.subheader("📊 Análisis de Clusters")
            
            # Resumen por cluster
            cluster_summary = df_clusters.groupby('Cluster').agg({
                col: ['mean', 'count'] for col in df_clusters.select_dtypes(include=[np.number]).columns 
                if col != 'Cluster'
            }).round(2)
            
            st.subheader("📈 Resumen por Cluster")
            
            # Mostrar estadísticas por cluster
            for cluster_id in sorted(df_clusters['Cluster'].unique()):
                with st.expander(f"📊 Cluster {cluster_id}"):
                    cluster_data = df_clusters[df_clusters['Cluster'] == cluster_id]
                    
                    # Métricas básicas
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("👥 Tamaño", len(cluster_data))
                    with col2:
                        st.metric("📊 % del Total", f"{len(cluster_data)/len(df_clusters)*100:.1f}%")
                    with col3:
                        # Variable más característica
                        if 'seg_selected_vars' in st.session_state:
                            vars_means = cluster_data[st.session_state['seg_selected_vars']].mean()
                            overall_means = df_clusters[st.session_state['seg_selected_vars']].mean()
                            diff = ((vars_means - overall_means) / overall_means * 100).abs()
                            top_var = diff.idxmax()
                            st.metric("🎯 Variable Distintiva", top_var)
                    
                    # Estadísticas detalladas
                    st.dataframe(cluster_data.describe())
            
            # Opciones de exportación
            st.subheader("💾 Opciones de Exportación")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📥 Descargar Tablas Cruzadas", type="primary"):
                    # Preparar datos para Excel
                    excel_data = {}
                    
                    # Hoja 1: Datos con clusters asignados
                    excel_data['Asignaciones_Cluster'] = df_clusters
                    
                    # Hoja 2: Resumen por cluster
                    summary_data = []
                    for cluster_id in sorted(df_clusters['Cluster'].unique()):
                        cluster_data = df_clusters[df_clusters['Cluster'] == cluster_id]
                        summary_row = {'Cluster': cluster_id, 'Tamaño': len(cluster_data)}
                        
                        if 'seg_selected_vars' in st.session_state:
                            for var in st.session_state['seg_selected_vars']:
                                summary_row[f'{var}_mean'] = cluster_data[var].mean()
                        
                        summary_data.append(summary_row)
                    
                    excel_data['Resumen_Clusters'] = pd.DataFrame(summary_data)
                    
                    # Hoja 3: Tabla cruzada si hay variables demográficas
                    if 'demo_data' in st.session_state:
                        excel_data['Variables_Demograficas'] = st.session_state['demo_data']
                    
                    excel_link = download_excel(excel_data, "segmentacion_resultados.xlsx")
                    st.markdown(excel_link, unsafe_allow_html=True)
            
            with col2:
                if st.button("📋 Descargar Solo Asignaciones"):
                    # Solo tabla de ID y cluster
                    if 'Respondent_ID' in df_clusters.columns:
                        assignments_df = df_clusters[['Respondent_ID', 'Cluster']]
                    else:
                        assignments_df = pd.DataFrame({
                            'Respondent_ID': range(1, len(df_clusters) + 1),
                            'Cluster': df_clusters['Cluster']
                        })
                    
                    excel_link = download_excel(assignments_df, "asignaciones_cluster.xlsx")
                    st.markdown(excel_link, unsafe_allow_html=True)
        
        else:
            st.warning("⚠️ Por favor, ejecuta el clustering primero.")

if __name__ == "__main__":
    main()
