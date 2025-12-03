import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Importar módulos del proyecto
from customer_data_simulator import CustomerDataSimulator
from customer_preprocessor import CustomerPreprocessor
from customer_cluster_model import CustomerClusterModel
from cluster_evaluator import ClusterEvaluator
from cluster_visualizer_kmeans import KMeansVisualizer
from cluster_visualizer_dbeans import DBSCANVisualizer


def main():    
    print("\n" + "="*80)
    print(" "*20 + "SISTEMA DE CLUSTERING DE CLIENTES")
    print("="*80)
    
    # ========================================================================
    # PASO 1: GENERACIÓN DE DATOS
    # ========================================================================
    simulator = CustomerDataSimulator(random_state=42)
    df_clientes = simulator.generate_customer_data(n_samples=2000)
    
    print(f"✓ Dataset generado exitosamente")
    print(f"  • Total de clientes: {df_clientes.shape[0]}")
    print(f"  • Características: {df_clientes.shape[1]}")
    print(f"\nPrimeras 5 filas:")
    print(df_clientes.head())
    
    print(f"\nEstadísticas descriptivas:")
    print(df_clientes.describe())
    
    
    # ========================================================================
    # PASO 2: PREPROCESAMIENTO
    # ========================================================================
    preprocessor = CustomerPreprocessor()
    
    # Definir columnas a usar
    numeric_cols = ['edad', 'ingreso_mensual', 'frecuencia_compra', 
                   'monto_promedio', 'antiguedad_dias', 'nivel_satisfaccion']
    categorical_cols = ['metodo_pago']
    
    # Ejecutar pipeline de preprocesamiento
    df_procesado = preprocessor.preprocess_pipeline(
        df_clientes,
        outlier_method='iqr',
        outlier_params={'factor': 1.5}
    )
    
    print(f"\n✓ Preprocesamiento completado")
    print(f"  • Dimensiones: {df_procesado.shape}")
    
    
    # ========================================================================
    # PASO 3: ENTRENAMIENTO DE MODELOS
    # ========================================================================
    cluster_model = CustomerClusterModel(random_state=42)
    
    # 3.1 Entrenar K-Means
    print("\n>>> Entrenando K-Means...")
    kmeans_results = cluster_model.train_kmeans(
        data=df_procesado,
        n_clusters=4
    )
    
    # 3.2 Entrenar DBSCAN
    print("\n>>> Entrenando DBSCAN...")
    dbscan_results = cluster_model.train_dbscan(
        data=df_procesado,
        eps=0.5,
        min_samples=10
    )
    
    
    # ========================================================================
    # PASO 4: EVALUACIÓN DE MODELOS
    # ========================================================================
    evaluator = ClusterEvaluator()
    
    # Evaluar K-Means
    kmeans_metrics = evaluator.evaluate_clustering(
        data=df_procesado,
        labels=kmeans_results['labels'],
        algorithm_name='K-Means'
    )
    
    # Evaluar DBSCAN
    dbscan_metrics = evaluator.evaluate_clustering(
        data=df_procesado,
        labels=dbscan_results['labels'],
        algorithm_name='DBSCAN'
    )
    
    # Comparar modelos
    comparison = evaluator.compare_models(['K-Means', 'DBSCAN'])
    
    
    # ========================================================================
    # PASO 5: RESÚMENES DE CLUSTERS
    # ========================================================================
    # Resumen K-Means
    print("\n>>> Resumen de Clusters - K-Means:")
    kmeans_summary = cluster_model.get_cluster_summary(
        df_original=df_clientes,
        algorithm='kmeans'
    )
    
    # Resumen DBSCAN
    print("\n>>> Resumen de Clusters - DBSCAN:")
    dbscan_summary = cluster_model.get_cluster_summary(
        df_original=df_clientes,
        algorithm='dbscan'
    )
    
    
    # ========================================================================
    # PASO 6: GENERACIÓN DE VISUALIZACIONES
    # ========================================================================
    print("\n\n[PASO 6/6] Generando visualizaciones...")
    print("-" * 80)
    
    # Crear directorios de salida
    import os
    os.makedirs('./visualizations/kmeans/', exist_ok=True)
    os.makedirs('./visualizations/dbscan/', exist_ok=True)
    
    # Visualizaciones K-Means
    print("\n>>> Generando reporte visual de K-Means...")
    kmeans_viz = KMeansVisualizer(figsize=(12, 8))
    kmeans_viz.generate_report(
        data_processed=df_procesado,
        df_original=df_clientes,
        labels=kmeans_results['labels'],
        output_dir='./visualizations/kmeans/'
    )
    
    # Visualizaciones DBSCAN
    print("\n>>> Generando reporte visual de DBSCAN...")
    dbscan_viz = DBSCANVisualizer(figsize=(12, 8))
    dbscan_viz.generate_report(
        data_processed=df_procesado,
        df_original=df_clientes,
        labels=dbscan_results['labels'],
        output_dir='./visualizations/dbscan/'
    )
    
    
    # ========================================================================
    # RESUMEN FINAL
    # ========================================================================
    print("\n\n" + "="*80)
    print(" "*25 + "RESUMEN FINAL DEL ANÁLISIS")
    print("="*80)
    
    print("\n📊 RESULTADOS PRINCIPALES:")
    print("-" * 80)
    
    print("\n1. K-MEANS:")
    print(f"   • Clusters formados: {kmeans_metrics['n_clusters']}")
    print(f"   • Silhouette Score: {kmeans_metrics['silhouette']:.4f}")
    print(f"   • Davies-Bouldin: {kmeans_metrics['davies_bouldin']:.4f}")
    
    print("\n2. DBSCAN:")
    print(f"   • Clusters formados: {dbscan_metrics['n_clusters']}")
    print(f"   • Silhouette Score: {dbscan_metrics['silhouette']:.4f}")
    print(f"   • Outliers detectados: {dbscan_metrics['n_noise']} ({dbscan_metrics['noise_percentage']:.1f}%)")
    
    print("\n3. ARCHIVOS GENERADOS:")
    print("   • Visualizaciones K-Means: ./visualizations/kmeans/")
    print("   • Visualizaciones DBSCAN: ./visualizations/dbscan/")
    
    print("\n" + "="*80)
    print("✓ ANÁLISIS COMPLETADO EXITOSAMENTE")
    print("="*80)
    
    # Opcional: Guardar DataFrames con etiquetas
    df_kmeans = cluster_model.add_labels_to_dataframe(df_clientes, 'kmeans')
    df_dbscan = cluster_model.add_labels_to_dataframe(df_clientes, 'dbscan')
    
    df_kmeans.to_csv('./visualizations/clientes_kmeans.csv', index=False)
    df_dbscan.to_csv('./visualizations/clientes_dbscan.csv', index=False)
    
    print("\n💾 DataFrames con etiquetas guardados:")
    print("   • clientes_kmeans.csv")
    print("   • clientes_dbscan.csv")
    
    return {
        'df_original': df_clientes,
        'df_procesado': df_procesado,
        'kmeans_labels': kmeans_results['labels'],
        'dbscan_labels': dbscan_results['labels'],
        'kmeans_metrics': kmeans_metrics,
        'dbscan_metrics': dbscan_metrics,
        'comparison': comparison
    }


if __name__ == "__main__":
    try:
        results = main()
        print("\n✓ Programa ejecutado exitosamente")
        print("\nPara visualizar los resultados:")
        print("  1. Revisa las carpetas ./visualizations/kmeans/ y ./visualizations/dbscan/")
        print("  2. Abre los archivos CSV generados para análisis adicional")
        print("  3. Las imágenes PNG contienen todos los gráficos del análisis")
        
    except Exception as e:
        print(f"\n❌ Error durante la ejecución: {str(e)}")
        import traceback
        traceback.print_exc()