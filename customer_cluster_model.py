import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score


class CustomerClusterModel:  
    def __init__(self, random_state=42):
        self.random_state = random_state
        
        # Modelos entrenados
        self.kmeans_model = None
        self.dbscan_model = None
        
        # Etiquetas generadas
        self.kmeans_labels = None
        self.dbscan_labels = None
        
        # Métricas de evaluación
        self.kmeans_metrics = {}
        self.dbscan_metrics = {}
    
    
    def train_kmeans(self, data, n_clusters=4):
        """
        Entrena KMeans para agrupar clientes.
        
        KMeans divide los datos en 'n_clusters' grupos minimizando
        la distancia de cada punto a su centroide más cercano.
        
        Args:
            data (array o DataFrame): Datos preprocesados (normalizados)
            n_clusters (int): Número de clusters deseado
        """
        print("\n" + "="*70)
        print(f"ENTRENANDO KMEANS (k={n_clusters})")
        print("="*70)
        
        # Convertir a array si es DataFrame
        if isinstance(data, pd.DataFrame):
            data = data.values
        
        # Entrenar modelo
        self.kmeans_model = KMeans(
            n_clusters=n_clusters,
            random_state=self.random_state,
            n_init=10,      # Número de inicializaciones
            max_iter=300    # Iteraciones máximas
        )
        
        self.kmeans_labels = self.kmeans_model.fit_predict(data)
        
        # Calcular métricas
        inertia = self.kmeans_model.inertia_
        silhouette = silhouette_score(data, self.kmeans_labels)
        
        self.kmeans_metrics = {
            'inertia': inertia,
            'silhouette': silhouette,
            'n_clusters': n_clusters
        }
        
        # Mostrar resultados
        print(f"\n✓ Entrenamiento completado:")
        print(f"  • Clusters: {n_clusters}")
        print(f"  • Inertia: {inertia:.2f} (menor = más compacto)")
        print(f"  • Silhouette: {silhouette:.4f} (cercano a 1 = bien separado)")
        
        # Distribución de clientes
        print(f"\n  Clientes por cluster:")
        for i in range(n_clusters):
            count = (self.kmeans_labels == i).sum()
            percentage = (count / len(data)) * 100
            print(f"    Cluster {i}: {count} ({percentage:.1f}%)")
        
        return {
            'model': self.kmeans_model,
            'labels': self.kmeans_labels,
            'metrics': self.kmeans_metrics
        }
    
    
    def train_dbscan(self, data, eps=0.5, min_samples=10):
        """
        Entrena DBSCAN para agrupar clientes.
        
        DBSCAN agrupa puntos que están "cerca" (dentro de eps) y tienen
        suficientes vecinos (min_samples). Detecta outliers automáticamente.
        
        Args:
            data (array o DataFrame): Datos preprocesados
            eps (float): Radio de vecindad (distancia máxima entre vecinos)
            min_samples (int): Mínimo de vecinos para formar un cluster
        """
        print("\n" + "="*70)
        print(f"ENTRENANDO DBSCAN (eps={eps}, min_samples={min_samples})")
        print("="*70)
        
        # Convertir a array si es DataFrame
        if isinstance(data, pd.DataFrame):
            data = data.values
        
        # Entrenar modelo
        self.dbscan_model = DBSCAN(
            eps=eps,
            min_samples=min_samples,
            metric='euclidean'
        )
        
        self.dbscan_labels = self.dbscan_model.fit_predict(data)
        
        # Calcular métricas
        n_clusters = len(set(self.dbscan_labels)) - (1 if -1 in self.dbscan_labels else 0)
        noise_points = (self.dbscan_labels == -1).sum()
        noise_percentage = (noise_points / len(data)) * 100
        
        # Silhouette (solo para puntos no ruidosos)
        if n_clusters > 1 and noise_points < len(data):
            mask = self.dbscan_labels != -1
            silhouette = silhouette_score(data[mask], self.dbscan_labels[mask])
        else:
            silhouette = 0.0
        
        self.dbscan_metrics = {
            'n_clusters': n_clusters,
            'noise_points': noise_points,
            'noise_percentage': noise_percentage,
            'silhouette': silhouette,
            'eps': eps,
            'min_samples': min_samples
        }
        
        # Mostrar resultados
        print(f"\n✓ Entrenamiento completado:")
        print(f"  • Clusters: {n_clusters}")
        print(f"  • Outliers: {noise_points} ({noise_percentage:.1f}%)")
        print(f"  • Silhouette: {silhouette:.4f}")
        
        # Distribución de clientes (sin outliers)
        if n_clusters > 0:
            print(f"\n  Clientes por cluster (excluyendo outliers):")
            labels_no_noise = self.dbscan_labels[self.dbscan_labels != -1]
            for i in range(n_clusters):
                count = (labels_no_noise == i).sum()
                percentage = (count / len(labels_no_noise)) * 100 if len(labels_no_noise) > 0 else 0
                print(f"    Cluster {i}: {count} ({percentage:.1f}%)")
        
        return {
            'model': self.dbscan_model,
            'labels': self.dbscan_labels,
            'metrics': self.dbscan_metrics
        }
    
    
    def add_labels_to_dataframe(self, df, algorithm='kmeans'):
        df_result = df.copy()
        
        if algorithm == 'kmeans' and self.kmeans_labels is not None:
            df_result['cluster'] = self.kmeans_labels
            print(f"✓ Etiquetas KMeans añadidas al DataFrame")
            
        elif algorithm == 'dbscan' and self.dbscan_labels is not None:
            df_result['cluster'] = self.dbscan_labels
            print(f"✓ Etiquetas DBSCAN añadidas al DataFrame")
            print(f"  Nota: cluster = -1 son outliers")
            
        else:
            print(f"⚠ No hay etiquetas disponibles para '{algorithm}'")
            print(f"  Debes entrenar el modelo primero usando train_{algorithm}()")
        
        return df_result
    
    
    def calculate_metrics(self, data, labels, algorithm_name):
        metrics = {}
        
        # Número de clusters
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        metrics['n_clusters'] = n_clusters
        
        # Silhouette Score
        if n_clusters > 1:
            # Excluir ruido si existe
            if -1 in labels:
                mask = labels != -1
                if mask.sum() > 1 and len(set(labels[mask])) > 1:
                    metrics['silhouette'] = silhouette_score(data[mask], labels[mask])
                else:
                    metrics['silhouette'] = 0.0
            else:
                metrics['silhouette'] = silhouette_score(data, labels)
        else:
            metrics['silhouette'] = 0.0
        
        print(f"\nMétricas para {algorithm_name}:")
        print(f"  • Clusters: {metrics['n_clusters']}")
        print(f"  • Silhouette: {metrics['silhouette']:.4f}")
        
        return metrics
    
    
    def compare_models(self):
        if not self.kmeans_metrics or not self.dbscan_metrics:
            print("⚠ Debes entrenar ambos modelos primero")
            return None
        
        print("\n" + "="*70)
        print("COMPARACIÓN: KMEANS vs DBSCAN")
        print("="*70)
        
        comparison = pd.DataFrame({
            'Métrica': [
                'Número de Clusters',
                'Silhouette Score',
                'Inertia',
                'Puntos de Ruido'
            ],
            'KMeans': [
                self.kmeans_metrics['n_clusters'],
                f"{self.kmeans_metrics['silhouette']:.4f}",
                f"{self.kmeans_metrics['inertia']:.2f}",
                '0 (no detecta outliers)'
            ],
            'DBSCAN': [
                self.dbscan_metrics['n_clusters'],
                f"{self.dbscan_metrics['silhouette']:.4f}",
                'N/A',
                f"{self.dbscan_metrics['noise_points']} ({self.dbscan_metrics['noise_percentage']:.1f}%)"
            ]
        })
        
        print(comparison.to_string(index=False))
        print("\n" + "="*70)
        
        # Interpretación simple
        print("\nINTERPRETACIÓN:")
        if self.kmeans_metrics['silhouette'] > self.dbscan_metrics['silhouette']:
            print("→ KMeans tiene mejor Silhouette Score")
        else:
            print("→ DBSCAN tiene mejor Silhouette Score")
        
        print(f"→ DBSCAN identificó {self.dbscan_metrics['noise_points']} outliers")
        
        return comparison
    
    
    def get_cluster_summary(self, df_original, algorithm='kmeans'):
        # Seleccionar etiquetas
        if algorithm == 'kmeans':
            labels = self.kmeans_labels
        elif algorithm == 'dbscan':
            labels = self.dbscan_labels
        else:
            print(f"⚠ Algoritmo '{algorithm}' no válido")
            return None
        
        if labels is None:
            print(f"⚠ Debes entrenar {algorithm} primero")
            return None
        
        # Añadir etiquetas al DataFrame
        df_with_labels = df_original.copy()
        df_with_labels['cluster'] = labels
        
        # Para DBSCAN, excluir outliers del resumen
        if algorithm == 'dbscan':
            df_with_labels = df_with_labels[df_with_labels['cluster'] != -1]
            print("\nNota: Outliers (cluster=-1) excluidos del resumen")
        
        # Calcular estadísticas por cluster
        summary = df_with_labels.groupby('cluster').agg({
            'edad': 'mean',
            'ingreso_mensual': 'mean',
            'frecuencia_compra': 'mean',
            'monto_promedio': 'mean',
            'antiguedad_dias': 'mean',
            'nivel_satisfaccion': 'mean'
        }).round(2)
        
        # Añadir conteo
        summary['total_clientes'] = df_with_labels.groupby('cluster').size()
        
        print(f"\n{'='*70}")
        print(f"RESUMEN DE CLUSTERS - {algorithm.upper()}")
        print(f"{'='*70}")
        print(summary)
        
        return summary

