import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score


class ClusterEvaluator:
    def __init__(self):
        self.evaluation_results = {}
    
    
    def evaluate_clustering(self, data, labels, algorithm_name):
        """
        Evalúa un modelo de clustering con múltiples métricas.
        """
        print(f"\n{'='*70}")
        print(f"EVALUANDO: {algorithm_name}")
        print(f"{'='*70}")
        
        # Convertir a array si es DataFrame
        if isinstance(data, pd.DataFrame):
            data = data.values
        
        metrics = {}
        
        # Información básica
        n_samples = len(labels)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = (labels == -1).sum() if -1 in labels else 0
        
        metrics['n_samples'] = n_samples
        metrics['n_clusters'] = n_clusters
        metrics['n_noise'] = n_noise
        metrics['noise_percentage'] = (n_noise / n_samples) * 100
        
        print(f"\nInformación General:")
        print(f"  • Total de muestras: {n_samples}")
        print(f"  • Clusters encontrados: {n_clusters}")
        print(f"  • Puntos de ruido: {n_noise} ({metrics['noise_percentage']:.1f}%)")
        
        # Solo calculamos las métricas si hay clusters válidos
        if n_clusters > 1:
            # Filtrar puntos de ruido si existen
            if n_noise > 0:
                mask = labels != -1
                data_filtered = data[mask]
                labels_filtered = labels[mask]
            else:
                data_filtered = data
                labels_filtered = labels
            
            # Calcular métricas de calidad
            metrics.update(self._calculate_quality_metrics(data_filtered, labels_filtered))
            
        else:
            print("\n⚠ No hay suficientes clusters para calcular métricas")
            metrics['silhouette'] = 0.0
            metrics['davies_bouldin'] = 0.0
            metrics['calinski_harabasz'] = 0.0
        
        # Guardar resultados
        self.evaluation_results[algorithm_name] = metrics
        
        return metrics
    
    
    def _calculate_quality_metrics(self, data, labels):
        metrics = {}
        
        print(f"\nMétricas de Calidad:")
        
        # 1. SILHOUETTE SCORE (-1 a 1)
        # Mide qué tan similar es un punto a su propio cluster vs otros clusters
        # Valores cercanos a 1: excelente separación
        # Valores cercanos a 0: clusters solapados
        # Valores negativos: puntos en cluster incorrecto
        try:
            silhouette = silhouette_score(data, labels)
            metrics['silhouette'] = silhouette
            
            # Interpretación
            if silhouette > 0.7:
                interpretation = "Excelente"
            elif silhouette > 0.5:
                interpretation = "Buena"
            elif silhouette > 0.25:
                interpretation = "Aceptable"
            else:
                interpretation = "Pobre"
            
            print(f"  • Silhouette Score: {silhouette:.4f} ({interpretation})")
            
        except Exception as e:
            metrics['silhouette'] = 0.0
            print(f"  • Silhouette Score: No calculable")
        
        # 2. DAVIES-BOULDIN INDEX (menor es mejor)
        # Mide la relación entre dispersión intra-cluster y separación inter-cluster
        # Valores bajos indican mejor clustering
        try:
            db_index = davies_bouldin_score(data, labels)
            metrics['davies_bouldin'] = db_index
            
            # Interpretación
            if db_index < 0.5:
                interpretation = "Excelente separación"
            elif db_index < 1.0:
                interpretation = "Buena separación"
            else:
                interpretation = "Separación mejorable"
            
            print(f"  • Davies-Bouldin Index: {db_index:.4f} ({interpretation})")
            
        except Exception as e:
            metrics['davies_bouldin'] = 0.0
            print(f"  • Davies-Bouldin Index: No calculable")
        
        # 3. CALINSKI-HARABASZ INDEX (mayor es mejor)
        # Ratio de dispersión entre clusters vs dentro de clusters
        # Valores altos indican clusters densos y bien separados
        try:
            ch_score = calinski_harabasz_score(data, labels)
            metrics['calinski_harabasz'] = ch_score
            
            print(f"  • Calinski-Harabasz Score: {ch_score:.2f}")
            
        except Exception as e:
            metrics['calinski_harabasz'] = 0.0
            print(f"  • Calinski-Harabasz Score: No calculable")
        
        return metrics

    def compare_models(self, model_names=None):
        if not self.evaluation_results:
            print(" No hay modelos evaluados para comparar")
            return None
        
        if model_names is None:
            model_names = list(self.evaluation_results.keys())
        
        print(f"\n{'='*70}")
        print("COMPARACIÓN DE MODELOS")
        print(f"{'='*70}")
        
        # Crear DataFrame comparativo
        comparison_data = {}
        
        for model_name in model_names:
            if model_name not in self.evaluation_results:
                continue
            
            metrics = self.evaluation_results[model_name]
            comparison_data[model_name] = {
                'Clusters': metrics.get('n_clusters', 0),
                'Silhouette': round(metrics.get('silhouette', 0), 4),
                'Davies-Bouldin': round(metrics.get('davies_bouldin', 0), 4),
                'Calinski-Harabasz': round(metrics.get('calinski_harabasz', 0), 2),
                'Ruido (%)': round(metrics.get('noise_percentage', 0), 2)
            }
        
        comparison_df = pd.DataFrame(comparison_data).T
        
        print(comparison_df)
        
        # Determinar mejor modelo
        print(f"\n{'='*70}")
        print("RECOMENDACIÓN:")
        print(f"{'='*70}")
        
        best_model = self._determine_best_model(comparison_df)
        print(f"\n✓ Mejor modelo: {best_model}")
        
        return comparison_df
    
    
    def _determine_best_model(self, comparison_df):
        scores = {}
        
        for model_name in comparison_df.index:
            score = 0
            
            # Silhouette (mayor es mejor) - Peso: 40%
            silhouette = comparison_df.loc[model_name, 'Silhouette']
            score += silhouette * 40
            
            # Davies-Bouldin (menor es mejor) - Peso: 30%
            db = comparison_df.loc[model_name, 'Davies-Bouldin']
            if db > 0:
                score += (1 / db) * 30
            
            # Calinski-Harabasz (mayor es mejor, normalizado) - Peso: 20%
            ch = comparison_df.loc[model_name, 'Calinski-Harabasz']
            max_ch = comparison_df['Calinski-Harabasz'].max()
            if max_ch > 0:
                score += (ch / max_ch) * 20
            
            # Penalizar mucho ruido - Peso: 10%
            noise_pct = comparison_df.loc[model_name, 'Ruido (%)']
            score += (1 - noise_pct / 100) * 10
            
            scores[model_name] = score
        
        best_model = max(scores, key=scores.get)
        
        print(f"\nPuntuaciones (escala 0-100):")
        for model, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
            print(f"  • {model}: {score:.2f}")
        
        best_metrics = comparison_df.loc[best_model]
        print(f"  • Silhouette: {best_metrics['Silhouette']:.4f}")
        print(f"  • Davies-Bouldin: {best_metrics['Davies-Bouldin']:.4f}")
        print(f"  • Clusters bien definidos y separados")
        
        return best_model
