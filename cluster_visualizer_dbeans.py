import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class DBSCANVisualizer:

    def __init__(self, figsize=(10, 6)):
        """
        Inicializa el visualizador de DBSCAN.
        
        Args:
            figsize (tuple): Tamaño por defecto de figuras
        """
        self.algorithm_name = 'DBSCAN'
        self.figsize = figsize
        
        # Configurar estilo profesional
        sns.set_style("whitegrid")
        sns.set_palette("husl")
        plt.rcParams['figure.figsize'] = figsize
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.labelsize'] = 11
        plt.rcParams['axes.titlesize'] = 12
        plt.rcParams['legend.fontsize'] = 9
    
    
    def plot_scatter_2d(self, data, labels, save_path=None):
        """
        Genera scatter plot 2D con reducción PCA.
        Destaca outliers (ruido) en color negro con marcador 'x'.
        
        Args:
            data (array o DataFrame): Datos procesados
            labels (array): Etiquetas de clusters (-1 = outlier)
            save_path (str): Ruta para guardar (opcional)
        """
        if isinstance(data, pd.DataFrame):
            data = data.values
        
        # Aplicar PCA
        pca = PCA(n_components=2, random_state=42)
        data_2d = pca.fit_transform(data)
        
        unique_labels = sorted(set(labels))
        n_clusters = len([l for l in unique_labels if l != -1])
        
        # Crear figura
        fig, ax = plt.subplots(figsize=(10, 7))
        colors = plt.cm.tab10(np.linspace(0, 1, max(10, n_clusters)))
        
        # Graficar cada cluster
        for label in unique_labels:
            if label == -1:
                # Outliers en negro con 'x'
                mask = labels == label
                ax.scatter(data_2d[mask, 0], data_2d[mask, 1],
                          c='black', marker='x', s=80, alpha=0.6,
                          label='Outliers/Ruido', linewidths=2)
            else:
                mask = labels == label
                ax.scatter(data_2d[mask, 0], data_2d[mask, 1],
                          c=[colors[label]], s=100, alpha=0.7,
                          edgecolors='black', linewidth=0.5,
                          label=f'Cluster {label}')
        
        # Configuración
        variance_explained = pca.explained_variance_ratio_
        ax.set_xlabel(f'PC1 ({variance_explained[0]:.1%} varianza)', fontsize=12)
        ax.set_ylabel(f'PC2 ({variance_explained[1]:.1%} varianza)', fontsize=12)
        ax.set_title(f'Visualización de Clusters - {self.algorithm_name}\n'
                    f'Proyección PCA en 2D (con detección de outliers)',
                    fontsize=14, fontweight='bold')
        ax.legend(loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  ✓ Scatter 2D guardado: {save_path}")
        
        plt.show()
    
    
    def plot_heatmap(self, df_original, labels, save_path=None):
        """
        Genera heatmap de perfiles promedio por cluster.
        Excluye outliers del análisis de perfiles.
        
        Args:
            df_original (DataFrame): Datos originales
            labels (array): Etiquetas de clusters
            save_path (str): Ruta para guardar (opcional)
        """
        # Crear DataFrame con etiquetas
        df_with_labels = df_original.copy()
        df_with_labels['cluster'] = labels
        
        # Excluir outliers (-1)
        if -1 in labels:
            df_with_labels = df_with_labels[df_with_labels['cluster'] != -1]
            print(f"  ℹ Outliers excluidos del heatmap")
        
        if len(df_with_labels) == 0:
            print(f"  ⚠ Todos los puntos son outliers - no se puede generar heatmap")
            return
        
        # Seleccionar columnas numéricas
        numeric_cols = df_with_labels.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != 'cluster']
        
        if len(numeric_cols) == 0:
            print(f"  ⚠ No hay columnas numéricas para heatmap")
            return
        
        # Calcular promedios por cluster
        cluster_profiles = df_with_labels.groupby('cluster')[numeric_cols].mean()
        
        # Normalizar para mejor visualización
        scaler = StandardScaler()
        cluster_profiles_normalized = pd.DataFrame(
            scaler.fit_transform(cluster_profiles.T).T,
            index=cluster_profiles.index,
            columns=cluster_profiles.columns
        )
        
        # Crear figura
        fig, ax = plt.subplots(figsize=(12, max(6, len(cluster_profiles) * 1.2)))
        
        # Generar heatmap
        sns.heatmap(
            cluster_profiles_normalized,
            annot=cluster_profiles.round(1),
            fmt='g',
            cmap='RdYlGn',
            center=0,
            linewidths=1,
            linecolor='gray',
            cbar_kws={'label': 'Valor Normalizado (Z-score)'},
            ax=ax
        )
        
        # Configuración
        ax.set_title(f'Perfiles de Clusters - {self.algorithm_name}\n'
                    f'Características Promedio por Cluster (sin outliers)',
                    fontsize=14, fontweight='bold', pad=15)
        ax.set_xlabel('Características', fontsize=12)
        ax.set_ylabel('Cluster', fontsize=12)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.set_yticklabels([f'Cluster {int(i)}' for i in cluster_profiles.index], 
                          rotation=0)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  ✓ Heatmap guardado: {save_path}")
        
        plt.show()
    
    
    def plot_violins(self, df_original, labels, variables=None, save_path=None):
        """
        Genera violinplots comparativos por variable.
        Excluye outliers para mejor visualización de clusters.
        
        Args:
            df_original (DataFrame): Datos originales
            labels (array): Etiquetas de clusters
            variables (list): Variables a graficar (None = todas numéricas)
            save_path (str): Ruta para guardar (opcional)
        """
        df_with_labels = df_original.copy()
        df_with_labels['cluster'] = labels
        
        # Excluir outliers
        if -1 in labels:
            df_with_labels = df_with_labels[df_with_labels['cluster'] != -1]
            print(f"  ℹ Outliers excluidos de violinplots")
        
        if len(df_with_labels) == 0:
            print(f"  ⚠ Todos los puntos son outliers - no se pueden generar violinplots")
            return
        
        # Seleccionar variables
        if variables is None:
            variables = df_with_labels.select_dtypes(include=[np.number]).columns
            variables = [col for col in variables if col != 'cluster']
        
        variables = list(variables)[:6]  # Limitar a 6
        n_vars = len(variables)
        n_cols = 3
        n_rows = (n_vars + n_cols - 1) // n_cols
        
        # Crear subplots
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 4))
        axes = axes.flatten() if n_vars > 1 else [axes]
        
        # Generar violinplot para cada variable
        for idx, var in enumerate(variables):
            ax = axes[idx]
            
            sns.violinplot(
                data=df_with_labels,
                x='cluster',
                y=var,
                palette='Set2',
                inner='box',
                ax=ax
            )
            
            # Añadir medias con diamantes rojos
            means = df_with_labels.groupby('cluster')[var].mean()
            for i, mean in enumerate(means):
                ax.plot(i, mean, 'D', color='red', markersize=8, 
                       markeredgecolor='white', markeredgewidth=1.5,
                       label='Media' if i == 0 else '', zorder=10)
            
            ax.set_title(f'{var}', fontsize=12, fontweight='bold')
            ax.set_xlabel('Cluster', fontsize=10)
            ax.set_ylabel(var, fontsize=10)
            ax.grid(True, alpha=0.3, axis='y')
            
            if idx == 0:
                ax.legend(loc='upper right', fontsize=8)
        
        # Ocultar ejes sobrantes
        for idx in range(n_vars, len(axes)):
            axes[idx].set_visible(False)
        
        plt.suptitle(f'Distribuciones por Cluster - {self.algorithm_name}\n'
                    '⬥ = Media | Caja interior = Cuartiles | Forma = Densidad',
                    fontsize=14, fontweight='bold', y=1.00)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  ✓ Violinplots guardados: {save_path}")
        
        plt.show()
    
    
    def plot_noise_analysis(self, df_original, labels, save_path=None):
        """
        Análisis específico de outliers para DBSCAN.
        Compara características de outliers vs puntos en clusters.
        
        Args:
            df_original (DataFrame): Datos originales
            labels (array): Etiquetas de clusters
            save_path (str): Ruta para guardar (opcional)
        """
        if -1 not in labels:
            print(f"  ℹ DBSCAN no detectó outliers - análisis no necesario")
            return
        
        # Crear columna de tipo
        df_with_labels = df_original.copy()
        df_with_labels['tipo'] = ['Outlier' if l == -1 else 'Cluster' 
                                   for l in labels]
        
        # Seleccionar variables numéricas
        numeric_cols = df_with_labels.select_dtypes(include=[np.number]).columns
        numeric_cols = list(numeric_cols)[:4]  # Primeras 4
        
        # Crear subplots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        for idx, var in enumerate(numeric_cols):
            ax = axes[idx]
            
            # Boxplot comparando outliers vs clusters
            sns.boxplot(
                data=df_with_labels,
                x='tipo',
                y=var,
                palette={'Cluster': 'lightgreen', 'Outlier': 'salmon'},
                ax=ax
            )
            
            # Añadir conteo
            outlier_count = np.sum(labels == -1)
            cluster_count = np.sum(labels != -1)
            
            ax.set_title(f'{var}\n(Clusters: {cluster_count} | Outliers: {outlier_count})',
                        fontsize=11, fontweight='bold')
            ax.set_xlabel('Tipo de Punto', fontsize=10)
            ax.set_ylabel(var, fontsize=10)
            ax.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle(f'Análisis de Outliers - {self.algorithm_name}\n'
                    'Comparación de características: Clusters vs Outliers',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  ✓ Análisis de outliers guardado: {save_path}")
        
        plt.show()
    
    
    def plot_distribution(self, labels, save_path=None):
        """
        Genera gráfico de barras con distribución de clientes por cluster.
        Muestra outliers en anotación especial.
        
        Args:
            labels (array): Etiquetas de clusters
            save_path (str): Ruta para guardar (opcional)
        """
        # Separar outliers
        labels_no_noise = labels[labels != -1]
        
        if len(labels_no_noise) == 0:
            print(f"  ⚠ Todos los puntos son outliers - no hay clusters para graficar")
            return
        
        unique_labels = sorted(set(labels_no_noise))
        counts = [np.sum(labels_no_noise == label) for label in unique_labels]
        percentages = [(count / len(labels_no_noise)) * 100 for count in counts]
        
        # Crear figura
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Gráfico de barras
        bars = ax.bar(
            [f'Cluster {i}' for i in unique_labels],
            counts,
            color=plt.cm.Set3(np.linspace(0, 1, len(unique_labels))),
            edgecolor='black',
            linewidth=1.5
        )
        
        # Añadir etiquetas con valores
        for bar, count, pct in zip(bars, counts, percentages):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{count}\n({pct:.1f}%)',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Configuración
        ax.set_ylabel('Número de Clientes', fontsize=12)
        ax.set_xlabel('Cluster', fontsize=12)
        ax.set_title(f'Distribución de Clientes por Cluster - {self.algorithm_name}\n'
                    f'Total en clusters: {len(labels_no_noise)} clientes',
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Info de outliers
        if -1 in labels:
            noise_count = np.sum(labels == -1)
            noise_pct = (noise_count / len(labels)) * 100
            ax.text(0.02, 0.98, 
                   f'⚠ Outliers detectados: {noise_count} ({noise_pct:.1f}%)',
                   transform=ax.transAxes, fontsize=11,
                   verticalalignment='top', fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='salmon', alpha=0.7))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  ✓ Distribución guardada: {save_path}")
        
        plt.show()
    
    
    def generate_report(self, data_processed, df_original, labels, 
                       output_dir='./visualizations/dbscan/'):
        """
        Genera reporte visual completo para DBSCAN.
        
        Args:
            data_processed (array): Datos procesados/normalizados
            df_original (DataFrame): Datos originales
            labels (array): Etiquetas de clusters de DBSCAN
            output_dir (str): Directorio de salida
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\n{'='*70}")
        print(f"GENERANDO REPORTE VISUAL: DBSCAN")
        print(f"{'='*70}\n")
        
        print("1. Scatter 2D (PCA) con outliers...")
        self.plot_scatter_2d(
            data_processed, labels,
            save_path=f"{output_dir}dbscan_scatter_2d.png"
        )
        
        print("2. Heatmap de perfiles...")
        self.plot_heatmap(
            df_original, labels,
            save_path=f"{output_dir}dbscan_heatmap.png"
        )
        
        print("3. Violinplots por variable...")
        self.plot_violins(
            df_original, labels,
            save_path=f"{output_dir}dbscan_violins.png"
        )
        
        print("4. Análisis específico de outliers...")
        self.plot_noise_analysis(
            df_original, labels,
            save_path=f"{output_dir}dbscan_noise_analysis.png"
        )
        
        print("5. Distribución de clusters...")
        self.plot_distribution(
            labels,
            save_path=f"{output_dir}dbscan_distribution.png"
        )
        
        print(f"\n{'='*70}")
        print(f"✓ REPORTE DBSCAN GENERADO EN: {output_dir}")
        print(f"{'='*70}\n")
