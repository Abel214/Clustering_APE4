import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class KMeansVisualizer:
    def __init__(self, figsize=(10, 6)):
        """
        Inicializa el visualizador de K-Means.
        
        Args:
            figsize (tuple): Tamaño por defecto de figuras
        """
        self.algorithm_name = 'K-Means'
        self.figsize = figsize
        
        # Configurar estilo profesional
        sns.set_style("whitegrid")
        sns.set_palette("husl")
        plt.rcParams['figure.figsize'] = figsize
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.labelsize'] = 11
        plt.rcParams['axes.titlesize'] = 12
        plt.rcParams['legend.fontsize'] = 9
    def plot_boxplots(self, df_original, labels, variables=None, save_path=None):
        """
        Genera boxplots comparativos por variable para cada cluster.

        Los boxplots permiten analizar la mediana, los cuartiles,
        la dispersión y la presencia de valores atípicos dentro de
        cada cluster.

        Args:
            df_original (DataFrame): Datos originales
            labels (array): Etiquetas de clusters
            variables (list): Variables a graficar (None = todas numéricas)
            save_path (str): Ruta para guardar (opcional)
        """
        df_with_labels = df_original.copy()
        df_with_labels['cluster'] = labels

        if variables is None:
            variables = df_with_labels.select_dtypes(include=[np.number]).columns
            variables = [col for col in variables if col != 'cluster']

        variables = list(variables)[:6]
        n_vars = len(variables)
        n_cols = 3
        n_rows = (n_vars + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 4))
        axes = axes.flatten() if n_vars > 1 else [axes]

        for idx, var in enumerate(variables):
            ax = axes[idx]

            sns.boxplot(
                data=df_with_labels,
                x='cluster',
                y=var,
                palette='Set3',
                ax=ax,
                showfliers=True
            )

            ax.set_title(f'{var}', fontsize=12, fontweight='bold')
            ax.set_xlabel('Cluster', fontsize=10)
            ax.set_ylabel(var, fontsize=10)
            ax.grid(True, alpha=0.3, axis='y')

        for idx in range(n_vars, len(axes)):
            axes[idx].set_visible(False)

        plt.suptitle(
            f'Boxplots por Cluster - {self.algorithm_name}\n'
            'Caja = Q1–Q3 | Línea = Mediana | Puntos = Outliers',
            fontsize=14, fontweight='bold', y=1.02
        )
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  ✓ Boxplots guardados: {save_path}")

        plt.show()


    def plot_scatter_2d(self, data, labels, save_path=None):
        """
        Genera scatter plot 2D con reducción PCA.
        """
        if isinstance(data, pd.DataFrame):
            data = data.values
        
        pca = PCA(n_components=2, random_state=42)
        data_2d = pca.fit_transform(data)
        
        unique_labels = sorted(set(labels))
        n_clusters = len(unique_labels)
        
        fig, ax = plt.subplots(figsize=(10, 7))
        colors = plt.cm.tab10(np.linspace(0, 1, max(10, n_clusters)))
        
        for label in unique_labels:
            mask = labels == label
            ax.scatter(
                data_2d[mask, 0], data_2d[mask, 1],
                c=[colors[label]], s=100, alpha=0.7,
                edgecolors='black', linewidth=0.5,
                label=f'Cluster {label}'
            )
        
        variance_explained = pca.explained_variance_ratio_
        ax.set_xlabel(f'PC1 ({variance_explained[0]:.1%} varianza)')
        ax.set_ylabel(f'PC2 ({variance_explained[1]:.1%} varianza)')
        ax.set_title(f'Clusters K-Means - Proyección PCA 2D', fontweight='bold')
        ax.legend()
        ax.grid(True)

        if save_path:
            plt.savefig(save_path, dpi=300)
            print(f"  ✓ Scatter 2D guardado: {save_path}")

        plt.show()
    def plot_heatmap(self, df_original, labels, save_path=None):
        """
        Genera heatmap de perfiles promedio por cluster.
        """
        df_with_labels = df_original.copy()
        df_with_labels['cluster'] = labels

        numeric_cols = df_with_labels.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != 'cluster']

        cluster_profiles = df_with_labels.groupby('cluster')[numeric_cols].mean()

        scaler = StandardScaler()
        cluster_profiles_normalized = pd.DataFrame(
            scaler.fit_transform(cluster_profiles),
            index=cluster_profiles.index,
            columns=cluster_profiles.columns
        )

        plt.figure(figsize=(12, 6))
        sns.heatmap(
            cluster_profiles_normalized,
            annot=cluster_profiles.round(1),
            cmap='RdYlGn',
            linewidths=1,
            center=0
        )

        plt.title(f'Perfiles Promedio por Cluster - K-Means', fontweight='bold')

        if save_path:
            plt.savefig(save_path, dpi=300)
            print(f"  ✓ Heatmap guardado: {save_path}")

        plt.show()
    def plot_distribution(self, labels, save_path=None):
        """
        Genera gráfico de barras con distribución de clientes por cluster.
        """
        unique_labels = sorted(set(labels))
        counts = [np.sum(labels == label) for label in unique_labels]
        percentages = [(c / len(labels)) * 100 for c in counts]

        plt.figure(figsize=(10, 6))
        bars = plt.bar(
            [f'Cluster {i}' for i in unique_labels],
            counts,
            edgecolor='black'
        )

        for bar, count, pct in zip(bars, counts, percentages):
            plt.text(
                bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{count}\n({pct:.1f}%)',
                ha='center', va='bottom', fontweight='bold'
            )

        plt.title(f'Distribución de Clientes por Cluster - K-Means', fontweight='bold')
        plt.ylabel('Número de Clientes')

        if save_path:
            plt.savefig(save_path, dpi=300)
            print(f"  ✓ Distribución guardada: {save_path}")

        plt.show()
    def generate_report(self, data_processed, df_original, labels, 
                       output_dir='./visualizations/kmeans/'):
        """
        Genera reporte visual completo para K-Means.
        """
        import os
        os.makedirs(output_dir, exist_ok=True)

        print(f"\n{'='*70}")
        print(f"GENERANDO REPORTE VISUAL: K-MEANS")
        print(f"{'='*70}\n")

        print("1. Scatter 2D (PCA)...")
        self.plot_scatter_2d(
            data_processed, labels,
            save_path=f"{output_dir}kmeans_scatter_2d.png"
        )
        print("2. Boxplots por variable (MÉTRICAS)...")
        self.plot_boxplots(
            df_original, labels,
            save_path=f"{output_dir}kmeans_boxplots.png"
        )

        print("3. Heatmap de perfiles...")
        self.plot_heatmap(
            df_original, labels,
            save_path=f"{output_dir}kmeans_heatmap.png"
        )

        print("4. Distribución de clusters...")
        self.plot_distribution(
            labels,
            save_path=f"{output_dir}kmeans_distribution.png"
        )

        print(f"\n{'='*70}")
        print(f"✅ REPORTE K-MEANS GENERADO EN: {output_dir}")
        print(f"{'='*70}\n")
