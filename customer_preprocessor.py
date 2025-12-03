import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


class CustomerPreprocessor:
    """
    Clase para limpiar y preparar datos de clientes para algoritmos de clustering.
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.encoder = OneHotEncoder(sparse_output=False, drop='first')
        self.column_transformer = None
        self.numeric_features = []
        self.categorical_features = []
        self.processed_feature_names = []
        
    
    def select_features(self, df, numeric_cols=None, categorical_cols=None):
        """
        Selecciona y separa las características numéricas y categóricas.
        """
        # Si no se especifican columnas, usar detección automática
        if numeric_cols is None:
            # Seleccionar automáticamente columnas numéricas
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if categorical_cols is None:
            # Seleccionar automáticamente columnas categóricas/objeto
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Guardar las características para uso posterior
        self.numeric_features = numeric_cols
        self.categorical_features = categorical_cols
        
        # Seleccionamos solo las columnas especificadas
        selected_cols = numeric_cols + categorical_cols
        df_selected = df[selected_cols].copy()
        
        print(f" Características seleccionadas:")
        print(f"  - Numéricas ({len(numeric_cols)}): {numeric_cols}")
        print(f"  - Categóricas ({len(categorical_cols)}): {categorical_cols}")
        
        return df_selected
    
    
    def handle_outliers_iqr(self, df, columns=None, factor=1.5):
        df_clean = df.copy()
        
        # Si no se especifican las columnas, se usan todas las variables numéricas
        if columns is None:
            columns = self.numeric_features
        
        outliers_info = {}
        
        for col in columns:
            if col not in df_clean.columns:
                continue
                
            # Calcular cuartiles y rango 
            Q1 = df_clean[col].quantile(0.25)  # Primer cuartil (25%)
            Q3 = df_clean[col].quantile(0.75)  # Tercer cuartil (75%)
            IQR = Q3 - Q1                       # Rango intercuartílico
            
            # Definir límites para outliers
            lower_bound = Q1 - factor * IQR
            upper_bound = Q3 + factor * IQR
            
            # Contar outliers antes de limpiar
            outliers_lower = (df_clean[col] < lower_bound).sum()
            outliers_upper = (df_clean[col] > upper_bound).sum()
            total_outliers = outliers_lower + outliers_upper
            
            # Recortar valores extremos a los límites
            df_clean[col] = df_clean[col].clip(lower=lower_bound, upper=upper_bound)
            
            # Guardar información para reporte
            if total_outliers > 0:
                outliers_info[col] = {
                    'total_outliers': total_outliers,
                    'outliers_bajos': outliers_lower,
                    'outliers_altos': outliers_upper,
                    'limite_inferior': round(lower_bound, 2),
                    'limite_superior': round(upper_bound, 2)
                }
        
        # Mostrar resumen
        print(f"\n✓ Outliers detectados y tratados (IQR con factor={factor}):")
        if outliers_info:
            for col, info in outliers_info.items():
                print(f"  - {col}: {info['total_outliers']} outliers "
                      f"(límites: [{info['limite_inferior']}, {info['limite_superior']}])")
        else:
            print("  - No se detectaron outliers significativos")
        
        return df_clean
    
    
    def handle_outliers_zscore(self, df, columns=None, threshold=3):
        df_clean = df.copy()
        
        if columns is None:
            columns = self.numeric_features
        
        outliers_info = {}
        
        for col in columns:
            if col not in df_clean.columns:
                continue
            
            # Calcular z-scores
            mean = df_clean[col].mean()
            std = df_clean[col].std()
            z_scores = np.abs((df_clean[col] - mean) / std)
            
            # Identificar outliers
            outliers_mask = z_scores > threshold
            n_outliers = outliers_mask.sum()
            
            if n_outliers > 0:
                # Reemplazar outliers con el valor del umbral
                lower_bound = mean - threshold * std
                upper_bound = mean + threshold * std
                df_clean[col] = df_clean[col].clip(lower=lower_bound, upper=upper_bound)
                
                outliers_info[col] = {
                    'total_outliers': n_outliers,
                    'limite_inferior': round(lower_bound, 2),
                    'limite_superior': round(upper_bound, 2)
                }
        
        print(f"\n✓ Outliers detectados y tratados (Z-Score con threshold={threshold}):")
        if outliers_info:
            for col, info in outliers_info.items():
                print(f"  - {col}: {info['total_outliers']} outliers "
                      f"(límites: [{info['limite_inferior']}, {info['limite_superior']}])")
        else:
            print("  - No se detectaron outliers significativos")
        
        return df_clean
    
    
    def normalize_features(self, df, method='standard'):
        """
        Normaliza las características numéricas.
        """
        df_normalized = df.copy()
        
        if method == 'standard':
            # StandardScaler: media=0, desviación=1
            # Fórmula: (x - mean) / std
            for col in self.numeric_features:
                if col in df_normalized.columns:
                    df_normalized[col] = self.scaler.fit_transform(
                        df_normalized[[col]]
                    )
            
            print(f"\n Normalización aplicada (StandardScaler):")
            print(f"  - Variables normalizadas: {self.numeric_features}")
            print(f"  - Método: (x - media) / desviación_estándar")
            print(f"  - Resultado: media ≈ 0, desviación ≈ 1")
        
        return df_normalized
    
    
    def encode_categorical(self, df):
        """
        Codificamos las variables categóricas usando One-Hot Encoding.
        
        One-Hot Encoding convierte categorías en columnas binarias.
        Ejemplo: 'metodo_pago' con valores ['Efectivo', 'Tarjeta']
                 se convierte en dos columnas: [metodo_pago_Tarjeta]
        """
        df_encoded = df.copy()
        
        if not self.categorical_features:
            print("\n✓ No hay variables categóricas para codificar")
            return df_encoded
        
        # Separar numéricas y categóricas
        df_numeric = df_encoded[self.numeric_features]
        df_categorical = df_encoded[self.categorical_features]
        
        # Aplicar One-Hot Encoding
        encoded_array = self.encoder.fit_transform(df_categorical)
        
        # Obtener nombres de las nuevas columnas
        encoded_feature_names = self.encoder.get_feature_names_out(
            self.categorical_features
        )
        
        # Crear DataFrame con variables codificadas
        df_encoded_cat = pd.DataFrame(
            encoded_array,
            columns=encoded_feature_names,
            index=df_encoded.index
        )
        
        # Combinar numéricas con categóricas codificadas
        df_final = pd.concat([df_numeric, df_encoded_cat], axis=1)
        
        print(f"\n✓ Variables categóricas codificadas (One-Hot Encoding):")
        print(f"  - Variables originales: {self.categorical_features}")
        print(f"  - Nuevas columnas creadas: {len(encoded_feature_names)}")
        print(f"  - Nombres: {list(encoded_feature_names)}")
        
        # Guardar nombres para referencia
        self.processed_feature_names = list(df_final.columns)
        
        return df_final
    
    
    def preprocess_pipeline(self, df, outlier_method='iqr', outlier_params=None):
        """
        Pipeline completo de preprocesamiento.
        Ejecuta todos los pasos en orden lógico.
        """
        print("="*70)
        print("INICIANDO PIPELINE DE PREPROCESAMIENTO")
        print("="*70)
        
        # Paso 1: Seleccionar características
        print("\n[1/4] Selección de características...")
        df_selected = self.select_features(df)
        
        # Paso 2: Manejar outliers
        print(f"\n[2/4] Manejo de valores atípicos (método: {outlier_method})...")
        if outlier_params is None:
            outlier_params = {}
        
        if outlier_method == 'iqr':
            df_clean = self.handle_outliers_iqr(df_selected, **outlier_params)
        elif outlier_method == 'zscore':
            df_clean = self.handle_outliers_zscore(df_selected, **outlier_params)
        else:
            print(f"  ⚠ Método '{outlier_method}' no reconocido, omitiendo paso")
            df_clean = df_selected
        
        # Paso 3: Codificar categóricas
        print(f"\n[3/4] Codificación de variables categóricas...")
        df_encoded = self.encode_categorical(df_clean)
        
        # Paso 4: Normalizar
        print(f"\n[4/4] Normalización de características numéricas...")
        df_final = self.normalize_features(df_encoded)
        
        print("\n" + "="*70)
        print("PREPROCESAMIENTO COMPLETADO")
        print("="*70)
        print(f"Dimensiones finales: {df_final.shape}")
        print(f"Total de características: {df_final.shape[1]}")
        
        return df_final
    
    
    def get_preprocessing_summary(self, df_original, df_processed):
        summary = {
            'registros_originales': df_original.shape[0],
            'registros_procesados': df_processed.shape[0],
            'columnas_originales': df_original.shape[1],
            'columnas_procesadas': df_processed.shape[1],
            'features_numericas': len(self.numeric_features),
            'features_categoricas': len(self.categorical_features),
            'features_finales': list(df_processed.columns)
        }
        return summary

