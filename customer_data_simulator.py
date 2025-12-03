import numpy as np
import pandas as pd

class CustomerDataSimulator:
    """
    Clase para generar dataset sintético de clientes.
    
    Generar 7 características principales:
    - Edad
    - Ingreso mensual
    - Frecuencia de compra
    - Monto promedio por compra
    - Antigüedad (días como cliente)
    - Método de pago preferido
    - Nivel de satisfacción
    """
    
    def __init__(self, random_state=42):
        """
        Inicializa el simulador de datos.
        
        Args:
            random_state (int): Semilla para reproducibilidad de resultados
        """
        self.random_state = random_state
        np.random.seed(random_state)
    
    def generate_customer_data(self, n_samples=2000):
        # ====================================================================
        # 1. EDAD - Distribución Normal (18-70 años)
        # ====================================================================
        # Mayoría de clientes entre 25-45 años (típico en comercio)
        edad = np.random.normal(loc=35, scale=12, size=n_samples)
        edad = np.clip(edad, 18, 70).astype(int)  # Limitar rango
        
        
        # ====================================================================
        # 2. INGRESO MENSUAL - Distribución Log-Normal ($500 - $50,000)
        # ====================================================================
        # Simula ingresos reales: mayoría en rango medio, pocos muy altos
        ingreso_mensual = np.random.lognormal(mean=9.0, sigma=0.6, size=n_samples)
        ingreso_mensual = np.clip(ingreso_mensual, 500, 50000)
        ingreso_mensual = np.round(ingreso_mensual, 2)
        
        
        # ====================================================================
        # 3. FRECUENCIA DE COMPRA - Distribución Poisson (1-12 compras/mes)
        # ====================================================================
        # Simula comportamiento típico: pocas compras frecuentes, muchas ocasionales
        frecuencia_compra = np.random.poisson(lam=4, size=n_samples)
        frecuencia_compra = np.clip(frecuencia_compra, 1, 12)
        
        
        # ====================================================================
        # 4. MONTO PROMEDIO - Distribución Gamma correlacionada con ingreso
        # ====================================================================
        # Clientes con más ingreso tienden a gastar más (correlación realista)
        base_amount = np.random.gamma(shape=2, scale=200, size=n_samples)
        # Ajustar según ingreso (factor de correlación)
        income_factor = ingreso_mensual / ingreso_mensual.mean()
        monto_promedio = base_amount * income_factor * 0.5
        monto_promedio = np.clip(monto_promedio, 10, 5000)
        monto_promedio = np.round(monto_promedio, 2)
        
        
        # ====================================================================
        # 5. ANTIGÜEDAD - Distribución Exponencial (30-2000 días)
        # ====================================================================
        # Más clientes nuevos que antiguos (típico en negocios en crecimiento)
        antiguedad_dias = np.random.exponential(scale=400, size=n_samples)
        antiguedad_dias = np.clip(antiguedad_dias, 30, 2000).astype(int)
        
        
        # ====================================================================
        # 6. MÉTODO DE PAGO - Distribución Categórica (5 opciones)
        # ====================================================================
        # Refleja preferencias actuales de pago
        metodos_pago = [
            'Tarjeta Crédito',    # 35% - Más popular
            'Tarjeta Débito',     # 30% - Segundo más usado
            'Transferencia',      # 15% - Creciendo
            'Efectivo',           # 12% - En declive
            'Billetera Digital'   # 8%  - Emergente
        ]
        probabilidades = [0.35, 0.30, 0.15, 0.12, 0.08]
        metodo_pago = np.random.choice(
            metodos_pago, 
            size=n_samples, 
            p=probabilidades
        )
        
        
        # ====================================================================
        # 7. NIVEL DE SATISFACCIÓN - Distribución Beta (1-10)
        # ====================================================================
        # Mayoría satisfechos (7-9), pocos insatisfechos (sesgo positivo realista)
        satisfaccion = np.random.beta(a=8, b=2, size=n_samples)
        nivel_satisfaccion = satisfaccion * 9 + 1  # Escalar a 1-10
        nivel_satisfaccion = np.round(nivel_satisfaccion, 1)
        
        
        # ====================================================================
        # CONSTRUCCIÓN DEL DATAFRAME
        # ====================================================================
        df = pd.DataFrame({
            'edad': edad,
            'ingreso_mensual': ingreso_mensual,
            'frecuencia_compra': frecuencia_compra,
            'monto_promedio': monto_promedio,
            'antiguedad_dias': antiguedad_dias,
            'metodo_pago': metodo_pago,
            'nivel_satisfaccion': nivel_satisfaccion
        })
        
        return df
    
    
    def get_dataset_info(self, df):
        info = {
            'total_clientes': df.shape[0],
            'total_variables': df.shape[1],
            'nombres_columnas': list(df.columns),
            'tipos_datos': df.dtypes.to_dict(),
            'valores_faltantes': df.isnull().sum().to_dict(),
            'memoria_usada_mb': round(df.memory_usage(deep=True).sum() / 1024**2, 2)
        }
        return info
    
    
    def get_statistics_summary(self, df):
        return df.describe()
