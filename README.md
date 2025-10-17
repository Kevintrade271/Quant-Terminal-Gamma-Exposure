# Quant Terminal - Gamma Exposure Analysis

**Fork & Reestructuración** https://github.com/Kevintrade271/Quant-Terminal-Gamma-Exposure

analisis de gamma exposure y volatilidad implícita SPY/QQQ etc
## Qué hace esto

- **Análisis GEX**: Calcula gamma exposure de dealers para encontrar niveles clave
- **Gamma Flip**: View cambio de dirección de régimen (momentum vs mean-reversion)
- **Call/Put Walls**: Resistencias y soportes probabilísticos
- **Dashboard**: Call API cada 5 minutos


## Instalación rápida

```bash
# Clonar
git clone https://github.com/TU_USUARIO/Quant-Terminal-Gamma-Exposure.git
cd Quant-Terminal-Gamma-Exposure

# Crear entorno virtual
python3 -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt
```

### Dashboard

```bash
source venv/bin/activate
python dashboards/unified_dashboard.py
```

Open http://localhost:8050

**Todo en uno**
- GEX & CHARM: Perfiles de gamma exposure y delta decay
- Volatilidad IV: Mapa de calor de volatilidad implícita


### Scripts individuales por ticker

```bash
source venv/bin/activate

# SPY
python spx.py --ticker SPY

# QQQ
python ndx.py --ticker QQQ

# Modo ODTE (solo opciones del día)
python spx.py --ticker SPY --odte_only

# Modo loop (actualiza cada 5 min y guarda PNGs)
python spx.py --ticker SPY --loop 300 --out_prefix output/images/spy --no_show
```

## Estructura (reestructurada)

```
├── dashboards/              # Dashboards web
│   ├── unified_dashboard.py # (todo en uno)
│   ├── trace.py            # GEX/CHARM individual
│   └── htm.py              # Volatilidad individual
├── spx.py, ndx.py, etc.    # Scripts de análisis por ticker
├── experimental/            # Scripts de prueba
├── output/                  # Aquí se guardan los gráficos
├── docs/                    # Documentación
└── venv/                    # Entorno virtual
```

**Gamma Positivo (verde)**: Dealers frenan el precio → tradea reversiones
**Gamma Negativo (rojo)**: Dealers aceleran el precio → tradea momentum
**Gamma Flip**: El punto donde cambia de uno a otro
**Put Wall**: Soporte fuerte (mucho gamma largo)
**Call Wall**: Resistencia fuerte (mucho gamma corto)

## Dependencias

- yfinance: Para descargar datos de Yahoo Finance
- plotly + dash: Para los dashboards interactivos
- numpy/scipy: Para cálculos Black-Scholes
- pandas: Para manejar datos

 `requirements.txt`

## Problemas comunes

**"No module named X"**: No activaste el venv → `source venv/bin/activate`

**"No se puede obtener datos"**: Yahoo Finance a veces falla, espera un minuto y prueba de nuevo

**El dashboard no abre**: Asegúrate que el puerto 8050 no esté ocupado

## Créditos

- Proyecto original de Kevin Trade 271: https://github.com/Kevintrade271/Quant-Terminal-Gamma-Exposure

## Contribuciones

- Toda contribución es bienvenida, si es muy grande hacer Issues primero.
