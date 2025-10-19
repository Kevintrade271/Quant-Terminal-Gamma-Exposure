# 🎯 Quant Terminal - Dashboard de Gamma Exposure

**Versión:** 2.1.0
**Estado:** Production-Ready ✅

Dashboard profesional para análisis de **Gamma Exposure (GEX)**, **Charm**, **Volatilidad Implícita** y **Niveles Operativos** de opciones sobre ETFs.

**Arquitectura:** FastAPI Backend + Next.js 15 Frontend

---

## 📋 Tabla de Contenidos

1. [¿Qué es este proyecto?](#qué-es-este-proyecto)
2. [Características principales](#características-principales)
3. [Instalación paso a paso (Windows)](#instalación-paso-a-paso-windows)
4. [Instalación paso a paso (Mac/Linux)](#instalación-paso-a-paso-maclinux)
5. [Cómo usar el dashboard](#cómo-usar-el-dashboard)
6. [Funcionalidades detalladas](#funcionalidades-detalladas)
7. [API Endpoints](#api-endpoints)
8. [Conceptos para traders](#conceptos-para-traders)
9. [Troubleshooting](#troubleshooting)
10. [Estructura del proyecto](#estructura-del-proyecto)

---

## 🎯 ¿Qué es este proyecto?

Este dashboard te permite analizar la exposición gamma de los **market makers** (dealers) para identificar niveles clave de soporte/resistencia en el mercado de opciones.

### ¿Por qué es importante la Gamma Exposure?

Cuando compras opciones, los **dealers** tienen que cubrirse (hedge) comprando o vendiendo el activo subyacente. El **GEX (Gamma Exposure)** mide cuánto tienen que comprar/vender cuando el precio se mueve.

- **Gamma Positivo:** Los dealers FRENAN los movimientos del precio (mejor para mean reversion)
- **Gamma Negativo:** Los dealers ACELERAN los movimientos del precio (mejor para momentum/tendencias)

Este dashboard te muestra:
- **Gamma Flip Point:** Nivel donde cambia de régimen positivo a negativo
- **Call Wall:** Resistencia fuerte (dealers compran agresivamente si sube)
- **Put Wall:** Soporte fuerte (dealers venden agresivamente si baja)
- **Operational Levels:** 4 niveles clave para operar

---

## ✨ Características principales

### 📊 4 Vistas Principales

1. **GEX & CHARM** - Barras espejo de exposición gamma y decay temporal
2. **Volatilidad IV** - Heatmap de volatilidad implícita por strike y expiración
3. **IV Skew** - Análisis del sesgo de volatilidad (miedo direccional)
4. **Advanced Analytics** - Perfil gamma continuo + niveles operativos

### 🎨 Features Implementadas (v2.1.0)

- ✅ **Selector de ticker dinámico** - SPY, QQQ, DIA, IWM, GLD, SLV
- ✅ **Modo ODTE** - Filtra solo opciones que expiran hoy (0 DTE)
- ✅ **Gamma Flip Point** - Línea amarilla discontinua en gráficos
- ✅ **Call/Put Walls** - Marcadores de resistencia/soporte
- ✅ **Net Gamma Profile** - Perfil continuo interpolado con scipy
- ✅ **Operational Levels** - Slip Risk, Gamma Pin, Exhaustion, Convex Hotspot
- ✅ **Export to PNG** - Descarga cualquier gráfico como imagen ⭐ NUEVO
- ✅ **Refresh Manual** - Botón para actualizar datos inmediatamente ⭐ NUEVO
- ✅ **Auto-refresh** - Actualización automática cada 5 minutos
- ✅ **VIX Integration** - Z-Score del VIX para contexto de mercado

---

## 🪟 Instalación paso a paso (Windows)

### Requisitos previos

1. **Python 3.10 o superior**
   - Descargar de: https://www.python.org/downloads/
   - ⚠️ IMPORTANTE: Durante la instalación, marcar "Add Python to PATH"

2. **Node.js 18 o superior**
   - Descargar de: https://nodejs.org/
   - Instalará automáticamente npm

3. **Git** (opcional, para clonar el repo)
   - Descargar de: https://git-scm.com/download/win

### Paso 1: Descargar el proyecto

**Opción A: Con Git**
```cmd
git clone https://github.com/tu-usuario/Quant-Terminal-Gamma-Exposure.git
cd Quant-Terminal-Gamma-Exposure
```

**Opción B: Sin Git**
- Descargar el ZIP desde GitHub
- Extraer en una carpeta (ej: `C:\Users\TuUsuario\Quant-Terminal`)
- Abrir CMD y navegar a esa carpeta:
```cmd
cd C:\Users\TuUsuario\Quant-Terminal-Gamma-Exposure
```

### Paso 2: Configurar el Backend (Python)

1. **Abrir CMD en la carpeta del proyecto**

2. **Crear entorno virtual:**
```cmd
python -m venv venv
```

3. **Activar entorno virtual:**
```cmd
venv\Scripts\activate
```
   - Tu terminal ahora debe mostrar `(venv)` al inicio

4. **Instalar dependencias:**
```cmd
pip install -r requirements.txt
```
   - Esto puede tomar 2-3 minutos

5. **Iniciar el backend:**
```cmd
cd backend
python main.py
```

✅ **El backend está listo cuando veas:**
```
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Application startup complete.
```

⚠️ **NO CIERRES ESTA VENTANA** - Déjala abierta mientras uses el dashboard

### Paso 3: Configurar el Frontend (Next.js)

1. **Abrir OTRA ventana de CMD** (la primera debe seguir corriendo)

2. **Navegar a la carpeta del proyecto:**
```cmd
cd C:\Users\TuUsuario\Quant-Terminal-Gamma-Exposure
```

3. **Ir a la carpeta frontend:**
```cmd
cd frontend
```

4. **Instalar dependencias:**
```cmd
npm install
```
   - Esto puede tomar 3-5 minutos la primera vez

5. **Iniciar el frontend:**
```cmd
npm run dev
```

✅ **El frontend está listo cuando veas:**
```
▲ Next.js 15.5.6 (Turbopack)
- Local:        http://localhost:3000
✓ Ready in 1562ms
```

### Paso 4: Abrir el Dashboard

Abre tu navegador y ve a:
```
http://localhost:3000
```

o si el puerto 3000 estaba ocupado:
```
http://localhost:3001
```

🎉 **¡Ya está funcionando!**

---

## 🍎 Instalación paso a paso (Mac/Linux)

### Requisitos previos

1. **Python 3.10+**
```bash
python3 --version
# Si no está instalado: brew install python@3.10 (Mac)
```

2. **Node.js 18+**
```bash
node --version
# Si no está instalado: brew install node (Mac)
```

### Instalación

1. **Clonar repositorio:**
```bash
git clone https://github.com/tu-usuario/Quant-Terminal-Gamma-Exposure.git
cd Quant-Terminal-Gamma-Exposure
```

2. **Backend:**
```bash
# Crear y activar entorno virtual
python3 -m venv venv
source venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt

# Iniciar backend
cd backend
python main.py
```

3. **Frontend (en otra terminal):**
```bash
cd frontend
npm install
npm run dev
```

4. **Abrir navegador:**
```
http://localhost:3000
```

---

## 🎮 Cómo usar el dashboard

### Interfaz Principal

1. **Selector de Ticker** - Elige entre SPY, QQQ, DIA, IWM, GLD, SLV
2. **Modo ODTE** - Toggle para filtrar solo opciones del día (0 DTE)
3. **Botón Refresh** - Actualiza datos manualmente sin esperar
4. **Tabs** - Navega entre las 4 vistas principales

### Exportar Gráficos

Cada gráfico tiene un botón **"Exportar PNG"** en la esquina superior derecha:
- Click en el botón
- El archivo se descarga automáticamente
- Nombre del archivo: `{TICKER}_{TIPO}_{FECHA}.png`
- Ejemplo: `SPY_GEX_2025-10-19.png`

### Interpretar los Gráficos

#### GEX & CHARM
- **Barras verdes (derecha):** Calls - Dealers venden cuando sube
- **Barras rojas (izquierda):** Puts - Dealers compran cuando baja
- **Línea amarilla discontinua:** Gamma Flip Point
- **Línea morada:** Call Wall (resistencia)
- **Línea roja:** Put Wall (soporte)

#### Volatility IV
- **Verde:** Alta volatilidad (miedo elevado, VIX alto)
- **Rojo:** Baja volatilidad (complacencia, VIX bajo)
- **Gris:** Volatilidad neutral

#### IV Skew
- **Curva pronunciada:** Alto miedo direccional
- **Curva plana:** Mercado neutral
- **Puts > Calls:** Miedo a caídas (normal en índices)

#### Advanced Analytics
- **Net Gamma Profile:** Línea azul continua con la exposición total
- **Operational Levels Panel:** 4 niveles clave para operar

---

## 🎯 Funcionalidades detalladas

### 1. Selector de Ticker Dinámico ✅

**Tickers disponibles:**
- **SPY** - S&P 500 ETF (más líquido)
- **QQQ** - Nasdaq 100 ETF (tech)
- **DIA** - Dow Jones ETF
- **IWM** - Russell 2000 ETF (small caps)
- **GLD** - Gold ETF (oro)
- **SLV** - Silver ETF (plata)

**Uso:** Selecciona del dropdown, cambia sin recargar la página.

### 2. Modo ODTE (Zero Days To Expiration) ✅

**Qué es:** Filtra solo opciones que expiran HOY.

**Por qué es útil:**
- Day traders operan opciones que expiran el mismo día
- Gamma extremadamente alta cerca de expiración
- Movimientos más explosivos

**Uso:** Toggle switch "Modo ODTE" ON/OFF

### 3. Gamma Flip Point Indicator ✅

**Qué es:** Nivel de precio donde el gamma total cambia de positivo a negativo.

**Cómo se calcula:**
```python
# Busca donde el GEX neto cruza cero
gamma_flip = strike donde sum(GEX) ≈ 0
```

**Trading:**
- **Arriba del Gamma Flip:** Dealers frenan movimientos (mean reversion)
- **Abajo del Gamma Flip:** Dealers aceleran movimientos (momentum)

**Visual:** Línea amarilla discontinua en gráficos GEX

### 4. Call/Put Wall Markers ✅

**Call Wall (morado):**
- Strike con máximo GEX positivo (calls)
- Resistencia fuerte
- Dealers tienen que COMPRAR si el precio sube
- Cálculo: `max(GEX_calls)`

**Put Wall (rojo):**
- Strike con máximo GEX negativo (puts)
- Soporte fuerte
- Dealers tienen que VENDER si el precio baja
- Cálculo: `max(GEX_puts)`

**Trading:**
- Usar como objetivos de profit
- Esperar rebotes en las walls
- Breakout violento si se cruzan

### 5. IV Skew Chart ✅

**Qué muestra:** Volatilidad implícita vs moneyness (Strike/Spot)

**Interpretación:**
- **Skew pronunciado hacia puts:** Miedo a caídas (normal en índices)
- **Skew plano:** Mercado neutral
- **Skew invertido:** Miedo a subidas (raro, ocurre en commodities)

**Moneyness:**
- < 1.0: In-the-money puts / Out-of-the-money calls
- = 1.0: At-the-money
- > 1.0: Out-of-the-money puts / In-the-money calls

### 6. Advanced Analytics ✅

#### a) Net Gamma Profile (Perfil Continuo)

**Qué es:** Interpolación suave del gamma neto total.

**Cómo se calcula:**
```python
# Usa interpolación cúbica sobre los strikes
from scipy.interpolate import interp1d
f = interp1d(strikes, gex_values, kind='cubic')
smooth_profile = f(dense_grid)
```

**Por qué es útil:**
- Vista más clara que barras discretas
- Identifica zonas de acumulación gamma
- Muestra la "forma" del perfil de riesgo

**Visual:** Línea azul continua con relleno

#### b) Operational Levels Panel (Niveles Operativos)

4 niveles clave basándose en la estructura gamma:

**🔴 Slip Risk (Put Wall)**
- **Definición:** Strike con máximo GEX negativo (puts)
- **Significado:** Soporte fuerte donde dealers venden si baja el precio
- **Trading:** Zona de rebote probable
- **Cálculo:** `max(GEX_total)` donde GEX < 0

**🟡 Gamma Pin**
- **Definición:** Zona de máxima concentración gamma cerca del spot
- **Significado:** El precio tiende a "pegarse" aquí
- **Trading:** Zona de consolidación esperada
- **Cálculo:** `max(|GEX|)` en ventana ±2% del spot

**🟣 Exhaustion (Call Wall)**
- **Definición:** Strike con máximo GEX positivo (calls)
- **Significado:** Resistencia fuerte donde dealers compran si sube el precio
- **Trading:** Techo probable a corto plazo
- **Cálculo:** `min(GEX_total)` donde GEX > 0

**🟢 Convex Hotspot**
- **Definición:** Zona de máxima curvatura en el perfil gamma
- **Significado:** Movimientos explosivos si se cruza este nivel
- **Trading:** Zona de breakout potencial
- **Cálculo:** Segunda derivada del perfil gamma

### 7. Export to PNG ⭐ NUEVO (v2.1.0)

**Qué es:** Botón para descargar cualquier gráfico como imagen PNG.

**Dónde:** Esquina superior derecha de cada gráfico.

**Implementación:**
- Componente reutilizable: `ExportChartButton.tsx`
- Usa Chart.js `toDataURL()` para gráficos canvas
- Usa `html2canvas` para elementos HTML (Volatility Heatmap)
- Nombre del archivo incluye ticker y fecha automáticamente

**Beneficio:** Compartir análisis fácilmente en Twitter, Discord, reportes, etc.

**Uso:**
1. Navega al gráfico que quieres exportar
2. Click en "Exportar PNG"
3. El archivo se descarga automáticamente

### 8. Refresh Manual ⭐ NUEVO (v2.1.0)

**Qué es:** Botón para actualizar datos sin esperar el auto-refresh (5 min).

**Dónde:** Junto al selector de ticker y modo ODTE.

**Implementación:** IconButton con RefreshIcon de Material-UI.

**Beneficio:** Control inmediato sobre cuando actualizar datos durante:
- Anuncios económicos importantes
- Movimientos bruscos del mercado
- Apertura/cierre de sesión

**Uso:** Click en el icono de refresh ⟳

---

## 🔌 API Endpoints

Base URL: `http://localhost:8000`

Documentación interactiva: `http://localhost:8000/docs`

### GET /api/greeks/{ticker}

**Descripción:** Obtiene datos de Gamma (GEX) y Charm.

**Parámetros opcionales:**
- `max_exp` (int): Máximo número de expiraciones (default: 6)
- `r` (float): Tasa libre de riesgo (default: 0.045)
- `q` (float): Dividend yield (default: 0.012)
- `min_oi` (int): Open Interest mínimo (default: 200)
- `odte_only` (bool): Solo opciones del día (default: false)

**Ejemplo:**
```bash
curl "http://localhost:8000/api/greeks/SPY?max_exp=10&min_oi=200&odte_only=false"
```

**Respuesta:**
```json
{
  "spot": 450.25,
  "ticker": "SPY",
  "timestamp": "2025-10-19T12:00:00",
  "gamma_flip": 448.50,
  "call_wall": 455.00,
  "put_wall": 445.00,
  "data": [
    {
      "exp": "2025-10-20",
      "K": 450.0,
      "side": "C",
      "GEX": 1500000.0,
      "CHARM": 25000.0
    }
  ]
}
```

### GET /api/volatility/{ticker}

**Descripción:** Obtiene la matriz de volatilidad implícita.

**Parámetros opcionales:**
- `max_exp` (int): Máximo número de expiraciones (default: 10)
- `strike_span` (int): Rango de strikes alrededor del spot (default: 40)
- `max_cols` (int): Máximo número de columnas (default: 25)
- `min_oi` (int): Open Interest mínimo (default: 100)

**Ejemplo:**
```bash
curl "http://localhost:8000/api/volatility/QQQ"
```

### GET /api/status/{ticker}

**Descripción:** Estado actual del ticker y VIX.

**Ejemplo:**
```bash
curl "http://localhost:8000/api/status/SPY"
```

**Respuesta:**
```json
{
  "ticker": "SPY",
  "spot": 450.25,
  "vix_current": 18.5,
  "vix_zscore": -0.35,
  "timestamp": "2025-10-19T12:00:00"
}
```

### GET /api/iv-skew/{ticker}

**Descripción:** Obtiene datos de IV Skew.

**Parámetros opcionales:**
- `max_exp` (int): Máximo número de expiraciones (default: 6)
- `min_oi` (int): Open Interest mínimo (default: 200)

**Ejemplo:**
```bash
curl "http://localhost:8000/api/iv-skew/SPY"
```

### GET /api/advanced-analytics/{ticker}

**Descripción:** Análisis avanzado con perfil gamma y niveles operativos.

**Parámetros opcionales:**
- `max_exp` (int): Máximo número de expiraciones (default: 6)
- `r` (float): Tasa libre de riesgo (default: 0.045)
- `q` (float): Dividend yield (default: 0.012)
- `min_oi` (int): Open Interest mínimo (default: 200)
- `odte_only` (bool): Solo opciones del día (default: false)

**Ejemplo:**
```bash
curl "http://localhost:8000/api/advanced-analytics/SPY?odte_only=false"
```

**Respuesta:**
```json
{
  "spot": 450.25,
  "ticker": "SPY",
  "timestamp": "2025-10-19T12:00:00",
  "gamma_levels": {
    "gamma_flip": 448.50,
    "call_wall": 455.00,
    "put_wall": 445.00,
    "call_wall_gex": 1500000,
    "put_wall_gex": -2000000
  },
  "net_gamma_profile": {
    "strikes": [440.0, 440.5, 441.0, ...],
    "net_gamma": [120000, 150000, ...]
  },
  "operational_levels": {
    "slip_risk": 445.00,
    "gamma_pin": 450.00,
    "exhaustion": 455.00,
    "convex_hotspot": 452.50
  },
  "volume_by_strike": {
    "450": {
      "call_volume": 50000,
      "put_volume": 30000,
      "call_oi": 100000,
      "put_oi": 80000
    }
  }
}
```

---

## 📚 Conceptos para traders

### ¿Cómo funciona el hedging de dealers?

Cuando **tú** compras una opción call:
1. El dealer te vende esa call
2. Para cubrirse (hedge), compra acciones del subyacente
3. Si el precio **sube**, tiene que comprar **más** acciones (gamma hedging)
4. Esto **acelera** el movimiento alcista

Cuando **tú** compras una opción put:
1. El dealer te vende esa put
2. Para cubrirse, vende acciones en corto
3. Si el precio **baja**, tiene que vender **más** acciones
4. Esto **acelera** el movimiento bajista

### Gamma Positivo vs Negativo

**Gamma Positivo (arriba del Gamma Flip):**
- Los dealers están largos gamma
- Tienen que vender cuando sube, comprar cuando baja
- **Efecto:** FRENAN los movimientos (reversión a la media)
- **Trading:** Fades, contrarian, sell premium

**Gamma Negativo (abajo del Gamma Flip):**
- Los dealers están cortos gamma
- Tienen que comprar cuando sube, vender cuando baja
- **Efecto:** ACELERAN los movimientos (momentum)
- **Trading:** Breakouts, tendencias, buy premium

### VIX Z-Score

**Qué es:** Mide cuántas desviaciones estándar está el VIX de su media.

**Interpretación:**
- **Z > +2.0:** Pánico extremo (oportunidad de compra)
- **Z > +1.0:** Miedo elevado
- **-1.0 < Z < +1.0:** Neutral
- **Z < -1.0:** Complacencia (peligro)
- **Z < -2.0:** Complacencia extrema (sell-off inminente)

### Uso combinado de niveles

**Ejemplo práctico con SPY:**

Supongamos:
- Spot: $450
- Gamma Flip: $448
- Put Wall: $445
- Call Wall: $455
- Gamma Pin: $450
- VIX Z-Score: +0.5 (neutral-alto)

**Escenario alcista:**
1. Precio sube de $450 → $452
2. Está arriba del Gamma Flip → Dealers frenan
3. Se acerca al Call Wall ($455) → Resistencia fuerte
4. **Estrategia:** Vender calls en $455, esperar reversión

**Escenario bajista:**
1. Precio baja de $450 → $447
2. Cruza el Gamma Flip ($448) → Dealers ahora aceleran
3. Próximo nivel: Put Wall ($445) → Soporte fuerte
4. **Estrategia:** Comprar puts, target $445

---

## 🐛 Troubleshooting

### Windows

**Error: "python no se reconoce como comando"**
```cmd
# Python no está en PATH
# Reinstalar Python y marcar "Add to PATH"
# O usar ruta completa:
C:\Users\TuUsuario\AppData\Local\Programs\Python\Python310\python.exe -m venv venv
```

**Error: "venv\Scripts\activate no funciona"**
```cmd
# PowerShell tiene restricciones de ejecución
# Usar CMD en lugar de PowerShell
# O ejecutar en PowerShell:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**Error: "Module not found: scipy"**
```cmd
# Activar entorno virtual primero
venv\Scripts\activate
# Luego reinstalar
pip install --upgrade -r requirements.txt
```

**Error: "npm no se reconoce"**
```
# Node.js no está instalado o no está en PATH
# Descargar e instalar desde nodejs.org
# Reiniciar CMD después de instalar
```

**Error: "Port 8000 already in use"**
```cmd
# Ver qué proceso usa el puerto
netstat -ano | findstr :8000
# Matar el proceso (reemplazar PID)
taskkill /PID <PID> /F
```

**Error: "Port 3000 already in use"**
```
# Next.js usará automáticamente el puerto 3001
# O matar el proceso:
netstat -ano | findstr :3000
taskkill /PID <PID> /F
```

### Mac/Linux

**Error: "Module not found"**
```bash
# Backend
source venv/bin/activate
pip install --upgrade -r requirements.txt

# Frontend
cd frontend
rm -rf node_modules package-lock.json
npm install
```

**Error: "Connection refused" en frontend**
```bash
# Verificar que el backend esté corriendo
curl http://localhost:8000/api/status/SPY

# Revisar CORS en backend/main.py
# Verificar NEXT_PUBLIC_API_URL en frontend/.env.local
```

**Error: Yahoo Finance no responde**
```
# Yahoo Finance tiene límites de tasa
# Esperar 1-2 minutos y reintentar
# Verificar conexión a internet
```

**Error: "Permission denied"**
```bash
# Mac puede bloquear scripts
chmod +x venv/bin/activate
source venv/bin/activate
```

**Puerto ocupado**
```bash
# Mac/Linux
lsof -ti:8000 | xargs kill -9  # Backend
lsof -ti:3000 | xargs kill -9  # Frontend
```

### Errores comunes de datos

**Gráficos vacíos o sin datos**
```
# Posibles causas:
1. Yahoo Finance está caído → Esperar
2. Ticker incorrecto → Usar SPY, QQQ, DIA, IWM, GLD, SLV
3. Mercado cerrado + sin datos ODTE → Desactivar modo ODTE
4. min_oi muy alto → Reducir en API call
```

**Datos desactualizados**
```
# Click en botón Refresh (⟳)
# O esperar al auto-refresh (5 min)
# Verificar que backend esté corriendo
```

**Gamma Flip = null**
```
# Normal si:
1. Todo el GEX es positivo (alcista extremo)
2. Todo el GEX es negativo (bajista extremo)
3. Mercado muy tranquilo
```

---

## 📁 Estructura del proyecto

```
Quant-Terminal-Gamma-Exposure/
│
├── backend/                              # Python FastAPI Backend
│   ├── api/
│   │   ├── routes/
│   │   │   ├── __init__.py
│   │   │   ├── greeks.py                 # Endpoints GEX/CHARM/Advanced
│   │   │   └── volatility.py             # Endpoints IV
│   │   ├── services/
│   │   │   ├── __init__.py
│   │   │   ├── greeks_calculator.py      # Cálculo de greeks
│   │   │   ├── volatility_calculator.py  # Cálculo de IV
│   │   │   ├── iv_skew_calculator.py     # Cálculo de IV Skew
│   │   │   └── advanced_analytics.py     # Perfil gamma + niveles
│   │   └── models/
│   │       ├── __init__.py
│   │       └── schemas.py                # Pydantic schemas
│   ├── main.py                           # Punto de entrada FastAPI
│   └── requirements.txt
│
├── frontend/                             # Next.js 15 Frontend
│   ├── app/
│   │   ├── page.tsx                      # Dashboard principal
│   │   ├── layout.tsx
│   │   └── globals.css
│   ├── components/
│   │   ├── GexCharmChart.tsx             # Gráficos GEX/CHARM
│   │   ├── VolatilityHeatmap.tsx         # Heatmap IV
│   │   ├── IVSkewChart.tsx               # Gráfico IV Skew
│   │   ├── AdvancedAnalyticsView.tsx     # Vista analytics avanzados
│   │   ├── OperationalLevelsPanel.tsx    # Panel niveles operativos
│   │   └── ExportChartButton.tsx         # Botón export PNG
│   ├── lib/
│   │   ├── api.ts                        # Cliente HTTP (axios)
│   │   ├── types.ts                      # TypeScript types
│   │   └── theme.ts                      # Material-UI theme
│   ├── package.json
│   └── tsconfig.json
│
├── spx.py, ndx.py, dia.py, ...           # Scripts standalone Python
├── requirements.txt                       # Dependencias Python
├── README.md                              # Este archivo
└── NUEVAS_FUNCIONALIDADES.md              # [DEPRECADO] Movido a README
```

---

## 🛠 Stack Tecnológico

### Backend
- **FastAPI** - Framework web moderno con documentación automática
- **Pydantic** - Validación de datos y schemas
- **NumPy** - Cálculos numéricos
- **SciPy** - Interpolación cúbica para perfil gamma
- **Pandas** - Manipulación de datos
- **yfinance** - Datos de opciones de Yahoo Finance
- **Uvicorn** - ASGI server

### Frontend
- **Next.js 15** - React framework con Turbopack
- **TypeScript** - Type safety
- **Material-UI (MUI)** - Componentes UI
- **Tailwind CSS** - Utility-first CSS
- **Chart.js** - Gráficos interactivos
- **react-chartjs-2** - Wrapper de Chart.js para React
- **chartjs-plugin-annotation** - Anotaciones en gráficos
- **html2canvas** - Export de HTML a imagen
- **Axios** - Cliente HTTP
- **SWR** - Data fetching hooks

---

## 🚀 Roadmap

### ✅ Implementado (v2.1.0)

- Backend FastAPI con endpoints REST
- Frontend Next.js con TypeScript
- Gráficos GEX, CHARM, IV Heatmap, IV Skew
- Advanced Analytics (Net Gamma Profile + Operational Levels)
- Actualización automática cada 5 minutos
- Refresh manual
- Export to PNG
- Multi-ticker support (SPY, QQQ, DIA, IWM, GLD, SLV)
- Modo ODTE
- Integración VIX con Z-Score
- Gamma Flip Point indicator
- Call/Put Wall markers
- Documentación Swagger completa

### 🔄 En desarrollo

- **Volume Overlay** - Superposición de volumen en gráficos GEX (backend listo)
- **Dealer Convention Toggle** - Cambiar entre convenciones de signo (spotgamma vs short_both)

### 📋 Próximas mejoras

**Complejidad Media:**
- **Price PDF (Breeden-Litzenberger)** - Distribución de probabilidad de precios
- **Zoom/Pan en gráficos** - Interactividad mejorada
- **Themes** - Light/Dark mode toggle
- **Configuración persistente** - Guardar preferencias del usuario

**Complejidad Alta:**
- **Historical Data** - Tracking temporal de niveles gamma
- **Time Series Charts** - Evolución de Gamma Flip, Walls, VIX
- **Alerts System** - Notificaciones cuando precio cruza niveles
- **WebSockets** - Updates en tiempo real sin polling
- **Database** - PostgreSQL/SQLite para histórico
- **User Authentication** - Login y configuración por usuario
- **Comparación Multi-ticker** - Ver múltiples tickers simultáneamente
- **Mobile App** - React Native app
- **Backtesting** - Probar estrategias con datos históricos

---

## 🤝 Contribuir

1. Fork del repositorio
2. Crea una rama para tu feature: `git checkout -b feature/nueva-feature`
3. Commit tus cambios: `git commit -m 'Add nueva feature'`
4. Push a la rama: `git push origin feature/nueva-feature`
5. Abre un Pull Request

---

## 📄 Licencia

MIT License

---

## 👥 Créditos

**Proyecto original:** [Kevin Trade 271](https://github.com/Kevintrade271/Quant-Terminal-Gamma-Exposure)

**Refactorización y nuevas features:** v2.0.0 - v2.1.0

---

## ⚠️ Disclaimer

Este software es para **fines educativos y de investigación** únicamente.

**NO constituye asesoramiento financiero.**

El trading de opciones conlleva riesgo sustancial de pérdida. Solo opera con capital que puedas permitirte perder.

Los datos provienen de Yahoo Finance y pueden tener retrasos o inexactitudes.

**Usa bajo tu propio riesgo.**

---

## 📞 Soporte

**Problemas técnicos:**
- Revisa la sección [Troubleshooting](#troubleshooting)
- Abre un Issue en GitHub

**Preguntas sobre trading:**
- Este proyecto NO provee asesoramiento financiero
- Consulta con un profesional financiero certificado

---

## 📈 Ejemplos de uso

### Estrategia 1: Fade en Gamma Positivo

```
Situación:
- SPY @ $450
- Gamma Flip @ $448
- Put Wall @ $445
- Precio arriba del Gamma Flip (gamma positivo)

Setup:
- Mercado abre con gap up a $452
- RSI > 70 (sobrecomprado)
- Cerca del Call Wall ($455)

Operación:
- Vender call spreads $455/$460
- Take profit en Put Wall ($445)
- Stop loss si cruza Call Wall con volumen
```

### Estrategia 2: Momentum en Gamma Negativo

```
Situación:
- QQQ @ $380
- Gamma Flip @ $385
- Precio abajo del Gamma Flip (gamma negativo)
- VIX Z-Score +1.5 (miedo)

Setup:
- Precio rompe Put Wall con volumen
- MACD cruza bajista
- Convex Hotspot roto

Operación:
- Comprar puts ATM
- Target: Próximo soporte técnico
- Stop: Volver arriba del Gamma Flip
```

### Estrategia 3: Iron Condor en Gamma Pin

```
Situación:
- SPY @ $450
- Gamma Pin @ $450 (máxima concentración)
- VIX bajo (< 15)
- Rango estrecho

Setup:
- Mercado lateral por varios días
- IV crushed
- Precio oscila ±1% del Gamma Pin

Operación:
- Vender Iron Condor centrado en Gamma Pin
- Short $445 put / $455 call
- Long $440 put / $460 call
- Aprovechar theta decay
```

---

**Última actualización:** 19 de Octubre, 2025
**Versión:** 2.1.0
**Estado:** Production-Ready ✅
