# Quant Terminal - Gamma Exposure Analysis

**Arquitectura moderna:** FastAPI Backend + Next.js Frontend

Fork y refactorización del proyecto original de [Kevin Trade 271](https://github.com/Kevintrade271/Quant-Terminal-Gamma-Exposure)

Dashboard para análisis de **Gamma Exposure (GEX)**, **Charm** y **Volatilidad Implícita** de opciones sobre ETFs (SPY, QQQ, DIA, etc.)

---

## ¿Qué hace este proyecto?

### Análisis de Gamma Exposure (GEX)
Calcula la exposición gamma de los dealers para identificar niveles de soporte/resistencia clave. Detecta el **Gamma Flip Point** (cambio de régimen) y los **Call/Put Walls**.

### Análisis de Charm
Mide el decay del delta respecto al tiempo, ayudando a entender cómo cambiará la cobertura de los dealers.

### Mapa de Volatilidad Implícita
Heatmap de IV por strike y expiración, con integración del VIX para análisis de régimen de mercado.

### Dashboard en Tiempo Real
Actualización automática cada 5 minutos con visualizaciones interactivas y API REST documentada.

---

## Instalación Rápida

### Requisitos
- Python 3.10+
- Node.js 18+ y npm
- Conexión a internet (Yahoo Finance)

### Backend (FastAPI)

```bash
# Clonar repositorio
git clone https://github.com/Kevintrade271/Quant-Terminal-Gamma-Exposure.git
cd Quant-Terminal-Gamma-Exposure

# Crear entorno virtual e instalar dependencias
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Iniciar servidor backend
cd backend
python main.py
```

Backend disponible en: http://localhost:8000
Documentación API: http://localhost:8000/docs

### Frontend (Next.js)

```bash
# En otra terminal
cd frontend
npm install
npm run dev
```

Dashboard disponible en: http://localhost:3000

---

## Estructura del Proyecto

```
Quant-Terminal-Gamma-Exposure/
│
├── backend/                          # Python FastAPI Backend
│   ├── api/
│   │   ├── routes/                   # Endpoints REST
│   │   ├── services/                 # Lógica de cálculo
│   │   └── models/                   # Schemas Pydantic
│   └── main.py
│
├── frontend/                         # Next.js Frontend
│   ├── app/
│   │   ├── page.tsx                  # Dashboard principal
│   │   ├── layout.tsx
│   │   └── globals.css
│   ├── components/
│   │   ├── GexCharmChart.tsx         # Gráficos GEX/CHARM
│   │   └── VolatilityHeatmap.tsx     # Heatmap IV
│   └── lib/
│       ├── api.ts                    # Cliente HTTP
│       ├── types.ts
│       └── theme.ts
│
├── dashboards/                       # [LEGACY] Dashboard Python/Dash original
└── spx.py, ndx.py, ...              # Scripts individuales por ticker
```

---

## Stack Tecnológico

**Backend:** FastAPI, Pydantic, NumPy, SciPy, Pandas, yfinance
**Frontend:** Next.js 14, TypeScript, Material-UI, Tailwind CSS, Chart.js, Axios

---

## API Endpoints

Base URL: `http://localhost:8000`

### GET /api/greeks/{ticker}
Obtiene datos de Gamma (GEX) y Charm.

Parámetros opcionales: `max_exp`, `r`, `q`, `min_oi`

```bash
curl "http://localhost:8000/api/greeks/SPY?max_exp=10&min_oi=200"
```

### GET /api/volatility/{ticker}
Obtiene la matriz de volatilidad implícita.

Parámetros opcionales: `max_exp`, `strike_span`, `max_cols`, `min_oi`

```bash
curl "http://localhost:8000/api/volatility/QQQ"
```

### GET /api/status/{ticker}
Estado actual del ticker y VIX.

```bash
curl "http://localhost:8000/api/status/SPY"
```

---

## Conceptos Clave

**Gamma Positivo:** Los dealers frenan el precio (mejor para reversión a la media)
**Gamma Negativo:** Los dealers aceleran el precio (mejor para momentum)
**Gamma Flip Point:** Nivel donde cambia de régimen
**Put Wall:** Soporte fuerte (alto gamma en puts)
**Call Wall:** Resistencia fuerte (alto gamma en calls)

---

## Configuración

### Variables de entorno

Crear `frontend/.env.local`:

```env
NEXT_PUBLIC_API_URL=http://localhost:8000
```

### Cambiar ticker

Editar `frontend/app/page.tsx`:

```typescript
const TICKER = 'QQQ'; // Cambiar de 'SPY' a otro ticker
```

### Cambiar intervalo de actualización

```typescript
const REFRESH_INTERVAL = 3 * 60 * 1000; // 3 minutos
```

---

## Troubleshooting

**Error: "Module not found"**
```bash
# Backend
pip install --upgrade -r requirements.txt

# Frontend
cd frontend && rm -rf node_modules package-lock.json && npm install
```

**Error: "Connection refused" en frontend**
- Verificar que el backend esté corriendo en puerto 8000
- Revisar `NEXT_PUBLIC_API_URL` en `.env.local`
- Verificar CORS en `backend/main.py`

**Error: Yahoo Finance no responde**
- Yahoo Finance tiene límites de tasa. Esperar 1-2 minutos.

**Puerto ocupado**
```bash
lsof -ti:8000 | xargs kill -9  # Backend
lsof -ti:3000 | xargs kill -9  # Frontend
```

---

## Build para Producción

```bash
# Backend
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000

# Frontend
cd frontend
npm run build
npm run start
```

---

## Roadmap

**Implementado:**
- Backend FastAPI con endpoints REST
- Frontend Next.js con TypeScript
- Gráficos GEX, CHARM y heatmap de volatilidad
- Actualización automática
- Integración VIX
- Documentación Swagger

**Próximas mejoras:**
- Selector de ticker dinámico en UI
- Exportar gráficos a PNG/PDF
- Histórico de datos
- Alertas configurables
- Modo ODTE
- Comparación multi-ticker
- Autenticación de usuario

---

## Contribuir

Fork del repositorio, crea una rama para tu feature, haz commit y abre un Pull Request.

---

## Licencia

MIT License

---

## Créditos

Proyecto original: [Kevin Trade 271](https://github.com/Kevintrade271/Quant-Terminal-Gamma-Exposure)

---

## Disclaimer

Software para fines educativos y de investigación. No constituye asesoramiento financiero.
