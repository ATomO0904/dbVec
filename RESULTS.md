# Resultados de benchmarks: pgvector vs Qdrant

Informe consolidado de los resultados obtenidos hasta la fecha. Los datos crudos están en `results/*.csv` y las gráficas en `results/*.png`. Para cómo ejecutar cada prueba, consulta [`BENCHMARKS.md`](BENCHMARKS.md).

---

## Resumen ejecutivo

| Benchmark                  | Condición                                | Ganador                                     | Magnitud                                                       |
| -------------------------- | ---------------------------------------- | ------------------------------------------- | -------------------------------------------------------------- |
| Inserción                  | 1 M vectores, lote 300, 1 run            | Qdrant                                      | ~22,8× más rápido                                              |
| Inserción                  | 10 k vectores, lote 100, 3 runs          | Qdrant                                      | ~15,0× más rápido (6210 vs 413 vec/s)                          |
| Inserción (índice después) | 10 k vectores, lote 100, 3 runs          | Qdrant                                      | ~1,7× más rápido contando carga + indexado                     |
| Búsqueda                   | 10 k vectores, 1000 consultas, k=10      | Qdrant                                      | ~2,1× más QPS, mitad de latencia                               |
| HNSW (semántico)           | 1 k vectores AG News, 50 consultas, k=10 | Empate en recall; pgvector en latencia baja | recall 0.988–0.998; pgvector 383–500 QPS vs Qdrant 380–433 QPS |
| Memoria RAM                | 100 k vectores, lote 1000                | Empate                                      | Pico residente +533,5 MiB vs +547,7 MiB                        |

> El HNSW se repitió con embeddings reales de AG News: ahora el recall es ~0.99 en ambos motores, que es lo esperado en un índice sano. La corrida anterior con 1 M de vectores gaussianos (recall 0,01–0,12) se conserva en `hnsw_results_synthetic.csv` y se explica en la sección 10 de `BENCHMARKS.md`: el dataset era isotrópico y adversarial para HNSW, no un fallo de los motores.

---

## 1. Inserción masiva

`insert_results.csv` · 1 000 000 de vectores · lote 300 · 1 run

| Motor      | Tiempo total |   Vectores/s | Media por lote | Máx. por lote |
| ---------- | -----------: | -----------: | -------------: | ------------: |
| **Qdrant** |     315.01 s | **3 174.49** |       85.00 ms |     311.84 ms |
| pgvector   |   7 179.85 s |       139.28 |    2 150.60 ms |   3 245.43 ms |

Ambos almacenaron y verificaron 1 000 000 de vectores. Qdrant fue ~22,8× más rápido. Al ser una sola repetición no hay estimación de variabilidad.

### 1.1 Inserción a 10 000 vectores (3 repeticiones)

`insert_results_10000.csv` · lote 100 · 3 runs · media ± desviación · ejecutado con `insert_benchmark_temp.py` sobre tablas/colecciones temporales.

| Motor      |      Tiempo total |           Vectores/s |      Media por lote |       Máx. por lote |
| ---------- | ----------------: | -------------------: | ------------------: | ------------------: |
| **Qdrant** | **1,61 ± 0,02 s** | **6 210,40 ± 65,39** | **14,12 ± 0,26 ms** | **30,18 ± 2,99 ms** |
| pgvector   |    24,24 ± 1,22 s |       413,17 ± 20,27 |   241,56 ± 12,24 ms |   333,02 ± 53,54 ms |

Qdrant vuelve a ganar (~15,0×), con una dispersión muy baja. La diferencia frente al 22,8× de 1 M se explica sobre todo por el tamaño de lote (100 aquí vs 300 allí): el `commit` por lote y el mantenimiento del índice HNSW de pgvector pesan más proporcionalmente con lotes pequeños. No son condiciones idénticas, así que compara la tendencia, no las cifras absolutas.

Los 6 registros individuales están en `insert_results_10000.csv` y el progreso de la ejecución en `insert_10000_run.log`.

### 1.2 Inserción con el índice construido después

`insert_index_after_results.csv` · 10 000 vectores · lote 100 · 3 runs · media ± desviación

Para responder a si pgvector es más lento «por el índice», se repitió la carga **sin índice HNSW** y se midió después cuánto tarda crearlo en cada motor: `CREATE INDEX ... USING hnsw` con `maintenance_work_mem=512MB` en pgvector, y bajada de `indexing_threshold` + espera a `indexed_vectors_count == 10000` en Qdrant.

| Motor      | Carga sin índice (vec/s) | Construir índice (s) | Carga + índice (s) | Carga + índice (vec/s) |
| ---------- | -----------------------: | -------------------: | -----------------: | ---------------------: |
| pgvector   |         4 049,20 ± 15,05 |          2,32 ± 0,12 |        4,79 ± 0,13 |       2 088,59 ± 55,63 |
| **Qdrant** |    **7 632,17 ± 182,52** |      **1,45 ± 0,00** |    **2,76 ± 0,04** |   **3 618,52 ± 47,02** |

| Escenario                             | pgvector (vec/s) | Qdrant (vec/s) | Ventaja Qdrant |
| ------------------------------------- | ---------------: | -------------: | -------------: |
| Solo carga, con índice presente (1.1) |           413,17 |       6 210,40 |          15,0× |
| Solo carga, sin índice                |         4 049,20 |       7 632,17 |           1,9× |
| **Carga + construcción del índice**   |         2 088,59 |       3 618,52 |       **1,7×** |

**Conclusión:** mantener el índice HNSW durante la inserción hace que pgvector sea ~9,8× más lento en la fase de carga. Si se carga sin índice y se construye después, pgvector tarda 4,79 s en total frente a 24,24 s del caso anterior. La comparación ajustada —dejar el índice listo en ambos motores— da 2,76 s para Qdrant contra 4,79 s para pgvector: Qdrant sigue ganando, pero por ~1,7× y no por 15×. En los 15× anteriores el indexado de Qdrant se solapaba con la carga y no se contabilizaba, mientras que pgvector lo pagaba lote a lote.

Detalle en `insert_index_after_results.csv` y progreso en `insert_index_after_run.log`.

---

## 2. Búsqueda Top-k

`search_results.csv` · 10 000 vectores · 1000 consultas · k=10 · 5 runs (media ± desviación)

| Motor      |               QPS |        Media (ms) |          p50 (ms) |          p95 (ms) |      p99 (ms) |
| ---------- | ----------------: | ----------------: | ----------------: | ----------------: | ------------: |
| **Qdrant** | **419.27 ± 8.80** | **2.382 ± 0.051** | **2.324 ± 0.013** | **2.698 ± 0.215** | 3.545 ± 1.118 |
| pgvector   |     198.76 ± 6.55 |     5.033 ± 0.168 |     4.906 ± 0.128 |     5.611 ± 0.567 | 6.626 ± 0.997 |

Qdrant sirve ~2,1× más consultas por segundo con aproximadamente la mitad de latencia. Salvedad metodológica: Qdrant usa `exact=True`, mientras pgvector no desactiva su índice HNSW en esta prueba.

---

## 3. HNSW: recall vs latencia (dataset semántico)

`hnsw_results.csv` · 1 000 vectores de AG News embebidos con `all-MiniLM-L6-v2` (384 dims) · 50 consultas · k=10 · 3 runs (media) · `m=16`, `ef_construct=100`

| Motor        | `ef_search` | recall@10 |        QPS | Media (ms) | p95 (ms) |
| ------------ | ----------: | --------: | ---------: | ---------: | -------: |
| **pgvector** |          32 | **0.998** | **500.32** |   **1.99** | **2.70** |
| Qdrant       |          32 |     0.988 |     432.95 |       2.30 |     3.40 |
| **pgvector** |          64 | **0.998** | **464.87** |   **2.14** |     3.06 |
| Qdrant       |          64 | **0.998** |     379.93 |       2.72 |     3.41 |
| **pgvector** |         128 | **0.998** | **428.51** |   **2.33** | **3.18** |
| Qdrant       |         128 | **0.998** |     394.92 |       2.53 |     3.85 |
| pgvector     |         256 | **0.998** |     383.32 |       2.60 |     3.65 |
| Qdrant       |         256 | **0.998** | **409.81** |       2.43 | **3.28** |

**Lectura:**

- Ambos índices funcionan correctamente: `recall@10` de 0.988–0.998 (solo se pierde 1 vecino de 500; el caso Qdrant `ef=32` es 49/50).
- Con 1 000 vectores, `ef=32` ya es casi exacto: subir a `ef=256` no mejora el recall y reduce el QPS de pgvector de ~500 a ~383.
- pgvector lidera en throughput con `ef=32` y `ef=64` (~500/465 QPS vs ~433/380). Con `ef=256`, Qdrant se pone delante (410 vs 383 QPS) y logra la menor latencia p95 (3,28 ms).
- Se forzó el uso real del índice en ambos motores: `indexing_threshold=100` en Qdrant (si no, con 1 000 vectores no construye HNSW) y `enable_seqscan=off` + comprobación de `EXPLAIN` en pgvector.

### 3.1 Corrida sintética histórica (no comparable)

`hnsw_results_synthetic.csv` · 1 000 000 de vectores gaussianos isotrópicos · 50 consultas · k=10

| Motor    | `ef=32` | `ef=64` | `ef=128` | `ef=256` |
| -------- | ------: | ------: | -------: | -------: |
| Qdrant   |   0.014 |   0.030 |    0.054 |    0.116 |
| pgvector |   0.018 |   0.020 |    0.032 |    0.066 |

Estos valores motivaron el diagnóstico de la sección 10 de `BENCHMARKS.md`: el ground truth era correcto y ambos motores usaban sus índices, pero vectores gaussianos de 384 dimensiones sin estructura son un caso adversarial para HNSW. Se conservan solo como registro histórico.

---

## 4. Consumo de RAM

`memory_results.csv` · `memory_samples.csv` · 100 000 vectores · lote 1000

Medido sobre los cgroups de los contenedores, restando el estado previo a la carga (los contenedores ya tenían 1 M de vectores). Baseline: pgvector 215,71 MiB residentes / 3 416,05 MiB `memory.current`; Qdrant 109,45 MiB residentes / 1 757,27 MiB `memory.current`.

| Métrica (incremento sobre baseline)                    |        pgvector |          Qdrant |
| ------------------------------------------------------ | --------------: | --------------: |
| **RAM residente en el pico**                           | **+533,51 MiB** | **+547,74 MiB** |
| RAM residente al final (100 k cargados)                |      +31,42 MiB |      +29,87 MiB |
| Heap anónimo en el pico                                |      +62,41 MiB | **+526,02 MiB** |
| Memoria compartida (`shmem`) en el pico                | **+509,23 MiB** |           0 MiB |
| Page cache en el pico                                  |     +484,46 MiB |     +390,52 MiB |
| `memory.current` en el pico (lo que ve `docker stats`) |     +845,56 MiB |     +881,53 MiB |
| `memory.current` al final                              |     +515,21 MiB |     +237,10 MiB |

**Lectura:**

- Pico de RAM residente prácticamente empatado (~2,6 % de diferencia). Repetibilidad: 531–534 MiB (pgvector) y 548–562 MiB (Qdrant) en dos ejecuciones.
- Composición distinta: pgvector crece en memoria compartida (+509 MiB, construcción del índice HNSW); Qdrant en heap privado (+526 MiB) que luego libera.
- Al terminar, solo ~+30 MiB residentes por motor: los vectores están en archivos mapeados y cuentan como page cache. `docker stats` sobreestima la huella real al incluirla.
- pgvector deja más caché residual (+515 MiB vs +237 MiB), relevante en equipos con memoria ajustada.

---

## 5. Gráficas

![Resumen de benchmarks](results/resumen_benchmarks.png)

![Consumo de RAM](results/memory_comparison.png)

![Inserción 1 M](results/insert_comparison.png)

![Búsqueda Top-k](results/search_comparison.png)

![HNSW recall/latencia](results/hnsw_comparison.png)

El resumen de cuatro paneles se regenera con `python -m benchmarks.plot_summary` directamente desde los CSV; el panel de inserción usa la corrida de 1 M y el de HNSW los datos semánticos actuales. Las corridas de inserción de 10 k se reportan en las secciones 1.1 y 1.2. Las imágenes individuales se regeneran con `python -m benchmarks.analyzer`, `python -m benchmarks.plot_search` y `python -m benchmarks.plot_hnsw` (todas guardan PNG en `results/` y no repiten pruebas).

Gráficas históricas: `results/pvvectorVsQdrant10000.png` y `results/pgvectorVsQdrant100000.png` (inserción con 10 k y 100 k vectores, anteriores a los CSV actuales). Log completo de la ejecución semántica: `results/semantic_hnsw_run.log`.

---

## 6. Experiencia de desarrollo: sintaxis en Python

Los ejemplos salen de `db/pgvector_db.py`, `models/models.py`, `repository/pgvector_repository.py` y `repository/qdrant_repository.py` de este proyecto. No son implementaciones alternativas: son las llamadas reales que usan los benchmarks.

### 6.1 pgvector con SQLAlchemy

**Conexión**

```python
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

DATABASE_URL = "postgresql+asyncpg://benchmark:benchmark@localhost:5433/vectors"

engine = create_async_engine(DATABASE_URL, pool_size=10, max_overflow=0)
session_local = async_sessionmaker(engine, expire_on_commit=False)
```

**Modelo e índice HNSW**

```python
from pgvector.sqlalchemy import VECTOR
from sqlalchemy import Index, Integer, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

DIMENSIONS = 384

class Base(DeclarativeBase): ...

class Document(Base):
    __tablename__ = "documents"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    text: Mapped[str] = mapped_column(Text)
    embedding: Mapped[list[float]] = mapped_column(VECTOR(DIMENSIONS))

hnsw_index = Index(
    "idx_documents_embedding_hnsw",
    Document.embedding,
    postgresql_using="hnsw",
    postgresql_with={"m": 16, "ef_construction": 100},
    postgresql_ops={"embedding": "vector_cosine_ops"},
)
```

**Inserción por lotes**

```python
from sqlalchemy.dialects.postgresql import insert as pg_insert

stmt = pg_insert(Document).values(documents)   # lista de dicts
stmt = stmt.on_conflict_do_update(
    index_elements=[Document.id],
    set_={"text": stmt.excluded.text, "embedding": stmt.excluded.embedding},
)
await session.execute(stmt)
await session.commit()
```

**Búsqueda exacta**

```python
from sqlalchemy import select

distance = Document.embedding.cosine_distance(query_vector)
stmt = (
    select(Document.id, Document.text, (1 - distance).label("score"))
    .order_by(distance)
    .limit(10)
)
rows = (await session.execute(stmt)).all()
```

**Búsqueda aproximada con HNSW**

```python
from sqlalchemy import text

await session.execute(
    text("SELECT set_config('hnsw.ef_search', :ef, true)"), {"ef": "64"}
)
distance = Document.embedding.cosine_distance(query_vector)
stmt = select(Document.id).order_by(distance).limit(10)
ids = [row.id for row in (await session.execute(stmt)).all()]
await session.rollback()
```

**Crear el índice después de cargar**

```python
async with engine.begin() as conn:
    await conn.execute(text("SET LOCAL maintenance_work_mem = '512MB'"))
    await conn.run_sync(lambda c: hnsw_index.create(c, checkfirst=True))
```

### 6.2 Cliente de Qdrant

**Colección**

```python
from qdrant_client import AsyncQdrantClient, models

client = AsyncQdrantClient(url="http://localhost:6333")
await client.create_collection(
    collection_name="documents",
    vectors_config=models.VectorParams(
        size=384, distance=models.Distance.COSINE
    ),
    hnsw_config=models.HnswConfigDiff(m=16, ef_construct=100),
)
```

**Inserción por lotes**

```python
points = [
    models.PointStruct(
        id=row["id"], vector=row["embedding"], payload={"text": row["text"]}
    )
    for row in documents
]
await client.upsert(collection_name="documents", wait=True, points=points)
```

**Búsqueda exacta**

```python
res = await client.query_points(
    collection_name="documents",
    query=query_vector,
    limit=10,
    with_payload=True,
    search_params=models.SearchParams(exact=True),
)
hits = [(p.id, p.payload["text"], p.score) for p in res.points]
```

**Búsqueda aproximada con HNSW**

```python
res = await client.query_points(
    collection_name="documents",
    query=query_vector,
    limit=10,
    with_payload=False,
    search_params=models.SearchParams(hnsw_ef=64, exact=False),
)
ids = [p.id for p in res.points]
```

**Crear el índice después de cargar**

```python
await client.update_collection(
    collection_name="documents",
    optimizers_config=models.OptimizersConfigDiff(
        indexing_threshold=100, default_segment_number=1
    ),
)
info = await client.get_collection("documents")
print(info.indexed_vectors_count, info.optimizer_status)
```

### 6.3 Qué se observa en este proyecto

| Tarea                 | pgvector + SQLAlchemy                                                                    | Cliente de Qdrant                                                   |
| --------------------- | ---------------------------------------------------------------------------------------- | ------------------------------------------------------------------- |
| Definir el índice     | `Index(..., postgresql_using="hnsw", ...)` en el modelo                                  | `HnswConfigDiff` al crear la colección                              |
| Insertar              | `pg_insert(...).on_conflict_do_update(...)` + `commit()`                                 | `upsert(..., wait=True)` con `PointStruct`                          |
| Alternar exacto / ANN | `set_config('hnsw.ef_search', …)` (local a la transacción) y `order_by(cosine_distance)` | `SearchParams(exact=True)` o `SearchParams(hnsw_ef=…, exact=False)` |
| Verificar el índice   | `pg_indexes` o `EXPLAIN`                                                                 | `get_collection()` → `indexed_vectors_count`                        |
| Índice tras la carga  | `CREATE INDEX` con `maintenance_work_mem`                                                | `update_collection()` + esperar al optimizador                      |

**Lectura:** ambas APIs son asíncronas. El cliente de Qdrant es más directo para tareas vectoriales: la configuración del índice y el modo de búsqueda viajan en el propio `create_collection` / `query_points`. Con pgvector se escribe más código y se interactúa con conceptos SQL (`set_config`, `rollback`, `EXPLAIN`, DDL), pero se gana control sobre la transacción y el plan de consulta. La elección depende de qué equipo vaya a mantenerlo: aquí la comparación es solo de sintaxis y flujo de trabajo, no de rendimiento.
