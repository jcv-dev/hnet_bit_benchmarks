# Respuesta a las apreciaciones del profesor (Juan Camilo)

Respuesta técnica a las dudas y preocupaciones planteadas sobre el documento de tesis, verificada contra el código fuente y los datos de las seis corridas. Referencias con formato `archivo:línea`.

---

## 1. Proceso de limpieza de los datos y el paso de txt a bytes

La aclaración es breve y directa: **el pipeline de esta tesis no realiza limpieza de texto**. Toda la limpieza ocurrió aguas arriba, en la publicación del dataset (`jhonparra18/spanish_billion_words_clean`, el sufijo `_clean` lo declara el autor del dataset). El código completo del "procesamiento" es (`data_spanish.py:70-74`):

```python
text = example.get("text", "")
if not text:
    continue
raw = text.encode("utf-8")
f.write(raw)
```

Es decir: se descartan las muestras vacías y el texto se codifica a UTF-8. No hay normalización Unicode, ni eliminación de HTML, ni lowercasing, ni deduplicación, ni filtros de longitud. La secuencia exacta txt → bytes es:

1. Descarga en streaming desde HuggingFace (~4.7 GB comprimidos en parquet; 46.9 millones de documentos).
2. Cada documento se codifica a UTF-8 y se concatena en un único `corpus.bin` plano (~8.7 GB), **sin separadores entre documentos** — las ventanas de entrenamiento cruzan fronteras de documento.
3. Modelos byte-level: ventanas fijas de 4096 bytes (vocabulario 0-255) sobre el archivo mapeado en memoria.
4. Transformer: el mismo corpus se tokeniza con BPE de GPT-2 en fragmentos de 10 MB (ventanas de 1280 tokens). Detalle menor: en las fronteras de fragmento, un carácter multi-byte partido se decodifica con `errors="replace"` (`data_spanish.py:46-48`), lo que corrompe ese carácter — sin efecto medible en un corpus de 8.7 GB.

Dos correcciones adicionales que ya se aplicaron al documento: (a) el capítulo 6 decía que el corpus "contiene 8.7 GB de texto comprimido, que se expanden a 25 000 millones de bytes" — lo correcto es ~4.7 GB comprimidos (parquet) → ~8.7 GB de texto plano UTF-8, y **25 GB es el presupuesto de entrenamiento**, que supera al corpus; por eso el DataLoader se recicla (`train_spanish.py:216-222`) y cada corrida completa **~3 pasadas** sobre la partición de entrenamiento del 95 % (el documento no lo mencionaba y ya se agregó). Verificado contra la API de HuggingFace: 46,925,295 filas, 4.70 GB de parquet, ~8.75 GB de texto UTF-8 medidos sobre los metadatos de los 18 archivos y dos archivos descargados íntegramente. (b) Se eliminó la mención de `byte2token_offsets.npy`, un archivo que se declara en el código pero nunca se escribe.

## 2. El 7.2 % y la demostración matemática del ×1.43

El documento decía "compresión global de ~14×"; el 7.2 % es la aritmética sobre ese dato (1/14 ≈ 7.1 %). El valor medido exacto, calculado en cada micro-lote como el producto de las fracciones de límites de las dos etapas (`train_spanish.py:242-245, 296-300`), es:

- Etapa 0 (350M): promedio **0.300** · Etapa 1: promedio **0.219** → producto global **0.066** (6.6 %).
- El producto crece de ~0.064 (paso 20k) a ~0.072 (pasos 160k-180k) — por eso el 7.2 % también coincide con el valor **final** del entrenamiento.
- 150M: 0.315 (31.5 %).

**La demostración que convierte esto en el argumento más fuerte de la tesis.** Cada bloque HGRN cuesta por token `8d² + 6dI` FLOPs (cuatro proyecciones d×d del núcleo recurrente más las proyecciones del MLP, con I = 256·⌈(2/3·d·4)/256⌉, configs en `model_factory.py:57-81`). La etapa s procesa Π mᵢ · L tokens. Con las compresiones medidas:

| | Etapa 0 | Etapa 1 | Etapa interna | Total |
|---|---|---|---|---|
| **150M** (c₀=0.315) | 260.9 G | — | 182.4 G | **443.3 GFLOPs/muestra** |
| **350M** (c₀=0.300, c₁=0.219) | 332.9 G | 198.4 G | 102.7 G | **634.0 GFLOPs/muestra** |

- **Cociente 634.0/443.3 = 1.43** → coincide con el cociente de tiempos medidos: 115.58 h / 80.57 h = **1.4345** (error 0.3 %).
- Equivalente denso (todas las etapas sobre los 4096 tokens): 840.7 → 2559.8 G = **×3.05**, proporcional a los parámetros (×3.03). O sea: sin chunking, el costo escalaría con el "cerebro", como en el Transformer.
- El chunking elimina el **47.3 %** de los FLOPs de bloques a 150M y el **75.2 %** a 350M. El desacople parámetros-tiempo que observaste (×3.03 parámetros = ×1.43 tiempo) no es una anomalía: es exactamente el efecto de que las etapas más anchas procesen cada vez menos tokens (31.5 % → 6.6 % en promedio de toda la corrida; el valor final del producto es 7.2 %, pero el tiempo de entrenamiento integra la compresión promedio de los 190,734 pasos).

Esto ya quedó como tabla (`tab:flops`) en el §Costo de entrenamiento del capítulo 7. Es reproducible: basta con las configs y los promedios de `training_steps_log.csv`.

**Matiz honesto para la comparación con MatMul-Free**: el híbrido 350M tiene ~4× menos FLOPs que el MatMul-Free 350M (634 G vs ~2.5 T) pero solo ~2× más throughput medido (60.7k vs 30.4k bytes/s). La diferencia se explica porque las GEMMs de la etapa interna operan sobre ~269 tokens y subutilizan la GPU, y porque el MatMul-Free usa kernels fusionados más eficientes. El ahorro de FLOPs es real y se materializa en tiempo, pero no al 100 %.

## 3. Memoria pico de entrenamiento (15.5 GB a 150M, 27.9 GB a 350M)

Tu hipótesis acierta en lo esencial, con una corrección técnica importante:

- **Correcto**: sin gradient checkpointing funcional (la bandera se activa pero el forward jerárquico nunca la lee — `hnet_bit/models/hnet_bit.py:815` es la única referencia), el autograd retiene **todas las activaciones intermedias de la jerarquía** — codificadores, decodificadores, proyecciones del enrutamiento (q, k), estados de los chunks y los broadcasts EMA del dechunk — hasta completar el backward. Eso domina: ~9-13 GB a 150M.
- **Corrección**: el núcleo recurrente interno **no almacena los estados ocultos de todos los chunks**. El kernel `fused_recurrent_hgrn` guarda solo (g, o, estado inicial) y **recomputa la trayectoria en el backward** mediante recurrencia inversa (`hnet_bit/ops/hgrn/recurrent_fuse.py:121-135, 168`). El cuello de botella no son los estados recurrentes, sino las activaciones de las capas que los rodean.

Componentes verificables del pico:
1. Pesos maestros FP32 + 2 momentos AdamW FP32 + gradientes: ≈ 2.2 GB (150M) y ≈ 6.7 GB (350M) — costo fijo independiente del chunking.
2. Activaciones retenidas sin recomputación: ≈ 9-13 GB (150M) y ≈ 21 GB (350M).
3. Fragmentación del asignador CUDA: 18.1 GB reservados vs 15.45 GB asignados (150M); la memoria por paso crece de ~11.6 GB (paso 10) a 15.45 GB (paso 160k) por fragmentación acumulada, no por el modelo.

**Mitigaciones posibles, en orden de impacto** (sin tocar la calidad — el checkpointing no cambia la función aprendida):
1. **Gradient checkpointing jerárquico**: envolver los stacks de codificador/decodificador y la llamada recursiva con `torch.utils.checkpoint`. La recursión hace natural el checkpoint por etapa. Estimación: pico 15.5 → ~8-10 GB a 150M (ya está listado en Trabajo Futuro).
2. **Activar `FusedBitLinear`**: el kernel ya existe y recomputa la salida en el backward (`hnet_bit/ops/fusedbitnet.py:379-409`); hoy el híbrido lo tiene desactivado (`use_fused_bitlinear=False`).
3. **Offload de los momentos AdamW a CPU**: −1.1 GB (150M) / −3.4 GB (350M).
4. **Descomposición de lote** (2×16 en vez de 4×8): las activaciones escalan con el lote; se mantiene el lote efectivo.
5. **Asignador `expandable_segments`**: elimina la fragmentación reservada (~17 %).

Sobre "un gradiente más agresivo": la palanca de memoria no es el gradiente (clipping o LR no tocan el pico), sino la **recomputación de activaciones** y la gestión del estado del optimizador — las opciones 1-5.

## 4. Los GB de inferencia (1.7 GB y 5.0 GB) — había un artefacto de medición

Tenías razón en desconfiar de esos números. Son correctos pero **no son memoria de inferencia**: la columna `Inference_Memory_MB` se mide dentro del proceso de entrenamiento, justo después de entrenar (`train_spanish.py:530-536` → `metrics_spanish.py:113-167`), cuando el GPU aún retiene los pesos FP32 y los momentos del optimizador. La aritmética lo demuestra:

- 150M: 1726 MB ≈ 553 MB (pesos) + 1106 MB (momentos) + ~67 MB de forward.
- 350M: 5010 MB ≈ 1678 + 3355 + ~0.

La medición honesta, en proceso limpio con el export de despliegue (`profile_inference.py`), es mucho menor:

| | Prefill 4096 | Decode BS8 |
|---|---|---|
| HNetBit 150M | **738 MB** | 653 MB |
| HNetBit 350M | **1863 MB** | 1769 MB |
| MatMul-Free 150M | 639 MB | 545 MB |
| MatMul-Free 350M | 1479 MB | 1354 MB |
| Transformer 150M | 1822 MB | 1038 MB |
| Transformer 350M | 3507 MB | 2723 MB |

Consecuencia: el documento subestimaba la ventaja del híbrido. La memoria de inferencia frente al Transformer es **1.9-2.9× menor** (no 1.5-2.3× como decía el texto), y las conclusiones que citan "menos de 700 MB" ya eran correctas. La tabla de síntesis y la sección de inferencia del capítulo 7 se corrigieron con los números limpios y una nota al pie que documenta el artefacto (las conclusiones y el capítulo 7 ahora coinciden).

## 5. Sobre "romper la ley de escalado de costo tradicional"

Tu lectura es correcta en los números, con un matiz de precisión: no se viola la ley de costo — el tiempo de entrenamiento está gobernado por los FLOPs, y lo que el chunking hace es **desacoplar los FLOPs de los parámetros**. La teoría clásica (FLOPs ∝ parámetros) aplica a arquitecturas densas; en HNetBit el ancho de las etapas internas crece, pero los tokens que las atraviesan caen de 31.5 % a 6.6 %, y el producto de ambos efectos da exactamente ×1.43 (sección 2). Para un proveedor de nube, la implicación es la que planteas: ×3 de "cerebro" con ×1.43 de cómputo, con la salvedad de que los costos ligados a parámetros (memoria del optimizador, ancho de banda de pesos) sí siguen escalando con los parámetros.

## 6. Cambios aplicados al documento

1. Cap. 6: corrección del tamaño del corpus (~4.7 GB comprimidos → ~8.7 GB de texto plano; 25 GB es el presupuesto), mención de las ~3 pasadas del DataLoader, eliminación de `byte2token_offsets.npy`.
2. Cap. 7: nueva tabla `tab:flops` con la demostración del ×1.43; precisión del 6.6 % medido (promedio) y del rango 0.064 → 0.072; memoria de inferencia con números limpios en la tabla de síntesis y en §3, con nota al pie del artefacto del optimizador.
3. Caps. 7, 8 y 9: magnitud de la ventaja de memoria de inferencia actualizada (1.9-2.9×).