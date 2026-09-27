# HNetBit: Explicación completa de todos los componentes

## El panorama general

HNetBit es un modelo de lenguaje que lee texto **byte por byte** y aprende solo a agrupar esos bytes en unidades más grandes (como palabras), sin que nadie le enseñe qué es una palabra. Lo hace de forma **jerárquica**: primero procesa bytes individuales, luego los agrupa en fragmentos, luego fragmentos más grandes, y así sucesivamente.

Tres innovaciones principales:
1. **Cuantización ternaria**: pesos en {-1, 0, +1} en vez de multiplicaciones de punto flotante
2. **Recurrencia HGRN**: en vez de atención cuadrática, un estado oculto lineal O(L)
3. **Chunking dinámico multi-etapa**: segmentación aprendida, sin tokenizador

---

## 1. INPUT: "Bytes raw 0–255"

**¿Qué es un byte?** La unidad más pequeña de información en una computadora. Un byte es un número entre 0 y 255. Cada letra, número o símbolo se convierte en uno o más bytes mediante codificación UTF-8.

**Ejemplo**: La palabra "Hola" sin acento en UTF-8:
```
H → 72
o → 111
l → 108
a → 97
```

**Ejemplo "aló" (con acento)**:
```
a  → 97                   (1 byte)
l  → 108                  (1 byte)
ó  → 195, 179             (2 bytes — el acento ocupa 2 bytes)
```

El ó con acento NO existe en ASCII (que solo va de 0 a 127). UTF-8 lo codifica como **dos bytes**: `195` y `179`. Cada uno está entre 0 y 255. Total: 4 bytes para "aló".

**¿Cómo representamos todo con 0-255 si UTF-8 puede tener millones de caracteres?** Porque el modelo no trabaja con caracteres. Trabaja con **bytes individuales**. Un carácter como 😊 (emoji) son 4 bytes: [240, 159, 152, 138]. El modelo no sabe que forman un emoji — solo ve 4 números. Aprende solo que esos 4 bytes suelen aparecer juntos.

El ByteTokenizer en el código (`hnet_bit/utils/tokenizers.py:13-90`):
- **Vocab**: 256 bytes fijos (0–255). Sin entrenamiento, sin estadísticas de corpus.
- **BOS** = 254: Beginning Of Sequence. Señal de inicio.
- **EOS** = 255: End Of Sequence. Señal de final.
- **pad** = 0: byte nulo para relleno.
- Todo en `np.uint8`.

**`encode(texts)`**: Convierte strings a UTF-8, opcionalmente antepone BOS (254) y apéndice EOS (255). Retorna arreglos `numpy.uint8`.

**`decode(tokens)`**: Inverso — `bytearray(tokens).decode("utf-8")`.

No hay subword, BPE ni nada aprendido. Cada carácter Unicode se parte en 1–4 bytes, y el modelo ve bytes crudos.

---

## 2. nn.Embedding (256 → d₀)

**El problema**: Tenemos bytes (números del 0 al 255), pero un solo número no captura suficiente información. El 72 (H) y el 111 (o) son solo etiquetas — no nos dicen nada sobre cómo se relacionan.

**La solución**: Convertir cada byte en un **vector denso** de 576 números (en el modelo 150M). Estos números se aprenden durante el entrenamiento.

Cada byte tiene una "ficha de Lego" con 576 conectores. Bytes que aparecen en contextos similares (como 'a' y 'e', que ambas son vocales) terminan con conectores parecidos.

```
Byte 72 (H) → [0.12, -0.45, 0.78, ..., 0.03]  (576 números)
Byte 111 (o) → [-0.21, 0.54, ..., -0.67]       (576 números)
```

La **tabla de embedding** es simplemente un diccionario de 256 entradas, cada una con un vector de 576 números. Se inicializa aleatoriamente y se ajusta durante el entrenamiento mediante backpropagation normal: el error final (CrossEntropyLoss) produce un gradiente que fluye hacia atrás hasta el embedding, y el optimizador AdamW ajusta los pesos.

---

## 3. Stage 0 — Outermost (La etapa exterior)

Aquí empieza la arquitectura jerárquica. La Stage 0 recibe la secuencia de 4096 bytes e identifica dónde terminan palabras para comprimir la secuencia.

### 3a. Encoder (HGRNBitStack, N_enc bloques)

Un stack de bloques. Cada bloque mezcla información de dos maneras:

**Paso 1 — Mezcla temporal (HGRNBitAttention)**:
Mantiene un "estado de memoria" que va pasando de byte a byte. Cuando lee el byte en posición t, el estado contiene información resumida de los bytes 0 hasta t-1.

```
Byte₁ → estado₁ → Byte₂ → estado₂ → Byte₃ → estado₃ → ...
```

Es como leer una frase palabra por palabra: cuando llegas a la cuarta palabra, ya tienes en mente el contexto de las tres primeras.

**Paso 2 — Mezcla de canales (HGRNBitMLP)**:
Dentro de cada byte individual, mezcla los N números para encontrar patrones entre ellos. Es como mirar una ficha de Lego y reorganizar sus conectores internos.

**ShortConvolution (opcional)**: antes de las proyecciones recurrentes, una convolución causal unidimensional profunda (kernel=4) captura patrones locales (bigramas, trigramas) a nivel de bytes. Dos modos:
- **Compartido**: una convolución sobre los estados ocultos antes de proyectar (más eficiente)
- **Separado**: convoluciones independientes sobre cada flujo proyectado i, f, g (más capacidad)

### 3b. Residual Proj — Proyección residual FP32

Guarda una copia exacta del estado actual en alta precisión (FP32). ¿Por qué FP32? Porque esta copia viajará a través de una operación discreta (el chunking, que decide cortar o no cortar). Para que la información no se pierda en ese proceso ruidoso, mantenemos alta precisión.

**¿Por qué inicializado en ceros?** Al principio del entrenamiento, el modelo no sabe nada. Inicializar en ceros significa "no añadas nada todavía". Gradualmente, los pesos se vuelven distintos de cero. Es como un cuaderno en blanco que se va llenando.

**Forma del estado**: Tensor (B, L, d₀):
```
Para batch=1, secuencia de 4 bytes:
Posición 0: [0.12, -0.45, 0.78, ..., 0.03]
Posición 1: [-0.21, 0.54, ..., -0.67]
Posición 2: [0.92, 0.11, ..., -0.42]
Posición 3: [-0.73, 0.88, ..., 0.05]
```

### 3c. RoutingModuleBit — El módulo de enrutamiento

Aquí ocurre la magia del chunking dinámico. Este módulo decide **dónde cortar** la secuencia en fragmentos.

**La idea**: Dos bytes dentro de la misma palabra (como 'H' y 'o' en "Hola") tienen **vectores parecidos** después del encoder. Dos bytes de palabras diferentes tienen **vectores diferentes**.

**¿Cómo mide la diferencia?** Con **similitud de coseno**. Imagina dos flechas que salen del origen. Si apuntan en la misma dirección, el coseno es 1. Si apuntan en direcciones opuestas, es -1. Si son perpendiculares, es 0.

```
coseno(H, o) ≈ 0.9  (muy parecidos → misma palabra)
coseno(a, espacio) ≈ 0.1 (diferentes → límite de palabra)
```

El "vector" aquí se refiere al vector de **estado procesado** después del encoder, NO al embedding crudo. Estos vectores han sido transformados por N_enc bloques HGRN y codifican información contextual.

**Fórmula completa**:
```
q_t = normalize(Linear_q(h_t))       → proyección del token actual
k_{t+1} = normalize(Linear_k(h_{t+1})) → proyección del siguiente
cos_sim_t = ⟨q_t, k_{t+1}⟩          → producto punto
p_t = clamp((1 - cos_sim_t) / 2, 0, 1) → probabilidad de límite
```

- coseno=0.9 → p=0.05 (casi no hay límite)
- coseno=0.1 → p=0.45 (probable límite)

La **primera posición** siempre es límite (p=1), para asegurar fragmentos no vacíos.

**Particularidad importante**: Este módulo usa precisión completa (`nn.Linear`), NO ternaria. ¿Por qué? Porque decidir dónde cortar es muy sensible al ruido. Si usáramos pesos ternarios {-1,0,+1}, las decisiones serían imprecisas. Las proyecciones se inicializan como matrices identidad para estabilizar el inicio del entrenamiento.

La decisión binaria se obtiene mediante **argmax**:
```
m_t = 1 si p_t > 0.5, sino 0
```

Esta operación no es diferenciable, pero el gradiente fluye a través de p_t mediante el **STE** en la conexión residual posterior.

**Durante generación**: se guarda un estado de inferencia (RoutingModuleState) con el último vector procesado. La probabilidad se calcula comparando el token actual contra el anterior:
```
p_step = clamp((1 - ⟨q_prev, k_curr⟩)/2, 0, 1)
```

### 3d. ChunkLayer — Cortar la secuencia

Toma las probabilidades de límite y las convierte en decisiones binarias, seleccionando solo los tokens en posiciones límite.

**Ejemplo con 6 bytes**:
```
Posición: 0     1     2     3     4     5
Byte:     BOS   H     o     l     a     ESP
¿Límite?: 1]    0     0     0     1]    0
```

De 6 posiciones → 2 límites → reducimos la secuencia.

**¿Cómo se implementa? Con el truco de argsort**. En PyTorch, seleccionar índices variables por batch no es eficiente. La solución:

1. **Crear claves**: para cada posición, clave = `posición + (NO_es_límite) × L`
```
Pos 0: 0 + (False=0) × 6 = 0   ← clave pequeña
Pos 1: 1 + (True=1) × 6 = 7    ← clave grande
Pos 2: 2 + 6 = 8               ← clave grande
Pos 3: 3 + 6 = 9               ← clave grande
Pos 4: 4 + 0 × 6 = 4           ← clave pequeña
Pos 5: 5 + 6 = 11              ← clave grande
```

2. **Ordenar** (argsort): las claves pequeñas quedan al inicio → índices [0, 4, 1, 2, 3, 5]

3. **Tomar los primeros M** = número de límites = 2 → posiciones 0 y 4

**Modo empaquetado** (packed mode): para secuencias de longitud variable concatenadas, la selección es directa mediante indexación booleana (sin argsort).

### 3e. HNetBit Stage=1 (Recursivo)

Los M tokens límite (~800 de 4096) pasan a la **siguiente etapa**.

**¿Qué significa recursivo?** Que Stage 0 contiene un HNetBit completo adentro (Stage 1). Si Stage 1 no es la más interna, contiene otro HNetBit (Stage 2), y así.

**Para 150M** (1 etapa de chunking):
- Stage 0: d₀=576, recibe 4096 bytes → comprime a ~800 chunks
- Stage 1 (innermost): d₁=768, procesa ~800 chunks con HGRNBitStack (10 bloques)

**Para 350M** (2 etapas de chunking):
- Stage 0: d₀=640, recibe 4096 → comprime a ~800
- Stage 1: d₁=896, recibe ~800 → comprime a ~160
- Stage 2 (innermost): d₂=1152, procesa ~160 chunks

Cada etapa ve la secuencia a un nivel de abstracción mayor: bytes → sílabas/palabras → frases. Y cada etapa tiene más dimensión que la anterior porque procesa conceptos de mayor nivel.

**¿Por qué la Stage 1 expande de d₀=4 a d₁=6 (en el ejemplo)?**

Porque los chunks representan conceptos de nivel superior. Es como la diferencia entre identificar una letra (poca info) vs identificar una palabra (más info). Cada etapa interna ve menos tokens pero con mayor dimensionalidad.

Cuando Stage 1 termina de procesar, produce vectores de dimensión d₁. Pero Stage 0 solo entiende d₀. La solución es recortar: `hidden_states[..., :D]` — tomar solo las primeras D dimensiones. Es como "traduce lo que aprendiste en Stage 1 al formato que entiende Stage 0".

### 3f. DeChunkLayer — Reconstruir la secuencia

Después de que Stage 1 procesa los ~800 chunks, necesitamos devolver la información a las 4096 posiciones originales. Se usa una **Media Móvil Exponencial (EMA)**.

**La intuición**: El chunk A (posiciones 0-3) produjo un vector V_A. El chunk B (posiciones 4-7) produjo V_B. Cada posición recibe un valor que es un promedio ponderado entre el chunk al que pertenece y la posición anterior:

```
out_t = p_t × chunk_t + (1 - p_t) × out_{t-1}
```

**Ejemplo concreto**:
```
out₀ = 1.0 × V_A + 0.0 × 0         = V_A      (p=1, primer límite)
out₁ = 0.0 × V_A + 1.0 × V_A       = V_A      (mismo chunk)
out₂ = 0.0 × V_A + 1.0 × V_A       = V_A
out₃ = 0.0 × V_A + 1.0 × V_A       = V_A
out₄ = 1.0 × V_B + 0.0 × V_A       = V_B      (nuevo límite)
out₅ = 0.0 × V_B + 1.0 × V_B       = V_B
```

En un modelo entrenado, p_t rara vez es exactamente 0 o 1. Si p_t = 0.85 en el límite:
```
out₄ = 0.85 × V_B + 0.15 × V_A     ← ¡mezcla suave!
```

**¿Por qué EMA en vez de copia directa?** Esta es una pregunta excelente. Si ya sabemos qué posiciones pertenecen a qué chunk, ¿por qué no simplemente copiar V_A a todas esas posiciones?

Tres razones:
1. **Gradiente continuo**: p_t es diferenciable → el enrutamiento aprende a ajustar los límites. Con copia directa, el gradiente no puede atravesar el corte abrupto para llegar al módulo de enrutamiento.
2. **Incertidumbre**: si p_t = 0.6, el límite es dudoso → el EMA mezcla suavemente en lugar de forzar un corte.
3. **Estabilidad**: la transición suave evita valores extremos en los bordes.

**Implementación**: La fórmula EMA es matemáticamente idéntica a la recurrencia HGRN:
```
h_t = g_t · h_{t-1} + x_t
```
con g = (1-p) y x = p · hidden. Por lo tanto, se implementa usando el mismo kernel fused_recurrent_hgrn que la recurrencia principal, operando sobre los M tokens límite.

**Broadcast a todas las posiciones**: Se construye un índice de retroceso (plug-back):
```
plug_back_idx[t] = cumsum(m, t) - 1
```
que mapea cada posición t al chunk que la contiene, permitiendo difusión paralela mediante `torch.gather`.

### 3g. Residual: out · STE(p) + R

Combina dos caminos:
1. **La salida del dechunk** (out): pasó por chunking → Stage 1 → dechunking
2. **La copia residual** (R): se guardó ANTES del chunking, sin compresión

**¿Por qué combinar ambas?** El proceso de chunking puede perder información (descartamos ~80% de los tokens). La copia residual provee un camino directo para que la información fluya sin compresión.

**STE(p)** (Straight-Through Estimator):
- **Forward**: tratamos p como 1 (la conexión del dechunk está activa)
- **Backward**: el gradiente fluye a través de p real (como número continuo)

Sin STE, el gradiente sería 0 para p (multiplicación por constante 1). Con STE, el enrutamiento aprende a ajustar p.

### 3h. Decoder (HGRNBitStack, N_dec bloques)

Otra pila de bloques HGRNBitBlock que **refina** la secuencia reconstruida. El proceso de chunk+dechunk introduce pequeñas imperfecciones (como comprimir y descomprimir una imagen). El decoder:
1. Quita el ruido del chunk/dechunk (información ligeramente distorsionada)
2. Quita el ruido de la cuantización ternaria (pesos {-1,0,+1} pierden precisión)

Hace lo mismo que el encoder: mezcla temporal (HGRNBitAttention) y mezcla de canales (HGRNBitMLP). La diferencia es que recibe la secuencia reconstruida después del dechunking + residual.

---

## 4. Stage 1 — Innermost (La etapa más interna)

Cuando llegamos a la última etapa, ya no hay más compresión. Simplemente procesamos los chunks con una pila de bloques HGRNBitBlock.

**Para 150M**: HGRNBitStack con 10 bloques, procesando ~800 chunks de dimensión 768.

**Variante Hybrid-Attn**: En algunos experimentos, los bloques HGRN se intercalan con bloques de **atención con ventana deslizante** siguiendo el patrón "xaxa":
```
[HGRN, Atención, HGRN, Atención, HGRN, Atención, HGRN, Atención, HGRN, HGRN]
```

La atención permite que cada token mire **directamente** a otros tokens relevantes (no solo a través de la memoria recurrente). Como la secuencia ya está comprimida a ~800 tokens, el costo cuadrático es manejable.

**Sliding-window attention**: Cada token atiende solo a los w=64 tokens anteriores, limitando la complejidad a O(L × w).

**RoPE (Rotary Position Embeddings)**: codifica posición relativa mediante rotaciones en el espacio de representación.

**Proyecciones FP16**: A diferencia del modelo principal, Q/K/V/O usan precisión completa porque los puntajes de atención requieren mayor precisión numérica.

---

## 5. BitLinear LM Head (d₀ → 256)

Después de toda la jerarquía, tenemos vectores de 576 números para cada posición. Necesitamos convertirlos de vuelta a predicciones de bytes.

La **LM Head** es una capa que proyecta:
```
576 números → 256 puntajes (logits)
```

Cada puntaje corresponde a un byte (0 a 255). El más alto indica el byte que el modelo cree más probable.

**Softmax**: convierte los puntajes en probabilidades:
```
probabilidad(byte=72) = exp(logit[72]) / Σ_v exp(logit[v])
```

Todas las posiciones se predicen simultáneamente. Para predecir la posición t, el modelo solo usa información de las posiciones 0..t-1 (es **causal**: no hace trampa mirando el futuro).

No se comparten los pesos con el embedding (tie_word_embeddings=False) porque la cuantización ternaria de la cabeza y la precisión completa del embedding tienen propiedades numéricas diferentes.

---

## 6. CrossEntropyLoss (La función de pérdida)

Mide qué tan equivocado estuvo el modelo:
```
Loss_t = -log(probabilidad del byte correcto en posición t)
Loss_total = promedio sobre todas las posiciones
```

**El modelo se entrena para minimizar esto**. Cada paso:
1. Toma una secuencia de 4096 bytes
2. Predice el siguiente byte para cada posición (4096 predicciones)
3. Compara cada predicción con el byte real que venía después
4. Calcula la pérdida promedio
5. Ajusta los parámetros un poquito en la dirección correcta (retropropagación)

**Shift-1**: La entrada son los bytes 0..L-1, la salida objetivo son los bytes 1..L:
```
Input:  [byte₀, byte₁, byte₂, ..., byte₄₀₉₄, byte₄₀₉₅]
Target: [byte₁, byte₂, byte₃, ..., byte₄₀₉₅, (ignorado)]
```
La posición 4095 no tiene "siguiente byte", así que se ignora (ignore_index=-100).

**Loss total**: L = L_CE + 0.01 × L_LB

---

## 7. Load-Balancing Loss — Pérdida de balanceo

El módulo de enrutamiento podría decidir que **todos** los bytes son límites (un fragmento por byte) o que **ninguno** lo es (un solo fragmento gigante). Ambas son inútiles.

La pérdida de balanceo penaliza las soluciones extremas:
```
L_LB(s) = (N/(N-1)) × [(1-ratio) × (1-prob) + ratio × prob × (N-1)]
```
donde:
- N = 5.0 (longitud promedio de palabra en español)
- ratio = fracción de límites predichos
- prob = probabilidad promedio de límite

**Mínimo en 1/N**: se alcanza cuando 1 de cada 5 bytes es un límite.

La pérdida total es:
```
Loss_total = CrossEntropyLoss + lambda_LB × Σ_s L_LB(s)
```
con lambda_LB = 0.01. El factor pequeño asegura que la prioridad sea predecir bien el texto, no solo balancear.

---

## 8. HGRN Recurrencia — El corazón del modelo

**La autoatención** del Transformer tiene complejidad O(L² · d). La **recurrencia HGRN** ofrece:
- O(L) en entrenamiento
- O(1) por paso en inferencia
- Estado oculto de tamaño fijo O(d) (no crece con la secuencia)

**Formulación**:
```
f_t = sigmoid(BitLinear_f(x_t))         → forget gate (qué olvidar)
i_t = SwiGLU(BitLinear_i(x_t), 1-f_t)  → input gate (qué recordar)
h_t = f_t · h_{t-1} + i_t              → state update
o_t = FusedRMSNormSwishGate(g_proj, h_t) → output
```

La compuerta de entrada usa **compuerta complementaria**: cuando f_t es alto (mucha retención), 1-f_t es bajo (poca entrada nueva). Es como decir "si no voy a olvidar el pasado, no me hace falta mucha información nueva".

### SwiGLU (activación):
```
SwiGLU(x, g) = Swish(x) · g = (x / (1+e^(-x))) · g
```

### Multi-cabeza:
El estado oculto se redimensiona de (B, L, d) a (B, H, L, D_head). Cada cabeza procesa su recurrencia independientemente.

### ShortConvolution:
Opcionalmente, antes de las proyecciones recurrentes se aplica una convolución causal depthwise (kernel=4) que captura patrones locales. Particularmente útil en el modelo híbrido para compensar la ausencia de atención local.

### Dos kernels de recurrencia:

| Propiedad | fused_recurrent | chunk_hgrn |
|---|---|---|
| Fórmula | h_t = g_t · h_{t-1} + x_t | h_t = exp(g_t) · h_{t-1} + x_t |
| Representación de g | Probabilidad (0,1) | Log-space ℝ |
| Complejidad serial | O(L) | O(L/C) con C=128 |
| Uso principal | Inferencia (paso a paso) | Entrenamiento (prefill) |

---

## 9. Estados h y la Cache de Inferencia

**Estado h** (hidden state) de HGRN:
```
h_t = f_t · h_{t-1} + i_t
```
- `h_t`: nuevo estado (para el byte siguiente)
- `h_{t-1}`: estado anterior (lo que recordaba)
- `f_t`: cuánto del pasado conservar
- `i_t`: información nueva del byte actual

Es como leer: cuando lees la quinta palabra de una oración, tienes en mente un resumen de las cuatro primeras. Ese resumen es h.

### HNetBitCache

Durante generación, el modelo guarda una estructura de caché recursiva:
```
HNetBitCache:
├── encoder_cache: [estados h de cada bloque del encoder, uno por capa]
├── routing_state: [último byte visto + si hay tokens procesados]
├── main_network_cache: HNetBitCache de la siguiente etapa (recursivo)
├── dechunk_state: [último valor EMA]
└── decoder_cache: [estados h de cada bloque del decoder]
```

**Diferencia con Transformer**:
- **Transformer**: cache KV crece con cada token (O(L · d) por capa)
- **HNetBit**: estado de tamaño fijo O(d) por capa, no crece

### Salto condicional de Stage 1 durante generación

Cuando el modelo genera byte por byte y el byte actual **no es límite**, la Stage 1 se salta por completo — solo se actualiza el EMA:

```
Generando: "El ___"
Byte 'E' → p=0.02 (no límite) → solo EMA, Stage 1 NO se ejecuta
Byte 'l' → p=0.01 (no límite) → solo EMA, Stage 1 NO se ejecuta
Byte ' ' → p=0.95 (límite) → procesar Stage 1 COMPLETO
```

~80% de los bytes no son límites → ~80% de los pasos de generación son mucho más rápidos → ~5× de aceleración total.

### Soporte para beam search
El método `reorder_cache(beam_idx)` recorre recursivamente la jerarquía y aplica `torch.index_select(beam_idx)` a todos los tensores de estado dependientes del lote.

---

## 10. Diferencia entre entrenamiento e inferencia

| Aspecto | Entrenamiento | Inferencia |
|---|---|---|
| Secuencia | Completa (4096 bytes de una vez, forward completo) | Byte por byte (step mode) |
| Chunking | Se seleccionan todos los límites simultáneamente | Se decide por cada byte (cada vez) |
| Stage 1 | Siempre procesa chunks completos | Se salta si el byte no es límite |
| Caché | No se usa (no hay pasado que guardar) | HNetBitCache acumula estados |
| Gradientes | Sí (backpropagation, autograd graph) | No (torch.no_grad()) |
| Pesos | {-1,0,+1} con STE durante forward, FP32 para gradientes | {-1,0,+1} fijos (freeze_ternary_weights) |
| Tiempo | ~15.5 GB VRAM (150M hybrid) | ~652 MB VRAM |
| Pipeline completo | Input → Embed → E → Route → Chunk → Stage1 → Dechunk → Residual → Decoder → LM Head → Loss | Input token → Embed → E step → Route step → (Chunk step + Stage1 step si es límite) → Dechunk step → Residual → Decoder step → LM Head → next token |

---

## 11. ¿Por qué 576? (dimensión del modelo)

576 no es mágico. Sale de decisiones de ingeniería:
1. Queremos ~138M parámetros totales para el modelo 150M
2. El número de parámetros depende de: d_model² × num_blocks × expansión
3. 576 es 24² y múltiplo de 64 (optimizado para GPUs)
4. Se prueba un tamaño y se verifica si cabe en GPU y si aprende bien

Para 350M: d_model = [640, 896, 1152] (más grande = más capacidad = más parámetros)
Para 750M: d_model = [896, 1152, 1536]

---

## 12. ¿Por qué 4096 bytes de secuencia?

Porque el presupuesto total es **25 mil millones de bytes**. Cada paso procesa 32 secuencias × 4096 bytes = 131,072 bytes/paso → 190,735 pasos.

4096 es potencia de 2 (2¹²), simplifica la implementación en GPU.

El transformer BPE usa 1280 tokens × ~4.5 bytes/token = ~5760 bytes/muestra. La diferencia (5760 vs 4096) da al transformer una ventaja potencial en contexto largo (ve más bytes por muestra). Se documenta en la tesis como limitación conocida.

---

## 13. Compresión de pesos ternarios

Cada peso {-1, 0, +1} necesita solo 2 bits:
```
-1 → 00
 0 → 01
+1 → 10
```
Se empaquetan 4 pesos por byte uint8. Por tensor se guarda una escala float32 (4 bytes).

| Modelo | Parámetros ternarios | Tamaño FP16 | Tamaño empaquetado | Compresión |
|--------|--------------------|-------------|-------------------|------------|
| Hybrid 150M | 138M | 264 MB | 38 MB | 7.0× |
| Hybrid 350M | 419M | 800 MB | 115 MB | 7.0× |
| Hybrid 750M | 1026M | 1958 MB | 283 MB | 6.9× |

---

## 14. Flujo completo de entrenamiento (resumen)

```
Entrada: 8 bytes [H, o, l, a, espacio, m, u, n]
         ↓
Embedding: (8, d₀) — cada byte → vector denso
         ↓
Encoder (N_enc bloques HGRN): (8, d₀)
         ↓
Routing: calcula p_t para cada posición → identifica límites
         ↓
Chunking: selecciona M tokens límite → (M, d₀) con M ≪ L
         ↓
Stage 1 (innermost, N_inner bloques): expande → (M, d₁)
         ↓
Dechunking: reconstruye L posiciones desde M chunks → (L, d₁)
         ↓
Recorte a d₀: (L, d₀)
         ↓
Residual: (L, d₀) + (L, d₀) = (L, d₀)
         ↓
Decoder (N_dec bloques): refina → (L, d₀)
         ↓
LM Head: proyecta a 256 bytes → (L, 256)
         ↓
CrossEntropyLoss: compara con los siguientes bytes reales → escalar
         ↓
Total = CE + 0.01 × LB → retropropagación → ajuste de pesos
```

---

## 15. Flujo de inferencia (generación paso a paso)

```
Estado inicial: h_cache = [ceros], ema_last = [ceros]

Para cada byte a generar:

  1. Embedding del byte actual → (1, d₀)
  
  2. Encoder step → (1, d₀)
  
  3. Routing step:
     - Compara coseno con el byte anterior (guardado en routing_state)
     - Si p > 0.5 (es límite):
         → ChunkLayer step: seleccionar el token
         → Stage 1 step: procesar con la jerarquía interna
         → Actualizar caché de Stage 1 (slice/merge)
     - Si p ≤ 0.5 (NO es límite):
         → SALTAR Stage 1 por completo
         → Solo mantener el valor anterior del EMA
  
  4. Dechunking step: out = p · chunk + (1-p) · ema_last
  
  5. Residual: out + encoder_output
  
  6. Decoder step → refina
  
  7. LM Head → logits (256 puntajes)
  
  8. argmax → selecciona el byte más probable
  
  9. Repite desde paso 1 con el nuevo byte
```

---

## 16. Variables del panel derecho del diagrama

El panel derecho del diagrama desglosa el mecanismo de **Dynamic Chunking** en 4 sub-paneles (a) a (d), de abajo hacia arriba. Cada uno usa su propio conjunto de variables.

### (a) Routing Module — Variables
```
x̂  →  estado oculto en cada posición DESPUÉS del encoder
        (no el embedding crudo, sino el vector procesado)
        Cada cuadrado = un token. Los colores (azul, rosa, verde, naranja)
        indican a qué chunk pertenecería ese token si se aplicara el corte.
        Los cuadrados blancos con borde punteado son posiciones que NO
        están en un límite de chunk (están dentro de un fragmento).

p   →  probabilidad de límite en cada posición (entre 0 y 1)
        Representado como un gráfico de pastel:
          - Círculo completamente oscuro = p=1 (límite seguro)
          - Círculo vacío = p=0 (no hay límite)
          - Medio lleno = p≈0.5 (límite dudoso)
        Se calcula como: pₜ = clamp((1 − cos⟨qₜ, kₜ₊₁⟩)/2, 0, 1)
        donde qₜ = normalize(Linear_q(x̂ₜ)), kₜ₊₁ = normalize(Linear_k(x̂ₜ₊₁))
        
        IMPORTANTE: p decide si HAY un límite ENTRE la posición t y t+1.
        pₜ alta → el token t+1 es el inicio de un nuevo chunk.
        p₀ siempre es 1 (el primer token siempre es límite).

Flechas verticales (x̂ → p):
        Cada cuadrado x̂ tiene su propio círculo p encima. La flecha
        representa que p depende del coseno entre x̂ en posiciones t y t+1.
```

### (b) Downsampler — Variables
```
x̂^s  →  los mismos estados x̂ de la etapa s, pero ahora con las
         decisiones de límite ya tomadas. Los cuadrados de color =
         tokens seleccionados como límites. Los blancos punteados =
         tokens descartados (no pasan a la siguiente etapa).

b^s  →  decisión binaria de límite para cada posición en la etapa s:
           ● negro = 1 (es límite → se selecciona)
           ○ blanco = 0 (no es límite → se descarta)
         Se obtiene como: b^sₜ = argmax([1-pₜ, pₜ])
         
x^{s+1} →  la representación COMPRIMIDA después del downsampling.
           Solo los M tokens seleccionados (los que tenían b^s=1).
           M ≪ L (muchos menos tokens que la secuencia original).
           Estos M tokens pasan al core model M para procesamiento.

Flechas convergentes (desde x̂^s → x^{s+1}):
        Solo los cuadrados de color tienen flecha hacia arriba.
        Los blancos punteados no tienen flecha — su información se pierde.
```

### (c) Smoothing Module — Variables
```
ẑ   →  los chunks PROCESADOS por el core model M, ANTES del suavizado.
        4 cuadrados (azul, rosa, verde, naranja) = 4 chunks procesados.
        Son los vectores de salida de Stage 1, con dimensión d₁.

z̄   →  los chunks SUAVIZADOS por el EMA, DESPUÉS del suavizado.
        Mismos 4 colores, pero ahora cada chunk contiene información
        mezclada con su vecino anterior debido al EMA.
        Fórmula: z̄ₜ = Pₜ · ẑₜ + (1−Pₜ) · z̄ₜ₋₁

        Compara ẑ con z̄: ẑ es el chunk "puro", z̄ es el chunk mezclado.
        En el diagrama z̄ tiene los mismos colores que ẑ, pero el
        degradado en el borde entre chunks representa la mezcla.

P_t, P_{t-1}, P_{t-2}, P_{t-3} →
        probabilidad de límite en la POSICIÓN del chunk dentro de la
        secuencia original. Se usan como pesos en el EMA:
        
        P_t = probabilidad de que el chunk t sea un límite nuevo
        (1−P_t) = contribución del chunk ANTERIOR al actual
        
        Flecha VERTICAL (ẑ → z̄): multiplica ẑ por P_t
        Flecha DIAGONAL (z̄ → ẑ del siguiente): multiplica z̄ por 1−P_t
        
        Ejemplo: si P_t = 0.85, entonces:
          z̄ₜ = 0.85·ẑₜ + 0.15·z̄ₜ₋₁
        El chunk t es 85% nuevo y 15% del chunk anterior (transición suave).

Flechas verticales ẑ → z̄:
        Cada chunk ẑ sube directamente a su posición correspondiente en z̄,
        multiplicado por su P_t.

Flechas diagonales z̄ → ẑ del siguiente:
        El z̄ del chunk anterior se mezcla con el ẑ del chunk actual,
        multiplicado por (1−P_t). Esto crea el degradado entre chunks.
```

### (d) Upsampler — Variables
```
z̄   →  los mismos 4 chunks suavizados de (c). Son la entrada al upsampler.
        Solo hay 4 porque hay 4 chunks.

c   →  coeficientes de EXPANSIÓN. Son las probabilidades de límite
        originales (pₜ) aplicadas a CADA posición de la secuencia larga.
        Arriba de cada posición de salida hay un círculo con gráfico de
        pastel que indica qué tanta influencia tiene ese chunk en esa
        posición. Círculo lleno = el chunk contribuye al 100%.
        Círculo vacío = no contribuye.

×   →  fila de multiplicación: cada posición de salida se multiplica
        por su coeficiente c antes de la expansión.

Filas de cuadrados (de 4 a 8):
        La expansión TOMA los 4 chunks (z̄) y los DISTRIBUYE a 8 posiciones
        (la longitud original L). Cada chunk se REPITE en las posiciones
        que le corresponden:
          [A] [A] [B] [B] [B] [C] [C] [D]
        
        Esto se implementa con el índice plug_back_idx:
        plug_back_idx[t] = cumsum(boundary_mask, t) − 1
        
        Los cuadrados blancos punteados representan posiciones donde
        el chunk se mezcla con el vecino (efecto del EMA).
        
        La fila de cuadrados con borde sólido y relleno sólido son las
        posiciones DONDE HAY LÍMITE (pₜ alta → el chunk es el dominante).
        Las de borde punteado son las posiciones DENTRO del chunk
        (sin límite, el valor viene del EMA del chunk anterior).

Flechas divergentes (z̄ → fila de 8):
        Cada uno de los 4 cuadrados z̄ envía flechas a todas las posiciones
        de salida que le corresponden. El azul va a 2 posiciones,
        el rosa a 3, el verde a 2, el naranja a 1.
        La cantidad de posiciones por chunk depende de dónde estén los
        límites originales.
```

### Resumen de todas las variables

| Variable | Significado | ¿Dónde aparece? |
|---|---|---|
| `x̂` (x hat) | Vector de estado después del encoder | Routing (a) |
| `p` | Probabilidad de límite (0 a 1) | Routing (a), Smoothing (c) |
| `x̂^s` | Estados en etapa s, con decisión de límite | Downsampler (b) |
| `b^s` | Decisión binaria (0/1): ¿es límite? | Downsampler (b) |
| `x^{s+1}` | Chunks comprimidos que entran al core M | Downsampler (b) |
| `ẑ` | Chunks procesados por M, antes del EMA | Smoothing (c) |
| `z̄` | Chunks después del EMA (suavizados) | Smoothing (c), Upsampler (d) |
| `P_t` | Peso del EMA para el chunk t | Smoothing (c) |
| `c` | Coeficientes de expansión por posición | Upsampler (d) |

### Glosario de notación

| Símbolo | Significado |
|---|---|
| `^` (hat) | Variable procesada por el paso anterior (ej: x̂ es x después del encoder) |
| `¯` (bar) | Variable suavizada/promediada (ej: z̄ es ẑ suavizado por EMA) |
| `s` superíndice | Número de etapa (s=0 es la más externa, s=1 la siguiente, etc.) |
| `t` subíndice | Posición en la secuencia |
| `M` | Número de chunks después del downsampling (≪ L) |
| `L` | Longitud original de la secuencia |

---

## 17. Correspondencia entre el panel izquierdo y derecho

```
Pipeline izquierdo:           Panel derecho:
────────────────────          ────────────────
Decoder D                     
Dechunking (trapecio rosa) →  (d) Upsampler + (c) Smoothing
Core M (Stage 1)           →  recibe x^{s+1}, produce ẑ
Chunking (trapecio amarillo)  (b) Downsampler + (a) Routing
Encoder E                     produce x̂
```

Los chunks coloreados (azul, rosa, verde, naranja) son los **mismos** en ambos paneles — el panel derecho es un "zoom" que muestra en detalle lo que los trapecios de chunking/dechunking hacen por dentro.

---

**Nota importante**: El script `ejemplo_minimo_output.txt` en la raíz del proyecto contiene la ejecución completa de un ejemplo mínimo con d₀=4, d₁=6, mostrando cada variable en cada paso. Léelo junto a esta explicación para ver los números reales en acción.

---

## 18. Jerarquía de clases — qué contiene cada una

### HNetBitForCausalLM
Modelo de lenguaje causal completo. Pipeline: bytes → embedding → HNetBit recursivo → logits → pérdida (entrenamiento) o predicción (inferencia).
Contiene: ByteTokenizer, nn.Embedding(256, d₀), HNetBit, BitLinear_LM.

### HNetBit
Modelo jerárquico recursivo. Cada etapa no-interna contiene otra instancia de HNetBit como red interna.
Flujo: encoder → routing → chunking → HNetBit(stage+1) → dechunking → residual → decoder.
La etapa más interna es solo una HGRNBitStack plana.

### HGRNBitStack
Pila secuencial de N bloques HGRNBitBlock + RMSNorm final.
Componente fundamental repetido en codificadores, decodificadores y etapa interna.

### HGRNBitBlock
Bloque atómico: atención recurrente + MLP feed-forward en estructura pre-norm residual.
x = x + HGRNBitAttention(RMSNorm(x)), luego x = x + HGRNBitMLP(RMSNorm(x)).

### HGRNBitAttention
Subcapa de mezcla de secuencia (reemplaza la autoatención del Transformer).
Contiene: Conv1D opcional, 4 BitLinear (i,f,g,o), SwiGLU, recurrencia HGRN, FusedRMSNormSwishGate.
Mantiene un estado recurrente h que se actualiza por paso.

### HGRNBitMLP
Subcapa de mezcla de canales (feed-forward). Contiene: BitLinear_gate (expansión), split en 2 mitades, SwiGLU, BitLinear_down (contracción).

### BitLinear
Capa lineal con cuantización ternaria de pesos (absmean) y activaciones (8-bit simétrica por token).
Usa STE (.detach()) para backprop en entrenamiento. Contiene RMSNorm internamente.

### FusedBitLinear (hereda de BitLinear)
Variante fusionada: RMSNorm + activation_quant + multiplicación ternaria en un solo kernel Triton.
Sin Triton disponible, hace fallback a BitLinear normal.

### RoutingModuleBit
Predice límites de fragmentos (chunks) vía similitud coseno entre proyecciones de tokens consecutivos.
Opera en FP32 (nn.Linear), no ternario. Dos modos: lote (entrenamiento) y paso a paso (generación).

### ChunkLayer
Sin parámetros. Extrae tokens marcados como límites usando un truco de argsort.
Modos: padded (entrenamiento, por defecto) y packed (secuencias concatenadas de largo variable).

### DeChunkLayer
Sin parámetros. Reconstruye secuencia original desde fragmentos procesados usando EMA.
El EMA se implementa con el kernel fused_recurrent_hgrn + plug_back_idx (cumsum).

### CausalMHABit (solo variante hybrid_attn, etapa interna)
Atención multi-cabeza con ventana deslizante (w=64) + RoPE.
Proyecciones Q,K,V,O en FP32 (nn.Linear), no ternarias.

### HNetBitCache (solo inferencia)
Caché recursiva que espeja la jerarquía del modelo. Contiene:
- encoder_cache: HGRNBlockCache por capa (conv_state, h_state)
- routing_state: RoutingModuleState (has_seen_tokens, last_hidden_state)
- main_network_cache: HNetBitCache recursivo de la etapa siguiente
- dechunk_state: DeChunkState (last_value del EMA)
- decoder_cache: HGRNBlockCache por capa

---

## 19. Tabla completa de componentes por régimen (entrenamiento vs inferencia)

### Tokenización y embedding

| Componente | Pertenece a | Qué hace | Régimen |
|-----------|-------------|----------|---------|
| ByteTokenizer | HNetBitForCausalLM | Añade BOS(254) al inicio y EOS(255) al final. Vocabulario=256. | Ambos |
| nn.Embedding(256, d₀) | HNetBitForCausalLM | Mapea cada byte ID a vector denso de dimensión d₀. FP32 (no ternario). | Ambos |

### Proyecciones lineales ternarias (todas usan BitLinear)

| Componente | Pertenece a | Qué hace | Régimen |
|-----------|-------------|----------|---------|
| BitLinear_i | HGRNBitAttention | Proyección ternaria para la compuerta de entrada | Ambos |
| BitLinear_f | HGRNBitAttention | Proyección ternaria para la compuerta de olvido → sigmoide → f_t ∈ (0,1)ᵈ | Ambos |
| BitLinear_g | HGRNBitAttention | Proyección ternaria para la compuerta de salida | Ambos |
| BitLinear_o | HGRNBitAttention | Proyección ternaria de salida del bloque de atención | Ambos |
| BitLinear_gate | HGRNBitMLP | Expansión ternaria (dim intermedia = 2/3·d·ratio, múltiplo de 256) | Ambos |
| BitLinear_down | HGRNBitMLP | Contracción ternaria de vuelta a d_hidden | Ambos |
| BitLinear_LM | HNetBitForCausalLM | Cabecera de lenguaje: proyecta estados ocultos → logits sobre 256 bytes | Ambos |
| weight_quant (absmean) | BitLinear | round_clamp(W·α, -1, 1)/α. Pesos en {-1,0,+1}. | Solo entrenamiento (cada forward). En despliegue se precuantiza una sola vez. |
| activation_quant (8-bit) | BitLinear | Cuantización simétrica por token: round_clamp(x·βₜ, -128, 127)/βₜ | Ambos |
| STE (detach) | BitLinear | quant(x)+(x-quant(x)).detach(): forward cuantizado, backward con valores completos. | Solo entrenamiento (inocuo bajo torch.no_grad()) |

### Normalización y activaciones

| Componente | Pertenece a | Qué hace | Régimen |
|-----------|-------------|----------|---------|
| RMSNorm | BitLinear, HGRNBitBlock, HGRNBitStack | x / sqrt(mean(x²)+ε) ⊙ γ. Pre-norm. | Ambos |
| SwiGLU | HGRNBitAttention, HGRNBitMLP | Swish(x) ⊙ g = x⊙g/(1+e⁻ˣ). Kernel CUDA con forward+backward fusionados. | Ambos |
| FusedRMSNormSwishGate | HGRNBitAttention | RMSNorm(BitLinear_g(x)) ⊙ h ⊙ σ(h). 4 operaciones fusionadas en 1 kernel. | Ambos |
| σ (sigmoide) | HGRNBitAttention | Convierte f en compuerta de olvido: σ(f) ∈ (0,1)ᵈ | Ambos |

### Recurrencia HGRN

| Componente | Pertenece a | Qué hace | Régimen |
|-----------|-------------|----------|---------|
| fused_recurrent_hgrn | HGRNBitAttention | hₜ = gₜ⊙hₜ₋₁ + xₜ, g en probability-space (0,1). Paralelo en heads+batch, serial en tiempo. | Principalmente inferencia. También para secuencias cortas en entrenamiento. |
| chunk_hgrn | HGRNBitAttention | hₜ = exp(gₜ)⊙hₜ₋₁ + xₜ, g en log-space para evitar underflow. Fragmentos C=128 en paralelo. | Solo entrenamiento (secuencias largas). |

### Convolución

| Componente | Pertenece a | Qué hace | Régimen |
|-----------|-------------|----------|---------|
| Conv1D causal (kernel=4) | HGRNBitAttention | Convolución depthwise antes de proyecciones recurrentes. Captura patrones locales. | Ambos (habilitado por defecto). |

### Chunking dinámico

| Componente | Pertenece a | Qué hace | Régimen |
|-----------|-------------|----------|---------|
| RoutingModuleBit | HNetBit | Similitud coseno entre q,k de tokens consecutivos. Produce p ∈ [0,1] y m ∈ {0,1}. FP32, inicializado como identidad. | Ambos (modo lote en entrenamiento, paso a paso en inferencia). |
| ChunkLayer | HNetBit | Sin parámetros. Extrae tokens límite con truco de argsort. | Ambos (padded mode en entrenamiento, packed para variables). |
| DeChunkLayer | HNetBit | Sin parámetros. Reconstruye secuencia vía EMA. Implementado con fused_recurrent_hgrn + plug_back_idx. | Ambos. |
| L_LB (load balancing) | HNetBitForCausalLM | Pérdida auxiliar que evita colapso: penaliza extremos (todos/nadie límite). | Solo entrenamiento. |
| RoutingModuleState | HNetBitCache | Estado del enrutador en inferencia: has_seen_tokens, last_hidden_state. | Solo inferencia. |
| DeChunkState | HNetBitCache | Estado del dechunking en inferencia: last_value del EMA. | Solo inferencia. |

### Atención con ventana (solo hybrid_attn, etapa interna)

| Componente | Pertenece a | Qué hace | Régimen |
|-----------|-------------|----------|---------|
| CausalMHABit | HGRNBitBlock | Atención multi-cabeza con ventana deslizante (w=64). Q,K,V,O en FP32. | Ambos (solo variante hybrid_attn). |
| RoPE | CausalMHABit | x⊙cos(mθ)+rotate_half(x)⊙sin(mθ). Codifica posición relativa. | Ambos (solo variante hybrid_attn). |
| Sliding window mask | CausalMHABit | Máscara causal con ventana w=64. | Ambos (solo variante hybrid_attn). |

### Conexiones residuales

| Componente | Pertenece a | Qué hace | Régimen |
|-----------|-------------|----------|---------|
| Pre-norm residual | HGRNBitBlock | x' = x + f(RMSNorm(x)). Normaliza antes de cada subcapa. | Ambos. |
| Linear_res (FP32, init=0) | HNetBit (etapa no-interna) | Proyección residual en FP32 inicializada a cero. | Ambos. |
| STE(p) en residual | HNetBit (etapa no-interna) | H_dechunk · STE(p) + R. STE permite gradiente a través de p. ∂/∂p(H·p) = H. | Solo entrenamiento (backprop). |

### Pérdidas

| Componente | Pertenece a | Qué hace | Régimen |
|-----------|-------------|----------|---------|
| L_CE (entropía cruzada) | HNetBitForCausalLM | -log(probabilidad del byte correcto), con shift-1 para next-token prediction. | Solo entrenamiento. |
| L_total | HNetBitForCausalLM | L_CE + λ_LB · Σ_s L_LB(s), con λ_LB = 0.01. | Solo entrenamiento. |

### Caché de inferencia

| Componente | Pertenece a | Qué hace | Régimen |
|-----------|-------------|----------|---------|
| HNetBitCache | HNetBitForCausalLM | Caché raíz recursiva que espeja la jerarquía. Tamaño O(d) por capa (vs O(L·d) del KV cache del Transformer). | Solo inferencia. |
| HGRNBlockCache | HNetBitCache | Almacena (conv_state, h_state) por capa HGRN. h_state tiene tamaño fijo d. | Solo inferencia. |
| slice/merge de caché | HNetBitCache | Particiona caché para tokens límite y fusiona resultados de vuelta. | Solo inferencia. |
| reorder_cache(beam_idx) | HNetBitCache | Reordena recursivamente todos los estados para beam search. | Solo inferencia. |
| Omisión condicional | HNetBit | Si el token no es límite, la etapa interna se salta. >80% de los pasos. | Solo inferencia. |

### Infraestructura (no componente del modelo, pero relevante)

| Componente | Qué hace | Régimen |
|-----------|----------|---------|
| Padding dimensional | Si d_s > d_{s-1}, concatena ceros aprendibles para igualar dimensión entre etapas. | Ambos. |
| Recorte dimensional | Trunca salida a dimensión d_{s-1} de la etapa padre. | Ambos. |
| model_deploy.pt | Export con pesos ternarios congelados y empaquetados (~2.1 bits/param). | Solo despliegue (inferencia). |

---

## 20. Resumen de diferencias entrenamiento vs inferencia

| Aspecto | Entrenamiento | Inferencia |
|---|---|---|
| Secuencia | Completa (4096 bytes, forward completo) | Byte por byte (step mode) |
| Chunking | Todos los límites simultáneamente (modo lote) | Un byte a la vez (RoutingModuleState) |
| Stage interna | Siempre procesa chunks completos | Se salta si el byte no es límite (>80% de pasos) |
| Caché | No se usa (kernel paralelo chunk_hgrn) | HNetBitCache acumula estados |
| Gradientes | Sí (backprop, autograd, STE activo) | No (torch.no_grad(), STE inocuo) |
| Pesos | {-1,0,+1} con STE en forward, FP32 para gradientes | {-1,0,+1} congelados (despliegue) o re-cuantizados (entrenamiento-like) |
| Kernel recurrencia | chunk_hgrn (log-space, C=128, paralelo) | fused_recurrent_hgrn (probability-space, secuencial) |
| Pérdidas | L_CE + λ_LB · Σ L_LB | No hay |
| VRAM (150M hybrid) | ~15.5 GB | ~652 MB |
| Pipeline | Input → Embed → E → Route → Chunk → Stage1 → Dechunk → Residual → D → LM Head → Loss | Input token → Embed → E step → Route step → (Chunk+Stage1 si límite) → Dechunk → Residual → D step → LM Head → next token |
