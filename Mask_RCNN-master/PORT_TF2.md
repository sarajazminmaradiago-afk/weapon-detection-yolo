# Port de Mask R-CNN a TensorFlow 2.x + GPU Blackwell

Este repositorio es el Mask R-CNN de Matterport (2019), escrito para
**TensorFlow 1.x + Keras 2.0**. Se ha portado a **TensorFlow 2.21 + tf-keras**
para poder usar la GPU RTX 5070 (Blackwell, `sm_120`).

Los ficheros originales estan intactos junto a los portados, con extension
`.tf1.bak` (`mrcnn/model.py.tf1.bak`, etc.). Para ver los cambios:

    diff mrcnn/model.py.tf1.bak mrcnn/model.py

## Uso

    source activate_mrcnn.sh          # activa venv + CUDA + Keras 2 legacy

Ese script es obligatorio: exporta `LD_LIBRARY_PATH` (TF no localiza solo las
librerias CUDA instaladas por pip) y `TF_USE_LEGACY_KERAS=1` (el repo usa la
API de Keras 2; TF 2.21 trae Keras 3, que es incompatible).

## Por que la GPU funciona pese al aviso

Al arrancar veras:

    TensorFlow was not built with CUDA kernel binaries compatible with
    compute capability 12.0a. CUDA kernels will be jit-compiled from PTX...

Es esperado: no hay binarios precompilados para Blackwell, asi que se compilan
desde PTX en la primera ejecucion (~35 s) y quedan cacheados en `~/.nv/ComputeCache`.
Despues: 0.17 s por imagen en inferencia, ~85 ms por paso de entrenamiento.

## Cambios aplicados

### Entorno / API
1-2. `import keras.*` -> `tensorflow.keras.*`; `keras.engine` (eliminado)
     -> `tensorflow.keras.layers` para la clase base `Layer`.
3.   `tf.log`, `tf.random_shuffle`, `tf.sets.set_intersection`,
     `tf.sparse_tensor_to_dense`, `tf.to_float` -> nombres de TF 2.x.
4.   `np.bool` (eliminado en numpy 1.24) -> `bool`.
5.   `os.name is 'nt'` -> `==`.
8.   Assert de version reescrito (`distutils` esta obsoleto); ahora comprueba
     que `tf.keras` apunta realmente a tf-keras y no a Keras 3.
16.  `skimage.transform.resize` ya no interpola `bool`: las mascaras se
     convierten a `float32` antes de redimensionar.

### Formas dinamicas
9.  `KL.Reshape` con `num_rois=None` -> se usa `-1`.
10. `tf.range(probs.shape[0])` -> `tf.shape(probs)[0]`.

### Modelo funcional de Keras 2 en TF2
12. Tres `KL.Lambda` capturaban `input_image` por clausura; ahora lo reciben
    como entrada explicita. `tf.Variable` dentro de una Lambda (el "hack" para
    las anclas) se sustituye por una constante replicada al lote.
17. La Lambda de anclas sufria ademas clausura tardia de Python: el nombre
    `anchors` acababa apuntando a la salida de la propia Lambda.

### compile() y bucle de entrenamiento
13. Las perdidas simbolicas solo pueden anadirse una vez (`train()` llama a
    `compile()` en cada etapa). Se usa un flag en lugar de vaciar `_losses`,
    atributo que en TF2 esta rastreado y no admite reasignacion.
14. La regularizacion L2 se pasa como *callable*, no como tensor evaluado; asi
    se reevalua tras cada `set_trainable()`.
15. `metrics_tensors` no existe -> `add_metric()`, y debe ejecutarse ANTES de
    `compile()`.
18-19. El generador cedia `(inputs, [])`; una lista de targets vacia rompe el
    contenedor de metricas. Ahora cede un dict con los nombres de las capas
    `Input`, que Keras interpreta inequivocamente como una unica `x`.

### Correccion numerica (importante)
20. `detection_targets_graph` recortaba las mascaras objetivo con
    `crop_and_resize` usando cajas derivadas de `positive_rois` **sin**
    `stop_gradient`. Eso abria una ruta de gradiente desde `mrcnn_mask_loss`
    hasta `rpn_bbox_pred`, atravesando la division `/gt_h` y el `tf.round`
    (gradiente 0): el producto `0*inf` producia `NaN`, que destruia los pesos
    del FPN y de la RPN.

    Sintoma: la perdida era finita pero el gradiente `NaN`, y el entrenamiento
    se arruinaba en aproximadamente la mitad de las ejecuciones tras unas
    decenas de pasos.

    Los objetivos son etiquetas y no deben propagar gradiente, asi que ahora
    llevan `stop_gradient` (lo mismo que ya hacia `PyramidROIAlign` con sus
    cajas, y lo que hacen Detectron y mmdetection con las propuestas).

## Verificado

- Construccion del modelo en modo `training` e `inference` (394 capas, 64.2M params).
- Inferencia con pesos COCO: detecciones correctas sobre `images/`.
- Carga de pesos con `exclude=` (81 clases -> 4), imprescindible para reentrenar.
- Entrenamiento en dos etapas (`heads` y luego `all`), checkpoints y recarga.
- Evaluacion con `utils.compute_ap`.
- Ausencia de NaN en 300 pasos x 4 semillas, incluidas dos que antes fallaban.

Prueba de aprendizaje de extremo a extremo sobre el dataset sintetico `shapes`
(transfer learning desde COCO, 8 epocas de cabezas + 16 de red completa):

    etapa 1 (heads): loss 1.521 -> 1.233
    etapa 2 (all)  : loss 1.294 -> 0.656
    detecciones/imagen: 1.8  (objetos reales: 1.9)
    mAP@IoU=0.5 = 0.933

Es decir, el port no solo se ejecuta: converge y alcanza la precision esperada.

## Limitaciones conocidas

- `mrcnn/parallel_model.py` (multi-GPU) se ha portado en imports pero NO se ha
  probado: solo hay una GPU en esta maquina.
- El aviso `map_fn_v2 ... dtype is deprecated` es inocuo.
