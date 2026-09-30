# Mask R-CNN ResNet-50 (torchvision) — termino de comparacion

Implementacion moderna de Mask R-CNN para comparar contra el port a TF2 del
repo Matterport que vive en `../Mask_RCNN-master`.

## Uso

    source activate.sh
    python scripts/infer_demo.py

No hace falta exportar `LD_LIBRARY_PATH` ni nada parecido: a diferencia de
TensorFlow, PyTorch encuentra sus librerias CUDA por si solo.

## Por que esta implementacion

`torchvision.models.detection.maskrcnn_resnet50_fpn_v2` es la implementacion de
referencia mantenida, con pesos COCO oficiales. Y sobre todo: **corre
nativamente en la RTX 5070**.

    torch.cuda.get_arch_list() -> ['sm_75','sm_80','sm_86','sm_90','sm_100','sm_120']

`sm_120` esta en la lista, asi que hay kernels precompilados para Blackwell. El
port de TensorFlow, en cambio, tiene que compilarlos desde PTX en la primera
ejecucion (~35 s).

## Modelos disponibles

| Modelo | Params | mAP COCO box | mAP COCO mask |
|---|---|---|---|
| `maskrcnn_resnet50_fpn`    | 44.4M | 37.9 | 34.6 |
| `maskrcnn_resnet50_fpn_v2` | 46.4M | 47.4 | 41.8 |

Se usa **v2** por defecto: mismo backbone, cabezas y receta de entrenamiento
mejores. Los pesos quedan cacheados en `~/.cache/torch/hub/checkpoints/`.

## Diseno de la comparacion

Para que la comparacion sea interpretable, **ambos lados van con ResNet-50**.
En Matterport basta poner en la config:

    BACKBONE = "resnet50"

Comprobado: al cargar `mask_rcnn_coco.h5` en un Matterport-R50, **el 100% de sus
131 capas con pesos encuentran correspondencia** (45.1M params, frente a 233
capas y 64.2M en R101). En la nomenclatura de Matterport, ResNet-50 es un
prefijo exacto de ResNet-101.

Matiz honesto que conviene recoger en la memoria: esos pesos se entrenaron
*dentro* de una ResNet-101, asi que Matterport-R50 arranca de un "warm start"
muy bueno pero no de una R50-COCO entrenada como tal. torchvision si trae una
R50-COCO nativa, y eso le da cierta ventaja de partida.

## Diferencias que ya se observan (pesos COCO, mismas imagenes)

Sobre `../Mask_RCNN-master/images`, umbral 0.7:

| | Matterport R101 (TF2) | torchvision R50-v2 |
|---|---|---|
| Tiempo 1a imagen | 35 s (JIT desde PTX) | 0.24 s |
| Tiempo siguientes | 0.17 s | 0.04-0.06 s |
| `2383514521` (jirafas) | jirafa, cebra, jirafa, **vaca 0.90** | cebra, jirafa, jirafa |

Ese "vaca 0.90" de Matterport es un falso positivo que v2 no comete. Es una
observacion anecdotica sobre 4 imagenes, no una medida: la comparacion seria
va con los datasets de armas y mAP.

## Pendiente

- Escribir el `WeaponDataset` para los XML de Pascal VOC del DaSCI, en los dos
  formatos (Matterport `utils.Dataset` y torchvision `Dataset`).
- Fijar particiones train/val identicas para ambos.
- Comparar con el mismo protocolo de evaluacion (mismo IoU, mismas imagenes).
