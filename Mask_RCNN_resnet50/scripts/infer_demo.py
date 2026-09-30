"""Inferencia de torchvision Mask R-CNN R50-FPN sobre las imagenes de ejemplo
del repo Matterport, para comparar con el port de TF2."""
import os, sys, time, torch
from PIL import Image
from torchvision.models.detection import (maskrcnn_resnet50_fpn_v2,
                                          MaskRCNN_ResNet50_FPN_V2_Weights)
import torchvision.transforms.functional as TF

# Relativa al script, para no depender de donde este el proyecto:
RAIZ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
IMGS = os.path.join(RAIZ, "Mask_RCNN-master", "images")
UMBRAL = 0.7

W = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
CLASES = W.meta["categories"]
modelo = maskrcnn_resnet50_fpn_v2(weights=W).eval().cuda()

archivos = sorted(os.listdir(IMGS))[:4]
with torch.inference_mode():
    for i, fn in enumerate(archivos):
        img = Image.open(os.path.join(IMGS, fn)).convert("RGB")
        x = TF.to_tensor(img).cuda()
        torch.cuda.synchronize(); t = time.time()
        r = modelo([x])[0]
        torch.cuda.synchronize(); dt = time.time() - t
        keep = r["scores"] >= UMBRAL
        etiquetas = [CLASES[c] for c in r["labels"][keep].tolist()]
        scores = r["scores"][keep].tolist()
        masks = r["masks"][keep]
        print("%-28s %5.2fs %2d obj  masks=%s" % (fn, dt, int(keep.sum()), tuple(masks.shape)))
        print("       ", ", ".join("%s %.2f" % (e, s) for e, s in list(zip(etiquetas, scores))[:8]))
