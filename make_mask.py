import cv2, torch, numpy as np
from ultralytics import YOLOWorld            # детектор
from segment_anything import sam_model_registry, SamPredictor
from PIL import Image

# 1. модели
det = YOLOWorld("yolov8s-worldv2.pt")        # 45 MB, RTX‑дружелюбно
det.set_classes(["cat"])                     # prompt
sam = sam_model_registry["vit_h"](
          checkpoint="sam_vit_h_4b8939.pth").cuda()
predictor = SamPredictor(sam)

# 2. детектируем котиков
img = cv2.imread("./cats_test/cat.png")[:,:,::-1]
y = det.predict(img, conf=0.3, iou=0.6)[0]   # y.boxes.xyxy in xyxy
boxes = y.boxes.xyxy.cpu().numpy()

# 3. сегментация SAM
predictor.set_image(img)
masks = [ predictor.predict(box=box, multimask_output=False)[0][0]
          for box in boxes ]

# 4. экспорт PNG с альфой
mask = (np.sum(masks, 0) > 0).astype("uint8")*255
out  = Image.fromarray(img).convert("RGBA")
out.putalpha(Image.fromarray(mask))
out.save("cat_cut.png")
