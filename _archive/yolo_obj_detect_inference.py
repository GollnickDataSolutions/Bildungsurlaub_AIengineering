#%% Pakete
import onnxruntime as ort
from PIL import Image
import numpy as np
#%% Model path
model_path = "C:\\Temp\\Bildungsurlaub_AIengineering\\runs\\detect\\train-3\\weights\\best.onnx"
# %% Load the ONNX model
session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

# %% Beispielbild laden
img = Image.open("../output.png")
img

#%% inference with session and img
# InvalidArgument: [ONNXRuntimeError] : 2 : INVALID_ARGUMENT : Got invalid dimensions for input: images for the following indices

img_resized = img.convert("RGB").resize((640, 640))
img_data = np.array(img_resized).astype('float32') / 255.0  # YOLO erwartet 0..1
img_data = np.transpose(img_data, (2, 0, 1))  # Change HWC to CHW
img_data = np.expand_dims(img_data, axis=0)
print(img_data.shape)

inputs = {session.get_inputs()[0].name: img_data}
outputs = session.run(None, inputs)
outputs

# %% Vorhersagen dekodieren (cx, cy, w, h + 80 Klassen-Scores)
import ast
import matplotlib.pyplot as plt
import matplotlib.patches as patches

names = ast.literal_eval(session.get_modelmeta().custom_metadata_map["names"])
conf_threshold = 0.25
iou_threshold = 0.45

pred = outputs[0][0].T  # (8400, 84)
boxes_xywh = pred[:, :4]
scores_all = pred[:, 4:]
class_ids = scores_all.argmax(axis=1)
scores = scores_all.max(axis=1)

keep = scores > conf_threshold
boxes_xywh, scores, class_ids = boxes_xywh[keep], scores[keep], class_ids[keep]

# xywh -> xyxy und auf die Originalgroesse zurueckskalieren
scale_x, scale_y = img.width / 640, img.height / 640
cx, cy, w, h = boxes_xywh.T
boxes = np.stack([(cx - w / 2) * scale_x,
                  (cy - h / 2) * scale_y,
                  (cx + w / 2) * scale_x,
                  (cy + h / 2) * scale_y], axis=1)


# %% Non-Maximum Suppression (Modell ist nicht end2end, liefert also Duplikate)
def nms(boxes, scores, iou_threshold):
    order = scores.argsort()[::-1]
    keep = []
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou < iou_threshold]
    return np.array(keep, dtype=int)


# klassenweise NMS ueber die Originalindizes
final = []
for c in np.unique(class_ids):
    idx = np.where(class_ids == c)[0]
    final.extend(idx[nms(boxes[idx], scores[idx], iou_threshold)])
final = np.array(final, dtype=int)

print(f"{len(final)} Objekte gefunden")
for i in final:
    print(f"  {names[int(class_ids[i])]}: {scores[i]:.2f} @ {boxes[i].round(1)}")


# %% Boxen ueber dem Bild anzeigen
fig, ax = plt.subplots(figsize=(10, 10 * img.height / img.width))
ax.imshow(img)
cmap = plt.get_cmap("tab20")

for i in final:
    x1, y1, x2, y2 = boxes[i]
    color = cmap(int(class_ids[i]) % 20)
    ax.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                   linewidth=2, edgecolor=color, facecolor="none"))
    ax.text(x1, y1 - 4, f"{names[int(class_ids[i])]} {scores[i]:.2f}",
            color="white", fontsize=10,
            bbox=dict(facecolor=color, edgecolor="none", pad=1.5))

ax.axis("off")
plt.tight_layout()
plt.show()

# %%
