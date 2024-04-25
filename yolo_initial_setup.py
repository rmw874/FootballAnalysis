from ultralytics import YOLO

model = YOLO('models/best.pt')

results = model.predict("data/798b45_4.mp4", project="runs", save=True)

for bbox in results[0].boxes:
    print(bbox)