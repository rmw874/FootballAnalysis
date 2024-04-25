from ultralytics import YOLO
import supervision as sv

class Tracker:
    def __init__(self, model_path): 
        #defining attributes
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    def detect_frames(self, frames):
        batch_size = 10
        detections = []
        for i in range(0, len(frames), batch_size):
            batch = frames[i:i+batch_size]
            batch_detections = self.model.predict(batch, conf=0.1)
            detections.extend(batch_detections)
        return detections

    def get_object_tracks(self, frames):
        detections = self.detect_frames(frames)
        for frame, detection in enumerate(detections):
            label = detection.names
            label_to_index = {val:key for key, val, in label.items()}
            # change format from ultralytics to supervision detection
            detection_sv = sv.Detections.from_ultralytics(detection)

            # overwriting goalkeeper to player object for simplicity
            for obj_index, label_index in enumerate(detection_sv.label_index):
                if label[label_index] == "goalkeeper":
                    detection_sv.label_index[obj_index] = label_to_index["player"]