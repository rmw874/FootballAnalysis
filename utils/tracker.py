from ultralytics import YOLO
import supervision as sv
import pickle
import os
import cv2
from .bbox_tools import get_bbox_center, get_bbox_width

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

class Tracker:
    def __init__(self, model_path): 
        #defining attributes
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    def detect_frames(self, frames):
        batch_size = 10
        raw_detections = []
        for i in range(0, len(frames), batch_size):
            batch = frames[i:i+batch_size]
            batch_detections = self.model.predict(batch, conf=0.1)
            raw_detections.extend(batch_detections)
        return raw_detections

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):

        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, "rb") as f:
                return pickle.load(f)


        raw_detections = self.detect_frames(frames)
        tracks = {
            "players": [],
            "ball": [],
            "referees": []
            #we don't have goalkeepers in the tracks, because we are considering them as players for simplicity.
        }


        for frame, frame_detection in enumerate(raw_detections):
            labels = frame_detection.names
            labels_id = {val:key for key, val, in labels.items()}
            # change format from ultralytics to supervision detection
            detection_sv = sv.Detections.from_ultralytics(frame_detection)

            # overwriting goalkeeper to player object for simplicity
            for obj_index, class_id in enumerate(detection_sv.class_id):
                if labels[class_id] == "goalkeeper":
                    detection_sv.class_id[obj_index] = labels_id["player"]
            
            # tracking objects
            tracked_detections = self.tracker.update_with_detections(detection_sv)

            # make dictionary of tracks
            tracks["players"].append({})
            tracks["ball"].append({})
            tracks["referees"].append({})

            for frame_detection in tracked_detections:
                bbox = frame_detection[0].tolist()
                label_id = frame_detection[3]
                track_id = frame_detection[4]

                #only track players and referees like this, as there is only one ball in play.
                if label_id == labels_id['player']:
                    tracks["players"][frame][track_id] = {"bbox":bbox}

                if label_id == labels_id['referee']:
                    tracks["referees"][frame][track_id] = {"bbox":bbox}

            # track ball
            for frame_detection in detection_sv:
                bbox = frame_detection[0].tolist()
                label_id = frame_detection[3]
                if label_id == labels_id['ball']:
                    tracks["ball"][frame][1] = {"bbox":bbox}

        if stub_path is not None:
            with open(stub_path, "wb") as f:
                pickle.dump(tracks, f)

        return tracks
    
    def draw_ellipse(self, frame, bbox, color, track_id=None, player=False):
        y2 = int(bbox[3])
        x_center, _ = get_bbox_center(bbox)
        width = get_bbox_width(bbox)

        cv2.ellipse(
            frame, 
            center = (x_center, y2), 
            axes = (int(width), int(width*0.4)),
            angle = 0.0,
            startAngle = -45,
            endAngle = 235,
            color = color,
            thickness = 2,
            lineType=cv2.LINE_4
            )
        
        if player:
            rectangle_color = color
            rectangle_width = 80
            rectangle_height = 20
            x1_rect = x_center - rectangle_width//2
            x2_rect = x_center + rectangle_width//2
            y1_rect = (y2 - rectangle_height//2 + 15)
            y2_rect = (y2 + rectangle_height//2 + 15)
            cv2.rectangle(
                frame, 
                (int(x1_rect), int(y1_rect)), 
                (int(x2_rect), int(y2_rect)), 
                color, 
                cv2.FILLED
                )
            x1_text = x1_rect + 5
            # if track_id > 99:
            #     x1_text -=10

            cv2.putText(
                frame, 
                f"uid: {track_id}", 
                (int(x1_text), int(y1_rect + 15)), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                fontScale=0.5, 
                color=(0, 0, 0), 
                thickness=2
                )

        return frame

    def draw_annots(self, frames, tracks):
        output_frames = []
        for frame_index, frame in enumerate(frames):
            frame = frame.copy()
            player_dict = tracks["players"][frame_index]
            ball_dict = tracks["ball"][frame_index]
            referee_dict = tracks["referees"][frame_index]

            for track_id, player in player_dict.items():
                frame = self.draw_ellipse(frame, player["bbox"], (255, 0, 0), track_id, player=True)

            for _, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee["bbox"], (255, 255, 255))

            output_frames.append(frame)

        return output_frames