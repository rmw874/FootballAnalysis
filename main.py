from utils import read_video, write_video, Tracker

def main():
    video_frames = read_video("data/0a2d9b_4.mp4")
    tracker = Tracker("models/best.pt")

    tracks = tracker.get_object_tracks(video_frames, 
                              read_from_stub=True, 
                              stub_path="stubs/track_stubs.pkl")
    
    output_frames = tracker.draw_annots(video_frames, tracks) 
    
    
    write_video(output_frames, "results/output_tracker.avi")

if __name__ == "__main__":
    main()