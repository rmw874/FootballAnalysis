from utils import read_video, write_video, Tracker

def main():
    video_frames = read_video("data/0a2d9b_4.mp4")
    tracker = Tracker("models/best.pt")
    tracker.get_object_tracks(video_frames)
    write_video(video_frames, "results/output_tracker.avi")

if __name__ == "__main__":
    main()