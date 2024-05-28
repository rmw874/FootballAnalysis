import cv2
from utils import read_video, write_video, Tracker, TeamAssign

def main():
    video_frames = read_video("data/0a2d9b_4.mp4")
    tracker = Tracker("models/best.pt")

    tracks = tracker.get_object_tracks(video_frames, 
                              read_from_stub=True, 
                              stub_path="stubs/track_stubs.pkl")

    team_assigner = TeamAssign()
    team_assigner.assign_col_to_team(video_frames[0], tracks['players'][0]) #0 is an arbitrary frame number

    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.assign_team(video_frames[frame_num],   
                                             track['bbox'],
                                             player_id)
            tracks['players'][frame_num][player_id]['team'] = team 
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_cols[team]
    
    #inspect team colors
    for team, color in team_assigner.team_cols.items():
        print(f"Team {team} color: {color}")

    output_frames = tracker.draw_annots(video_frames, tracks) 
    write_video(output_frames, "results/output_tracker.avi")

if __name__ == "__main__":
    main()