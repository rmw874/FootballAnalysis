import numpy as np
from sklearn.cluster import KMeans

#Remove warnings
import warnings
warnings.filterwarnings("ignore")

class TeamAssign:
    def __init__(self):
        self.team_cols = {}
        self.player_team_dict = {} # player_id: team

    def get_player_color(self, frame, bbox):
        img = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
        img_2d = img.reshape(-1, 3) #3 channels (RGB)

        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=1, random_state=0).fit(img_2d)
        labels = kmeans.labels_
        clustered_img = labels.reshape(img.shape[0], img.shape[1])

        corner_cluster = [clustered_img[0, 0], clustered_img[0, -1], clustered_img[-1, 0], clustered_img[-1, -1]]
        bg_cluster = max(set(corner_cluster), key = corner_cluster.count)
        player_cluster = 1 - bg_cluster
        player_center = np.mean(img[clustered_img == player_cluster], axis=0)
        return player_center

    def assign_col_to_team(self, frame, player):
        player_cols = []
        for _, player in player.items():
            bbox = player['bbox']
            player_col = self.get_player_color(frame, bbox)
            player_cols.append(player_col)
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=1).fit(player_cols)
        
        self.kmeans = kmeans

        self.team_cols[0] = kmeans.cluster_centers_[0]
        self.team_cols[1] = kmeans.cluster_centers_[1] 
    
    def assign_team(self, frame, player_bbox, player_id):
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]
        player_col = self.get_player_color(frame, player_bbox)
        team = self.kmeans.predict(player_col.reshape(1,-1))[0]
        self.player_team_dict[player_id] = team
        return team