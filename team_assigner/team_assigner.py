from sklearn.cluster import KMeans

class TeamAssigner:
    """
    The TeamAssigner class is responsible for assigning players to teams based on the dominant color in their bounding boxes.
    It uses K-means clustering to classify players into two distinct teams based on their uniform colors.
    """

    def __init__(self):
        """
        Initializes the TeamAssigner with empty dictionaries for storing team colors and the player-team assignments.
        """
        self.team_colors = {}  # Dictionary to store the color of each team
        self.player_team_dict = {}  # Dictionary to map players to their assigned team

    def get_clustering_model(self, image):
        """
        Applies K-means clustering on the provided image to separate the image into 2 clusters.

        Args:
            image (np.array): The image data (assumed to be a 2D array of RGB values).

        Returns:
            KMeans: Fitted KMeans object with 2 clusters.
        """
        # Reshape the image into a 2D array of pixels (each pixel represented by 3 values for RGB)
        image_2d = image.reshape(-1, 3)

        # Perform K-means clustering to classify the image pixels into 2 clusters (e.g., player and non-player colors)
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=1)
        kmeans.fit(image_2d)

        return kmeans

    def get_player_color(self, frame, bbox):
        """
        Extracts the dominant player color from the bounding box of the player.

        Args:
            frame (np.array): The video frame containing the player.
            bbox (list): The bounding box [x1, y1, x2, y2] specifying the region of the player in the frame.

        Returns:
            np.array: The RGB values representing the player's color.
        """
        # Crop the image to the player's bounding box
        image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]

        # Consider only the top half of the image (likely to contain the jersey)
        top_half_image = image[0:int(image.shape[0] / 2), :]

        # Get the clustering model for the top half of the player's image
        kmeans = self.get_clustering_model(top_half_image)

        # Get the cluster labels for each pixel
        labels = kmeans.labels_

        # Reshape the labels to match the original shape of the image
        clustered_image = labels.reshape(top_half_image.shape[0], top_half_image.shape[1])

        # Identify which cluster represents the player by analyzing the corner pixels
        corner_clusters = [clustered_image[0, 0], clustered_image[0, -1], clustered_image[-1, 0], clustered_image[-1, -1]]
        non_player_cluster = max(set(corner_clusters), key=corner_clusters.count)
        player_cluster = 1 - non_player_cluster  # Opposite of non-player cluster

        # Get the color corresponding to the player's cluster
        player_color = kmeans.cluster_centers_[player_cluster]

        return player_color

    def assign_team_color(self, frame, player_detections):
        """
        Assigns a team color to each team based on the colors detected in the player bounding boxes.

        Args:
            frame (np.array): The video frame containing the players.
            player_detections (dict): A dictionary containing player detections with their bounding boxes.
        """
        player_colors = []
        # Loop through all detected players and extract their colors
        for _, player_detection in player_detections.items():
            bbox = player_detection["bbox"]
            player_color = self.get_player_color(frame, bbox)
            player_colors.append(player_color)
        
        # Apply K-means clustering to group the players into two teams based on their colors
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=10)
        kmeans.fit(player_colors)

        self.kmeans = kmeans  # Store the trained K-means model for later use

        # Assign the cluster centroids as the team colors
        self.team_colors[1] = kmeans.cluster_centers_[0]
        self.team_colors[2] = kmeans.cluster_centers_[1]

    def get_player_team(self, frame, player_bbox, player_id):
        """
        Predicts and assigns a team to a player based on the player's color and the K-means model.

        Args:
            frame (np.array): The video frame containing the player.
            player_bbox (list): The bounding box of the player [x1, y1, x2, y2].
            player_id (int): The unique ID of the player.

        Returns:
            int: The team ID (either 1 or 2) that the player belongs to.
        """
        # If the player is already assigned to a team, return the team assignment
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]

        # Otherwise, get the player's color and use the K-means model to predict the team
        player_color = self.get_player_color(frame, player_bbox)
        team_id = self.kmeans.predict(player_color.reshape(1, -1))[0]
        team_id += 1  # Adjust team ID to be 1-based (1 or 2)

        # Hardcode goalkeeper assignments to specific teams
        if player_id in [73, 102, 67, 59]:
            team_id = 1  # Assign to team 1
        if player_id == 168:
            team_id = 2  # Assign to team 2

        # Save the player's team assignment
        self.player_team_dict[player_id] = team_id

        return team_id
