import numpy as np

class StdMontageMapper:
    """
    Maps signals from a set of original channels to a set of standard channels.
    """
    def __init__(self, original_channels, sensors, threshold_distance=0.06):
        self.original_channels = original_channels
        self.sensors = sensors
        self.threshold_distance = threshold_distance

    @property    
    def original_ch_locs(self):
        return np.array([c['loc'][:3] for c in self.original_channels])
    
    @property
    def original_ch_names(self):
        return np.array([c['ch_name'] for c in self.original_channels])
        
    @property
    def standard_locs(self):
        return np.array(list(self.sensors.values()))

    @property
    def standard_sensor_names(self):
        return np.array(list(self.sensors.keys()))
    
    def calculate_unit_vectors(self):
        """
        Calculates unit vectors from each standard location to each original location.
        """
        vectors = self.standard_locs[:, None, :] - self.original_ch_locs
        norms = np.linalg.norm(vectors, axis=2, keepdims=True)
        unit_vectors = vectors / (norms + 1e-10)  # Add small value to avoid division by zero
        return unit_vectors
    
    def calculate_influence_map(self, distance_factor=1.5, angle_factor=1.5):
        """
        Calculates the influence of each original channel on each standard channel.
        :param distance_factor: The power to which the distance should be raised in the influence calculation. Larger values give more weight to closer channels.
        :param angle_factor: The power to which the cosine of the angle between two channels should be raised in the influence calculation. Larger values penalize channels that are further away in the same direction more.
        :return: A numpy array of shape (n_standards, n_originals) with influence values.
        """
        distances = np.linalg.norm(self.standard_locs[:, None] - self.original_ch_locs, axis=2)
        treshold_mask = distances <= self.threshold_distance
        influence_map = 1 / (distances**distance_factor + 1e-10)
        influence_map[~treshold_mask] = 0

        # Reduce influence of channels that are further away in the same direction
        unit_vectors = self.calculate_unit_vectors()
        for i in range(len(self.standard_locs)):
            for j in range(len(self.original_ch_locs)):
                if influence_map[i, j] == 0:
                    continue
                for k in range(j + 1, len(self.original_ch_locs)):
                    if influence_map[i, k] == 0:
                        continue
                    # direction similarity
                    cos_similarity = np.dot(unit_vectors[i, j], unit_vectors[i, k])
                    
                    # adjustment factor, 1 for anything with more than 90 degrees difference, otherwise 1 - cos_similarity
                    # adjustment_factor = min(1, 1 - cos_similarity)

                    # adjustment factor that grows faster as the angle increases and slows down as it approaches 90 degrees, larger angle_factor means faster growth and slower decrease
                    if cos_similarity < 0:
                        adjustment_factor = 1
                    else:
                        adjustment_factor = np.cos(np.pi/2 * cos_similarity ** angle_factor)

                    # Determine which channel is further and adjust
                    if distances[i, j] < distances[i, k]:
                        influence_map[i, k] *= adjustment_factor
                    else:
                        influence_map[i, j] *= adjustment_factor

        # Normalize influence map
        for i in range(len(self.standard_locs)):
            row_sum = np.sum(influence_map[i])
            if row_sum > 0:
                influence_map[i] /= row_sum

        return influence_map
    
    def map_signals(self, original_values):
        """
        Maps the signals from original channels to standard channels based on the calculated influence map.
        :param original_values: A numpy array containing the signal values of the original channels.
        :return: A numpy array containing the mapped signal values of the standard channels.
        """
        self.influence_map = self.calculate_influence_map()
        # print(self.influence_map.shape) # (n_standards, n_originals)
        # print(original_values.shape) # (n_samples, n_originals, n_timepoints)
        mapped_values = np.einsum('ij, kjt -> kit', self.influence_map, original_values)
        return mapped_values
