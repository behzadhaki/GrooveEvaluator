import numpy as np
from tqdm import tqdm
from sklearn.neighbors import KernelDensity
from scipy.stats import  gaussian_kde
from scipy import stats, integrate
from GrooveEvaluator.settings import FEATURES_TO_EXTRACT
import functools

import sklearn
from bokeh.io import output_file, show

class Feature_Extractor_From_HVO_Set:

    def __init__(
            self,
            hvo_dataset,
            features_to_extract=FEATURES_TO_EXTRACT,
            name=None
    ):
        """
        Extracts features for each sample within a list containing hvo_sequences
        (compatible with HVO_Sequence Version >= 0.1.0)

        :param hvo_dataset:                 List of HVO_Sequences
        :param features_to_extract:         Dictionary of features (for which extractors are implemented in hvo_sequence)
                                            Must be formatted same way as GrooveEvaluator.settings.FEATURES_TO_EXTRACT
        :param name:                        Name of dataset
        """

        self.__name = name

        # Add the dataset to self
        self.hvo_dataset = hvo_dataset

        self.features_to_extract = FEATURES_TO_EXTRACT

        # __extracted_features is a dictionary with same structure as FEATURES_TO_EXTRACT dict
        # the extracted features will be stored here upon calling the extract() method
        self.__extracted_features = None

    @property
    def name(self):
        return self.__name

    @name.setter
    def name(self, name_):
        self.__name = name_

    @property
    def extracted_features(self):
        # If extract() hasn't been called previously, do so to extract features
        # If features extracted, No need to do so anymore
        """if self.__extracted_features is None:
            self.extract()"""
        return self.__extracted_features

    def extract(self, use_tqdm=True, force_extract=False):

        should_extract = [False]
        if self.extracted_features is None:
            should_extract.append(True)
            self.__extracted_features = self.get_empty_extracted_features_dict(self.features_to_extract)

        if force_extract is True:
            should_extract.append(True)

        if any(should_extract):
            if use_tqdm:
                for ix in tqdm(range(len(self.hvo_dataset)), desc="Extracting Features from HVO_Sequence Set"):
                    sample_hvo = self.hvo_dataset[ix]
                    self.update_statistical_features(sample_hvo)
                    self.update_syncopation_features(sample_hvo)
                    self.update_autocorrelation_features(sample_hvo)
                    self.update_microtiming_features(sample_hvo)

            else:
                for ix in range(len(self.hvo_dataset)):
                    sample_hvo = self.hvo_dataset[ix]
                    self.update_statistical_features(sample_hvo)
                    self.update_syncopation_features(sample_hvo)
                    self.update_autocorrelation_features(sample_hvo)
                    self.update_microtiming_features(sample_hvo)

    def update_statistical_features(self, sample_hvo):

        statistical_keys = self.__extracted_features["Statistical"].keys()

        if "NoI" in statistical_keys:
            self.__extracted_features["Statistical"]["NoI"] = np.append(
                self.__extracted_features["Statistical"]["NoI"],
                sample_hvo.get_number_of_active_voices()
            )

        if "Total Step Density" in statistical_keys:
            self.__extracted_features["Statistical"]["Total Step Density"] = np.append(
                self.__extracted_features["Statistical"]["Total Step Density"],
                sample_hvo.get_total_step_density()
            )

        if "Avg Voice Density" in statistical_keys:
            self.__extracted_features["Statistical"]["Avg Voice Density"] = np.append(
                self.__extracted_features["Statistical"]["Avg Voice Density"],
                sample_hvo.get_average_voice_density()
            )

        if any(x in statistical_keys for x in ["Lowness", "Midness", "Hiness"]):
            lowness, midness, hiness = sample_hvo.get_lowness_midness_hiness()
            if "Lowness" in statistical_keys:
                self.__extracted_features["Statistical"]["Lowness"] = np.append(
                    self.__extracted_features["Statistical"]["Lowness"],
                    lowness
                )
            if "Midness" in statistical_keys:
                self.__extracted_features["Statistical"]["Midness"] = np.append(
                    self.__extracted_features["Statistical"]["Midness"],
                    midness
                )
            if "Hiness" in statistical_keys:
                self.__extracted_features["Statistical"]["Hiness"] = np.append(
                    self.__extracted_features["Statistical"]["Hiness"],
                    hiness
                )

        if "Vel Similarity Score" in statistical_keys:
            self.__extracted_features["Statistical"]["Vel Similarity Score"] = np.append(
                self.__extracted_features["Statistical"]["Vel Similarity Score"],
                sample_hvo.get_velocity_score_symmetry()
            )

        if "Weak to Strong Ratio" in statistical_keys:
            self.__extracted_features["Statistical"]["Weak to Strong Ratio"] = np.append(
                self.__extracted_features["Statistical"]["Weak to Strong Ratio"],
                sample_hvo.get_total_weak_to_strong_ratio()
            )

        if any(x in statistical_keys for x in ["Poly Velocity Mean", "Poly Velocity std"]):
            mean, std = sample_hvo.get_polyphonic_velocity_mean_stdev()
            if "Poly Velocity Mean" in statistical_keys:
                self.__extracted_features["Statistical"]["Poly Velocity Mean"] = np.append(
                    self.__extracted_features["Statistical"]["Poly Velocity Mean"],
                    mean
                )
            if "Poly Velocity std" in statistical_keys:
                self.__extracted_features["Statistical"]["Poly Velocity std"] = np.append(
                    self.__extracted_features["Statistical"]["Poly Velocity std"],
                    std
                )

        if any(x in statistical_keys for x in ["Poly Offset Mean", "Poly Offset std"]):
            mean, std = sample_hvo.get_polyphonic_offset_mean_stdev()
            if "Poly Offset Mean" in statistical_keys:
                self.__extracted_features["Statistical"]["Poly Offset Mean"] = np.append(
                    self.__extracted_features["Statistical"]["Poly Offset Mean"],
                    mean
                )
            if "Poly Offset std" in statistical_keys:
                self.__extracted_features["Statistical"]["Poly Offset std"] = np.append(
                    self.__extracted_features["Statistical"]["Poly Offset std"],
                    std
                )

    def update_syncopation_features(self, sample_hvo):
        sync_keys = self.__extracted_features["Syncopation"].keys()

        if "Combined" in sync_keys:
            self.__extracted_features["Syncopation"]["Combined"] = np.append(
                self.__extracted_features["Syncopation"]["Combined"],
                sample_hvo.get_combined_syncopation()
            )

        if "Polyphonic" in sync_keys:
            self.__extracted_features["Syncopation"]["Polyphonic"] = np.append(
                self.__extracted_features["Syncopation"]["Polyphonic"],
                sample_hvo.get_witek_polyphonic_syncopation()
            )

        if any(shared_feats in sync_keys for shared_feats in ["Lowsync", "Midsync", "Hisync",
                                                                       "Lowsyness", "Midsyness", "Hisyness"]):

            lmh_sync_info = sample_hvo.get_low_mid_hi_syncopation_info()

            for feat in ["Lowsync", "Midsync", "Hisync", "Lowsyness", "Midsyness", "Hisyness"]:
                if feat in sync_keys:
                    self.__extracted_features["Syncopation"][feat] = np.append(
                        self.__extracted_features["Syncopation"][feat],
                        lmh_sync_info[feat.lower()]
                    )

        if "Complexity" in sync_keys:
            self.__extracted_features["Syncopation"]["Complexity"] = np.append(
                self.__extracted_features["Syncopation"]["Complexity"],
                sample_hvo.get_total_complexity()
            )

    def update_autocorrelation_features(self, sample_hvo):
        autocorrelation_keys = self.__extracted_features["Auto-Correlation"].keys()

        if any(shared_feats in autocorrelation_keys for shared_feats in [
            "Skewness", "Max", "Centroid", "Harmonicity"]
               ):
            autocorrelation_features = sample_hvo.get_velocity_autocorrelation_features()

            for feat in ["Skewness", "Max", "Centroid", "Harmonicity"]:
                if feat in autocorrelation_keys:
                    self.__extracted_features["Auto-Correlation"][feat] = np.append(
                        self.__extracted_features["Auto-Correlation"][feat],
                        autocorrelation_features[feat.lower()]
                    )

    def update_microtiming_features(self, sample_hvo):
        microtiming_keys = self.__extracted_features["Micro-Timing"].keys()

        if "Swingness" in microtiming_keys:
            self.__extracted_features["Micro-Timing"]["Swingness"] = np.append(
                self.__extracted_features["Micro-Timing"]["Swingness"],
                sample_hvo.swingness()
            )

        if "Laidbackness" in microtiming_keys:
            self.__extracted_features["Micro-Timing"]["Laidbackness"] = np.append(
                self.__extracted_features["Micro-Timing"]["Laidbackness"],
                sample_hvo.laidbackness()
            )

        if "Accuracy" in microtiming_keys:
            self.__extracted_features["Micro-Timing"]["Accuracy"] = np.append(
                self.__extracted_features["Micro-Timing"]["Accuracy"],
                sample_hvo.get_timing_accuracy()
            )

    def get_empty_extracted_features_dict(self, _features_to_extract):
        # Create a new dictionary with same structure as _features_to_extract
        # Initialize to an empty np array and remove non required features
        extracted_features = \
            {
                type_key:
                    {
                        feat_key: np.array([], dtype=np.float16) for feat_key in _features_to_extract[type_key].keys()
                    } for type_key in _features_to_extract.keys()
            }

        for type in _features_to_extract.keys():
            for feat in _features_to_extract[type].keys():
                if _features_to_extract[type][feat] is not True:
                    extracted_features[type].pop(feat)

        return extracted_features

    def get_major_minor_field_keys(self):
        # returns top level and specific feature levels
        # exp:  major   ["statistical", "micro-timing"]
        #       minor   ["noi", "lowness", "swing"]
        major_keys = list()
        minor_keys = list()

        for major_key in self.features_to_extract.keys():
            major_keys.append(major_key)
            for minor_key in self.features_to_extract[major_key].keys():
                minor_keys.append(minor_key)

        return list(set(major_keys)), list(set(minor_keys))


class Intraset_Distance_Calculator:
    def __init__(self, feature_dictionary, name=None):

        self.feature_dictionary = feature_dictionary
        self.name = name
        self.__intraset_distances_per_feat = None

    @property
    def intraset_distances_per_feat(self):
        if self.__intraset_distances_per_feat is None:
            self.calculate_distances()
        return self.__intraset_distances_per_feat

    def calculate_distances(self):

        # For each feature distance matrix will be a symmetrical MxM matrix with zeros on diagonal
        for feature, feature_values in zip(self.feature_dictionary.keys(), self.feature_dictionary.values()):
            n_samples = len(feature_values)
            distance_matrix_for_feature = np.zeros((n_samples, n_samples))
            for sample_ix, feature_value in enumerate(feature_values):
                dist_from_val_in_feature_set = np.linalg.norm(feature_values - feature_value)
                distance_matrix_for_feature[sample_ix,:] = dist_from_val_in_feature_set

            if self.__intraset_distances_per_feat is None:
                self.__intraset_distances_per_feat = {feature: distance_matrix_for_feature}
            else:
                self.__intraset_distances_per_feat.update({feature: distance_matrix_for_feature})

        return self.__intraset_distances_per_feat


class Interset_Distance_Calculator:
    def __init__(self, feature_dictionary_a, feature_dictionary_b, name_a=None, name_b=None):

        self.feature_dictionary_a = feature_dictionary_a
        self.name_a = name_a
        self.feature_dictionary_b = feature_dictionary_b
        self.name_b = name_b

        self.__inter_distances_per_feat = None

    @property
    def interset_distances_per_feat(self):
        if self.__inter_distances_per_feat is None:
            self.calculate_distances()
        return self.__inter_distances_per_feat

    def calculate_distances(self):

        # For each feature distance matrix will be a symmetrical MxM matrix with zeros on diagonal
        for feature, feature_values_a in zip(self.feature_dictionary_a.keys(), self.feature_dictionary_a.values()):
            if feature in self.feature_dictionary_b.keys():
                # Get the values in second set corresponding to feature
                feature_values_b = self.feature_dictionary_b[feature]

                # Calculate number of samples in both sets
                n_samples_a = len(feature_values_a)
                n_samples_b = len(feature_values_b)

                # Create an empty distance matrix of shape n_samples_a, n_samples_b
                distance_matrix_for_feature = np.zeros((n_samples_a, n_samples_b))

                # Fill in the distance matrix
                for sample_ix_a, feature_value_a in enumerate(feature_values_a):
                    dist_from_val_in_feature_set = np.linalg.norm(feature_values_b - feature_value_a)
                    distance_matrix_for_feature[sample_ix_a, :] = dist_from_val_in_feature_set

                if self.__inter_distances_per_feat is None:
                    self.__inter_distances_per_feat = {feature: distance_matrix_for_feature}
                else:
                    self.__inter_distances_per_feat.update({feature: distance_matrix_for_feature})

        return self.__inter_distances_per_feat


def convert_distances_dict_to_gaussian_pdfs(distances_dict):
    distances_pdfs = dict()
    X_plot = np.linspace(-5, 10, 1000)
    for feature in distances_dict.keys():
        distances_in_feat = distances_dict[feature]

        if np.count_nonzero(distances_in_feat) > 0:
            # Find kernel bandwidth using Scott's Rule of Thumb
            # https://en.wikipedia.org/wiki/Histogram#Scott's_normal_reference_rule
            bandwidth = 3.49 * distances_in_feat.std() / (distances_in_feat.shape[0]) ** (1.0 / 3.0)
            kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(distances_in_feat)
            log_dens = kde.score_samples(X_plot)


            hist, bin_edges = np.histogram(
                distances_for_feat, bins="scott",
                density=True
            )

def convert_distances_dict_to_pdf_histograms_dict (distances_dict):
    distances_histograms = dict()
    for feature in distances_dict.keys():
        # Find number of bins using Scott's Rule of Thumb
        # https://en.wikipedia.org/wiki/Histogram#Scott's_normal_reference_rule
        distances_for_feat = distances_dict[feature]
        if np.count_nonzero(distances_for_feat) > 0:
            hist, bin_edges = np.histogram(
                distances_for_feat, bins="scott",
                density=True
            )
            distances_histograms.update({feature: (hist, bin_edges)})
    return distances_histograms


class Distance_to_PDF_Converter:
    def __init__(self, intraset_distances_per_features, interset_distances_per_features):
        self.intraset_distances_per_features = intraset_distances_per_features
        self.interset_distances_per_features = interset_distances_per_features

        self.__intraset_pdfs_per_features = None
        self.__interset_pdfs_per_features = None

    @property
    def intraset_pdfs_per_features(self):
        if self.__intraset_pdfs_per_features is None:
            self.calculate_pdfs()
        return self.__intraset_pdfs_per_features

    @property
    def interset_pdfs_per_features(self):
        if self.__interset_pdfs_per_features is None:
            self.calculate_pdfs()
        return self.__interset_pdfs_per_features

    def calculate_pdfs(self):
        # Calculate pdfs for interset distances per each feature
        self.__intraset_pdfs_per_features = convert_distances_dict_to_pdf_histograms_dict(
            self.intraset_distances_per_features)
        self.__interset_pdfs_per_features = convert_distances_dict_to_pdf_histograms_dict(
            self.interset_distances_per_features)
        return self.__intraset_pdfs_per_features, self.__interset_pdfs_per_features

"""

def c_dist(A, B, mode='None', normalize=0):
    c_dist = np.zeros(len(B))
    for i in range(0, len(B)):
        if mode == 'None':
            c_dist[i] = np.linalg.norm(A - B[i])
        elif mode == 'EMD':
            if normalize == 1:
                A_ = sklearn.preprocessing.normalize(A.reshape(1, -1), norm='l1')[0]
                B_ = sklearn.preprocessing.normalize(B[i].reshape(1, -1), norm='l1')[0]
            else:
                A_ = A.reshape(1, -1)[0]
                B_ = B[i].reshape(1, -1)[0]

            c_dist[i] = stats.wasserstein_distance(A_, B_)

        elif mode == 'KL':
            if normalize == 1:
                A_ = sklearn.preprocessing.normalize(A.reshape(1, -1), norm='l1')[0]
                B_ = sklearn.preprocessing.normalize(B[i].reshape(1, -1), norm='l1')[0]
            else:
                A_ = A.reshape(1, -1)[0]
                B_ = B[i].reshape(1, -1)[0]

            B_[B_ == 0] = 0.00000001
            c_dist[i] = stats.entropy(A_, B_)
    return c_dist

"""