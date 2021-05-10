import numpy as np
from tqdm import tqdm
from sklearn.neighbors import KernelDensity
from scipy.stats import  gaussian_kde
from scipy import stats, integrate
import sklearn

available_features_in_HVO_Sequence = [
    # GrooveToolbox.RhythmFeatures
    # features implemented in Hvo_Sequence.__version__ = 0.1.0
    # --------------------------------------
    "combined_syncopation",
    "polyphonic_syncopation",
    "low_syncopation",
    "mid_syncopation",
    "high_syncopation",
    "low_density",
    "mid_density",
    "high_density",
    "total_density",
    "hiness",
    "midness",
    "lowness",
    "hisyncness",
    "midsyncness",
    "lowsyncness",
    "total_autocorrelation_curve",             # SUPER SLOW!!!!
    "autocorrelation_skew",
    "autocorrelation_max_amplitude",
    "autocorrelation_centroid",
    "autocorrelation_harmonicity",
    "total_symmetry",
    "total_average_intensity",
    "total_weak_to_strong_ratio",
    "total_complexity",

    # GrooveToolbox.MicrotimingFeatures
    # implemented in Hvo_Sequence.__version__ = 0.1.0
    # --------------------------------------
    "swingness",
    "is_swung",
    "laidbackness",
    "timing_accuracy",
]

class Feature_Extractor_From_HVO_Set:

    def __init__(
            self,
            hvo_dataset,
            feature_list_to_extract=[None],            # Extracts all if [None]
            name=None
    ):
        """
        Extracts features for each sample within a list containing hvo_sequences
        (compatible with HVO_Sequence Version >= 0.1.0)

        :param hvo_dataset:                 List of HVO_Sequences
        :param feature_list_to_extract:     List of features (for which extractors are implemented in hvo_sequence)
        :param name:                        Name of dataset
        """

        # Add the dataset to self
        self.hvo_dataset = hvo_dataset

        # If no specific features specified, make sure all features are extracted
        if feature_list_to_extract == [None]:
            self.feature_list_to_extract = available_features_in_HVO_Sequence
        else:
            self.feature_list_to_extract = feature_list_to_extract

        self.__extracted_features = None
        self.name = name

    @property
    def extracted_features(self):
        # If extract() hasn't been called previously, do so to extract features
        # If features extracted, No need to do so anymore
        if self.__extracted_features is None:
            self.extract()
        return self.__extracted_features

    def get_features_for_sample(self, sample_ix):
        sample_hvo = self.hvo_dataset[sample_ix]
        feature_dict = sample_hvo.get_analysis_features(feature_list=self.feature_list_to_extract)
        return feature_dict

    def extract(self, use_tqdm=True):

        # Create an empty dictionary with required fields
        extracted_set = {}
        for feature in self.feature_list_to_extract:
            extracted_set[feature] = np.array([])

        if use_tqdm:
            for ix in tqdm(range(len(self.hvo_dataset)), desc="Extracting Features from HVO_Sequence Set"):
                sample_hvo = self.hvo_dataset[ix]
                feature_dict = sample_hvo.get_analysis_features(feature_list=self.feature_list_to_extract)
                """for feature in self.feature_list_to_extract:
                    extracted_set[feature] = np.append(extracted_set[feature], feature_dict[feature])"""
        else:
            for ix in range(len(self.hvo_dataset)):
                sample_hvo = self.hvo_dataset[ix]
                feature_dict = sample_hvo.get_analysis_features(feature_list=self.feature_list_to_extract)
                for feature in #self.feature_list_to_extract:
                    extracted_set[feature] = np.append(extracted_set[feature], feature_dict[feature])

        self.__extracted_features = extracted_set
        return extracted_set


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
        #print("distances_for_feat", distances_for_feat)
        if np.count_nonzero(distances_for_feat) > 0:
            hist, bin_edges = np.histogram(
                distances_for_feat, bins="scott",
                density=True
            )
            print("mean, std in feature ", feature, np.mean(distances_for_feat), np.std(distances_for_feat))
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

