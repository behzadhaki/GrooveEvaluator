import numpy as np
import pandas as pd
from GrooveEvaluator import evaluator  # import your version of evaluator!!
import pickle

def get_pd_feats_from_evaluator(evaluator_):
    # extracts the prediction features from a evaluator
    return evaluator_.prediction_SubSet_Evaluator.feature_extractor.get_global_features_dicts(True)


def get_gt_feats_from_evaluator(evaluator_):
    # extracts the ground truth features from a evaluator
    return evaluator_.gt_SubSet_Evaluator.feature_extractor.get_global_features_dicts(True)


def flatten_subset_genres(feature_dict):
    # combines the subset samples irregardless of their genre
    flattened_feature_dict = {x: np.array([]) for x in feature_dict.keys()}
    for feature_key in flattened_feature_dict.keys():
        for subset_key, subset_samples in feature_dict[feature_key].items():
            flattened_feature_dict[feature_key] = np.append(flattened_feature_dict[feature_key], subset_samples)
    return flattened_feature_dict


def get_absolute_measures_for_single_set(flat_feature_dict, csv_file=None):
    # Gets absolute measures of a set according to
    # Yang, Li-Chia, and Alexander Lerch. "On the evaluation of generative models in music."
    #           Neural Computing and Applications 32.9 (2020): 4773-4784.

    stats = []  # list of lists stats[i] corresponds to [mean, std, min, max, median, q1, q3]
    labels = []

    for key in flat_feature_dict.keys():
        data = flat_feature_dict[key]
        # Calc stats
        stats.append(
            [np.mean(data), np.std(data), np.min(data), np.max(data), np.percentile(data, 50), np.percentile(data, 25),
             np.percentile(data, 75)])
        labels.append(key)

    df2 = pd.DataFrame(np.array(stats).transpose(),
                       ["mean", "std", "min", "max", "median", "q1", "q3"],
                       labels).transpose()

    if csv_file is not None:
        df2.to_csv(csv_file)

    return df2


def get_absolute_measures_for_multiple_sets(sets_of_flat_feature_dict, csv_file=None):

    sets_dfs = []
    sets_df_keys = []
    for set_tag, set_feat_dict in sets_of_flat_feature_dict.items():
        sets_df_keys.append(set_tag)
        sets_dfs.append(get_absolute_measures_for_single_set(set_feat_dict))
        print(f"--------- Finished Calculating Absolute Measures for set {set_tag} --------------")

    pd_final = pd.concat(sets_dfs, keys=sets_df_keys)

    if csv_file is not None:
        pd_final.to_csv(csv_file)

    return pd_final

if __name__ == '__main__':

    # Compile data (flatten styles)
    sets = {
        "gmd": flatten_subset_genres(get_gt_feats_from_evaluator(pickle.load(open(
            f"post_processing_scripts/evaluators_monotonic_groove_transformer_v1/"
            f"validation_set_evaluator_run_misunderstood-bush-246_Epoch_26.Eval","rb")))),
        "hopeful": flatten_subset_genres(get_pd_feats_from_evaluator(pickle.load(open(
            f"post_processing_scripts/evaluators_monotonic_groove_transformer_v1/"
            f"validation_set_evaluator_run_hopeful-gorge-252_Epoch_90.Eval","rb")))),
        "misunderstood": flatten_subset_genres(get_pd_feats_from_evaluator(pickle.load(open(
            f"post_processing_scripts/evaluators_monotonic_groove_transformer_v1/"
            f"validation_set_evaluator_run_misunderstood-bush-246_Epoch_26.Eval", "rb")))),
        "rosy": flatten_subset_genres(get_pd_feats_from_evaluator(pickle.load(open(
            f"post_processing_scripts/evaluators_monotonic_groove_transformer_v1/"
            f"validation_set_evaluator_run_rosy-durian-248_Epoch_26.Eval", "rb")))),
        "solar": flatten_subset_genres(get_pd_feats_from_evaluator(pickle.load(open(
            f"post_processing_scripts/evaluators_monotonic_groove_transformer_v1/"
            f"validation_set_evaluator_run_solar-shadow-247_Epoch_41.Eval", "rb")))),
        "groovae": flatten_subset_genres(get_pd_feats_from_evaluator(pickle.load(open(
            f"post_processing_scripts/evaluators_monotonic_groove_transformer_v1/"
            f"validation_set_evaluator_run_groovae.Eval", "rb")))),
    }

    # ================================================================
    # ---- Absolute Measures According to
    # Yang, Li-Chia, and Alexander Lerch. "On the evaluation of generative models in music."
    #           Neural Computing and Applications 32.9 (2020): 4773-4784.
    # ================================================================

    # Compile Absolute Measures
    csv_path = "post_processing_scripts/evaluators_monotonic_groove_transformer_v1/mgeval_results/absolute_measures.csv"
    get_absolute_measures_for_multiple_sets(sets, csv_file=csv_path)

    # ================================================================
    # Intra_set Calculations according to
    # Yang, Li-Chia, and Alexander Lerch. "On the evaluation of generative models in music."
    #           Neural Computing and Applications 32.9 (2020): 4773-4784.
    # ================================================================


    #sets["gmd"]