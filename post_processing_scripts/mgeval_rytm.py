from post_processing_scripts.mgeval_rytm_utils import *


if __name__ == '__main__':

    gmd_eval = pickle.load(open(
            f"post_processing_scripts/evaluators_monotonic_groove_transformer_v1/"
            f"validation_set_evaluator_run_misunderstood-bush-246_Epoch_26.Eval","rb"))

    down_size = 1024
    final_indices = sample_uniformly(gmd_eval, num_samples=down_size) if down_size < 1024 else list(range(1024))

    # Compile data (flatten styles)
    sets_evals = {
        "groovae":
            pickle.load(open(f"post_processing_scripts/evaluators_monotonic_groove_transformer_v1/"
                             f"validation_set_evaluator_run_groovae.Eval", "rb")),
        "rosy":
            pickle.load(open(f"post_processing_scripts/evaluators_monotonic_groove_transformer_v1/"
                             f"validation_set_evaluator_run_rosy-durian-248_Epoch_26.Eval", "rb")),
        "hopeful":
            pickle.load(open(f"post_processing_scripts/evaluators_monotonic_groove_transformer_v1/"
                             f"validation_set_evaluator_run_hopeful-gorge-252_Epoch_90.Eval", "rb")),
        "solar":
            pickle.load(open(f"post_processing_scripts/evaluators_monotonic_groove_transformer_v1/"
                             f"validation_set_evaluator_run_solar-shadow-247_Epoch_41.Eval", "rb")),
        "misunderstood":
            pickle.load(open(f"post_processing_scripts/evaluators_monotonic_groove_transformer_v1/"
                             f"validation_set_evaluator_run_misunderstood-bush-246_Epoch_26.Eval", "rb"))
    }

    # compile and flatten features
    feature_sets = {"gmd": flatten_subset_genres(get_gt_feats_from_evaluator(list(sets_evals.values())[0]))}
    feature_sets.update({
        set_name:flatten_subset_genres(get_pd_feats_from_evaluator(eval)) for (set_name, eval) in sets_evals.items()
    })

    # ----- grab selected indices (samples)
    for set_name, set_dict in feature_sets.items():
        for key, array in set_dict.items():
            feature_sets[set_name][key] = array[final_indices]

    # --- remove unnecessary features
    allowed_analysis = ["Statistical::NoI", "Statistical::Total Step Density", "Statistical::Avg Voice Density",
                        "Statistical::Lowness", "Statistical::Midness", "Statistical::Hiness",
                        "Statistical::Vel Similarity Score", "Statistical::Weak to Strong Ratio",
                        "Syncopation::Lowsync", "Syncopation::Midsync", "Syncopation::Hisync",
                        "Syncopation::Lowsyness", "Syncopation::Midsyness", "Syncopation::Hisyness", "Syncopation::Complexity"]

    for set_name in feature_sets.keys():
        for key in list(feature_sets[set_name].keys()):
            if key not in allowed_analysis:
                feature_sets[set_name].pop(key)

    # ================================================================
    # ---- Analysis 0: Accuracy Vs. Precision
    # from sklearn.metrics import precision_score, accuracy_score
    # ================================================================
    stats_sets = get_positive_negative_hit_stats(sets_evals)
    fig_path = "post_processing_scripts/evaluators_monotonic_groove_transformer_v1/mgeval_results/"
    boxplot_absolute_measures(stats_sets, fs=12, legend_fs=10, legend_ncols=3, fig_path=fig_path, show=True, ncols=4,
                              figsize=(20, 10), color_map="tab20c", filename="Stats.png")



    vel_stats_sets = get_positive_negative_vel_stats(sets_evals)
    boxplot_absolute_measures(vel_stats_sets, fs=12, legend_fs=10, legend_ncols=3, fig_path=fig_path, show=True, ncols=4,
                              figsize=(20, 10), color_map="tab20c", filename="Stats_velocities.png", force_ylim = (-0.2, 1))

    ut_stats_sets = get_positive_negative_utiming_stats(sets_evals)
    boxplot_absolute_measures(ut_stats_sets, fs=12, legend_fs=10, legend_ncols=3, fig_path=fig_path, show=True,
                              ncols=4,
                              figsize=(20, 10), color_map="tab20c", filename="Stats_utimings.png", force_ylim = (-0.7, 0.7))

    # ================================================================
    # ---- Analysis 1: Absolute Measures According to
    # Yang, Li-Chia, and Alexander Lerch. "On the evaluation of generative models in music."
    #           Neural Computing and Applications 32.9 (2020): 4773-4784.
    # ================================================================

    # Compile Absolute Measures
    csv_path = "post_processing_scripts/evaluators_monotonic_groove_transformer_v1/mgeval_results/absolute_measures.csv"
    pd_final = get_absolute_measures_for_multiple_sets(feature_sets, csv_file=csv_path)

    fig_path = "post_processing_scripts/evaluators_monotonic_groove_transformer_v1/mgeval_results/"
    boxplot_absolute_measures(feature_sets, fs=12, legend_fs=8, legend_ncols=2, fig_path=fig_path, show=False, ncols=5,
                              figsize=(18, 10), color_map="tab20c")



    # ================================================================
    # ---- Analysis 2: Comparing the 4 models we made with each other
    # 1.a. Calculate intraset distances of gmd and 4 models
    #   b. Calculate mean and std of each
    # 2.a. Calculate interset distances of each 4 models from gmd
    #   b. for each set, calculate KLD and OLD against gmd
    # 3. Create a table similar to Table 4 in Yang et. al.
    # ================================================================
    # 1.a.


    set_labels = ['gmd', 'groovae', 'misunderstood'] # always put gt on the very left

    gt = feature_sets[set_labels[0]]
    set1 = feature_sets[set_labels[1]]
    set2 = feature_sets[set_labels[2]]




    # Export Analysis to Table
    #csv_path = f"post_processing_scripts/evaluators_monotonic_groove_transformer_v1/mgeval_results/{set_labels[0]}_{set_labels[1]}_{set_labels[2]}/table4_compiled.csv"
    df, raw_data = compare_two_sets_against_ground_truth(gt, set1, set2, set_labels=set_labels, csv_path=csv_path)


    # Generate inter_intra_pdfs feature plots
    fig_path = f"post_processing_scripts/evaluators_monotonic_groove_transformer_v1/mgeval_results/{set_labels[0]}_{set_labels[1]}_{set_labels[2]}"
    plot_inter_intra_pdfs(raw_data, fig_path, set_labels, show=False)

    # Generate per feature plots
    plot_intersets(df, fig_path, set_labels, show=False)

