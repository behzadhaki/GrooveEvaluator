import sys
sys.path.insert(1, "../../")
sys.path.insert(1, "../")
import wandb

from GrooveEvaluator.evaluator import Evaluator

from bokeh.io import output_file, show, save


pickle_source_path = "../preprocessed_dataset/datasets_extracted_locally/GrooveMidi/hvo_0.4.2/" \
                     "Processed_On_17_05_2021_at_22_32_hrs"

# Create Subset Filters (styles both in train and test)
styles = ["hiphop", "funk", "reggae", "soul", "latin", "jazz", "pop", "afrobeat", "highlife", "punk", "rock"]


list_of_filter_dicts_for_subsets = []
for style in styles:
    list_of_filter_dicts_for_subsets.append(
        {"style_primary": [style], "beat_type": ["beat"], "time_signature": ["4-4"]}
    )

# todo implement distance difference calculator between hvo_sequences
train_set_evaluator = Evaluator(
    pickle_source_path=pickle_source_path, set_subfolder="GrooveMIDI_processed_train",
    hvo_pickle_filename="hvo_sequence_data.obj",
    list_of_filter_dicts_for_subsets=list_of_filter_dicts_for_subsets,
    max_hvo_shape=(32, 27),
    n_samples_to_use=256,
    n_samples_to_synthesize_visualize_per_subset=5,
    disable_tqdm=False,
    analyze_heatmap=True,
    analyze_global_features=True
)


gt_hvos_array = train_set_evaluator.get_ground_truth_hvos_array()
train_set_evaluator.add_predictions(gt_hvos_array)

rhythmic_distances = train_set_evaluator.get_rhythmic_distances()

gt_log_dict, predicted_log_dict = train_set_evaluator.get_wandb_logging_media()

show(gt_log_dict['piano_rolls'])
show(predicted_log_dict['piano_rolls'])

train_set_evaluator.dump()

# Pass to model
# predicted_hvos_array = model.predict(gt_hvos_array)

train_set_evaluator.add_predictions(gt_hvos_array)


tags1, _ = train_set_evaluator.prediction_SubSet_Evaluator.tags_subsets