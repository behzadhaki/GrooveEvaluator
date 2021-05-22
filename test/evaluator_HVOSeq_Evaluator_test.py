import sys
sys.path.insert(1, "../../")
sys.path.insert(1, "../")
import wandb


from preprocessed_dataset.Subset_Creators import subsetters
# , Set_Sampler, convert_hvos_array_to_subsets
from GrooveEvaluator.evaluator import HVOSeq_SubSet_Evaluator, Evaluator

from bokeh.io import output_file, show, save


pickle_source_path = "../preprocessed_dataset/datasets_extracted_locally/GrooveMidi/hvo_0.4.2/" \
                     "Processed_On_17_05_2021_at_22_32_hrs"

# Create Subset Filters
styles = ["afrobeat", "afrocuban", "blues", "country", "dance", "funk", "gospel", "highlife", "hiphop", "jazz",
          "latin", "middleeastern", "neworleans", "pop", "punk", "reggae", "rock", "soul"]

list_of_filter_dicts_for_subsets = []
for style in styles:
    list_of_filter_dicts_for_subsets.append(
        {"style_primary": [style], "beat_type": ["beat"], "time_signature": ["4-4"]}
    )


train_set_evaluator = Evaluator(
    pickle_source_path=pickle_source_path, set_subfolder="GrooveMIDI_processed_train",
    hvo_pickle_filename="hvo_sequence_data.obj",
    list_of_filter_dicts_for_subsets=list_of_filter_dicts_for_subsets,
    max_hvo_shape=(32, 27),
    n_samples_to_use=1024
)

gt_hvos_array = train_set_evaluator.get_ground_truth_hvos_array()
train_set_evaluator.dump()

train_set_evaluator.receive_prediction_hvos_array(gt_hvos_array)



