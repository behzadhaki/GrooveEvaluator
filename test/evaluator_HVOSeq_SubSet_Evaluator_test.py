import sys
sys.path.insert(1, "../../")
sys.path.insert(1, "../")
import wandb
from bokeh.embed import file_html
from bokeh.resources import CDN

from preprocessed_dataset.Subset_Creators import subsetters

from GrooveEvaluator.evaluator import HVOSeq_SubSet_Evaluator

from bokeh.io import output_file, show, save


# Create Subset Filters
styles = ["afrobeat", "afrocuban", "blues", "country", "dance", "funk", "gospel", "highlife", "hiphop", "jazz",
          "latin", "middleeastern", "neworleans", "pop", "punk", "reggae", "rock", "soul"]

list_of_filter_dicts_for_subsets = []
for style in styles:
    list_of_filter_dicts_for_subsets.append({"style_primary": [style], "beat_type": ["beat"], "time_signature": ["4-4"]})

tags_by_style_and_beat, subsets_by_style_and_beat = subsetters.GrooveMidiSubsetter(
    pickle_source_path="../preprocessed_dataset/datasets_extracted_locally/GrooveMidi/hvo_0.4.2/"
                       "Processed_On_17_05_2021_at_22_32_hrs",
    subset="GrooveMIDI_processed_train",
    hvo_pickle_filename="hvo_sequence_data.obj",
    list_of_filter_dicts_for_subsets=list_of_filter_dicts_for_subsets,
    max_len=32,
    ).create_subsets()

train_ground_evaluator = HVOSeq_SubSet_Evaluator (
    set_subsets=subsets_by_style_and_beat,              # Ground Truth typically
    set_tags=tags_by_style_and_beat,
    set_identifier="train_ground_truth",
    analyze_heatmap=True,
    analyze_global_features=True,
    n_samples_to_analyze=20,
    synthesize_sequences=True,
    n_samples_to_synthesize=10,
    shuffle_audio_samples=False,                 # if false, it will reuse the same samples
    disable_tqdm=False,
    group_by_minor_keys=False
)

'''p = train_ground_evaluator.get_global_features_bokeh_figure()
save(p, "misc/{}.html".format("temp_global_features"))

p = train_ground_evaluator.get_vel_heatmap_bokeh_figures()
save(p, "misc/{}.html".format("temp_heatmap"))

captions_audios_tuples = train_ground_evaluator.get_audios(
    sf_paths = ["../hvo_sequence/hvo_sequence/soundfonts/Standard_Drum_Kit.sf2"])'''

wandb_dict = train_ground_evaluator.get_wandb_logging_dict(
    sf_paths=["../hvo_sequence/hvo_sequence/soundfonts/Standard_Drum_Kit.sf2"])

wandb_log = {

    "Velocity_Profile":
        {
            train_ground_evaluator.set_identifier:
                [
                    wandb.Html(file_html(wandb_dict["global_features_heatmaps"], CDN, "my plot"))
                ]
        },
    "global_features_heatmaps":
        {
            train_ground_evaluator.set_identifier:
                [
                    wandb.Html(open("misc/{}.html".format("temp_global_features")))
                ]
        },
    "Audios":
        {
            train_ground_evaluator.set_identifier:
            [
                wandb.Audio(c_a[1], caption=c_a[0], sample_rate=44100) for c_a in wandb_dict["captions_audios_tuples"]
            ]
        }
}

wandb.init(project="GMD Analysis", entity="behzadhaki")
wandb.log(wandb_log)