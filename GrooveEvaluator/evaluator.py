import numpy as np

from GrooveEvaluator.feature_extractor import Feature_Extractor_From_HVO_SubSets
from GrooveEvaluator.plotting_utils import global_features_plotter, velocity_timing_heatmaps_scatter_plotter
from GrooveEvaluator.plotting_utils import separate_figues_by_tabs

"""class HVOSeq_SubSets_Comparator:
    # Eval 1. Test set loss
    #
    # todo 1. From Training Subsets, grab n_samples and pass to model

    def __init__(
            self,
            set_a_subsets,              # Ground Truth typically
            set_a_tags,
            set_b_subsets,              # Predictions typically
            set_b_tags,
            set_a_identifier="Train",
            set_b_identifier="Test",
            analyze_heatmap_a=False,
            analyze_heatmap_b=True,
            analyze_feature_pdfs_a=False,
            analyze_feature_pdfs_b=True,
            max_samples_to_compare = None
            #todo distance measures KL, OVerlap, intra and interset
    ):"""


PATH_DICT_TEMPLATE = {"root_dir": "", "project_name": '', "run_name": '', "eval_data": '', "epoch": ''}


class HVOSeq_SubSet_Evaluator:
    # Eval 1. Test set loss
    #
    # todo 1. From Training Subsets, grab n_samples and pass to model

    def __init__(
            self,
            set_subsets,              # Ground Truth typically
            set_tags,
            set_identifier="Train",
            path_dict=None,
            analyze_heatmap=True,
            analyze_global_features=True,
            n_samples_to_analyze=None,
            synthesize_sequences=True,
            n_samples_to_synthesize=10,
            shuffle_audio_samples=False,
            disable_tqdm=True,
            group_by_minor_keys=True                # if True, plots are grouped per feature/drum voice or,
                                                    # otherwise grouped by style
    ):

        self.__version__ = "0.0.0"

        self.path_dict = PATH_DICT_TEMPLATE if path_dict is None else path_dict

        self.disable_tqdm = disable_tqdm
        self.set_identifier = set_identifier
        self.group_by_minor_keys = group_by_minor_keys

        self.feature_extractor = None   # instantiates in

        self.n_samples_to_analyze = n_samples_to_analyze
        self.analyze_heatmap = analyze_heatmap
        self.analyze_global_features = analyze_global_features
        self.vel_heatmaps_dict = None
        self.vel_scatters_dict = None
        self.vel_heatmaps_bokeh_fig = None
        self.global_features_dict = None
        self.global_features_bokeh_fig = None

        # Store subsets and tags locally, and auto-extract
        self.__set_subsets = None
        self.__set_tags = None
        self.tags_subsets = (set_tags, set_subsets)

        # Flag to re-extract if data changed
        self.__analyze_Flag = True

        # Audio Params
        self.synthesize_sequences = synthesize_sequences
        self. n_samples_to_synthesize = n_samples_to_synthesize
        self.shuffle_audio_samples = shuffle_audio_samples
        self._sampled_hvos = None

    @property
    def tags_subsets(self):
        return self.__set_tags, self.__set_subsets

    @tags_subsets.setter
    def tags_subsets(self, tags_subsets_tuple):
        tags = tags_subsets_tuple[0]
        subsets = tags_subsets_tuple[1]
        assert len(tags) == len(subsets), "Length mismatch between Tags and HVO Subsets : {} Tags vs " \
                                          "{} HVO_Seq Subsets".format(len(tags), len(subsets))
        self.__set_tags = tags
        self.__set_subsets = subsets

        # Reset calculator Flag
        self.__analyze_Flag = True

        # Create a new feature extractor for subsets
        self.feature_extractor = Feature_Extractor_From_HVO_SubSets(
            hvo_subsets=self.__set_subsets,
            tags=self.__set_tags,
            auto_extract=False,
            max_samples_in_subset=self.n_samples_to_analyze,
        )

        if self.analyze_global_features:
            self.feature_extractor.extract(use_tqdm=not self.disable_tqdm)
            self.global_features_dict = self.feature_extractor.get_global_features_dicts(
                regroup_by_feature=self.group_by_minor_keys)

        self.vel_heatmaps_dict, self.vel_scatters_dict = self.feature_extractor.get_velocity_timing_heatmap_dicts(
            s=(4, 10),
            bins=[32*8, 127],
            regroup_by_drum_voice=self.group_by_minor_keys
        ) if self.analyze_heatmap else None

        # todo
        #self.global_features_dict = self.feature_extractor.

    def get_vel_heatmap_bokeh_figures(
            self, plot_width=800, plot_height_per_set=100, legend_fnt_size="8px",
            synchronize_plots=True,
            downsample_heat_maps_by=1
    ):
        p = velocity_timing_heatmaps_scatter_plotter(
            self.vel_heatmaps_dict,
            self.vel_scatters_dict,
            number_of_loops_per_subset_dict=self.feature_extractor.number_of_loops_in_sets,
            number_of_unique_performances_per_subset_dict=self.feature_extractor.number_of_unique_performances_in_sets,
            organized_by_drum_voice=self.group_by_minor_keys,
            title_prefix=self.set_identifier,
            plot_width=plot_width, plot_height_per_set=plot_height_per_set, legend_fnt_size=legend_fnt_size,
            synchronize_plots=synchronize_plots,
            downsample_heat_maps_by=downsample_heat_maps_by
        )
        tabs = separate_figues_by_tabs(p, tab_titles=list(self.vel_heatmaps_dict.keys()))
        return tabs

    def get_global_features_bokeh_figure(self, plot_width=800, plot_height=1200,
                                         legend_fnt_size="8px", resolution=100):
        p = global_features_plotter(
            self.global_features_dict,
            title_prefix=self.set_identifier,
            normalize_data=False,
            analyze_combined_sets=True,
            force_extract=False, plot_width=plot_width, plot_height=plot_height,
            legend_fnt_size=legend_fnt_size,
            scale_y=False, resolution=resolution)
        tabs = separate_figues_by_tabs(p, tab_titles=list(self.global_features_dict.keys()))
        return tabs

    def get_audios(self, sf_paths):
        if self._sampled_hvos is None or self.shuffle_audio_samples:
            self._sampled_hvos = self.feature_extractor.get_few_hvo_samples(self.n_samples_to_synthesize)

        audios = []
        captions = []
        for sample_hvo in self._sampled_hvos:
            # randomly select a sound font
            sf_path = sf_paths[np.random.randint(0, len(sf_paths))]
            audios.append(sample_hvo.synthesize(sf_path=sf_path))
            captions.append("{}_{}_{}.wav".format(
                self.set_identifier, sample_hvo.metadata.style_primary, sample_hvo.metadata.master_id.replace("/","_")
            ))

        return list(zip(captions, audios))

    def get_wandb_logging_dict(self, sf_paths):
        # todo check if some fields are not needed
        heatmap_html = self.get_global_features_bokeh_figure()
        global_features_html = self.get_vel_heatmap_bokeh_figures()
        captions_audios_tuples = self.get_audios(sf_paths)  # todo add to self in case no shuffling is needed
        wandb_dict = {
            "velocity_heatmaps":    heatmap_html,
            "global_features_heatmaps":     global_features_html,
            #"global_features_dict": self.global_features_dict, # todo think about how to log mean/std of these
            "captions_audios_tuples": captions_audios_tuples
        }
        return wandb_dict


