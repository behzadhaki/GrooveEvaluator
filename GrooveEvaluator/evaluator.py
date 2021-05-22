import numpy as np
import zipfile
import wandb

from GrooveEvaluator.feature_extractor import Feature_Extractor_From_HVO_SubSets
from GrooveEvaluator.plotting_utils import global_features_plotter, velocity_timing_heatmaps_scatter_plotter
from GrooveEvaluator.plotting_utils import separate_figues_by_tabs

from bokeh.embed import file_html
from bokeh.resources import CDN

from preprocessed_dataset.Subset_Creators import subsetters

from copy import deepcopy
import pickle as pk
import os

import sys


class Evaluator:
    # Eval 1. Test set loss
    #
    # todo 1. From Training Subsets, grab n_samples and pass to model

    def __init__(
            self,
            pickle_source_path,
            set_subfolder,
            hvo_pickle_filename,
            list_of_filter_dicts_for_subsets,
            gt_identifier="Train",
            analyze_heatmap_gt=True,
            analyze_heatmap_pred=True,
            analyze_global_features_gt=True,
            analyze_global_features_pred=True,
            max_hvo_shape=(32, 27),
            n_samples_to_use=1024,
            synthesize_sequences=True,
            n_samples_to_synthesize_visualize=10,
            shuffle_samples_for_synthesizing_visualizing=False,
            # todo distance measures KL, Overlap, intra and interset
    ):
        """
        This class will perform a thorough Intra- and Inter- evaluation between ground truth data and predictions

        :param pickle_source_path:          "../preprocessed_dataset/datasets_extracted_locally/GrooveMidi/hvo_0.4.2/"
        :param set_subfolder:               "GrooveMIDI_processed_{train/test/evaluation}"
        :param hvo_pickle_filename:         "hvo_sequence_data.obj"
        :param list_of_filter_dicts_for_subsets
        :param gt_identifier:               Text identifier for set comparison --> Train if dealing with evaluating
                                            predictions of the training set. Test if evaluating performance on test set
        :param analyze_heatmap_gt:          True/False if gt heat maps required (Typically set to False, as this only
                                            needs to be done once, rather than every time - UNLESS, you want to randomly
                                            sample a subset every time
        :param analyze_heatmap_pred:        True/False ---> if you need velocity_profile_heatmaps set to True

        :param analyze_global_features_gt:     True/False ---> Feature PDF plotter for gt --> Same description
                                                            as gt_identifier

        :param analyze_global_features_pred:   True/False
        :param max_hvo_shape:               tuple of (steps, 3*n_drum_voices) --> fits all sequences to this shape
                                                    by trimming or padding them
        :param n_samples_to_use:            number of samples to use for evaluation (uniformly samples n_samples_to_use
                                            from all classes in the ground truth set)
        :param synthesize_sequences:        True/False --> if audio files are needed for evaluation

        """

        gt_subsetter_sampler = subsetters.GrooveMidiSubsetterAndSampler(
            pickle_source_path=pickle_source_path, subset=set_subfolder, hvo_pickle_filename=hvo_pickle_filename,
            list_of_filter_dicts_for_subsets=list_of_filter_dicts_for_subsets,
            number_of_samples=n_samples_to_use,
            max_hvo_shape=max_hvo_shape
        )

        self._gt_identifier = gt_identifier

        self._gt_tags, self._gt_subsets = gt_subsetter_sampler.get_subsets()
        self._gt_hvos_array_tags, self._gt_hvos_array, self._prediction_hvo_seq_templates = \
            gt_subsetter_sampler.get_hvos_array()
        self._prediction_tags, self._prediction_subsets = None, None        # Empty place holder for predictions

        self.gt_SubSet_Evaluator = HVOSeq_SubSet_Evaluator(
            self._gt_subsets,              # Ground Truth typically
            self._gt_tags,
            "Ground Truth _ {} Set".format(self._gt_identifier),             # a name for the subset
            analyze_heatmap=analyze_heatmap_gt,
            analyze_global_features=analyze_global_features_gt,
            synthesize_sequences=synthesize_sequences,
            n_samples_to_synthesize_visualize=n_samples_to_synthesize_visualize,
            shuffle_samples_for_synthesizing_visualizing=shuffle_samples_for_synthesizing_visualizing,
            disable_tqdm=True,
            group_by_minor_keys=True  )

    def get_ground_truth_hvos_array(self):
        return self._gt_hvos_array

    def receive_prediction_hvos_array(self, prediction_hvos_array):
        self.prediction_tags, self.prediction_subsets = \
            subsetters.convert_hvos_array_to_subsets(
                self._gt_hvos_array_tags,
                prediction_hvos_array,
                self._prediction_hvo_seq_templates
            )

    def dump(self, path=None, auto_zip=True):          # todo implement in comparator
        if path is None:
            path = os.path.join("misc", self._gt_identifier)
        if not os.path.exists(path):
            os.makedirs(path)

        fname = os.path.join(path, "evaluator.Eval")
        f = open(fname, "wb")
        pk.dump(self, f)

        if auto_zip is True:
            zipObj = zipfile.ZipFile(fname.replace(".Eval", ".zip"), 'w')
            zipObj.write(fname)
            zipObj.close()
            os.remove(fname)


PATH_DICT_TEMPLATE = {
    "root_dir": "",                 # ROOT_DIR to save data
    "project_name": '',             # GROOVE_TRANSFORMER_INFILL or GROOVE_TRANSFORMER_TAP2DRUM
    "run_name": '',                 # WANDB RUN NAME run = wandb.init(...
    "set_identifier": '',  # TRAIN OR TEST
    "epoch": '',
}


class HVOSeq_SubSet_Evaluator (object):
    # Eval 1. Test set loss
    #
    # todo 1. From Training Subsets, grab n_samples and pass to model

    def __init__(
            self,
            set_subsets,              # Ground Truth typically
            set_tags,
            set_identifier,             # a name for the subset
            analyze_heatmap=True,
            analyze_global_features=True,
            n_samples_to_analyze=None,
            synthesize_sequences=True,
            n_samples_to_synthesize_visualize=10,
            shuffle_samples_for_synthesizing_visualizing=False,
            disable_tqdm=True,
            group_by_minor_keys=True                # if True, plots are grouped per feature/drum voice or,
                                                    # otherwise grouped by style
    ):

        self.__version__ = "0.0.0"

        self.set_identifier = set_identifier

        self.disable_tqdm = disable_tqdm
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
        self. n_samples_to_synthesize_visualize = n_samples_to_synthesize_visualize
        self.shuffle_samples_to_synthesize_visualize = shuffle_samples_for_synthesizing_visualizing
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
            self, plot_width=1200, plot_height_per_set=100, legend_fnt_size="8px",
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
        if self._sampled_hvos is None or self.shuffle_samples_to_synthesize_visualize:
            self._sampled_hvos = self.feature_extractor.get_few_hvo_samples(self.n_samples_to_synthesize_visualize)

        if not isinstance(sf_paths, list):
            sf_paths = [sf_paths]

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

    def get_piano_rolls(self):
        if self._sampled_hvos is None or self.shuffle_samples_to_synthesize_visualize:
            self._sampled_hvos = self.feature_extractor.get_few_hvo_samples(self.n_samples_to_synthesize_visualize)

        piano_rolls = []
        titles = []
        for sample_hvo in self._sampled_hvos:
            # randomly select a sound font
            titles.append("{}_{}_{}".format(
                self.set_identifier, sample_hvo.metadata.style_primary, sample_hvo.metadata.master_id.replace("/", "_")
            ))
            piano_rolls.append(sample_hvo.to_html_plot(filename=titles[-1]))
        return separate_figues_by_tabs(piano_rolls, [str(x) for x in range(len(piano_rolls))])

    def get_features_statistics_dict(self):
        wandb_features_data = {}
        for major_key in self.global_features_dict.keys():
            for minor_key in self.global_features_dict[major_key].keys():
                feature_data = self.global_features_dict[major_key][minor_key]
                main_key = "{}.{}.".format(major_key.split('_AND_')[0].replace("['", ''),
                                           minor_key.split('_AND_')[0].replace("']", ''))
                wandb_features_data.update(
                    {
                        main_key + "mean": feature_data.mean(), main_key + "std": feature_data.std(),
                        main_key + "median": np.percentile(feature_data, 50),
                        main_key + "q1": np.percentile(feature_data, 25),
                        main_key + "q3": np.percentile(feature_data, 75)}
                )
        return wandb_features_data

    def get_logging_dict(self, velocity_heatmap_html=True, global_features_html=True,
                         piano_roll_html=True, audio_files=True, sf_paths=None):
        if audio_files is True:
            assert sf_paths is not None, "Provide sound_file path(s) for synthesizing samples"

        logging_dict = {}
        if velocity_heatmap_html is True:
            logging_dict.update({"velocity_heatmaps": self.get_vel_heatmap_bokeh_figures()})
        if global_features_html is True:
            logging_dict.update({"global_feature_pdfs": self.get_global_features_bokeh_figure()})
        if audio_files is True:
            captions_audios_tuples = self.get_audios(sf_paths)  # todo add to self in case no shuffling is needed
            captions_audios = [(c_a[0], c_a[1]) for c_a in captions_audios_tuples]
            logging_dict.update({"captions_audios": captions_audios})
        if piano_roll_html is True:
            logging_dict.update({"piano_rolls": self.get_piano_rolls()})

        return logging_dict

    def get_wandb_logging_media(self, velocity_heatmap_html=True, global_features_html=True,
                                piano_roll_html=True, audio_files=True, sf_paths=None):

        logging_dict = self.get_logging_dict(velocity_heatmap_html, global_features_html,
                                             piano_roll_html, audio_files, sf_paths)

        wandb_media_dict = {}
        for key in logging_dict.keys():
            if velocity_heatmap_html is True:
                wandb_media_dict.update(
                    {
                        "{}_velocity_heatmaps".format(self.set_identifier): wandb.Html(file_html(
                            logging_dict["velocity_heatmaps"], CDN, "vel_heatmap_"+self.set_identifier))
                    }
                )

            if global_features_html is True:
                wandb_media_dict.update(
                    {
                        "{}_global_feature_pdfs".format(self.set_identifier): wandb.Html(file_html(
                            logging_dict["global_feature_pdfs"], CDN, "feature_pdfs_"+self.set_identifier))
                    }
                )

            if audio_files is True:
                captions_audios_tuples = logging_dict["captions_audios"]
                wandb_media_dict.update(
                    {
                        "{}_audios".format(self.set_identifier): [
                            wandb.Audio(c_a[1], caption=c_a[0], sample_rate=44100) for c_a in captions_audios_tuples
                        ]
                    }
                )

            if piano_roll_html is True:
                wandb_media_dict.update(
                    {
                        "{}_piano_roll_html".format(self.set_identifier): wandb.Html(file_html(
                            logging_dict["piano_rolls"], CDN, "piano_rolls_"+self.set_identifier))
                    }
                )

        return wandb_media_dict

    def dump(self, path=None, auto_zip=True):          # todo implement in comparator
        if path is None:
            path = os.path.join("misc", self.set_identifier)
        if not os.path.exists(path):
            os.makedirs(path)

        fname = os.path.join(path, "subset_evaluator.SubEval")
        f = open(fname, "wb")
        pk.dump(self, f)

        if auto_zip is True:
            zipObj = zipfile.ZipFile(fname.replace(".SubEval", ".zip"), 'w')
            zipObj.write(fname)
            zipObj.close()
            os.remove(fname)
