import argparse
import os
from argparse import ArgumentDefaultsHelpFormatter
from utils import str2bool


def get_non_default(parsed, parser):
    non_default = {
        opt.dest: getattr(parsed, opt.dest)
        for opt in parser._option_string_actions.values()
        if hasattr(parsed, opt.dest) and opt.default != getattr(parsed, opt.dest)
    }
    return non_default


def default_arg_parser(
    description="", conflict_handler="resolve", parents=[], is_lowest_leaf=False
):
    """
    Generate the default parser - Helper for readability

    Args:
        description (str, optional): name of the parser - usually project name. Defaults to ''.
        conflict_handler (str, optional): wether to raise error on conflict or resolve(take last). Defaults to 'resolve'.
        parents (list, optional): [the name of parent argument managers]. Defaults to [].

    Returns:
        [type]: [description]
    """
    description = (
        parents[0].description + description
        if len(parents) != 0 and parents[0] != None and parents[0].description != None
        else description
    )
    parser = argparse.ArgumentParser(
        description=description,
        add_help=is_lowest_leaf,
        formatter_class=ArgumentDefaultsHelpFormatter,
        conflict_handler=conflict_handler,
        parents=parents,
    )

    return parser


def add_dataset_specific_args(parser, task_name, dataset_name, is_lowest_leaf=False):
        parser.add_argument("--num_points", type=int, default=1024, help="Number of points in point cloud")

        parser.add_argument("--split", default="train", help="Which data to fetch")

        parser.add_argument("--num_neighs", type=int, default=40, help="Num of nearest neighbors to use")
        parser.add_argument("--tosca_all_pairs", nargs="?", default=False, type=str2bool, const=True)
        parser.add_argument("--tosca_cross_pairs", nargs="?", default=False, type=str2bool, const=True)

        if(parser.parse_known_args()[0].dataset_name in ['surreal','smal']):
            parser.set_defaults(limit_train_batches=1000,limit_val_batches=200)
        return parser


def init_parse_argparse_default_params(parser, dataset_name=None):
    TASK_OPTIONS = ["shape_corr", "complition"]

    parser.add_argument(
        "--task_name",
        type=str,
        default="shape_corr",
        choices=TASK_OPTIONS,
        help="The task to solve",
    )
    task_name = parser.parse_known_args()[0].task_name

    ## Dataset and augmentations parameters
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="tosca",
        # choices=DATASET_OPTIONS[task_name],
        help="The dataset to evaluate on",
    )

    parser.add_argument(
        "--test_on_shrec", type=str2bool, nargs="?", const=True, default=False
    )
    parser.add_argument(
        "--test_on_tosca", type=str2bool, nargs="?", const=True, default=False
    )
    parser.add_argument(
        "--test_on_surreal", type=str2bool, nargs="?", const=True, default=False
    )
    parser.add_argument(
        "--rotate_factor",
        type=float,
        default=40,
        help="the rotation factor.",
    )

    parser.add_argument(
        "--scale_factor",
        type=float,
        default=1.0,
        help="the scale factor.",
    )

    parser.add_argument(
        "--noise_factor",
        type=float,
        default=0.001,
        help="the noise factor.",
    )

    parser.add_argument(
        "--crop_factor",
        type=float,
        default=1.0,
        help="the crop factor. If 1. no crop, higher means less crop",
    )

    dataset_name = dataset_name or parser.parse_known_args()[0].dataset_name

    ## General learning parameters
    parser.add_argument(
        "--batch_size",
        type=int,
        default={"complition": 128, "shape_corr": 1}[task_name],
        help="Number of samples in batch",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=1,
        help="Number of samples in train batch",
    )
    parser.add_argument(
        "--val_batch_size", type=int, default=1, help="Number of samples in val batch"
    )
    parser.add_argument(
        "--test_batch_size", type=int, default=1, help="Number of samples in test batch"
    )
    parser.add_argument(
        "--max_epochs",
        default={"complition": 200, "shape_corr": 50}[task_name],
        type=int,
        help="Number of epochs to train",
    )
    parser.add_argument(
        "--lr",
        "--learning_rate",
        type=float,
        default={"complition": 5e-4, "shape_corr": 1e-3}[task_name],
        help="Learning rate",
    )
    parser.add_argument("--optimizer", default="adam", help="Optimizer to use")
    parser.add_argument("--weight_decay", default=5e-3, type=float, help="weight decay")

    ## Input Output parameters
    parser.add_argument(
        "--default_root_dir",
        default=os.path.join(os.getcwd(), "output", task_name),
        help="The path to store this run output",
    )
    parser.add_argument(
        "--show_vis",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="If true logs visualizations (run time)",
    )
    parser.add_argument(
        "--show_recon_vis",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="If true logs visualizations of reconstructions (run time)",
    )
    parser.add_argument(
        "--show_corr_vis",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="If true logs visualizations of pair correspondence (run time)",
    )
    parser.add_argument(
        "--rotate_pc_for_vis",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="If true, rotate point cloud for visualization",
    )
    parser.add_argument(
        "--rotate_pc_angles",
        nargs="+",
        default=[0, 0, 0],
        help="Rotation angle (in degrees) of point cloud for visualization",
    )
    parser.add_argument(
        "--log_html",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="If true logs the html assests as well (storage)",
    )
    parser.add_argument(
        "--write_image",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="If true, write the image assests as well (for offline logger) (storage)",
    )
    parser.add_argument(
        "--write_html",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="If true, write the html assests as well (for offline logger) (storage)",
    )
    parser.add_argument(
        "--vis_mitsuba",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="If true, write mitsuba xml file for rendering",
    )
    parser.add_argument(
        "--mitsuba_debug_mode", type=str2bool, nargs="?", const=True, default=True
    )
    parser.add_argument(
        "--save_data_for_vis",
        nargs="?",
        default=False,
        type=str2bool,
        const=True,
        help="Whether to save data for visualization (storage)",
    )
    parser.add_argument(
        "--vis_idx_list",
        nargs="+",
        default=[],
        help="List of batch indices for visualization (empty list [] for all batches)",
    )
    parser.add_argument(
        "--do_train",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="Train the model",
    )
    parser.add_argument(
        "--predict", type=str2bool, nargs="?", const=True, default=False
    )
    parser.add_argument(
        "--display_id",
        type=int,
        help="For headless mutlithreaded we might want to specify display numbe",
    )
    parser.add_argument(
        "--train_val_split",
        default=0.8,
        type=float,
    )
    parser.add_argument(
        "--test_during_train", type=str2bool, nargs="?", const=True, default=True
    )
    parser.add_argument("--test_during_train_interval", type=int, default=4)

    parser.add_argument(
        "--dropout",
        default={"complition": 0.3, "shape_corr": 0.3}[task_name],
        help="The amount of features to drop",
    )
    parser.add_argument(
        "--latent_dim",
        default={"complition": 64, "shape_corr": 256}[task_name],
        type=int,
        help="The the latent dimention of the bottle-neck feature",
    )
    parser.add_argument(
        "--out_feature_dim",
        default=128,
        type=int,
        help="The the output feature dimention",
    )
    parser.add_argument(
        "--in_features_dim", default=3, help="feature length of input samples"
    )

    ### Auxiliary parameters
    parser.add_argument(
        "--DEBUG_MODE",
        action="store_true",
        help="Important: Set true for single batch per epoch, 1 percent of data(overfit) and log norm",
    )
    parser.add_argument(
        "--offline_logger", type=str2bool, nargs="?", const=True, default=True
    )
    parser.add_argument(
        "--OVERFIT_singel_pair",
        default=None,
        type=str,
        help="Should be set in the format : 8815_9583 And this will be the only pair in the training",
    )

    parser.add_argument("--gpus", default="0", type=str)
    parser.add_argument(
        "--num_data_workers", default=0, type=int, help="for parallel data load"
    )
    parser.add_argument("--config_file", type=str, help="Configuration file yaml file")
    parser.add_argument("--exp_name", type=str, default=None, help="experiment name")
    parser.add_argument("--train_vis_interval", default=400, type=int)
    parser.add_argument("--val_vis_interval", default=50, type=int)
    parser.add_argument("--test_vis_interval", default=50, type=int)
    parser.add_argument("--train_on_limited_data", default=None, type=int)

    parser.add_argument(
        "--flush_logs_every_n_steps",
        default=1,
        type=int,
        help="flush_logs_every_n_steps",
    )
    parser.add_argument(
        "--log_every_n_steps", default=50, type=int, help="log_every_n_steps"
    )
    parser.add_argument(
        "--metric_to_track",
        default="val_tot_loss",
        type=str,
        help="The metric the checkpoint manager will track",
    )
    parser.add_argument(
        "--metric_score_cutoff",
        default=0.05,
        type=float,
        help="The result(min or max) to kill the run if not achieved in K minutes",
    )

    ##misc
    parser.add_argument(
        "--without_logger", action="store_true", help="If true, will not log"
    )
    parser.add_argument("--tag", default="", help="Set a description of the run")

    parser.add_argument(
        "--accumulate_grad_batches",
        type=int,
        default=1,
        help="For sanity check of the network, will randomlly permute the points and check for same results",
    )
