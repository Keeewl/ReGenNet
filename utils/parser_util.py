from argparse import ArgumentParser
import argparse
import os
import json


def parse_and_load_from_model(parser):
    # args according to the loaded model
    # do not try to specify them from cmd line since they will be overwritten
    add_data_options(parser)
    add_model_options(parser)
    add_diffusion_options(parser)
    args = parser.parse_args()
    args_to_overwrite = []
    for group_name in ['dataset', 'model', 'diffusion']:
        args_to_overwrite += get_args_per_group_name(parser, args, group_name)

    # load args from model
    model_path = get_model_path_from_args()
    args_path = os.path.join(os.path.dirname(model_path), 'args.json')
    assert os.path.exists(args_path), 'Arguments json file was not found!'
    with open(args_path, 'r') as fr:
        model_args = json.load(fr)

    for a in args_to_overwrite:
        if a in model_args.keys():
            setattr(args, a, model_args[a])

        elif 'cond_mode' in model_args: # backward compitability
            unconstrained = (model_args['cond_mode'] == 'no_cond')
            setattr(args, 'unconstrained', unconstrained)

        else:
            print('Warning: was not able to load [{}], using default value [{}] instead.'.format(a, args.__dict__[a]))

    if args.cond_mask_prob == 0:
        args.guidance_param = 1
    return args

def parse_and_load_from_model_wo_data(parser):
    # args according to the loaded model
    # do not try to specify them from cmd line since they will be overwritten
    add_model_options(parser)
    add_diffusion_options(parser)
    args = parser.parse_args()
    args_to_overwrite = []
    for group_name in ['model', 'diffusion']:
        args_to_overwrite += get_args_per_group_name(parser, args, group_name)

    # load args from model
    model_path = get_model_path_from_args()
    args_path = os.path.join(os.path.dirname(model_path), 'args.json')
    assert os.path.exists(args_path), 'Arguments json file was not found!'
    with open(args_path, 'r') as fr:
        model_args = json.load(fr)

    for a in args_to_overwrite:
        if a in model_args.keys():
            setattr(args, a, model_args[a])

        elif 'cond_mode' in model_args: # backward compitability
            unconstrained = (model_args['cond_mode'] == 'no_cond')
            setattr(args, 'unconstrained', unconstrained)

        else:
            print('Warning: was not able to load [{}], using default value [{}] instead.'.format(a, args.__dict__[a]))

    if args.cond_mask_prob == 0:
        args.guidance_param = 1
    return args


def get_args_per_group_name(parser, args, group_name):
    for group in parser._action_groups:
        if group.title == group_name:
            group_dict = {a.dest: getattr(args, a.dest, None) for a in group._group_actions}
            return list(argparse.Namespace(**group_dict).__dict__.keys())
    return ValueError('group_name was not found.')

def get_model_path_from_args():
    try:
        dummy_parser = ArgumentParser()
        dummy_parser.add_argument('model_path')
        dummy_args, _ = dummy_parser.parse_known_args()
        return dummy_args.model_path
    except:
        raise ValueError('model_path argument must be specified.')


def add_base_options(parser):
    group = parser.add_argument_group('base')
    group.add_argument("--cuda", default=True, type=bool, help="Use cuda device, otherwise use CPU.")
    group.add_argument("--device", default=0, type=int, help="Device id to use.")
    group.add_argument("--seed", default=10, type=int, help="For fixing random seed.")
    group.add_argument("--batch_size", default=64, type=int, help="Batch size during training.")
    group.add_argument("--use_ddim", action='store_true',
                       help="Use DDIM to accelerate the inference or not.")
    group.add_argument("--timestep_respacing", default="", type=str, help="ddim timestep respacing.")

def add_online_options(parser):
    group = parser.add_argument_group('online')
    group.add_argument("--reaction_mode", default='offline', choices=['offline', 'online'], type=str,
                       help="Run full-sequence offline or strict-online windowed training/eval.")
    group.add_argument("--online_strategy", default='sliding_window', choices=['sliding_window', 'autoregressive'], type=str,
                       help="Online strategy for generation/training.")
    group.add_argument("--window_size", default=30, type=int,
                       help="Window size for online sliding-window.")
    group.add_argument("--window_stride", default=10, type=int,
                       help="Window stride for online sliding-window.")
    group.add_argument("--window_emit", default='stride', choices=['last', 'stride'], type=str,
                       help="Which portion of window is emitted/supervised.")
    group.add_argument("--window_overlap_handling", default='latest', choices=['latest'], type=str,
                       help="How to resolve overlaps when stitching windows.")
    group.add_argument("--window_pad_mode", default='edge', choices=['edge', 'zero'], type=str,
                       help="Padding mode when window shorter than window_size.")
    group.add_argument("--online_train_random_offset", action='store_true',
                       help="Randomize window start during online training.")
    group.add_argument("--online_eval_use_same_noise", action='store_true',
                       help="Reuse noise seeds across overlapping windows (optional).")


def add_diffusion_options(parser):
    group = parser.add_argument_group('diffusion')
    group.add_argument("--noise_schedule", default='cosine', choices=['linear', 'cosine'], type=str,
                       help="Noise schedule type")
    group.add_argument("--diffusion_steps", default=1000, type=int,
                       help="Number of diffusion steps (denoted T in the paper)")
    group.add_argument("--sigma_small", default=True, type=bool, help="Use smaller sigma values.")


def add_model_options(parser):
    group = parser.add_argument_group('model')
    group.add_argument("--setting", default='mdm', choices=['mdm', 'cmdm', 'cnet', 'cnet_v2', 'cnet_v3', 'cnet_v4', 'cnet_v5'], type=str,
                       help="Training MDM, CMDM, CNet, or CNetV2/V3/V4/V5 framework")
    group.add_argument("--baseline_family", default='regennet', choices=['regennet', 'mdm'], type=str,
                       help="Stage1 baseline family. 'mdm' keeps the actor-conditioned reaction shell but disables ReGenNet explicit interaction losses.")
    group.add_argument("--arch", default='trans_enc',
                       choices=['trans_enc', 'trans_dec', 'gru', 'mlp', 'online', 'offline'], type=str,
                       help="Architecture types as reported in the paper.")
    group.add_argument("--emb_trans_dec", default=False, type=bool,
                       help="For trans_dec architecture only, if true, will inject condition as a class token"
                            " (in addition to cross-attention).")
    group.add_argument("--wo_pos_emb", action='store_true',
                       help="Add positional embedding or not.")
    group.add_argument("--cm_mode", default='concat', # TODO
                       choices=['add', 'concat', 'concat2'], type=str,
                       help="Conditional modeling modes as reported in the paper.")
    group.add_argument("--layers", default=8, type=int,
                       help="Number of layers.")
    group.add_argument("--latent_dim", default=512, type=int,
                       help="Transformer/GRU width.")
    group.add_argument("--cond_mask_prob", default=.1, type=float,
                       help="The probability of masking the condition during training."
                            " For classifier-free guidance learning.")
    group.add_argument("--lambda_rcxyz", default=0.0, type=float, help="Joint positions loss.")
    group.add_argument("--lambda_vel", default=0.0, type=float, help="Joint velocity loss.")
    group.add_argument("--lambda_fc", default=0.0, type=float, help="Foot contact loss.")
    group.add_argument("--lambda_orient", default=1.0, type=float, help="Explicit orientation loss.")
    group.add_argument("--lambda_body", default=1.0, type=float, help="Explicit body pose loss.")
    group.add_argument("--lambda_transl", default=1.0, type=float, help="Explicit root translation loss.")
    group.add_argument("--unconstrained", action='store_true',
                       help="Model is trained unconditionally. That is, it is constrained by neither text nor action. "
                            "Currently intended for unconditional training.")



def add_data_options(parser):
    group = parser.add_argument_group('dataset')
    group.add_argument("--dataset", default='chi3d', choices=['chi3d', 'interx'], type=str,
                       help="Dataset name.")
    group.add_argument("--data_dir", default="", type=str,
                       help="If empty, will use defaults according to the specified dataset.")
    group.add_argument("--num_person", default=1, type=int, help="number of persons")
    group.add_argument("--data_path", default="", type=str, help="Path of the data")
    group.add_argument("--pose_rep", default="rot6d", help="xyz or rotvec etc")
    group.add_argument("--body_model", default='smpl', choices=['smpl', 'smplx'], type=str,
                       help="Use SMPL model or SMPl-X model.")
    group.add_argument("--vel_threshold", default=0.01, type=float, help="Threshold of the velocity.")
    group.add_argument("--shuffle", action='store_true', help="Shuffle the actor-reactor order during training.")

def add_training_options(parser):
    group = parser.add_argument_group('training')
    group.add_argument("--save_dir", required=True, type=str,
                       help="Path to save checkpoints and results.")
    group.add_argument("--overwrite", action='store_true',
                       help="If True, will enable to use an already existing save_dir.")
    group.add_argument("--train_platform_type", default='NoPlatform', choices=['NoPlatform', 'ClearmlPlatform', 'TensorboardPlatform'], type=str,
                       help="Choose platform to log results. NoPlatform means no logging.")
    group.add_argument("--lr", default=1e-4, type=float, help="Learning rate.")
    group.add_argument("--weight_decay", default=0.0, type=float, help="Optimizer weight decay.")
    group.add_argument("--lr_anneal_steps", default=0, type=int, help="Number of learning rate anneal steps.")
    group.add_argument("--eval_batch_size", default=32, type=int,
                       help="Batch size during evaluation loop. Do not change this unless you know what you are doing. "
                            "T2m precision calculation is based on fixed batch size 32.")
    group.add_argument("--eval_split", default='test', choices=['val', 'test'], type=str,
                       help="Which split to evaluate on during training.")
    group.add_argument("--eval_during_training", action='store_true',
                       help="If True, will run evaluation during training.")
    group.add_argument("--eval_rep_times", default=3, type=int,
                       help="Number of repetitions for evaluation loop during training.")
    group.add_argument("--eval_num_samples", default=1_000, type=int,
                       help="If -1, will use all samples in the specified split.")
    group.add_argument("--log_interval", default=1_000, type=int,
                       help="Log losses each N steps")
    group.add_argument("--save_interval", default=10_000, type=int, # 50_000 original
                       help="Save checkpoints and run evaluation each N steps")
    group.add_argument("--num_steps", default=600_000, type=int,
                       help="Training will stop after the specified number of steps.")
    group.add_argument("--num_frames", default=60, type=int,
                       help="Limit for the maximal number of frames.")
    group.add_argument("--resume_checkpoint", default="", type=str,
                       help="If not empty, will start from the specified checkpoint (path to model###.pt file).")


def _legacy_rnet_error(entrypoint):
    raise RuntimeError(
        "RNet Stage2 is archived under legacy/rnet and is not supported in the main CLI. "
        "Use hand-contact Stage2 (contact refiner) instead."
    )


def add_refine_training_options(parser):
    _legacy_rnet_error("add_refine_training_options")


def add_refine_sampling_options(parser):
    _legacy_rnet_error("add_refine_sampling_options")


def add_sampling_options(parser):
    group = parser.add_argument_group('sampling')
    group.add_argument("--model_path", required=True, type=str,
                       help="Path to model####.pt file to be sampled.")
    group.add_argument("--output_dir", default='', type=str,
                       help="Path to results dir (auto created by the script). "
                            "If empty, will create dir in parallel to checkpoint.")
    group.add_argument("--num_samples", default=10, type=int,
                       help="Maximal number of prompts to sample, "
                            "if loading dataset from file, this field will be ignored.")
    group.add_argument("--num_repetitions", default=3, type=int,
                       help="Number of repetitions, per sample (text prompt/action)")
    group.add_argument("--guidance_param", default=2.5, type=float,
                       help="For classifier-free sampling - specifies the s parameter, as defined in the paper.")


def add_generate_options(parser):
    group = parser.add_argument_group('generate')
    group.add_argument("--motion_length", default=60, type=float,
                       help="The length of the sampled motion [in frames]. ")
    group.add_argument(
        "--interaction_order",
        default="",
        type=str,
        help="Optional path to Inter-X interaction_order.pkl for actor/reactor mapping.",
    )
    group.add_argument("--input_text", default='', type=str,
                       help="Path to a text file lists text prompts to be synthesized. If empty, will take text prompts from dataset.")
    group.add_argument("--action_file", default='', type=str,
                       help="Path to a text file that lists names of actions to be synthesized.")
    group.add_argument("--text_prompt", default='', type=str,
                       help="A text prompt to be generated. If empty, will take text prompts from dataset.")
    group.add_argument("--action_name", default='', type=str,
                       help="An action name to be generated. If empty, will take text prompts from dataset.")


def add_evaluation_options(parser):
    group = parser.add_argument_group('eval')
    group.add_argument("--model_path", required=True, type=str,
                       help="Path to model####.pt file to be sampled.")
    group.add_argument("--rec_model_path", required=True, type=str,
                       help="Path to model####.pt of the action recognition model.")
    group.add_argument("--eval_mode", default='debug', type=str, help="Evaluation mode.")
    group.add_argument("--eval_tag", default='', type=str,
                       help="Optional suffix appended to the saved evaluation yaml filename.")
    group.add_argument("--guidance_param", default=2.5, type=float,
                       help="For classifier-free sampling - specifies the s parameter, as defined in the paper.")
    group.add_argument("--auto_regressive", action='store_true',
                       help="Auto-regressive evaluation or not.")


def train_args():
    parser = ArgumentParser()
    add_base_options(parser)
    add_online_options(parser)
    add_data_options(parser)
    add_model_options(parser)
    add_diffusion_options(parser)
    add_training_options(parser)
    return parser.parse_args()


def refine_train_args():
    _legacy_rnet_error("refine_train_args")


def refine_sample_args():
    _legacy_rnet_error("refine_sample_args")


def generate_args():
    parser = ArgumentParser()
    # args specified by the user: (all other will be loaded from the model)
    add_base_options(parser)
    add_online_options(parser)
    add_sampling_options(parser)
    add_generate_options(parser)
    return parse_and_load_from_model(parser)

def cgenerate_args():
    parser = ArgumentParser()
    # args specified by the user: (all other will be loaded from the model)
    add_base_options(parser)
    add_online_options(parser)
    add_data_options(parser)
    add_sampling_options(parser)
    add_generate_options(parser)
    return parse_and_load_from_model_wo_data(parser)


def evaluation_parser():
    parser = ArgumentParser()
    # args specified by the user: (all other will be loaded from the model)
    add_base_options(parser)
    add_online_options(parser)
    add_evaluation_options(parser)
    return parse_and_load_from_model(parser)


def refine_evaluation_parser():
    _legacy_rnet_error("refine_evaluation_parser")


def add_contact_proposal_training_options(parser):
    group = parser.add_argument_group('contact_proposal_training')
    group.add_argument("--cache_path", required=True, type=str,
                       help="Path to coarse cache (.npz or .h5).")
    group.add_argument("--save_dir", required=True, type=str,
                       help="Path to save checkpoints and logs.")
    group.add_argument("--overwrite", action='store_true',
                       help="If True, will enable to use an already existing save_dir.")
    group.add_argument("--num_steps", default=100_000, type=int,
                       help="Training will stop after the specified number of steps.")
    group.add_argument("--log_interval", default=100, type=int,
                       help="Log losses each N steps.")
    group.add_argument("--save_interval", default=2_000, type=int,
                       help="Save checkpoints each N steps.")
    group.add_argument("--lr", default=1e-4, type=float, help="Learning rate.")
    group.add_argument("--weight_decay", default=0.0, type=float, help="Optimizer weight decay.")
    group.add_argument("--resume_checkpoint", default="", type=str,
                       help="If not empty, will start from the specified checkpoint.")
    group.add_argument("--num_workers", default=4, type=int, help="DataLoader workers.")
    group.add_argument("--max_batches", default=-1, type=int,
                       help="Limit the number of batches per epoch (debug).")
    group.add_argument("--train_platform_type", default='NoPlatform',
                       choices=['NoPlatform', 'ClearmlPlatform', 'TensorboardPlatform'],
                       type=str, help="Logging backend.")

    group.add_argument("--body_model", default="smplx", type=str, help="Body model name.")
    group.add_argument("--pose_rep", default="rot6d", type=str, help="Pose representation.")
    group.add_argument("--topk", default=3, type=int, help="Top-k distances for relations.")
    group.add_argument("--sigma", default=0.1, type=float, help="Sigma for soft contact.")
    group.add_argument("--proposal_density", default="small", choices=["small", "medium"], type=str,
                       help="Sparse mesh density for restored-shape proposal features.")
    group.add_argument("--proposal_softmin_beta", default=30.0, type=float,
                       help="Softmin beta for restored-shape mesh proximity proposal features.")
    group.add_argument("--hidden_dim", default=64, type=int, help="Proposal hidden dim.")
    group.add_argument("--num_temporal_blocks", default=2, type=int,
                       help="Number of temporal blocks.")
    group.add_argument("--dropout", default=0.1, type=float, help="Proposal dropout.")

    group.add_argument("--tau_contact", default=0.10, type=float, help="Contact threshold.")
    group.add_argument("--tau_near", default=0.18, type=float, help="Near threshold.")
    group.add_argument("--delta_target", default=0.02, type=float, help="Target hysteresis delta.")
    group.add_argument("--epsilon_move", default=0.01, type=float, help="Motion delta for phase.")
    group.add_argument("--epsilon_hold", default=0.005, type=float, help="Hold delta for phase.")
    group.add_argument("--recent_window", default=3, type=int, help="Recent window for phase.")

    group.add_argument("--lambda_smooth", default=0.1, type=float, help="Smoothness loss weight.")
    group.add_argument("--lambda_consistency", default=0.1, type=float, help="Consistency loss weight.")
    group.add_argument("--use_focal", action='store_true', help="Use focal BCE for active.")
    group.add_argument("--focal_gamma", default=2.0, type=float, help="Focal gamma.")
    group.add_argument("--focal_alpha", default=0.25, type=float, help="Focal alpha.")
    group.add_argument("--log_events", action='store_true',
                       help="Log simple event-level stats.")


def contact_proposal_train_args():
    parser = ArgumentParser()
    add_base_options(parser)
    add_contact_proposal_training_options(parser)
    return parser.parse_args()



def add_contact_refiner_training_options(parser):
    group = parser.add_argument_group('contact_refiner_training')
    group.add_argument("--cache_path", required=True, type=str,
                       help="Path to coarse cache (.npz or .h5).")
    group.add_argument("--save_dir", required=True, type=str,
                       help="Path to save checkpoints and logs.")
    group.add_argument("--overwrite", action='store_true',
                       help="If True, will enable to use an already existing save_dir.")
    group.add_argument("--num_steps", default=50_000, type=int,
                       help="Training will stop after the specified number of steps.")
    group.add_argument("--log_interval", default=100, type=int,
                       help="Log losses each N steps.")
    group.add_argument("--save_interval", default=2_000, type=int,
                       help="Save checkpoints each N steps.")
    group.add_argument("--lr", default=1e-4, type=float, help="Learning rate.")
    group.add_argument("--weight_decay", default=0.0, type=float, help="Optimizer weight decay.")
    group.add_argument("--resume_checkpoint", default="", type=str,
                       help="If not empty, will start from the specified checkpoint.")
    group.add_argument("--num_workers", default=4, type=int, help="DataLoader workers.")
    group.add_argument("--max_batches", default=-1, type=int,
                       help="Limit the number of batches per epoch (debug).")
    group.add_argument("--train_platform_type", default='NoPlatform',
                       choices=['NoPlatform', 'ClearmlPlatform', 'TensorboardPlatform'],
                       type=str, help="Logging backend.")

    group.add_argument("--body_model", default="smplx", type=str, help="Body model name.")
    group.add_argument("--pose_rep", default="rot6d", type=str, help="Pose representation.")
    group.add_argument("--window_size", default=12, type=int, help="Window size for refiner.")
    group.add_argument("--window_pad", default=2, type=int, help="Window padding for refiner.")
    group.add_argument("--include_buffer", action='store_true',
                       help="Include tiny forearm buffer joints (18, 19).")

    group.add_argument("--hidden_dim", default=128, type=int, help="Refiner hidden dim.")
    group.add_argument("--num_temporal_blocks", default=2, type=int, help="Temporal blocks.")
    group.add_argument("--num_cross_blocks", default=2, type=int, help="Cross-attn blocks.")
    group.add_argument("--num_spatial_blocks", default=1, type=int, help="Spatial blocks.")
    group.add_argument("--dropout", default=0.1, type=float, help="Refiner dropout.")
    group.add_argument("--delta_max", default=0.15, type=float, help="Residual bound.")

    group.add_argument("--topk", default=3, type=int, help="Top-k distances for relations.")
    group.add_argument("--sigma", default=0.1, type=float, help="Sigma for soft contact.")

    group.add_argument("--lambda_wrist_res", default=1.0, type=float, help="Wrist residual loss.")
    group.add_argument("--lambda_hand_res", default=1.0, type=float, help="Hand residual loss.")
    group.add_argument("--lambda_contact_align", default=0.5, type=float, help="Contact alignment loss.")
    group.add_argument("--lambda_smooth", default=0.1, type=float, help="Smoothness loss.")
    group.add_argument("--lambda_identity", default=0.1, type=float, help="Identity loss.")
    group.add_argument("--lambda_delta_reg", default=0.01, type=float, help="Delta regularization.")
    group.add_argument("--lambda_buffer", default=0.05, type=float, help="Buffer drift penalty.")

    group.add_argument("--window_source", default="teacher", choices=["teacher", "predicted", "mixed"],
                       type=str, help="Window source mode.")
    group.add_argument("--window_source_debug", default="", choices=["", "teacher", "mix", "predict"],
                       type=str, help="Debug override for window source.")
    group.add_argument("--proposal_checkpoint", default="", type=str,
                       help="Proposal checkpoint for predicted windows.")
    group.add_argument("--proposal_ckpt", default="", type=str,
                       help="Alias for proposal checkpoint.")
    group.add_argument("--active_threshold", default=0.5, type=float,
                       help="Active threshold for predicted windows.")
    group.add_argument("--pred_window_ratio", default=0.5, type=float,
                       help="Ratio of predicted windows in mixed mode.")


    group.add_argument("--teacher_stage_ratio", default=0.3, type=float, help="Teacher stage ratio.")
    group.add_argument("--mix_stage_ratio", default=0.4, type=float, help="Mix stage ratio.")
    group.add_argument("--predict_stage_ratio", default=0.3, type=float, help="Predict stage ratio.")
    group.add_argument("--mix_mode", default="per_sample", choices=["per_sample", "per_batch"],
                       type=str, help="Mix mode for teacher/predict.")
    group.add_argument("--eval_pure_predict_only", default=True, type=bool,
                       help="Eval/test uses pure predict windows.")
    group.add_argument("--log_events", action='store_true',
                       help="Log window stats during training.")
    group.add_argument("--log_contact_metrics", action='store_true',
                       help="Log placeholder contact metrics.")


def contact_refiner_train_args():
    parser = ArgumentParser()
    add_base_options(parser)
    add_contact_refiner_training_options(parser)
    return parser.parse_args()
