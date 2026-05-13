"""
Test script for BC algorithms. Each test trains a variant of BC
for a handful of gradient steps and tries one rollout with 
the model. Excludes stdout output by default (pass --verbose
to see stdout output).
"""
import argparse
from collections import OrderedDict
from types import SimpleNamespace

import numpy as np
import torch

import robomimic.models.policy_nets as PolicyNets
from robomimic.algo.bc import MixedActionBCMixin
from robomimic.utils.dataset import EventAwareSequenceDataset, action_stats_to_normalization_stats
import robomimic.utils.test_utils as TestUtils
from robomimic.utils.log_utils import silence_stdout
from robomimic.utils.torch_utils import dummy_context_mgr


def get_algo_base_config():
    """
    Base config for testing BC algorithms.
    """

    # config with basic settings for quick training run
    config = TestUtils.get_base_config(algo_name="bc")

    # low-level obs (note that we define it here because @observation structure might vary per algorithm, 
    # for example HBC)
    config.observation.modalities.obs.low_dim = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "object"]
    config.observation.modalities.obs.rgb = []

    # by default, vanilla BC
    config.algo.gaussian.enabled = False
    config.algo.gmm.enabled = False
    config.algo.vae.enabled = False
    config.algo.rnn.enabled = False

    return config


def convert_config_for_images(config):
    """
    Modify config to use image observations.
    """

    # using high-dimensional images - don't load entire dataset into memory, and smaller batch size
    config.train.hdf5_cache_mode = "low_dim"
    config.train.num_data_workers = 0
    config.train.batch_size = 16

    # replace object with rgb modality
    config.observation.modalities.obs.low_dim = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
    config.observation.modalities.obs.rgb = ["agentview_image"]

    # set up visual encoders
    config.observation.encoder.rgb.core_class = "VisualCore"
    config.observation.encoder.rgb.core_kwargs.feature_dimension = 64
    config.observation.encoder.rgb.core_kwargs.backbone_class = 'ResNet18Conv'                         # ResNet backbone for image observations (unused if no image observations)
    config.observation.encoder.rgb.core_kwargs.backbone_kwargs.pretrained = False                # kwargs for visual core
    config.observation.encoder.rgb.core_kwargs.backbone_kwargs.input_coord_conv = False
    config.observation.encoder.rgb.core_kwargs.pool_class = "SpatialSoftmax"                # Alternate options are "SpatialMeanPool" or None (no pooling)
    config.observation.encoder.rgb.core_kwargs.pool_kwargs.num_kp = 32                      # Default arguments for "SpatialSoftmax"
    config.observation.encoder.rgb.core_kwargs.pool_kwargs.learnable_temperature = False    # Default arguments for "SpatialSoftmax"
    config.observation.encoder.rgb.core_kwargs.pool_kwargs.temperature = 1.0                # Default arguments for "SpatialSoftmax"
    config.observation.encoder.rgb.core_kwargs.pool_kwargs.noise_std = 0.0

    # observation randomizer class - set to None to use no randomization, or 'CropRandomizer' to use crop randomization
    config.observation.encoder.rgb.obs_randomizer_class = None

    return config


def test_mixed_binary_actions_keep_identity_normalization():
    action_stats = OrderedDict(
        actions=dict(
            n=4,
            mean=np.array([[40.0, 0.5]], dtype=np.float32),
            sqdiff=np.ones((1, 2), dtype=np.float32),
            min=np.array([[-50.0, 0.0]], dtype=np.float32),
            max=np.array([[130.0, 1.0]], dtype=np.float32),
        )
    )

    dataset = object.__new__(EventAwareSequenceDataset)
    dataset._hdf5_file = None
    dataset.action_config = {"actions": {"normalization": "min_max"}}
    dataset.action_keys = ("actions",)
    dataset.binary_indices = [1]

    identity_indices = dataset.get_action_identity_indices()
    assert identity_indices == {"actions": [1]}

    action_config = {
        "actions": {
            "normalization": "min_max",
            "identity_indices": identity_indices["actions"],
        }
    }
    stats = action_stats_to_normalization_stats(action_stats, action_config)

    np.testing.assert_allclose(stats["actions"]["scale"][0, 1], 1.0)
    np.testing.assert_allclose(stats["actions"]["offset"][0, 1], 0.0)

    raw_actions = np.array([[-50.0, 0.0], [130.0, 1.0]], dtype=np.float32)
    normalized = (raw_actions - stats["actions"]["offset"]) / stats["actions"]["scale"]
    np.testing.assert_allclose(normalized[:, 1], raw_actions[:, 1])
    assert normalized[0, 0] < -0.99
    assert normalized[1, 0] > 0.99


def _event_dataset(actions, mode="change"):
    dataset = object.__new__(EventAwareSequenceDataset)
    dataset.continuous_indices = [0]
    dataset.binary_indices = [1]
    dataset.noop_continuous_raw_values = np.array([0.0], dtype=np.float32)
    dataset.continuous_event_mode = mode
    dataset.continuous_eps = 1e-6
    dataset.continuous_delta_eps = 1.0
    dataset._hdf5_file = None
    dataset._raw_actions_for_demo = lambda _: np.asarray(actions, dtype=np.float32)
    return dataset


def test_event_sampler_change_mode_labels_setpoint_transitions():
    events, binary = _event_dataset([
        [50.0, 1.0],
        [50.0, 1.0],
        [50.5, 1.0],
        [55.0, 1.0],
        [55.0, 0.0],
        [55.0, 0.0],
    ])._traces_for_demo("demo")

    np.testing.assert_array_equal(events, np.array([1, 0, 0, 1, 1, 0], dtype=np.float32))
    np.testing.assert_array_equal(binary[:, 0], np.array([1, 1, 1, 1, 0, 0], dtype=np.float32))


def test_event_sampler_nonzero_mode_keeps_legacy_hold_labels():
    events, _ = _event_dataset([[50.0, 1.0], [50.0, 1.0]], mode="nonzero")._traces_for_demo("demo")
    np.testing.assert_array_equal(events, np.array([1, 1], dtype=np.float32))


def test_event_sampler_change_mode_weights_background_pre_event_and_event():
    dataset = object.__new__(EventAwareSequenceDataset)
    dataset.seq_length = 1
    dataset.event_halo = 2
    dataset.event_mixture = dict(background=0.1, pre_event=0.3, event=0.6)
    dataset.supervision_mode = "first"
    dataset.pad_frame_stack = True
    dataset.n_frame_stack = 1
    dataset.total_num_sequences = 6
    dataset._hdf5_file = None
    dataset._index_to_demo_id = ["demo"] * 6
    dataset._demo_id_to_start_indices = {"demo": 0}
    dataset._demo_id_to_demo_length = {"demo": 6}
    dataset._event_traces = {"demo": np.array([0, 0, 0, 1, 0, 0], dtype=np.float32)}

    buckets, weights = dataset._build_sample_weights()

    np.testing.assert_array_equal(buckets, np.array([0, 1, 1, 2, 0, 0]))
    assert weights[3] > weights[1] > weights[0]


def _mixed_loss_subject():
    subject = object.__new__(MixedActionBCMixin)
    subject._continuous_indices = [0]
    subject._binary_indices = [1]
    subject._action_norm_scale = None
    subject.algo_config = SimpleNamespace(action_head=SimpleNamespace(
        loss_weights=SimpleNamespace(continuous=1.0, binary=1.0)))
    subject.nets = SimpleNamespace(training=True)
    return subject


def test_mixed_action_content_losses_use_sample_weight_directly():
    subject = _mixed_loss_subject()
    batch = dict(
        actions=torch.tensor([[[0.0, 1.0], [10.0, 1.0], [30.0, 0.0]]]),
        event_label=torch.tensor([[[0.0], [1.0], [0.0]]]),
        event_weight=torch.ones(1, 3, 1),
        pad_mask=torch.ones(1, 3, 1),
        binary_labels=torch.tensor([[[1.0], [1.0], [0.0]]]),
    )
    predictions = OrderedDict(continuous=torch.zeros(1, 3, 1), binary_logits=torch.zeros(1, 3, 1))

    losses = subject._compute_losses(predictions, batch)

    torch.testing.assert_close(losses["continuous_loss"], torch.tensor(13.0))
    torch.testing.assert_close(losses["binary_loss"], torch.tensor(np.log(2.0), dtype=torch.float32))
    torch.testing.assert_close(losses["event_rate"], torch.tensor(1.0 / 3.0))
    assert set(losses) == {"continuous_loss", "binary_loss", "event_rate", "action_loss"}


def test_mixed_action_rollout_uses_current_continuous_and_binary_predictions():
    subject = object.__new__(MixedActionBCMixin)
    subject.device = torch.device("cpu")
    subject.ac_dim = 2
    subject._continuous_indices = [0]
    subject._binary_indices = [1]
    subject._action_norm_offset = torch.tensor([50.0, 0.0])
    subject._action_norm_scale = torch.tensor([10.0, 1.0])

    first = subject._mixed_predictions_to_action(OrderedDict(
        continuous=torch.tensor([[0.2]]),
        binary_logits=torch.tensor([[-10.0]]),
    ))
    second = subject._mixed_predictions_to_action(OrderedDict(
        continuous=torch.tensor([[-0.8]]),
        binary_logits=torch.tensor([[10.0]]),
    ))

    torch.testing.assert_close(first, torch.tensor([[0.2, 0.0]]))
    torch.testing.assert_close(second, torch.tensor([[-0.8, 1.0]]))


def test_event_dataset_chunk_targets_mask_post_demo_offsets():
    actions = np.array([
        [0.0, 1.0],
        [10.0, 1.0],
        [20.0, 0.0],
        [30.0, 1.0],
        [40.0, 0.0],
    ], dtype=np.float32)
    dataset = object.__new__(EventAwareSequenceDataset)
    dataset._hdf5_file = None
    dataset.seq_length = 3
    dataset.chunk_horizon = 4
    dataset.action_keys = ("actions",)
    dataset.binary_indices = [1]
    dataset._demo_id_to_demo_length = {"demo": actions.shape[0]}
    dataset._event_traces = {"demo": np.array([0, 1, 0, 1, 0], dtype=np.float32)}
    dataset._binary_traces = {"demo": actions[:, [1]]}
    dataset._raw_actions_for_demo = lambda _: actions
    dataset.get_action_normalization_stats = lambda: OrderedDict(
        actions=dict(
            offset=np.zeros((1, 2), dtype=np.float32),
            scale=np.ones((1, 2), dtype=np.float32),
        )
    )
    meta = {"actions": np.zeros((dataset.seq_length, actions.shape[-1]), dtype=np.float32)}

    dataset._add_chunk_targets(meta, "demo", index_in_demo=2)

    assert meta["action_chunk"].shape == (3, 4, 2)
    assert meta["event_label_chunk"].shape == (3, 4, 1)
    assert meta["event_weight_chunk"].shape == (3, 4, 1)
    assert meta["pad_mask_chunk"].shape == (3, 4, 1)
    assert meta["binary_labels_chunk"].shape == (3, 4, 1)
    np.testing.assert_array_equal(
        meta["pad_mask_chunk"][:, :, 0],
        np.array([[1, 1, 1, 0], [1, 1, 0, 0], [1, 0, 0, 0]], dtype=np.float32),
    )
    np.testing.assert_array_equal(meta["action_chunk"][0, :3], actions[2:5])
    np.testing.assert_array_equal(meta["action_chunk"][0, 3], np.zeros(2, dtype=np.float32))
    np.testing.assert_array_equal(meta["binary_labels_chunk"][0, :3, 0], actions[2:5, 1])


def test_rnn_mixed_actor_network_chunk_output_shapes(monkeypatch):
    net = object.__new__(PolicyNets.RNNMixedActorNetwork)
    net.obs_shapes = OrderedDict(obs=(5,))
    net.continuous_dim = 2
    net.binary_dim = 1
    net.chunk_horizon = 3
    net._is_goal_conditioned = False

    decoder_shapes = net._get_output_shapes()

    def fake_rnn_forward(_self, obs, goal=None, rnn_init_state=None, return_state=False):
        outputs = OrderedDict(
            continuous=torch.zeros(2, 4, 3, 2),
            binary_logits=torch.zeros(2, 4, 3, 1),
        )
        return (outputs, None) if return_state else outputs

    monkeypatch.setattr(PolicyNets.RNN_MIMO_MLP, "forward", fake_rnn_forward)
    predictions = net.forward(OrderedDict(obs=torch.zeros(2, 4, 5)))

    assert decoder_shapes["continuous"] == (3, 2)
    assert decoder_shapes["binary_logits"] == (3, 1)
    assert predictions["continuous"].shape == (2, 4, 3, 2)
    assert predictions["binary_logits"].shape == (2, 4, 3, 1)
    assert set(predictions) == {"continuous", "binary_logits"}


def test_chunked_mixed_action_loss_uses_chunk_pad_mask():
    subject = _mixed_loss_subject()
    subject._chunk_enabled = True
    subject._chunk_horizon = 2
    batch = dict(
        action_chunk=torch.tensor([[[[2.0, 1.0], [100.0, 1.0]], [[4.0, 0.0], [8.0, 1.0]]]]),
        event_label_chunk=torch.ones(1, 2, 2, 1),
        event_weight_chunk=torch.ones(1, 2, 2, 1),
        pad_mask_chunk=torch.tensor([[[[1.0], [0.0]], [[1.0], [1.0]]]]),
        binary_labels_chunk=torch.tensor([[[[1.0], [1.0]], [[0.0], [1.0]]]]),
    )
    predictions = OrderedDict(
        continuous=torch.zeros(1, 2, 2, 1),
        binary_logits=torch.zeros(1, 2, 2, 1),
    )

    losses = subject._compute_losses(predictions, batch)

    expected_continuous = torch.tensor((1.5 + 3.5 + 7.5) / 3.0)
    torch.testing.assert_close(losses["continuous_loss"], expected_continuous)
    assert set(losses) == {"continuous_loss", "binary_loss", "event_rate", "action_loss"}


def test_chunk_temporal_ensembling_and_reset_behavior():
    subject = object.__new__(MixedActionBCMixin)
    subject._chunk_enabled = True
    subject._chunk_horizon = 3
    subject._chunk_temporal_ensemble_enabled = True
    subject._chunk_temporal_ensemble_decay = 0.0
    subject._chunk_prediction_history = []
    subject._chunk_rollout_batch_size = None

    first = OrderedDict(
        continuous=torch.tensor([[[0.0], [10.0], [20.0]]]),
        binary_logits=torch.tensor([[[0.0], [2.0], [4.0]]]),
    )
    second = OrderedDict(
        continuous=torch.tensor([[[100.0], [110.0], [120.0]]]),
        binary_logits=torch.tensor([[[10.0], [12.0], [14.0]]]),
    )

    subject._aggregate_chunk_predictions(first, current_step=0)
    ensembled = subject._aggregate_chunk_predictions(second, current_step=1)

    torch.testing.assert_close(ensembled["continuous"], torch.tensor([[55.0]]))
    torch.testing.assert_close(ensembled["binary_logits"], torch.tensor([[6.0]]))

    subject._reset_chunk_rollout_state()
    assert subject._chunk_prediction_history == []
    assert subject._chunk_rollout_batch_size is None

    batched = OrderedDict(
        continuous=torch.zeros(2, 3, 1),
        binary_logits=torch.zeros(2, 3, 1),
    )
    subject._aggregate_chunk_predictions(batched, current_step=2)
    assert len(subject._chunk_prediction_history) == 1
    assert subject._chunk_rollout_batch_size == 2


def make_image_modifier(config_modifier):
    """
    Turn a config modifier into its image version. Note that
    this explicit function definition is needed for proper
    scoping of @config_modifier.
    """
    return lambda x: config_modifier(convert_config_for_images(x))


# mapping from test name to config modifier functions
MODIFIERS = OrderedDict()
def register_mod(test_name):
    def decorator(config_modifier):
        MODIFIERS[test_name] = config_modifier
    return decorator


@register_mod("bc")
def bc_modifier(config):
    # no-op
    return config


@register_mod("bc-gaussian")
def bc_gaussian_modifier(config):
    config.algo.gaussian.enabled = True
    return config


@register_mod("bc-gmm")
def bc_gmm_modifier(config):
    config.algo.gmm.enabled = True
    return config


@register_mod("bc-vae, N(0, 1) prior")
def bc_vae_modifier_1(config):
    # N(0, 1) prior
    config.algo.vae.enabled = True
    config.algo.vae.prior.learn = False
    config.algo.vae.prior.is_conditioned = False
    return config


@register_mod("bc-vae, Gaussian prior (obs-independent)")
def bc_vae_modifier_2(config):
    # learn parameters of Gaussian prior (obs-independent)
    config.algo.vae.enabled = True
    config.algo.vae.prior.learn = True
    config.algo.vae.prior.is_conditioned = False
    config.algo.vae.prior.use_gmm = False
    config.algo.vae.prior.use_categorical = False
    return config


@register_mod("bc-vae, Gaussian prior (obs-dependent)")
def bc_vae_modifier_3(config):
    # learn parameters of Gaussian prior (obs-dependent)
    config.algo.vae.enabled = True
    config.algo.vae.prior.learn = True
    config.algo.vae.prior.is_conditioned = True
    config.algo.vae.prior.use_gmm = False
    config.algo.vae.prior.use_categorical = False
    return config


@register_mod("bc-vae, GMM prior (obs-independent, weights-fixed)")
def bc_vae_modifier_4(config):
    # learn parameters of GMM prior (obs-independent, weights-fixed)
    config.algo.vae.enabled = True
    config.algo.vae.prior.learn = True
    config.algo.vae.prior.is_conditioned = False
    config.algo.vae.prior.use_gmm = True
    config.algo.vae.prior.gmm_learn_weights = False
    config.algo.vae.prior.use_categorical = False
    return config


@register_mod("bc-vae, GMM prior (obs-independent, weights-learned)")
def bc_vae_modifier_5(config):
    # learn parameters of GMM prior (obs-independent, weights-learned)
    config.algo.vae.enabled = True
    config.algo.vae.prior.learn = True
    config.algo.vae.prior.is_conditioned = False
    config.algo.vae.prior.use_gmm = True
    config.algo.vae.prior.gmm_learn_weights = True
    config.algo.vae.prior.use_categorical = False
    return config


@register_mod("bc-vae, GMM prior (obs-dependent, weights-fixed)")
def bc_vae_modifier_6(config):
    # learn parameters of GMM prior (obs-dependent, weights-fixed)
    config.algo.vae.enabled = True
    config.algo.vae.prior.learn = True
    config.algo.vae.prior.is_conditioned = True
    config.algo.vae.prior.use_gmm = True
    config.algo.vae.prior.gmm_learn_weights = False
    config.algo.vae.prior.use_categorical = False
    return config


@register_mod("bc-vae, GMM prior (obs-dependent, weights-learned)")
def bc_vae_modifier_7(config):
    # learn parameters of GMM prior (obs-dependent, weights-learned)
    config.algo.vae.enabled = True
    config.algo.vae.prior.learn = True
    config.algo.vae.prior.is_conditioned = True
    config.algo.vae.prior.use_gmm = True
    config.algo.vae.prior.gmm_learn_weights = True
    config.algo.vae.prior.use_categorical = False
    return config


@register_mod("bc-vae, uniform categorical prior")
def bc_vae_modifier_8(config):
    # uniform categorical prior
    config.algo.vae.enabled = True
    config.algo.vae.prior.learn = False
    config.algo.vae.prior.is_conditioned = False
    config.algo.vae.prior.use_gmm = False
    config.algo.vae.prior.use_categorical = True
    return config


@register_mod("bc-vae, categorical prior (obs-independent)")
def bc_vae_modifier_9(config):
    # learn parameters of categorical prior (obs-independent)
    config.algo.vae.enabled = True
    config.algo.vae.prior.learn = True
    config.algo.vae.prior.is_conditioned = False
    config.algo.vae.prior.use_gmm = False
    config.algo.vae.prior.use_categorical = True
    return config


@register_mod("bc-vae, categorical prior (obs-dependent)")
def bc_vae_modifier_10(config):
    # learn parameters of categorical prior (obs-dependent)
    config.algo.vae.enabled = True
    config.algo.vae.prior.learn = True
    config.algo.vae.prior.is_conditioned = True
    config.algo.vae.prior.use_gmm = False
    config.algo.vae.prior.use_categorical = True
    return config


@register_mod("bc-rnn")
def bc_rnn_modifier(config):
    config.algo.rnn.enabled = True
    config.algo.rnn.horizon = 10
    config.train.seq_length = 10
    return config


@register_mod("bc-rnn-gmm")
def bc_rnn_gmm_modifier(config):
    config.algo.gmm.enabled = True
    config.algo.rnn.enabled = True
    config.algo.rnn.horizon = 10
    config.train.seq_length = 10
    return config


@register_mod("bc-transformer")
def bc_transformer_modifier(config):
    config.algo.transformer.enabled = True
    config.train.frame_stack = 10
    config.train.seq_length = 1
    return config


@register_mod("bc-transformer-gmm")
def bc_transformer_gmm_modifier(config):
    config.algo.gmm.enabled = True
    config.algo.transformer.enabled = True
    config.train.frame_stack = 10
    config.train.seq_length = 1
    return config


# add image version of all tests
image_modifiers = OrderedDict()
for test_name in MODIFIERS:
    lst = test_name.split("-")
    name = "-".join(lst[:1] + ["rgb"] + lst[1:])
    image_modifiers[name] = make_image_modifier(MODIFIERS[test_name])
MODIFIERS.update(image_modifiers)


# test for image crop randomization
@register_mod("bc-image-crop")
def bc_image_crop_modifier(config):
    config = convert_config_for_images(config)

    # observation randomizer class - using Crop randomizer
    config.observation.encoder.rgb.obs_randomizer_class = "CropRandomizer"

    # kwargs for observation randomizers (for the CropRandomizer, this is size and number of crops)
    config.observation.encoder.rgb.obs_randomizer_kwargs.crop_height = 76
    config.observation.encoder.rgb.obs_randomizer_kwargs.crop_width = 76
    config.observation.encoder.rgb.obs_randomizer_kwargs.num_crops = 1
    config.observation.encoder.rgb.obs_randomizer_kwargs.pos_enc = False
    return config


def test_bc(silence=True):
    for test_name in MODIFIERS:
        context = silence_stdout() if silence else dummy_context_mgr()
        with context:
            base_config = get_algo_base_config()
            res_str = TestUtils.test_run(base_config=base_config, config_modifier=MODIFIERS[test_name])
        print("{}: {}".format(test_name, res_str))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--verbose",
        action='store_true',
        help="don't suppress stdout during tests",
    )
    args = parser.parse_args()

    test_bc(silence=(not args.verbose))
