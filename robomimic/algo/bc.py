"""
Implementation of Behavioral Cloning (BC).
"""
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D

import robomimic.models.base_nets as BaseNets
import robomimic.models.obs_nets as ObsNets
import robomimic.models.policy_nets as PolicyNets
import robomimic.models.vae_nets as VAENets
import robomimic.utils.loss_utils as LossUtils
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.obs_utils as ObsUtils

from robomimic.algo import register_algo_factory_func, PolicyAlgo


@register_algo_factory_func("bc")
def algo_config_to_class(algo_config):
    """
    Maps algo config to the BC algo class to instantiate, along with additional algo kwargs.

    Args:
        algo_config (Config instance): algo config

    Returns:
        algo_class: subclass of Algo
        algo_kwargs (dict): dictionary of additional kwargs to pass to algorithm
    """

    # note: we need the check below because some configs import BCConfig and exclude
    # some of these options
    gaussian_enabled = ("gaussian" in algo_config and algo_config.gaussian.enabled)
    gmm_enabled = ("gmm" in algo_config and algo_config.gmm.enabled)
    vae_enabled = ("vae" in algo_config and algo_config.vae.enabled)
    action_head_type = algo_config.action_head.type if ("action_head" in algo_config) else "continuous"
    mixed_action_enabled = (action_head_type == "mixed")

    rnn_enabled = algo_config.rnn.enabled
    # support legacy configs that do not have "transformer" item
    transformer_enabled = ("transformer" in algo_config) and algo_config.transformer.enabled

    if mixed_action_enabled:
        if gaussian_enabled or gmm_enabled or vae_enabled or transformer_enabled:
            raise NotImplementedError("action_head.type='mixed' is only implemented for deterministic BC and BC-RNN")
        if rnn_enabled:
            algo_class, algo_kwargs = BC_RNN_MixedAction, {}
        else:
            algo_class, algo_kwargs = BC_MixedAction, {}
    elif action_head_type != "continuous":
        raise ValueError("unsupported BC action_head.type '{}'".format(action_head_type))
    elif gaussian_enabled:
        if rnn_enabled:
            raise NotImplementedError
        elif transformer_enabled:
            raise NotImplementedError
        else:
            algo_class, algo_kwargs = BC_Gaussian, {}
    elif gmm_enabled:
        if rnn_enabled:
            algo_class, algo_kwargs = BC_RNN_GMM, {}
        elif transformer_enabled:
            algo_class, algo_kwargs = BC_Transformer_GMM, {}
        else:
            algo_class, algo_kwargs = BC_GMM, {}
    elif vae_enabled:
        if rnn_enabled:
            raise NotImplementedError
        elif transformer_enabled:
            raise NotImplementedError
        else:
            algo_class, algo_kwargs = BC_VAE, {}
    else:
        if rnn_enabled:
            algo_class, algo_kwargs = BC_RNN, {}
        elif transformer_enabled:
            algo_class, algo_kwargs = BC_Transformer, {}
        else:
            algo_class, algo_kwargs = BC, {}

    return algo_class, algo_kwargs


class BC(PolicyAlgo):
    """
    Normal BC training.
    """
    def _create_networks(self):
        """
        Creates networks and places them into @self.nets.
        """
        self.nets = nn.ModuleDict()
        self.nets["policy"] = PolicyNets.ActorNetwork(
            obs_shapes=self.obs_shapes,
            goal_shapes=self.goal_shapes,
            ac_dim=self.ac_dim,
            mlp_layer_dims=self.algo_config.actor_layer_dims,
            encoder_kwargs=ObsUtils.obs_encoder_kwargs_from_config(self.obs_config.encoder),
        )
        self.nets = self.nets.float().to(self.device)

    def process_batch_for_training(self, batch):
        """
        Processes input batch from a data loader to filter out
        relevant information and prepare the batch for training.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader

        Returns:
            input_batch (dict): processed and filtered batch that
                will be used for training 
        """
        input_batch = dict()
        input_batch["obs"] = {k: batch["obs"][k][:, 0, :] for k in batch["obs"]}
        input_batch["goal_obs"] = batch.get("goal_obs", None) # goals may not be present
        input_batch["actions"] = batch["actions"][:, 0, :]
        # we move to device first before float conversion because image observation modalities will be uint8 -
        # this minimizes the amount of data transferred to GPU
        return TensorUtils.to_float(TensorUtils.to_device(input_batch, self.device))


    def train_on_batch(self, batch, epoch, validate=False):
        """
        Training on a single batch of data.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

            epoch (int): epoch number - required by some Algos that need
                to perform staged training and early stopping

            validate (bool): if True, don't perform any learning updates.

        Returns:
            info (dict): dictionary of relevant inputs, outputs, and losses
                that might be relevant for logging
        """
        with TorchUtils.maybe_no_grad(no_grad=validate):
            info = super(BC, self).train_on_batch(batch, epoch, validate=validate)
            predictions = self._forward_training(batch)
            losses = self._compute_losses(predictions, batch)

            info["predictions"] = TensorUtils.detach(predictions)
            info["losses"] = TensorUtils.detach(losses)

            if not validate:
                step_info = self._train_step(losses)
                info.update(step_info)

        return info

    def _forward_training(self, batch):
        """
        Internal helper function for BC algo class. Compute forward pass
        and return network outputs in @predictions dict.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

        Returns:
            predictions (dict): dictionary containing network outputs
        """
        predictions = OrderedDict()
        actions = self.nets["policy"](obs_dict=batch["obs"], goal_dict=batch["goal_obs"])
        predictions["actions"] = actions
        return predictions

    def _compute_losses(self, predictions, batch):
        """
        Internal helper function for BC algo class. Compute losses based on
        network outputs in @predictions dict, using reference labels in @batch.

        Args:
            predictions (dict): dictionary containing network outputs, from @_forward_training
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

        Returns:
            losses (dict): dictionary of losses computed over the batch
        """
        losses = OrderedDict()
        a_target = batch["actions"]
        actions = predictions["actions"]
        losses["l2_loss"] = nn.MSELoss()(actions, a_target)
        losses["l1_loss"] = nn.SmoothL1Loss()(actions, a_target)
        # cosine direction loss on eef delta position
        losses["cos_loss"] = LossUtils.cosine_loss(actions[..., :3], a_target[..., :3])

        action_losses = [
            self.algo_config.loss.l2_weight * losses["l2_loss"],
            self.algo_config.loss.l1_weight * losses["l1_loss"],
            self.algo_config.loss.cos_weight * losses["cos_loss"],
        ]
        action_loss = sum(action_losses)
        losses["action_loss"] = action_loss
        return losses

    def _train_step(self, losses):
        """
        Internal helper function for BC algo class. Perform backpropagation on the
        loss tensors in @losses to update networks.

        Args:
            losses (dict): dictionary of losses computed over the batch, from @_compute_losses
        """

        # gradient step
        info = OrderedDict()
        policy_grad_norms = TorchUtils.backprop_for_loss(
            net=self.nets["policy"],
            optim=self.optimizers["policy"],
            loss=losses["action_loss"],
            max_grad_norm=self.global_config.train.max_grad_norm,
        )
        info["policy_grad_norms"] = policy_grad_norms
        return info

    def log_info(self, info):
        """
        Process info dictionary from @train_on_batch to summarize
        information to pass to tensorboard for logging.

        Args:
            info (dict): dictionary of info

        Returns:
            loss_log (dict): name -> summary statistic
        """
        log = super(BC, self).log_info(info)
        log["Loss"] = info["losses"]["action_loss"].item()
        if "l2_loss" in info["losses"]:
            log["L2_Loss"] = info["losses"]["l2_loss"].item()
        if "l1_loss" in info["losses"]:
            log["L1_Loss"] = info["losses"]["l1_loss"].item()
        if "cos_loss" in info["losses"]:
            log["Cosine_Loss"] = info["losses"]["cos_loss"].item()
        if "policy_grad_norms" in info:
            log["Policy_Grad_Norms"] = info["policy_grad_norms"]
        return log

    def get_action(self, obs_dict, goal_dict=None):
        """
        Get policy action outputs.

        Args:
            obs_dict (dict): current observation
            goal_dict (dict): (optional) goal

        Returns:
            action (torch.Tensor): action tensor
        """
        assert not self.nets.training
        return self.nets["policy"](obs_dict, goal_dict=goal_dict)


class MixedActionBCMixin(object):
    """
    Shared helpers for deterministic BC variants with mixed continuous and binary
    action heads.
    """
    def _setup_mixed_action_config(self):
        cfg = self.algo_config.action_head
        self._continuous_indices = [int(i) for i in cfg.continuous_indices]
        self._binary_indices = [int(i) for i in cfg.binary_indices]
        self._gate_enabled = bool(cfg.gate.enabled)
        self._gate_threshold = float(cfg.gate.threshold)
        self._binary_mode = str(cfg.noop.binary_mode)

        if cfg.type != "mixed":
            raise ValueError("Mixed action BC requires algo.action_head.type='mixed'")
        if self._binary_mode != "repeat_last":
            raise NotImplementedError("only action_head.noop.binary_mode='repeat_last' is supported")

        all_indices = self._continuous_indices + self._binary_indices
        if len(all_indices) != len(set(all_indices)):
            raise ValueError("mixed action continuous_indices and binary_indices must not overlap")
        if sorted(all_indices) != list(range(self.ac_dim)):
            raise ValueError(
                "mixed action indices must cover every action dimension exactly once; "
                "got continuous_indices={} binary_indices={} for ac_dim={}".format(
                    self._continuous_indices, self._binary_indices, self.ac_dim)
            )

        self._noop_continuous_raw_values = torch.as_tensor(
            list(cfg.noop.continuous_raw_values), dtype=torch.float32, device=self.device)
        if self._noop_continuous_raw_values.numel() != len(self._continuous_indices):
            raise ValueError("action_head.noop.continuous_raw_values must match continuous_indices length")

        self._initial_binary_values = torch.as_tensor(
            list(cfg.noop.initial_binary_values), dtype=torch.float32, device=self.device)
        if self._initial_binary_values.numel() != len(self._binary_indices):
            raise ValueError("action_head.noop.initial_binary_values must match binary_indices length")

        self._last_raw_binary = None
        self._action_norm_offset = None
        self._action_norm_scale = None

    def set_action_normalization_stats(self, action_normalization_stats):
        action_key = self.global_config.train.action_keys[0]
        stats = action_normalization_stats[action_key]
        self._action_norm_offset = torch.as_tensor(
            stats["offset"], dtype=torch.float32, device=self.device).reshape(-1)
        self._action_norm_scale = torch.as_tensor(
            stats["scale"], dtype=torch.float32, device=self.device).reshape(-1)

    def _add_mixed_action_batch_fields(self, input_batch, batch, take_first_step):
        if take_first_step:
            input_batch["event_label"] = batch["event_label"][:, 0, :]
            input_batch["event_weight"] = batch["event_weight"][:, 0, :]
            input_batch["pad_mask"] = batch["pad_mask"][:, 0, :]
            input_batch["binary_labels"] = batch["binary_labels"][:, 0, :]
        else:
            input_batch["event_label"] = batch["event_label"]
            input_batch["event_weight"] = batch["event_weight"]
            input_batch["pad_mask"] = batch["pad_mask"]
            input_batch["binary_labels"] = batch["binary_labels"]
        return input_batch

    def _forward_training(self, batch):
        return self.nets["policy"](obs_dict=batch["obs"], goal_dict=batch["goal_obs"])

    def _weighted_mean(self, loss, weight):
        weight = weight.to(dtype=loss.dtype)
        return (loss * weight).sum() / weight.sum().clamp(min=1e-6)

    def _compute_losses(self, predictions, batch):
        losses = OrderedDict()

        event_label = batch["event_label"].squeeze(-1)
        event_weight = batch["event_weight"].squeeze(-1)
        pad_mask = batch["pad_mask"].squeeze(-1).to(dtype=event_label.dtype)
        sample_weight = event_weight * pad_mask
        value_mask = sample_weight * event_label if self._gate_enabled else sample_weight

        continuous_target = batch["actions"][..., self._continuous_indices]
        binary_target = batch["binary_labels"]

        if self._gate_enabled:
            gate_loss = F.binary_cross_entropy_with_logits(
                predictions["gate_logits"], event_label, reduction="none")
            losses["gate_loss"] = self._weighted_mean(gate_loss, sample_weight)
            losses["gate_positive_rate"] = torch.sigmoid(predictions["gate_logits"]).mean()

            if not self.nets.training:
                valid_mask = pad_mask > 0
                with torch.no_grad():
                    pred_positive = torch.sigmoid(predictions["gate_logits"]) > self._gate_threshold
                    true_positive_label = event_label > 0.5
                    tp = (pred_positive & true_positive_label & valid_mask).sum().to(dtype=torch.float32)
                    fp = (pred_positive & ~true_positive_label & valid_mask).sum().to(dtype=torch.float32)
                    fn = (~pred_positive & true_positive_label & valid_mask).sum().to(dtype=torch.float32)
                    pred_count = tp + fp
                    true_count = tp + fn
                    valid_count = valid_mask.sum().to(dtype=torch.float32)
                    precision = tp / pred_count.clamp(min=1.0)
                    recall = tp / true_count.clamp(min=1.0)
                    f1 = (2.0 * precision * recall) / (precision + recall).clamp(min=1e-6)
                    losses["gate_precision"] = precision
                    losses["gate_recall"] = recall
                    losses["gate_f1"] = f1
                    losses["gate_true_rate"] = true_count / valid_count.clamp(min=1.0)
                    losses["gate_pred_rate"] = pred_count / valid_count.clamp(min=1.0)
                    losses["gate_tp"] = tp
                    losses["gate_fp"] = fp
                    losses["gate_fn"] = fn
                    losses["gate_pred_count"] = pred_count
                    losses["gate_true_count"] = true_count
                    losses["gate_valid_count"] = valid_count
        else:
            losses["gate_loss"] = sample_weight.sum() * 0.0
            losses["gate_positive_rate"] = sample_weight.sum() * 0.0
            if not self.nets.training:
                zero = sample_weight.sum() * 0.0
                losses["gate_precision"] = zero
                losses["gate_recall"] = zero
                losses["gate_f1"] = zero
                losses["gate_true_rate"] = zero
                losses["gate_pred_rate"] = zero
                losses["gate_tp"] = zero
                losses["gate_fp"] = zero
                losses["gate_fn"] = zero
                losses["gate_pred_count"] = zero
                losses["gate_true_count"] = zero
                losses["gate_valid_count"] = zero

        continuous_loss = F.smooth_l1_loss(
            predictions["continuous"], continuous_target, reduction="none")
        binary_loss = F.binary_cross_entropy_with_logits(
            predictions["binary_logits"], binary_target, reduction="none")

        losses["continuous_loss"] = self._weighted_mean(continuous_loss, value_mask.unsqueeze(-1))
        losses["binary_loss"] = self._weighted_mean(binary_loss, value_mask.unsqueeze(-1))
        losses["event_rate"] = self._weighted_mean(event_label, pad_mask)

        loss_weights = self.algo_config.action_head.loss_weights
        losses["action_loss"] = (
            loss_weights.gate * losses["gate_loss"] +
            loss_weights.continuous * losses["continuous_loss"] +
            loss_weights.binary * losses["binary_loss"]
        )
        return losses

    def _normalize_raw_dims(self, dims, raw_values):
        if self._action_norm_offset is None or self._action_norm_scale is None:
            return raw_values
        dims = torch.as_tensor(dims, dtype=torch.long, device=self.device)
        return (raw_values - self._action_norm_offset[dims]) / self._action_norm_scale[dims]

    def _ensure_mixed_rollout_state(self, batch_size):
        if len(self._binary_indices) == 0:
            return
        if self._last_raw_binary is None or self._last_raw_binary.shape[0] != batch_size:
            self._last_raw_binary = self._initial_binary_values.unsqueeze(0).expand(batch_size, -1).clone()

    def _mixed_predictions_to_action(self, predictions):
        batch_size = predictions["continuous"].shape[0]
        self._ensure_mixed_rollout_state(batch_size=batch_size)

        if self._gate_enabled:
            should_act = torch.sigmoid(predictions["gate_logits"]) > self._gate_threshold
        else:
            should_act = torch.ones((batch_size,), dtype=torch.bool, device=self.device)

        actions = torch.zeros((batch_size, self.ac_dim), dtype=torch.float32, device=self.device)

        if len(self._continuous_indices) > 0:
            noop_continuous = self._noop_continuous_raw_values.unsqueeze(0).expand(batch_size, -1)
            normalized_noop = self._normalize_raw_dims(self._continuous_indices, noop_continuous)
            actions[:, self._continuous_indices] = torch.where(
                should_act.unsqueeze(-1),
                predictions["continuous"],
                normalized_noop,
            )

        if len(self._binary_indices) > 0:
            predicted_raw_binary = (torch.sigmoid(predictions["binary_logits"]) > 0.5).to(dtype=torch.float32)
            next_raw_binary = torch.where(
                should_act.unsqueeze(-1),
                predicted_raw_binary,
                self._last_raw_binary,
            )
            actions[:, self._binary_indices] = self._normalize_raw_dims(self._binary_indices, next_raw_binary)
            self._last_raw_binary = next_raw_binary.detach()

        return actions

    def get_action(self, obs_dict, goal_dict=None):
        assert not self.nets.training
        predictions = self.nets["policy"](obs_dict, goal_dict=goal_dict)
        return self._mixed_predictions_to_action(predictions)

    def reset(self):
        self._last_raw_binary = None

    def log_info(self, info):
        log = PolicyAlgo.log_info(self, info)
        log["Loss"] = info["losses"]["action_loss"].item()
        log["Gate_Loss"] = info["losses"]["gate_loss"].item()
        log["Continuous_Loss"] = info["losses"]["continuous_loss"].item()
        log["Binary_Loss"] = info["losses"]["binary_loss"].item()
        log["Event_Rate"] = info["losses"]["event_rate"].item()
        log["Gate_Positive_Rate"] = info["losses"]["gate_positive_rate"].item()
        if "gate_true_rate" in info["losses"]:
            log["Gate_True_Rate"] = info["losses"]["gate_true_rate"].item()
        if "gate_pred_rate" in info["losses"]:
            log["Gate_Pred_Rate"] = info["losses"]["gate_pred_rate"].item()
        if "gate_precision" in info["losses"]:
            log["Gate_Precision"] = info["losses"]["gate_precision"].item()
        if "gate_recall" in info["losses"]:
            log["Gate_Recall"] = info["losses"]["gate_recall"].item()
        if "gate_f1" in info["losses"]:
            log["Gate_F1"] = info["losses"]["gate_f1"].item()
        for loss_key, log_key in (
            ("gate_tp", "Gate_TP"),
            ("gate_fp", "Gate_FP"),
            ("gate_fn", "Gate_FN"),
            ("gate_pred_count", "Gate_Pred_Count"),
            ("gate_true_count", "Gate_True_Count"),
            ("gate_valid_count", "Gate_Valid_Count"),
        ):
            if loss_key in info["losses"]:
                log[log_key] = info["losses"][loss_key].item()
        if "policy_grad_norms" in info:
            log["Policy_Grad_Norms"] = info["policy_grad_norms"]
        return log


class BC_MixedAction(MixedActionBCMixin, BC):
    """
    Deterministic BC with separate continuous, binary, and optional gate heads.
    """
    def _create_networks(self):
        self._setup_mixed_action_config()
        self.nets = nn.ModuleDict()
        self.nets["policy"] = PolicyNets.MixedActorNetwork(
            obs_shapes=self.obs_shapes,
            goal_shapes=self.goal_shapes,
            ac_dim=self.ac_dim,
            mlp_layer_dims=self.algo_config.actor_layer_dims,
            continuous_dim=len(self._continuous_indices),
            binary_dim=len(self._binary_indices),
            gate_enabled=self._gate_enabled,
            encoder_kwargs=ObsUtils.obs_encoder_kwargs_from_config(self.obs_config.encoder),
        )
        self.nets = self.nets.float().to(self.device)

    def process_batch_for_training(self, batch):
        input_batch = super(BC_MixedAction, self).process_batch_for_training(batch)
        input_batch = self._add_mixed_action_batch_fields(input_batch, batch, take_first_step=True)
        return TensorUtils.to_float(TensorUtils.to_device(input_batch, self.device))


class BC_Gaussian(BC):
    """
    BC training with a Gaussian policy.
    """
    def _create_networks(self):
        """
        Creates networks and places them into @self.nets.
        """
        assert self.algo_config.gaussian.enabled

        self.nets = nn.ModuleDict()
        self.nets["policy"] = PolicyNets.GaussianActorNetwork(
            obs_shapes=self.obs_shapes,
            goal_shapes=self.goal_shapes,
            ac_dim=self.ac_dim,
            mlp_layer_dims=self.algo_config.actor_layer_dims,
            fixed_std=self.algo_config.gaussian.fixed_std,
            init_std=self.algo_config.gaussian.init_std,
            std_limits=(self.algo_config.gaussian.min_std, 7.5),
            std_activation=self.algo_config.gaussian.std_activation,
            low_noise_eval=self.algo_config.gaussian.low_noise_eval,
            encoder_kwargs=ObsUtils.obs_encoder_kwargs_from_config(self.obs_config.encoder),
        )

        self.nets = self.nets.float().to(self.device)

    def _forward_training(self, batch):
        """
        Internal helper function for BC algo class. Compute forward pass
        and return network outputs in @predictions dict.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

        Returns:
            predictions (dict): dictionary containing network outputs
        """
        dists = self.nets["policy"].forward_train(
            obs_dict=batch["obs"], 
            goal_dict=batch["goal_obs"],
        )

        # make sure that this is a batch of multivariate action distributions, so that
        # the log probability computation will be correct
        assert len(dists.batch_shape) == 1
        log_probs = dists.log_prob(batch["actions"])

        predictions = OrderedDict(
            log_probs=log_probs,
        )
        return predictions

    def _compute_losses(self, predictions, batch):
        """
        Internal helper function for BC algo class. Compute losses based on
        network outputs in @predictions dict, using reference labels in @batch.

        Args:
            predictions (dict): dictionary containing network outputs, from @_forward_training
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

        Returns:
            losses (dict): dictionary of losses computed over the batch
        """

        # loss is just negative log-likelihood of action targets
        action_loss = -predictions["log_probs"].mean()
        return OrderedDict(
            log_probs=-action_loss,
            action_loss=action_loss,
        )

    def log_info(self, info):
        """
        Process info dictionary from @train_on_batch to summarize
        information to pass to tensorboard for logging.

        Args:
            info (dict): dictionary of info

        Returns:
            loss_log (dict): name -> summary statistic
        """
        log = PolicyAlgo.log_info(self, info)
        log["Loss"] = info["losses"]["action_loss"].item()
        log["Log_Likelihood"] = info["losses"]["log_probs"].item() 
        if "policy_grad_norms" in info:
            log["Policy_Grad_Norms"] = info["policy_grad_norms"]
        return log


class BC_GMM(BC_Gaussian):
    """
    BC training with a Gaussian Mixture Model policy.
    """
    def _create_networks(self):
        """
        Creates networks and places them into @self.nets.
        """
        assert self.algo_config.gmm.enabled

        self.nets = nn.ModuleDict()
        self.nets["policy"] = PolicyNets.GMMActorNetwork(
            obs_shapes=self.obs_shapes,
            goal_shapes=self.goal_shapes,
            ac_dim=self.ac_dim,
            mlp_layer_dims=self.algo_config.actor_layer_dims,
            num_modes=self.algo_config.gmm.num_modes,
            min_std=self.algo_config.gmm.min_std,
            std_activation=self.algo_config.gmm.std_activation,
            low_noise_eval=self.algo_config.gmm.low_noise_eval,
            encoder_kwargs=ObsUtils.obs_encoder_kwargs_from_config(self.obs_config.encoder),
        )

        self.nets = self.nets.float().to(self.device)


class BC_VAE(BC):
    """
    BC training with a VAE policy.
    """
    def _create_networks(self):
        """
        Creates networks and places them into @self.nets.
        """
        self.nets = nn.ModuleDict()
        self.nets["policy"] = PolicyNets.VAEActor(
            obs_shapes=self.obs_shapes,
            goal_shapes=self.goal_shapes,
            ac_dim=self.ac_dim,
            device=self.device,
            encoder_kwargs=ObsUtils.obs_encoder_kwargs_from_config(self.obs_config.encoder),
            **VAENets.vae_args_from_config(self.algo_config.vae),
        )
        
        self.nets = self.nets.float().to(self.device)

    def train_on_batch(self, batch, epoch, validate=False):
        """
        Update from superclass to set categorical temperature, for categorical VAEs.
        """
        if self.algo_config.vae.prior.use_categorical:
            temperature = self.algo_config.vae.prior.categorical_init_temp - epoch * self.algo_config.vae.prior.categorical_temp_anneal_step
            temperature = max(temperature, self.algo_config.vae.prior.categorical_min_temp)
            self.nets["policy"].set_gumbel_temperature(temperature)
        return super(BC_VAE, self).train_on_batch(batch, epoch, validate=validate)

    def _forward_training(self, batch):
        """
        Internal helper function for BC algo class. Compute forward pass
        and return network outputs in @predictions dict.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

        Returns:
            predictions (dict): dictionary containing network outputs
        """
        vae_inputs = dict(
            actions=batch["actions"],
            obs_dict=batch["obs"],
            goal_dict=batch["goal_obs"],
            freeze_encoder=batch.get("freeze_encoder", False),
        )

        vae_outputs = self.nets["policy"].forward_train(**vae_inputs)
        predictions = OrderedDict(
            actions=vae_outputs["decoder_outputs"],
            kl_loss=vae_outputs["kl_loss"],
            reconstruction_loss=vae_outputs["reconstruction_loss"],
            encoder_z=vae_outputs["encoder_z"],
        )
        if not self.algo_config.vae.prior.use_categorical:
            with torch.no_grad():
                encoder_variance = torch.exp(vae_outputs["encoder_params"]["logvar"])
            predictions["encoder_variance"] = encoder_variance
        return predictions

    def _compute_losses(self, predictions, batch):
        """
        Internal helper function for BC algo class. Compute losses based on
        network outputs in @predictions dict, using reference labels in @batch.

        Args:
            predictions (dict): dictionary containing network outputs, from @_forward_training
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

        Returns:
            losses (dict): dictionary of losses computed over the batch
        """

        # total loss is sum of reconstruction and KL, weighted by beta
        kl_loss = predictions["kl_loss"]
        recons_loss = predictions["reconstruction_loss"]
        action_loss = recons_loss + self.algo_config.vae.kl_weight * kl_loss
        return OrderedDict(
            recons_loss=recons_loss,
            kl_loss=kl_loss,
            action_loss=action_loss,
        )

    def log_info(self, info):
        """
        Process info dictionary from @train_on_batch to summarize
        information to pass to tensorboard for logging.

        Args:
            info (dict): dictionary of info

        Returns:
            loss_log (dict): name -> summary statistic
        """
        log = PolicyAlgo.log_info(self, info)
        log["Loss"] = info["losses"]["action_loss"].item()
        log["KL_Loss"] = info["losses"]["kl_loss"].item()
        log["Reconstruction_Loss"] = info["losses"]["recons_loss"].item()
        if self.algo_config.vae.prior.use_categorical:
            log["Gumbel_Temperature"] = self.nets["policy"].get_gumbel_temperature()
        else:
            log["Encoder_Variance"] = info["predictions"]["encoder_variance"].mean().item()
        if "policy_grad_norms" in info:
            log["Policy_Grad_Norms"] = info["policy_grad_norms"]
        return log


class BC_RNN(BC):
    """
    BC training with an RNN policy.
    """
    def _create_networks(self):
        """
        Creates networks and places them into @self.nets.
        """
        self.nets = nn.ModuleDict()
        self.nets["policy"] = PolicyNets.RNNActorNetwork(
            obs_shapes=self.obs_shapes,
            goal_shapes=self.goal_shapes,
            ac_dim=self.ac_dim,
            mlp_layer_dims=self.algo_config.actor_layer_dims,
            encoder_kwargs=ObsUtils.obs_encoder_kwargs_from_config(self.obs_config.encoder),
            **BaseNets.rnn_args_from_config(self.algo_config.rnn),
        )

        self._rnn_hidden_state = None
        self._rnn_horizon = self.algo_config.rnn.horizon
        self._rnn_counter = 0
        self._rnn_is_open_loop = self.algo_config.rnn.get("open_loop", False)

        self.nets = self.nets.float().to(self.device)

    def process_batch_for_training(self, batch):
        """
        Processes input batch from a data loader to filter out
        relevant information and prepare the batch for training.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader

        Returns:
            input_batch (dict): processed and filtered batch that
                will be used for training
        """
        input_batch = dict()
        input_batch["obs"] = batch["obs"]
        input_batch["goal_obs"] = batch.get("goal_obs", None) # goals may not be present
        input_batch["actions"] = batch["actions"]

        if self._rnn_is_open_loop:
            # replace the observation sequence with one that only consists of the first observation.
            # This way, all actions are predicted "open-loop" after the first observation, based
            # on the rnn hidden state.
            n_steps = batch["actions"].shape[1]
            obs_seq_start = TensorUtils.index_at_time(batch["obs"], ind=0)
            input_batch["obs"] = TensorUtils.unsqueeze_expand_at(obs_seq_start, size=n_steps, dim=1)

        # we move to device first before float conversion because image observation modalities will be uint8 -
        # this minimizes the amount of data transferred to GPU
        return TensorUtils.to_float(TensorUtils.to_device(input_batch, self.device))

    def get_action(self, obs_dict, goal_dict=None):
        """
        Get policy action outputs.

        Args:
            obs_dict (dict): current observation
            goal_dict (dict): (optional) goal

        Returns:
            action (torch.Tensor): action tensor
        """
        assert not self.nets.training

        if self._rnn_hidden_state is None or self._rnn_counter % self._rnn_horizon == 0:
            batch_size = list(obs_dict.values())[0].shape[0]
            self._rnn_hidden_state = self.nets["policy"].get_rnn_init_state(batch_size=batch_size, device=self.device)

            if self._rnn_is_open_loop:
                # remember the initial observation, and use it instead of the current observation
                # for open-loop action sequence prediction
                self._open_loop_obs = TensorUtils.clone(TensorUtils.detach(obs_dict))

        obs_to_use = obs_dict
        if self._rnn_is_open_loop:
            # replace current obs with last recorded obs
            obs_to_use = self._open_loop_obs

        self._rnn_counter += 1
        action, self._rnn_hidden_state = self.nets["policy"].forward_step(
            obs_to_use, goal_dict=goal_dict, rnn_state=self._rnn_hidden_state)
        return action

    def reset(self):
        """
        Reset algo state to prepare for environment rollouts.
        """
        self._rnn_hidden_state = None
        self._rnn_counter = 0


class BC_RNN_MixedAction(MixedActionBCMixin, BC_RNN):
    """
    BC-RNN with separate continuous, binary, and optional gate heads.
    """
    def _create_networks(self):
        self._setup_mixed_action_config()
        self.nets = nn.ModuleDict()
        self.nets["policy"] = PolicyNets.RNNMixedActorNetwork(
            obs_shapes=self.obs_shapes,
            goal_shapes=self.goal_shapes,
            ac_dim=self.ac_dim,
            mlp_layer_dims=self.algo_config.actor_layer_dims,
            continuous_dim=len(self._continuous_indices),
            binary_dim=len(self._binary_indices),
            gate_enabled=self._gate_enabled,
            encoder_kwargs=ObsUtils.obs_encoder_kwargs_from_config(self.obs_config.encoder),
            **BaseNets.rnn_args_from_config(self.algo_config.rnn),
        )

        self._rnn_hidden_state = None
        self._rnn_horizon = self.algo_config.rnn.horizon
        self._rnn_counter = 0
        self._rnn_is_open_loop = self.algo_config.rnn.get("open_loop", False)
        self._open_loop_obs = None

        self.nets = self.nets.float().to(self.device)

    def process_batch_for_training(self, batch):
        input_batch = super(BC_RNN_MixedAction, self).process_batch_for_training(batch)
        input_batch = self._add_mixed_action_batch_fields(input_batch, batch, take_first_step=False)
        return TensorUtils.to_float(TensorUtils.to_device(input_batch, self.device))

    def get_action(self, obs_dict, goal_dict=None):
        assert not self.nets.training

        if self._rnn_hidden_state is None or self._rnn_counter % self._rnn_horizon == 0:
            batch_size = list(obs_dict.values())[0].shape[0]
            self._rnn_hidden_state = self.nets["policy"].get_rnn_init_state(batch_size=batch_size, device=self.device)
            self._ensure_mixed_rollout_state(batch_size=batch_size)

            if self._rnn_is_open_loop:
                self._open_loop_obs = TensorUtils.clone(TensorUtils.detach(obs_dict))

        obs_to_use = obs_dict
        if self._rnn_is_open_loop:
            obs_to_use = self._open_loop_obs

        self._rnn_counter += 1
        predictions, self._rnn_hidden_state = self.nets["policy"].forward_step(
            obs_to_use, goal_dict=goal_dict, rnn_state=self._rnn_hidden_state)

        return self._mixed_predictions_to_action(predictions)

    def reset(self):
        BC_RNN.reset(self)
        self._last_raw_binary = None


class BC_RNN_GMM(BC_RNN):
    """
    BC training with an RNN GMM policy.
    """
    def _create_networks(self):
        """
        Creates networks and places them into @self.nets.
        """
        assert self.algo_config.gmm.enabled
        assert self.algo_config.rnn.enabled

        self.nets = nn.ModuleDict()
        self.nets["policy"] = PolicyNets.RNNGMMActorNetwork(
            obs_shapes=self.obs_shapes,
            goal_shapes=self.goal_shapes,
            ac_dim=self.ac_dim,
            mlp_layer_dims=self.algo_config.actor_layer_dims,
            num_modes=self.algo_config.gmm.num_modes,
            min_std=self.algo_config.gmm.min_std,
            std_activation=self.algo_config.gmm.std_activation,
            low_noise_eval=self.algo_config.gmm.low_noise_eval,
            encoder_kwargs=ObsUtils.obs_encoder_kwargs_from_config(self.obs_config.encoder),
            **BaseNets.rnn_args_from_config(self.algo_config.rnn),
        )

        self._rnn_hidden_state = None
        self._rnn_horizon = self.algo_config.rnn.horizon
        self._rnn_counter = 0
        self._rnn_is_open_loop = self.algo_config.rnn.get("open_loop", False)

        self.nets = self.nets.float().to(self.device)

    def _forward_training(self, batch):
        """
        Internal helper function for BC algo class. Compute forward pass
        and return network outputs in @predictions dict.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

        Returns:
            predictions (dict): dictionary containing network outputs
        """
        dists = self.nets["policy"].forward_train(
            obs_dict=batch["obs"], 
            goal_dict=batch["goal_obs"],
        )

        # make sure that this is a batch of multivariate action distributions, so that
        # the log probability computation will be correct
        assert len(dists.batch_shape) == 2 # [B, T]
        log_probs = dists.log_prob(batch["actions"])

        predictions = OrderedDict(
            log_probs=log_probs,
        )
        return predictions

    def _compute_losses(self, predictions, batch):
        """
        Internal helper function for BC algo class. Compute losses based on
        network outputs in @predictions dict, using reference labels in @batch.

        Args:
            predictions (dict): dictionary containing network outputs, from @_forward_training
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

        Returns:
            losses (dict): dictionary of losses computed over the batch
        """

        # loss is just negative log-likelihood of action targets
        action_loss = -predictions["log_probs"].mean()
        return OrderedDict(
            log_probs=-action_loss,
            action_loss=action_loss,
        )

    def log_info(self, info):
        """
        Process info dictionary from @train_on_batch to summarize
        information to pass to tensorboard for logging.

        Args:
            info (dict): dictionary of info

        Returns:
            loss_log (dict): name -> summary statistic
        """
        log = PolicyAlgo.log_info(self, info)
        log["Loss"] = info["losses"]["action_loss"].item()
        log["Log_Likelihood"] = info["losses"]["log_probs"].item() 
        if "policy_grad_norms" in info:
            log["Policy_Grad_Norms"] = info["policy_grad_norms"]
        return log


class BC_Transformer(BC):
    """
    BC training with a Transformer policy.
    """
    def _create_networks(self):
        """
        Creates networks and places them into @self.nets.
        """
        assert self.algo_config.transformer.enabled

        self.nets = nn.ModuleDict()
        self.nets["policy"] = PolicyNets.TransformerActorNetwork(
            obs_shapes=self.obs_shapes,
            goal_shapes=self.goal_shapes,
            ac_dim=self.ac_dim,
            encoder_kwargs=ObsUtils.obs_encoder_kwargs_from_config(self.obs_config.encoder),
            **BaseNets.transformer_args_from_config(self.algo_config.transformer),
        )
        self._set_params_from_config()
        self.nets = self.nets.float().to(self.device)
        
    def _set_params_from_config(self):
        """
        Read specific config variables we need for training / eval.
        Called by @_create_networks method
        """
        self.context_length = self.algo_config.transformer.context_length
        self.supervise_all_steps = self.algo_config.transformer.supervise_all_steps
        self.pred_future_acs = self.algo_config.transformer.pred_future_acs
        if self.pred_future_acs:
            assert self.supervise_all_steps is True

    def process_batch_for_training(self, batch):
        """
        Processes input batch from a data loader to filter out
        relevant information and prepare the batch for training.
        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader
        Returns:
            input_batch (dict): processed and filtered batch that
                will be used for training
        """
        input_batch = dict()
        h = self.context_length
        input_batch["obs"] = {k: batch["obs"][k][:, :h, :] for k in batch["obs"]}
        input_batch["goal_obs"] = batch.get("goal_obs", None) # goals may not be present

        if self.supervise_all_steps:
            # supervision on entire sequence (instead of just current timestep)
            if self.pred_future_acs:
                ac_start = h - 1
            else:
                ac_start = 0
            input_batch["actions"] = batch["actions"][:, ac_start:ac_start+h, :]
        else:
            # just use current timestep
            input_batch["actions"] = batch["actions"][:, h-1, :]

        if self.pred_future_acs:
            assert input_batch["actions"].shape[1] == h

        input_batch = TensorUtils.to_device(TensorUtils.to_float(input_batch), self.device)
        return input_batch

    def _forward_training(self, batch, epoch=None):
        """
        Internal helper function for BC_Transformer algo class. Compute forward pass
        and return network outputs in @predictions dict.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

        Returns:
            predictions (dict): dictionary containing network outputs
        """
        # ensure that transformer context length is consistent with temporal dimension of observations
        TensorUtils.assert_size_at_dim(
            batch["obs"], 
            size=(self.context_length), 
            dim=1, 
            msg="Error: expect temporal dimension of obs batch to match transformer context length {}".format(self.context_length),
        )

        predictions = OrderedDict()
        predictions["actions"] = self.nets["policy"](obs_dict=batch["obs"], actions=None, goal_dict=batch["goal_obs"])
        if not self.supervise_all_steps:
            # only supervise final timestep
            predictions["actions"] = predictions["actions"][:, -1, :]
        return predictions

    def get_action(self, obs_dict, goal_dict=None):
        """
        Get policy action outputs.
        Args:
            obs_dict (dict): current observation
            goal_dict (dict): (optional) goal
        Returns:
            action (torch.Tensor): action tensor
        """
        assert not self.nets.training

        output = self.nets["policy"](obs_dict, actions=None, goal_dict=goal_dict)

        if self.supervise_all_steps:
            if self.algo_config.transformer.pred_future_acs:
                output = output[:, 0, :]
            else:
                output = output[:, -1, :]
        else:
            output = output[:, -1, :]

        return output

        

class BC_Transformer_GMM(BC_Transformer):
    """
    BC training with a Transformer GMM policy.
    """
    def _create_networks(self):
        """
        Creates networks and places them into @self.nets.
        """
        assert self.algo_config.gmm.enabled
        assert self.algo_config.transformer.enabled

        self.nets = nn.ModuleDict()
        self.nets["policy"] = PolicyNets.TransformerGMMActorNetwork(
            obs_shapes=self.obs_shapes,
            goal_shapes=self.goal_shapes,
            ac_dim=self.ac_dim,
            num_modes=self.algo_config.gmm.num_modes,
            min_std=self.algo_config.gmm.min_std,
            std_activation=self.algo_config.gmm.std_activation,
            low_noise_eval=self.algo_config.gmm.low_noise_eval,
            encoder_kwargs=ObsUtils.obs_encoder_kwargs_from_config(self.obs_config.encoder),
            **BaseNets.transformer_args_from_config(self.algo_config.transformer),
        )
        self._set_params_from_config()
        self.nets = self.nets.float().to(self.device)

    def _forward_training(self, batch, epoch=None):
        """
        Modify from super class to support GMM training.
        """
        # ensure that transformer context length is consistent with temporal dimension of observations
        TensorUtils.assert_size_at_dim(
            batch["obs"], 
            size=(self.context_length), 
            dim=1, 
            msg="Error: expect temporal dimension of obs batch to match transformer context length {}".format(self.context_length),
        )

        dists = self.nets["policy"].forward_train(
            obs_dict=batch["obs"], 
            actions=None,
            goal_dict=batch["goal_obs"],
            low_noise_eval=False,
        )

        # make sure that this is a batch of multivariate action distributions, so that
        # the log probability computation will be correct
        assert len(dists.batch_shape) == 2 # [B, T]

        if not self.supervise_all_steps:
            # only use final timestep prediction by making a new distribution with only final timestep.
            # This essentially does `dists = dists[:, -1]`
            component_distribution = D.Normal(
                loc=dists.component_distribution.base_dist.loc[:, -1],
                scale=dists.component_distribution.base_dist.scale[:, -1],
            )
            component_distribution = D.Independent(component_distribution, 1)
            mixture_distribution = D.Categorical(logits=dists.mixture_distribution.logits[:, -1])
            dists = D.MixtureSameFamily(
                mixture_distribution=mixture_distribution,
                component_distribution=component_distribution,
            )

        log_probs = dists.log_prob(batch["actions"])

        predictions = OrderedDict(
            log_probs=log_probs,
        )
        return predictions

    def _compute_losses(self, predictions, batch):
        """
        Internal helper function for BC_Transformer_GMM algo class. Compute losses based on
        network outputs in @predictions dict, using reference labels in @batch.
        Args:
            predictions (dict): dictionary containing network outputs, from @_forward_training
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training
        Returns:
            losses (dict): dictionary of losses computed over the batch
        """

        # loss is just negative log-likelihood of action targets
        action_loss = -predictions["log_probs"].mean()
        return OrderedDict(
            log_probs=-action_loss,
            action_loss=action_loss,
        )

    def log_info(self, info):
        """
        Process info dictionary from @train_on_batch to summarize
        information to pass to tensorboard for logging.
        Args:
            info (dict): dictionary of info
        Returns:
            loss_log (dict): name -> summary statistic
        """
        log = PolicyAlgo.log_info(self, info)
        log["Loss"] = info["losses"]["action_loss"].item()
        log["Log_Likelihood"] = info["losses"]["log_probs"].item() 
        if "policy_grad_norms" in info:
            log["Policy_Grad_Norms"] = info["policy_grad_norms"]
        return log
