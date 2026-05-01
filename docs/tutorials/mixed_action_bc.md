# Mixed-Action BC

Mixed-action BC is an opt-in deterministic BC feature for sparse action spaces
that contain both continuous command dimensions and binary state dimensions.
It is intended for cases where most timesteps are no-op frames, but rare
intervention windows need to be seen and optimized directly.

Enable the mixed head through `algo.action_head`:

```json
"algo": {
  "action_head": {
    "type": "mixed",
    "continuous_indices": [0],
    "binary_indices": [1],
    "gate": {
      "enabled": true,
      "threshold": 0.5
    },
    "noop": {
      "continuous_raw_values": [0.0],
      "binary_mode": "repeat_last",
      "initial_binary_values": [1.0]
    },
    "loss_weights": {
      "gate": 1.0,
      "continuous": 1.0,
      "binary": 1.0
    }
  }
}
```

With `gate.enabled=true`, the model predicts whether to emit an intervention.
Continuous and binary value losses are then masked to event timesteps, so no-op
frames do not dominate the value heads. Binary dimensions are trained with
BCE-with-logits instead of continuous regression.

Event-aware resampling is configured separately under `train.event_sampler`:

```json
"train": {
  "event_sampler": {
    "enabled": true,
    "halo": 3,
    "continuous_eps": 1e-6,
    "mixture": {
      "event": 0.30,
      "pre_event": 0.20,
      "background": 0.50
    }
  }
}
```

The sampler changes which sequence starts appear during training. It does not
change the real-world frequency of no-op states, and validation keeps the
natural dataset distribution. Use it to increase exposure to rare event and
pre-event windows, then check validation and rollout behavior under the
natural distribution.

`gate.threshold` is used only at inference. Tune it on validation metrics or
rollouts by measuring false no-op decisions against false interventions. Lower
thresholds increase intervention recall; higher thresholds make the policy more
conservative.

Current scope:

- deterministic BC and BC-RNN are supported
- Transformer and Diffusion Policy mixed heads are not implemented
- action dictionaries with multiple action keys are not supported for event
  labeling in v1
- action normalization still comes from `train.action_config`; no-op raw
  values are converted back to normalized action values before robomimic's
  rollout unnormalization
