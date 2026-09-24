"""Native Astribot state/action IO, ported from the pi0.5 training pipeline.

The pose and delta math follows ``StateActionTransform`` in
``robot-learning-framework-aug/src/utils/BaseProcessor.py``. A pose contains
xyz followed by the first two *rows* of its rotation matrix. This module is
self contained: it does not import the reference repository or normalize data.
"""

from collections.abc import Mapping

import torch


class Pi05IO:
    """Select native dimensions, transform actions, and restore native outputs.

    ``action_valid_dim`` order defines the packed model layout. Positive
    ``action_delta_mask`` entries transform that segment; negative entries keep
    it absolute. Unselected dimensions are restored from the raw state.

    Optional ``model_indices`` maps each packed coordinate to a model input /
    output slot. This can retain existing xyz, gripper, and base slots while
    keeping pretrained projection shapes. Rotation slots still need adaptation
    when their representation changes; slot reuse does not preserve semantics.

    ``preprocess`` and ``postprocess`` process one observation and an optional
    action chunk, or batches shaped (B, D) and (B, T, D). Both preserve their
    inputs. Normalization belongs between these methods, and ``postprocess``
    must receive the original raw state.
    """

    def __init__(self, config: dict):
        if not isinstance(config, Mapping):
            raise TypeError("pi05 IO config must be a mapping")
        ranges = config.get("action_valid_dim")
        if not isinstance(ranges, Mapping) or not ranges:
            raise ValueError("action_valid_dim must be a nonempty ordered mapping")
        self.action_valid_dim = {}
        self._packed_ranges = {}
        occupied = set()
        offset = 0
        for name, bounds in ranges.items():
            if not isinstance(name, str) or not name:
                raise ValueError("action_valid_dim names must be nonempty strings")
            if not isinstance(bounds, (tuple, list)) or len(bounds) != 2:
                raise ValueError(f"{name}: expected [start, end]")
            start, end = bounds
            if any(type(v) is not int for v in bounds) or not 0 <= start < end:
                raise ValueError(f"{name}: invalid dimension range {bounds!r}")
            indices = set(range(start, end))
            if occupied & indices:
                raise ValueError(f"{name}: action_valid_dim ranges overlap")
            occupied.update(indices)
            self.action_valid_dim[name] = (start, end)
            self._packed_ranges[name] = (offset, offset + end - start)
            offset += end - start
        self.selected_dim = offset
        self.action_max_dim = config.get("action_max_dim", 32)
        self.raw_dim = config.get("raw_dim", max(end for _, end in self.action_valid_dim.values()))
        for name, value in (("action_max_dim", self.action_max_dim), ("raw_dim", self.raw_dim)):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.selected_dim > self.action_max_dim:
            raise ValueError(f"selected_dim={self.selected_dim} exceeds action_max_dim={self.action_max_dim}")
        if max(occupied) >= self.raw_dim:
            raise ValueError("action_valid_dim exceeds raw_dim")
        model_indices = config.get("model_indices", list(range(self.action_max_dim)))
        if not isinstance(model_indices, (list, tuple)) or len(model_indices) != self.action_max_dim:
            raise ValueError("model_indices must have exactly action_max_dim entries")
        if any(type(index) is not int or index < 0 for index in model_indices):
            raise ValueError("model_indices must contain nonnegative integers")
        if len(set(model_indices)) != len(model_indices):
            raise ValueError("model_indices must be unique")
        self.model_indices = tuple(model_indices)

        self.action_mode = config.get("action_mode", "delta_so3")
        self.state_axis = config.get("state_axis", "base")
        self.action_axis = config.get("action_axis", "self")
        if self.action_mode not in ("delta_so3", "delta_joints", "absolute"):
            raise ValueError(f"unsupported action_mode: {self.action_mode!r}")
        if self.state_axis not in ("base", "torso", "abstorso"):
            raise ValueError(f"unsupported state_axis: {self.state_axis!r}")
        if self.action_axis not in ("self", "base", "torso"):
            raise ValueError(f"unsupported action_axis: {self.action_axis!r}")
        if self.action_mode == "delta_joints" and (self.state_axis != "base" or self.action_axis == "torso"):
            raise ValueError("delta_joints supports state_axis=base and action_axis=self/base")
        if self.state_axis != "base" or self.action_axis == "torso":
            torso_range = self._packed_ranges.get("torso")
            if torso_range is None or torso_range[1] - torso_range[0] != 9:
                raise ValueError("torso frame requires a selected 9D torso pose")

        widths = [end - start for start, end in self.action_valid_dim.values()]
        mask = config.get("action_delta_mask")
        if mask is None:
            if self.action_mode != "absolute":
                raise ValueError("action_delta_mask is required for delta modes")
            mask = widths
        if not isinstance(mask, (tuple, list)) or len(mask) != len(widths):
            raise ValueError("action_delta_mask must have one entry per selected segment")
        if any(type(value) is not int or abs(value) != width for value, width in zip(mask, widths)):
            raise ValueError("action_delta_mask lengths must match action_valid_dim in order")
        self.action_delta_mask = tuple(mask)
        self._segments = [
            (slice(*bounds), width, value > 0)
            for bounds, width, value in zip(self._packed_ranges.values(), widths, mask)
        ]

    @staticmethod
    def _tensor(value, name: str) -> torch.Tensor:
        try:
            result = torch.as_tensor(value).to(dtype=torch.float32)
        except (TypeError, ValueError, RuntimeError) as exc:
            raise ValueError(f"{name} must be a numeric tensor") from exc
        if result.ndim == 0 or not torch.isfinite(result).all():
            raise ValueError(f"{name} must have a dimension axis and contain only finite values")
        return result

    def select(self, raw_tensor: torch.Tensor) -> torch.Tensor:
        """Select configured segments and pad to action_max_dim, without aliasing."""
        raw = self._tensor(raw_tensor, "raw_tensor")
        if raw.shape[-1] != self.raw_dim:
            raise ValueError(f"expected native width {self.raw_dim}, got {raw.shape[-1]}")
        packed = raw.new_zeros((*raw.shape[:-1], self.action_max_dim))
        for name, (start, end) in self.action_valid_dim.items():
            dest = slice(*self._packed_ranges[name])
            packed[..., dest] = raw[..., start:end]
        return packed

    def _model_index_tensor(self, max_dim: int, device=None) -> torch.Tensor:
        if type(max_dim) is not int or max_dim <= 0:
            raise ValueError("model max_dim must be a positive integer")
        if max(self.model_indices) >= max_dim:
            raise ValueError(f"model_indices exceed model width {max_dim}")
        return torch.tensor(self.model_indices, dtype=torch.long, device=device)

    def pack_model(self, packed, max_dim: int) -> torch.Tensor:
        """Scatter packed coordinates into model slots, zeroing other slots.

        Normalize in packed space before calling this method. Any leading
        dimensions are preserved; the last dimension must be action_max_dim.
        """
        packed = self._tensor(packed, "packed")
        if packed.shape[-1] != self.action_max_dim:
            raise ValueError(f"expected packed width {self.action_max_dim}, got {packed.shape[-1]}")
        indices = self._model_index_tensor(max_dim, packed.device)
        model = packed.new_zeros((*packed.shape[:-1], max_dim))
        model.index_copy_(-1, indices, packed)
        return model

    def unpack_model(self, model_tensor) -> torch.Tensor:
        """Gather model slots back to packed order before denormalization.

        Index selection follows model_indices exactly. Boolean-mask selection
        would sort by model position and corrupt a nonmonotonic mapping.
        """
        model = self._tensor(model_tensor, "model_tensor")
        indices = self._model_index_tensor(model.shape[-1], model.device)
        return model.index_select(-1, indices)

    def model_mask(self, max_dim: int, device=None) -> torch.Tensor:
        """Return a bool mask for the selected, supervised model coordinates."""
        indices = self._model_index_tensor(max_dim, device)
        mask = torch.zeros(max_dim, dtype=torch.bool, device=device)
        mask[indices[:self.selected_dim]] = True
        return mask

    @staticmethod
    def _rotation(six: torch.Tensor) -> torch.Tensor:
        """Reconstruct a rotation from two rows using pi0.5 Gram-Schmidt."""
        first, second = six[..., :3], six[..., 3:6]
        first_norm = torch.linalg.vector_norm(first, dim=-1, keepdim=True)
        if not torch.isfinite(first_norm).all() or (first_norm < 1e-8).any():
            raise ValueError("SO3 6D first row is degenerate")
        first = first / first_norm.clamp(min=1e-8)
        second = second - (second * first).sum(dim=-1, keepdim=True) * first
        second_norm = torch.linalg.vector_norm(second, dim=-1, keepdim=True)
        if not torch.isfinite(second_norm).all() or (second_norm < 1e-8).any():
            raise ValueError("SO3 6D rows are collinear or degenerate")
        second = second / second_norm.clamp(min=1e-8)
        third = torch.cross(first, second, dim=-1)
        third = third / torch.linalg.vector_norm(third, dim=-1, keepdim=True).clamp(min=1e-8)
        return torch.stack((first, second, third), dim=-2)

    @staticmethod
    def _pose(position: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
        return torch.cat((position, rotation[..., :2, :].reshape(*rotation.shape[:-2], 6)), dim=-1)

    def _reference(self, state: torch.Tensor, axis: str) -> torch.Tensor:
        if axis in ("torso", "abstorso"):
            return state[..., slice(*self._packed_ranges["torso"])].clone()
        identity = state.new_zeros((*state.shape[:-1], 9))
        identity[..., 3] = identity[..., 7] = 1
        return identity

    def _in_frame(self, pose: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        if pose.ndim > reference.ndim:
            reference = reference.unsqueeze(-2)
        rotation = self._rotation(reference[..., 3:]).transpose(-1, -2)
        return self._pose(
            (rotation @ (pose[..., :3] - reference[..., :3]).unsqueeze(-1)).squeeze(-1),
            rotation @ self._rotation(pose[..., 3:]),
        )

    def _from_frame(self, pose: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        if pose.ndim > reference.ndim:
            reference = reference.unsqueeze(-2)
        rotation = self._rotation(reference[..., 3:])
        return self._pose(
            (rotation @ pose[..., :3].unsqueeze(-1)).squeeze(-1) + reference[..., :3],
            rotation @ self._rotation(pose[..., 3:]),
        )

    def _delta(self, state: torch.Tensor, actions: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        state = state.unsqueeze(-2)
        reference = reference.unsqueeze(-2)
        state_rot = self._rotation(state[..., 3:])
        action_rot = self._rotation(actions[..., 3:])
        ref_rot = self._rotation(reference[..., 3:])
        delta_rot = ref_rot.transpose(-1, -2) @ (action_rot @ state_rot.transpose(-1, -2)) @ ref_rot
        delta_pos = (ref_rot.transpose(-1, -2) @ (actions[..., :3] - state[..., :3]).unsqueeze(-1)).squeeze(-1)
        return self._pose(delta_pos, delta_rot)

    def _restore(self, state: torch.Tensor, delta: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        state = state.unsqueeze(-2)
        reference = reference.unsqueeze(-2)
        state_rot = self._rotation(state[..., 3:])
        delta_rot = self._rotation(delta[..., 3:])
        ref_rot = self._rotation(reference[..., 3:])
        rotation = (ref_rot @ delta_rot @ ref_rot.transpose(-1, -2)) @ state_rot
        position = (ref_rot @ delta[..., :3].unsqueeze(-1)).squeeze(-1) + state[..., :3]
        return self._pose(position, rotation)

    def preprocess(self, raw_state, raw_actions=None):
        """Return selected, transformed state/actions before normalization."""
        state = self.select(raw_state)
        if state.ndim not in (1, 2):
            raise ValueError("raw_state must have shape (raw_dim,) or (B, raw_dim)")
        actions = None if raw_actions is None else self.select(raw_actions)
        if actions is not None and (actions.ndim != state.ndim + 1 or actions.shape[:-2] != state.shape[:-1] or actions.shape[-2] == 0):
            raise ValueError("raw_actions must match state batch dimensions and have shape (..., T, raw_dim), T > 0")
        if actions is not None and actions.device != state.device:
            raise ValueError("raw_state and raw_actions must use the same device")
        original = state.clone()
        state_ref = self._reference(original, self.state_axis)
        action_ref = self._reference(original, self.action_axis)
        for seg, width, active in self._segments:
            if not active:
                continue
            if self.action_mode != "delta_joints" and width == 9:
                anchor = original[..., seg]
                if actions is not None:
                    if self.action_mode == "delta_so3":
                        ref = anchor if self.action_axis == "self" else action_ref
                        actions[..., seg] = self._delta(anchor, actions[..., seg], ref)
                    elif self.action_axis != "self":
                        actions[..., seg] = self._in_frame(actions[..., seg], action_ref)
                state[..., seg] = self._in_frame(anchor, state_ref)
            elif actions is not None and self.action_mode != "absolute":
                actions[..., seg] -= original[..., seg].unsqueeze(-2)
        # Match the reference implementation: retain absolute torso state while
        # expressing the other poses in the torso frame (despite its old comment).
        if self.state_axis == "abstorso":
            state[..., slice(*self._packed_ranges["torso"])] = state_ref
        if not torch.isfinite(state).all() or (actions is not None and not torch.isfinite(actions).all()):
            raise ValueError("transformed state/actions contain nonfinite values")
        return state, actions

    def postprocess(self, raw_state, denormalized_selected_actions):
        """Restore absolute native actions using the untouched observation anchor.

        The result has raw_dim columns. In particular, a 32D policy that excludes
        the head returns 34D commands with the head copied from raw_state.
        """
        raw = self._tensor(raw_state, "raw_state")
        if raw.ndim not in (1, 2):
            raise ValueError("raw_state must have shape (raw_dim,) or (B, raw_dim)")
        state = self.select(raw)
        actions = self._tensor(denormalized_selected_actions, "denormalized_selected_actions").clone()
        if actions.ndim != raw.ndim + 1 or actions.shape[:-2] != raw.shape[:-1] or actions.shape[-2] == 0 or actions.shape[-1] not in (self.selected_dim, self.action_max_dim):
            raise ValueError(f"actions must match state batch dimensions and have shape (..., T, {self.selected_dim}) or (..., T, {self.action_max_dim}), T > 0")
        if actions.device != state.device:
            raise ValueError("raw_state and actions must use the same device")
        action_ref = self._reference(state, self.action_axis)
        for seg, width, active in self._segments:
            if not active:
                continue
            if self.action_mode != "delta_joints" and width == 9:
                if self.action_mode == "delta_so3":
                    ref = state[..., seg] if self.action_axis == "self" else action_ref
                    actions[..., seg] = self._restore(state[..., seg], actions[..., seg], ref)
                elif self.action_axis != "self":
                    actions[..., seg] = self._from_frame(actions[..., seg], action_ref)
            elif self.action_mode != "absolute":
                actions[..., seg] += state[..., seg].unsqueeze(-2)
        result = raw.unsqueeze(-2).expand(*raw.shape[:-1], actions.shape[-2], self.raw_dim).clone()
        for name, (start, end) in self.action_valid_dim.items():
            result[..., start:end] = actions[..., slice(*self._packed_ranges[name])]
        if not torch.isfinite(result).all():
            raise ValueError("restored native actions contain nonfinite values")
        return result
