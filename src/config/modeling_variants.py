from dataclasses import dataclass
from typing import List, Tuple


@dataclass(frozen=True)
class ModelingSpec:
    mode: str
    probabilistic: bool
    mixture_model: bool
    num_mixtures: int
    num_classes: List[int]
    classes_grouping: List[Tuple[int, ...]]
    num_vector_fields: int
    channel_multiplier: int
    center_field_indices: List[int]
    orientation_field_indices: List[int]

    @property
    def total_channels(self) -> int:
        return sum(self.num_classes)


def get_modeling_spec(
    mode: str,
    *,
    num_center_direction_fields: int = 3,
    orientation_dims: int = 1,
    include_auxiliary_channel: bool = True,
    orientation_confidence: bool = False,
    num_mixtures: int = 1,
) -> ModelingSpec:
    """Build a modeling specification used to configure heads, losses and estimators.

    Args:
        mode: One of ``deterministic``, ``single_gaussian`` or ``gaussian_mixture``.
        num_center_direction_fields: Base fields for center direction (sin, cos, r).
        orientation_dims: Number of orientation axes (1 for planar, 3 for 6-DoF).
        include_auxiliary_channel: Whether to reserve an auxiliary field (e.g. heatmap).
        orientation_confidence: Whether orientation confidence is predicted explicitly.
        num_mixtures: Number of mixture components (used when ``mode`` is ``gaussian_mixture``).

    Returns:
        ModelingSpec describing the required channel layout.
    """

    mode_key = mode.lower()
    if mode_key not in {"deterministic", "single_gaussian", "gaussian_mixture"}:
        raise ValueError(f"Unsupported modeling mode '{mode}'.")

    if orientation_dims < 0:
        raise ValueError("orientation_dims must be non-negative.")

    # Base field indexing -------------------------------------------------
    center_fields = list(range(num_center_direction_fields))
    next_index = num_center_direction_fields

    orientation_sin = list(range(next_index, next_index + orientation_dims))
    next_index += orientation_dims

    orientation_cos = list(range(next_index, next_index + orientation_dims))
    next_index += orientation_dims

    orientation_conf_indices: List[int] = []
    if orientation_confidence:
        orientation_conf_indices = [next_index]
        next_index += 1

    aux_indices: List[int] = []
    if include_auxiliary_channel:
        aux_indices = [next_index]
        next_index += 1

    num_vector_fields = next_index

    center_field_indices = center_fields + aux_indices
    orientation_field_indices = orientation_sin + orientation_cos + orientation_conf_indices

    if not orientation_field_indices:
        raise ValueError("Orientation field set cannot be empty – check orientation_dims/confidence settings.")

    # Mode-specific channel expansion ------------------------------------
    if mode_key == "deterministic":
        probabilistic = False
        mixture_model = False
        effective_mixtures = 1
        channel_multiplier = 1

        def expand_indices(field_indices: List[int]) -> List[int]:
            return list(field_indices)

    elif mode_key == "single_gaussian":
        probabilistic = True
        mixture_model = False
        effective_mixtures = 1
        channel_multiplier = 2

        def expand_indices(field_indices: List[int]) -> List[int]:
            mu = list(field_indices)
            log_var = [idx + num_vector_fields for idx in field_indices]
            return mu + log_var

    else:  # gaussian_mixture
        if num_mixtures < 2:
            raise ValueError("Gaussian mixture mode requires num_mixtures >= 2.")

        probabilistic = True
        mixture_model = True
        effective_mixtures = num_mixtures
        channel_multiplier = 3 * num_mixtures

        total_fields = num_vector_fields * num_mixtures

        def expand_indices(field_indices: List[int]) -> List[int]:
            mu: List[int] = []
            log_var: List[int] = []
            alpha: List[int] = []
            for base_idx in field_indices:
                base_offset = base_idx * num_mixtures
                mu.extend(range(base_offset, base_offset + num_mixtures))

                log_var_offset = total_fields + base_offset
                log_var.extend(range(log_var_offset, log_var_offset + num_mixtures))

                alpha_offset = 2 * total_fields + base_offset
                alpha.extend(range(alpha_offset, alpha_offset + num_mixtures))

            return mu + log_var + alpha

    center_group = tuple(expand_indices(center_field_indices))
    orientation_group = tuple(expand_indices(orientation_field_indices))

    classes_grouping = [center_group, orientation_group]
    num_classes = [len(center_group), len(orientation_group)]

    return ModelingSpec(
        mode=mode_key,
        probabilistic=probabilistic,
        mixture_model=mixture_model,
        num_mixtures=effective_mixtures,
        num_classes=num_classes,
        classes_grouping=classes_grouping,
        num_vector_fields=num_vector_fields,
        channel_multiplier=channel_multiplier,
        center_field_indices=center_field_indices,
        orientation_field_indices=orientation_field_indices,
    )
