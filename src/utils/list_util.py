import torch


def group_tensor_pairs(*data: torch.Tensor) -> tuple[dict[str, torch.Tensor], ...]:
    """
    Groups a flat sequence of tensors into structured (upper_air, surface) pairs.

    The input tensors must alternate in the following order:

        upper_air_0, surface_0,
        upper_air_1, surface_1,
        ...

    The number of input tensors must be even.

    Args:
        *data (torch.Tensor):
            Variable-length sequence of tensors alternating between
            "upper_air" and "surface" data for each step.

    Returns:
        tuple[dict[str, torch.Tensor]]:
            A tuple where each element corresponds to one step and contains:

                {
                    "upper_air": Tensor of shape [B, Z, H, W, C],
                    "surface":   Tensor of shape [B, 1, H, W, C],
                }
    """
    if len(data) % 2 != 0:
        raise ValueError(
            f"_group_pairs expects even number of tensors (upper/surface pairs), "
            f"got {len(data)}"
        )
    return tuple([{"upper_air": data[2*i], "surface": data[2*i+1]} for i in range(len(data)//2)])


def ungroup_tensor_pairs(*data: dict[str, torch.Tensor]) -> tuple[torch.Tensor, ...]:
    """
    Flattens structured (upper_air, surface) dictionaries into a flat tensor tuple.

    This function performs the inverse operation of `_group_pairs`. Each input
    dictionary represents one step and must contain exactly two keys:

        "upper_air" and "surface".

    The tensors are flattened in the following order for each step:

        upper_air_0, surface_0,
        upper_air_1, surface_1,
        ...

    Args:
        *data (dict[str, torch.Tensor]):
            Variable-length sequence of dictionaries, where each dictionary
            has the structure:

                {
                    "upper_air": Tensor of shape [B, Z, H, W, C],
                    "surface":   Tensor of shape [B, 1, H, W, C],
                }

    Returns:
        tuple[torch.Tensor]:
            A flat tuple of tensors alternating between "upper_air" and
            "surface" for each step.
    """
    return tuple([data[i][k] for i in range(len(data)) for k in ["upper_air", "surface"]])
