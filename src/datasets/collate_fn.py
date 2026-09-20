import logging
import numpy as np
import torch

log = logging.getLogger(__name__)


def safe_collate_fn(batch):
    """
    Custom collate function that handles None samples (missing files) and converts to torch tensors.

    This function filters out None samples (which represent missing/unavailable data)
    and returns torch tensors instead of numpy arrays. Returns None if all samples are invalid.

    Parameters:
        batch (list): List of samples from the dataset (can include None values).

    Returns:
        tuple: (inputs, outputs) as torch tensors, or None if no valid samples.
    """
    # Filter out None samples (which represent missing files)
    valid_samples = [s for s in batch if s is not None]

    if not valid_samples:
        # Return None to signal empty batch - DataLoader will skip it
        log.debug("All samples in batch are None (missing files)")
        return None

    # Collate valid samples - assuming they are tuples of (input, output)
    if isinstance(valid_samples[0], tuple) and len(valid_samples[0]) == 2:
        inputs, outputs = zip(*valid_samples)

        # Handle dict of arrays (different levels: surface, upper_air, etc.)
        if isinstance(inputs[0], dict):
            collated_inputs = {}
            collated_outputs = {}

            for key in inputs[0].keys():
                # Stack numpy arrays and convert to torch tensors
                inp_array = np.stack([inp[key] for inp in inputs])
                out_array = np.stack([out[key] for out in outputs])

                collated_inputs[key] = torch.from_numpy(inp_array).float()
                collated_outputs[key] = torch.from_numpy(out_array).float()

            return collated_inputs, collated_outputs
        else:
            # Simple case: stack and convert to torch tensors
            inp_array = np.stack(inputs)
            out_array = np.stack(outputs)
            return torch.from_numpy(inp_array).float(), torch.from_numpy(out_array).float()

    raise ValueError(f"Unexpected batch format: {type(valid_samples[0])}")