import torch
from torch.nn.functional import relu

def compute_physics_loss(pred, physics_data, graph, delta_t=1200.0):
    """
    Incorporated from https://github.com/MehdiTaghizadehUVa/modulus/blob/main/examples/weather/flood_modeling/hydrographnet/utils.py
    """
    batch = graph.batch # Returns a tensor of shape (num_nodes,) with the batch index for each node
    unique_ids = torch.unique(batch)
    predicted_diff = pred[:, 1]  # Predicted volume difference (normalized)
    physics_losses = []

    for uid in unique_ids:
        mask = (batch == uid)
        pred_diff_sum = predicted_diff[mask].sum()

        idx = (unique_ids == uid).nonzero(as_tuple=False).item()
        past_volume_norm = physics_data["past_volume"][idx]
        future_volume_norm = physics_data["future_volume"][idx]
        # For term1: use average inflow and precipitation
        denorm_avg_inflow = physics_data["avg_inflow"][idx]
        denorm_avg_precip = physics_data["avg_precipitation"][idx]
        # For term2: use next step inflow and precipitation
        denorm_next_inflow = physics_data["next_inflow"][idx]
        denorm_next_precip = physics_data["next_precip"][idx]

        volume_mean = physics_data["volume_mean"][idx]
        volume_std = physics_data["volume_std"][idx]
        num_nodes = physics_data["num_nodes"][idx]
        area_sum = physics_data["area_sum"][idx]
        infiltration_area_sum = physics_data["infiltration_area_sum"][idx]

        # Denormalize past and future volumes.
        past_volume_denorm = past_volume_norm * volume_std + num_nodes * volume_mean
        future_volume_denorm = future_volume_norm * volume_std + num_nodes * volume_mean

        # Compute the predicted total volume.
        pred_total_volume = past_volume_denorm + volume_std * pred_diff_sum

        # Compute effective precipitation terms.
        new_precip_term = denorm_avg_precip * infiltration_area_sum
        new_next_precip_term = denorm_next_precip * infiltration_area_sum

        # Compute continuity terms using ReLU to enforce non-negativity.
        term1 = relu((pred_total_volume - (
                    past_volume_denorm + delta_t * (denorm_avg_inflow + new_precip_term))) / area_sum) ** 2
        term2 = relu((future_volume_denorm - pred_total_volume - delta_t * (
                    denorm_next_inflow + new_next_precip_term)) / area_sum) ** 2

        physics_losses.append(term1 + term2)

    if physics_losses:
        return torch.stack(physics_losses).mean()
    else:
        return torch.tensor(0.0, device=pred.device)
