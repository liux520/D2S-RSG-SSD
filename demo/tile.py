import torch
from numpy import pi, exp, sqrt
import numpy as np


def _gaussian_weights(tile_width, tile_height, nbatches, device):
    """Generates a gaussian mask of weights for tile contributions"""
    latent_width = tile_width
    latent_height = tile_height

    var = 0.01
    midpoint = (latent_width - 1) / 2  # -1 because index goes from 0 to latent_width - 1
    x_probs = [
        exp(-(x - midpoint) * (x - midpoint) / (latent_width * latent_width) / (2 * var)) / sqrt(2 * pi * var) for x
        in range(latent_width)]
    midpoint = latent_height / 2
    y_probs = [
        exp(-(y - midpoint) * (y - midpoint) / (latent_height * latent_height) / (2 * var)) / sqrt(2 * pi * var) for
        y in range(latent_height)]

    weights = np.outer(y_probs, x_probs)
    return torch.tile(torch.tensor(weights, device=device), (nbatches, 3, 1, 1))


def tile_process_s1(
        tile_size,
        tile_overlap,
        stage_s1_func,
        model_s1,
        lq_latent,
        gt_latent,
        pd_factor,
        basename,
        device
):
    # print(f"[Tiled Latent]: the input size is {lq.shape[-2]}x{lq.shape[-1]}, need to tiled")
    # tile_weights = self._gaussian_weights(tile_size, tile_size, 1)
    b, c ,h, w = lq_latent.shape
    tile_size = min(tile_size, min(h, w))
    tile_weights = _gaussian_weights(tile_size, tile_size, 1, device)

    grid_rows = 0
    cur_x = 0
    while cur_x < lq_latent.size(-1):
        cur_x = max(grid_rows * tile_size - tile_overlap * grid_rows, 0) + tile_size
        grid_rows += 1

    grid_cols = 0
    cur_y = 0
    while cur_y < lq_latent.size(-2):
        cur_y = max(grid_cols * tile_size - tile_overlap * grid_cols, 0) + tile_size
        grid_cols += 1

    print(f's1 tile size: {tile_size}, tile overlap: {tile_overlap}, grid_rows: {grid_rows}, grid_cols: {grid_cols}')
    print(f's1 tile num: {grid_rows * grid_cols}')

    input_list = []
    noise_preds = []
    for row in range(grid_rows):
        noise_preds_row = []
        for col in range(grid_cols):
            if col < grid_cols - 1 or row < grid_rows - 1:
                # extract tile from input image
                ofs_x = max(row * tile_size - tile_overlap * row, 0)
                ofs_y = max(col * tile_size - tile_overlap * col, 0)
                # input tile area on total image
            if row == grid_rows - 1:
                ofs_x = w - tile_size
            if col == grid_cols - 1:
                ofs_y = h - tile_size

            input_start_x = ofs_x
            input_end_x = ofs_x + tile_size
            input_start_y = ofs_y
            input_end_y = ofs_y + tile_size

            # input tile dimensions
            input_tile = lq_latent[:, :, input_start_y:input_end_y, input_start_x:input_end_x]
            input_list.append(input_tile)

            if len(input_list) == 1 or col == grid_cols - 1:
                input_list_t = torch.cat(input_list, dim=0)
                # predict the noise residual
                # model, noisy, coarse, gt, pd_factor, r3_num, r3_factor, basename
                model_out = stage_s1_func(model_s1, input_list_t, gt_latent, pd_factor, basename)
                input_list = []
            noise_preds.append(model_out)

            print(f'row: {row}, col: {col}, ofs_x: {ofs_x}, ofs_y: {ofs_y}, input_start_x: {input_start_x}, input_end_x: {input_end_x}, input_start_y: {input_start_y}, input_end_y: {input_end_y}')

    # Stitch noise predictions for all tiles
    noise_pred = torch.zeros(lq_latent.shape, device=lq_latent.device)
    contributors = torch.zeros(lq_latent.shape, device=lq_latent.device)
    # Add each tile contribution to overall latents
    for row in range(grid_rows):
        for col in range(grid_cols):
            if col < grid_cols - 1 or row < grid_rows - 1:
                # extract tile from input image
                ofs_x = max(row * tile_size - tile_overlap * row, 0)
                ofs_y = max(col * tile_size - tile_overlap * col, 0)
                # input tile area on total image
            if row == grid_rows - 1:
                ofs_x = w - tile_size
            if col == grid_cols - 1:
                ofs_y = h - tile_size

            input_start_x = ofs_x
            input_end_x = ofs_x + tile_size
            input_start_y = ofs_y
            input_end_y = ofs_y + tile_size

            noise_pred[:, :, input_start_y:input_end_y, input_start_x:input_end_x] += noise_preds[
                                                                                          row * grid_cols + col] * tile_weights
            contributors[:, :, input_start_y:input_end_y, input_start_x:input_end_x] += tile_weights
    # Average overlapping areas with more than 1 contributor
    noise_pred /= contributors
    model_pred = noise_pred

    return model_pred


def tile_process_s2(
        tile_size,
        tile_overlap,
        stage_s2_func,
        model_s2,
        lq_latent,
        s1_output,
        gt_latent,
        pd_factor,
        r3_num,
        r3_factor,
        basename,
        device
):
    # print(f"[Tiled Latent]: the input size is {lq.shape[-2]}x{lq.shape[-1]}, need to tiled")
    # tile_weights = self._gaussian_weights(tile_size, tile_size, 1)
    b, c ,h, w = lq_latent.shape
    tile_size = min(tile_size, min(h, w))
    tile_weights = _gaussian_weights(tile_size, tile_size, 1, device)

    grid_rows = 0
    cur_x = 0
    while cur_x < lq_latent.size(-1):
        cur_x = max(grid_rows * tile_size - tile_overlap * grid_rows, 0) + tile_size
        grid_rows += 1

    grid_cols = 0
    cur_y = 0
    while cur_y < lq_latent.size(-2):
        cur_y = max(grid_cols * tile_size - tile_overlap * grid_cols, 0) + tile_size
        grid_cols += 1

    print(f's2 tile size: {tile_size}, tile overlap: {tile_overlap}, grid_rows: {grid_rows}, grid_cols: {grid_cols}')
    print(f's2 tile num: {grid_rows * grid_cols}')

    input_list = []
    s1_input_list = []
    noise_preds = []
    for row in range(grid_rows):
        noise_preds_row = []
        for col in range(grid_cols):
            if col < grid_cols - 1 or row < grid_rows - 1:
                # extract tile from input image
                ofs_x = max(row * tile_size - tile_overlap * row, 0)
                ofs_y = max(col * tile_size - tile_overlap * col, 0)
                # input tile area on total image
            if row == grid_rows - 1:
                ofs_x = w - tile_size
            if col == grid_cols - 1:
                ofs_y = h - tile_size

            input_start_x = ofs_x
            input_end_x = ofs_x + tile_size
            input_start_y = ofs_y
            input_end_y = ofs_y + tile_size

            # input tile dimensions
            input_tile = lq_latent[:, :, input_start_y:input_end_y, input_start_x:input_end_x]
            s1_input_tile = s1_output[:, :, input_start_y:input_end_y, input_start_x:input_end_x]
            input_list.append(input_tile)
            s1_input_list.append(s1_input_tile)

            if len(input_list) == 1 or col == grid_cols - 1:
                input_list_t = torch.cat(input_list, dim=0)
                s1_input_list_t = torch.cat(s1_input_list, dim=0)
                # predict the noise residual
                # model, noisy, coarse, gt, pd_factor, r3_num, r3_factor, basename
                model_out = stage_s2_func(model_s2, input_list_t, s1_input_list_t, gt_latent, pd_factor, r3_num, r3_factor, basename)
                input_list = []
                s1_input_list = []
            noise_preds.append(model_out)

            print(f'row: {row}, col: {col}, ofs_x: {ofs_x}, ofs_y: {ofs_y}, input_start_x: {input_start_x}, input_end_x: {input_end_x}, input_start_y: {input_start_y}, input_end_y: {input_end_y}')

    # Stitch noise predictions for all tiles
    noise_pred = torch.zeros(lq_latent.shape, device=lq_latent.device)
    contributors = torch.zeros(lq_latent.shape, device=lq_latent.device)
    # Add each tile contribution to overall latents
    for row in range(grid_rows):
        for col in range(grid_cols):
            if col < grid_cols - 1 or row < grid_rows - 1:
                # extract tile from input image
                ofs_x = max(row * tile_size - tile_overlap * row, 0)
                ofs_y = max(col * tile_size - tile_overlap * col, 0)
                # input tile area on total image
            if row == grid_rows - 1:
                ofs_x = w - tile_size
            if col == grid_cols - 1:
                ofs_y = h - tile_size

            input_start_x = ofs_x
            input_end_x = ofs_x + tile_size
            input_start_y = ofs_y
            input_end_y = ofs_y + tile_size

            noise_pred[:, :, input_start_y:input_end_y, input_start_x:input_end_x] += noise_preds[
                                                                                          row * grid_cols + col] * tile_weights
            contributors[:, :, input_start_y:input_end_y, input_start_x:input_end_x] += tile_weights
    # Average overlapping areas with more than 1 contributor
    noise_pred /= contributors
    model_pred = noise_pred

    return model_pred


def tile_process_s1_s2_full_pipeline(
        tile_size,
        tile_overlap,
        stage_s1_func,
        model_s1,
        stage_s2_func,
        model_s2,
        lq_latent,
        gt_latent,
        pd_factor,
        r3_num,
        r3_factor,
        basename,
        device
):
    # print(f"[Tiled Latent]: the input size is {lq.shape[-2]}x{lq.shape[-1]}, need to tiled")
    # tile_weights = self._gaussian_weights(tile_size, tile_size, 1)
    b, c ,h, w = lq_latent.shape
    tile_size = min(tile_size, min(h, w))
    tile_weights = _gaussian_weights(tile_size, tile_size, 1, device)

    grid_rows = 0
    cur_x = 0
    while cur_x < lq_latent.size(-1):
        cur_x = max(grid_rows * tile_size - tile_overlap * grid_rows, 0) + tile_size
        grid_rows += 1

    grid_cols = 0
    cur_y = 0
    while cur_y < lq_latent.size(-2):
        cur_y = max(grid_cols * tile_size - tile_overlap * grid_cols, 0) + tile_size
        grid_cols += 1

    print(f's2 tile size: {tile_size}, tile overlap: {tile_overlap}, grid_rows: {grid_rows}, grid_cols: {grid_cols}')
    print(f's2 tile num: {grid_rows * grid_cols}')

    input_list = []
    noise_preds = []
    for row in range(grid_rows):
        noise_preds_row = []
        for col in range(grid_cols):
            if col < grid_cols - 1 or row < grid_rows - 1:
                # extract tile from input image
                ofs_x = max(row * tile_size - tile_overlap * row, 0)
                ofs_y = max(col * tile_size - tile_overlap * col, 0)
                # input tile area on total image
            if row == grid_rows - 1:
                ofs_x = w - tile_size
            if col == grid_cols - 1:
                ofs_y = h - tile_size

            input_start_x = ofs_x
            input_end_x = ofs_x + tile_size
            input_start_y = ofs_y
            input_end_y = ofs_y + tile_size

            # input tile dimensions
            input_tile = lq_latent[:, :, input_start_y:input_end_y, input_start_x:input_end_x]
            input_list.append(input_tile)

            if len(input_list) == 1 or col == grid_cols - 1:
                input_list_t = torch.cat(input_list, dim=0)
                # s1: model, noisy, gt, pd_factor, basename
                s1_model_out = stage_s1_func(model_s1, input_list_t, gt_latent, pd_factor, basename)
                # s2: model, noisy, coarse, gt, pd_factor, r3_num, r3_factor, basename
                model_out = stage_s2_func(model_s2, input_list_t, s1_model_out, gt_latent, pd_factor, r3_num, r3_factor, basename)
                input_list = []
            noise_preds.append(model_out)

            print(f'row: {row}, col: {col}, ofs_x: {ofs_x}, ofs_y: {ofs_y}, input_start_x: {input_start_x}, input_end_x: {input_end_x}, input_start_y: {input_start_y}, input_end_y: {input_end_y}')

    # Stitch noise predictions for all tiles
    noise_pred = torch.zeros(lq_latent.shape, device=lq_latent.device)
    contributors = torch.zeros(lq_latent.shape, device=lq_latent.device)
    # Add each tile contribution to overall latents
    for row in range(grid_rows):
        for col in range(grid_cols):
            if col < grid_cols - 1 or row < grid_rows - 1:
                # extract tile from input image
                ofs_x = max(row * tile_size - tile_overlap * row, 0)
                ofs_y = max(col * tile_size - tile_overlap * col, 0)
                # input tile area on total image
            if row == grid_rows - 1:
                ofs_x = w - tile_size
            if col == grid_cols - 1:
                ofs_y = h - tile_size

            input_start_x = ofs_x
            input_end_x = ofs_x + tile_size
            input_start_y = ofs_y
            input_end_y = ofs_y + tile_size

            noise_pred[:, :, input_start_y:input_end_y, input_start_x:input_end_x] += noise_preds[
                                                                                          row * grid_cols + col] * tile_weights
            contributors[:, :, input_start_y:input_end_y, input_start_x:input_end_x] += tile_weights
    # Average overlapping areas with more than 1 contributor
    noise_pred /= contributors
    model_pred = noise_pred

    return model_pred