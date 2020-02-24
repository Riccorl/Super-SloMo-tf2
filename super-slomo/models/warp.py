# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Image warping using per-pixel flow vectors."""

from typing import Optional

import numpy as np
import tensorflow as tf
from tensorflow_addons.utils import types


@tf.function
def interpolate_bilinear(
    grid: types.TensorLike,
    query_points: types.TensorLike,
    indexing: str = "ij",
    name: Optional[str] = None,
) -> tf.Tensor:
    """Similar to Matlab's interp2 function.

    Finds values for query points on a grid using bilinear interpolation.

    Args:
      grid: a 4-D float `Tensor` of shape `[batch, height, width, channels]`.
      query_points: a 3-D float `Tensor` of N points with shape
        `[batch, N, 2]`.
      indexing: whether the query points are specified as row and column (ij),
        or Cartesian coordinates (xy).
      name: a name for the operation (optional).

    Returns:
      values: a 3-D `Tensor` with shape `[batch, N, channels]`

    Raises:
      ValueError: if the indexing mode is invalid, or if the shape of the
        inputs invalid.
    """
    if indexing != "ij" and indexing != "xy":
        raise ValueError("Indexing mode must be 'ij' or 'xy'")

    with tf.name_scope(name or "interpolate_bilinear"):
        grid = tf.convert_to_tensor(grid)
        query_points = tf.convert_to_tensor(query_points)

        # grid shape checks
        grid_static_shape = grid.shape
        grid_shape = tf.shape(grid)
        if grid_static_shape.dims is not None:
            if len(grid_static_shape) != 4:
                raise ValueError("Grid must be 4D Tensor")
            if grid_static_shape[1] is not None and grid_static_shape[1] < 2:
                raise ValueError("Grid height must be at least 2.")
            if grid_static_shape[2] is not None and grid_static_shape[2] < 2:
                raise ValueError("Grid width must be at least 2.")
        else:
            with tf.control_dependencies(
                [
                    tf.debugging.assert_greater_equal(
                        grid_shape[1], 2, message="Grid height must be at least 2."
                    ),
                    tf.debugging.assert_greater_equal(
                        grid_shape[2], 2, message="Grid width must be at least 2."
                    ),
                    tf.debugging.assert_less_equal(
                        tf.cast(
                            grid_shape[0] * grid_shape[1] * grid_shape[2],
                            dtype=tf.dtypes.float32,
                        ),
                        np.iinfo(np.int32).max / 8.0,
                        message="The image size or batch size is sufficiently "
                        "large that the linearized addresses used by "
                        "tf.gather may exceed the int32 limit.",
                    ),
                ]
            ):
                pass

        # query_points shape checks
        query_static_shape = query_points.shape
        query_shape = tf.shape(query_points)
        if query_static_shape.dims is not None:
            if len(query_static_shape) != 3:
                raise ValueError("Query points must be 3 dimensional.")
            query_hw = query_static_shape[2]
            if query_hw is not None and query_hw != 2:
                raise ValueError("Query points last dimension must be 2.")
        else:
            with tf.control_dependencies(
                [
                    tf.debugging.assert_equal(
                        query_shape[2],
                        2,
                        message="Query points last dimension must be 2.",
                    )
                ]
            ):
                pass

        batch_size, height, width, channels = (
            grid_shape[0],
            grid_shape[1],
            grid_shape[2],
            grid_shape[3],
        )

        num_queries = query_shape[1]

        query_type = query_points.dtype
        grid_type = grid.dtype

        alphas = []
        floors = []
        ceils = []
        index_order = [0, 1] if indexing == "ij" else [1, 0]
        unstacked_query_points = tf.unstack(query_points, axis=2, num=2)

        for i, dim in enumerate(index_order):
            with tf.name_scope("dim-" + str(dim)):
                queries = unstacked_query_points[dim]

                size_in_indexing_dimension = grid_shape[i + 1]

                # max_floor is size_in_indexing_dimension - 2 so that max_floor + 1
                # is still a valid index into the grid.
                max_floor = tf.cast(size_in_indexing_dimension - 2, query_type)
                min_floor = tf.constant(0.0, dtype=query_type)
                floor = tf.math.minimum(
                    tf.math.maximum(min_floor, tf.math.floor(queries)), max_floor
                )
                int_floor = tf.cast(floor, tf.dtypes.int32)
                floors.append(int_floor)
                ceil = int_floor + 1
                ceils.append(ceil)

                # alpha has the same type as the grid, as we will directly use alpha
                # when taking linear combinations of pixel values from the image.
                alpha = tf.cast(queries - floor, grid_type)
                min_alpha = tf.constant(0.0, dtype=grid_type)
                max_alpha = tf.constant(1.0, dtype=grid_type)
                alpha = tf.math.minimum(tf.math.maximum(min_alpha, alpha), max_alpha)

                # Expand alpha to [b, n, 1] so we can use broadcasting
                # (since the alpha values don't depend on the channel).
                alpha = tf.expand_dims(alpha, 2)
                alphas.append(alpha)

            flattened_grid = tf.reshape(grid, [batch_size * height * width, channels])
            batch_offsets = tf.reshape(
                tf.range(batch_size) * height * width, [batch_size, 1]
            )

        # This wraps tf.gather. We reshape the image data such that the
        # batch, y, and x coordinates are pulled into the first dimension.
        # Then we gather. Finally, we reshape the output back. It's possible this
        # code would be made simpler by using tf.gather_nd.
        def gather(y_coords, x_coords, name):
            with tf.name_scope("gather-" + name):
                linear_coordinates = batch_offsets + y_coords * width + x_coords
                gathered_values = tf.gather(flattened_grid, linear_coordinates)
                return tf.reshape(gathered_values, [batch_size, num_queries, channels])

        # grab the pixel values in the 4 corners around each query point
        top_left = gather(floors[0], floors[1], "top_left")
        top_right = gather(floors[0], ceils[1], "top_right")
        bottom_left = gather(ceils[0], floors[1], "bottom_left")
        bottom_right = gather(ceils[0], ceils[1], "bottom_right")

        # now, do the actual interpolation
        with tf.name_scope("interpolate"):
            interp_top = alphas[1] * (top_right - top_left) + top_left
            interp_bottom = alphas[1] * (bottom_right - bottom_left) + bottom_left
            interp = alphas[0] * (interp_bottom - interp_top) + interp_top

        return interp


@tf.function
def dense_image_warp(
    image: types.TensorLike, flow: types.TensorLike, name: Optional[str] = None
) -> tf.Tensor:
    """Image warping using per-pixel flow vectors.

    Apply a non-linear warp to the image, where the warp is specified by a
    dense flow field of offset vectors that define the correspondences of
    pixel values in the output image back to locations in the source image.
    Specifically, the pixel value at output[b, j, i, c] is
    images[b, j - flow[b, j, i, 0], i - flow[b, j, i, 1], c].

    The locations specified by this formula do not necessarily map to an int
    index. Therefore, the pixel value is obtained by bilinear
    interpolation of the 4 nearest pixels around
    (b, j - flow[b, j, i, 0], i - flow[b, j, i, 1]). For locations outside
    of the image, we use the nearest pixel values at the image boundary.

    Args:
      image: 4-D float `Tensor` with shape `[batch, height, width, channels]`.
      flow: A 4-D float `Tensor` with shape `[batch, height, width, 2]`.
      name: A name for the operation (optional).

      Note that image and flow can be of type tf.half, tf.float32, or
      tf.float64, and do not necessarily have to be the same type.

    Returns:
      A 4-D float `Tensor` with shape`[batch, height, width, channels]`
        and same type as input image.

    Raises:
      ValueError: if height < 2 or width < 2 or the inputs have the wrong
        number of dimensions.
    """
    with tf.name_scope(name or "dense_image_warp"):
        image = tf.convert_to_tensor(image)
        flow = tf.convert_to_tensor(flow)
        batch_size, height, width, channels = (
            tf.shape(image)[0],
            tf.shape(image)[1],
            tf.shape(image)[2],
            tf.shape(image)[3],
        )

        # The flow is defined on the image grid. Turn the flow into a list of query
        # points in the grid space.
        grid_x, grid_y = tf.meshgrid(tf.range(width), tf.range(height))
        stacked_grid = tf.cast(tf.stack([grid_y, grid_x], axis=2), flow.dtype)
        batched_grid = tf.expand_dims(stacked_grid, axis=0)
        query_points_on_grid = batched_grid - flow
        query_points_flattened = tf.reshape(
            query_points_on_grid, [batch_size, height * width, 2]
        )
        # Compute values at the query points, then reshape the result back to the
        # image grid.
        interpolated = interpolate_bilinear(image, query_points_flattened)
        interpolated = tf.reshape(interpolated, [batch_size, height, width, channels])
        return interpolated


@tf.function
def flow_back_wrap(x, v, resize=False, normalize=False, crop=None, out="CONSTANT"):
    """
      Args:
        x - Input tensor [N, H, W, C]
        v - Vector flow tensor [N, H, W, 2], tf.float32
        (optional)
        resize - Whether to resize v as same size as x
        normalize - Whether to normalize v from scale 1 to H (or W).
                    h : [-1, 1] -> [-H/2, H/2]
                    w : [-1, 1] -> [-W/2, W/2]
        crop - Setting the region to sample. 4-d list [h0, h1, w0, w1]
        out  - Handling out of boundary value.
               Zero value is used if out="CONSTANT".
               Boundary values are used if out="EDGE".
    """

    def _get_grid_array(N, H, W, h, w):
        N_i = tf.range(N)
        H_i = tf.range(h + 1, h + H + 1)
        W_i = tf.range(w + 1, w + W + 1)
        n, h, w, = tf.meshgrid(N_i, H_i, W_i, indexing="ij")
        n = tf.expand_dims(n, axis=3)  # [N, H, W, 1]
        h = tf.expand_dims(h, axis=3)  # [N, H, W, 1]
        w = tf.expand_dims(w, axis=3)  # [N, H, W, 1]
        n = tf.cast(n, tf.float32)  # [N, H, W, 1]
        h = tf.cast(h, tf.float32)  # [N, H, W, 1]
        w = tf.cast(w, tf.float32)  # [N, H, W, 1]

        return n, h, w

    shape = tf.shape(x)  # TRY : Dynamic shape
    N = shape[0]
    if crop is None:
        H_ = H = shape[1]
        W_ = W = shape[2]
        h = w = 0
    else:
        H_ = shape[1]
        W_ = shape[2]
        H = crop[1] - crop[0]
        W = crop[3] - crop[2]
        h = crop[0]
        w = crop[2]

    if resize:
        if callable(resize):
            v = resize(v, [H, W])
        else:
            v = tf.image.resize(v, [H, W])

    if out == "CONSTANT":
        x = tf.pad(x, ((0, 0), (1, 1), (1, 1), (0, 0)), mode="CONSTANT")
    elif out == "EDGE":
        x = tf.pad(x, ((0, 0), (1, 1), (1, 1), (0, 0)), mode="REFLECT")

    vy, vx = tf.split(v, 2, axis=3)
    if normalize:
        vy = vy * tf.cast(
            H, dtype=tf.float32
        )  # TODO: Check why  vy * (H/2) didn't work
        vy = vy / 2
        vx = vy * tf.cast(W, dtype=tf.float32)
        vx = vx / 2

    n, h, w = _get_grid_array(N, H, W, h, w)  # [N, H, W, 3]
    vx0 = tf.floor(vx)
    vy0 = tf.floor(vy)
    vx1 = vx0 + 1
    vy1 = vy0 + 1  # [N, H, W, 1]

    H_1 = tf.cast(H_ + 1, tf.float32)
    W_1 = tf.cast(W_ + 1, tf.float32)
    iy0 = tf.clip_by_value(vy0 + h, 0.0, H_1)
    iy1 = tf.clip_by_value(vy1 + h, 0.0, H_1)
    ix0 = tf.clip_by_value(vx0 + w, 0.0, W_1)
    ix1 = tf.clip_by_value(vx1 + w, 0.0, W_1)

    i00 = tf.concat([n, iy0, ix0], 3)
    i01 = tf.concat([n, iy1, ix0], 3)
    i10 = tf.concat([n, iy0, ix1], 3)
    i11 = tf.concat([n, iy1, ix1], 3)  # [N, H, W, 3]
    i00 = tf.cast(i00, tf.int32)
    i01 = tf.cast(i01, tf.int32)
    i10 = tf.cast(i10, tf.int32)
    i11 = tf.cast(i11, tf.int32)
    x00 = tf.gather_nd(x, i00)
    x01 = tf.gather_nd(x, i01)
    x10 = tf.gather_nd(x, i10)
    x11 = tf.gather_nd(x, i11)
    w00 = tf.cast((vx1 - vx) * (vy1 - vy), tf.float32)
    w01 = tf.cast((vx1 - vx) * (vy - vy0), tf.float32)
    w10 = tf.cast((vx - vx0) * (vy1 - vy), tf.float32)
    w11 = tf.cast((vx - vx0) * (vy - vy0), tf.float32)
    output = tf.add_n([w00 * x00, w01 * x01, w10 * x10, w11 * x11])

    return output


@tf.function
def get_pixel_value(img, x, y):
    """
    Utility function to get pixel value for coordinate
    vectors x and y from a  4D tensor image.
    Input
    -----
    - img: tensor of shape (B, H, W, C)
    - x: flattened tensor of shape (B*H*W,)
    - y: flattened tensor of shape (B*H*W,)
    Returns
    -------
    - output: tensor of shape (B, H, W, C)
    """
    shape = tf.shape(x)
    batch_size = shape[0]
    height = shape[1]
    width = shape[2]

    batch_idx = tf.range(0, batch_size)
    batch_idx = tf.reshape(batch_idx, (batch_size, 1, 1))
    b = tf.tile(batch_idx, (1, height, width))

    indices = tf.stack([b, y, x], 3)

    return tf.gather_nd(img, indices)


@tf.function
def bilinear_sampler(img, x, y):
    """
    Performs bilinear sampling of the input images according to the
    normalized coordinates provided by the sampling grid. Note that
    the sampling is done identically for each channel of the input.
    To test if the function works properly, output image should be
    identical to input image when theta is initialized to identity
    transform.
    Input
    -----
    - img: batch of images in (B, H, W, C) layout.
    - grid: x, y which is the output of affine_grid_generator.
    Returns
    -------
    - out: interpolated images according to grids. Same size as grid.
    """
    H = tf.shape(img)[1]
    W = tf.shape(img)[2]
    max_y = tf.cast(H - 1, "int32")
    max_x = tf.cast(W - 1, "int32")
    zero = tf.zeros([], dtype="int32")

    # rescale x and y to [0, W-1/H-1]
    x = tf.cast(x, "float32")
    y = tf.cast(y, "float32")
    x = 0.5 * ((x + 1.0) * tf.cast(max_x - 1, "float32"))
    y = 0.5 * ((y + 1.0) * tf.cast(max_y - 1, "float32"))

    # grab 4 nearest corner points for each (x_i, y_i)
    x0 = tf.cast(tf.floor(x), "int32")
    x1 = x0 + 1
    y0 = tf.cast(tf.floor(y), "int32")
    y1 = y0 + 1

    # clip to range [0, H-1/W-1] to not violate img boundaries
    x0 = tf.clip_by_value(x0, zero, max_x)
    x1 = tf.clip_by_value(x1, zero, max_x)
    y0 = tf.clip_by_value(y0, zero, max_y)
    y1 = tf.clip_by_value(y1, zero, max_y)

    # get pixel value at corner coords
    Ia = get_pixel_value(img, x0, y0)
    Ib = get_pixel_value(img, x0, y1)
    Ic = get_pixel_value(img, x1, y0)
    Id = get_pixel_value(img, x1, y1)

    # recast as float for delta calculation
    x0 = tf.cast(x0, "float32")
    x1 = tf.cast(x1, "float32")
    y0 = tf.cast(y0, "float32")
    y1 = tf.cast(y1, "float32")

    # calculate deltas
    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)

    # add dimension for addition
    wa = tf.expand_dims(wa, axis=3)
    wb = tf.expand_dims(wb, axis=3)
    wc = tf.expand_dims(wc, axis=3)
    wd = tf.expand_dims(wd, axis=3)

    # compute output
    out = tf.add_n([wa * Ia, wb * Ib, wc * Ic, wd * Id])

    return out
