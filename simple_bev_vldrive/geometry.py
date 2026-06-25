import torch

DEVICE = torch.device("mps")


def get_image_grid(height: int, width: int) -> torch.Tensor:
    """Return an image coordinate grid of shape (H*W, 2) in pixel coordinates.

    Coordinates are arranged as (x, y) where x is horizontal pixel index
    and y is vertical pixel index.
    """
    ys = torch.arange(0, height, device=DEVICE, dtype=torch.float32)
    xs = torch.arange(0, width, device=DEVICE, dtype=torch.float32)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="xy")
    return torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=1)


def pixel_to_camera(points_2d: torch.Tensor, depths: torch.Tensor, intrinsics: torch.Tensor) -> torch.Tensor:
    """Project 2D pixel coordinates into 3D camera space.

    Args:
        points_2d: Tensor of shape (N, 2) with (x, y) pixel coordinates.
        depths: Tensor of shape (N,) or (N, 1) with depth values in camera frame.
        intrinsics: Tensor of shape (3, 3) camera intrinsic matrix.

    Returns:
        Tensor of shape (N, 3) in camera coordinates.
    """
    if points_2d.device != DEVICE:
        points_2d = points_2d.to(DEVICE)
    if depths.device != DEVICE:
        depths = depths.to(DEVICE)
    if intrinsics.device != DEVICE:
        intrinsics = intrinsics.to(DEVICE)

    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]

    x = (points_2d[:, 0] - cx) / fx
    y = (points_2d[:, 1] - cy) / fy
    z = depths.view(-1)
    camera_x = x * z
    camera_y = y * z
    camera_z = z
    return torch.stack([camera_x, camera_y, camera_z], dim=1)


def transform_camera_to_world(points_camera: torch.Tensor, extrinsics: torch.Tensor) -> torch.Tensor:
    """Transform camera-space points into world coordinates using a 4x4 extrinsic matrix.

    Args:
        points_camera: Tensor of shape (N, 3).
        extrinsics: Tensor of shape (4, 4) representing camera-to-world transform.

    Returns:
        Tensor of shape (N, 3) in world coordinates.
    """
    if points_camera.device != DEVICE:
        points_camera = points_camera.to(DEVICE)
    if extrinsics.device != DEVICE:
        extrinsics = extrinsics.to(DEVICE)

    num_points = points_camera.shape[0]
    homo = torch.cat([points_camera, torch.ones((num_points, 1), device=DEVICE, dtype=points_camera.dtype)], dim=1)
    world_points = homo @ extrinsics.T
    return world_points[:, :3]


def world_points_to_bev(points_world: torch.Tensor, bev_shape: tuple[int, int], bev_bounds: tuple[float, float, float, float]) -> torch.Tensor:
    """Rasterize world points into a top-down BEV occupancy grid.

    Args:
        points_world: Tensor of shape (N, 3) in world coordinates.
        bev_shape: (height, width) of the BEV map.
        bev_bounds: (x_min, x_max, y_min, y_max) world extents represented in meters.

    Returns:
        Tensor of shape (1, H, W) with occupancy counts or a normalized BEV feature map.
    """
    if points_world.device != DEVICE:
        points_world = points_world.to(DEVICE)

    x_min, x_max, y_min, y_max = bev_bounds
    bev_h, bev_w = bev_shape
    x_range = x_max - x_min
    y_range = y_max - y_min
    if x_range <= 0 or y_range <= 0:
        raise ValueError("BEV bounds must define a positive area.")

    x_coords = points_world[:, 0]
    y_coords = points_world[:, 1]
    valid_mask = (
        (x_coords >= x_min) & (x_coords < x_max) &
        (y_coords >= y_min) & (y_coords < y_max)
    )
    if valid_mask.sum() == 0:
        return torch.zeros((1, bev_h, bev_w), device=DEVICE, dtype=torch.float32)

    x_norm = (x_coords[valid_mask] - x_min) / x_range
    y_norm = (y_coords[valid_mask] - y_min) / y_range
    x_indices = torch.clamp((x_norm * bev_w).long(), min=0, max=bev_w - 1)
    y_indices = torch.clamp((y_norm * bev_h).long(), min=0, max=bev_h - 1)

    bev = torch.zeros((bev_h, bev_w), device=DEVICE, dtype=torch.float32)
    bev.index_put_((y_indices, x_indices), torch.ones_like(x_indices, dtype=bev.dtype), accumulate=True)
    return bev.unsqueeze(0)


def image_to_bev(
    image_height: int,
    image_width: int,
    depth_map: torch.Tensor,
    intrinsics: torch.Tensor,
    extrinsics: torch.Tensor,
    bev_shape: tuple[int, int],
    bev_bounds: tuple[float, float, float, float],
) -> torch.Tensor:
    """Convert an image depth map into a Bird's Eye View aggregate map.

    Args:
        image_height: Image height in pixels.
        image_width: Image width in pixels.
        depth_map: Tensor of shape (H, W) with depth values.
        intrinsics: Tensor of shape (3, 3).
        extrinsics: Tensor of shape (4, 4).
        bev_shape: Output BEV grid shape (H_bev, W_bev).
        bev_bounds: World spatial bounds (x_min, x_max, y_min, y_max).

    Returns:
        Tensor of shape (1, H_bev, W_bev) representing the BEV map.
    """
    if depth_map.device != DEVICE:
        depth_map = depth_map.to(DEVICE)
    points_2d = get_image_grid(image_height, image_width)
    depths = depth_map.reshape(-1)
    camera_points = pixel_to_camera(points_2d, depths, intrinsics)
    world_points = transform_camera_to_world(camera_points, extrinsics)
    return world_points_to_bev(world_points, bev_shape, bev_bounds)
