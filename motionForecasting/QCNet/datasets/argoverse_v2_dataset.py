"""Argoverse 2 motion-forecasting dataset.

Ported from the official QCNet ``datasets/argoverse_v2_dataset.py``. The feature-extraction
logic (``get_agent_features`` / ``get_map_features``) is kept faithful and still uses the
``av2`` API for map/scenario parsing. The only change is that this is a plain
``torch.utils.data.Dataset`` emitting a nested ``dict`` (with tuple keys for the heterogeneous
edges) instead of a ``torch_geometric`` ``HeteroData`` object, so the rest of the pipeline is
PyG-free. Processed scenes are cached as pickles next to the raw data.
"""
import math
import os
import pickle
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from utils import safe_list_index
from utils import side_to_directed_lineseg

from av2.geometry.interpolate import compute_midpoint_line
from av2.map.map_api import ArgoverseStaticMap
from av2.map.map_primitives import Polyline
from av2.utils.io import read_json_file


class ArgoverseV2Dataset(Dataset):

    def __init__(self,
                 root: str,
                 split: str,
                 raw_dir: Optional[str] = None,
                 processed_dir: Optional[str] = None,
                 transform: Optional[Callable] = None,
                 dim: int = 3,
                 num_historical_steps: int = 50,
                 num_future_steps: int = 60,
                 predict_unseen_agents: bool = False,
                 vector_repr: bool = True,
                 max_scenarios: Optional[int] = None) -> None:
        super().__init__()
        root = os.path.expanduser(os.path.normpath(root))
        if split not in ('train', 'val', 'test'):
            raise ValueError(f'{split} is not a valid split')
        self.root = root
        self.split = split
        self.transform = transform

        # The official layout is root/split/raw/<scenario_id>/; the av2 tars however extract
        # to root/split/<scenario_id>/. Accept either.
        if raw_dir is None:
            cand_raw = os.path.join(root, split, 'raw')
            raw_dir = cand_raw if os.path.isdir(cand_raw) else os.path.join(root, split)
        self.raw_dir = os.path.expanduser(os.path.normpath(raw_dir))
        if processed_dir is None:
            processed_dir = os.path.join(root, split, 'processed')
        self.processed_dir = os.path.expanduser(os.path.normpath(processed_dir))
        os.makedirs(self.processed_dir, exist_ok=True)

        if os.path.isdir(self.raw_dir):
            self.raw_file_names = sorted(name for name in os.listdir(self.raw_dir)
                                         if os.path.isdir(os.path.join(self.raw_dir, name)))
        else:
            self.raw_file_names = []
        if max_scenarios is not None:
            self.raw_file_names = self.raw_file_names[:max_scenarios]

        self.dim = dim
        self.num_historical_steps = num_historical_steps
        self.num_future_steps = num_future_steps
        self.num_steps = num_historical_steps + num_future_steps
        self.predict_unseen_agents = predict_unseen_agents
        self.vector_repr = vector_repr
        self._agent_types = ['vehicle', 'pedestrian', 'motorcyclist', 'cyclist', 'bus', 'static', 'background',
                             'construction', 'riderless_bicycle', 'unknown']
        self._agent_categories = ['TRACK_FRAGMENT', 'UNSCORED_TRACK', 'SCORED_TRACK', 'FOCAL_TRACK']
        self._polygon_types = ['VEHICLE', 'BIKE', 'BUS', 'PEDESTRIAN']
        self._polygon_is_intersections = [True, False, None]
        self._point_types = ['DASH_SOLID_YELLOW', 'DASH_SOLID_WHITE', 'DASHED_WHITE', 'DASHED_YELLOW',
                             'DOUBLE_SOLID_YELLOW', 'DOUBLE_SOLID_WHITE', 'DOUBLE_DASH_YELLOW', 'DOUBLE_DASH_WHITE',
                             'SOLID_YELLOW', 'SOLID_WHITE', 'SOLID_DASH_WHITE', 'SOLID_DASH_YELLOW', 'SOLID_BLUE',
                             'NONE', 'UNKNOWN', 'CROSSWALK', 'CENTERLINE']
        self._point_sides = ['LEFT', 'RIGHT', 'CENTER']
        self._polygon_to_polygon_types = ['NONE', 'PRED', 'SUCC', 'LEFT', 'RIGHT']

    # ------------------------------------------------------------------ processing
    def _process_one(self, raw_file_name: str) -> Dict:
        """Parse one scenario folder into the nested-dict scene representation the model
        consumes. The folder holds a tracks parquet + an HD-map JSON; the av2 API decodes
        both. Returns a dict with keys 'scenario_id', 'city', 'agent', 'map_polygon',
        'map_point', and the tuple-keyed heterogeneous edges."""
        # agent tracks: one row per (track_id, timestep) with position/heading/velocity/type
        df = pd.read_parquet(os.path.join(self.raw_dir, raw_file_name, f'scenario_{raw_file_name}.parquet'))
        map_dir = Path(self.raw_dir) / raw_file_name
        map_path = map_dir / sorted(map_dir.glob('log_map_archive_*.json'))[0]
        map_data = read_json_file(map_path)
        # lane centerlines as polylines (xyz) keyed by lane-segment id
        centerlines = {lane_segment['id']: Polyline.from_json_data(lane_segment['centerline'])
                       for lane_segment in map_data['lane_segments'].values()}
        map_api = ArgoverseStaticMap.from_json(map_path)  # gives lane segments, crosswalks, connectivity
        data = dict()
        data['scenario_id'] = self.get_scenario_id(df)
        data['city'] = self.get_city(df)
        data['agent'] = self.get_agent_features(df)         # -> dict of [A, 110, ...] agent tensors
        data.update(self.get_map_features(map_api, centerlines))  # -> map_polygon / map_point / edges
        return data

    @staticmethod
    def get_scenario_id(df: pd.DataFrame) -> str:
        return df['scenario_id'].values[0]

    @staticmethod
    def get_city(df: pd.DataFrame) -> str:
        return df['city'].values[0]

    def get_agent_features(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Build per-agent tensors of shape [num_agents, num_steps(=110), ...].
        Key fields: valid_mask (observed?), predict_mask (supervise?), position/heading/
        velocity, type (vehicle/pedestrian/...), category (3 = scored FOCAL agent)."""
        if not self.predict_unseen_agents:  # filter out agents that are unseen during the historical time steps
            historical_df = df[df['timestep'] < self.num_historical_steps]
            agent_ids = list(historical_df['track_id'].unique())
            df = df[df['track_id'].isin(agent_ids)]
        else:
            agent_ids = list(df['track_id'].unique())

        num_agents = len(agent_ids)
        av_idx = agent_ids.index('AV')

        # initialization
        valid_mask = torch.zeros(num_agents, self.num_steps, dtype=torch.bool)
        current_valid_mask = torch.zeros(num_agents, dtype=torch.bool)
        predict_mask = torch.zeros(num_agents, self.num_steps, dtype=torch.bool)
        agent_id: List[Optional[str]] = [None] * num_agents
        agent_type = torch.zeros(num_agents, dtype=torch.uint8)
        agent_category = torch.zeros(num_agents, dtype=torch.uint8)
        position = torch.zeros(num_agents, self.num_steps, self.dim, dtype=torch.float)
        heading = torch.zeros(num_agents, self.num_steps, dtype=torch.float)
        velocity = torch.zeros(num_agents, self.num_steps, self.dim, dtype=torch.float)

        for track_id, track_df in df.groupby('track_id'):
            agent_idx = agent_ids.index(track_id)
            agent_steps = track_df['timestep'].values

            valid_mask[agent_idx, agent_steps] = True
            current_valid_mask[agent_idx] = valid_mask[agent_idx, self.num_historical_steps - 1]
            predict_mask[agent_idx, agent_steps] = True
            if self.vector_repr:  # a time step t is valid only when both t and t-1 are valid
                valid_mask[agent_idx, 1: self.num_historical_steps] = (
                        valid_mask[agent_idx, :self.num_historical_steps - 1] &
                        valid_mask[agent_idx, 1: self.num_historical_steps])
                valid_mask[agent_idx, 0] = False
            predict_mask[agent_idx, :self.num_historical_steps] = False
            if not current_valid_mask[agent_idx]:
                predict_mask[agent_idx, self.num_historical_steps:] = False

            agent_id[agent_idx] = track_id
            agent_type[agent_idx] = self._agent_types.index(track_df['object_type'].values[0])
            agent_category[agent_idx] = track_df['object_category'].values[0]
            position[agent_idx, agent_steps, :2] = torch.from_numpy(np.stack([track_df['position_x'].values,
                                                                              track_df['position_y'].values],
                                                                             axis=-1)).float()
            heading[agent_idx, agent_steps] = torch.from_numpy(track_df['heading'].values).float()
            velocity[agent_idx, agent_steps, :2] = torch.from_numpy(np.stack([track_df['velocity_x'].values,
                                                                              track_df['velocity_y'].values],
                                                                             axis=-1)).float()

        if self.split == 'test':
            predict_mask[current_valid_mask
                         | (agent_category == 2)
                         | (agent_category == 3), self.num_historical_steps:] = True

        return {
            'num_nodes': num_agents,
            'av_index': av_idx,
            'valid_mask': valid_mask,
            'predict_mask': predict_mask,
            'id': agent_id,
            'type': agent_type,
            'category': agent_category,
            'position': position,
            'heading': heading,
            'velocity': velocity,
        }

    def get_map_features(self,
                         map_api: ArgoverseStaticMap,
                         centerlines: Mapping[str, Polyline]) -> Dict[Union[str, Tuple[str, str, str]], Any]:
        """Build the map graph. Each lane segment / crosswalk becomes a 'polygon' node
        (position = first centerline point, orientation, type, is_intersection). Each
        polygon owns 'point' nodes = its left/right boundary + centerline segments (position,
        orientation, magnitude, type, side). Edges: map_point->map_polygon (membership) and
        map_polygon->map_polygon (PRED/SUCC/LEFT/RIGHT lane connectivity). Crosswalks are
        added twice (both traversal directions)."""
        lane_segment_ids = map_api.get_scenario_lane_segment_ids()
        cross_walk_ids = list(map_api.vector_pedestrian_crossings.keys())
        polygon_ids = lane_segment_ids + cross_walk_ids
        num_polygons = len(lane_segment_ids) + len(cross_walk_ids) * 2

        # initialization
        polygon_position = torch.zeros(num_polygons, self.dim, dtype=torch.float)
        polygon_orientation = torch.zeros(num_polygons, dtype=torch.float)
        polygon_height = torch.zeros(num_polygons, dtype=torch.float)
        polygon_type = torch.zeros(num_polygons, dtype=torch.uint8)
        polygon_is_intersection = torch.zeros(num_polygons, dtype=torch.uint8)
        point_position: List[Optional[torch.Tensor]] = [None] * num_polygons
        point_orientation: List[Optional[torch.Tensor]] = [None] * num_polygons
        point_magnitude: List[Optional[torch.Tensor]] = [None] * num_polygons
        point_height: List[Optional[torch.Tensor]] = [None] * num_polygons
        point_type: List[Optional[torch.Tensor]] = [None] * num_polygons
        point_side: List[Optional[torch.Tensor]] = [None] * num_polygons

        for lane_segment in map_api.get_scenario_lane_segments():
            lane_segment_idx = polygon_ids.index(lane_segment.id)
            centerline = torch.from_numpy(centerlines[lane_segment.id].xyz).float()
            polygon_position[lane_segment_idx] = centerline[0, :self.dim]
            polygon_orientation[lane_segment_idx] = torch.atan2(centerline[1, 1] - centerline[0, 1],
                                                                centerline[1, 0] - centerline[0, 0])
            polygon_height[lane_segment_idx] = centerline[1, 2] - centerline[0, 2]
            polygon_type[lane_segment_idx] = self._polygon_types.index(lane_segment.lane_type.value)
            polygon_is_intersection[lane_segment_idx] = self._polygon_is_intersections.index(
                lane_segment.is_intersection)

            left_boundary = torch.from_numpy(lane_segment.left_lane_boundary.xyz).float()
            right_boundary = torch.from_numpy(lane_segment.right_lane_boundary.xyz).float()
            point_position[lane_segment_idx] = torch.cat([left_boundary[:-1, :self.dim],
                                                          right_boundary[:-1, :self.dim],
                                                          centerline[:-1, :self.dim]], dim=0)
            left_vectors = left_boundary[1:] - left_boundary[:-1]
            right_vectors = right_boundary[1:] - right_boundary[:-1]
            center_vectors = centerline[1:] - centerline[:-1]
            point_orientation[lane_segment_idx] = torch.cat([torch.atan2(left_vectors[:, 1], left_vectors[:, 0]),
                                                             torch.atan2(right_vectors[:, 1], right_vectors[:, 0]),
                                                             torch.atan2(center_vectors[:, 1], center_vectors[:, 0])],
                                                            dim=0)
            point_magnitude[lane_segment_idx] = torch.norm(torch.cat([left_vectors[:, :2],
                                                                      right_vectors[:, :2],
                                                                      center_vectors[:, :2]], dim=0), p=2, dim=-1)
            point_height[lane_segment_idx] = torch.cat([left_vectors[:, 2], right_vectors[:, 2], center_vectors[:, 2]],
                                                       dim=0)
            left_type = self._point_types.index(lane_segment.left_mark_type.value)
            right_type = self._point_types.index(lane_segment.right_mark_type.value)
            center_type = self._point_types.index('CENTERLINE')
            point_type[lane_segment_idx] = torch.cat(
                [torch.full((len(left_vectors),), left_type, dtype=torch.uint8),
                 torch.full((len(right_vectors),), right_type, dtype=torch.uint8),
                 torch.full((len(center_vectors),), center_type, dtype=torch.uint8)], dim=0)
            point_side[lane_segment_idx] = torch.cat(
                [torch.full((len(left_vectors),), self._point_sides.index('LEFT'), dtype=torch.uint8),
                 torch.full((len(right_vectors),), self._point_sides.index('RIGHT'), dtype=torch.uint8),
                 torch.full((len(center_vectors),), self._point_sides.index('CENTER'), dtype=torch.uint8)], dim=0)

        for crosswalk in map_api.get_scenario_ped_crossings():
            crosswalk_idx = polygon_ids.index(crosswalk.id)
            edge1 = torch.from_numpy(crosswalk.edge1.xyz).float()
            edge2 = torch.from_numpy(crosswalk.edge2.xyz).float()
            start_position = (edge1[0] + edge2[0]) / 2
            end_position = (edge1[-1] + edge2[-1]) / 2
            polygon_position[crosswalk_idx] = start_position[:self.dim]
            polygon_position[crosswalk_idx + len(cross_walk_ids)] = end_position[:self.dim]
            polygon_orientation[crosswalk_idx] = torch.atan2((end_position - start_position)[1],
                                                             (end_position - start_position)[0])
            polygon_orientation[crosswalk_idx + len(cross_walk_ids)] = torch.atan2((start_position - end_position)[1],
                                                                                   (start_position - end_position)[0])
            polygon_height[crosswalk_idx] = end_position[2] - start_position[2]
            polygon_height[crosswalk_idx + len(cross_walk_ids)] = start_position[2] - end_position[2]
            polygon_type[crosswalk_idx] = self._polygon_types.index('PEDESTRIAN')
            polygon_type[crosswalk_idx + len(cross_walk_ids)] = self._polygon_types.index('PEDESTRIAN')
            polygon_is_intersection[crosswalk_idx] = self._polygon_is_intersections.index(None)
            polygon_is_intersection[crosswalk_idx + len(cross_walk_ids)] = self._polygon_is_intersections.index(None)

            if side_to_directed_lineseg((edge1[0] + edge1[-1]) / 2, start_position, end_position) == 'LEFT':
                left_boundary = edge1
                right_boundary = edge2
            else:
                left_boundary = edge2
                right_boundary = edge1
            num_centerline_points = math.ceil(torch.norm(end_position - start_position, p=2, dim=-1).item() / 2.0) + 1
            centerline = torch.from_numpy(
                compute_midpoint_line(left_ln_boundary=left_boundary.numpy(),
                                      right_ln_boundary=right_boundary.numpy(),
                                      num_interp_pts=int(num_centerline_points))[0]).float()

            point_position[crosswalk_idx] = torch.cat([left_boundary[:-1, :self.dim],
                                                       right_boundary[:-1, :self.dim],
                                                       centerline[:-1, :self.dim]], dim=0)
            point_position[crosswalk_idx + len(cross_walk_ids)] = torch.cat(
                [right_boundary.flip(dims=[0])[:-1, :self.dim],
                 left_boundary.flip(dims=[0])[:-1, :self.dim],
                 centerline.flip(dims=[0])[:-1, :self.dim]], dim=0)
            left_vectors = left_boundary[1:] - left_boundary[:-1]
            right_vectors = right_boundary[1:] - right_boundary[:-1]
            center_vectors = centerline[1:] - centerline[:-1]
            point_orientation[crosswalk_idx] = torch.cat(
                [torch.atan2(left_vectors[:, 1], left_vectors[:, 0]),
                 torch.atan2(right_vectors[:, 1], right_vectors[:, 0]),
                 torch.atan2(center_vectors[:, 1], center_vectors[:, 0])], dim=0)
            point_orientation[crosswalk_idx + len(cross_walk_ids)] = torch.cat(
                [torch.atan2(-right_vectors.flip(dims=[0])[:, 1], -right_vectors.flip(dims=[0])[:, 0]),
                 torch.atan2(-left_vectors.flip(dims=[0])[:, 1], -left_vectors.flip(dims=[0])[:, 0]),
                 torch.atan2(-center_vectors.flip(dims=[0])[:, 1], -center_vectors.flip(dims=[0])[:, 0])], dim=0)
            point_magnitude[crosswalk_idx] = torch.norm(torch.cat([left_vectors[:, :2],
                                                                   right_vectors[:, :2],
                                                                   center_vectors[:, :2]], dim=0), p=2, dim=-1)
            point_magnitude[crosswalk_idx + len(cross_walk_ids)] = torch.norm(
                torch.cat([-right_vectors.flip(dims=[0])[:, :2],
                           -left_vectors.flip(dims=[0])[:, :2],
                           -center_vectors.flip(dims=[0])[:, :2]], dim=0), p=2, dim=-1)
            point_height[crosswalk_idx] = torch.cat([left_vectors[:, 2], right_vectors[:, 2], center_vectors[:, 2]],
                                                    dim=0)
            point_height[crosswalk_idx + len(cross_walk_ids)] = torch.cat(
                [-right_vectors.flip(dims=[0])[:, 2],
                 -left_vectors.flip(dims=[0])[:, 2],
                 -center_vectors.flip(dims=[0])[:, 2]], dim=0)
            crosswalk_type = self._point_types.index('CROSSWALK')
            center_type = self._point_types.index('CENTERLINE')
            point_type[crosswalk_idx] = torch.cat([
                torch.full((len(left_vectors),), crosswalk_type, dtype=torch.uint8),
                torch.full((len(right_vectors),), crosswalk_type, dtype=torch.uint8),
                torch.full((len(center_vectors),), center_type, dtype=torch.uint8)], dim=0)
            point_type[crosswalk_idx + len(cross_walk_ids)] = torch.cat(
                [torch.full((len(right_vectors),), crosswalk_type, dtype=torch.uint8),
                 torch.full((len(left_vectors),), crosswalk_type, dtype=torch.uint8),
                 torch.full((len(center_vectors),), center_type, dtype=torch.uint8)], dim=0)
            point_side[crosswalk_idx] = torch.cat(
                [torch.full((len(left_vectors),), self._point_sides.index('LEFT'), dtype=torch.uint8),
                 torch.full((len(right_vectors),), self._point_sides.index('RIGHT'), dtype=torch.uint8),
                 torch.full((len(center_vectors),), self._point_sides.index('CENTER'), dtype=torch.uint8)], dim=0)
            point_side[crosswalk_idx + len(cross_walk_ids)] = torch.cat(
                [torch.full((len(right_vectors),), self._point_sides.index('LEFT'), dtype=torch.uint8),
                 torch.full((len(left_vectors),), self._point_sides.index('RIGHT'), dtype=torch.uint8),
                 torch.full((len(center_vectors),), self._point_sides.index('CENTER'), dtype=torch.uint8)], dim=0)

        num_points = torch.tensor([point.size(0) for point in point_position], dtype=torch.long)
        point_to_polygon_edge_index = torch.stack(
            [torch.arange(num_points.sum(), dtype=torch.long),
             torch.arange(num_polygons, dtype=torch.long).repeat_interleave(num_points)], dim=0)
        polygon_to_polygon_edge_index = []
        polygon_to_polygon_type = []
        for lane_segment in map_api.get_scenario_lane_segments():
            lane_segment_idx = polygon_ids.index(lane_segment.id)
            pred_inds = []
            for pred in lane_segment.predecessors:
                pred_idx = safe_list_index(polygon_ids, pred)
                if pred_idx is not None:
                    pred_inds.append(pred_idx)
            if len(pred_inds) != 0:
                polygon_to_polygon_edge_index.append(
                    torch.stack([torch.tensor(pred_inds, dtype=torch.long),
                                 torch.full((len(pred_inds),), lane_segment_idx, dtype=torch.long)], dim=0))
                polygon_to_polygon_type.append(
                    torch.full((len(pred_inds),), self._polygon_to_polygon_types.index('PRED'), dtype=torch.uint8))
            succ_inds = []
            for succ in lane_segment.successors:
                succ_idx = safe_list_index(polygon_ids, succ)
                if succ_idx is not None:
                    succ_inds.append(succ_idx)
            if len(succ_inds) != 0:
                polygon_to_polygon_edge_index.append(
                    torch.stack([torch.tensor(succ_inds, dtype=torch.long),
                                 torch.full((len(succ_inds),), lane_segment_idx, dtype=torch.long)], dim=0))
                polygon_to_polygon_type.append(
                    torch.full((len(succ_inds),), self._polygon_to_polygon_types.index('SUCC'), dtype=torch.uint8))
            if lane_segment.left_neighbor_id is not None:
                left_idx = safe_list_index(polygon_ids, lane_segment.left_neighbor_id)
                if left_idx is not None:
                    polygon_to_polygon_edge_index.append(
                        torch.tensor([[left_idx], [lane_segment_idx]], dtype=torch.long))
                    polygon_to_polygon_type.append(
                        torch.tensor([self._polygon_to_polygon_types.index('LEFT')], dtype=torch.uint8))
            if lane_segment.right_neighbor_id is not None:
                right_idx = safe_list_index(polygon_ids, lane_segment.right_neighbor_id)
                if right_idx is not None:
                    polygon_to_polygon_edge_index.append(
                        torch.tensor([[right_idx], [lane_segment_idx]], dtype=torch.long))
                    polygon_to_polygon_type.append(
                        torch.tensor([self._polygon_to_polygon_types.index('RIGHT')], dtype=torch.uint8))
        if len(polygon_to_polygon_edge_index) != 0:
            polygon_to_polygon_edge_index = torch.cat(polygon_to_polygon_edge_index, dim=1)
            polygon_to_polygon_type = torch.cat(polygon_to_polygon_type, dim=0)
        else:
            polygon_to_polygon_edge_index = torch.tensor([[], []], dtype=torch.long)
            polygon_to_polygon_type = torch.tensor([], dtype=torch.uint8)

        map_data = {
            'map_polygon': {},
            'map_point': {},
            ('map_point', 'to', 'map_polygon'): {},
            ('map_polygon', 'to', 'map_polygon'): {},
        }
        map_data['map_polygon']['num_nodes'] = num_polygons
        map_data['map_polygon']['position'] = polygon_position
        map_data['map_polygon']['orientation'] = polygon_orientation
        if self.dim == 3:
            map_data['map_polygon']['height'] = polygon_height
        map_data['map_polygon']['type'] = polygon_type
        map_data['map_polygon']['is_intersection'] = polygon_is_intersection
        if len(num_points) == 0:
            map_data['map_point']['num_nodes'] = 0
            map_data['map_point']['position'] = torch.tensor([], dtype=torch.float)
            map_data['map_point']['orientation'] = torch.tensor([], dtype=torch.float)
            map_data['map_point']['magnitude'] = torch.tensor([], dtype=torch.float)
            if self.dim == 3:
                map_data['map_point']['height'] = torch.tensor([], dtype=torch.float)
            map_data['map_point']['type'] = torch.tensor([], dtype=torch.uint8)
            map_data['map_point']['side'] = torch.tensor([], dtype=torch.uint8)
        else:
            map_data['map_point']['num_nodes'] = num_points.sum().item()
            map_data['map_point']['position'] = torch.cat(point_position, dim=0)
            map_data['map_point']['orientation'] = torch.cat(point_orientation, dim=0)
            map_data['map_point']['magnitude'] = torch.cat(point_magnitude, dim=0)
            if self.dim == 3:
                map_data['map_point']['height'] = torch.cat(point_height, dim=0)
            map_data['map_point']['type'] = torch.cat(point_type, dim=0)
            map_data['map_point']['side'] = torch.cat(point_side, dim=0)
        map_data['map_point', 'to', 'map_polygon']['edge_index'] = point_to_polygon_edge_index
        map_data['map_polygon', 'to', 'map_polygon']['edge_index'] = polygon_to_polygon_edge_index
        map_data['map_polygon', 'to', 'map_polygon']['type'] = polygon_to_polygon_type

        return map_data

    # ------------------------------------------------------------------ Dataset API
    def __len__(self) -> int:
        return len(self.raw_file_names)

    def __getitem__(self, idx: int) -> Dict:
        raw_file_name = self.raw_file_names[idx]
        processed_path = os.path.join(self.processed_dir, f'{raw_file_name}.pkl')
        if os.path.isfile(processed_path):
            with open(processed_path, 'rb') as handle:
                data = pickle.load(handle)
        else:
            data = self._process_one(raw_file_name)
            with open(processed_path, 'wb') as handle:
                pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        if self.transform is not None:
            data = self.transform(data)
        return data
