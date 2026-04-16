import copy
from typing import List

import numpy as np
import numpy.typing as npt
from shapely import creation

from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.actor_state.tracked_objects_types import AGENT_TYPES
from nuplan.common.maps.maps_datatypes import SemanticMapLayer
from nuplan.planning.metrics.utils.collision_utils import CollisionType
from nuplan.planning.simulation.observation.idm.utils import is_agent_ahead, is_agent_behind

from navsim.planning.simulation.planner.pdm_planner.observation.pdm_observation import PDMObservation
from navsim.planning.simulation.planner.pdm_planner.observation.pdm_occupancy_map import PDMDrivableMap
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_comfort_metrics import ego_is_comfortable
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import (
    PDMScorer as BasePDMScorer,
    PDMScorerConfig,
)
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer_utils import get_collision_type
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import (
    BBCoordsIndex,
    EgoAreaIndex,
    MultiMetricIndex,
    StateIndex,
    WeightedMetricIndex,
)
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_path import PDMPath


class PDMScorer(BasePDMScorer):
    def score_proposals(
        self,
        states: npt.NDArray[np.float64],
        observation: PDMObservation,
        centerline: PDMPath,
        route_lane_ids: List[str],
        drivable_area_map: PDMDrivableMap,
        pdm_progress: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        self._reset(states, observation, centerline, route_lane_ids, drivable_area_map)
        self._calculate_ego_area()
        self._calculate_no_at_fault_collision()
        self._calculate_drivable_area_compliance()
        self._calculate_driving_direction_compliance()
        self._calculate_progress()
        try:
            self._calculate_ttc()
        except Exception:
            self._weighted_metrics[WeightedMetricIndex.TTC] = 2.0
        self._calculate_is_comfortable()
        self.pdm_progress = pdm_progress
        return self._aggregate_scores()

    def _aggregate_scores(self) -> npt.NDArray[np.float64]:
        multiplicate_metric_scores = self._multi_metrics.prod(axis=0)
        raw_progress = self._progress_raw * multiplicate_metric_scores
        max_raw_progress = np.maximum(raw_progress, self.pdm_progress)

        fast_mask = max_raw_progress > self._config.progress_distance_threshold
        normalized_progress = np.ones([len(raw_progress)], dtype=np.float64)
        normalized_progress[fast_mask] = raw_progress[fast_mask] / max_raw_progress[fast_mask]
        normalized_progress[(~fast_mask) & (multiplicate_metric_scores == 0)] = 0
        self._weighted_metrics[WeightedMetricIndex.PROGRESS] = normalized_progress

        weighted_metrics_array = self._config.weighted_metrics_array
        weighted_metric_scores = (self._weighted_metrics * weighted_metrics_array[..., None]).sum(axis=0)
        weighted_metric_scores /= weighted_metrics_array.sum()

        return self._multi_metrics.prod(axis=0) * weighted_metric_scores

    def _calculate_no_at_fault_collision(self) -> None:
        no_collision_scores = np.ones(self._num_proposals, dtype=np.float64)

        self.proposal_collided_track_ids = {
            proposal_idx: copy.deepcopy(self._observation.collided_track_ids)
            for proposal_idx in range(self._num_proposals)
        }
        self.proposal_fault_collided_track_ids = {
            proposal_idx: copy.deepcopy(self._observation.collided_track_ids)
            for proposal_idx in range(self._num_proposals)
        }

        for time_idx in range(self.proposal_sampling.num_poses + 1):
            ego_polygons = self._ego_polygons[:, time_idx]
            intersecting = self._observation[time_idx].query(ego_polygons, predicate="intersects")

            if len(intersecting) == 0:
                continue

            for proposal_idx, geometry_idx in zip(intersecting[0], intersecting[1]):
                token = self._observation[time_idx].tokens[geometry_idx]
                if (self._observation.red_light_token in token) or (
                    token in self.proposal_collided_track_ids[proposal_idx]
                ):
                    continue

                ego_in_multiple_lanes_or_nondrivable_area = (
                    self._ego_areas[proposal_idx, time_idx, EgoAreaIndex.MULTIPLE_LANES]
                    or self._ego_areas[proposal_idx, time_idx, EgoAreaIndex.NON_DRIVABLE_AREA]
                )

                tracked_object = self._observation.unique_objects[token]
                collision_type: CollisionType = get_collision_type(
                    self._states[proposal_idx, time_idx],
                    self._ego_polygons[proposal_idx, time_idx],
                    tracked_object,
                    self._observation[time_idx][token],
                )
                collisions_at_stopped_track_or_active_front = collision_type in [
                    CollisionType.ACTIVE_FRONT_COLLISION,
                    CollisionType.STOPPED_TRACK_COLLISION,
                ]
                collision_at_lateral = collision_type == CollisionType.ACTIVE_LATERAL_COLLISION

                if collisions_at_stopped_track_or_active_front or (
                    ego_in_multiple_lanes_or_nondrivable_area and collision_at_lateral
                ):
                    no_at_fault_collision_score = 0.0 if tracked_object.tracked_object_type in AGENT_TYPES else 0.5
                    no_collision_scores[proposal_idx] = np.minimum(
                        no_collision_scores[proposal_idx], no_at_fault_collision_score
                    )
                    self._collision_time_idcs[proposal_idx] = min(time_idx, self._collision_time_idcs[proposal_idx])
                    self.proposal_fault_collided_track_ids[proposal_idx].append(token)
                else:
                    self.proposal_collided_track_ids[proposal_idx].append(token)

        self._multi_metrics[MultiMetricIndex.NO_COLLISION] = no_collision_scores

    def _calculate_drivable_area_compliance(self) -> None:
        drivable_area_compliance_scores = np.ones(self._num_proposals, dtype=np.float64)
        self.off_road = self._ego_areas[:, :, EgoAreaIndex.NON_DRIVABLE_AREA]
        off_road_mask = self.off_road.any(axis=-1)
        drivable_area_compliance_scores[off_road_mask] = 0.0
        self._multi_metrics[MultiMetricIndex.DRIVABLE_AREA] = drivable_area_compliance_scores

    def _calculate_ttc(self):
        ttc_scores = np.ones(self._num_proposals, dtype=np.float64)
        self.temp_collided_track_ids = {
            proposal_idx: copy.deepcopy(self._observation.collided_track_ids)
            for proposal_idx in range(self._num_proposals)
        }
        self.ttc_collided_track_ids = {
            proposal_idx: copy.deepcopy(self._observation.collided_track_ids)
            for proposal_idx in range(self._num_proposals)
        }

        future_time_idcs = np.arange(0, 10, 3)
        n_future_steps = len(future_time_idcs)

        coords_exterior = self._ego_coords.copy()
        coords_exterior[:, :, BBCoordsIndex.CENTER, :] = coords_exterior[:, :, BBCoordsIndex.FRONT_LEFT, :]
        coords_exterior_time_steps = np.repeat(coords_exterior[:, :, None], n_future_steps, axis=2)

        speeds = np.hypot(
            self._states[..., StateIndex.VELOCITY_X],
            self._states[..., StateIndex.VELOCITY_Y],
        )

        dxy_per_s = np.stack(
            [
                np.cos(self._states[..., StateIndex.HEADING]) * speeds,
                np.sin(self._states[..., StateIndex.HEADING]) * speeds,
            ],
            axis=-1,
        )

        for idx, future_time_idx in enumerate(future_time_idcs):
            delta_t = float(future_time_idx) * self.proposal_sampling.interval_length
            coords_exterior_time_steps[:, :, idx] = coords_exterior_time_steps[:, :, idx] + dxy_per_s[:, :, None] * delta_t

        polygons = creation.polygons(coords_exterior_time_steps)

        for time_idx in range(self.proposal_sampling.num_poses + 1):
            for step_idx, future_time_idx in enumerate(future_time_idcs):
                current_time_idx = time_idx + future_time_idx
                polygons_at_time_step = polygons[:, time_idx, step_idx]
                intersecting = self._observation[current_time_idx].query(polygons_at_time_step, predicate="intersects")

                if len(intersecting) == 0:
                    continue

                for proposal_idx, geometry_idx in zip(intersecting[0], intersecting[1]):
                    token = self._observation[current_time_idx].tokens[geometry_idx]
                    if (
                        (self._observation.red_light_token in token)
                        or (token in self.temp_collided_track_ids[proposal_idx])
                        or (speeds[proposal_idx, time_idx] < self._config.stopped_speed_threshold)
                    ):
                        continue

                    ego_in_multiple_lanes_or_nondrivable_area = (
                        self._ego_areas[proposal_idx, time_idx, EgoAreaIndex.MULTIPLE_LANES]
                        or self._ego_areas[proposal_idx, time_idx, EgoAreaIndex.NON_DRIVABLE_AREA]
                    )
                    ego_rear_axle = StateSE2(*self._states[proposal_idx, time_idx, StateIndex.STATE_SE2])

                    centroid = self._observation[current_time_idx][token].centroid
                    track_heading = self._observation.unique_objects[token].box.center.heading
                    track_state = StateSE2(centroid.x, centroid.y, track_heading)

                    if is_agent_ahead(ego_rear_axle, track_state) or (
                        (
                            ego_in_multiple_lanes_or_nondrivable_area
                            or self._drivable_area_map.is_in_layer(ego_rear_axle.point, layer=SemanticMapLayer.INTERSECTION)
                        )
                        and not is_agent_behind(ego_rear_axle, track_state)
                    ):
                        ttc_scores[proposal_idx] = np.minimum(ttc_scores[proposal_idx], 0.0)
                        self._ttc_time_idcs[proposal_idx] = min(time_idx, self._ttc_time_idcs[proposal_idx])
                        self.ttc_collided_track_ids[proposal_idx].append(token)
                    else:
                        self.temp_collided_track_ids[proposal_idx].append(token)

        self._weighted_metrics[WeightedMetricIndex.TTC] = ttc_scores

    def _calculate_is_comfortable(self) -> None:
        time_point_s = (
            np.arange(0, self.proposal_sampling.num_poses + 1).astype(np.float64)
            * self.proposal_sampling.interval_length
        )
        is_comfortable = ego_is_comfortable(self._states, time_point_s)
        self._weighted_metrics[WeightedMetricIndex.COMFORTABLE] = np.all(is_comfortable, axis=-1)

