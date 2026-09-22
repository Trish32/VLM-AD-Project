"""Modular end-to-end AD pipeline: dense + object perception, uncertainty, safety.

See README.md for the stage diagram and frame contracts.
"""

from .freespace import (FREE_CLASS, GROUND_CLASSES, OCC_CLASSES, TRAVERSABLE_CLASSES,
                        FreeSpace, FreeSpaceExtractor, GridConfig)
from .pipeline import (E2EPipeline, PipelineOutput, lidar_boxes_to_agents,
                       to_controller_waypoints)
from .closed_loop import (ClosedLoopRunner, GTWorldModel, LoopConfig,
                          constant_velocity_planner,
                          diffusiondrive_anchor_planner)
from .metrics import StepRecord, evaluate, format_report, obb_overlap
from .safety_filter import (CandidateVerdict, FeasibilityLimits, FilterResult,
                            SafetyFilter)
from .scene import (Agent, EgoState, SceneRepresentation, TrajectoryDistribution,
                    ego_footprint_corners, yaw_from_waypoints)
from .uncertainty import (RiskModel, RiskReport, TrackCovarianceTracker,
                          constant_velocity_prediction)

__all__ = [
    "Agent", "EgoState", "SceneRepresentation", "TrajectoryDistribution",
    "ego_footprint_corners", "yaw_from_waypoints",
    "FreeSpace", "FreeSpaceExtractor", "GridConfig",
    "OCC_CLASSES", "FREE_CLASS", "TRAVERSABLE_CLASSES", "GROUND_CLASSES",
    "TrackCovarianceTracker", "RiskModel", "RiskReport",
    "constant_velocity_prediction",
    "SafetyFilter", "FeasibilityLimits", "CandidateVerdict", "FilterResult",
    "E2EPipeline", "PipelineOutput", "lidar_boxes_to_agents",
    "to_controller_waypoints",
    "ClosedLoopRunner", "GTWorldModel", "LoopConfig",
    "constant_velocity_planner", "diffusiondrive_anchor_planner",
    "StepRecord", "evaluate", "format_report", "obb_overlap",
]
