# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Interaction-field challenge API for SHOW3D.

Dataset-generic path resolution and pose JSON parsing live in
``show3d.dataset``. This module only defines the
challenge-specific sample manifest, directed-field labels, submission schema,
and local evaluator.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np

from ..dataset import (
    array_from_json,
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_HAND_POSE_VERSION,
    default_object_mesh_provider,
    DEFAULT_OBJECT_POSE_VERSION,
    DEFAULT_VIDEO_FPS,
    FloatArray,
    HandPoseFrame,
    object_alias_from_scene_id,
    Show3DDataset,
    Show3DFrameData,
    Show3DFrameRef,
    ViewFrame,
)


DEFAULT_SAMPLING_FPS: float = 10.0

# Hands are 21-joint UmeTrack skeletons, so each hand-anchored field is a fixed
# ``(NUM_HAND_LANDMARKS, 3)`` target regardless of the object.
NUM_HAND_LANDMARKS: int = 21

LEFT_TO_OBJECT: str = "left_to_object"
RIGHT_TO_OBJECT: str = "right_to_object"
# Hand-anchored fields only: one 3D vector from each (fixed, 21-joint) hand
# landmark to the nearest object surface point. The object->hand directions are
# intentionally excluded so the target is fixed-size per image (the object vertex
# count varies across objects).
FIELD_NAMES: tuple[str, ...] = (
    LEFT_TO_OBJECT,
    RIGHT_TO_OBJECT,
)


@dataclass(frozen=True)
class InteractionFieldSample:
    """One challenge sample: a single frame in one SHOW3D scene."""

    sample_id: str
    subject_id: str
    scene_id: str
    frame_index: int
    sampling_fps: float = DEFAULT_SAMPLING_FPS
    video_fps: float = DEFAULT_VIDEO_FPS
    object_alias: str | None = None
    has_left_hand: bool = False
    has_right_hand: bool = False

    @classmethod
    def from_json(cls, row: Mapping[str, object]) -> "InteractionFieldSample":
        sample_id = _required_str(row, "sample_id")
        subject_id = _required_str(row, "subject_id")
        scene_id = _required_str(row, "scene_id")
        return cls(
            sample_id=sample_id,
            subject_id=subject_id,
            scene_id=scene_id,
            frame_index=_required_int(row, "frame_index"),
            sampling_fps=_optional_float(row, "sampling_fps", DEFAULT_SAMPLING_FPS),
            video_fps=_optional_float(row, "video_fps", DEFAULT_VIDEO_FPS),
            object_alias=_optional_str(row, "object_alias")
            or object_alias_from_scene_id(scene_id),
            has_left_hand=_optional_bool(row, "has_left_hand", False),
            has_right_hand=_optional_bool(row, "has_right_hand", False),
        )

    def to_frame_ref(self) -> Show3DFrameRef:
        return Show3DFrameRef(
            subject_id=self.subject_id,
            scene_id=self.scene_id,
            frame_index=self.frame_index,
            video_fps=self.video_fps,
            object_alias=self.object_alias,
        )

    def to_json(self) -> dict[str, object]:
        row: dict[str, object] = {
            "sample_id": self.sample_id,
            "subject_id": self.subject_id,
            "scene_id": self.scene_id,
            "frame_index": self.frame_index,
            "sampling_fps": self.sampling_fps,
            "video_fps": self.video_fps,
            "has_left_hand": self.has_left_hand,
            "has_right_hand": self.has_right_hand,
        }
        if self.object_alias is not None:
            row["object_alias"] = self.object_alias
        return row


@dataclass(frozen=True)
class InteractionFieldLabels:
    """Hand-anchored interaction-field targets for one frame.

    Each field is ``(21, 3)`` -- one vector from each hand joint to the nearest
    object surface point -- or ``None`` when that hand is invalid under the
    configured confidence threshold.
    """

    left_to_object: FloatArray | None = None
    right_to_object: FloatArray | None = None

    @property
    def is_valid(self) -> bool:
        """True when at least one hand yields a target (object pose confident and
        that hand valid under the confidence threshold)."""
        return self.left_to_object is not None or self.right_to_object is not None

    def get(self, field_name: str) -> FloatArray | None:
        if field_name == LEFT_TO_OBJECT:
            return self.left_to_object
        if field_name == RIGHT_TO_OBJECT:
            return self.right_to_object
        raise ValueError(f"Unknown field name: {field_name}")

    def to_json(self) -> dict[str, object]:
        out: dict[str, object] = {}
        for field_name in FIELD_NAMES:
            value = self.get(field_name)
            out[field_name] = None if value is None else value.tolist()
        return out


@dataclass(frozen=True)
class InteractionFieldExample:
    """Challenge dataloader item."""

    sample: InteractionFieldSample
    frame_data: Show3DFrameData
    labels: InteractionFieldLabels | None

    @property
    def views(self) -> dict[str, ViewFrame]:
        return self.frame_data.views

    @property
    def headset_tracking_valid(self) -> bool:
        """False when headset tracking failed on this frame (synthesized pose)."""
        return self.frame_data.headset_tracking_valid

    @property
    def is_valid(self) -> bool:
        """True when this frame is usable: headset tracking succeeded AND the
        object pose and at least one hand are valid (a target exists)."""
        return (
            self.frame_data.headset_tracking_valid
            and self.labels is not None
            and self.labels.is_valid
        )

    @property
    def object_pose_path(self) -> Path:
        return self.frame_data.object_pose_path

    @property
    def hand_pose_path(self) -> Path:
        return self.frame_data.hand_pose_path


@dataclass(frozen=True)
class PredictionRecord:
    """One submission row."""

    sample_id: str
    fields: dict[str, FloatArray]

    @classmethod
    def from_json(cls, row: Mapping[str, object]) -> "PredictionRecord":
        sample_id = _required_str(row, "sample_id")
        fields: dict[str, FloatArray] = {}
        for field_name in FIELD_NAMES:
            value = row.get(field_name)
            if value is not None:
                fields[field_name] = array_from_json(value, columns=3)
        return cls(sample_id=sample_id, fields=fields)

    def to_json(self) -> dict[str, object]:
        row: dict[str, object] = {"sample_id": self.sample_id}
        for field_name, value in self.fields.items():
            if field_name not in FIELD_NAMES:
                raise ValueError(f"Unknown field name: {field_name}")
            row[field_name] = value.tolist()
        return row


@dataclass(frozen=True)
class LabelRecord:
    """One reference-label row for local or hosted evaluation."""

    sample_id: str
    labels: InteractionFieldLabels

    @classmethod
    def from_json(cls, row: Mapping[str, object]) -> "LabelRecord":
        return cls(
            sample_id=_required_str(row, "sample_id"),
            labels=_labels_from_json(row),
        )

    def to_json(self) -> dict[str, object]:
        row: dict[str, object] = {"sample_id": self.sample_id}
        row.update(self.labels.to_json())
        return row


@dataclass(frozen=True)
class FieldMetric:
    """Aggregated metric for one directed field."""

    num_samples: int
    num_points: int
    missing_predictions: int
    ade_mm: float | None
    accuracy_by_threshold_mm: dict[float, float]


@dataclass(frozen=True)
class EvaluationResult:
    """Local evaluator output."""

    fields: dict[str, FieldMetric]

    @property
    def mean_ade_mm(self) -> float | None:
        values = [
            metric.ade_mm
            for metric in self.fields.values()
            if metric.ade_mm is not None
        ]
        if not values:
            return None
        return float(sum(values) / len(values))


class Show3DInteractionFieldDataset:
    """Challenge dataset over a local SHOW3D mirror and interaction manifest."""

    def __init__(
        self,
        root: str | Path,
        manifest_path: str | Path,
        *,
        hand_pose_version: str = DEFAULT_HAND_POSE_VERSION,
        object_pose_version: str = DEFAULT_OBJECT_POSE_VERSION,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        load_labels: bool = True,
        object_mesh_provider: Callable[[str], FloatArray | None] | None = None,
        multiview: bool = True,
        decode_images: bool = False,
    ) -> None:
        self.samples: list[InteractionFieldSample] = read_manifest_jsonl(manifest_path)
        self.confidence_threshold: float = confidence_threshold
        self.load_labels: bool = load_labels
        # Maps an object alias -> its canonical mesh vertices (object frame, mm).
        # Defaults to the meshes bundled in the repo, so labels build with no
        # external assets; pass a custom provider to override.
        if object_mesh_provider is None and load_labels:
            object_mesh_provider = default_object_mesh_provider()
        self._object_mesh_provider: Callable[[str], FloatArray | None] | None = (
            object_mesh_provider
        )
        self._dataset: Show3DDataset = Show3DDataset(
            root,
            [sample.to_frame_ref() for sample in self.samples],
            hand_pose_version=hand_pose_version,
            object_pose_version=object_pose_version,
            load_poses=load_labels,
            multiview=multiview,
            decode_images=decode_images,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> InteractionFieldExample:
        sample = self.samples[index]
        frame_data = self._dataset[index]
        labels: InteractionFieldLabels | None = None
        # Skip targets on frames where headset tracking failed -- their world-space
        # poses are unreliable, so callers cannot train on them by accident.
        if self.load_labels and frame_data.headset_tracking_valid:
            object_pose = frame_data.object_pose
            object_vertices_world_mm: FloatArray | None = None
            object_confidence = 0.0
            if object_pose is not None:
                object_confidence = object_pose.confidence
                if self._object_mesh_provider is not None:
                    alias = sample.object_alias or object_alias_from_scene_id(
                        sample.scene_id
                    )
                    mesh = self._object_mesh_provider(alias)
                    if mesh is not None:
                        object_vertices_world_mm = object_pose.pose_vertices(mesh)
            labels = build_interaction_labels(
                object_vertices_world_mm=object_vertices_world_mm,
                object_confidence=object_confidence,
                left_hand=frame_data.left_hand,
                right_hand=frame_data.right_hand,
                confidence_threshold=self.confidence_threshold,
            )
        return InteractionFieldExample(
            sample=sample, frame_data=frame_data, labels=labels
        )


def make_sample_id(subject_id: str, scene_id: str, frame_index: int) -> str:
    return f"{subject_id}/{scene_id}:{frame_index:06d}"


def sampled_frame_indices(
    num_frames: int,
    sampling_fps: float,
    *,
    video_fps: float = DEFAULT_VIDEO_FPS,
    start_frame: int = 0,
) -> list[int]:
    """Deterministically subsample source-frame indices for the challenge."""
    if num_frames <= 0:
        return []
    if sampling_fps <= 0.0:
        raise ValueError("sampling_fps must be positive")
    if video_fps <= 0.0:
        raise ValueError("video_fps must be positive")
    if start_frame < 0:
        raise ValueError("start_frame must be non-negative")
    if start_frame >= num_frames:
        return []

    interval = video_fps / sampling_fps
    rounded_stride = round(interval)
    if rounded_stride > 0 and math.isclose(
        interval, float(rounded_stride), rel_tol=0.0, abs_tol=1e-6
    ):
        return list(range(start_frame, num_frames, int(rounded_stride)))

    out: list[int] = []
    seen: set[int] = set()
    k = 0
    while True:
        frame = int(round(start_frame + k * interval))
        if frame >= num_frames:
            break
        if frame not in seen:
            out.append(frame)
            seen.add(frame)
        k += 1
    return out


def make_manifest_rows(
    subject_id: str,
    scene_id: str,
    num_frames: int,
    *,
    sampling_fps: float = DEFAULT_SAMPLING_FPS,
    video_fps: float = DEFAULT_VIDEO_FPS,
    has_left_hand: bool = False,
    has_right_hand: bool = False,
) -> list[InteractionFieldSample]:
    object_alias = object_alias_from_scene_id(scene_id)
    return [
        InteractionFieldSample(
            sample_id=make_sample_id(subject_id, scene_id, frame_index),
            subject_id=subject_id,
            scene_id=scene_id,
            frame_index=frame_index,
            sampling_fps=sampling_fps,
            video_fps=video_fps,
            object_alias=object_alias,
            has_left_hand=has_left_hand,
            has_right_hand=has_right_hand,
        )
        for frame_index in sampled_frame_indices(
            num_frames, sampling_fps, video_fps=video_fps
        )
    ]


def read_manifest_jsonl(path: str | Path) -> list[InteractionFieldSample]:
    rows: list[InteractionFieldSample] = []
    with Path(path).open() as f:
        for line_number, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            raw = json.loads(stripped)
            if not isinstance(raw, dict):
                raise ValueError(f"Manifest line {line_number} is not a JSON object")
            rows.append(InteractionFieldSample.from_json(cast(dict[str, object], raw)))
    return rows


def write_manifest_jsonl(
    path: str | Path, samples: Iterable[InteractionFieldSample]
) -> None:
    with Path(path).open("w") as f:
        for sample in samples:
            f.write(json.dumps(sample.to_json(), sort_keys=True))
            f.write("\n")


def build_interaction_labels(
    *,
    object_vertices_world_mm: FloatArray | None,
    object_confidence: float,
    left_hand: HandPoseFrame | None,
    right_hand: HandPoseFrame | None,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> InteractionFieldLabels:
    """Hand-anchored nearest-neighbor field: for each hand joint, the vector to
    the nearest object surface point (fixed-size, 21x3 per hand).

    ``object_vertices_world_mm`` is the object's mesh posed into world space
    (e.g. via ``ObjectPoseFrame.pose_vertices``). Returns empty labels when the
    object pose is missing or at/below ``confidence_threshold``.
    """
    if (
        object_vertices_world_mm is None
        or object_confidence <= confidence_threshold
        or object_vertices_world_mm.size == 0
    ):
        return InteractionFieldLabels()

    object_points = object_vertices_world_mm
    left_to_object: FloatArray | None = None
    right_to_object: FloatArray | None = None

    if _hand_is_valid(left_hand, confidence_threshold):
        left_points = cast(HandPoseFrame, left_hand).landmarks_world_mm
        assert left_points is not None
        left_to_object = nearest_neighbor_vectors(
            source_points=left_points,
            target_points=object_points,
        )

    if _hand_is_valid(right_hand, confidence_threshold):
        right_points = cast(HandPoseFrame, right_hand).landmarks_world_mm
        assert right_points is not None
        right_to_object = nearest_neighbor_vectors(
            source_points=right_points,
            target_points=object_points,
        )

    return InteractionFieldLabels(
        left_to_object=left_to_object,
        right_to_object=right_to_object,
    )


def nearest_neighbor_vectors(
    *,
    source_points: FloatArray,
    target_points: FloatArray,
) -> FloatArray:
    """Vector from each source point to its nearest target point."""
    _validate_points(source_points, "source_points")
    _validate_points(target_points, "target_points")
    if source_points.shape[0] == 0 or target_points.shape[0] == 0:
        return np.empty((source_points.shape[0], 3), dtype=np.float64)
    deltas = target_points[None, :, :] - source_points[:, None, :]
    squared_distances = np.sum(deltas * deltas, axis=2)
    nearest = np.argmin(squared_distances, axis=1)
    return target_points[nearest] - source_points


def read_submission_jsonl(path: str | Path) -> dict[str, PredictionRecord]:
    records: dict[str, PredictionRecord] = {}
    with Path(path).open() as f:
        for line_number, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            raw = json.loads(stripped)
            if not isinstance(raw, dict):
                raise ValueError(f"Submission line {line_number} is not a JSON object")
            record = PredictionRecord.from_json(cast(dict[str, object], raw))
            if record.sample_id in records:
                raise ValueError(
                    f"Duplicate prediction for sample_id={record.sample_id}"
                )
            records[record.sample_id] = record
    return records


def write_submission_jsonl(
    path: str | Path, records: Iterable[PredictionRecord]
) -> None:
    with Path(path).open("w") as f:
        for record in records:
            f.write(json.dumps(record.to_json(), sort_keys=True))
            f.write("\n")


def read_label_jsonl(path: str | Path) -> list[LabelRecord]:
    records: list[LabelRecord] = []
    with Path(path).open() as f:
        for line_number, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            raw = json.loads(stripped)
            if not isinstance(raw, dict):
                raise ValueError(f"Label line {line_number} is not a JSON object")
            records.append(LabelRecord.from_json(cast(dict[str, object], raw)))
    return records


def write_label_jsonl(path: str | Path, records: Iterable[LabelRecord]) -> None:
    with Path(path).open("w") as f:
        for record in records:
            f.write(json.dumps(record.to_json(), sort_keys=True))
            f.write("\n")


def evaluate_submission_jsonl(
    dataset: Show3DInteractionFieldDataset,
    submission_path: str | Path,
    *,
    accuracy_thresholds_mm: tuple[float, ...] = (10.0, 50.0, 100.0),
) -> EvaluationResult:
    predictions = read_submission_jsonl(submission_path)
    return evaluate_predictions(
        dataset,
        predictions,
        accuracy_thresholds_mm=accuracy_thresholds_mm,
    )


def evaluate_label_jsonl(
    reference_path: str | Path,
    submission_path: str | Path,
    *,
    accuracy_thresholds_mm: tuple[float, ...] = (10.0, 50.0, 100.0),
) -> EvaluationResult:
    return evaluate_prediction_records(
        read_label_jsonl(reference_path),
        read_submission_jsonl(submission_path),
        accuracy_thresholds_mm=accuracy_thresholds_mm,
    )


def evaluate_predictions(
    dataset: Show3DInteractionFieldDataset,
    predictions: Mapping[str, PredictionRecord],
    *,
    accuracy_thresholds_mm: tuple[float, ...] = (10.0, 50.0, 100.0),
) -> EvaluationResult:
    references: list[LabelRecord] = []
    for index in range(len(dataset)):
        example = dataset[index]
        if example.labels is None:
            if not dataset.load_labels:
                raise ValueError("Dataset must be constructed with load_labels=True")
            # Tracking-invalid frame: no reference target to score against.
            continue
        references.append(
            LabelRecord(sample_id=example.sample.sample_id, labels=example.labels)
        )
    return evaluate_prediction_records(
        references,
        predictions,
        accuracy_thresholds_mm=accuracy_thresholds_mm,
    )


def evaluate_prediction_records(
    references: Iterable[LabelRecord],
    predictions: Mapping[str, PredictionRecord],
    *,
    accuracy_thresholds_mm: tuple[float, ...] = (10.0, 50.0, 100.0),
) -> EvaluationResult:
    accumulators = {
        field_name: _MetricAccumulator(accuracy_thresholds_mm)
        for field_name in FIELD_NAMES
    }
    seen_sample_ids: set[str] = set()
    for reference in references:
        if reference.sample_id in seen_sample_ids:
            raise ValueError(f"Duplicate label for sample_id={reference.sample_id}")
        seen_sample_ids.add(reference.sample_id)
        prediction = predictions.get(reference.sample_id)
        for field_name in FIELD_NAMES:
            target = reference.labels.get(field_name)
            if target is None:
                continue
            if prediction is None or field_name not in prediction.fields:
                accumulators[field_name].add_missing()
                continue
            accumulators[field_name].add_prediction(
                prediction.fields[field_name],
                target,
            )
    return EvaluationResult(
        fields={
            field_name: accumulator.to_metric()
            for field_name, accumulator in accumulators.items()
        }
    )


class _MetricAccumulator:
    def __init__(self, accuracy_thresholds_mm: tuple[float, ...]) -> None:
        self._accuracy_thresholds_mm: tuple[float, ...] = accuracy_thresholds_mm
        self._num_samples: int = 0
        self._num_points: int = 0
        self._missing_predictions: int = 0
        self._sum_error_mm: float = 0.0
        self._correct_by_threshold: dict[float, int] = dict.fromkeys(
            accuracy_thresholds_mm, 0
        )

    def add_missing(self) -> None:
        self._num_samples += 1
        self._missing_predictions += 1

    def add_prediction(self, prediction: FloatArray, target: FloatArray) -> None:
        if prediction.shape != target.shape:
            raise ValueError(
                f"Prediction shape {prediction.shape} does not match target "
                f"shape {target.shape}"
            )
        errors = np.linalg.norm(prediction - target, axis=1)
        self._num_samples += 1
        self._num_points += int(errors.shape[0])
        self._sum_error_mm += float(np.sum(errors))
        for threshold in self._accuracy_thresholds_mm:
            self._correct_by_threshold[threshold] += int(np.sum(errors <= threshold))

    def to_metric(self) -> FieldMetric:
        ade_mm = self._sum_error_mm / self._num_points if self._num_points > 0 else None
        accuracy_by_threshold = {
            threshold: (
                self._correct_by_threshold[threshold] / self._num_points
                if self._num_points > 0
                else 0.0
            )
            for threshold in self._accuracy_thresholds_mm
        }
        return FieldMetric(
            num_samples=self._num_samples,
            num_points=self._num_points,
            missing_predictions=self._missing_predictions,
            ade_mm=ade_mm,
            accuracy_by_threshold_mm=accuracy_by_threshold,
        )


def _labels_from_json(row: Mapping[str, object]) -> InteractionFieldLabels:
    values: dict[str, FloatArray | None] = {}
    for field_name in FIELD_NAMES:
        value = row.get(field_name)
        values[field_name] = (
            array_from_json(value, columns=3) if value is not None else None
        )
    return InteractionFieldLabels(
        left_to_object=values[LEFT_TO_OBJECT],
        right_to_object=values[RIGHT_TO_OBJECT],
    )


def _hand_is_valid(
    hand_pose: HandPoseFrame | None, confidence_threshold: float
) -> bool:
    return (
        hand_pose is not None
        and hand_pose.confidence > confidence_threshold
        and hand_pose.landmarks_world_mm is not None
        and hand_pose.landmarks_world_mm.size > 0
    )


def _validate_points(points: FloatArray, name: str) -> None:
    if len(points.shape) != 2 or points.shape[1] != 3:
        raise ValueError(f"{name} must have shape Nx3, got {points.shape}")


def _required_str(row: Mapping[str, object], key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Required string field missing: {key}")
    return value


def _optional_str(row: Mapping[str, object], key: str) -> str | None:
    value = row.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"Expected string field for {key}")
    return value


def _required_int(row: Mapping[str, object], key: str) -> int:
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"Required int field missing: {key}")
    return value


def _optional_float(row: Mapping[str, object], key: str, default: float) -> float:
    value = row.get(key)
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError(f"Expected numeric field for {key}")
    return float(value)


def _optional_bool(row: Mapping[str, object], key: str, default: bool) -> bool:
    value = row.get(key)
    if value is None:
        return default
    if not isinstance(value, bool):
        raise ValueError(f"Expected bool field for {key}")
    return value
