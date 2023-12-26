# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


from typing import List, Optional, Set, Tuple, Union

import dcgm_fields
import torch
from dcgm_fields import DcgmFieldGetById
from dcgm_structs import DCGM_GROUP_EMPTY, DCGM_OPERATION_MODE_AUTO
from pydcgm import DcgmFieldGroup, DcgmGroup, DcgmHandle

from .profiler import _Profiler, logger


class DCGMProfiler:
    """Profiler that triggers start of DCGM profiler."""

    def __init__(
        self,
        main_profiler: "_Profiler",
        gpus_to_profile: Optional[Tuple[int, ...]] = None,
        field_ids_to_profile=(
            dcgm_fields.DCGM_FI_PROF_SM_ACTIVE,
            dcgm_fields.DCGM_FI_PROF_SM_OCCUPANCY,
            dcgm_fields.DCGM_FI_PROF_PIPE_TENSOR_ACTIVE,
            dcgm_fields.DCGM_FI_PROF_DRAM_ACTIVE,
            dcgm_fields.DCGM_FI_PROF_PCIE_TX_BYTES,
            dcgm_fields.DCGM_FI_PROF_PCIE_RX_BYTES,
            dcgm_fields.DCGM_FI_PROF_NVLINK_TX_BYTES,
            dcgm_fields.DCGM_FI_PROF_NVLINK_RX_BYTES,
        ),
        updateFreq: int = 5000,  # in microseconds
    ) -> None:
        """
        Args:
            main_profiler: The main profiler object.
            gpus_to_profile: A tuple of integers representing the GPUs to profile. If `None`,
                then the "default" GPU is used.
            field_ids_to_profile:
                See https://github.com/NVIDIA/DCGM/blob/master/testing/python3/dcgm_fields.py#L436
                for a full list of available fields. Note that not all fields are profilable.
            updateFreq: The interval of two consecutive updates of each field. Defaults to 5000 microseconds.
                This is a good tradeoff between performance and accuracy.
                An even smaller updateFreq is not supported well by A100.
                If the step to profile takes more than 5000 microseconds, then a larger updateFreq could also be used.
        """
        self.main_profiler = main_profiler
        self.updateFreq = updateFreq

        self.dcgmHandle = DcgmHandle(
            ipAddress="127.0.0.1", opMode=DCGM_OPERATION_MODE_AUTO
        )

        if gpus_to_profile is None:
            default_gpu: int = torch.empty([], device="cuda").device.index
            self.dcgmGroup = self.create_dcgm_group((default_gpu,))
        else:
            self.dcgmGroup = self.create_dcgm_group(gpus_to_profile)

        self.dcgmFieldGroup = self.create_profiling_field_group(field_ids_to_profile)

    def create_dcgm_group(
        self, gpus_to_profile: Union[Tuple[int], Tuple[int, ...]]
    ) -> Optional[DcgmGroup]:
        if self.dcgmHandle is None:
            return None

        dcgmSystem = self.dcgmHandle.GetSystem()
        supportedGPUs = dcgmSystem.discovery.GetAllSupportedGpuIds()

        valid_gpus_to_profile: List[int] = [
            gpu for gpu in gpus_to_profile if gpu in supportedGPUs
        ]
        if len(valid_gpus_to_profile) < 1:
            logger.warning(
                f"The provided GPUs are not supported on this system: "
                f"provided {gpus_to_profile}, supported {supportedGPUs}. "
                f"No data will be captured."
            )
            return None

        dcgmGroup = DcgmGroup(
            self.dcgmHandle,
            groupName="DCGMProfiler",
            groupType=DCGM_GROUP_EMPTY,
        )

        for gpu in valid_gpus_to_profile:
            dcgmGroup.AddGpu(gpu)

        return dcgmGroup

    def get_profilable_fields(self) -> Set[int]:
        assert self.dcgmGroup is not None

        dcgmMetricGroups = self.dcgmGroup.profiling.GetSupportedMetricGroups()
        profilableFieldIds = set()
        for group_idx in range(dcgmMetricGroups.numMetricGroups):
            metric_group = dcgmMetricGroups.metricGroups[group_idx]
            for field_id in metric_group.fieldIds[: metric_group.numFieldIds]:
                profilableFieldIds.add(field_id)
        return profilableFieldIds

    def create_profiling_field_group(
        self,
        fieldIdsToProfile: Optional[Tuple[int, ...]],
    ) -> Optional[DcgmFieldGroup]:
        if self.dcgmGroup is None:
            return None

        # Get all field ids that can be profiled.
        profilableFieldIds = self.get_profilable_fields()

        # Check which of the provided field ids are valid and invalid.
        if fieldIdsToProfile is None:
            validFieldIds = list(profilableFieldIds)
            invalidFieldIds = []
        else:
            validFieldIds = [
                field_id
                for field_id in fieldIdsToProfile
                if field_id in profilableFieldIds
            ]
            invalidFieldIds = [
                field_id
                for field_id in fieldIdsToProfile
                if field_id not in profilableFieldIds
            ]

        if not validFieldIds:
            logger.warning(
                "None of the provided field ids could be profiled.\n"
                f"  Provided: {fieldIdsToProfile}\n"
                f"  Supported: {profilableFieldIds}\n"
                "No data will be captured."
            )
            return None

        if invalidFieldIds:
            logger.warning(
                f"The following field ids cannot be profiled: {invalidFieldIds}. "
                f"Profiling {validFieldIds} only."
            )
        dcgmFieldGroup = DcgmFieldGroup(
            self.dcgmHandle, name="Profiling", fieldIds=validFieldIds
        )
        return dcgmFieldGroup

    def __enter__(self) -> None:
        if self.dcgmGroup is not None and self.dcgmFieldGroup is not None:
            self.dcgmGroup.samples.WatchFields(
                self.dcgmFieldGroup, self.updateFreq, 3600, 0
            )

            # Start collecting the profiling results run in background.
            self.profiling_results = self.dcgmGroup.samples.GetAllSinceLastCall(
                None, self.dcgmFieldGroup
            )

            # It is necessary to call GetAllSinceLastCall and EmptyValues twice
            # to clear old data from previous profilings
            # (otherwise the new profiling data is appended to the old data from previous profiling).
            self.dcgmGroup.samples.GetAllSinceLastCall(
                self.profiling_results, self.dcgmFieldGroup
            )
            self.profiling_results.EmptyValues()

            self.dcgmGroup.samples.GetAllSinceLastCall(
                self.profiling_results, self.dcgmFieldGroup
            )
            self.profiling_results.EmptyValues()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self.dcgmGroup is not None and self.dcgmFieldGroup is not None:
            self.dcgmGroup.samples.UnwatchFields(self.dcgmFieldGroup)

            # Delete the group.
            self.dcgmGroup.Delete()
            del self.dcgmGroup
            self.dcgmGroup = None

            # Disconnect from the hostengine by deleting the DcgmHandle object.
            del self.dcgmHandle
            self.dcgmHandle = None

    def step(self) -> None:
        if self.dcgmGroup is not None and self.dcgmFieldGroup is not None:
            # Collect the profiling results.
            self.dcgmGroup.samples.GetAllSinceLastCall(
                self.profiling_results, self.dcgmFieldGroup
            )

            # Save profiling results to log.
            for gpu_id in self.profiling_results.values.keys():
                for field_id in self.profiling_results.values[gpu_id].keys():
                    field_name = DcgmFieldGetById(field_id).tag

                    field_avg_val = 0.0
                    num_vals = 0
                    for gpu_field_time in self.profiling_results.values[gpu_id][
                        field_id
                    ]:
                        if gpu_field_time.value is not None:
                            field_avg_val = (
                                field_avg_val * num_vals + gpu_field_time.value
                            ) / (num_vals + 1)
                            num_vals += 1
                    self.main_profiler.summary.append(
                        (f"GPU {gpu_id}, {field_name}({field_id})", f"{field_avg_val}")
                    )

            # Clear the profiling results to get ready for the next collection.
            self.profiling_results.EmptyValues()
