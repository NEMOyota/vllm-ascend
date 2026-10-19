# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vllm_ascend project
"""Mooncake KV-cache transfer connector (disaggregated prefill / decode)."""

from importlib import import_module

__all__ = [
    "MooncakeTransferMetadata",
    "MooncakeBaseConnector",
    "MooncakeBaseConnectorScheduler",
    "MooncakeBaseConnectorWorker",
    "MooncakeConnector",
    "MooncakeConnectorMetadata",
    "MooncakeKVConnectorStats",
    "MooncakePullConnector",
    "MooncakePullConnectorScheduler",
    "MooncakePullConnectorWorker",
]


def __getattr__(name: str):
    # Scheduler/config users do not need to import device worker dependencies.
    modules = {
        "MooncakeTransferMetadata": "metadata",
        "MooncakeConnectorMetadata": "metadata",
        "MooncakeBaseConnector": "connector",
        "MooncakeConnector": "connector",
        "MooncakePullConnector": "connector",
        "MooncakeBaseConnectorScheduler": "base_scheduler",
        "MooncakeBaseConnectorWorker": "base_worker",
        "MooncakeKVConnectorStats": "stats",
        "MooncakePullConnectorScheduler": "pull_scheduler",
        "MooncakePullConnectorWorker": "pull_worker",
    }
    if name not in modules:
        raise AttributeError(name)
    return getattr(import_module(f"{__name__}.{modules[name]}"), name)
