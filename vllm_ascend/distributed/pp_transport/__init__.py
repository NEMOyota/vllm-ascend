# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Model-independent, bounded pipeline transport.

Keep configuration imports light: AscendConfig is imported during platform
discovery, before torch_npu and the distributed runtime are initialized.
"""

from .config import WanPPConfig

__all__ = ["CreditBudget", "PipelineTransportError", "ReceiveTicket", "WanPPConfig", "WanPPTransport"]


def __getattr__(name: str):
    if name == "CreditBudget":
        from .credits import CreditBudget

        return CreditBudget
    if name in ("PipelineTransportError", "ReceiveTicket", "WanPPTransport"):
        from .transport import PipelineTransportError, ReceiveTicket, WanPPTransport

        return {
            "PipelineTransportError": PipelineTransportError,
            "ReceiveTicket": ReceiveTicket,
            "WanPPTransport": WanPPTransport,
        }[name]
    raise AttributeError(name)
