"""Compatibility alias for the active V8 implementation.

User-directed in-place development: this module is not a frozen algorithm.
Historical experiment outputs retain their original meaning and metadata.
"""

from gov_mem.backbones.govmem_v8_late_governance import GovMemV8LateGovernanceBackbone


GOVMEM_V8_DEV1_VERSION = "Gov-Mem-v8-Late-Governance-dev1"


class GovMemV8Dev1LateGovernanceBackbone(GovMemV8LateGovernanceBackbone):
    pass


__all__ = ["GOVMEM_V8_DEV1_VERSION", "GovMemV8Dev1LateGovernanceBackbone"]
