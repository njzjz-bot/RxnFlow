import enum
import re
from dataclasses import dataclass
from functools import cached_property

from ..gflownet.types import Action, ActionType


class RxnActionType(ActionType):
    # Forward actions
    FirstBlock = enum.auto()
    BiRxn = enum.auto()
    UniRxn = enum.auto()
    Stop = enum.auto()

    # Backward actions
    BckFirstBlock = enum.auto()
    BckBiRxn = enum.auto()
    BckUniRxn = enum.auto()
    BckStop = enum.auto()

    @cached_property
    def cname(self) -> str:
        return re.sub(r"(?<!^)(?=[A-Z])", "_", self.name).lower()

    @cached_property
    def mask_name(self) -> str:
        return self.cname + "_mask"

    @cached_property
    def is_backward(self) -> bool:
        return self.name.startswith("Bck")


@dataclass(frozen=True, slots=True)
class RxnAction(Action):
    """A single graph-building action

    Parameters
    ----------
    action: GraphActionType
        the action type
    block: str, optional
        the block smi object
    block_cluster_idx: int, optional
        the block cluster idx in the workflow
    block_idx: int, optional
        the block idx in the cluster
    """

    action: RxnActionType
    block: str = ""
    block_cluster_idx: int = -1
    block_idx: int = -1

    def __post_init__(self):
        # Validate the action parameters
        assert self.block_cluster_idx >= -1, "Block cluster idx must be -1 or greater"
        assert self.block_idx >= -1, "Block idx must be -1 or greater"

        # Validate the action parameters for each action types
        match self.action:
            case RxnActionType.FirstBlock | RxnActionType.BckFirstBlock:
                assert self.block != "", "Block must be set for FirstBlock action"
                assert self.block_cluster_idx >= 0, (
                    "Block cluster idx must be set for FirstBlock action"
                )
                assert self.block_idx >= 0, "Block idx must be set for FirstBlock action"
            case RxnActionType.BiRxn | RxnActionType.BckBiRxn:
                assert self.block != "", "Block must be set for BiRxn action"
                assert self.block_cluster_idx >= 0, (
                    "Block cluster idx must be set for BiRxn action"
                )
                assert self.block_idx >= 0, "Block idx must be set for BiRxn action"
            case RxnActionType.UniRxn | RxnActionType.BckUniRxn:
                assert self.block == "", "Block must not be set for UniRxn action"
                assert self.block_cluster_idx == -1, (
                    "Block cluster idx must not be set for UniRxn action"
                )
                assert self.block_idx == -1, "Block idx must not be set for UniRxn action"

    def __repr__(self):
        return f"RxnAction({self.action}: {self.workflow}, {self.protocol_order}, {self.block}[{self.block_cluster_idx}, {self.block_idx}])"

    @property
    def is_fwd(self) -> bool:
        return self.action in (
            RxnActionType.FirstBlock,
            RxnActionType.BiRxn,
            RxnActionType.UniRxn,
        )


class Protocol:
    action_type: RxnActionType

    def __init__(
        self,
        name: str,
        type: RxnActionType,
        forward: str | None = None,
        reverse: str | None = None,
        block_type: str | None = None,
    ):
        # type check
        match type:
            case RxnActionType.FirstBlock:
                assert block_type is not None and forward is None and reverse is None
            case RxnActionType.BiRxn:
                assert (
                    block_type is not None and forward is not None and reverse is not None
                )
            case RxnActionType.UniRxn:
                assert block_type is None and forward is not None and reverse is not None
            case _:
                raise ValueError(f"Unsupported action type: {type}")
        self.name: str = name
        self.type: RxnActionType = type
        self._block_type: str | None = block_type
        self._forward: str | None = forward
        self._rxn_forward: Reaction | None = Reaction(forward) if forward else None
        self._reverse: str | None = reverse
        self._rxn_reverse: Reaction | None = Reaction(reverse) if reverse else None

    @property
    def block_type(self) -> str:
        assert self._block_type is not None
        return self._block_type

    @property
    def forward(self) -> str:
        assert self._forward is not None
        return self._forward

    @property
    def rxn_forward(self) -> Reaction:
        assert self._rxn_forward is not None
        return self._rxn_forward

    @property
    def reverse(self) -> str:
        assert self._reverse is not None
        return self._reverse

    @property
    def rxn_reverse(self) -> Reaction:
        assert self._rxn_reverse is not None
        return self._rxn_reverse
