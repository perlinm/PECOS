# Copyright 2023 The PECOS Developers
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License.You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

from __future__ import annotations

from typing import TYPE_CHECKING

from pecos.machines.machine_abc import Machine
from pecos.reps.pypmir.op_types import QOp

if TYPE_CHECKING:
    from pecos.error_models.error_model_abc import ErrorModel
    from pecos.reps.pypmir.op_types import MOp


class GenericMachine(Machine):
    """Represents generic, abstract machine."""

    def __init__(self, error_model: ErrorModel | None = None, num_qubits: int | None = None) -> None:
        super().__init__(error_model, num_qubits)
        self.leaked_qubits = None
        self.lost_qubits = None

    def reset(self) -> None:
        """Reset state to initialization state."""
        self.leaked_qubits = set()
        self.lost_qubits = set()

    def init(self, machine_params: dict | None = None, num_qubits: int | None = None) -> None:
        if machine_params:
            self.machine_params = machine_params

        self.num_qubits = num_qubits

    def shot_reinit(self) -> None:
        self.reset()

    def process(self, op_buffer: list[QOp | MOp]) -> list:
        for op in op_buffer:
            if "mop" in op:
                print("MOP >", op)

        return op_buffer

    def leak(self, qubits: set) -> list[QOp]:
        """Starts tracking qubits as leaked qubits and calls the quantum simulation appropriately to trigger leakage."""
        self.leaked_qubits |= qubits
        return [QOp(name="Init", args=list(qubits), metadata={})]

    def unleak(self, qubits: set) -> None:
        """Untrack qubits as leaked qubits and calls the quantum simulation appropriately to trigger leakage."""
        self.leaked_qubits -= qubits

    def meas_leaked(self, qubits: set) -> list[QOp]:
        self.leaked_qubits -= qubits
        return [
            QOp(name="Init -Z", args=list(qubits), metadata={}),
        ]