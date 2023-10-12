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

from pecos.simulators.projectq.state import ProjectQSim
from pecos.simulators.sparsesim.state import SparseSim

if TYPE_CHECKING:
    from pecos.reps.pypmir.op_types import QOp


class QuantumSimulator:
    def __init__(self, backend: str | object | None = None) -> None:
        self.num_qubits = None
        self.state = None
        self.backend = backend

    def reset(self):
        self.num_qubits = None
        self.state = None

    def init(self, num_qubits: int):
        self.num_qubits = num_qubits

        if isinstance(self.backend, str) and self.backend == "stabilizer":
            self.state = SparseSim

        if self.backend is None:
            self.state = ProjectQSim

        self.state = self.state(num_qubits=num_qubits)

    def shot_reinit(self):
        """Run all code needed at the beginning of each shot, e.g., resetting state."""
        self.state.reset()

    def run(self, qops: list[QOp]) -> list:
        """Given a list of quantum operations, run them, update the state, and return any measurement results that
        are generated in the form {qid: result, ...}.
        """
        meas = []
        for op in qops:
            output = self.state.run_gate(op.name, op.args, **op.metadata)
            if op.returns:
                temp = {}
                bitflips = op.metadata.get("bitflips")
                for q, r in zip(op.args, op.returns):
                    out = output.get(q, 0)
                    if bitflips and q in bitflips:
                        out ^= 1

                    temp[tuple(r)] = out

                meas.append(temp)

        return meas