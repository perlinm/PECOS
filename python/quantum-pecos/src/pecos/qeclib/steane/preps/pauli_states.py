"""Pauli eigenstate preparation for the Steane 7-qubit code.

This module provides implementations for preparing logical Pauli eigenstates (|0⟩, |1⟩, |+⟩, |-⟩ |+i⟩, |-i⟩) in the
Steane 7-qubit code using both fault-tolerant and non-fault-tolerant encoding methods.
"""

# Copyright 2024 The PECOS Developers
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License.You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

from pecos.qeclib import qubit
from pecos.qeclib.qubit import Prep
from pecos.qeclib.steane.gates_sq import sqrt_paulis
from pecos.qeclib.steane.gates_sq.hadamards import H
from pecos.qeclib.steane.gates_sq.paulis import X, Z
from pecos.slr import Barrier, Bit, Block, Comment, If, QReg, Qubit, Repeat, CReg
from pecos.qeclib.steane.syn_extract.three_parallel_flagging import (
    ThreeParallelFlaggingXZZ,
    ThreeParallelFlaggingZXX,
)

class PrepEncodingNonFTZero(Block):
    """Represents the non-fault-tolerant encoding circuit for the Steane code."""

    def __init__(self, q: QReg) -> None:
        """Initialize PrepEncodingNonFTZero block for non-fault-tolerant zero state preparation.

        Args:
            q: Quantum register containing exactly 7 qubits for the Steane code.

        Raises:
            Exception: If the register does not contain exactly 7 qubits.
        """
        if len(q.elems) != 7:
            msg = f"Size of register {len(q.elems)} != 7"
            raise Exception(msg)

        super().__init__(
            qubit.H(
                q[0],
                q[4],
                q[6],
            ),
            Comment(),
            qubit.CX(
                (q[4], q[5]),
                (q[0], q[1]),
                (q[6], q[3]),
                (q[4], q[2]),
                (q[6], q[5]),
                (q[0], q[3]),
                (q[4], q[1]),
                (q[3], q[2]),
            ),
        )


class PrepZeroVerify(Block):
    """Verify the initialization of InitEncodingNonFTZero."""

    def __init__(
        self,
        qubits: QReg,
        ancilla: Qubit,
        init_bit: Bit,
        *,
        reset_ancilla: bool = True,
    ) -> None:
        """Initialize PrepZeroVerify block for verification of zero state preparation.

        Args:
            qubits: Data register containing the 7 qubits of the Steane code.
            ancilla: Ancilla qubit used for verification.
            init_bit: Bit to store the verification result.
            reset_ancilla: Whether to reset the ancilla qubit. Defaults to True.
        """
        q = qubits
        a = ancilla
        c = init_bit

        super().__init__(
            Comment(),
            Barrier(a, q[1], q[3], q[5]),
            Comment("verification step"),
        )

        if reset_ancilla:
            self.extend(
                Comment(),
                Prep(a),
            )

        self.extend(
            qubit.CX(
                (q[5], a),
                (q[1], a),
                (q[3], a),
            ),
            qubit.Measure(a) > c,
            Comment(""),
        )


class PrepEncodingFTZero(Block):
    """Represents the non-fault-tolerant encoding circuit for the Steane code.

    Args:
        data (QReg[7]):
        ancilla (Qubit):
        init_bit (Bit):
        reset (bool):
    """

    def __init__(
        self,
        data: QReg,
        ancilla: Qubit,
        init_bit: Bit,
        *,
        reset: bool = True,
    ) -> None:
        """Initialize PrepEncodingFTZero block for fault-tolerant zero state preparation.

        Args:
            data: Data register containing the 7 qubits of the Steane code.
            ancilla: Ancilla qubit used for verification.
            init_bit: Bit to store the initialization result.
            reset: Whether to reset qubits before preparation. Defaults to True.
        """
        q = data
        a = ancilla

        super().__init__()

        self.extend(
            Comment(),
            Barrier(q[0], q[1], q[2], q[3], q[4], q[5], q[6], a),
            Comment(),
        )

        if reset:
            self.extend(
                Prep(q),
                Prep(a),
                Barrier(q, a),
            )

        self.extend(
            PrepEncodingNonFTZero(data),
            # reset_ancilla to False because it is reset earlier
            PrepZeroVerify(data, ancilla, init_bit, reset_ancilla=False),
        )


class PrepEncodingFTZero_fullqed(Block):
    """Represents the non-fault-tolerant encoding circuit for the Steane code.

    Args:
        data (QReg[7]):
        ancilla (Qubit):
        reject (Bit):
        reset (bool):
    """

    def __init__(
        self,
        data: QReg,
        ancilla: QReg,
        reject: Bit,
        flag_x: CReg,
        flag_z: CReg,
        flags: CReg,
        last_raw_syn_x: CReg,
        last_raw_syn_z: CReg, 
        *,
        reset: bool = True,
    ) -> None:
        """Initialize PrepEncodingFTZero block for fault-tolerant zero state preparation.

        Args:
            data: Data register containing the 7 qubits of the Steane code.
            ancilla: Ancilla qubit used for verification.
            init_bit: Bit to store the initialization result.
            reset: Whether to reset qubits before preparation. Defaults to True.
        """
        q = data
        a = ancilla

        super().__init__()

        self.extend(
            Comment(),
            Barrier(q[0], q[1], q[2], q[3], q[4], q[5], q[6], a),
            Comment(),
        )

        if reset:
            self.extend(
                Prep(q),
                Prep(a),
                Barrier(q, a),
            )

        self.extend(
            PrepEncodingNonFTZero(data),
            # reset_ancilla to False because it is reset earlier
            PrepZeroVerify(data, ancilla[0], reject, reset_ancilla=False),
        )

        # QED
        # self.extend(
        #     ThreeParallelFlaggingXZZ(q, a, flag_x, flag_z, flags, last_raw_syn_x, last_raw_syn_z),
        # )

        # self.extend(
        #     reject.set(reject | flags[0] | flags[1] | flags[2]),
        #     # Reject on the results of the `reject` bit. 0 is good. 1 means the prep failed.
        # )

        self.extend(
            qubit.Prep(a[0]),
            qubit.Prep(a[1]),
            qubit.H(a[0]),
            qubit.CX(a[0],a[1]),
            qubit.CZ(a[0],q[0]),
            qubit.CZ(a[0],q[1]),
            qubit.CZ(a[0],q[2]),
            qubit.CZ(a[0],q[3]),
            qubit.CX(a[0],a[1]),
            qubit.H(a[0]),
            qubit.Measure(a[0]) > flag_z[0],
            qubit.Measure(a[1]) > flag_z[1],
        )

        self.extend(
            reject.set(reject | flag_z[0] | flag_z[1] ),
            # Reject on the results of the `reject` bit. 0 is good. 1 means the prep failed.
        )

        self.extend(
            qubit.Prep(a[0]),
            qubit.Prep(a[1]),
            qubit.H(a[0]),
            qubit.CX(a[0],a[1]),
            qubit.CZ(a[0],q[0]),
            qubit.CZ(a[0],q[2]),
            qubit.CZ(a[0],q[4]),
            qubit.CZ(a[0],q[6]),
            qubit.CX(a[0],a[1]),
            qubit.H(a[0]),
            qubit.Measure(a[0]) > flag_z[0],
            qubit.Measure(a[1]) > flag_z[1],
        )

        self.extend(
            reject.set(reject | flag_z[0] | flag_z[1] ),
            # Reject on the results of the `reject` bit. 0 is good. 1 means the prep failed.
        )

        # self.extend(
        #     ThreeParallelFlaggingZXX(q, a, flag_x, flag_z, flags, last_raw_syn_x, last_raw_syn_z),
        # )

        # self.extend(
        #     If(flags == 0).Then(
        #         ThreeParallelFlaggingZXX(q, a, flag_x, flag_z, flags, last_raw_syn_x, last_raw_syn_z),
        #     ),
        # )

        


class PrepRUS(Block):
    """Use repeat-until-success to initialize a logical qubit."""

    def __init__(
        self,
        q: QReg,
        a: Qubit,
        init: Bit,
        limit: int,
        state: str = "|0>",
        *,
        first_round_reset: bool = True,
    ) -> None:
        """Initialize PrepRUS block for repeat-until-success state preparation.

        Args:
            q: Data register containing the 7 qubits of the Steane code.
            a: Ancilla qubit used for verification.
            init: Bit to track initialization success.
            limit: Maximum number of preparation attempts.
            state: Target Pauli eigenstate to prepare. Defaults to '|0>'.
            first_round_reset: Whether to reset on first round. Defaults to True.
        """
        super().__init__(
            PrepEncodingFTZero(q, a, init, reset=first_round_reset),
            Repeat(limit - 1).block(
                If(init == 1).Then(
                    PrepEncodingFTZero(q, a, init, reset=True),
                ),
            ),
        )
        if limit == 1:
            self.extend(
                Comment(),
            )

        self.extend(
            # Rotate to the Pauli basis of choice
            LogZeroRot(q, state),
        )

class PrepRUS_fullqed(Block):
    """Use repeat-until-success to initialize a logical qubit."""

    def __init__(
        self,
        q: QReg,
        a: QReg,
        reject: Bit,
        limit: int,
        flag_x: CReg,
        flag_z: CReg,
        flags: CReg,
        last_raw_syn_x: CReg,
        last_raw_syn_z: CReg, 
        state: str = "|0>",
        *,
        first_round_reset: bool = True,
    ) -> None:
        """Initialize PrepRUS block for repeat-until-success state preparation.

        Args:
            q: Data register containing the 7 qubits of the Steane code.
            a: Ancilla qubit used for verification.
            init: Bit to track initialization success.
            limit: Maximum number of preparation attempts.
            state: Target Pauli eigenstate to prepare. Defaults to '|0>'.
            first_round_reset: Whether to reset on first round. Defaults to True.
        """
        super().__init__(
            PrepEncodingFTZero_fullqed(q, a, reject, flag_x,
            flag_z,
            flags,
            last_raw_syn_x,
            last_raw_syn_z, 
            reset=first_round_reset),
            Repeat(limit - 1).block(
                If(reject == 1).Then(
                    PrepEncodingFTZero_fullqed(q, a, reject, flag_x,
                    flag_z,
                    flags,
                    last_raw_syn_x,
                    last_raw_syn_z, reset=True),
                ),
            ),
        )
        if limit == 1:
            self.extend(
                Comment(),
            )

        self.extend(
            # Rotate to the Pauli basis of choice
            LogZeroRot(q, state),
        )

class LogZeroRot(Block):
    """Rotate logical |0> to appropriate Pauli state."""

    def __init__(self, q: QReg, state: str) -> None:
        """Initialize LogZeroRot block to rotate logical |0> to target Pauli state.

        Args:
            q: Data register containing the 7 qubits of the Steane code.
            state: Target state notation (e.g., '|0>', '|1>', '|+>', '|->', '|+i>', '|-i>').

        Raises:
            Exception: If the target state is not supported.
        """
        super().__init__()

        match state:
            case "|1>" | "-Z":
                self.extend(
                    X(q),
                )
            case "|0>" | "+Z" | "Z":
                pass
            case "|+>" | "+X" | "X":
                self.extend(
                    H(q),
                )
            case "|->" | "-X":
                self.extend(
                    H(q),
                    Z(q),
                )
            case "|+i>" | "+Y" | "Y":
                self.extend(
                    sqrt_paulis.SXdg(q),
                )
            case "|-i>" | "-Y":
                self.extend(
                    sqrt_paulis.SX(q),
                )
            case _:
                msg = f"Unsupported init state '{state}'"
                raise Exception(msg)
