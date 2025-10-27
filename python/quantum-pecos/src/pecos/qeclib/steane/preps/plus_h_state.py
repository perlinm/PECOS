"""Logical |+H⟩ state preparation for the Steane 7-qubit code.

This module provides implementations for preparing the logical |+H⟩ state in the Steane 7-qubit code using
fault-tolerant distillation and verification protocols.
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

from numpy import pi

from pecos.qeclib import qubit
from pecos.qeclib.generic.check_1flag import Check1Flag
from pecos.qeclib.steane.preps.encoding_circ import EncodingCircuit
from pecos.qeclib.steane.syn_extract.three_parallel_flagging import (
    ThreeParallelFlaggingXZZ,
    ThreeParallelFlaggingZXX,
)
from pecos.slr import Bit, Block, Comment, CReg, If, QReg, Repeat


class PrepHStateFT(Block):
    """Prepare a |+H> state fault tolerantly.

    By using an encoding circuit to prepare logical|+H>, measuring the logical
    Hadamard with a flag, doing a QED round, and post-selecting based on non-trivial measurements.

    Arguments:
        d: Data qubits (size 7)
        a: Axillary qubits (size 2)
        out: Measurement outputs (size 2). out[0] is the Measure H result and out[1] is the flag result.
        reject: Whether the procedure failed and should be rejected. 0 it is good, 1 prep failed.
    """

    def __init__(
        self,
        d: QReg,
        a: QReg,
        out: CReg,
        reject: Bit,
        flag_x: CReg,
        flag_z: CReg,
        flags: CReg,
        last_raw_syn_x: CReg,
        last_raw_syn_z: CReg,
        *,
        condition_qed: bool = True,
    ) -> None:
        """Initialize PrepHStateFT block for fault-tolerant |+H> state preparation.

        Args:
            d: Data qubits (size 7) for the Steane code.
            a: Ancillary qubits (size 2) for measurements.
            out: Measurement outputs (size 2). out[0] is the Hadamard measurement,
                out[1] is the flag result.
            reject: Bit indicating preparation failure (0 for success, 1 for failure).
            flag_x: Classical register for X stabilizer flags.
            flag_z: Classical register for Z stabilizer flags.
            flags: Combined flags register.
            last_raw_syn_x: Previous X syndrome measurements.
            last_raw_syn_z: Previous Z syndrome measurements.
            condition_qed: Whether to condition second QED round on first. Defaults to True.
        """
        super().__init__()

        # non-fault-tolerantly encode logical |+H>
        # ----------------------------------------
        self.extend(
            qubit.Prep(d[6]),
            qubit.RY[pi / 4](d[6]),
            EncodingCircuit(d),
        )

        # flagged logical H measurement
        # -----------------------------
        self.extend(
            Check1Flag(
                d=[d[0], d[1], d[2], d[3], d[4], d[5], d[6]],
                ops="HHHHHHH",
                a=a[0],
                flag=a[1],
                out=out[0],
                out_flag=out[1],
            ),
        )

        # QED
        self.extend(
            ThreeParallelFlaggingXZZ(
                d,
                a,
                flag_x,
                flag_z,
                flags,
                last_raw_syn_x,
                last_raw_syn_z,
            ),
        )

        if condition_qed:
            self.extend(
                If(flags == 0).Then(
                    ThreeParallelFlaggingZXX(
                        d,
                        a,
                        flag_x,
                        flag_z,
                        flags,
                        last_raw_syn_x,
                        last_raw_syn_z,
                    ),
                ),
            )
        else:
            self.extend(
                ThreeParallelFlaggingZXX(
                    d,
                    a,
                    flag_x,
                    flag_z,
                    flags,
                    last_raw_syn_x,
                    last_raw_syn_z,
                ),
            )

        self.extend(
            reject.set(out[0] | out[1] | flags[0] | flags[1] | flags[2]),
            # Reject on the results of the `reject` bit. 0 is good. 1 means the prep failed.
        )

class PrepHStateFT_8cnot(Block):
    """
    Prepare a |+H> state fault tolerantly by using an encoding circuit to prepare logical|+H>, measuring the logical
    Hadamard with a flag, doing a QED round, and post-selecting based on non-trivial measurements.

    Arguments:
        d: Data qubits (size 7)
        a: Axillary qubits (size 2)
        out: Measurement outputs (size 2). out[0] is the Measure H result and out[1] is the flag result.
        reject: Whether the procedure failed and should be rejected. 0 it is good, 1 prep failed.
    """

    def __init__(
        self,
        d: QReg,
        a: QReg,
        out: CReg,
        reject: Bit,
        flag_x: CReg,
        flag_z: CReg,
        flags: CReg,
        last_raw_syn_x: CReg,
        last_raw_syn_z: CReg,
        *,
        condition_qed: bool = True,
    ):
        super().__init__()

        # non-fault-tolerantly encode logical |+H>
        # ----------------------------------------
        self.extend(
            Comment('\n Start Non-FT encoding a |H> state with 8 cx'),
            qubit.Prep(d[0],d[1],d[2],d[3],d[4],d[5],d[6]),
            qubit.H(d[0],
                    d[4],
                    d[6]),
            qubit.CX(
                (d[0],d[1]),
                (d[4],d[5]),
                (d[6],d[3]),
                (d[6],d[5]),
                (d[4],d[2]),
                (d[0],d[3]),
                (d[4],d[1]),
            ),
            ####### 
            # ''' qubit.RXX[pi/4](d[3],d[4]) '''
            qubit.H(d[3]),
            qubit.H(d[4]),
            # qubit.RZZ[pi/4](d[3],d[4]),
            qubit.CX(d[3],d[4]),
            qubit.RZ[pi/4](d[4]),
            qubit.CX(d[3],d[4]),
            qubit.H(d[3]),
            qubit.H(d[4]), 
            #########
            qubit.CX(d[3],d[2]),
            qubit.SZdg(d[0],d[1],d[2],d[3],d[4],d[5],d[6]),
            Comment('\n End Non-FT encoding a |H> state with 8 cx'),
        )
        
        # flagged logical H measurement
        # -----------------------------
        self.extend(
            Check1Flag(
                d=[d[0], d[1], d[2], d[3], d[4], d[5], d[6]],
                ops="HHHHHHH",
                a=a[0],
                flag=a[1],
                out=out[0],
                out_flag=out[1],
            ),
        )

        # QED
        self.extend(
            ThreeParallelFlaggingXZZ(d, a, flag_x, flag_z, flags, last_raw_syn_x, last_raw_syn_z),
        )

        if condition_qed:
            self.extend(
                If(flags == 0).Then(
                    ThreeParallelFlaggingZXX(d, a, flag_x, flag_z, flags, last_raw_syn_x, last_raw_syn_z),
                ),
            )
        else:
            self.extend(
                ThreeParallelFlaggingZXX(d, a, flag_x, flag_z, flags, last_raw_syn_x, last_raw_syn_z),
            )

        self.extend(
            reject.set(out[0] | out[1] | flags[0] | flags[1] | flags[2]),
            # Reject on the results of the `reject` bit. 0 is good. 1 means the prep failed.
        )

class PrepHStateFT_noveri(Block):
    """
    Prepare a |+H> state fault tolerantly by using an encoding circuit to prepare logical|+H>, measuring the logical
    Hadamard with a flag, doing a QED round, and post-selecting based on non-trivial measurements.

    Arguments:
        d: Data qubits (size 7)
        a: Axillary qubits (size 2)
        out: Measurement outputs (size 2). out[0] is the Measure H result and out[1] is the flag result.
        reject: Whether the procedure failed and should be rejected. 0 it is good, 1 prep failed.
    """

    def __init__(
        self,
        d: QReg,
        a: QReg,
        out: CReg,
        reject: Bit,
        flag_x: CReg,
        flag_z: CReg,
        flags: CReg,
        last_raw_syn_x: CReg,
        last_raw_syn_z: CReg,
        *,
        condition_qed: bool = True,
    ):
        super().__init__()

        # non-fault-tolerantly encode logical |+H>
        # ----------------------------------------
        self.extend(
            Comment('\n Start Non-FT encoding a |H> state without verification'),
            qubit.Prep(d[0],d[1],d[2],d[3],d[4],d[5],d[6]),
            qubit.H(d[0],
                    d[4],
                    d[6]),
            qubit.CX(
                (d[0],d[1]),
                (d[4],d[5]),
                (d[6],d[3]),
                (d[6],d[5]),
                (d[4],d[2]),
                (d[0],d[3]),
                (d[4],d[1]),
            ),
            ####### 
            # ''' qubit.RXX[pi/4](d[3],d[4]) '''
            qubit.H(d[3]),
            qubit.H(d[4]),
            # qubit.RZZ[pi/4](d[3],d[4]),
            qubit.CX(d[3],d[4]),
            qubit.RZ[pi/4](d[4]),
            qubit.CX(d[3],d[4]),
            qubit.H(d[3]),
            qubit.H(d[4]), 
            #########
            qubit.CX(d[3],d[2]),
            qubit.SZdg(d[0],d[1],d[2],d[3],d[4],d[5],d[6]),
            Comment('\n End Non-FT encoding a |H> state with 8 cx'),
        )
        
        # flagged logical H measurement
        # -----------------------------
        self.extend(
            Check1Flag(
                d=[d[0], d[1], d[2], d[3], d[4], d[5], d[6]],
                ops="HHHHHHH",
                a=a[0],
                flag=a[1],
                out=out[0],
                out_flag=out[1],
            ),
        )

        # QED
        # self.extend(
        #     ThreeParallelFlaggingXZZ(d, a, flag_x, flag_z, flags, last_raw_syn_x, last_raw_syn_z),
        # )

        # if condition_qed:
        #     self.extend(
        #         If(flags == 0).Then(
        #             ThreeParallelFlaggingZXX(d, a, flag_x, flag_z, flags, last_raw_syn_x, last_raw_syn_z),
        #         ),
        #     )
        # else:
        #     self.extend(
        #         ThreeParallelFlaggingZXX(d, a, flag_x, flag_z, flags, last_raw_syn_x, last_raw_syn_z),
        #     )

        # self.extend(
        #     reject.set(out[0] | out[1] | flags[0] | flags[1] | flags[2]),
        #     # Reject on the results of the `reject` bit. 0 is good. 1 means the prep failed.
        # )

class PrepHStateFT_30gate_v0(Block):
    """
    Prepare a |+H> state fault tolerantly by using an encoding circuit to prepare logical|+H>, measuring the logical
    Hadamard with a flag, doing a QED round, and post-selecting based on non-trivial measurements.

    Arguments:
        d: Data qubits (size 7)
        a: Axillary qubits (size 2)
        out: Measurement outputs (size 2). out[0] is the Measure H result and out[1] is the flag result.
        reject: Whether the procedure failed and should be rejected. 0 it is good, 1 prep failed.
    """

    def __init__(
        self,
        d: QReg,
        a: QReg,
        out: CReg,
        reject: Bit,
        flag_x: CReg,
        flag_z: CReg,
        flags: CReg,
        last_raw_syn_x: CReg,
        last_raw_syn_z: CReg,
        *,
        condition_qed: bool = True,
    ):
        super().__init__()
        if a.size < 6:
            msg = "need at least 6 physical ancilla qubits"
            raise Exception(msg)
            
        # non-fault-tolerantly encode logical |+H>
        # ----------------------------------------
        self.extend(
            Comment('\n Start FT preparing a |H> state with 30 two-qubit gates'),
            qubit.Prep(d[0],d[1],d[2],d[3],d[4],d[5],d[6]),
            qubit.Prep(a[0],a[1],a[2],a[3],a[4],a[5]),
            qubit.H(d[0],
                    d[4],
                    d[6],
                    a[0], 
                    a[1],
                    a[2],
                   ),
            qubit.CX(
                (d[0],d[1]),
                (d[4],d[5]),
                (d[6],d[3]),
            ),
            qubit.CX((a[1],d[3])),
            qubit.CX(
                (d[6],d[5]),
                (d[4],d[2]),
                (d[0],d[3]),
            ),
            qubit.CX((a[0],d[2])),
            qubit.CX(
                (d[4],d[1]),
            ),
            ####### 
            # ''' qubit.RXX[pi/4](d[3],d[4]) '''
            qubit.H(d[3]),
            qubit.H(d[4]),
            # qubit.RZZ[pi/4](d[3],d[4]),
            qubit.CX(d[3],d[4]),
            qubit.RZ[pi/4](d[4]),
            qubit.CX(d[3],d[4]),
            qubit.H(d[3]),
            qubit.H(d[4]), 
            
            qubit.CX((a[1],d[3])),
            #########
            qubit.CX(d[3],d[2]),
            qubit.CX((a[0],d[2])),
            qubit.SZdg(d[0],d[1],d[2],d[3],d[4],d[5],d[6]),
        )
        
        # flagged logical H measurement
        # -----------------------------
        self.extend(
         qubit.H(a[0]),   
         qubit.H(a[1]),
         qubit.CX(a[2],a[3]),
         qubit.CH(a[2],d[0]),
         qubit.CH(a[2],d[1]),
         qubit.CH(a[2],d[2]),
         qubit.CH(a[2],d[3]),
         qubit.CH(a[2],d[4]),
         qubit.CH(a[2],d[5]),
         qubit.CH(a[2],d[6]),
         qubit.CX(a[2],a[3]),
         qubit.H(a[2]),
        )
        
        self.extend(
         qubit.Measure(a[0]) > flag_z[0],
         qubit.Measure(a[1]) > flag_z[1],
         qubit.Measure(a[2]) > flag_z[2],
        )
        
        # measure rest stabilizers
        self.extend(
         qubit.CX(d[0], a[4]),
         qubit.H(a[5]),
         qubit.CX(a[5], d[5]),
        )
        
        self.extend(
         qubit.CX(d[5], a[4]),
         qubit.CX(a[5], d[0]),
        )
        
        self.extend(
         qubit.CX(d[4], a[4]),
         qubit.CX(a[5], d[3]),
        )
        
        self.extend(
         qubit.CX(d[3], a[4]),
         qubit.CX(a[5], d[4]),
        )
        
        self.extend(
         qubit.Measure(a[3]) > flag_x[0],
         qubit.Measure(a[4]) > flag_x[1],
         qubit.H(a[5]),
         qubit.Measure(a[5]) > flag_x[2],
        )
        
        self.extend(
            reject.set(flag_z[0] | flag_z[1] | flag_z[2] | flag_x[0] | flag_x[1] | flag_x[2]),
            # Reject on the results of the `reject` bit. 0 is good. 1 means the prep failed.
        )
        Comment('\n End FT preparing a |H> state with 30 two-qubit gates'),

class PrepHStateFT_30gate(Block):
    """
    Prepare a |+H> state fault tolerantly by using an encoding circuit to prepare logical|+H>, measuring the logical
    Hadamard with a flag, doing a QED round, and post-selecting based on non-trivial measurements.

    Arguments:
        d: Data qubits (size 7)
        a: Axillary qubits (size 2)
        out: Measurement outputs (size 2). out[0] is the Measure H result and out[1] is the flag result.
        reject: Whether the procedure failed and should be rejected. 0 it is good, 1 prep failed.
    """

    def __init__(
        self,
        d: QReg,
        a: QReg,
        out: CReg,
        reject: Bit,
        flag_x: CReg,
        flag_z: CReg,
        flags: CReg,
        last_raw_syn_x: CReg,
        last_raw_syn_z: CReg,
        *,
        condition_qed: bool = True,
    ):
        super().__init__()
        if a.size < 6:
            msg = "need at least 6 physical ancilla qubits"
            raise Exception(msg)
            
        # non-fault-tolerantly encode logical |+H>
        # ----------------------------------------
        self.extend(
            Comment('\n Start FT preparing a |H> state with 30 two-qubit gates'),
            qubit.Prep(d[0],d[1],d[2],d[3],d[4],d[5],d[6]),
            qubit.Prep(a[0],a[1],a[2],a[3],a[4],a[5]),
            qubit.H(d[0],
                    d[4],
                    d[6],
                   ),
            qubit.CX(
                (d[0],d[1]),
                (d[4],d[5]),
                (d[6],d[3]),
            ),
            
            qubit.H(a[2]),
            qubit.CX(
                (d[6],d[5]),
                (d[4],d[2]),
                (d[0],d[3]),
            ),
            qubit.CX((a[2],a[3])),
            
            qubit.H(
                    a[0],
                    a[1],
                   ),
            qubit.CX(
                (a[1],d[3]),
            ),
            qubit.SZdg(d[0]),
            # qubit.CH(a[2],d[0]),
            qubit.RY[-pi/4](d[0]),
            qubit.CZ(a[2],d[0]),
            qubit.RY[pi/4](d[0]),
            
            qubit.SZdg(d[5]),
            # qubit.CH(a[3], d[5]),
            qubit.RY[-pi/4](d[5]),
            qubit.CZ(a[3],d[5]),
            qubit.RY[pi/4](d[5]),
            qubit.CX(a[0],d[2]),
            qubit.CX(d[4],d[1]),
            
            qubit.SZdg(d[1]),
            # qubit.CH(a[2],d[1]),
            qubit.RY[-pi/4](d[1]),
            qubit.CZ(a[2],d[1]),
            qubit.RY[pi/4](d[1]),
            qubit.SZdg(d[6]),
            # qubit.CH(a[3],d[6]),
            qubit.RY[-pi/4](d[6]),
            qubit.CZ(a[3],d[6]),
            qubit.RY[pi/4](d[6]),
            
            ####### 
            # ''' qubit.RXX[pi/4](d[3],d[4]) '''
            qubit.H(d[3]),
            qubit.H(d[4]),
            # qubit.RZZ[pi/4](d[3],d[4]),
            qubit.CX(d[3],d[4]),
            qubit.RZ[pi/4](d[4]),
            qubit.CX(d[3],d[4]),
            qubit.H(d[3]),
            qubit.H(d[4]), 
            #########
            qubit.CX((d[0],a[4])),
            qubit.H(a[5]),
            qubit.CX((a[5],d[5])),
            
            qubit.CX(d[5],a[4]),
            qubit.CX((a[5],d[0])),
            qubit.SZdg(d[4]),
            # qubit.CH(a[2],d[4]),
            qubit.RY[-pi/4](d[4]),
            qubit.CZ(a[2],d[4]),
            qubit.RY[pi/4](d[4]),
            qubit.CX(a[1],d[3]),
            qubit.H(a[1]),
            qubit.Measure(a[1]) > flag_z[1],
            
            qubit.CX(d[3], d[2]),
            qubit.CX(d[4], a[4]),

            qubit.SZdg(d[3]),
            # qubit.CH(a[3], d[3]),
            qubit.RY[-pi/4](d[3]),
            qubit.CZ(a[3],d[3]),
            qubit.RY[pi/4](d[3]),
            qubit.CX(a[0], d[2]),
            qubit.H(a[0]),
            qubit.Measure(a[0]) > flag_z[0],

            qubit.CX(a[5], d[3]),
            qubit.SZdg(d[2]),
            # qubit.CH(a[2], d[2]),
            qubit.RY[-pi/4](d[2]),
            qubit.CZ(a[2],d[2]),
            qubit.RY[pi/4](d[2]),

            qubit.CX(a[2], a[3]),
            qubit.CX(d[3], a[4]),
            qubit.Measure(a[3]) > flag_x[0],
            qubit.Measure(a[4]) > flag_x[1],
            qubit.CX(a[5], d[4]),
            qubit.H(a[5]),
            qubit.Measure(a[5]) > flag_x[2],
            qubit.H(a[2]),
            qubit.Measure(a[2]) > flag_z[2],
        )
        
        self.extend(
            reject.set(flag_z[0] | flag_z[1] | flag_z[2] | flag_x[0] | flag_x[1] | flag_x[2]),
            # Reject on the results of the `reject` bit. 0 is good. 1 means the prep failed.
        )
        Comment('\n End FT preparing a |H> state with 30 two-qubit gates'),
        
class PrepHStateFTRUS(Block):
    """Prepare a |+H> state fault tolerantly using repeat-until-success.

    By using an encoding circuit to prepare logical|+H>, measuring the logical
    Hadamard with a flag, doing a QED round, and post-selecting based on non-trivial measurements.

    Arguments:
        d: Data qubits (size 7)
        a: Axillary qubits (size 2)
        out: Measurement outputs (size 2). out[0] is the Measure H result and out[1] is the flag result.
        reject: Whether the procedure failed and should be rejected. 0 it is good, 1 prep failed.
    """

    def __init__(
        self,
        d: QReg,
        a: QReg,
        out: CReg,
        reject: Bit,
        flag_x: CReg,
        flag_z: CReg,
        flags: CReg,
        last_raw_syn_x: CReg,
        last_raw_syn_z: CReg,
        limit: int,
    ) -> None:
        """Initialize PrepHStateFTRUS block for repeat-until-success |+H> preparation.

        Args:
            d: Data qubits (size 7) for the Steane code.
            a: Ancillary qubits (size 2) for measurements.
            out: Measurement outputs (size 2). out[0] is the Hadamard measurement,
                out[1] is the flag result.
            reject: Bit indicating preparation failure (0 for success, 1 for failure).
            flag_x: Classical register for X stabilizer flags.
            flag_z: Classical register for Z stabilizer flags.
            flags: Combined flags register.
            last_raw_syn_x: Previous X syndrome measurements.
            last_raw_syn_z: Previous Z syndrome measurements.
            limit: Maximum number of preparation attempts.
        """
        super().__init__(
            # PrepHStateFT_8cnot PrepHStateFT PrepHStateFT_30gate
            PrepHStateFT_30gate(
                d,
                a,
                out,
                reject,
                flag_x,
                flag_z,
                flags,
                last_raw_syn_x,
                last_raw_syn_z,
            ),
            Repeat(limit - 1).block(
                If(reject != 0).Then(
                    # PrepHStateFT_8cnot PrepHStateFT PrepHStateFT_30gate
                    PrepHStateFT_30gate(
                        d,
                        a,
                        out,
                        reject,
                        flag_x,
                        flag_z,
                        flags,
                        last_raw_syn_x,
                        last_raw_syn_z,
                        condition_qed=False,
                    ),
                ),
            ),
        )

        if limit == 1:
            self.extend(Comment())

class PrepHStateFTRUS_8cx(Block):
    """Prepare a |+H> state fault tolerantly using repeat-until-success.

    By using an encoding circuit to prepare logical|+H>, measuring the logical
    Hadamard with a flag, doing a QED round, and post-selecting based on non-trivial measurements.

    Arguments:
        d: Data qubits (size 7)
        a: Axillary qubits (size 2)
        out: Measurement outputs (size 2). out[0] is the Measure H result and out[1] is the flag result.
        reject: Whether the procedure failed and should be rejected. 0 it is good, 1 prep failed.
    """

    def __init__(
        self,
        d: QReg,
        a: QReg,
        out: CReg,
        reject: Bit,
        flag_x: CReg,
        flag_z: CReg,
        flags: CReg,
        last_raw_syn_x: CReg,
        last_raw_syn_z: CReg,
        limit: int,
    ) -> None:
        """Initialize PrepHStateFTRUS block for repeat-until-success |+H> preparation.

        Args:
            d: Data qubits (size 7) for the Steane code.
            a: Ancillary qubits (size 2) for measurements.
            out: Measurement outputs (size 2). out[0] is the Hadamard measurement,
                out[1] is the flag result.
            reject: Bit indicating preparation failure (0 for success, 1 for failure).
            flag_x: Classical register for X stabilizer flags.
            flag_z: Classical register for Z stabilizer flags.
            flags: Combined flags register.
            last_raw_syn_x: Previous X syndrome measurements.
            last_raw_syn_z: Previous Z syndrome measurements.
            limit: Maximum number of preparation attempts.
        """
        super().__init__(
            # PrepHStateFT_8cnot PrepHStateFT PrepHStateFT_30gate
            PrepHStateFT_8cnot(
                d,
                a,
                out,
                reject,
                flag_x,
                flag_z,
                flags,
                last_raw_syn_x,
                last_raw_syn_z,
            ),
            Repeat(limit - 1).block(
                If(reject != 0).Then(
                    # PrepHStateFT_8cnot PrepHStateFT PrepHStateFT_30gate
                    PrepHStateFT_8cnot(
                        d,
                        a,
                        out,
                        reject,
                        flag_x,
                        flag_z,
                        flags,
                        last_raw_syn_x,
                        last_raw_syn_z,
                        condition_qed=False,
                    ),
                ),
            ),
        )

        if limit == 1:
            self.extend(Comment())

class PrepHStateFTRUS_noveri(Block):
    """Prepare a |+H> state fault tolerantly using repeat-until-success.

    By using an encoding circuit to prepare logical|+H>, measuring the logical
    Hadamard with a flag, doing a QED round, and post-selecting based on non-trivial measurements.

    Arguments:
        d: Data qubits (size 7)
        a: Axillary qubits (size 2)
        out: Measurement outputs (size 2). out[0] is the Measure H result and out[1] is the flag result.
        reject: Whether the procedure failed and should be rejected. 0 it is good, 1 prep failed.
    """

    def __init__(
        self,
        d: QReg,
        a: QReg,
        out: CReg,
        reject: Bit,
        flag_x: CReg,
        flag_z: CReg,
        flags: CReg,
        last_raw_syn_x: CReg,
        last_raw_syn_z: CReg,
        limit: int,
    ) -> None:
        """Initialize PrepHStateFTRUS block for repeat-until-success |+H> preparation.

        Args:
            d: Data qubits (size 7) for the Steane code.
            a: Ancillary qubits (size 2) for measurements.
            out: Measurement outputs (size 2). out[0] is the Hadamard measurement,
                out[1] is the flag result.
            reject: Bit indicating preparation failure (0 for success, 1 for failure).
            flag_x: Classical register for X stabilizer flags.
            flag_z: Classical register for Z stabilizer flags.
            flags: Combined flags register.
            last_raw_syn_x: Previous X syndrome measurements.
            last_raw_syn_z: Previous Z syndrome measurements.
            limit: Maximum number of preparation attempts.
        """
        super().__init__(
            # PrepHStateFT_8cnot PrepHStateFT PrepHStateFT_30gate
            PrepHStateFT_noveri(
                d,
                a,
                out,
                reject,
                flag_x,
                flag_z,
                flags,
                last_raw_syn_x,
                last_raw_syn_z,
            ),
            Repeat(limit - 1).block(
                If(reject != 0).Then(
                    # PrepHStateFT_8cnot PrepHStateFT PrepHStateFT_30gate
                    PrepHStateFT_noveri(
                        d,
                        a,
                        out,
                        reject,
                        flag_x,
                        flag_z,
                        flags,
                        last_raw_syn_x,
                        last_raw_syn_z,
                        condition_qed=False,
                    ),
                ),
            ),
        )

        if limit == 1:
            self.extend(Comment())
