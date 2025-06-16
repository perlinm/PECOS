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
from pecos.slr import Block, Comment, QReg
from numpy import pi

class SX(Block):
    """
    Square root of X.

    X -> X
    Z -> -Y

    Y -> Z
    """

    def __init__(self, q: QReg):
        if len(q.elems) != 7:
            msg = f"Size of register {len(q.elems)} != 7"
            raise Exception(msg)

        super().__init__(
            Comment("Logical SX"),
            qubit.SXdg(q[0]),
            qubit.SXdg(q[1]),
            qubit.SXdg(q[2]),
            qubit.SXdg(q[3]),
            qubit.SXdg(q[4]),
            qubit.SXdg(q[5]),
            qubit.SXdg(q[6]),
        )


class SXdg(Block):
    """
    Hermitian adjoint of the square root of X.

    X -> X
    Z -> Y

    Y -> -Z
    """

    def __init__(self, q: QReg):
        if len(q.elems) != 7:
            msg = f"Size of register {len(q.elems)} != 7"
            raise Exception(msg)

        super().__init__(
            Comment("Logical SXdg"),
            # qubit.SX(q),
            qubit.SX(q[0]),
            qubit.SX(q[1]),
            qubit.SX(q[2]),
            qubit.SX(q[3]),
            qubit.SX(q[4]),
            qubit.SX(q[5]),
            qubit.SX(q[6]),
        )


class SY(Block):
    """
    Square root of Y.

    X -> -Z
    Z -> X

    Y -> Y
    """

    def __init__(self, q: QReg):
        if len(q.elems) != 7:
            msg = f"Size of register {len(q.elems)} != 7"
            raise Exception(msg)

        super().__init__(
            Comment("Logical SY"),
            # qubit.SY(q),
            qubit.SY(q[0]),
            qubit.SY(q[1]),
            qubit.SY(q[2]),
            qubit.SY(q[3]),
            qubit.SY(q[4]),
            qubit.SY(q[5]),
            qubit.SY(q[6]),
        )


class SYdg(Block):
    """
    Square root of X.

    X -> Z
    Z -> -X

    Y -> Y
    """

    def __init__(self, q: QReg):
        if len(q.elems) != 7:
            msg = f"Size of register {len(q.elems)} != 7"
            raise Exception(msg)

        super().__init__(
            Comment("Logical SYdg"),
            # qubit.SYdg(q),
            qubit.SYdg(q[0]),
            qubit.SYdg(q[1]),
            qubit.SYdg(q[2]),
            qubit.SYdg(q[3]),
            qubit.SYdg(q[4]),
            qubit.SYdg(q[5]),
            qubit.SYdg(q[6]),
        )


class SZ(Block):
    """
    Square root of Z. Also known as the S gate.
    diag(1, i)

    X -> Y
    Z -> Z

    Y -> -X
    """

    def __init__(self, q: QReg):
        if len(q.elems) != 7:
            msg = f"Size of register {len(q.elems)} != 7"
            raise Exception(msg)

        super().__init__(
            Comment("Logical SZ"),
            # qubit.SZdg(q),
            qubit.SZdg(q[0]),
            qubit.SZdg(q[1]),
            qubit.SZdg(q[2]),
            qubit.SZdg(q[3]),
            qubit.SZdg(q[4]),
            qubit.SZdg(q[5]),
            qubit.SZdg(q[6]),
        )


class SZdg(Block):
    """
    Hermitian adjoint of the square root of Z. Also known as the Sdg gate.
    diag(1, -i)

    X -> -Y
    Z -> Z

    Y -> X
    """

    def __init__(self, q: QReg):
        if len(q.elems) != 7:
            msg = f"Size of register {len(q.elems)} != 7"
            raise Exception(msg)

        super().__init__(
            Comment("Logical SZdg"),
            # qubit.SZ(q),
            qubit.SZ(q[0]),
            qubit.SZ(q[1]),
            qubit.SZ(q[2]),
            qubit.SZ(q[3]),
            qubit.SZ(q[4]),
            qubit.SZ(q[5]),
            qubit.SZ(q[6]),
        )

class direct_t(Block):
    """
    The direct implementation of a T gate
    """

    def __init__(self, q: QReg):
        if len(q.elems) != 7:
            msg = f"Size of register {len(q.elems)} != 7"
            raise Exception(msg)

        super().__init__(
            Comment("Logical direct T"),
            qubit.RZ[pi/4](q[4]),
            qubit.RZ[pi/4](q[5]),
            qubit.RZ[pi/4](q[6]),
        )