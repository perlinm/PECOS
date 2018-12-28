#   Copyright 2018 National Technology & Engineering Solutions of Sandia,
#   LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS,
#   the U.S. Government retains certain rights in this software.
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

"""
A simple wrapper for the ProjectQ simulator.

Compatibility checked for: cirq version (Git July 26, 2018)
"""

import cirq
from . import bindings
from .bindings import MakeFunc


class State(object):
    """
    Represents the stabilizer state.
    """
    gate_dict = bindings.gate_dict

    def __init__(self, num_qubits):
        """
        Initializes the stabilizer state.

        :param num_qubits: Number of qubits to represent.
        """

        if not isinstance(num_qubits, int):
            raise Exception('``num_qubits`` should be of type ``int.``')

        self.num_qubits = num_qubits
        self.qubits = [cirq.LineQubit(q) for q in range(num_qubits)]
        self.simulator = cirq.google.XmonSimulator()

        initq = cirq.Circuit()
        initq.append([cirq.ops.measure(q) for q in self.qubits])
        results = self.simulator.simulate(initq)
        self.state = results.final_state

    def add_gate(self, symbol, gate_obj, make_func=True, angle=False):

        if symbol in self.gate_dict:
            print('WARNING: Can not add gate as the symbol has already been taken.')
        else:
            if make_func:
                self.gate_dict[symbol] = MakeFunc(gate_obj, angle=angle).func
            else:
                self.gate_dict[symbol] = gate_obj

    def run_gate(self, symbol, locations, flush=True, **gate_kwargs):
        """
        Applies the gate indicated by `symbol` to the gate locations in the set `locations`.

        Args:
            symbol (str): String representing gate.
            locations (set of int, set of tuple of int):
            flush (bool): Whether to flush. Note: Measurements and initializations will flush anyway.
            **gate_kwargs: A dictionary specifying extra parameters for the gate.

        Returns:

        """

        # TODO: Why do we have the output option?

        circuit = cirq.Circuit()

        output = {}

        # results = self.gate_dict[symbol](self, location, **gate_kwargs)
        circuit.append([self.gate_dict[symbol](self, location, **gate_kwargs) for location in locations])

        results = self.simulator.simulate(circuit, initial_state=self.state)
        self.state = results.final_state

        # results.measurements
        # output[location] = meas_result

        return output

    def run_circuit(self, circuit, output=True):
        """

        Args:
            circuit (QuantumCircuit): A circuit instance or object with an appropriate items() generator.
            output (bool): Whether to return an output.

        Returns (list): If output is True then the circuit output is returned. Note that this output format may differ
        from what a ``circuit_runner`` will return for the same method named ``run_circuit``.

        """

        results = []

        for symbol, locations, gate_kwargs in circuit.items(params=True):
            gate_output = self.run_gate(symbol, locations, output, flush=False, **gate_kwargs)
            if output:
                results.append(gate_output)

        self.eng.flush()
        return results
