use field::Field;
use std::fmt;

use crate::poly::MultilinearPoly;

#[derive(Debug, Clone, PartialEq)]
/// Representation of `coeff * left * right`
pub struct QuadTerm<F> {
    pub coeff: F,
    pub left: usize,
    pub right: usize,
}

impl<F> QuadTerm<F> {
    pub fn new(coeff: F, left: usize, right: usize) -> Self {
        Self {
            coeff,
            left: left.min(right),
            right: right.max(left),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
/// Representation of a sum of the form `k_1*a_1*b_1 + k_2*a_2*b_2 + ...`
pub struct QuadGate<F> {
    pub terms: Vec<QuadTerm<F>>,
}

impl<F: Field> QuadGate<F> {
    pub fn new(terms: Vec<QuadTerm<F>>) -> Self {
        Self { terms }
    }

    pub fn evaluate(&self, prev_layer_values: &[F]) -> F {
        let mut result = F::ZERO;

        for term in &self.terms {
            let mut contrib = term.coeff;
            contrib.mul_assign(&prev_layer_values[term.left]);
            contrib.mul_assign(&prev_layer_values[term.right]);

            result.add_assign(&contrib);
        }

        result
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum CircuitError {
    LayerNotPowerOfTwo {
        layer_idx: usize,
        size: usize,
    },
    InputsNotPowerOfTwo {
        num_inputs: usize,
    },
    InvalidWireIndex {
        layer_idx: usize,
        gate_idx: usize,
        wire_idx: usize,
        prev_layer_size: usize,
    },
    EmptyCircuit,
    InvalidNumberOfInputs {
        expected: usize,
        given: usize,
    },
}

impl fmt::Display for CircuitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CircuitError::LayerNotPowerOfTwo { layer_idx, size } => {
                write!(
                    f,
                    "Layer {} has size {}, which is not a power of 2",
                    layer_idx, size
                )
            }
            CircuitError::InputsNotPowerOfTwo { num_inputs } => {
                write!(f, "Number of inputs {} is not a power of 2", num_inputs)
            }
            CircuitError::InvalidWireIndex {
                layer_idx,
                gate_idx,
                wire_idx,
                prev_layer_size,
            } => {
                write!(
                    f,
                    "Layer {} gate {} references wire {} but previous layer has only {} wires",
                    layer_idx, gate_idx, wire_idx, prev_layer_size
                )
            }
            CircuitError::EmptyCircuit => write!(f, "Circuit has no layers"),
            CircuitError::InvalidNumberOfInputs { expected, given } => {
                write!(f, "Expected {} inputs, got {}", expected, given)
            }
        }
    }
}

impl std::error::Error for CircuitError {}

#[derive(Debug, Clone, PartialEq)]
pub struct CircuitLayer<F> {
    pub gates: Vec<QuadGate<F>>,
    pub num_vars: usize,
}

impl<F: Field> CircuitLayer<F> {
    pub fn new(gates: Vec<QuadGate<F>>) -> Result<Self, CircuitError> {
        if gates.is_empty() {
            return Ok(Self {
                gates: vec![],
                num_vars: 0,
            });
        }

        let num_vars = gates.len().next_power_of_two().ilog2() as usize;

        Ok(Self { gates, num_vars })
    }

    pub fn evaluate(&self, prev_layer_values: &[F]) -> Vec<F> {
        self.gates
            .iter()
            .map(|gate| gate.evaluate(prev_layer_values))
            .collect()
    }

    pub fn num_gates(&self) -> usize {
        self.gates.len()
    }

    pub fn is_empty(&self) -> bool {
        self.gates.is_empty()
    }

    /// Creates an MLE over (gate, left_wire, right_wire)
    pub fn selector(&self, num_wire_vars: usize) -> MultilinearPoly<F> {
        let num_vars = self.num_vars + 2 * num_wire_vars;

        let mut evals = vec![F::ZERO; 1 << num_vars];
        for (gate_idx, gate) in self.gates.iter().enumerate() {
            for term in &gate.terms {
                let idx =
                    term.right + (term.left << num_wire_vars) + (gate_idx << (2 * num_wire_vars));
                evals[idx] = term.coeff;
            }
        }

        MultilinearPoly::new(evals)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Circuit<F> {
    pub layers: Vec<CircuitLayer<F>>,
    pub num_inputs: usize,
    pub log_inputs: usize,
}

impl<F: Field> Circuit<F> {
    pub fn new(layers: Vec<CircuitLayer<F>>, num_inputs: usize) -> Result<Self, CircuitError> {
        if layers.is_empty() {
            return Err(CircuitError::EmptyCircuit);
        }

        let log_inputs = num_inputs.next_power_of_two().ilog2() as usize;

        // check that wire indices are valid
        for (layer_idx, layer) in layers.iter().enumerate() {
            let prev_layer_size = if layer_idx + 1 < layers.len() {
                layers[layer_idx + 1].num_gates()
            } else {
                num_inputs
            };

            for (gate_idx, gate) in layer.gates.iter().enumerate() {
                for term in &gate.terms {
                    if term.left > prev_layer_size {
                        return Err(CircuitError::InvalidWireIndex {
                            layer_idx,
                            gate_idx,
                            wire_idx: term.left,
                            prev_layer_size,
                        });
                    }
                    if term.right > prev_layer_size {
                        return Err(CircuitError::InvalidWireIndex {
                            layer_idx,
                            gate_idx,
                            wire_idx: term.right,
                            prev_layer_size,
                        });
                    }
                }
            }
        }

        Ok(Self {
            layers,
            num_inputs,
            log_inputs,
        })
    }

    pub fn evaluate(&self, inputs: &[F]) -> Result<Vec<Vec<F>>, CircuitError> {
        if inputs.len() != self.num_inputs {
            return Err(CircuitError::InvalidNumberOfInputs {
                expected: self.num_inputs,
                given: inputs.len(),
            });
        }

        let mut layer_values = Vec::with_capacity(self.layers.len());

        let mut current_values = inputs.to_vec();

        for layer in self.layers.iter().rev() {
            current_values = layer.evaluate(&current_values);
            layer_values.push(current_values.clone());
        }

        layer_values.reverse();

        Ok(layer_values)
    }

    /// Get the number of variables for a specific layer
    pub fn num_vars_at(&self, layer_idx: usize) -> Option<usize> {
        if layer_idx < self.layers.len() {
            Some(self.layers[layer_idx].num_vars)
        } else if layer_idx == self.layers.len() {
            Some(self.log_inputs)
        } else {
            None
        }
    }

    pub fn depth(&self) -> usize {
        self.layers.len()
    }

    pub fn output_size(&self) -> usize {
        self.layers.first().map(|l| l.num_gates()).unwrap_or(0)
    }

    pub fn to_u32s(&self) -> Vec<u32> {
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use field::Mersenne31Field as F;

    #[test]
    fn test_quad_gate_evaluate() {
        let prev_values: Vec<F> = [5u32, 7, 11]
            .into_iter()
            .map(F::from_nonreduced_u32)
            .collect();

        let gate = QuadGate::new(vec![
            QuadTerm::new(F::from_nonreduced_u32(2), 0, 1), // 2*5*7 = 70
            QuadTerm::new(F::from_nonreduced_u32(3), 1, 2), // 3*7*11 = 231
        ]);
        let result = gate.evaluate(&prev_values);

        assert_eq!(result, F::from_nonreduced_u32(301));
    }

    #[test]
    fn test_circuit_layer() {
        let prev_values: Vec<F> = [1u32, 5, 7]
            .into_iter()
            .map(F::from_nonreduced_u32)
            .collect();

        let gates = vec![
            QuadGate::new(vec![QuadTerm::new(F::ONE, 0, 1)]), // 1*1*5
            QuadGate::new(vec![QuadTerm::new(F::ONE, 0, 2)]), // 1*1*7
        ];

        let layer = CircuitLayer::new(gates).unwrap();
        assert_eq!(layer.num_vars, 1);

        let result = layer.evaluate(&prev_values);
        assert_eq!(
            result,
            vec![F::from_nonreduced_u32(5), F::from_nonreduced_u32(7)]
        );
    }

    #[test]
    fn test_simple_circuit() {
        let layer0 = CircuitLayer::new(vec![QuadGate::new(vec![
            QuadTerm::new(F::ONE, 0, 1), // 1 * input[0] * input[1] = a * b
        ])])
        .unwrap();

        let circuit = Circuit::new(vec![layer0], 2).unwrap();

        let inputs = vec![F::from_nonreduced_u32(3), F::from_nonreduced_u32(5)];
        let result = circuit.evaluate(&inputs).unwrap();

        assert_eq!(result[0][0], F::from_nonreduced_u32(15));
    }

    #[test]
    fn test_selector() {
        use crate::poly::Polynomial;

        // Layer with 2 gates, previous layer has 4 wires (2 wire vars)
        // Gate 0: 3*w[0]*w[1] + 5*w[1]*w[2]
        // Gate 1: 7*w[2]*w[3]
        let gates = vec![
            QuadGate::new(vec![
                QuadTerm::new(F::from_nonreduced_u32(3), 0, 1),
                QuadTerm::new(F::from_nonreduced_u32(5), 1, 2),
            ]),
            QuadGate::new(vec![QuadTerm::new(F::from_nonreduced_u32(7), 2, 3)]),
        ];

        let layer = CircuitLayer::new(gates).unwrap();
        let num_wire_vars = 2;

        let selector = layer.selector(num_wire_vars);

        // Check num_vars: 1 (gate) + 2*2 (left+right) = 5
        assert_eq!(selector.num_vars(), 5);

        // Check specific entries
        // Gate 0, left=0, right=1: should be 3
        let idx_0_0_1 = 1 + (0 << 2) + (0 << 4); // = 1
        assert_eq!(selector.evals()[idx_0_0_1], F::from_nonreduced_u32(3));

        // Gate 0, left=1, right=2: should be 5
        let idx_0_1_2 = 2 + (1 << 2) + (0 << 4); // = 2 + 4 = 6
        assert_eq!(selector.evals()[idx_0_1_2], F::from_nonreduced_u32(5));

        // Gate 1, left=2, right=3: should be 7
        let idx_1_2_3 = 3 + (2 << 2) + (1 << 4); // = 3 + 8 + 16 = 27
        assert_eq!(selector.evals()[idx_1_2_3], F::from_nonreduced_u32(7));

        // Some zero entry (gate 0, left=0, right=0)
        let idx_0_0_0 = 0 + (0 << 2) + (0 << 4); // = 0
        assert_eq!(selector.evals()[idx_0_0_0], F::ZERO);
    }
}
