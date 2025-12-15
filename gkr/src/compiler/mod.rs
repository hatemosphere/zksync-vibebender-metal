mod circuit_builder;
mod node;
mod scheduler;

#[cfg(test)]
mod tests {
    use super::circuit_builder::CircuitBuilder;
    use field::{Field, Mersenne31Field as F};

    #[test]
    fn test_compile_simple_multiplication() {
        let mut builder = CircuitBuilder::<F>::new();

        let x = builder.input();
        let y = builder.input();
        let z = builder.mul(x, y);
        builder.output(z);

        let circuit = builder.compile().unwrap();

        assert_eq!(circuit.depth(), 1);
        assert_eq!(circuit.num_inputs, 3);

        let inputs = vec![F::ONE, F::from_nonreduced_u32(3), F::from_nonreduced_u32(5)];
        let result = circuit.evaluate(&inputs).unwrap();

        assert_eq!(result[0][0], F::from_nonreduced_u32(15));
    }

    #[test]
    fn test_compile_addition() {
        let mut builder = CircuitBuilder::<F>::new();

        let x = builder.input();
        let y = builder.input();
        let z = builder.add(x, y);
        builder.output(z);

        let circuit = builder.compile().unwrap();

        assert_eq!(circuit.depth(), 1);
        assert_eq!(circuit.num_inputs, 3);

        let inputs = vec![
            F::ONE,
            F::from_nonreduced_u32(7),
            F::from_nonreduced_u32(11),
        ];
        let result = circuit.evaluate(&inputs).unwrap();

        assert_eq!(result[0][0], F::from_nonreduced_u32(18));
    }

    #[test]
    fn test_compile_complex_expression() {
        let mut builder = CircuitBuilder::<F>::new();

        // Circuit: (a + b) * (c + d)
        let a = builder.input();
        let b = builder.input();
        let c = builder.input();
        let d = builder.input();

        let sum1 = builder.add(a, b);
        let sum2 = builder.add(c, d);
        let product = builder.mul(sum1, sum2);
        builder.output(product);

        let circuit = builder.compile().unwrap();
        assert_eq!(circuit.num_inputs, 5);
        assert_eq!(circuit.depth(), 2);

        // Evaluate: (2 + 3) * (4 + 5) = 5 * 9 = 45
        let inputs = vec![
            F::ONE,
            F::from_nonreduced_u32(2),
            F::from_nonreduced_u32(3),
            F::from_nonreduced_u32(4),
            F::from_nonreduced_u32(5),
        ];
        let result = circuit.evaluate(&inputs).unwrap();
        assert_eq!(result[0][0], F::from_nonreduced_u32(45));
    }

    #[test]
    fn test_compile_with_constants() {
        let mut builder = CircuitBuilder::<F>::new();

        let x = builder.input();
        let c = builder.constant(F::from_nonreduced_u32(10));
        let z = builder.mul(x, c);
        builder.output(z);

        let circuit = builder.compile().unwrap();

        // Evaluate: 7 * 10 = 70
        let inputs = vec![F::ONE, F::from_nonreduced_u32(7)];
        let result = circuit.evaluate(&inputs).unwrap();

        assert_eq!(result[0][0], F::from_nonreduced_u32(70));
    }

    #[test]
    fn test_compile_constant_folding() {
        let mut builder = CircuitBuilder::<F>::new();

        // Constants should be folded at compile time
        let c1 = builder.constant(F::from_nonreduced_u32(3));
        let c2 = builder.constant(F::from_nonreduced_u32(5));
        let product = builder.mul(c1, c2);
        builder.output(product);

        let circuit = builder.compile().unwrap();

        assert_eq!(circuit.depth(), 1);

        let result = circuit.evaluate(&[F::ONE]).unwrap();
        assert_eq!(result[0][0], F::from_nonreduced_u32(15));
    }

    #[test]
    fn test_compile_scale_op() {
        let mut builder = CircuitBuilder::<F>::new();

        let x = builder.input();
        let scaled = builder.scale(F::from_nonreduced_u32(7), x);
        builder.output(scaled);

        let circuit = builder.compile().unwrap();

        // Evaluate: 7 * 4 = 28
        let inputs = vec![F::ONE, F::from_nonreduced_u32(4)];
        let result = circuit.evaluate(&inputs).unwrap();
        assert_eq!(result[0][0], F::from_nonreduced_u32(28));
    }

    #[test]
    fn test_compile_sub() {
        let mut builder = CircuitBuilder::<F>::new();

        let x = builder.input();
        let y = builder.input();
        let diff = builder.sub(x, y);
        builder.output(diff);

        let circuit = builder.compile().unwrap();

        // Evaluate: 20 - 7 = 13
        let inputs = vec![
            F::ONE,
            F::from_nonreduced_u32(20),
            F::from_nonreduced_u32(7),
        ];
        let result = circuit.evaluate(&inputs).unwrap();

        assert_eq!(result[0][0], F::from_nonreduced_u32(13));
    }

    #[test]
    fn test_compile_multiple_outputs() {
        let mut builder = CircuitBuilder::<F>::new();

        let x = builder.input();
        let y = builder.input();

        let sum = builder.add(x, y);
        let product = builder.mul(x, y);

        builder.output(sum);
        builder.output(product);

        let circuit = builder.compile().unwrap();
        assert_eq!(circuit.num_inputs, 3);

        // Evaluate: x=3, y=5 -> sum=8, product=15
        let inputs = vec![F::ONE, F::from_nonreduced_u32(3), F::from_nonreduced_u32(5)];
        let result = circuit.evaluate(&inputs).unwrap();

        // Check both outputs are in the output layer
        let mut outputs = result[0].clone();
        assert!(outputs.len() >= 2);
        outputs.sort();

        assert_eq!(outputs[0], F::from_nonreduced_u32(8));
        assert_eq!(outputs[1], F::from_nonreduced_u32(15));
    }

    #[test]
    fn test_compile_poly_eval() {
        let mut builder = CircuitBuilder::<F>::new();

        // Polynomial: f(x) = 2x^2 + 3x + 1
        let x = builder.input();

        let x_linear = builder.linear(x);
        let x_squared = builder.mul(x, x_linear);

        // 2x^2
        let two_x_squared = builder.scale(F::from_nonreduced_u32(2), x_squared);

        // 3x
        let three_x = builder.scale(F::from_nonreduced_u32(3), x);

        // 2x^2 + 3x
        let sum1 = builder.add(two_x_squared, three_x);

        // 2x^2 + 3x + 1
        let one = builder.constant(F::ONE);
        let result = builder.add(sum1, one);

        builder.output(result);

        let circuit = builder.compile().unwrap();

        // Evaluate f(4) = 2*16 + 3*4 + 1 = 32 + 12 + 1 = 45
        let inputs = vec![F::ONE, F::from_nonreduced_u32(4)];
        let eval_result = circuit.evaluate(&inputs).unwrap();

        assert_eq!(eval_result[0][0], F::from_nonreduced_u32(45));
    }

    #[test]
    fn test_compile_axpy_operation() {
        let mut builder = CircuitBuilder::<F>::new();

        let x = builder.input();
        let y = builder.input();
        let result = builder.axpy(y, F::from_nonreduced_u32(3), x);
        builder.output(result);

        let circuit = builder.compile().unwrap();

        // y+3x = 10 + 3*4 = 22
        let inputs = vec![
            F::ONE,
            F::from_nonreduced_u32(4),
            F::from_nonreduced_u32(10),
        ];
        let eval_result = circuit.evaluate(&inputs).unwrap();

        assert_eq!(eval_result[0][0], F::from_nonreduced_u32(22));
    }

    #[test]
    fn test_compile_powers() {
        let mut builder = CircuitBuilder::<F>::new();

        // circuit: x * x * x * x
        let x = builder.input();
        let x_lin = builder.linear(x);
        let x2 = builder.mul(x, x_lin); // depth 1
        let x3 = builder.mul(x2, x); // depth 2
        let x4 = builder.mul(x3, x); // depth 3
        builder.output(x4);

        let circuit = builder.compile().unwrap();

        assert_eq!(circuit.depth(), 3);

        let inputs = vec![F::ONE, F::from_nonreduced_u32(2)];
        let result = circuit.evaluate(&inputs).unwrap();

        assert_eq!(result[0][0], F::from_nonreduced_u32(16));
    }

    #[test]
    fn test_compile_big_sum() {
        let mut builder = CircuitBuilder::<F>::new();
        let mut sum = builder.input();
        for _ in 2..=100 {
            let y = builder.input();
            sum = builder.add(sum, y);
        }

        builder.output(sum);
        let circuit = builder.compile().unwrap();

        let mut inputs = Vec::with_capacity(101);
        inputs.push(F::ONE);
        inputs.extend((1..=100).map(F::from_nonreduced_u32));

        let result = circuit.evaluate(&inputs).unwrap();
        assert_eq!(result[0][0], F::from_nonreduced_u32(5050));
    }

    #[test]
    fn test_compile_factorial() {
        let mut builder = CircuitBuilder::<F>::new();
        let mut prod = builder.input();
        for _ in 2..10 {
            let y = builder.input();
            prod = builder.mul(prod, y);
        }
        builder.output(prod);

        let circuit = builder.compile().unwrap();

        let mut inputs = Vec::with_capacity(11);
        inputs.push(F::ONE);
        inputs.extend((1..10).map(F::from_nonreduced_u32));

        let result = circuit.evaluate(&inputs).unwrap();
        assert_eq!(result[0][0], F::from_nonreduced_u32(362880));
    }

    #[test]
    fn test_cse_optimization() {
        let mut builder = CircuitBuilder::<F>::new();

        let x = builder.input();
        let y = builder.input();

        // Compute x*y multiple times - should be CSE'd to a single multiplication
        let xy1 = builder.mul(x, y);
        let xy2 = builder.mul(x, y);
        let xy3 = builder.mul(x, y);

        // Use all three results
        let sum = builder.add(xy1, xy2);
        let result = builder.add(sum, xy3);

        builder.output(result);

        let circuit = builder.compile().unwrap();

        // Should only have 1 multiplication gate (x*y), not 3
        // Count multiplication gates (depth 1 for this circuit)
        let mult_layer = &circuit.layers[circuit.depth() - 1];
        assert_eq!(mult_layer.num_gates(), 1);

        // Verify 3*(x*y)
        let inputs = vec![
            F::ONE,
            F::from_nonreduced_u32(7),
            F::from_nonreduced_u32(11),
        ];
        let eval_result = circuit.evaluate(&inputs).unwrap();
        assert_eq!(eval_result[0][0], F::from_nonreduced_u32(3 * 7 * 11));
    }
}
