use crate::{circuit::{Circuit, CircuitError}, sumcheck::{SumcheckInstanceProof, batch_prove}, transcript::Transcript};
use field::{Field, FieldExtension};

pub mod layer_sumcheck;
use layer_sumcheck::LayerSumcheckProver;

pub struct GKRProof<F, E> {
    inputs: Vec<F>,
    outputs: Vec<F>,
    layer_proofs: Vec<SumcheckInstanceProof<F, E>>,
}

impl From<CircuitError> for GKRError {
    fn from(value: CircuitError) -> Self {
        Self::CircuitError(value)
    }
}

pub enum GKRError {
    CircuitError(CircuitError)
}

pub fn prove<F, E, T>(
    circuit: Circuit<F>,
    inputs: Vec<F>
) -> Result<GKRProof<F, E>, GKRError> 
where
    F: Field,
    E: FieldExtension<F> + Field,
    T: Transcript<F, Extension = E>
{
    // let layer_values = circuit.evaluate(&inputs)?;
    // let mut transcript = Transcript::commit_initial(&circuit.to_u32s());

    // for layer in circuit.layers {
    //     let (x, y) = batch_prove::<F, E, T>(
    //         vec![LayerSumcheckProver::new(z, layer, prev_layer_values, claim)], 
    //         &mut transcript
    //     );
    // }

    todo!()
}