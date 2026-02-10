use super::queries::*;
use super::*;
use crate::merkle_trees::MerkleTreeCapVarLength;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct WhirCommitment<F: PrimeField, T: ColumnMajorMerkleTreeConstructor<F>> {
    pub coset_caps: Vec<MerkleTreeCapVarLength>,
    pub _marker: core::marker::PhantomData<(F, T)>,
}

impl<F: PrimeField, T: ColumnMajorMerkleTreeConstructor<F>> Default for WhirCommitment<F, T> {
    fn default() -> Self {
        Self {
            coset_caps: vec![],
            _marker: core::marker::PhantomData,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Default)]
pub struct WhirBaseLayerCommitmentAndQueries<
    F: PrimeField,
    E: FieldExtension<F> + Field,
    T: ColumnMajorMerkleTreeConstructor<F>,
> {
    pub commitment: WhirCommitment<F, T>,
    pub num_columns: usize,
    pub evals: Vec<E>, // num_columns
    pub queries: Vec<BaseFieldQuery<F, T>>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Default)]
pub struct WhirIntermediateCommitmentAndQueries<
    F: PrimeField,
    E: FieldExtension<F> + Field,
    T: ColumnMajorMerkleTreeConstructor<F>,
> {
    pub commitment: WhirCommitment<F, T>,
    pub queries: Vec<ExtensionFieldQuery<F, E, T>>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Default)]
pub struct WhirPolyCommitProof<
    F: PrimeField,
    E: FieldExtension<F> + Field,
    T: ColumnMajorMerkleTreeConstructor<F>,
> {
    pub setup_commitment: WhirBaseLayerCommitmentAndQueries<F, E, T>,
    pub memory_commitment: WhirBaseLayerCommitmentAndQueries<F, E, T>,
    pub witness_commitment: WhirBaseLayerCommitmentAndQueries<F, E, T>,
    pub intermediate_whir_oracles: Vec<WhirIntermediateCommitmentAndQueries<F, E, T>>,
    pub sumcheck_polys: Vec<[E; 3]>,
    pub final_monomials: Vec<E>,
}
