use crate::definitions::DIGEST_SIZE_U32_WORDS;

use super::*;

#[derive(Clone, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(bound = "F: serde::Serialize + serde::de::DeserializeOwned")]
pub struct BaseFieldQuery<F: PrimeField, T: ColumnMajorMerkleTreeConstructor<F>> {
    pub index: usize,
    pub leaf_values_concatenated: Vec<F>,
    pub path: Vec<[u32; DIGEST_SIZE_U32_WORDS]>,
    pub _marker: core::marker::PhantomData<T>,
}

impl<F: PrimeField, T: ColumnMajorMerkleTreeConstructor<F>> Default for BaseFieldQuery<F, T> {
    fn default() -> Self {
        Self {
            index: 0,
            leaf_values_concatenated: vec![],
            path: vec![],
            _marker: core::marker::PhantomData,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(
    bound = "F: serde::Serialize + serde::de::DeserializeOwned, E: serde::Serialize + serde::de::DeserializeOwned"
)]
pub struct ExtensionFieldQuery<
    F: PrimeField,
    E: FieldExtension<F> + Field,
    T: ColumnMajorMerkleTreeConstructor<F>,
> {
    pub index: usize,
    pub leaf_values_concatenated: Vec<E>,
    pub path: Vec<[u32; DIGEST_SIZE_U32_WORDS]>,
    pub _marker: core::marker::PhantomData<T>,
}

impl<F: PrimeField, E: FieldExtension<F> + Field, T: ColumnMajorMerkleTreeConstructor<F>> Default
    for ExtensionFieldQuery<F, E, T>
{
    fn default() -> Self {
        Self {
            index: 0,
            leaf_values_concatenated: vec![],
            path: vec![],
            _marker: core::marker::PhantomData,
        }
    }
}
