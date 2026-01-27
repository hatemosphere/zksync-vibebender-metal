use super::*;

pub const DECODER_LOOKUP_FORMAL_SET_INDEX: usize = usize::MAX;

#[derive(Clone, Debug, Hash, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct NoFieldSingleColumnLookupRelation {
    pub input: NoFieldLinearRelation,
    pub lookup_set_index: usize,   
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct NoFieldVectorLookupRelation {
    pub columns: Box<[NoFieldLinearRelation]>,
    pub lookup_set_index: usize,   
}
