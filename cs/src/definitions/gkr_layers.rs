#[derive(
    Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize, serde::Deserialize,
)]
pub enum GKRAddress {
    BaseLayerWitness(usize),
    BaseLayerMemory(usize),
    InnerLayer { layer: usize, offset: usize },
    Setup(usize),
    OptimizedOut(usize),
    Cached { layer: usize, offset: usize },
}

impl GKRAddress {
    pub const fn placeholder() -> Self {
        Self::OptimizedOut(0)
    }

    #[inline(always)]
    pub const fn offset(&self) -> usize {
        match self {
            Self::BaseLayerWitness(offset) => *offset,
            Self::BaseLayerMemory(offset) => *offset,
            Self::Setup(offset) => *offset,
            Self::InnerLayer { offset, .. } => *offset,
            Self::OptimizedOut(offset) => *offset,
            Self::Cached { offset, .. } => *offset,
        }
    }

    pub fn as_memory(&self) -> usize {
        let Self::BaseLayerMemory(offset) = self else {
            panic!("expected memory location")
        };
        *offset
    }
}
