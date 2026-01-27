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

    pub const fn is_cache(&self) -> bool {
        if let Self::Cached { .. } = self {
            true
        } else {
            false
        }
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

    #[track_caller]
    pub fn assert_as_dependency_for_layer(&self, output_layer: usize) {
        match self {
            Self::BaseLayerWitness(..) | Self::BaseLayerMemory(..) | Self::Setup(..) => {
                assert_eq!(output_layer, 1)
            }
            Self::InnerLayer { layer, .. } => assert_eq!(output_layer, *layer + 1),
            Self::OptimizedOut(..) => unreachable!(),
            Self::Cached { layer, .. } => assert_eq!(output_layer, *layer + 1),
        }
    }

    #[track_caller]
    pub fn assert_as_layer(&self, output_layer: usize) {
        match self {
            Self::BaseLayerWitness(..) | Self::BaseLayerMemory(..) => {
                assert_eq!(output_layer, 0)
            }
            Self::InnerLayer { .. } | Self::Setup(..) | Self::OptimizedOut(..)  => {
                unreachable!();
            },

            Self::Cached { layer, .. } => assert_eq!(output_layer, *layer),
        }
    }
}
