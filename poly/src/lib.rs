mod eq_poly;
mod multilinear;
mod split_eq_poly;
mod univariate;

const PAR_THRESHOLD: usize = 16;

pub use eq_poly::EqPoly;
pub use multilinear::MultiLinearPoly;
pub use split_eq_poly::SplitEqPoly;
pub use univariate::UniPoly;
