use field::{Field, FieldExtension};

pub trait Transcript<F: Field> {
    type Extension: FieldExtension<F> + Field;

    fn commit_initial(input: &[u32]) -> Self;

    fn absorb_base(&mut self, s: &F);
    fn absorb_bases(&mut self, s: &[F]) {
        for x in s {
            self.absorb_base(x);
        }
    }
    fn absorb_extension(&mut self, s: &Self::Extension) {
        for coeff in s.coeffs_in_base() {
            self.absorb_base(coeff);
        }
    }
    fn absorb_extensions(&mut self, s: &[Self::Extension]) {
        for x in s {
            self.absorb_extension(x);
        }
    }

    fn draw_challenge_base(&mut self) -> F;
    fn draw_challenge_extension(&mut self) -> Self::Extension;

    fn draw_challenges_base(&mut self, len: usize) -> Vec<F> {
        (0..len).map(|_| self.draw_challenge_base()).collect()
    }

    fn draw_challenges_extension(&mut self, len: usize) -> Vec<Self::Extension> {
        (0..len).map(|_| self.draw_challenge_extension()).collect()
    }
}

#[cfg(test)]
pub struct Blake2sTranscript {
    seed: transcript::Seed,
}

#[cfg(test)]
impl Transcript<field::Mersenne31Field> for Blake2sTranscript {
    type Extension = field::Mersenne31Complex;

    fn commit_initial(input: &[u32]) -> Self {
        let seed = transcript::Blake2sTranscript::commit_initial(input);

        Self { seed }
    }

    fn absorb_base(&mut self, s: &field::Mersenne31Field) {
        transcript::Blake2sTranscript::commit_with_seed(&mut self.seed, &[s.to_reduced_u32()]);
    }

    fn draw_challenge_base(&mut self) -> field::Mersenne31Field {
        let mut dst = [0u32; 8];
        transcript::Blake2sTranscript::draw_randomness(&mut self.seed, &mut dst);

        field::Mersenne31Field::from_nonreduced_u32(dst[0])
    }

    fn draw_challenges_base(&mut self, len: usize) -> Vec<field::Mersenne31Field> {
        let mut dst = vec![0u32; ((len + 7) / 8) * 8];

        transcript::Blake2sTranscript::draw_randomness(&mut self.seed, &mut dst);

        (0..len)
            .map(|i| field::Mersenne31Field::from_nonreduced_u32(dst[i]))
            .collect()
    }

    fn draw_challenge_extension(&mut self) -> Self::Extension {
        let mut dst = [0u32; 8];

        transcript::Blake2sTranscript::draw_randomness(&mut self.seed, &mut dst);

        field::Mersenne31Complex::from_coeffs_in_base(&[
            field::Mersenne31Field::from_nonreduced_u32(dst[0]),
            field::Mersenne31Field::from_nonreduced_u32(dst[1]),
        ])
    }

    fn draw_challenges_extension(&mut self, len: usize) -> Vec<Self::Extension> {
        let mut dst = vec![0u32; ((2 * len + 7) / 8) * 8];

        transcript::Blake2sTranscript::draw_randomness(&mut self.seed, &mut dst);

        (0..2 * len)
            .step_by(2)
            .map(|i| {
                field::Mersenne31Complex::from_coeffs_in_base(&[
                    field::Mersenne31Field::from_nonreduced_u32(dst[i]),
                    field::Mersenne31Field::from_nonreduced_u32(dst[i + 1]),
                ])
            })
            .collect()
    }
}
