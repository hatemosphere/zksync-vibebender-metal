use crate::witness_proxy::WitnessProxy;
use cs::oracle::*;
use cs::witness_placer::scalar_witness_type_set::ScalarWitnessTypeSet;
use cs::tables::TableDriver;
use field::{Mersenne31Field, PrimeField};

pub struct SimpleWitnessProxy<'a, O: Oracle<F> + 'a, F: PrimeField = Mersenne31Field> {
    pub(crate) witness_row: &'a mut [F],
    pub(crate) memory_row: &'a mut [F],
    pub(crate) scratch_space: &'a mut [F],
    pub(crate) table_driver: &'a TableDriver<F>,
    pub(crate) multiplicity_counting_scratch: &'a mut [u32],
    pub(crate) lookup_mapping_row: &'a mut [u32],
    pub(crate) oracle: &'a O,
    pub(crate) absolute_row_idx: usize,
}

impl<'a, O: Oracle<F> + 'a, F: PrimeField> WitnessProxy<F, ScalarWitnessTypeSet<F, true>>
    for SimpleWitnessProxy<'a, O, F>
{
    #[inline(always)]
    fn get_memory_place(&self, idx: usize) -> F {
        debug_assert!(idx < self.memory_row.len());
        unsafe { *self.memory_row.get_unchecked(idx) }
    }

    #[inline(always)]
    fn get_witness_place(&self, idx: usize) -> F {
        debug_assert!(idx < self.witness_row.len());
        unsafe { *self.witness_row.get_unchecked(idx) }
    }

    #[inline(always)]
    fn get_scratch_place(&self, idx: usize) -> F {
        debug_assert!(idx < self.scratch_space.len());
        unsafe { *self.scratch_space.get_unchecked(idx) }
    }

    #[inline(always)]
    fn get_oracle_value(&self, placeholder: Placeholder, subindex: usize) -> F {
        self.oracle
            .get_witness_from_placeholder(placeholder, subindex, self.absolute_row_idx)
    }

    #[inline(always)]
    fn get_oracle_value_u32(&self, placeholder: Placeholder) -> u32 {
        self.oracle
            .get_u32_witness_from_placeholder(placeholder, self.absolute_row_idx)
    }

    #[inline(always)]
    fn get_oracle_value_u16(&self, placeholder: Placeholder) -> u16 {
        self.oracle
            .get_u16_witness_from_placeholder(placeholder, self.absolute_row_idx)
    }

    #[inline(always)]
    fn get_oracle_value_u8(&self, placeholder: Placeholder) -> u8 {
        self.oracle
            .get_u8_witness_from_placeholder(placeholder, self.absolute_row_idx)
    }

    #[inline(always)]
    fn get_oracle_value_boolean(&self, placeholder: Placeholder) -> bool {
        self.oracle
            .get_boolean_witness_from_placeholder(placeholder, self.absolute_row_idx)
    }

    #[inline(always)]
    fn set_memory_place(&mut self, idx: usize, value: F) {
        debug_assert!(idx < self.memory_row.len());
        unsafe {
            *self.memory_row.get_unchecked_mut(idx) = value;
        }
    }

    #[inline(always)]
    fn set_witness_place(&mut self, idx: usize, value: F) {
        debug_assert!(idx < self.witness_row.len());
        unsafe {
            *self.witness_row.get_unchecked_mut(idx) = value;
        }
    }

    #[inline(always)]
    fn set_scratch_place(&mut self, idx: usize, value: F) {
        debug_assert!(idx < self.scratch_space.len());
        unsafe {
            *self.scratch_space.get_unchecked_mut(idx) = value;
        }
    }

    #[inline(always)]
    fn lookup<const M: usize, const N: usize>(
        &mut self,
        inputs: &[F; M],
        table_id: u16,
        lookup_mapping_idx: usize,
    ) -> [F; N] {
        let (absolute_index, values) = self
            .table_driver
            .lookup_values_and_get_absolute_index::<N>(inputs, table_id as u32);
        self.multiplicity_counting_scratch[absolute_index] += 1;
        // assert_eq!(self.lookup_mapping_row[lookup_mapping_idx], 0);
        self.lookup_mapping_row[lookup_mapping_idx] = absolute_index as u32;

        values
    }

    #[inline(always)]
    fn maybe_lookup<const M: usize, const N: usize>(
        &mut self,
        inputs: &[F; M],
        table_id: u16,
        condition: bool,
    ) -> [F; N] {
        if condition {
            let (_absolute_index, values) = self
                .table_driver
                .lookup_values_and_get_absolute_index::<N>(inputs, table_id as u32);
            // we do not need to count multiplicity here
            values
        } else {
            [F::ZERO; N]
        }
    }

    #[inline(always)]
    fn lookup_enforce<const M: usize>(
        &mut self,
        inputs: &[F; M],
        table_id: u16,
        lookup_mapping_idx: usize,
    ) {
        let absolute_index = self
            .table_driver
            .enforce_values_and_get_absolute_index::<M>(inputs, table_id as u32);

        self.multiplicity_counting_scratch[absolute_index] += 1;
        // assert_eq!(self.lookup_mapping_row[lookup_mapping_idx], 0);
        self.lookup_mapping_row[lookup_mapping_idx] = absolute_index as u32;
    }
}
