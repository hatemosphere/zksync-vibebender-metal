use super::*;

pub fn create_u16_get_sign_table<F: PrimeField>(id: u32) -> LookupTable<F> {
    let keys = key_for_continuous_log2_range::<F, 1>(16);
    const TABLE_NAME: &'static str = "U16 get sign table";

    LookupTable::create_table_from_key_and_pure_generation_fn(
        &keys,
        TABLE_NAME.to_string(),
        1,
        1,
        |keys| {
            let a = keys[0].as_u32_reduced();
            assert!(a < (1u32 << 16), "input value is 0x{:08x}", a);

            let sign = a >> 15;

            let mut result = ArrayVec::new();
            result.push(F::from_u32_unchecked(sign));

            (a as usize, result)
        },
        Some(first_key_index_gen_fn::<F>),
        id,
    )
}

pub fn create_reg_is_zero_table<F: PrimeField>(id: u32) -> LookupTable<F> {
    let keys = key_for_continuous_log2_range::<F, 1>(17);
    const TABLE_NAME: &'static str = "Reg is zero table";

    LookupTable::create_table_from_key_and_pure_generation_fn(
        &keys,
        TABLE_NAME.to_string(),
        1,
        1,
        |keys| {
            let a = keys[0].as_u32_reduced();
            assert!(a < (1u32 << 17), "input value is 0x{:08x}", a);

            let is_zero = a == 0;

            let mut result = ArrayVec::new();
            result.push(F::from_u32_unchecked(is_zero as u32));

            (a as usize, result)
        },
        Some(first_key_index_gen_fn::<F>),
        id,
    )
}
