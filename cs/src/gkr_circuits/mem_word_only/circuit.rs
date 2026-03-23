use super::*;
use crate::constraint::Constraint;
use crate::constraint::Term;
use crate::cs::circuit::LookupQueryTableType;
use crate::cs::circuit_trait::*;
use crate::gkr_circuits::utils::update_intermediate_carry_value;
use crate::oracle::Placeholder;
use crate::tables::TableDriver;
use crate::types::*;
use crate::witness_placer::*;
use field::PrimeField;

const TABLES_TOTAL_WIDTH: usize = 3;

pub fn mem_word_only_tables() -> Vec<TableType> {
    vec![
        // TableType::RegIsZero,
        // TableType::U16GetSign,
        // TableType::ConditionalJmpBranchSlt,
        // TableType::JumpCleanupOffset,
    ]
}

pub fn mem_word_only_table_addition_fn<F: PrimeField, CS: Circuit<F>>(cs: &mut CS) {
    for el in mem_word_only_tables() {
        cs.materialize_table::<TABLES_TOTAL_WIDTH>(el);
    }
}

pub fn mem_word_only_table_driver_fn<F: PrimeField>(table_driver: &mut TableDriver<F>) {
    for el in mem_word_only_tables() {
        table_driver.materialize_table::<TABLES_TOTAL_WIDTH>(el);
    }
}

fn apply_mem_word_only_inner<F: PrimeField, CS: Circuit<F>>(
    cs: &mut CS,
    inputs: OpcodeFamilyCircuitState<F>,
    decoder: WordOnlyMemoryFamilyCircuitMask,
) {
    // LW: rd                          <- mem[addr] || rom[addr]  with +0 offset accepted
    // SW: mem[addr] || trap rom[addr] <- rs2                     with +0 offset accepted
    // NOTE: by preprocessing (decoder lookup) we have rd == 0 for loads not possible
    // so we do NOT need to mask rd value

    if let Some(circuit_family_extra_mask) =
        cs.get_value(inputs.decoder_data.circuit_family_extra_mask)
    {
        println!(
            "circuit_family_extra_mask = 0b{:08b}",
            circuit_family_extra_mask.as_u32_reduced()
        );
    }

    // read rs1, to compute address
    let MemoryAccess::RegisterOnly(RegisterAccess { read_value: WordRepresentation::U16Limbs(rs1), .. }) = cs.request_mem_access(
        MemoryAccessRequest::RegisterRead {
            reg_idx: inputs.decoder_data.rs1_index,
            read_value_placeholder: Placeholder::ShuffleRamReadValue(0),
            split_as_u8: false,
        },
        "rs1",
        0,
    ) else {unreachable!()};

    

    // strategies:
    // - we perform an initial setup: computing the addr, and fetching rom data.
    //   we get rom data from a lookup (that also manages traps),
    // - then we manage 3 orthogonal edge cases: load*!rom, load*rom, store*!rom (and store*rom is trapped)
    // - bump pc

    let isstore = decoder.perform_write();
    let isload = decoder.perform_write().toggle();
    let addr = {
        let [rs1_low, rs1_high] = rs1;
        let [imm_low, imm_high] = inputs.decoder_data.imm;
        let low: Variable = cs.add_named_variable("addr_low"); // range checked by addr cleanup table
        let high = cs.add_named_variable("addr_high"); // range checked by rom addr sep. table
        // cs.require_invariant(low, Invariant::RangeChecked { width: 16 });
        // cs.require_invariant(high, Invariant::RangeChecked { width: 16 });
        let of_low = cs.add_named_boolean_variable("low overflow: rs1 +u16 imm");
        let of_high = cs.add_named_boolean_variable("high overflow: rs1 +u16 imm");
        let shift16 = Term::from(1<<16);
        {
            // TODO: witgen
        }
        cs.add_constraint_allow_explicit_linear(Term::from(rs1_low) + Term::from(imm_low) - Term::from(low) - shift16*Term::from(of_low));
        cs.add_constraint_allow_explicit_linear(Term::from(of_low) + Term::from(rs1_high) + Term::from(imm_high) - Term::from(high) - shift16*Term::from(of_high));
        [low, high]
    };
    let [isrom, romaddr_hi] = {
        let isrom = cs.add_named_variable("flag: are we in rom addr range?");
        let romaddr_hi = cs.add_named_variable("address high 16bits truncated/wrapped to rom range (eg 6 bits)");
        let [_, addr_hi] = addr;
        let inputs = &[addr_hi].map(LookupInput::from);
        let output_variables = &[isrom, romaddr_hi];
        let table_type = LookupQueryTableType::Constant(TableType::RomAddressSpaceSeparator);
        cs.set_variables_from_lookup_constrained(inputs, output_variables, table_type);
        // trap store*rom
        cs.add_constraint(Constraint::from(isrom) * Constraint::from(isstore));
        [isrom, romaddr_hi]
    };
    let rom = {
        let lo = cs.add_named_variable("romread low 16bits");
        let hi = cs.add_named_variable("romread high 16bits");
        // trap +1/2/3 word offsets
        let [addr_lo, _] = addr;
        let inputs = &[addr_lo, romaddr_hi].map(LookupInput::from);
        let output_variables = &[lo, hi];
        let table_type = LookupQueryTableType::Constant(TableType::RomReadWordonlyRangechecked);
        cs.set_variables_from_lookup_constrained(inputs, output_variables, table_type);
        [lo, hi]
    };

    // now we are ready to read mem/rs2

    // this is gonna be its own variables anyway
    // so let's just do register choose (using NEW cs.choose ..)
    let readonly_address = {
        let lo = cs.add_named_variable_from_constraint(constraint, "read-only mem/rs2 address");
        [lo, hi]
    };
    let maccess = cs.request_mem_access(MemoryAccessRequest::RegisterOrRamRead { 
        is_register: isstore, 
        address: cs.choose, 
        read_value_placeholder: (), 
        split_as_u8: () }, "mem/rs2 read", 1);
    let _isreg = isstore;
    let _addr = {
        let [cleanaddr_lo, cleanaddr_hi] = cleanaddr;
        let rs2_idx = Constraint::from(inputs.decoder_data.rs2_index);
        let lo = isload * cleanaddr_lo + isstore * rs2_idx;
        let hi = isload * cleanaddr_hi;
        [lo, hi]
    };
    let _read_value_placeholder = Placeholder::ShuffleRamReadValue(1);
    let _split_as_u8 = false;
    let _name = "mem/rs2 read";
    let _local_timestamp_in_cycle = 1;
    let requesttbd = todo!();
    let MemoryAccess::RegisterOrRam(RegisterOrRamAccess { read_value: WordRepresentation::U16Limbs(mem), .. }) = requesttbd else {unreachable!()};

    // now we may overwrite rd/mem
    let _isreg: Constraint<F> = isload;
    let _addr = {
        let [cleanaddr_lo, cleanaddr_hi] = cleanaddr;
        let rd_idx = Constraint::from(inputs.decoder_data.rd_index);
        let lo = isstore * cleanaddr_lo + isload * rd_idx;
        let hi = isstore * cleanaddr_hi;
        [lo, hi]
    };
    let _read_value_placeholder = Placeholder::ShuffleRamReadValue(2);
    let _write_value_placeholder = Placeholder::ShuffleRamWriteValue(2);
    let _split_read_as_u8 = false;
    let _split_write_as_u8 = false;
    let _name = "mem/rd overwrite";
    let _local_timestamp_in_cycle = 2;
    let requesttbd = todo!();
    let MemoryAccess::RegisterOrRam(RegisterOrRamAccess { read_value: WordRepresentation::U16Limbs(old), write_value: WordRepresentation::U16Limbs(write), .. }) = requesttbd else {unreachable!()};

    // now we may proceed with our "write" calculations
    // due to STORE opcode limitations, we will be creating 2 scratch values
    let isloadrom = Constraint::from(isrom);
    let isloadram = isload - Constraint::from(isrom);
    let isstoreram = Constraint::from(isstore);
    let out = {
        let lo = cs.add_named_variable("first orthogonal lookup output variable (low or chunk)");
        let hi = cs.add_named_variable("second orthogonal lookup output variable (high or pad)");
        let limb = {
            let [mem_lo, mem_hi] = mem;
            let [rom_lo, rom_hi] = rom;
            let [old_lo, old_hi] = old;
            let sel_lo = Term::from(cs.add_intermediate_named_variable_from_constraint(isloadram * Term::from(mem_lo) + isloadrom * Term::from(rom_lo) + isstoreram * Term::from(old_lo), "selection of correct read (low bits)"));
            let sel_hi = Term::from(cs.add_intermediate_named_variable_from_constraint(isloadram * Term::from(mem_hi) + isloadrom * Term::from(rom_hi) + isstoreram * Term::from(old_hi), "selection of correct read (high bits)"));
            cs.add_intermediate_named_variable_from_constraint(Term::from(b1) * sel_hi + Constraint::from(b1.toggle()) * sel_lo, "selected low/high limb to use")
        };
        // let input_lo = 
        let tableid_lo = todo!();
        let tableid_hi = todo!();
        //
        let inputs = &[limb_lo, offset, funct3_from_mask.clone()].map(LookupInput::from);
        let output_variables = &[lo];
        let table_type = LookupQueryTableType::Expression(LookupInput::from(tableid_lo));
        cs.set_variables_from_lookup_constrained(inputs, output_variables, table_type);
        let inputs = &[limb_hi, offset, funct3_from_mask].map(LookupInput::from);
        let output_variables = &[lo];
        let table_type = LookupQueryTableType::Expression(LookupInput::from(tableid_lo));
        cs.set_variables_from_lookup_constrained(inputs, output_variables, table_type);
        [lo, hi]
    };








    // bump PC
    use crate::gkr_circuits::utils::calculate_pc_next_no_overflows_with_range_checks;
    calculate_pc_next_no_overflows_with_range_checks(
        cs,
        inputs.cycle_start_state.pc,
        inputs.cycle_end_state.pc,
    );
}

pub fn mem_word_only_circuit_with_preprocessed_bytecode_for_gkr<F: PrimeField, CS: Circuit<F>>(
    cs: &mut CS,
) {
    let (input, bitmask) = cs.allocate_machine_state(false, false, WORD_ONLY_MEMORY_FAMILY_NUM_FLAGS);
    let bitmask: [_; WORD_ONLY_MEMORY_FAMILY_NUM_FLAGS] = bitmask.try_into().unwrap();
    let bitmask = bitmask.map(|el| Boolean::Is(el));
    let decoder = WordOnlyMemoryFamilyCircuitMask::from_mask(bitmask);
    apply_mem_word_only_inner(cs, input, decoder);
}

#[cfg(test)]
mod test {
    use test_utils::skip_if_ci;

    use super::*;
    use crate::gkr_compiler::compile_unrolled_circuit_state_transition_into_gkr;
    use crate::gkr_compiler::dump_ssa_witness_eval_form_for_unrolled_circuit;
    use crate::utils::serialize_to_file;

    #[test]
    fn compile_mem_word_only_circuit_into_gkr() {
        skip_if_ci!();
        use ::field::baby_bear::base::BabyBearField;

        let gkr_compiled = compile_unrolled_circuit_state_transition_into_gkr::<BabyBearField>(
            &|cs| mem_word_only_table_addition_fn(cs),
            &|cs| mem_word_only_circuit_with_preprocessed_bytecode_for_gkr(cs),
            1 << 20,
            24,
        );

        serialize_to_file(
            &gkr_compiled,
            "compiled_circuits/mem_word_only_preprocessed_layout_gkr.json",
        );
    }

    #[test]
    fn compile_mem_word_only_gkr_witness_graph() {
        skip_if_ci!();
        use ::field::baby_bear::base::BabyBearField;

        let ssa_forms = dump_ssa_witness_eval_form_for_unrolled_circuit::<BabyBearField>(
            &|cs| mem_word_only_table_addition_fn(cs),
            &|cs| mem_word_only_circuit_with_preprocessed_bytecode_for_gkr(cs),
        );
        serialize_to_file(
            &ssa_forms,
            "compiled_circuits/mem_word_only_preprocessed_ssa_gkr.json",
        );
    }
}
