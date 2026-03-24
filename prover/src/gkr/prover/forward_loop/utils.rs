use super::*;
use crate::gkr::prover::apply_row_wise;
use cs::definitions::gkr::NoFieldVectorLookupRelation;
use field::{Field, FieldExtension, PrimeField};
use std::alloc::Global;

pub(crate) fn materialize_vector_lookup_input<F: PrimeField, E: FieldExtension<F> + Field>(
    rel: &NoFieldVectorLookupRelation,
    witness_trace: &mut GKRFullWitnessTrace<F, Global, Global>,
    trace_len: usize,
    preprocessed_generic_lookup: &[E],
    worker: &Worker,
) -> Box<[E]> {
    // we materialize it, but the good thing is that we have a cache of lookups
    let lookup_set_index = rel.lookup_set_index;
    let mut destination = Box::<[E], Global>::new_uninit_slice(trace_len);
    let ext_destination = vec![&mut destination[..]];
    let mapping_ref = if lookup_set_index != DECODER_LOOKUP_FORMAL_SET_INDEX {
        // println!("Mapping lookup access number {}", lookup_set_index);
        &witness_trace.generic_lookup_mapping[lookup_set_index]
    } else {
        // println!("Mapping decoder lookup");
        assert!(witness_trace.generic_lookup_mapping.len() > 0);
        witness_trace.generic_lookup_mapping.last().unwrap()
    };
    apply_row_wise::<F, _>(
        vec![],
        ext_destination,
        trace_len,
        worker,
        |_, ext_dest, chunk_start, chunk_size| {
            assert_eq!(ext_dest.len(), 1);
            let mut ext_dest = ext_dest;
            let dest = ext_dest.pop().unwrap();
            for i in 0..chunk_size {
                let mapping_index = mapping_ref[chunk_start + i];
                let mapped_value = preprocessed_generic_lookup[mapping_index as usize];
                dest[i].write(mapped_value);
            }
        },
    );
    let destination = unsafe { destination.assume_init() };

    destination
}
