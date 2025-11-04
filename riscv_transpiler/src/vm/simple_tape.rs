use crate::{ir::Instruction, vm::InstructionTape};

pub struct SimpleTape {
    instructions: Box<[Instruction]>,
}

impl SimpleTape {
    pub fn new(instructions: &[Instruction]) -> Self {
        Self {
            instructions: instructions.to_vec().into_boxed_slice(),
        }
    }
}

impl InstructionTape for SimpleTape {
    #[inline(always)]
    fn prefetch_instruction(&self, pc: u32) {
        debug_assert_eq!(pc % 4, 0);
        let word = (pc >> 2) as usize;
        debug_assert!(word < self.instructions.len());
        unsafe {
            core::intrinsics::prefetch_read_data::<_, 2>(self.instructions.get_unchecked(word) as *const Instruction);
        }
    }

    #[inline(always)]
    fn read_instruction(&self, pc: u32) -> Instruction {
        unsafe {
            debug_assert_eq!(pc % 4, 0);
            let word = (pc >> 2) as usize;
            debug_assert!(word < self.instructions.len());
            *self.instructions.get_unchecked(word)
        }
    }
}
