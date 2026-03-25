#[repr(C, u32)]
#[derive(Clone, Copy)]
#[allow(dead_code)]
pub enum Placeholder {
    XregsInit,
    XregsFin,
    XregInit(u32),
    XregFin(u32),
    Instruction,
    MemSlot,
    PcInit,
    PcFin,
    StatusInit,
    StatusFin,
    IeInit,
    IeFin,
    IpInit,
    IpFin,
    TvecInit,
    TvecFin,
    ScratchInit,
    ScratchFin,
    EpcInit,
    EpcFin,
    CauseInit,
    CauseFin,
    TvalInit,
    TvalFin,
    ModeInit,
    ModeFin,
    MemorySaptInit,
    MemorySaptFin,
    ContinueExecutionInit,
    ContinueExecutionFin,
    ExternalOracle,
    Trapped,
    InvalidEncoding,
    FirstRegMem,
    SecondRegMem,
    MemoryLoadOp,
    WriteRdReadSetWitness,
    ShuffleRamLazyInitAddressThis,
    ShuffleRamLazyInitAddressNext,
    ShuffleRamAddress(u32),
    ShuffleRamReadTimestamp(u32),
    ShuffleRamReadValue(u32),
    ShuffleRamIsRegisterAccess(u32),
    ShuffleRamWriteValue(u32),
    ExecuteDelegation,
    DelegationType,
    DelegationABIOffset,
    DelegationWriteTimestamp,
    DelegationMemoryReadValue(u32),
    DelegationMemoryReadTimestamp(u32),
    DelegationMemoryWriteValue(u32),
    DelegationRegisterReadValue(u32),
    DelegationRegisterReadTimestamp(u32),
    DelegationRegisterWriteValue(u32),
    DelegationIndirectReadValue {
        register_index: u32,
        word_index: u32,
    },
    DelegationIndirectReadTimestamp {
        register_index: u32,
        word_index: u32,
    },
    DelegationIndirectWriteValue {
        register_index: u32,
        word_index: u32,
    },
    DelegationNondeterminismAccess(u32),
    DelegationNondeterminismAccessNoSplits(u32),
}

impl Default for Placeholder {
    fn default() -> Self {
        Placeholder::XregsInit
    }
}

impl From<cs::cs::placeholder::Placeholder> for Placeholder {
    fn from(value: cs::cs::placeholder::Placeholder) -> Self {
        match value {
            cs::cs::placeholder::Placeholder::XregsInit => Placeholder::XregsInit,
            cs::cs::placeholder::Placeholder::XregsFin => Placeholder::XregsFin,
            cs::cs::placeholder::Placeholder::XregInit(x) => Placeholder::XregInit(x as u32),
            cs::cs::placeholder::Placeholder::XregFin(x) => Placeholder::XregFin(x as u32),
            cs::cs::placeholder::Placeholder::Instruction => Placeholder::Instruction,
            cs::cs::placeholder::Placeholder::MemSlot => Placeholder::MemSlot,
            cs::cs::placeholder::Placeholder::PcInit => Placeholder::PcInit,
            cs::cs::placeholder::Placeholder::PcFin => Placeholder::PcFin,
            cs::cs::placeholder::Placeholder::StatusInit => Placeholder::StatusInit,
            cs::cs::placeholder::Placeholder::StatusFin => Placeholder::StatusFin,
            cs::cs::placeholder::Placeholder::IeInit => Placeholder::IeInit,
            cs::cs::placeholder::Placeholder::IeFin => Placeholder::IeFin,
            cs::cs::placeholder::Placeholder::IpInit => Placeholder::IpInit,
            cs::cs::placeholder::Placeholder::IpFin => Placeholder::IpFin,
            cs::cs::placeholder::Placeholder::TvecInit => Placeholder::TvecInit,
            cs::cs::placeholder::Placeholder::TvecFin => Placeholder::TvecFin,
            cs::cs::placeholder::Placeholder::ScratchInit => Placeholder::ScratchInit,
            cs::cs::placeholder::Placeholder::ScratchFin => Placeholder::ScratchFin,
            cs::cs::placeholder::Placeholder::EpcInit => Placeholder::EpcInit,
            cs::cs::placeholder::Placeholder::EpcFin => Placeholder::EpcFin,
            cs::cs::placeholder::Placeholder::CauseInit => Placeholder::CauseInit,
            cs::cs::placeholder::Placeholder::CauseFin => Placeholder::CauseFin,
            cs::cs::placeholder::Placeholder::TvalInit => Placeholder::TvalInit,
            cs::cs::placeholder::Placeholder::TvalFin => Placeholder::TvalFin,
            cs::cs::placeholder::Placeholder::ModeInit => Placeholder::ModeInit,
            cs::cs::placeholder::Placeholder::ModeFin => Placeholder::ModeFin,
            cs::cs::placeholder::Placeholder::MemorySaptInit => Placeholder::MemorySaptInit,
            cs::cs::placeholder::Placeholder::MemorySaptFin => Placeholder::MemorySaptFin,
            cs::cs::placeholder::Placeholder::ContinueExecutionInit => {
                Placeholder::ContinueExecutionInit
            }
            cs::cs::placeholder::Placeholder::ContinueExecutionFin => {
                Placeholder::ContinueExecutionFin
            }
            cs::cs::placeholder::Placeholder::ExternalOracle => Placeholder::ExternalOracle,
            cs::cs::placeholder::Placeholder::Trapped => Placeholder::Trapped,
            cs::cs::placeholder::Placeholder::InvalidEncoding => Placeholder::InvalidEncoding,
            cs::cs::placeholder::Placeholder::FirstRegMem => Placeholder::FirstRegMem,
            cs::cs::placeholder::Placeholder::SecondRegMem => Placeholder::SecondRegMem,
            cs::cs::placeholder::Placeholder::MemoryLoadOp => Placeholder::MemoryLoadOp,
            cs::cs::placeholder::Placeholder::WriteRdReadSetWitness => {
                Placeholder::WriteRdReadSetWitness
            }
            cs::cs::placeholder::Placeholder::ShuffleRamLazyInitAddressThis => {
                Placeholder::ShuffleRamLazyInitAddressThis
            }
            cs::cs::placeholder::Placeholder::ShuffleRamLazyInitAddressNext => {
                Placeholder::ShuffleRamLazyInitAddressNext
            }
            cs::cs::placeholder::Placeholder::ShuffleRamAddress(x) => {
                Placeholder::ShuffleRamAddress(x as u32)
            }
            cs::cs::placeholder::Placeholder::ShuffleRamReadTimestamp(x) => {
                Placeholder::ShuffleRamReadTimestamp(x as u32)
            }
            cs::cs::placeholder::Placeholder::ShuffleRamReadValue(x) => {
                Placeholder::ShuffleRamReadValue(x as u32)
            }
            cs::cs::placeholder::Placeholder::ShuffleRamIsRegisterAccess(x) => {
                Placeholder::ShuffleRamIsRegisterAccess(x as u32)
            }
            cs::cs::placeholder::Placeholder::ShuffleRamWriteValue(x) => {
                Placeholder::ShuffleRamWriteValue(x as u32)
            }
            cs::cs::placeholder::Placeholder::ExecuteDelegation => Placeholder::ExecuteDelegation,
            cs::cs::placeholder::Placeholder::DelegationType => Placeholder::DelegationType,
            cs::cs::placeholder::Placeholder::DegelationABIOffset => {
                Placeholder::DelegationABIOffset
            }
            cs::cs::placeholder::Placeholder::DelegationWriteTimestamp => {
                Placeholder::DelegationWriteTimestamp
            }
            cs::cs::placeholder::Placeholder::DelegationMemoryReadValue(x) => {
                Placeholder::DelegationMemoryReadValue(x as u32)
            }
            cs::cs::placeholder::Placeholder::DelegationMemoryReadTimestamp(x) => {
                Placeholder::DelegationMemoryReadTimestamp(x as u32)
            }
            cs::cs::placeholder::Placeholder::DelegationMemoryWriteValue(x) => {
                Placeholder::DelegationMemoryWriteValue(x as u32)
            }
            cs::cs::placeholder::Placeholder::DelegationRegisterReadValue(x) => {
                Placeholder::DelegationRegisterReadValue(x as u32)
            }
            cs::cs::placeholder::Placeholder::DelegationRegisterReadTimestamp(x) => {
                Placeholder::DelegationRegisterReadTimestamp(x as u32)
            }
            cs::cs::placeholder::Placeholder::DelegationRegisterWriteValue(x) => {
                Placeholder::DelegationRegisterWriteValue(x as u32)
            }
            cs::cs::placeholder::Placeholder::DelegationIndirectReadValue {
                register_index,
                word_index,
            } => Placeholder::DelegationIndirectReadValue {
                register_index: register_index as u32,
                word_index: word_index as u32,
            },
            cs::cs::placeholder::Placeholder::DelegationIndirectReadTimestamp {
                register_index,
                word_index,
            } => Placeholder::DelegationIndirectReadTimestamp {
                register_index: register_index as u32,
                word_index: word_index as u32,
            },
            cs::cs::placeholder::Placeholder::DelegationIndirectWriteValue {
                register_index,
                word_index,
            } => Placeholder::DelegationIndirectWriteValue {
                register_index: register_index as u32,
                word_index: word_index as u32,
            },
            cs::cs::placeholder::Placeholder::DelegationNondeterminismAccess(x) => {
                Placeholder::DelegationNondeterminismAccess(x as u32)
            }
            cs::cs::placeholder::Placeholder::DelegationNondeterminismAccessNoSplits(x) => {
                Placeholder::DelegationNondeterminismAccessNoSplits(x as u32)
            }
        }
    }
}
