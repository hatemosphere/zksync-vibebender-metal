#pragma once

#include "common.metal"

namespace airbender {
namespace witness {
namespace placeholder {

using namespace ::airbender::witness;

enum PlaceholderTag : u32 {
  XregsInit,
  XregsFin,
  XregInit,
  XregFin,
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
  ShuffleRamAddress,
  ShuffleRamReadTimestamp,
  ShuffleRamReadValue,
  ShuffleRamIsRegisterAccess,
  ShuffleRamWriteValue,
  ExecuteDelegation,
  DelegationType,
  DelegationABIOffset,
  DelegationWriteTimestamp,
  DelegationMemoryReadValue,
  DelegationMemoryReadTimestamp,
  DelegationMemoryWriteValue,
  DelegationRegisterReadValue,
  DelegationRegisterReadTimestamp,
  DelegationRegisterWriteValue,
  DelegationIndirectReadValue,
  DelegationIndirectReadTimestamp,
  DelegationIndirectWriteValue,
  DelegationNondeterminismAccess,
  DelegationNondeterminismAccessNoSplits,
};

struct DelegationPayload {
  u32 register_index;
  u32 word_index;
};

union PlaceholderPayload {
  u32 u32_val;
  DelegationPayload delegation_payload;
};

struct Placeholder {
  PlaceholderTag tag;
  PlaceholderPayload payload;
};

} // namespace placeholder
} // namespace witness
} // namespace airbender
