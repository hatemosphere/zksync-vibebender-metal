// Tiny ObjC shim for os_signpost — the C API functions are inline/macro
// and can't be called directly from Rust FFI.
#import <os/signpost.h>

static os_log_t _prover_log = NULL;

void prover_signpost_init(void) {
    if (!_prover_log) {
        _prover_log = os_log_create("com.matterlabs.gpu-prover", "proving");
    }
}

uint64_t prover_signpost_begin(const char *name) {
    if (!_prover_log) prover_signpost_init();
    os_signpost_id_t spid = os_signpost_id_generate(_prover_log);
    os_signpost_interval_begin(_prover_log, spid, "prover", "%s", name);
    return (uint64_t)spid;
}

void prover_signpost_end(uint64_t spid, const char *name) {
    if (!_prover_log) return;
    os_signpost_interval_end(_prover_log, (os_signpost_id_t)spid, "prover", "%s", name);
}
