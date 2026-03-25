//! GPU profiling infrastructure for Metal prover.
//!
//! When the `log_gpu_stages_timings` feature is enabled, records per-kernel
//! dispatch timing via Metal command buffer GPU timestamps. Prints a summary
//! table at the end of proving and outputs a Chrome trace JSON for visualization.

use std::sync::Mutex;
use std::time::Instant;

/// CPU span timing — records start/end of named CPU operations.
#[derive(Clone, Debug)]
pub struct CpuSpan {
    pub label: String,
    pub start_us: u64,
    pub end_us: u64,
}

/// A single profiling event (one commit_and_wait boundary).
#[derive(Clone, Debug)]
pub struct GpuEvent {
    pub label: String,
    pub wall_start_us: u64,
    pub wall_end_us: u64,
    pub gpu_start_us: u64,
    pub gpu_end_us: u64,
    pub dispatches: u32,
}

/// Global GPU profiler. Collects events across the entire proving run.
pub struct GpuProfiler {
    events: Vec<GpuEvent>,
    cpu_spans: Vec<CpuSpan>,
    epoch: Instant,
    /// Stack of active labels (set by push_label/pop_label)
    label_stack: Vec<String>,
    /// Current pending dispatches since last commit
    pending_dispatches: Vec<String>,
    /// Output directory for trace JSON (set via set_trace_dir)
    trace_dir: Option<String>,
}

static PROFILER: Mutex<Option<GpuProfiler>> = Mutex::new(None);
static TRACE_DIR: Mutex<Option<String>> = Mutex::new(None);

impl GpuProfiler {
    fn new() -> Self {
        Self {
            events: Vec::new(),
            cpu_spans: Vec::new(),
            epoch: Instant::now(),
            label_stack: Vec::new(),
            pending_dispatches: Vec::new(),
            trace_dir: None,
        }
    }

    fn current_label(&self) -> String {
        if !self.pending_dispatches.is_empty() {
            // Use the most recent kernel name
            self.pending_dispatches.last().unwrap().clone()
        } else if !self.label_stack.is_empty() {
            self.label_stack.last().unwrap().clone()
        } else {
            "unknown".to_string()
        }
    }
}

/// Initialize the global profiler. Call at start of proving.
pub fn init() {
    let mut profiler = GpuProfiler::new();
    // Pick up trace dir if it was set before init
    if let Some(dir) = TRACE_DIR.lock().unwrap().take() {
        profiler.trace_dir = Some(dir);
    }
    *PROFILER.lock().unwrap() = Some(profiler);
}

/// Mark a CPU span boundary. Closes the previous span (if any) and starts a new one.
/// Call at the start of each stage. Call `cpu_mark_end()` after the last stage.
pub fn cpu_mark(label: &str) {
    if let Some(ref mut p) = *PROFILER.lock().unwrap() {
        let now = p.epoch.elapsed().as_micros() as u64;
        // Close previous open span
        if let Some(last) = p.cpu_spans.last_mut() {
            if last.end_us == 0 {
                last.end_us = now;
            }
        }
        // Start new span
        p.cpu_spans.push(CpuSpan {
            label: label.to_string(),
            start_us: now,
            end_us: 0, // open until next mark
        });
    }
}

/// Close the last open CPU span.
pub fn cpu_mark_end() {
    if let Some(ref mut p) = *PROFILER.lock().unwrap() {
        let now = p.epoch.elapsed().as_micros() as u64;
        if let Some(last) = p.cpu_spans.last_mut() {
            if last.end_us == 0 {
                last.end_us = now;
            }
        }
    }
}

/// Convenience macro for CPU span markers (feature-gated).
#[macro_export]
macro_rules! cpu_span {
    ($label:expr) => {
        #[cfg(feature = "log_gpu_stages_timings")]
        $crate::metal_runtime::profiler::cpu_mark($label);
    };
}

/// Set the output directory for the trace JSON file.
/// Can be called before init() — the dir is stored and picked up on init.
pub fn set_trace_dir(dir: &str) {
    *TRACE_DIR.lock().unwrap() = Some(dir.to_string());
    // Also set on existing profiler if already initialized
    if let Some(ref mut p) = *PROFILER.lock().unwrap() {
        p.trace_dir = Some(dir.to_string());
    }
}

/// Push a label onto the profiler stack (e.g., "stage_1_commit").
pub fn push_label(label: &str) {
    if let Some(ref mut p) = *PROFILER.lock().unwrap() {
        p.label_stack.push(label.to_string());
    }
}

/// Pop a label from the profiler stack.
pub fn pop_label() {
    if let Some(ref mut p) = *PROFILER.lock().unwrap() {
        p.label_stack.pop();
    }
}

/// Record a kernel dispatch name (called from dispatch_kernel).
pub fn record_dispatch(kernel_name: &str) {
    if let Some(ref mut p) = *PROFILER.lock().unwrap() {
        p.pending_dispatches.push(kernel_name.to_string());
    }
}

/// Record a commit_and_wait event with GPU timestamps.
/// `gpu_start` and `gpu_end` are from MTLCommandBuffer.GPUStartTime/GPUEndTime.
pub fn record_commit(gpu_start: f64, gpu_end: f64) {
    if let Some(ref mut p) = *PROFILER.lock().unwrap() {
        let wall_now = p.epoch.elapsed().as_micros() as u64;
        let gpu_duration_us = ((gpu_end - gpu_start) * 1_000_000.0) as u64;
        let label = if p.pending_dispatches.len() <= 3 {
            p.pending_dispatches.join("+")
        } else {
            let first = &p.pending_dispatches[0];
            let last = p.pending_dispatches.last().unwrap();
            format!("{}..{} ({})", first, last, p.pending_dispatches.len())
        };
        let context = if !p.label_stack.is_empty() {
            format!("[{}] {}", p.label_stack.last().unwrap(), label)
        } else {
            label
        };
        let dispatches = p.pending_dispatches.len() as u32;
        p.events.push(GpuEvent {
            label: context,
            wall_start_us: wall_now.saturating_sub(gpu_duration_us),
            wall_end_us: wall_now,
            gpu_start_us: (gpu_start * 1_000_000.0) as u64,
            gpu_end_us: (gpu_end * 1_000_000.0) as u64,
            dispatches,
        });
        p.pending_dispatches.clear();
    }
}

/// Print the profiling summary and write Chrome trace JSON.
pub fn finish_and_report() {
    let mut p = PROFILER.lock().unwrap();
    let profiler = match p.take() {
        Some(p) => p,
        None => return,
    };

    if profiler.events.is_empty() {
        return;
    }

    // Print summary table
    eprintln!("\n{:=<90}", "");
    eprintln!("  GPU PROFILING SUMMARY");
    eprintln!("{:=<90}", "");
    eprintln!(
        "{:<55} {:>8} {:>8} {:>5}",
        "Operation", "GPU(ms)", "Wall(ms)", "Disp"
    );
    eprintln!("{:-<90}", "");

    let mut total_gpu_us: u64 = 0;
    let mut total_wall_us: u64 = 0;
    for event in &profiler.events {
        let gpu_ms = (event.gpu_end_us - event.gpu_start_us) as f64 / 1000.0;
        let wall_ms = (event.wall_end_us - event.wall_start_us) as f64 / 1000.0;
        total_gpu_us += event.gpu_end_us - event.gpu_start_us;
        total_wall_us += event.wall_end_us - event.wall_start_us;
        eprintln!(
            "{:<55} {:>7.1} {:>7.1} {:>5}",
            truncate_label(&event.label, 55),
            gpu_ms,
            wall_ms,
            event.dispatches
        );
    }
    eprintln!("{:-<90}", "");
    eprintln!(
        "{:<55} {:>7.1} {:>7.1} {:>5}",
        "TOTAL",
        total_gpu_us as f64 / 1000.0,
        total_wall_us as f64 / 1000.0,
        profiler.events.iter().map(|e| e.dispatches).sum::<u32>()
    );
    eprintln!("{:=<90}\n", "");

    // CPU spans summary
    if !profiler.cpu_spans.is_empty() {
        eprintln!("{:=<90}", "");
        eprintln!("  CPU SPANS");
        eprintln!("{:=<90}", "");
        eprintln!("{:<55} {:>8}", "Operation", "CPU(ms)");
        eprintln!("{:-<90}", "");
        let mut total_cpu_ms = 0.0f64;
        for span in &profiler.cpu_spans {
            let ms = (span.end_us - span.start_us) as f64 / 1000.0;
            if ms >= 0.1 {
                eprintln!("{:<55} {:>7.1}", truncate_label(&span.label, 55), ms);
            }
            total_cpu_ms += ms;
        }
        eprintln!("{:-<90}", "");
        eprintln!("{:<55} {:>7.1}", "TOTAL CPU SPANS", total_cpu_ms);
        eprintln!("{:=<90}\n", "");
    }

    // Write Chrome trace JSON (chrome://tracing format)
    if let Some(ref dir) = profiler.trace_dir {
        let trace_path = format!("{}/gpu_trace.json", dir);
        if let Ok(mut file) = std::fs::File::create(&trace_path) {
            use std::io::Write;
            let _ = write!(file, "[");
            let mut first = true;
            // GPU events on tid=1
            for event in &profiler.events {
                if !first { let _ = write!(file, ","); }
                first = false;
                let dur = event.gpu_end_us - event.gpu_start_us;
                let _ = write!(
                    file,
                    r#"{{"name":"{}","cat":"gpu","ph":"X","ts":{},"dur":{},"pid":1,"tid":1}}"#,
                    event.label.replace('"', "'"),
                    event.wall_start_us,
                    dur
                );
            }
            // CPU spans on tid=2
            for span in &profiler.cpu_spans {
                if !first { let _ = write!(file, ","); }
                first = false;
                let dur = span.end_us - span.start_us;
                let _ = write!(
                    file,
                    r#"{{"name":"{}","cat":"cpu","ph":"X","ts":{},"dur":{},"pid":1,"tid":2}}"#,
                    span.label.replace('"', "'"),
                    span.start_us,
                    dur
                );
            }
            let _ = write!(file, "]");
            eprintln!("Trace written to: {}", trace_path);
            eprintln!("Open in: chrome://tracing or https://ui.perfetto.dev\n");
        }
    }
}

fn truncate_label(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        format!("{}...", &s[..max - 3])
    }
}
