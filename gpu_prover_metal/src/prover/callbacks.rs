//! Callback scheduling for Metal.
//! In CUDA, host functions are enqueued on streams and run asynchronously
//! when the stream reaches that point. On Metal with synchronous execution,
//! callbacks run immediately when scheduled since all prior GPU work is
//! already complete (auto-commit per dispatch).

pub(crate) struct Callbacks<'a>(Vec<Box<dyn FnOnce() + Send + 'a>>);

impl<'a> Callbacks<'a> {
    pub fn new() -> Self {
        Self(vec![])
    }

    /// Schedule a callback. Runs immediately since Metal execution is synchronous.
    pub fn schedule<F: FnOnce() + Send + 'a>(
        &mut self,
        func: F,
    ) {
        // Run immediately — all GPU work is already committed and complete
        func();
    }

    pub fn extend(&mut self, other: Self) {
        self.0.extend(other.0);
    }
}
