use std::sync::atomic::{AtomicUsize, Ordering};

pub(crate) fn atomic_saturating_sub(atomic: &AtomicUsize, sub: usize) -> (usize, usize) {
    let mut old = atomic.load(Ordering::Relaxed);

    loop {
        let new = old.saturating_sub(sub);

        match atomic.compare_exchange_weak(old, new, Ordering::SeqCst, Ordering::Relaxed) {
            Ok(_) => return (old, new),
            Err(actual) => old = actual,
        }
    }
}
