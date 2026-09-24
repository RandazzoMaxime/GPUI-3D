use std::{
    cell::Cell,
    marker::PhantomData,
    sync::atomic::{AtomicUsize, Ordering},
};
use thread_local::ThreadLocal;

pub(crate) struct Dropper<T: Send> {
    counter: AtomicUsize,
    garbage: ThreadLocal<GarbageVec<T>>,
}

impl<T: Send> Dropper<T> {
    pub(crate) fn pause(&self) -> Pause<'_, T> {
        let mut count = self.counter.load(Ordering::Relaxed);

        loop {
            if count == usize::MAX {
                panic!("Too many pauses!");
            }

            match self.counter.compare_exchange(
                count,
                count + 1,
                Ordering::AcqRel,
                Ordering::Relaxed,
            ) {
                Ok(_) => break Pause {
                    gc: self,
                    // TODO: Do we need had_list?
                    _no_send: PhantomData
                },
                Err(new) => count = new,
            }
        }
    }
}

pub(crate) struct Pause<'gc, T: Send> {
    gc: &'gc Dropper<T>,
    _no_send: PhantomData<*mut ()>,
}

impl<'gc, T: Send> Pause<'gc, T> {
    pub(crate) fn gc(&self) -> &Dropper<T> {
        self.gc
    }

    pub(crate) fn queue_drop(&self, value: T) {
        match self.gc.counter.load(Ordering::Acquire) {
            // Only pause active, and it's this thread.
            1 => {
                self.gc.garbage.get_or_default().clear();
                drop(value);
            }
            // Not safe to drop right now.
            _ => self.gc.garbage.get_or_default().push(value),
        }
    }
}

impl<'gc, T: Send> Drop for Pause<'gc, T> {
    fn drop(&mut self) {
        if self.gc.counter.fetch_sub(1, Ordering::AcqRel) == 1 {
            self.gc.garbage.get().map(GarbageVec::clear);
        }
    }
}

impl<'gc, T: Send> Clone for Pause<'gc, T> {
    fn clone(&self) -> Self {
        self.gc.pause()
    }
}

struct GarbageVec<T>(Cell<Vec<T>>);

impl<T> GarbageVec<T> {
    fn push(&self, value: T) {
        let mut vec = self.0.replace(vec![]);
        vec.push(value);
        self.0.replace(vec);
    }

    fn clear(&self) {
        self.0.replace(vec![]);
    }
}

impl<T> Default for GarbageVec<T> {
    fn default() -> Self {
        Self(Cell::new(vec![]))
    }
}