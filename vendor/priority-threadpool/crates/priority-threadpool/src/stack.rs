use std::{
    ptr::NonNull,
    sync::atomic::{AtomicPtr, Ordering},
};

use owned_alloc::OwnedAlloc;

use crate::dropper::Dropper;

struct Node<T> {
    value: T,
    next: *mut Node<T>,
}

pub(crate) struct Stack<T: Send> {
    top: AtomicPtr<Node<T>>,
    gc: Dropper<OwnedAlloc<Node<T>>>,
}

impl<T: Send> Stack<T> {
    pub(crate) fn push(&self, value: T) {
        let mut node = OwnedAlloc::new(Node {
            value,
            next: self.top.load(Ordering::Acquire),
        });

        'cas: loop {
            let new_top = node.raw().as_ptr();

            match self.top.compare_exchange(
                node.next,
                new_top,
                Ordering::Release,
                Ordering::Relaxed,
            ) {
                Ok(_) => {
                    // We're going to manually drop this, later.
                    std::mem::forget(node);
                    break 'cas;
                }
                // CAS failed, someone could've updated the top before us.
                // Let's update our node to point to it.
                Err(x) => node.next = x,
            }
        }
    }

    pub(crate) fn pop(&self) -> Option<T> {
        let pause = self.gc.pause();
        let mut top = self.top.load(Ordering::Acquire);

        loop {
            let mut ptr = NonNull::new(top)?;

            match self.top.compare_exchange(
                top,
                unsafe { ptr.as_ref().next },
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => {
                    let value = unsafe { (&mut ptr.as_mut().value as *mut T).read() };

                    pause.queue_drop(unsafe { OwnedAlloc::from_raw(ptr) });

                    break Some(value);
                }
                _ => todo!(),
            }
        }
    }
}
