use std::{
    marker::PhantomData,
    mem::MaybeUninit,
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
    thread::JoinHandle,
};

use async_task::Runnable;
use lockfree::stack::Stack;

//mod stack;
mod util;
//mod dropper;

/// Type-erased `Runnable<M>` stored inline (pointer-sized).
struct ErasedRunnable {
    storage: MaybeUninit<*mut ()>,
    run_fn: unsafe fn(*mut u8) -> bool,
    drop_fn: unsafe fn(*mut u8),
}

// Safety: Runnable<M> is Send + Sync for all M.
unsafe impl Send for ErasedRunnable {}
unsafe impl Sync for ErasedRunnable {}

impl ErasedRunnable {
    fn new<M: Send + Sync + 'static>(runnable: Runnable<M>) -> Self {
        assert!(
            std::mem::size_of::<Runnable<M>>() <= std::mem::size_of::<*mut ()>(),
            "Runnable<M> is larger than expected",
        );
        assert!(
            std::mem::align_of::<Runnable<M>>() <= std::mem::align_of::<*mut ()>(),
            "Runnable<M> has stricter alignment than expected",
        );

        let mut storage = MaybeUninit::<*mut ()>::uninit();
        unsafe {
            std::ptr::write(storage.as_mut_ptr().cast::<Runnable<M>>(), runnable);
        }

        unsafe fn run_erased<M>(ptr: *mut u8) -> bool {
            let runnable = unsafe { std::ptr::read(ptr.cast::<Runnable<M>>()) };
            runnable.run()
        }

        unsafe fn drop_erased<M>(ptr: *mut u8) {
            unsafe { std::ptr::drop_in_place(ptr.cast::<Runnable<M>>()) };
        }

        ErasedRunnable {
            storage,
            run_fn: run_erased::<M>,
            drop_fn: drop_erased::<M>,
        }
    }

    fn run(mut self) -> bool {
        let result = unsafe { (self.run_fn)(self.storage.as_mut_ptr().cast::<u8>()) };
        // run() consumed the inner Runnable, skip Drop.
        std::mem::forget(self);
        result
    }
}

impl Drop for ErasedRunnable {
    fn drop(&mut self) {
        unsafe { (self.drop_fn)(self.storage.as_mut_ptr().cast::<u8>()) };
    }
}

pub trait Priority {
    const COUNT: usize;

    fn index(&self) -> usize;
}

#[derive(Clone)]
struct PriorityQueue<P: Priority> {
    stacks: Arc<Vec<Stack<Job>>>,
    _phantom: PhantomData<P>,
}

struct Job {
    // Execute after duration.
    after: Option<std::time::Duration>,
    runnable: ErasedRunnable,
}

impl Job {
    fn run(self) -> bool {
        // TODO(mdeand): This could be much better instead of occupying an entire thread.

        if let Some(after) = self.after {
            std::thread::sleep(after);
        }

        self.runnable.run()
    }
}

impl<P: Priority> PriorityQueue<P> {
    fn pop(&self) -> Option<Job> {
        for ix in 0..P::COUNT {
            let stack = &self.stacks[ix];

            if let Some(value) = stack.pop() {
                return Some(value);
            }
        }

        None
    }

    fn push(&self, priority: &P, job: Job) {
        let index = priority.index();

        assert!(index < P::COUNT);
        assert!(index < self.stacks.len());

        self.stacks[priority.index()].push(job);
    }

    pub fn new() -> Self {
        Self {
            stacks: Arc::new((0..P::COUNT).into_iter().map(|_| Stack::new()).collect()),
            _phantom: PhantomData,
        }
    }
}

pub struct ThreadPool<P: Priority + Clone> {
    jobs_queued: Arc<AtomicUsize>,
    should_stop: Arc<AtomicBool>,
    waiting: Vec<Arc<AtomicBool>>,
    threads: Vec<JoinHandle<()>>,
    queue: PriorityQueue<P>,
}

impl<P> ThreadPool<P>
where
    P: Priority + Clone + Send + 'static,
{
    pub fn new(nworkers: usize) -> Self {
        let jobs_queued = Arc::new(AtomicUsize::new(0));

        let should_stop = Arc::new(AtomicBool::new(false));

        let waiting: Vec<_> = (0..nworkers)
            .into_iter()
            .map(|_| Arc::new(AtomicBool::new(false)))
            .collect();

        let queue = PriorityQueue::new();

        let threads: Vec<_> = (0..nworkers)
            .into_iter()
            .map(|ix| {
                let jobs_queued = jobs_queued.clone();
                let should_stop = should_stop.clone();
                let thread_waiting = waiting[ix].clone();
                let queue = queue.clone();

                std::thread::Builder::new()
                    .name(format!("ThreadPool worker {}", ix))
                    .spawn(move || {
                        thread_waiting.store(true, Ordering::Release);

                        while !should_stop.load(Ordering::Relaxed) {
                            // TODO(mdeand): This probably doesn't need to happen here
                            match util::atomic_saturating_sub(&jobs_queued, 1) {
                                (old, new) if old > 0 => {
                                    if let Some(runnable) = queue.pop() {
                                        thread_waiting.store(false, Ordering::Release);

                                        /*
                                        println!(
                                            "Thread {:?} running job...",
                                            std::thread::current().id()
                                        );
                                        */

                                        runnable.run();
                                    }
                                }
                                // If there's no jobs in queue, do nothing.
                                _ => std::thread::park(),
                            };

                            thread_waiting.store(true, Ordering::Release);
                        }
                    })
                    .unwrap()
            })
            .collect();

        Self {
            jobs_queued,
            should_stop,
            waiting,
            threads,
            queue,
        }
    }

    pub fn block_til_ready(&self) {
        // Wait for all threads to park
        for is_waiting in &self.waiting {
            while !is_waiting.load(Ordering::Acquire) {
                std::hint::spin_loop();
            }
        }
    }

    pub fn signal_stop(&self) {
        self.should_stop.store(false, Ordering::Release);
    }

    pub fn wake(&self) {
        for ix in 0..self.threads.len() {
            let handle = &self.threads[ix];
            let waiting = &self.waiting[ix];

            if waiting.load(Ordering::Acquire) {
                handle.thread().unpark();
            }
        }
    }

    pub fn queue<M: Send + Sync + 'static>(&self, priority: &P, runnable: Runnable<M>) {
        // Le job AVANT le crédit : un worker qui prend le crédit entre les deux
        // fait `pop()` sur une pile vide et le job reste orphelin à jamais (LIFO).
        // Timer de `ui_boot` perdu ⇒ ticks figés (2026-09-17).
        self.queue.push(
            priority,
            Job {
                after: None,
                runnable: ErasedRunnable::new(runnable),
            },
        );
        self.jobs_queued.fetch_add(1, Ordering::SeqCst);

        self.wake();
    }

    pub fn queue_delayed<M: Send + Sync + 'static>(
        &self,
        priority: &P,
        duration: std::time::Duration,
        runnable: Runnable<M>,
    ) {
        // Le job AVANT le crédit : un worker qui prend le crédit entre les deux
        // fait `pop()` sur une pile vide et le job reste orphelin à jamais (LIFO).
        // Timer de `ui_boot` perdu ⇒ ticks figés (2026-09-17).
        self.queue.push(
            priority,
            Job {
                after: Some(duration),
                runnable: ErasedRunnable::new(runnable),
            },
        );
        self.jobs_queued.fetch_add(1, Ordering::SeqCst);

        self.wake();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;

    #[derive(Clone)]
    struct P(usize);
    impl Priority for P {
        const COUNT: usize = 3;
        fn index(&self) -> usize {
            self.0
        }
    }

    /// Échoue 6/6 sur `bb1ab91` (1 à 2 jobs perdus sur 80 000), passe avec le
    /// crédit publié après le job.
    #[test]
    fn no_job_is_orphaned_under_contention() {
        let pool = Arc::new(ThreadPool::<P>::new(8));
        pool.block_til_ready();
        let ran = Arc::new(AtomicUsize::new(0));
        let (producers, per) = (4, 20_000);
        let hs: Vec<_> = (0..producers)
            .map(|_| {
                let (pool, ran) = (pool.clone(), ran.clone());
                std::thread::spawn(move || {
                    for i in 0..per {
                        let (ran, pool2) = (ran.clone(), pool.clone());
                        let (runnable, task) = async_task::Builder::new().metadata(()).spawn(
                            move |_| async move {
                                ran.fetch_add(1, Ordering::SeqCst);
                            },
                            move |r| {
                                if i % 50 == 0 {
                                    pool2.queue_delayed(&P(2), std::time::Duration::from_micros(200), r)
                                } else {
                                    pool2.queue(&P(i % 3), r)
                                }
                            },
                        );
                        runnable.schedule();
                        task.detach();
                    }
                })
            })
            .collect();
        for h in hs {
            h.join().unwrap();
        }
        let expected = producers * per;
        let t0 = std::time::Instant::now();
        while ran.load(Ordering::SeqCst) < expected && t0.elapsed() < std::time::Duration::from_secs(5) {
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        assert_eq!(ran.load(Ordering::SeqCst), expected, "jobs perdus");
    }
}
