use std::{os::windows::thread, sync::Arc, time::Duration};

use async_task::Runnable;

#[derive(Clone)]
enum MyPriority {
    High,
    Medium,
    Low,
}

impl priority_threadpool::Priority for MyPriority {
    const COUNT: usize = 3;

    fn index(&self) -> usize {
        match self {
            Self::High => 2,
            Self::Medium => 1,
            Self::Low => 0,
        }
    }
}

fn main() {
    let nworkers = 1024 * 5;

    let threadpool = Arc::new(priority_threadpool::ThreadPool::<MyPriority>::new(nworkers));
    threadpool.block_til_ready();

    let mut tasks = vec![];

    for ix in 0..(nworkers * 4) {
        let schedule = {
            let threadpool = threadpool.clone();
            move |x| threadpool.queue(&MyPriority::High, x)
        };

        let future = async move {
            println!("Hi on thread {:?}", std::thread::current().id());
            std::thread::sleep(Duration::from_secs(1));
        };

        let (runnable, task) = async_task::spawn(future, schedule);

        runnable.schedule();
        tasks.push(task);
    }

    threadpool.wake();

    for task in &tasks {
        while !task.is_finished() {
            std::hint::spin_loop();
        }
    } 

   // threadpool.join();
}
