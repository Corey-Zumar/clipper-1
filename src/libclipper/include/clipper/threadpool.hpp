#ifndef CLIPPER_LIB_THREADPOOL_HPP
#define CLIPPER_LIB_THREADPOOL_HPP

#include <mutex>
#include <queue>
#include <thread>

#include <boost/thread.hpp>

#include "config.hpp"

namespace clipper {

template <typename T>
class ThreadSafeQueue {
 public:
  /**
   * Destructor.
   */
  ~ThreadSafeQueue(void) { invalidate(); }

  /**
   * Attempt to get the first value in the queue.
   * Returns true if a value was successfully written to the out parameter,
   * false otherwise.
   */
  bool try_pop(T& out) {
    std::lock_guard<std::mutex> lock{mutex_};
    if (queue_.empty() || !valid_) {
      return false;
    }
    out = std::move(queue_.front());
    queue_.pop();
    return true;
  }

  /**
   * Get the first value in the queue.
   * Will block until a value is available unless clear is called or the
   * instance is destructed.
   * Returns true if a value was successfully written to the out parameter,
   * false otherwise.
   */
  bool wait_pop(T& out) {
    std::unique_lock<std::mutex> lock{mutex_};
    condition_.wait(lock, [this]() { return !queue_.empty() || !valid_; });
    /*
     * Using the condition in the predicate ensures that spurious wakeups with a
     * valid
     * but empty queue will not proceed, so only need to check for validity
     * before proceeding.
     */
    if (!valid_) {
      return false;
    }
    out = std::move(queue_.front());
    queue_.pop();
    return true;
  }

  /**
   * Push a new value onto the queue.
   */
  void push(T value) {
    std::lock_guard<std::mutex> lock{mutex_};
    queue_.push(std::move(value));
    condition_.notify_one();
  }

  /**
   * Check whether or not the queue is empty.
   */
  bool empty(void) const {
    std::lock_guard<std::mutex> lock{mutex_};
    return queue_.empty();
  }

  /**
   * Clear all items from the queue.
   */
  void clear(void) {
    std::lock_guard<std::mutex> lock{mutex_};
    while (!queue_.empty()) {
      queue_.pop();
    }
    condition_.notify_all();
  }

  /**
   * Invalidate the queue.
   * Used to ensure no conditions are being waited on in wait_pop when
   * a thread or the application is trying to exit.
   * The queue is invalid after calling this method and it is an error
   * to continue using a queue after this method has been called.
   */
  void invalidate(void) {
    std::lock_guard<std::mutex> lock{mutex_};
    valid_ = false;
    condition_.notify_all();
  }

  /**
   * Returns whether or not this queue is valid.
   */
  bool is_valid(void) const {
    std::lock_guard<std::mutex> lock{mutex_};
    return valid_;
  }

 private:
  std::atomic_bool valid_{true};
  mutable std::mutex mutex_;
  std::queue<T> queue_;
  std::condition_variable condition_;
};

class ThreadPool {
 private:
  class IThreadTask {
   public:
    IThreadTask(void) = default;
    virtual ~IThreadTask(void) = default;
    IThreadTask(const IThreadTask& rhs) = delete;
    IThreadTask& operator=(const IThreadTask& rhs) = delete;
    IThreadTask(IThreadTask&& other) = default;
    IThreadTask& operator=(IThreadTask&& other) = default;

    /**
     * Run the task.
     */
    virtual void execute() = 0;
  };

  template <typename Func>
  class ThreadTask : public IThreadTask {
   public:
    ThreadTask(Func&& func) : m_func{std::move(func)} {}

    ~ThreadTask(void) override = default;
    ThreadTask(const ThreadTask& rhs) = delete;
    ThreadTask& operator=(const ThreadTask& rhs) = delete;
    ThreadTask(ThreadTask&& other) = default;
    ThreadTask& operator=(ThreadTask&& other) = default;

    /**
     * Run the task.
     */
    void execute() override { m_func(); }

   private:
    Func m_func;
  };

 public:
  /**
   * Constructor.
   */
  ThreadPool(void)
      : ThreadPool{std::max(std::thread::hardware_concurrency(), 2u) - 1u} {
    /*
     * Always create at least one thread.  If hardware_concurrency() returns 0,
     * subtracting one would turn it to UINT_MAX, so get the maximum of
     * hardware_concurrency() and 2 before subtracting 1.
     */
  }

  /**
   * Constructor.
   */
  explicit ThreadPool(const std::uint32_t numThreads)
      : m_done{false}, m_workQueue{}, m_threads{} {
    try {
      for (std::uint32_t i = 0u; i < numThreads; ++i) {
        m_threads.emplace_back(&ThreadPool::worker, this);
      }
    } catch (...) {
      destroy();
      throw;
    }
  }

  /**
   * Non-copyable.
   */
  ThreadPool(const ThreadPool& rhs) = delete;

  /**
   * Non-assignable.
   */
  ThreadPool& operator=(const ThreadPool& rhs) = delete;

  /**
   * Destructor.
   */
  ~ThreadPool(void) { destroy(); }

  /**
   * Submit a job to be run by the thread pool.
   */
  template <typename Func, typename... Args>
  auto submit(Func&& func, Args&&... args) {
    auto boundTask =
        std::bind(std::forward<Func>(func), std::forward<Args>(args)...);
    using ResultType = std::result_of_t<decltype(boundTask)()>;
    using PackagedTask = boost::packaged_task<ResultType()>;
    using TaskType = ThreadTask<PackagedTask>;
    PackagedTask task{std::move(boundTask)};
    auto result_future = task.get_future();

    m_workQueue.push(std::make_unique<TaskType>(std::move(task)));
    return result_future;
  }

 private:
  /**
   * Constantly running function each thread uses to acquire work items from the
   * queue.
   */
  void worker(void) {
    while (!m_done) {
      std::unique_ptr<IThreadTask> pTask{nullptr};
      if (m_workQueue.wait_pop(pTask)) {
        pTask->execute();
      }
    }
  }

  /**
   * Invalidates the queue and joins all running threads.
   */
  void destroy(void) {
    m_done = true;
    m_workQueue.invalidate();
    for (auto& thread : m_threads) {
      if (thread.joinable()) {
        thread.join();
      }
    }
  }

 private:
  std::atomic_bool m_done;
  ThreadSafeQueue<std::unique_ptr<IThreadTask>> m_workQueue;
  std::vector<std::thread> m_threads;
};

namespace DefaultThreadPool {

/**
 * Get the default thread pool for the application.
 * This pool is created with 4 threads.
 */
inline ThreadPool& get_thread_pool(void) {
  static ThreadPool defaultPool(get_config().get_default_threadpool_size());
  return defaultPool;
}

/**
 * Submit a job to the default thread pool.
 */
template <typename Func, typename... Args>
inline auto submit_job(Func&& func, Args&&... args) {
  return get_thread_pool().submit(std::forward<Func>(func),
                                  std::forward<Args>(args)...);
}
}  // namespace DefaultThreadPool
}

#endif  // CLIPPER_LIB_THREADPOOL_HPP
