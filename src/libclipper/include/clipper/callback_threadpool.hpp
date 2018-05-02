#ifndef CLIPPER_LIB_CALLBACK_THREADPOOL_HPP
#define CLIPPER_LIB_CALLBACK_THREADPOOL_HPP

#include <mutex>
#include <queue>
#include <thread>
#include <unordered_map>

#include <boost/thread.hpp>
#include <blockingconcurrentqueue.h>

#include "datatypes.hpp"
#include "logging.hpp"
#include "threadpool.hpp"
#include "metrics.hpp"

namespace clipper {

// const std::string LOGGING_TAG_THREADPOOL = "CALLBACKTHREADPOOL";

/// Implementation adapted from
/// https://goo.gl/Iav87R

class CallbackThreadPool {
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
    ThreadTask(Func&& func) : func_{std::move(func)} {}

    ~ThreadTask(void) override = default;
    ThreadTask(const ThreadTask& rhs) = delete;
    ThreadTask& operator=(const ThreadTask& rhs) = delete;
    ThreadTask(ThreadTask&& other) = default;
    ThreadTask& operator=(ThreadTask&& other) = default;

    /**
     * Run the task.
     */
    void execute() override { func_(); }

   private:
    Func func_;
  };

 public:
  CallbackThreadPool(const std::string name,
      const std::uint32_t numThreads) : done_{false}, queue_{100000}, threads_{},
    queue_submit_latency_hist_(metrics::MetricsRegistry::get_metrics().create_histogram(
        name + ":queue_submit_latency",
        "microseconds", 4096)) {
    try {
      for(std::uint32_t i = 0u; i < numThreads; ++i) {
        threads_.emplace_back(&CallbackThreadPool::worker, this);
      }
    }
    catch(...) {
      destroy();
      throw;
    }
  }

  /**
   * Non-copyable.
   */
  CallbackThreadPool(const CallbackThreadPool& rhs) = delete;

  /**
   * Non-assignable.
   */
  CallbackThreadPool& operator=(const CallbackThreadPool& rhs) = delete;

  /**
   * Destructor.
   */
  ~CallbackThreadPool(void) { destroy(); }

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
    std::chrono::time_point<std::chrono::system_clock> start_time =
        std::chrono::system_clock::now();
    queue_.enqueue(std::make_unique<TaskType>(std::move(task)));
    std::chrono::time_point<std::chrono::system_clock> current_time =
        std::chrono::system_clock::now();

    auto submit_latency = current_time - start_time;
    long submit_latency_micros =
        std::chrono::duration_cast<std::chrono::microseconds>(submit_latency)
            .count();
    queue_submit_latency_hist_->insert(static_cast<int64_t>(submit_latency_micros));
    return result_future;
  }

 private:
  /**
   * Constantly running function each thread uses to acquire work items from the
   * queue.
   */
  void worker() {
    while (!done_) {
      std::unique_ptr<IThreadTask> pTask{nullptr};
      // NOTE: This is a blocking call. In this threadpool, we want the workers to block
      // instead of spin if there is no work.
      queue_.wait_dequeue(pTask);
      pTask->execute();
    }
  }

  /**
   * Invalidates the queue and joins all running threads.
   */
  void destroy(void) {
    log_info(LOGGING_TAG_THREADPOOL, "Destroying threadpool");
    done_ = true;
    // queue_.invalidate();
    for (auto& thread : threads_) {
      if (thread.joinable()) {
        thread.join();
      }
    }
  }

 private:
  std::atomic_bool done_;
  // ThreadSafeQueue<std::unique_ptr<IThreadTask>> queue_;
  moodycamel::BlockingConcurrentQueue<std::unique_ptr<IThreadTask>> queue_;
  std::vector<std::thread> threads_;
  std::shared_ptr<metrics::Histogram> queue_submit_latency_hist_;
};

}


#endif  // CLIPPER_LIB_CALLBACK_THREADPOOL_HPP
