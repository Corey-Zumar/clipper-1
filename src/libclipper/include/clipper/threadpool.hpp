#ifndef CLIPPER_LIB_THREADPOOL_HPP
#define CLIPPER_LIB_THREADPOOL_HPP

#include <concurrentqueue.h>
#include <condition_variable>
#include <mutex>
#include <queue>
#include <thread>
#include <unordered_map>

#include <boost/thread.hpp>

#include "containers.hpp"
#include "datatypes.hpp"
#include "logging.hpp"

namespace clipper {

const std::string LOGGING_TAG_THREADPOOL = "THREADPOOL";

/// Implementation adapted from
/// https://goo.gl/Iav87R

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
  ThreadPool(void) : done_{false}, queues_{}, threads_{} {}

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

  bool create_queue(ContainerId container_id) {
    boost::unique_lock<boost::shared_mutex> l(queues_mutex_);
    auto queue = queues_.find(static_cast<size_t>(container_id));
    if (queue != queues_.end()) {
      log_error_formatted(LOGGING_TAG_THREADPOOL,
                          "Work queue already exists for container with id {}",
                          std::to_string(container_id));
      return false;
    } else {
      queues_.emplace(std::piecewise_construct, std::forward_as_tuple(container_id),
                      std::forward_as_tuple());
      threads_.emplace(std::piecewise_construct, std::forward_as_tuple(container_id),
                       std::forward_as_tuple(&ThreadPool::worker, this, container_id));
      log_info_formatted(LOGGING_TAG_THREADPOOL, "Work queue created for container with id {}",
                         std::to_string(container_id));
      return true;
    }
  }

  /**
   * Submit a job to be run by the thread pool.
   */
  template <typename Func, typename... Args>
  auto submit(ContainerId container_id, Func&& func, Args&&... args) {
    auto boundTask = std::bind(std::forward<Func>(func), std::forward<Args>(args)...);
    using ResultType = std::result_of_t<decltype(boundTask)()>;
    using PackagedTask = boost::packaged_task<ResultType()>;
    using TaskType = ThreadTask<PackagedTask>;
    PackagedTask task{std::move(boundTask)};
    auto result_future = task.get_future();

    boost::shared_lock<boost::shared_mutex> l(queues_mutex_);
    auto queue = queues_.find(static_cast<size_t>(container_id));
    if (queue != queues_.end()) {
      queue->second.enqueue(std::make_unique<TaskType>(std::move(task)));
    } else {
      std::stringstream error_msg;
      error_msg << "No work queue for container with id: " << std::to_string(container_id);
      log_error(LOGGING_TAG_THREADPOOL, error_msg.str());
      throw std::runtime_error(error_msg.str());
    }
    return result_future;
  }

 private:
  /**
   * Constantly running function each thread uses to acquire work items from the
   * queue.
   */
  void worker(size_t worker_id) {
    while (!done_) {
      std::unique_ptr<IThreadTask> pTask{nullptr};
      bool work_to_do = false;
      {
        boost::shared_lock<boost::shared_mutex> l(queues_mutex_);
        // NOTE: The use of try_pop here means the worker will spin instead of
        // block while waiting for work. This is intentional. We defer to the
        // submitted tasks to block when no work is available.
        work_to_do = queues_[worker_id].try_dequeue(pTask);
      }
      if (work_to_do) {
        pTask->execute();
      }
    }
    auto thread_id = std::this_thread::get_id();
    std::stringstream ss;
    ss << thread_id;
    log_info_formatted(LOGGING_TAG_THREADPOOL, "Worker {}, thread {} is shutting down",
                       std::to_string(worker_id), ss.str());
  }

  /**
   * Invalidates the queue and joins all running threads.
   */
  void destroy(void) {
    log_info(LOGGING_TAG_THREADPOOL, "Destroying threadpool");
    done_ = true;
    // for (auto& queue : queues_) {
    // queue.second.invalidate();
    // }
    for (auto& thread : threads_) {
      if (thread.second.joinable()) {
        thread.second.join();
      }
    }
  }

 private:
  std::atomic_bool done_;
  boost::shared_mutex queues_mutex_;
  std::unordered_map<size_t, moodycamel::ConcurrentQueue<std::unique_ptr<IThreadTask>>> queues_;
  std::unordered_map<size_t, std::thread> threads_;
};

namespace TaskExecutionThreadPool {

/**
 * Convenience method to get the task execution thread pool for the application.
 */
inline ThreadPool& get_thread_pool(void) {
  static ThreadPool taskExecutionPool;
  return taskExecutionPool;
}

/**
 * Submit a job to the task execution thread pool.
 */
template <typename Func, typename... Args>
inline auto submit_job(ContainerId container_id, Func&& func, Args&&... args) {
  return get_thread_pool().submit(container_id, std::forward<Func>(func),
                                  std::forward<Args>(args)...);
}

inline void create_queue(ContainerId container_id) {
  get_thread_pool().create_queue(container_id);
}

}  // namespace DefaultThreadPool
}  // namespace clipper

#endif  // CLIPPER_LIB_THREADPOOL_HPP
