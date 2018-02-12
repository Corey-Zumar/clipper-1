#ifndef CLIPPER_CLOCK_HPP
#define CLIPPER_CLOCK_HPP

#include <chrono>

namespace clipper {

namespace clock {

class ClipperClock {
 public:
  static ClipperClock &get_clock() {
    static ClipperClock instance;
    return instance;
  }

  /**
   * Obtains the time, in microseconds, since
   * this clock was created
   */
  long long get_uptime() {
    return std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::system_clock::now() - start_time_).count();
  }

 private:
  ClipperClock()
      : start_time_(std::chrono::system_clock::now()) {}

  std::chrono::time_point<std::chrono::system_clock> start_time_;
};

} // namespace clock

} // namespace clipper

#endif //CLIPPER_CLOCK_HPP
