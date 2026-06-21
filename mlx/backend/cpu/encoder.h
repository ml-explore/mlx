// Copyright © 2025 Apple Inc.

#pragma once

#include <exception>
#include <unordered_map>

#include "mlx/array.h"
#include "mlx/event.h"
#include "mlx/scheduler.h"

namespace mlx::core::cpu {

// Number of dispatches per scheduler task
constexpr int DISPATCHES_PER_TASK = 10;

struct MLX_API CommandEncoder {
  CommandEncoder(Stream stream) : stream_(stream) {}

  CommandEncoder(const CommandEncoder&) = delete;
  CommandEncoder& operator=(const CommandEncoder&) = delete;
  CommandEncoder(CommandEncoder&&) = delete;
  CommandEncoder& operator=(CommandEncoder&&) = delete;

  void set_input_array(const array& a) {}
  void set_output_array(array& a) {}

  // Hold onto a temporary until any already scheduled tasks which use it as
  // an input are complete.
  void add_temporary(array arr) {
    temporaries_.push_back(std::move(arr));
  }

  void add_temporaries(std::vector<array> arrays) {
    temporaries_.insert(
        temporaries_.end(),
        std::make_move_iterator(arrays.begin()),
        std::make_move_iterator(arrays.end()));
  }

  void set_error_event(Event event) {
    event_ = std::move(event);
  }

  std::vector<array>& temporaries() {
    return temporaries_;
  }

  template <class F, class... Args>
  void dispatch(F&& f, Args&&... args) {
    dispatch_impl(true, std::forward<F>(f), std::forward<Args>(args)...);
  }

  template <class F, class... Args>
  void dispatch_unchecked(F&& f, Args&&... args) {
    dispatch_impl(false, std::forward<F>(f), std::forward<Args>(args)...);
  }

 private:
  template <class F, class... Args>
  void dispatch_impl(bool skip_on_error, F&& f, Args&&... args) {
    num_ops_ = (num_ops_ + 1) % DISPATCHES_PER_TASK;
    auto task = std::bind(std::forward<F>(f), std::forward<Args>(args)...);
    auto event = event_;
    if (num_ops_ == 0) {
      scheduler::notify_new_task(stream_);
      auto task_wrap = [s = stream_,
                        event = std::move(event),
                        skip_on_error,
                        task = std::move(task)]() mutable {
        struct CompletionNotifier {
          Stream stream;
          ~CompletionNotifier() {
            scheduler::notify_task_completion(stream);
          }
        } completion{s};
        if (skip_on_error && event.valid() && event.error()) {
          return;
        }
        try {
          task();
        } catch (...) {
          if (event.valid()) {
            event.set_error(std::current_exception());
          } else {
            throw;
          }
        }
      };
      scheduler::enqueue(stream_, std::move(task_wrap));
    } else {
      scheduler::enqueue(
          stream_,
          [event = std::move(event),
           skip_on_error,
           task = std::move(task)]() mutable {
            if (skip_on_error && event.valid() && event.error()) {
              return;
            }
            try {
              task();
            } catch (...) {
              if (event.valid()) {
                event.set_error(std::current_exception());
              } else {
                throw;
              }
            }
          });
    }
  }

  Stream stream_;
  Event event_;
  std::vector<array> temporaries_;
  int num_ops_{0};
};

MLX_API CommandEncoder& get_command_encoder(Stream s);

std::unordered_map<int, CommandEncoder>& get_command_encoders();
std::unordered_map<int, CommandEncoder>& get_global_command_encoders();

} // namespace mlx::core::cpu
