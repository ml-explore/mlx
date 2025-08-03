// Copyright © 2023-2024 Apple Inc.

#pragma once

namespace mlx::core {

/**
 * Structure used to manage thread-local autograd state flags
 * similar to PyTorch's autograd state management.
 */
struct AutogradState {
  static AutogradState& get_tls_state();
  static void set_tls_state(AutogradState state);

  AutogradState(bool grad_mode = true) : grad_mode_(grad_mode) {}

  void set_grad_mode(bool enabled) {
    grad_mode_ = enabled;
  }
  bool get_grad_mode() const {
    return grad_mode_;
  }

 private:
  bool grad_mode_;
};

/**
 * Global gradient mode control functions
 */
struct GradMode {
  static bool is_enabled();
  static void set_enabled(bool enabled);
};

/**
 * A RAII, thread local guard that enables or disables grad mode upon
 * construction, and sets it back to the original value upon destruction.
 */
struct AutoGradMode {
  AutoGradMode(bool enabled) : prev_mode(GradMode::is_enabled()) {
    GradMode::set_enabled(enabled);
  }
  AutoGradMode(const AutoGradMode&) = delete;
  AutoGradMode(AutoGradMode&&) = delete;
  AutoGradMode& operator=(const AutoGradMode&) = delete;
  AutoGradMode& operator=(AutoGradMode&&) = delete;
  ~AutoGradMode() {
    GradMode::set_enabled(prev_mode);
  }
  bool prev_mode;
};

/**
 * A RAII, thread local guard that stops future operations from building
 * gradients.
 */
struct NoGradGuard : public AutoGradMode {
  NoGradGuard() : AutoGradMode(/*enabled=*/false) {}
};

/**
 * A RAII, thread local guard that enables gradient computation.
 */
struct EnableGradGuard : public AutoGradMode {
  EnableGradGuard() : AutoGradMode(/*enabled=*/true) {}
};

} // namespace mlx::core