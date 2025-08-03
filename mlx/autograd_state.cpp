// Copyright © 2023-2024 Apple Inc.

#include "mlx/autograd_state.h"

namespace mlx::core {

AutogradState& AutogradState::get_tls_state() {
  thread_local static AutogradState tls_state{true}; // gradients enabled by default
  return tls_state;
}

void AutogradState::set_tls_state(AutogradState state) {
  get_tls_state() = state;
}

bool GradMode::is_enabled() {
  return AutogradState::get_tls_state().get_grad_mode();
}

void GradMode::set_enabled(bool enabled) {
  AutogradState::get_tls_state().set_grad_mode(enabled);
}

} // namespace mlx::core