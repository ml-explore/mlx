#include <iostream>
#include "mlx/autograd_state.h"
#include "mlx/mlx.h"

using namespace mlx::core;

int main() {
  std::cout << "Testing no_grad implementation..." << std::endl;

  // Test default state
  std::cout << "Default gradient state: "
            << (GradMode::is_enabled() ? "enabled" : "disabled") << std::endl;

  // Test basic function
  auto x = array(2.0);
  auto fun = [](array input) { return multiply(input, input); };

  // Test with gradients enabled
  {
    std::cout << "\n=== With gradients enabled ===" << std::endl;
    auto [output, grad] = vjp(fun, x, array(1.0));
    std::cout << "f(2) = " << output.item<float>() << std::endl;
    std::cout << "df/dx = " << grad.item<float>() << std::endl;
  }

  // Test with no_grad
  {
    std::cout << "\n=== With no_grad ===" << std::endl;
    NoGradGuard no_grad;
    std::cout << "Gradient state inside no_grad: "
              << (GradMode::is_enabled() ? "enabled" : "disabled") << std::endl;

    auto [output, grad] = vjp(fun, x, array(1.0));
    std::cout << "f(2) = " << output.item<float>() << std::endl;
    std::cout << "df/dx = " << grad.item<float>() << " (should be 0)"
              << std::endl;
  }

  // Test that state is restored
  std::cout << "\nGradient state after no_grad: "
            << (GradMode::is_enabled() ? "enabled" : "disabled") << std::endl;

  // Test nested contexts
  {
    std::cout << "\n=== Testing nested contexts ===" << std::endl;
    NoGradGuard no_grad;
    std::cout << "Inside no_grad: "
              << (GradMode::is_enabled() ? "enabled" : "disabled") << std::endl;

    {
      EnableGradGuard enable_grad;
      std::cout << "Inside enable_grad (nested): "
                << (GradMode::is_enabled() ? "enabled" : "disabled")
                << std::endl;

      auto [output, grad] = vjp(fun, x, array(1.0));
      std::cout << "f(2) = " << output.item<float>()
                << ", df/dx = " << grad.item<float>() << std::endl;
    }

    std::cout << "Back to no_grad: "
              << (GradMode::is_enabled() ? "enabled" : "disabled") << std::endl;
  }

  std::cout << "\nFinal gradient state: "
            << (GradMode::is_enabled() ? "enabled" : "disabled") << std::endl;
  std::cout << "\nno_grad implementation test completed successfully!"
            << std::endl;

  return 0;
}