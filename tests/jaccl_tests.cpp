// Copyright © 2026 Apple Inc.

#include "doctest/doctest.h"

#include <stdexcept>
#include <string>

#include "jaccl/rdma.h"

TEST_CASE("test JACCL shared buffer rejects null protection domain") {
  jaccl::SharedBuffer buffer(64);

  try {
    buffer.register_to_protection_domain(nullptr);
    FAIL("Expected null protection domain registration to throw");
  } catch (const std::runtime_error& e) {
    std::string message(e.what());
    CHECK(message.find("protection domain") != std::string::npos);
    CHECK(message.find("ibv_devices") != std::string::npos);
  }
}
