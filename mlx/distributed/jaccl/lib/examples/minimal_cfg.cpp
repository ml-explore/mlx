#include <jaccl/jaccl.h>
#include <iostream>

int main() {
  auto cfg = jaccl::Config()
                 .set_rank(0)
                 .set_coordinator("192.168.1.1:32132")
                 .set_devices(
                     {{{}, {"rdma_en5"}, {"rdma_en4"}, {"rdma_en3"}},
                      {{"rdma_en5"}, {}, {"rdma_en3"}, {"rdma_en4"}},
                      {{"rdma_en4"}, {"rdma_en3"}, {}, {"rdma_en5"}},
                      {{"rdma_en3"}, {"rdma_en4"}, {"rdma_en5"}, {}}});
  auto group = jaccl::init(cfg);
  if (!group) {
    std::cerr << "Failed to initialize JACCL" << std::endl;
    return 1;
  }

  std::cout << "Rank " << group->rank() << " of " << group->size() << std::endl;

  // Perform all-reduce sum
  float input[10] = {
      1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f};
  float output[10];

  group->all_sum(input, output, sizeof(input), jaccl::Float32);

  std::cout << "Result: " << output[0] << std::endl;

  return 0;
}
