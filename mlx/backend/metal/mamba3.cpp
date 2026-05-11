// mlx/backend/metal/mamba3.cpp
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/array.h"
#include "mlx/stream.h"
#include "mlx/utils.h"      // <-- RCA FIXED: For StreamOrDevice and to_stream
#include "mlx/primitives.h" // <-- RCA FIXED: For Primitive base class
#include "mlx/fast.h"        // <-- RCA FIXED: Connects the MLX_API export tag
#include "mlx/allocator.h" // <-- RCA FIXED: We need the MLX Memory Allocator

namespace mlx::core::fast {

// -----------------------------------------------------------------------------
// 1. The Low-Level Metal Dispatcher
// -----------------------------------------------------------------------------
void mamba3_ssd_gpu(
    const array& q, const array& k, const array& v,
    const array& dt, const array& trap, const array& angles,
    array& out, const Stream& stream) {

  auto& d = metal::device(stream.device);
  auto& compute_encoder = metal::get_command_encoder(stream);

  int batch = q.shape(0);
  int seqlen = q.shape(1);
  int nheads = q.shape(2);
  int headdim_qk = q.shape(3);
  int headdim_v = v.shape(3);
  
  int chunk_size = 64; 
  int num_chunks = (seqlen + chunk_size - 1) / chunk_size;

  std::string kernel_name = "mamba3_ssd_fused_" + type_to_name(q.dtype()); 
  auto kernel = d.get_kernel(kernel_name);
  
  compute_encoder.set_compute_pipeline_state(kernel);

  compute_encoder.set_input_array(q, 0);
  compute_encoder.set_input_array(k, 1);
  compute_encoder.set_input_array(v, 2);
  compute_encoder.set_input_array(dt, 3);
  compute_encoder.set_input_array(trap, 4);
  compute_encoder.set_input_array(angles, 5);
  
  compute_encoder.set_output_array(out, 6);

  struct Mamba3Params {
    int seqlen, headdim_qk, headdim_v, chunk_size, num_chunks;
  };
  Mamba3Params params{seqlen, headdim_qk, headdim_v, chunk_size, num_chunks};
  
  compute_encoder.set_bytes(params, 7);

  MTL::Size grid_dims = MTL::Size::Make(nheads, batch, 1);
  MTL::Size group_dims = MTL::Size::Make(256, 1, 1);

  compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
}

// -----------------------------------------------------------------------------
// 2. The Autograd Graph Primitive
// -----------------------------------------------------------------------------
class Mamba3SSD : public Primitive {
public:
  explicit Mamba3SSD(Stream stream) : Primitive(stream) {}

  void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) override {
      // RCA FIXED: The correct MLX API is allocator::malloc
      outputs[0].set_data(allocator::malloc(outputs[0].nbytes()));
      
      mamba3_ssd_gpu(inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5], outputs[0], stream());
  }

  void eval_cpu(const std::vector<array>& inputs, std::vector<array>& outputs) override {
      throw std::runtime_error("Mamba-3 SSD is only supported on Apple Silicon GPUs.");
  }

  const char* name() const override { return "Mamba3SSD"; }

  bool is_equivalent(const Primitive& other) const override {
      return typeid(*this) == typeid(other);
  }
};

// -----------------------------------------------------------------------------
// 3. The High-Level Frontend Function
// -----------------------------------------------------------------------------
array mamba3_ssd(
    const array& q, const array& k, const array& v,
    const array& dt, const array& trap, const array& angles,
    StreamOrDevice s) {
    
    return array(
        v.shape(), 
        v.dtype(), 
        std::make_shared<Mamba3SSD>(to_stream(s)), 
        {q, k, v, dt, trap, angles}
    );
}

} // namespace mlx::core::fast