#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

// Standalone repro for the Metal framework assertion fixed by this change.
//
// Compile:
//   clang++ -fobjc-arc -framework Foundation -framework Metal \
//     tests/metal_completed_handler_repro.mm \
//     -o /tmp/metal_completed_handler_repro
//
// Run:
//   /tmp/metal_completed_handler_repro
//   /tmp/metal_completed_handler_repro fixed
//
// On affected macOS versions, the first command aborts when addCompletedHandler
// is called while the compute encoder is still active. The second command ends
// encoding first and exits successfully.

int main(int argc, const char** argv) {
  @autoreleasepool {
    BOOL end_before_handler = argc > 1 && strcmp(argv[1], "fixed") == 0;

    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (device == nil) {
      fprintf(stderr, "No Metal device available\n");
      return 2;
    }

    id<MTLCommandQueue> queue = [device newCommandQueue];
    id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder =
        [command_buffer computeCommandEncoder];

    NSString* source =
        @"#include <metal_stdlib>\n"
         "using namespace metal;\n"
         "kernel void add_one(device float* values [[buffer(0)]],"
         " uint id [[thread_position_in_grid]]) {\n"
         "  values[id] += 1.0f;\n"
         "}\n";
    NSError* error = nil;
    id<MTLLibrary> library = [device newLibraryWithSource:source
                                                  options:nil
                                                    error:&error];
    if (library == nil) {
      fprintf(stderr, "%s\n", error.localizedDescription.UTF8String);
      return 3;
    }

    id<MTLFunction> function = [library newFunctionWithName:@"add_one"];
    id<MTLComputePipelineState> pipeline =
        [device newComputePipelineStateWithFunction:function error:&error];
    if (pipeline == nil) {
      fprintf(stderr, "%s\n", error.localizedDescription.UTF8String);
      return 4;
    }

    float value = 1.0f;
    id<MTLBuffer> buffer =
        [device newBufferWithBytes:&value
                            length:sizeof(value)
                           options:MTLResourceStorageModeShared];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:buffer offset:0 atIndex:0];
    [encoder dispatchThreads:MTLSizeMake(1, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];

    if (end_before_handler) {
      [encoder endEncoding];
    }

    [command_buffer addCompletedHandler:^(id<MTLCommandBuffer> buffer) {
      (void)buffer;
    }];

    if (!end_before_handler) {
      [encoder endEncoding];
    }

    [command_buffer commit];
    [command_buffer waitUntilCompleted];

    if (*static_cast<float*>(buffer.contents) != 2.0f) {
      fprintf(stderr, "Unexpected result\n");
      return 5;
    }
  }

  return 0;
}
