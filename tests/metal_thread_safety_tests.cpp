#include "doctest/doctest.h"
#include "mlx/mlx.h"
#include "mlx/backend/metal/device.h"

#include <thread>
#include <vector>
#include <atomic>
#include <chrono>
#include <mutex>
#include <iostream> 

using namespace mlx::core;

// Helper function to run operations across multiple threads with pre-created streams
void run_in_threads(int num_threads, const std::function<void(int, Stream)>& func, 
                   const std::vector<Stream>& streams) {
    std::vector<std::thread> threads;
    threads.reserve(num_threads);
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back(func, i, streams[i % streams.size()]);
    }
    for (auto& t : threads) {
        if (t.joinable()) {
            t.join();
        }
    }
}

// Helper function for tasks not requiring streams (e.g., using default stream)
void run_in_threads_default(int num_threads, const std::function<void(int)>& func) {
    std::vector<std::thread> threads;
    threads.reserve(num_threads);
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back(func, i);
    }
    for (auto& t : threads) {
        if (t.joinable()) {
            t.join();
        }
    }
}

// Thread-safe result collection
struct TestResults {
    std::mutex mutex;
    std::vector<bool> shape_checks;
    std::vector<bool> availability_checks;
    std::vector<bool> value_checks;
    std::vector<float> expected_values;
    std::vector<float> actual_values;
    
    void record_result(bool shape_ok, bool available_ok, bool value_ok, 
                      float expected, float actual) {
        std::lock_guard<std::mutex> lock(mutex);
        shape_checks.push_back(shape_ok);
        availability_checks.push_back(available_ok);
        value_checks.push_back(value_ok);
        expected_values.push_back(expected);
        actual_values.push_back(actual);
    }
};

TEST_CASE("test metal concurrent eval operations") {
    Device D_GPU = Device::gpu;
    const int num_threads = std::thread::hardware_concurrency() > 0 ? std::thread::hardware_concurrency() : 8;
    const int ops_per_thread = 10;
    const int array_size = 32;
    std::atomic<int> completed_ops{0};
    TestResults results;
    
    // Pre-create streams to avoid concurrent stream creation
    std::vector<Stream> streams;
    for (int i = 0; i < num_threads; ++i) {
        streams.push_back(new_stream(D_GPU));
    }
    synchronize(); // Ensure stream creation is complete
    
    auto task = [&](int thread_id, Stream s) {
        try {
            for (int i = 0; i < ops_per_thread; ++i) {
                float val1 = static_cast<float>(thread_id * ops_per_thread + i + 1);
                float val2 = val1 * 2.0f;
                
                auto x = full({array_size, array_size}, val1, s);
                auto y = full({array_size, array_size}, val2, s);
                auto z = add(x, y);
                eval(z);
                
                bool shape_ok = (z.shape() == Shape{array_size, array_size});
                bool available_ok = z.is_available();
                
                // Get a value from the array
                int mid = array_size/2;
                auto sample = slice(z, {mid, mid}, {mid+1, mid+1});
                float actual = sample.item<float>();
                float expected = val1 + val2;
                
                bool values_match = (std::abs(actual - expected) < 1e-5);
                
                results.record_result(shape_ok, available_ok, values_match, expected, actual);
                
                if (shape_ok && available_ok && values_match) {
                    completed_ops++;
                }
            }
        } catch (const std::exception& e) {
            std::cerr << "Thread " << thread_id << " exception: " << e.what() << std::endl;
        }
    };

    // Run the threads with pre-created streams
    CHECK_NOTHROW(run_in_threads(num_threads, task, streams));
    
    // Check all results outside of threads
    for (size_t i = 0; i < results.shape_checks.size(); ++i) {
        CAPTURE(i); // Help identify which operation failed
        CHECK(results.shape_checks[i]);
        CHECK(results.availability_checks[i]);
        CHECK(results.value_checks[i]);
        if (!results.value_checks[i]) {
            CAPTURE(results.expected_values[i]);
            CAPTURE(results.actual_values[i]);
        }
    }
    
    // Verify all operations completed successfully
    CHECK_EQ(completed_ops.load(), num_threads * ops_per_thread);
}

TEST_CASE("test metal high contention on default stream eval") {
    Device D_GPU = Device::gpu;
    const int num_threads = std::thread::hardware_concurrency() > 0 ? std::thread::hardware_concurrency() : 8;
    const int ops_per_thread = 5;
    const int array_size = 16;
    Stream default_gpu_stream = default_stream(D_GPU);
    std::atomic<int> successful_ops{0};
    std::vector<std::string> thread_errors;
    std::mutex errors_mutex;
    TestResults results;

    auto task = [&](int thread_id) {
        try {
            for (int i = 0; i < ops_per_thread; ++i) {
                float val = static_cast<float>(thread_id * 100 + i + 1);
                auto x = full({array_size, array_size}, val, default_gpu_stream);
                auto y = full({array_size, array_size}, val * 0.5f, default_gpu_stream);
                auto z = multiply(x, y);
                eval(z);
                
                // Sample a value
                auto sample = slice(z, {0, 0}, {1, 1});
                float actual = sample.item<float>();
                float expected = val * val * 0.5f;
                
                bool shape_ok = (z.shape() == Shape{array_size, array_size});
                bool available_ok = z.is_available();
                bool values_match = (std::abs(actual - expected) < 1e-5);
                
                results.record_result(shape_ok, available_ok, values_match, expected, actual);
                
                if (shape_ok && available_ok && values_match) {
                    successful_ops++;
                }
            }
        } catch (const std::exception& e) {
            std::lock_guard<std::mutex> lock(errors_mutex);
            thread_errors.push_back(std::string("Thread ") + 
                                   std::to_string(thread_id) + 
                                   " exception: " + e.what());
        }
    };
    
    // Use the default helper for this test since it uses the default stream
    CHECK_NOTHROW(run_in_threads_default(num_threads, task));
    
    // Check for thread errors
    CHECK(thread_errors.empty());
    if (!thread_errors.empty()) {
        for (const auto& err : thread_errors) {
            CAPTURE(err);
        }
    }
    
    // Check all results
    for (size_t i = 0; i < results.shape_checks.size(); ++i) {
        CAPTURE(i);
        CHECK(results.shape_checks[i]);
        CHECK(results.availability_checks[i]);
        CHECK(results.value_checks[i]);
        if (!results.value_checks[i]) {
            CAPTURE(results.expected_values[i]);
            CAPTURE(results.actual_values[i]);
        }
    }
    
    // Verify operation count
    CHECK_EQ(successful_ops.load(), num_threads * ops_per_thread);
}

TEST_CASE("test metal concurrent graph eval from different threads") {
    Device D_GPU = Device::gpu;
    const int num_threads = std::thread::hardware_concurrency() > 0 ? std::thread::hardware_concurrency() : 4; // Keep modest for clarity
    const int array_size = 64;
    TestResults all_results;

    // Pre-create streams
    std::vector<Stream> streams;
    for (int i = 0; i < num_threads; ++i) {
        streams.push_back(new_stream(D_GPU));
    }
    synchronize();

    auto task = [&](int thread_id, Stream s) {
        try {
            float val1_base = static_cast<float>(thread_id + 1) * 10.0f;
            auto x = full({array_size, array_size}, val1_base, s);
            auto y = full({array_size, array_size}, val1_base + 1.0f, s);
            auto z = add(x, y);
            auto w = multiply(z, x);
            eval(w);

            float expected_val = (val1_base + (val1_base + 1.0f)) * val1_base;
            auto sample = slice(w, {0,0}, {1,1});
            float actual_val = sample.item<float>();

            bool shape_ok = (w.shape() == Shape{array_size, array_size});
            bool available_ok = w.is_available();
            bool value_ok = (std::abs(actual_val - expected_val) < 1e-4);

            all_results.record_result(shape_ok, available_ok, value_ok, expected_val, actual_val);

        } catch (const std::exception& e) {
            std::cerr << "Thread " << thread_id << " exception in concurrent graph eval: " << e.what() << std::endl;
        }
    };

    CHECK_NOTHROW(run_in_threads(num_threads, task, streams));

    CHECK_EQ(all_results.shape_checks.size(), num_threads); // One result per thread
    for (size_t i = 0; i < num_threads; ++i) {
        CAPTURE(i);
        CHECK(all_results.shape_checks[i]);
        CHECK(all_results.availability_checks[i]);
        CHECK(all_results.value_checks[i]);
        if (!all_results.value_checks[i]) {
            CAPTURE(all_results.expected_values[i]);
            CAPTURE(all_results.actual_values[i]);
        }
    }
}