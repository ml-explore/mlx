// Copyright © 2025 Apple Inc.
// Windows CUDA initialization support
//
// This file is intentionally minimal. With dynamic CUDA runtime linking,
// the cudart DLL handles kernel registration automatically when loaded.
// No early initialization is needed.

#ifdef _WIN32
// Empty - using dynamic CUDA runtime handles initialization automatically
#endif // _WIN32
