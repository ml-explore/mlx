#include <iostream>
#include <vector>
#include <cassert>
#include "your_header_file.h" // Replace with the actual header file name

void test_diag_1d() {
    std::vector<int> input = {1, 2, 3, 4, 5};
    std::vector<std::vector<int>> expected_output = {
        {1, 0, 0, 0, 0},
        {0, 2, 0, 0, 0},
        {0, 0, 3, 0, 0},
        {0, 0, 0, 4, 0},
        {0, 0, 0, 0, 5}
    };

    std::vector<std::vector<int>> result = diag(input);

    assert(result == expected_output);
    std::cout << "Test for diag with 1D input passed!" << std::endl;
}

void test_diag_2d() {
    std::vector<std::vector<int>> input = {
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9}
    };
    std::vector<int> expected_output = {1, 5, 9};

    std::vector<int> result = diag(input);

    assert(result == expected_output);
    std::cout << "Test for diag with 2D input passed!" << std::endl;
}

int main() {
    test_diag_1d();
    test_diag_2d();

    return 0;
}