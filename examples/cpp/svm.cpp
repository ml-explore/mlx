#include "SVM.h"
#include "mlx/mlx.h"
#include <iostream>

namespace mx = mlx::core;


int main() {

    // --------------------------------
    // Training data
    // --------------------------------

    mx::array X(
        {
            2.0f, 3.0f,
            3.0f, 4.0f,
            4.0f, 2.0f,
            5.0f, 3.0f,

            -2.0f, -3.0f,
            -3.0f, -4.0f,
            -4.0f, -2.0f,
            -5.0f, -3.0f
        },
        {8, 2}
    );


    // Labels must be -1 or +1

    mx::array Y(
        {
            1.0f,
            1.0f,
            1.0f,
            1.0f,

            -1.0f,
            -1.0f,
            -1.0f,
            -1.0f
        },
        {8}
    );


    // --------------------------------
    // Create SVM
    // --------------------------------

    SVM svm(
        2,          // Number of features
        0.01f,      // Learning rate
        0.01f       // Lambda
    );


    // --------------------------------
    // Train
    // --------------------------------

    svm.fit(
        X,
        Y,
        1000
    );


    // --------------------------------
    // Evaluate
    // --------------------------------

    float acc =
        svm.accuracy(X, Y);


    std::cout
        << "\nTraining Accuracy: "
        << acc * 100.0f
        << "%"
        << std::endl;


    // --------------------------------
    // Prediction
    // --------------------------------

    auto predictions =
        svm.predict(X);


    std::cout
        << "\nPredictions:\n"
        << predictions
        << std::endl;


    return 0;
}