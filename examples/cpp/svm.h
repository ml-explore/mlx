#pragma once
#include "mlx/mlx.h"
#include <iostream>

namespace mx = mlx::core; 


class SVM {
    // Private Attributes
private :
    mx::array weights_;
    mx::array bias_; 
 

    float lr_; // learning rate
    float lambda_;  // lambda 

public : 
    // --------------------------------
    // Constructor
    // --------------------------------
    
    SVM(int n_features,float lr = 0.01f, float lambda = 0.01f): 
        weights_(mx::zeros({n_features}, mx::float32)),
        bias_(mx::zeros({}, mx::float32)),
        lr_(lr),
        lambda_(lambda) {
            if (n_features <=0 ) {
                throw std::invalid_argument("n_features must be positive");
            }
            if (lr <= 0.0f ) {
                throw std::invalid_argument("learning must be positive");
            }
            if (lambda < 0.0f) {
                throw std::invalid_argument("Lambda must be non-negayive");
            }

        };

    
 
    // --------------------------------
    // Loss function
    // --------------------------------

    mx::array loss(const mx::array& X, const mx::array& Y) {
        auto scores = mx::add(mx::matmul(X, weights_), bias_);
        auto margins = mx::multiply(Y, scores);
        auto hinge = mx::maximum(mx::subtract(mx::array(1.0f), margins), mx::array(0.0f));
        auto hinge_loss = mx::mean(hinge);
        auto regularization = mx::multiply(mx::array(lambda_) / mx::array(2.0f), mx::sum(mx::square(weights_)));

        return mx::add(
            hinge_loss,
            regularization
        );

    };

    // --------------------------------
    // Predict function
    // --------------------------------
    mx::array predict(const mx::array& X) const {
        auto scores = mx::add(mx::matmul(X, weights_), bias_);
        return mx::where(mx::greater_equal(scores, mx::array(0.0f)), mx::array(1.0f), mx::array(-1.0f));
    };

    // --------------------------------
    // Training function
    // --------------------------------

    void fit(const mx::array& X, const mx::array& Y, int epochs) {
        auto loss_fn = [this, &X, &Y](const std::vector<mx::array>& params) -> mx::array { 
            const auto& weights = params[0]; 
            const auto& bias = params[1];
            auto scores = mx::add(mx::matmul(X, weights), bias);
            auto margins = mx::multiply(Y, scores); 
            auto hinge = mx::maximum(mx::subtract(mx::array(1.0f), margins), mx::array(0.0f));
            auto hinge_loss = mx::mean(hinge); 
            auto regularization =
                    mx::multiply(
                        mx::array(lambda_ / 2.0f),
                        mx::sum(mx::square(weights))
                    );

            return mx::add(hinge_loss, regularization);

        };

        auto value_and_grad_fn = mx::value_and_grad(
            loss_fn,
            std::vector<int>{0, 1}
        );


        for (int epoch = 0; epoch < epochs; ++epoch) {

            std::vector<mx::array> params = {
                weights_,
                bias_
            };


            auto [values, gradients] = value_and_grad_fn(params);


            auto loss_value = values;


            auto grad_weights = gradients[0];
            auto grad_bias = gradients[1];


            // Gradient descent

            weights_ =
                mx::subtract(
                    weights_,
                    mx::multiply(
                        mx::array(lr_),
                        grad_weights
                    )
                );


            bias_ =
                mx::subtract(
                    bias_,
                    mx::multiply(
                        mx::array(lr_),
                        grad_bias
                    )
                );


            // Evaluate computation graph

            mx::eval(
                weights_,
                bias_,
                loss_value
            );


            if (epoch % 100 == 0) {

                std::cout
                    << "Epoch: "
                    << epoch
                    << " | Loss: "
                    << loss_value.item<float>()
                    << std::endl;
            }
        }

    };


    // --------------------------------
    // Accuracy
    // --------------------------------
    // TODO: accuracy func is decleared const type
    float accuracy(
        const mx::array& X,
        const mx::array& Y
    ) const {

        auto predictions = predict(X);

        auto correct =
            mx::equal(
                predictions,
                Y
            );

        auto accuracy =
            mx::mean(
                mx::astype(
                    correct,
                    mx::float32
                )
            );

        return accuracy.item<float>();
    }

};




