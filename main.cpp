#include <iostream>
using namespace std;

// Function to calculate Mean Squared Error (MSE)
double mse(double y[], double y_hat[], size_t size) {
    double err = 0;
    for (size_t i = 0; i < size; ++i) {
        err += (y[i] - y_hat[i]) * (y[i] - y_hat[i]);
    }
    return err / size;
}

// Function to calculate the predicted output
double* linear(double x[], double w, double b, size_t size) {
    double* y_hat = new double[size];
    for (size_t i = 0; i < size; ++i) {
        y_hat[i] = w * x[i] + b;
    }
    return y_hat;
}

// Function to compute gradients and update weights
void backpropagation(double x[], double y[], double& w, double& b, size_t size, double learning_rate) {
    // Step 1: Calculate y_hat
    double* y_hat = linear(x, w, b, size);

    // Step 2: Compute gradients
    double grad_w = 0.0;
    double grad_b = 0.0;
    for (size_t i = 0; i < size; ++i) {
        double error = y_hat[i] - y[i];
        grad_w += error * x[i];
        grad_b += error;
    }
    grad_w *= (2.0 / size);
    grad_b *= (2.0 / size);

    // Step 3: Update weights and bias
    w -= learning_rate * grad_w;
    b -= learning_rate * grad_b;

    // Clean up dynamically allocated memory
    delete[] y_hat;
}

int main() {
    double x[10] = {1, 3, 4, 6, 11, 14, 15, 17, 16, 19};
    double y[10] = {2.2, 3.9, 5.1, 7.3, 12.1, 15.2, 17.4, 19.1, 18.0, 21.7};

    // Initial weight and bias
    double w = 0.0;
    double b = 0.0;

    // Calculate initial prediction and error
    double* y_hat = linear(x, w, b, 10);
    double error = mse(y, y_hat, 10);

    // Print initial error rate
    cout << "Initial error rate: " << error << endl;

    // Learning rate and number of iterations
    double learning_rate = 0.01;
    size_t iterations = 1000;

    // Training loop
    for (size_t i = 0; i < iterations; ++i) {
        backpropagation(x, y, w, b, 10, learning_rate);
    }

    // Final weight and bias
    cout << "Final weight (w): " << w << endl;
    cout << "Final bias (b): " << b << endl;

    // Recalculate y_hat and error with updated weights
    y_hat = linear(x, w, b, 10);
    error = mse(y, y_hat, 10);
    cout << "Final error rate: " << error << endl;

    // Free the allocated memory
    delete[] y_hat;

    return 0;
}
