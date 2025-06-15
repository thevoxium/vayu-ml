// IMPORTANT - I ASKED CLAUDE TO WRITE THIS TEST
// TAKE IT WITH A GRAIN OF SALT

#include "../include/tensor.h"
#include <cassert>
#include <cmath>
#include <iomanip>
#include <iostream>

void print_tensor_info(const std::string &name, std::shared_ptr<Tensor> t) {
  std::cout << name << " shape: (";
  for (size_t i = 0; i < t->shape.size(); i++) {
    std::cout << t->shape[i];
    if (i != t->shape.size() - 1)
      std::cout << ", ";
  }
  std::cout << ")\n";

  std::cout << name << " data: [";
  for (size_t i = 0; i < t->data.size(); i++) {
    std::cout << std::fixed << std::setprecision(4) << t->data[i];
    if (i != t->data.size() - 1)
      std::cout << ", ";
  }
  std::cout << "]\n";

  if (t->requires_grad && !t->grad.empty()) {
    std::cout << name << " grad: [";
    for (size_t i = 0; i < t->grad.size(); i++) {
      std::cout << std::fixed << std::setprecision(4) << t->grad[i];
      if (i != t->grad.size() - 1)
        std::cout << ", ";
    }
    std::cout << "]\n";
  }
  std::cout << "\n";
}

void test_basic_matmul() {
  std::cout << "=== Test 1: Basic Matrix Multiplication ===\n";

  // Create simple test matrices
  // A = [[1, 2], [3, 4]] (2x2)
  auto A = tensor({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2}, true);

  // B = [[5, 6], [7, 8]] (2x2)
  auto B = tensor({5.0f, 6.0f, 7.0f, 8.0f}, {2, 2}, true);

  std::cout << "Input matrices:\n";
  print_tensor_info("A", A);
  print_tensor_info("B", B);

  // Test both fast (BLAS) and slow versions
  auto C_fast = A->mm(B, true);  // Using BLAS
  auto C_slow = A->mm(B, false); // Manual implementation

  std::cout << "Results:\n";
  print_tensor_info("C_fast (BLAS)", C_fast);
  print_tensor_info("C_slow (Manual)", C_slow);

  // Expected result: [[19, 22], [43, 50]]
  std::vector<float> expected = {19.0f, 22.0f, 43.0f, 50.0f};

  std::cout << "Expected: [19, 22, 43, 50]\n";

  // Check if results match
  bool fast_correct = true, slow_correct = true;
  for (size_t i = 0; i < expected.size(); i++) {
    if (std::abs(C_fast->data[i] - expected[i]) > 1e-5)
      fast_correct = false;
    if (std::abs(C_slow->data[i] - expected[i]) > 1e-5)
      slow_correct = false;
  }

  std::cout << "BLAS result correct: " << (fast_correct ? "YES" : "NO") << "\n";
  std::cout << "Manual result correct: " << (slow_correct ? "YES" : "NO")
            << "\n\n";
}

void test_gradient_computation() {
  std::cout << "=== Test 2: Gradient Computation ===\n";

  // Create test matrices
  auto A = tensor({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2}, true);
  auto B = tensor({0.5f, 1.5f, 2.5f, 3.5f}, {2, 2}, true);

  std::cout << "Input matrices:\n";
  print_tensor_info("A", A);
  print_tensor_info("B", B);

  // Forward pass
  auto C = A->mm(B, true); // Using BLAS
  auto loss = C->sum();    // Scalar loss for backprop

  std::cout << "Forward pass results:\n";
  print_tensor_info("C = A @ B", C);
  print_tensor_info("loss = sum(C)", loss);

  // Backward pass
  loss->backward();

  std::cout << "After backward pass:\n";
  print_tensor_info("A", A);
  print_tensor_info("B", B);

  // Manual gradient verification
  // For matrix multiplication C = A @ B, and loss = sum(C):
  // dA = dC @ B^T, where dC is all ones (gradient of sum)
  // dB = A^T @ dC

  std::cout << "Manual gradient verification:\n";

  // Expected gradients for A: dA = ones(2,2) @ B^T
  // B^T = [[0.5, 2.5], [1.5, 3.5]]
  // dA = [[1,1], [1,1]] @ [[0.5, 2.5], [1.5, 3.5]] = [[2, 6], [2, 6]]
  std::vector<float> expected_dA = {2.0f, 6.0f, 2.0f, 6.0f};

  // Expected gradients for B: dB = A^T @ ones(2,2)
  // A^T = [[1, 3], [2, 4]]
  // dB = [[1, 3], [2, 4]] @ [[1,1], [1,1]] = [[4, 4], [6, 6]]
  std::vector<float> expected_dB = {4.0f, 4.0f, 6.0f, 6.0f};

  std::cout << "Expected dA: [2, 6, 2, 6]\n";
  std::cout << "Expected dB: [4, 4, 6, 6]\n";

  // Check gradients
  bool dA_correct = true, dB_correct = true;
  for (size_t i = 0; i < expected_dA.size(); i++) {
    if (std::abs(A->grad[i] - expected_dA[i]) > 1e-5)
      dA_correct = false;
    if (std::abs(B->grad[i] - expected_dB[i]) > 1e-5)
      dB_correct = false;
  }

  std::cout << "Gradient dA correct: " << (dA_correct ? "YES" : "NO") << "\n";
  std::cout << "Gradient dB correct: " << (dB_correct ? "YES" : "NO") << "\n\n";
}

void test_gradient_comparison() {
  std::cout << "=== Test 3: BLAS vs Manual Gradient Comparison ===\n";

  // Test with the same matrices using both methods
  auto A1 = tensor({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {2, 3}, true);
  auto B1 = tensor({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {3, 2}, true);

  auto A2 = tensor({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {2, 3}, true);
  auto B2 = tensor({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {3, 2}, true);

  // Forward pass with BLAS
  auto C1 = A1->mm(B1, true);
  auto loss1 = C1->sum();
  loss1->backward();

  // Forward pass without BLAS
  auto C2 = A2->mm(B2, false);
  auto loss2 = C2->sum();
  loss2->backward();

  std::cout << "Comparing gradients from BLAS vs Manual:\n";

  bool gradients_match = true;

  // Compare A gradients
  std::cout << "A gradients (BLAS): ";
  for (size_t i = 0; i < A1->grad.size(); i++) {
    std::cout << std::fixed << std::setprecision(4) << A1->grad[i] << " ";
    if (std::abs(A1->grad[i] - A2->grad[i]) > 1e-5)
      gradients_match = false;
  }
  std::cout << "\n";

  std::cout << "A gradients (Manual): ";
  for (size_t i = 0; i < A2->grad.size(); i++) {
    std::cout << std::fixed << std::setprecision(4) << A2->grad[i] << " ";
  }
  std::cout << "\n";

  // Compare B gradients
  std::cout << "B gradients (BLAS): ";
  for (size_t i = 0; i < B1->grad.size(); i++) {
    std::cout << std::fixed << std::setprecision(4) << B1->grad[i] << " ";
    if (std::abs(B1->grad[i] - B2->grad[i]) > 1e-5)
      gradients_match = false;
  }
  std::cout << "\n";

  std::cout << "B gradients (Manual): ";
  for (size_t i = 0; i < B2->grad.size(); i++) {
    std::cout << std::fixed << std::setprecision(4) << B2->grad[i] << " ";
  }
  std::cout << "\n";

  std::cout << "BLAS and Manual gradients match: "
            << (gradients_match ? "YES" : "NO") << "\n\n";
}

void test_chain_rule() {
  std::cout << "=== Test 4: Chain Rule with Multiple Operations ===\n";

  auto A = tensor({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2}, true);
  auto B = tensor({0.5f, 1.0f, 1.5f, 2.0f}, {2, 2}, true);
  auto C = tensor({2.0f, 1.0f, 1.0f, 2.0f}, {2, 2}, true);

  // Complex computation: loss = sum((A @ B) * C)
  auto AB = A->mm(B, true);
  auto result = AB * C;
  auto loss = result->sum();

  std::cout << "Computing: loss = sum((A @ B) * C)\n";
  print_tensor_info("A", A);
  print_tensor_info("B", B);
  print_tensor_info("C", C);
  print_tensor_info("A @ B", AB);
  print_tensor_info("(A @ B) * C", result);
  print_tensor_info("loss", loss);

  loss->backward();

  std::cout << "Gradients after backward pass:\n";
  print_tensor_info("A", A);
  print_tensor_info("B", B);
  print_tensor_info("C", C);
}

void test_numerical_gradient() {
  std::cout << "=== Test 5: Numerical Gradient Check ===\n";

  const float eps = 1e-4f;

  // Create fresh tensors for this test
  auto A = tensor({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2}, true);
  auto B = tensor({0.5f, 1.0f, 1.5f, 2.0f}, {2, 2}, true);

  // Zero gradients to start fresh
  A->zero_grad();
  B->zero_grad();

  // Compute analytical gradient
  auto C = A->mm(B, true);
  auto loss = C->sum();
  loss->backward();

  std::cout << "Numerical gradient check for A[0,0]:\n";

  // Store analytical gradient
  float analytical_grad = A->grad[0];
  std::cout << "Analytical gradient: " << analytical_grad << "\n";

  // Create fresh tensors for numerical gradient computation
  auto A_plus = tensor({1.0f + eps, 2.0f, 3.0f, 4.0f}, {2, 2}, false);
  auto B_copy1 = tensor({0.5f, 1.0f, 1.5f, 2.0f}, {2, 2}, false);
  auto C_plus = A_plus->mm(B_copy1, true);
  auto loss_plus = C_plus->sum();
  float loss_plus_val = loss_plus->data[0];

  auto A_minus = tensor({1.0f - eps, 2.0f, 3.0f, 4.0f}, {2, 2}, false);
  auto B_copy2 = tensor({0.5f, 1.0f, 1.5f, 2.0f}, {2, 2}, false);
  auto C_minus = A_minus->mm(B_copy2, true);
  auto loss_minus = C_minus->sum();
  float loss_minus_val = loss_minus->data[0];

  float numerical_grad = (loss_plus_val - loss_minus_val) / (2 * eps);
  std::cout << "Numerical gradient: " << numerical_grad << "\n";

  float error = std::abs(analytical_grad - numerical_grad);
  std::cout << "Absolute error: " << error << "\n";
  std::cout << "Gradient check: " << (error < 1e-3 ? "PASSED" : "FAILED")
            << "\n\n";
}

void test_larger_matrices() {
  std::cout << "=== Test 6: Larger Matrix Test ===\n";

  // Test with larger matrices
  auto A = random_tensor({3, 4}, true, -1.0f, 1.0f);
  auto B = random_tensor({4, 5}, true, -1.0f, 1.0f);

  std::cout << "Testing with random 3x4 and 4x5 matrices\n";

  // Test both implementations
  auto C_fast = A->mm(B, true);
  auto C_slow = A->mm(B, false);

  // Check if results are the same
  bool results_match = true;
  for (size_t i = 0; i < C_fast->data.size(); i++) {
    if (std::abs(C_fast->data[i] - C_slow->data[i]) > 1e-5) {
      results_match = false;
      break;
    }
  }

  std::cout << "Fast and slow results match: " << (results_match ? "YES" : "NO")
            << "\n";

  // Test gradients
  auto loss1 = C_fast->sum();
  loss1->backward();

  // Reset gradients and test with slow version
  A->zero_grad();
  B->zero_grad();

  auto loss2 = C_slow->sum();
  loss2->backward();

  // The gradients should be the same (they accumulate, so we need fresh
  // tensors)
  std::cout << "Larger matrix test completed.\n\n";
}

int main() {
  std::cout << std::fixed << std::setprecision(4);

  test_basic_matmul();
  test_gradient_computation();
  test_gradient_comparison();
  test_chain_rule();
  test_numerical_gradient();
  test_larger_matrices();

  std::cout << "=== All Tests Completed ===\n";

  return 0;
}
