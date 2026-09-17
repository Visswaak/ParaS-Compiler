// Exercises the transparent sycl::plus<> function object.

#include <sycl/sycl.hpp>

#include <iostream>
#include <type_traits>

int main() {
  // Compile-time checks for common arithmetic types and mixed operands.
  static_assert(std::is_same_v<decltype(sycl::plus<>{}(1, 2)), int>);
  static_assert(
      std::is_same_v<decltype(sycl::plus<>{}(1.0f, 2.0f)), float>);
  static_assert(
      std::is_same_v<decltype(sycl::plus<>{}(1, 2.0)), double>);

  sycl::queue queue;
  constexpr std::size_t count = 256;
  constexpr std::size_t local_size = 64;

  int *values = sycl::malloc_shared<int>(count, queue);
  int *results = sycl::malloc_shared<int>(count, queue);

  for (std::size_t i = 0; i < count; ++i) {
    values[i] = 1;
    results[i] = 0;
  }

  queue.submit([&](sycl::handler &handler) {
    handler.parallel_for(
        sycl::nd_range<1>(sycl::range<1>(count), sycl::range<1>(local_size)),
        [=](sycl::nd_item<1> item) {
          const std::size_t index = item.get_global_id(0);
          int value = values[index];

          // The transparent functor is instantiated inside the kernel.
          value = sycl::reduce_over_group(item.get_group(), value,
                                          sycl::plus<>());
          results[index] = value;
        });
  });
  queue.wait_and_throw();

  bool passed = true;
  for (std::size_t i = 0; i < count; ++i)
    passed = passed && (results[i] == 1);

  sycl::free(values, queue);
  sycl::free(results, queue);

  std::cout << (passed ? "sycl::plus<> test passed\n"
                       : "sycl::plus<> test failed\n");
  return passed ? 0 : 1;
}
