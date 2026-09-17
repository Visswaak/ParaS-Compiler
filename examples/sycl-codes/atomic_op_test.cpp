#include <iostream>
#include <sycl/sycl.hpp>

class IntKernel;
class ShortKernel;

int main() {
    sycl::queue q;

    int *a = sycl::malloc_shared<int>(1, q);
    short *b = sycl::malloc_shared<short>(1, q);

    *a = 0;
    *b = 0;

    q.parallel_for<IntKernel>(sycl::range<1>(1000), [=](sycl::id<1>) {
        sycl::atomic_ref<int> x(*a);
        x.fetch_add(1);
    });

    q.parallel_for<ShortKernel>(sycl::range<1>(1000), [=](sycl::id<1>) {
        sycl::atomic_ref<short> x(*b);
        x.fetch_add(1);
    });

    q.wait();

    std::cout << "int   = " << *a << " (expected 1000)\n";
    std::cout << "short = " << *b << " (expected 1000)\n";

    sycl::free(a, q);
    sycl::free(b, q);

    return 0;
}