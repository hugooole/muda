#include "catch2/catch.hpp"
#include "hello_muda/hello_muda.h"
#include "hello_muda/muda_vs_cuda.h"
#include "muda/container.h"

TEST_CASE("muda_overview", "[quick_start]")
{
    quick_overview();
}

TEST_CASE("hello_muda", "[quick_start]")
{
    hello_muda();
}

TEST_CASE("muda_vs_cuda", "[quick_start]")
{
    muda_vs_cuda();
}

TEST_CASE("vector_add", "[quick start]")
{
    HostVector<float> gt_C, C;
    vector_add(gt_C, C);
    REQUIRE(gt_C == C);
}