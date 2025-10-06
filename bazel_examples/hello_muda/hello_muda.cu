#include <muda/muda.h>
#include <muda/buffer.h>
#include <example_common.h>
#include "hello_muda/hello_muda.h"

using namespace muda;
void hello_muda()
{
    example_desc("say hello in muda");
    Launch(1, 1).apply([] __device__() { print("hello muda!\n"); }).wait();
}

void quick_overview()
{
    example_desc("use parallel_for to say hello.");
    Stream s;
    on(s)
        .next(ParallelFor(2, 32))
        .apply(4,
               [] __device__(int i)
               { print("hello muda for %d/4 rounds\n", i + 1); })
        .wait();
}
