// This function does nothing. This is mostly used to trigger a warm start of
// the kaas server without poluting the cache or other state.
extern "C"
__global__ void noop(void)
{
    return;
}
