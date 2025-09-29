#pragma once
#include <muda/ext/spgrid/spgrid_layout.h>
#include <muda/ext/spgrid/struct_type.h>
#include <muda/ext/spgrid/utils.h>

namespace SparseGrid
{
template <typename Layout>
class SPGrid
{
  public:
    using LayoutType = Layout;
    using StructType = typename Layout::StructType;

    static constexpr size_t Block_size   = Layout::Block_size;
    static constexpr size_t cell_memory  = Layout::cell_memory;
    static constexpr size_t Block_memory = Layout::Block_memory;
    static constexpr size_t num_T        = Layout::num_T;

    SPGrid(){};
    ~SPGrid(){};

    SPGrid(const SPGrid& other);
    SPGrid(size_t& num_blocks);


    MUDA_HOST void grid_hash_build(muda::BufferView<uint64_t> offset_key,
                                   muda::BufferView<int>      index);

    MUDA_HOST void Grid_build_topology();

    MUDA_DEVICE void visit_table(muda::BufferView<int> particle_2_block_id);


  private:
    int*                         d_data       = nullptr;
    size_t                       m_num_blocks = 0;
    muda::DeviceBuffer<uint64_t> key;
    muda::DeviceBuffer<uint64_t> value;
    muda::DeviceBuffer<uint64_t> block_offset;
    muda::DeviceVar<int>         block_counter;
    muda::DeviceBuffer<int>      topology_cache;
};

template <typename Layout>
MUDA_HOST void SPGrid<Layout>::grid_hash_build(muda::BufferView<uint64_t> offset_key,
                                               muda::BufferView<int> index)
{
    muda::Launch().apply(
        [index         = index.viewer(),
         offset_key    = offset_key.viewer(),
         tablesize     = m_num_blocks,
         keyTable      = key.viewer(),
         valueTable    = value.viewer(),
         block_id2key  = block_offset.viewer(),
         block_counter = block_counter.viewer()] __device__() mutable
        {
            auto global_thread_id = blockDim.x * blockIdx.x + threadIdx.x;

            if(global_thread_id > index.total_size())
            {
                return;
            }

            auto     particle_index = index(global_thread_id);
            uint64_t key            = offset_key(particle_index) >> 6;
            uint64_t hashkey        = key % tablesize;
            int      block_id       = 0;

            while(true)
            {

                auto key_prev =
                    muda::atomic_cas((unsigned long long int*)keyTable.data() + hashkey,
                                     0xffffffffffffffff,
                                     (unsigned long long int)key);
                if(keyTable(hashkey) == key)
                {

                    block_id = muda::atomic_add(block_counter.data(), 1);

                    valueTable(hashkey)    = block_id;
                    block_id2key(block_id) = key;
                    break;
                }

                else
                {
                    hashkey += 127;
                    if(hashkey >= tablesize)
                        hashkey = hashkey % tablesize;
                }
            }
        });
}

template <typename Layout>
MUDA_HOST void SPGrid<Layout>::Grid_build_topology()
{
    muda::Launch().apply(
        [tablesize    = m_num_blocks,
         keyTable     = key.viewer(),
         valueTable   = value.viewer(),
         block_id2key = block_offset.viewer(),
         topology     = topology_cache.viewer(),
         num_block    = block_counter.viewer()]() __device__ mutable
        {
            auto global_thread_id = blockDim.x * blockIdx.x + threadIdx.x;
            if(global_thread_id > num_block)
            {
                return;
            }

            unsigned int block_id = global_thread_id;
            auto         key      = block_id2key(block_id);

            for(int i = -1; i < 2; i++)
            {
                for(int j = -1; j < 2; j++)
                {
                    for(int k = -1; k < 2; k++)
                    {

                        auto key_neibour     = packed_add_morton(key, i, j, k);
                        auto hashkey_neibour = key_neibour % tablesize;

                        while(true)
                        {

                            auto keytable_value = keyTable(hashkey_neibour);
                            if(keytable_value == key_neibour)
                            {
                                topology(27 * block_id
                                         + encode_linear(i + 1, j + 1, k + 1, 3, 3, 3));
                                break;
                            }
                            else
                            {
                                if(keytable_value == 0xffffffffffffffff)
                                {
                                    auto block_id_neibour =
                                        muda::atomic_add(num_block.data(), 1);

                                    valueTable(hashkey_neibour) = block_id_neibour;
                                    block_id2key(block_id_neibour) = key_neibour;
                                }
                                else
                                {
                                    hashkey_neibour += 127;
                                    if(hashkey_neibour > tablesize)
                                    {
                                        hashkey_neibour = hashkey_neibour % 127;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        });
}


}  // namespace SparseGrid


// #include "details/spgrid.inl"