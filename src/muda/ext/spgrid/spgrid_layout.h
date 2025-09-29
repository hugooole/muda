#pragma once
#include <cuda_runtime.h>
#include "StructType.hpp"
#include "StructReflect.hpp"
namespace SparseGrid{


template <typename Struct>
struct StructSOA{};

template <typename Struct>
struct StructReflect{};


template<typename S , size_t nx , size_t ny , size_t nz>
class SPGridLayout{

    public:

    static constexpr size_t Block_size = nx * ny * nz;
    static constexpr size_t cell_memory = StructSOA<S>::offset_all() ;
    static constexpr size_t Block_memory = cell_memory * Block_size;
    static constexpr size_t num_T = Block_memory / sizeof(int);
    
    StructSOA<S> soa_info;
    StructReflect<S> reflect_info;

};

template<typename ChannelType>
constexpr static unsigned int compute_offset()
requires SparseGrid::ScalarType<ChannelType> || SparseGrid::VectorType<ChannelType> ||
         SparseGrid::MatrixType<ChannelType>
{
    if constexpr (SparseGrid::is_scalar_v<ChannelType>){
        return sizeof(typename ChannelType::type);
    }
    else if constexpr (SparseGrid::is_vector_v<ChannelType>){
        return sizeof(typename ChannelType::type) * ChannelType::size;
    }
    else if constexpr (SparseGrid::is_matrix_v<ChannelType>){
        return sizeof(typename ChannelType::type) * ChannelType::rows * ChannelType::cols;
    }
};










};