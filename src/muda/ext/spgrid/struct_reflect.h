#pragma once
#include "Macro.hpp"
#include <type_traits>
#include <string_view>
#include "StructType.hpp"
#include "SPGridLayout.hpp"

#define MUDA_SPGRID_MAKE_MEMBER_TYPE(m) \
    using member_type_##m = std::_Remove_cvref_t< \
            decltype(std::declval<this_type>().m)>;

#define MUDA_SOA_MEMBER_OFFSET(m)\
SparseGrid::compute_offset<member_type_##m>()


#define MUDA_SPGRID_STRUCT(S , ...)\
    MUDA_SPGRID_SOA_IMPL(S , __VA_ARGS__)\
    MUDA_SPGRID_REFLECT_STRUCT_IMPL(S , __VA_ARGS__)




#define MUDA_SPGRID_REFLECT_STRUCT_IMPL(S, ...) \
    template<> \
    class SparseGrid::StructReflect<S> { \
        private: \
            using this_type = S; \
        public:    \
        MUDA_MAP(MUDA_SPGRID_MAKE_MEMBER_TYPE, __VA_ARGS__) \
        static constexpr size_t _member_index(std::string_view name) noexcept { \
                constexpr const std::string_view member_names[]{ \
                    MUDA_MAP_LIST(MUDA_STRINGIFY, __VA_ARGS__) \
                }; \
                return std::find(std::begin(member_names), \
                                std::end(member_names), \
                                name) - \
                       std::begin(member_names); \
            } \
            static constexpr std::string_view  _member_name(size_t index) noexcept { \
                constexpr const std::string_view _member_names[]{ \
                    MUDA_MAP_LIST(MUDA_STRINGIFY, __VA_ARGS__) \
                }; \
                return _member_names[index]; \
            } \
    };

#define MUDA_SPGRID_SOA_IMPL(S , ...)\
    template<> \
    class SparseGrid::StructSOA<S>{\
    private:\
        using this_type = S;\
        MUDA_MAP(MUDA_SPGRID_MAKE_MEMBER_TYPE, __VA_ARGS__)\
        constexpr static unsigned int SOA_offset[]{\
        MUDA_MAP_LIST(MUDA_SOA_MEMBER_OFFSET, __VA_ARGS__) \
        }; \
    public:\
    static constexpr size_t SOA_Channel_count = sizeof(SOA_offset) / sizeof(SOA_offset[0]);\
        static constexpr unsigned int offset_all(){\
            constexpr unsigned int total_offset = [](){\
                unsigned int sum = 0;\
                for(size_t i = 0 ; i < sizeof(SOA_offset) / sizeof(SOA_offset[0]) ; i++){\
                    sum += SOA_offset[i];\
                }\
                return sum;}();\
            return total_offset;};\
        };
    