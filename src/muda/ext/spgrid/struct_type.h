#pragma once
#include <muda/ext/spgrid/macro.h>
#include <type_traits>
#include <concepts>

namespace SparseGrid{

template <typename T>
struct Scalar{
    using type = T;

    T value;
};

template <typename T ,size_t N>
struct Vector{
    using type = T;
    static constexpr size_t size = N;

    T value[N];
};

template <typename T , size_t row , size_t col>
struct Matrix{
    using type = T;
    static constexpr size_t rows = row;
    static constexpr size_t cols = col;

    T value[row*col];
};

// Type traits for Scalar
template<typename T>
struct is_scalar : std::false_type {};

template<typename T>
struct is_scalar<Scalar<T>> : std::true_type {};

template<typename T>
inline constexpr bool is_scalar_v = is_scalar<T>::value;

// Type traits for Vector
template<typename T>
struct is_vector : std::false_type {};

template<typename T, size_t N>
struct is_vector<Vector<T, N>> : std::true_type {};

template<typename T>
inline constexpr bool is_vector_v = is_vector<T>::value;

// Type traits for Matrix
template<typename T>
struct is_matrix : std::false_type {};

template<typename T, size_t row, size_t col>
struct is_matrix<Matrix<T, row, col>> : std::true_type {};

template<typename T>
inline constexpr bool is_matrix_v = is_matrix<T>::value;

template<typename T>
concept ScalarType = is_scalar_v<T>;

template<typename T>
concept VectorType = is_vector_v<T>;

template<typename T>
concept MatrixType = is_matrix_v<T>;

template<typename T>
struct underlying_type {
    using type = T; // 默认情况
};

template<typename T>
struct underlying_type<Scalar<T>> {
    using type = T;
};

template<typename T, size_t N>
struct underlying_type<Vector<T, N>> {
    using type = T;
};

template<typename T, size_t row, size_t col>
struct underlying_type<Matrix<T, row, col>> {
    using type = T;
};

template<typename T>
using underlying_type_t = typename underlying_type<T>::type;

}



