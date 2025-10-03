
//copy write LuisaCompute/include/core/Macro

#pragma once

#define MUDA_MACRO_EVAL0(...) __VA_ARGS__
#define MUDA_MACRO_EVAL1(...) MUDA_MACRO_EVAL0(MUDA_MACRO_EVAL0(MUDA_MACRO_EVAL0(__VA_ARGS__)))
#define MUDA_MACRO_EVAL2(...) MUDA_MACRO_EVAL1(MUDA_MACRO_EVAL1(MUDA_MACRO_EVAL1(__VA_ARGS__)))
#define MUDA_MACRO_EVAL3(...) MUDA_MACRO_EVAL2(MUDA_MACRO_EVAL2(MUDA_MACRO_EVAL2(__VA_ARGS__)))
#define MUDA_MACRO_EVAL4(...) MUDA_MACRO_EVAL3(MUDA_MACRO_EVAL3(MUDA_MACRO_EVAL3(__VA_ARGS__)))
#define MUDA_MACRO_EVAL5(...) MUDA_MACRO_EVAL4(MUDA_MACRO_EVAL4(MUDA_MACRO_EVAL4(__VA_ARGS__)))
#define MUDA_MACRO_EVAL(...) MUDA_MACRO_EVAL5(__VA_ARGS__)

#define MUDA_MACRO_EMPTY()
#define MUDA_MACRO_DEFER(id) id MUDA_MACRO_EMPTY()

// macro reverse
#define MUDA_REVERSE_END(...)
#define MUDA_REVERSE_OUT

#define MUDA_REVERSE_GET_END2() 0, MUDA_REVERSE_END
#define MUDA_REVERSE_GET_END1(...) MUDA_REVERSE_GET_END2
#define MUDA_REVERSE_GET_END(...) MUDA_REVERSE_GET_END1
#define MUDA_REVERSE_NEXT0(test, next, ...) next MUDA_REVERSE_OUT
#define MUDA_REVERSE_NEXT1(test, next)    \
    MUDA_MACRO_DEFER(MUDA_REVERSE_NEXT0) \
    (test, next, 0)
#define MUDA_REVERSE_NEXT(test, next) MUDA_REVERSE_NEXT1(MUDA_REVERSE_GET_END test, next)

#define MUDA_REVERSE0(x, peek, ...)                            \
    MUDA_MACRO_DEFER(MUDA_REVERSE_NEXT(peek, MUDA_REVERSE1)) \
    (peek, __VA_ARGS__) x,
#define MUDA_REVERSE1(x, peek, ...)                            \
    MUDA_MACRO_DEFER(MUDA_REVERSE_NEXT(peek, MUDA_REVERSE0)) \
    (peek, __VA_ARGS__) x,
#define MUDA_REVERSE2(x, peek, ...)                            \
    MUDA_MACRO_DEFER(MUDA_REVERSE_NEXT(peek, MUDA_REVERSE1)) \
    (peek, __VA_ARGS__) x
#define MUDA_REVERSE(...) MUDA_MACRO_EVAL(MUDA_REVERSE2(__VA_ARGS__, ()()(), ()()(), ()()(), 0))

// macro map
#define MUDA_MAP_END(...)
#define MUDA_MAP_OUT

#define MUDA_MAP_GET_END2() 0, MUDA_MAP_END
#define MUDA_MAP_GET_END1(...) MUDA_MAP_GET_END2
#define MUDA_MAP_GET_END(...) MUDA_MAP_GET_END1
#define MUDA_MAP_NEXT0(test, next, ...) next MUDA_MAP_OUT
#define MUDA_MAP_NEXT1(test, next)    \
    MUDA_MACRO_DEFER(MUDA_MAP_NEXT0) \
    (test, next, 0)
#define MUDA_MAP_NEXT(test, next) MUDA_MAP_NEXT1(MUDA_MAP_GET_END test, next)

#define MUDA_MAP0(f, x, peek, ...) f(x) MUDA_MACRO_DEFER(MUDA_MAP_NEXT(peek, MUDA_MAP1))(f, peek, __VA_ARGS__)
#define MUDA_MAP1(f, x, peek, ...) f(x) MUDA_MACRO_DEFER(MUDA_MAP_NEXT(peek, MUDA_MAP0))(f, peek, __VA_ARGS__)

#define MUDA_MAP_LIST0(f, x, peek, ...) , f(x) MUDA_MACRO_DEFER(MUDA_MAP_NEXT(peek, MUDA_MAP_LIST1))(f, peek, __VA_ARGS__)
#define MUDA_MAP_LIST1(f, x, peek, ...) , f(x) MUDA_MACRO_DEFER(MUDA_MAP_NEXT(peek, MUDA_MAP_LIST0))(f, peek, __VA_ARGS__)
#define MUDA_MAP_LIST2(f, x, peek, ...) f(x) MUDA_MACRO_DEFER(MUDA_MAP_NEXT(peek, MUDA_MAP_LIST1))(f, peek, __VA_ARGS__)

// Applies the function macro `f` to each of the remaining parameters.
#define MUDA_MAP(f, ...) MUDA_MACRO_EVAL(MUDA_MAP1(f, __VA_ARGS__, ()()(), ()()(), ()()(), 0))

// Applies the function macro `f` to each of the remaining parameters and inserts commas between the results.
#define MUDA_MAP_LIST(f, ...) MUDA_MACRO_EVAL(MUDA_MAP_LIST2(f, __VA_ARGS__, ()()(), ()()(), ()()(), 0))

// other useful list operations...
#define MUDA_TAIL_IMPL(x, ...) __VA_ARGS__
#define MUDA_HEAD_IMPL(x, ...) x
#define MUDA_HEAD(...) MUDA_HEAD_IMPL(__VA_ARGS__)
#define MUDA_TAIL(...) MUDA_TAIL_IMPL(__VA_ARGS__)
#define MUDA_LAST(...) MUDA_HEAD(MUDA_REVERSE(__VA_ARGS__))
#define MUDA_POP_LAST(...) MUDA_REVERSE(MUDA_TAIL(MUDA_REVERSE(__VA_ARGS__)))

// inc & dec
#define MUDA_INC_0() 1
#define MUDA_INC_1() 2
#define MUDA_INC_2() 3
#define MUDA_INC_3() 4
#define MUDA_INC_4() 5
#define MUDA_INC_5() 6
#define MUDA_INC_6() 7
#define MUDA_INC_7() 8
#define MUDA_INC_8() 9
#define MUDA_INC_9() 10
#define MUDA_INC_10() 11
#define MUDA_INC_11() 12
#define MUDA_INC_12() 13
#define MUDA_INC_13() 14
#define MUDA_INC_14() 15
#define MUDA_INC_15() 16
#define MUDA_INC_16() 17
#define MUDA_INC_17() 18
#define MUDA_INC_18() 19
#define MUDA_INC_19() 20
#define MUDA_INC_20() 21
#define MUDA_INC_21() 22
#define MUDA_INC_22() 23
#define MUDA_INC_23() 24
#define MUDA_INC_24() 25
#define MUDA_INC_25() 26
#define MUDA_INC_26() 27
#define MUDA_INC_27() 28
#define MUDA_INC_28() 29
#define MUDA_INC_29() 30
#define MUDA_INC_30() 31
#define MUDA_INC_31() 32
#define MUDA_INC_32() 33
#define MUDA_INC_33() 34
#define MUDA_INC_34() 35
#define MUDA_INC_35() 36
#define MUDA_INC_36() 37
#define MUDA_INC_37() 38
#define MUDA_INC_38() 39
#define MUDA_INC_39() 40
#define MUDA_INC_40() 41
#define MUDA_INC_41() 42
#define MUDA_INC_42() 43
#define MUDA_INC_43() 44
#define MUDA_INC_44() 45
#define MUDA_INC_45() 46
#define MUDA_INC_46() 47
#define MUDA_INC_47() 48
#define MUDA_INC_48() 49
#define MUDA_INC_49() 50
#define MUDA_INC_50() 51
#define MUDA_INC_51() 52
#define MUDA_INC_52() 53
#define MUDA_INC_53() 54
#define MUDA_INC_54() 55
#define MUDA_INC_55() 56
#define MUDA_INC_56() 57
#define MUDA_INC_57() 58
#define MUDA_INC_58() 59
#define MUDA_INC_59() 60
#define MUDA_INC_60() 61
#define MUDA_INC_61() 62
#define MUDA_INC_62() 63
#define MUDA_INC_63() 64
#define MUDA_INC_IMPL(x) MUDA_INC_##x()
#define MUDA_INC(x) MUDA_INC_IMPL(x)

#define MUDA_DEC_1() 0
#define MUDA_DEC_2() 1
#define MUDA_DEC_3() 2
#define MUDA_DEC_4() 3
#define MUDA_DEC_5() 4
#define MUDA_DEC_6() 5
#define MUDA_DEC_7() 6
#define MUDA_DEC_8() 7
#define MUDA_DEC_9() 8
#define MUDA_DEC_10() 9
#define MUDA_DEC_11() 10
#define MUDA_DEC_12() 11
#define MUDA_DEC_13() 12
#define MUDA_DEC_14() 13
#define MUDA_DEC_15() 14
#define MUDA_DEC_16() 15
#define MUDA_DEC_17() 16
#define MUDA_DEC_18() 17
#define MUDA_DEC_19() 18
#define MUDA_DEC_20() 19
#define MUDA_DEC_21() 20
#define MUDA_DEC_22() 21
#define MUDA_DEC_23() 22
#define MUDA_DEC_24() 23
#define MUDA_DEC_25() 24
#define MUDA_DEC_26() 25
#define MUDA_DEC_27() 26
#define MUDA_DEC_28() 27
#define MUDA_DEC_29() 28
#define MUDA_DEC_30() 29
#define MUDA_DEC_31() 30
#define MUDA_DEC_32() 31
#define MUDA_DEC_33() 32
#define MUDA_DEC_34() 33
#define MUDA_DEC_35() 34
#define MUDA_DEC_36() 35
#define MUDA_DEC_37() 36
#define MUDA_DEC_38() 37
#define MUDA_DEC_39() 38
#define MUDA_DEC_40() 39
#define MUDA_DEC_41() 40
#define MUDA_DEC_42() 41
#define MUDA_DEC_43() 42
#define MUDA_DEC_44() 43
#define MUDA_DEC_45() 44
#define MUDA_DEC_46() 45
#define MUDA_DEC_47() 46
#define MUDA_DEC_48() 47
#define MUDA_DEC_49() 48
#define MUDA_DEC_50() 49
#define MUDA_DEC_51() 50
#define MUDA_DEC_52() 51
#define MUDA_DEC_53() 52
#define MUDA_DEC_54() 53
#define MUDA_DEC_55() 54
#define MUDA_DEC_56() 55
#define MUDA_DEC_57() 56
#define MUDA_DEC_58() 57
#define MUDA_DEC_59() 58
#define MUDA_DEC_60() 59
#define MUDA_DEC_61() 60
#define MUDA_DEC_62() 61
#define MUDA_DEC_63() 62
#define MUDA_DEC_64() 63
#define MUDA_DEC_IMPL(x) MUDA_DEC_##x()
#define MUDA_DEC(x) MUDA_DEC_IMPL(x)

#define MUDA_RANGE_GEN_1() 0
#define MUDA_RANGE_GEN_2() 0, 1
#define MUDA_RANGE_GEN_3() 0, 1, 2
#define MUDA_RANGE_GEN_4() 0, 1, 2, 3
#define MUDA_RANGE_GEN_5() 0, 1, 2, 3, 4
#define MUDA_RANGE_GEN_6() 0, 1, 2, 3, 4, 5
#define MUDA_RANGE_GEN_7() 0, 1, 2, 3, 4, 5, 6
#define MUDA_RANGE_GEN_8() 0, 1, 2, 3, 4, 5, 6, 7
#define MUDA_RANGE_GEN_9() 0, 1, 2, 3, 4, 5, 6, 7, 8
#define MUDA_RANGE_GEN_10() 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
#define MUDA_RANGE_GEN_11() 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
#define MUDA_RANGE_GEN_12() 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11
#define MUDA_RANGE_GEN_13() 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12
#define MUDA_RANGE_GEN_14() 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13
#define MUDA_RANGE_GEN_15() 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14
#define MUDA_RANGE_GEN_16() 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
#define MUDA_RANGE_GEN_17() 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
#define MUDA_RANGE_GEN_18() 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17
#define MUDA_RANGE_GEN_19() 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18
#define MUDA_RANGE_GEN_20() 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19
#define MUDA_RANGE_GEN_21() 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20
#define MUDA_RANGE_GEN_22() 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21
#define MUDA_RANGE_GEN_23() 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22
#define MUDA_RANGE_GEN_24() 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23
#define MUDA_RANGE_GEN_25() 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24
#define MUDA_RANGE_GEN_26() 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25
#define MUDA_RANGE_GEN_27() 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26
#define MUDA_RANGE_GEN_28() 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27
#define MUDA_RANGE_GEN_29() 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28
#define MUDA_RANGE_GEN_30() 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29
#define MUDA_RANGE_GEN_31() 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30
#define MUDA_RANGE_GEN_32() 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31
#define MUDA_RANGE_GEN_33() 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32
#define MUDA_RANGE_GEN_34() 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33
#define MUDA_RANGE_GEN_35() 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34
#define MUDA_RANGE_GEN_36() 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35
#define MUDA_RANGE_GEN_37() 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36
#define MUDA_RANGE_GEN_38() 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37
#define MUDA_RANGE_GEN_39() 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38
#define MUDA_RANGE_GEN_40() 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39
#define MUDA_RANGE_GEN_41() 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40
#define MUDA_RANGE_GEN_42() 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41
#define MUDA_RANGE_GEN_43() 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42
#define MUDA_RANGE_GEN_44() 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43
#define MUDA_RANGE_GEN_45() 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44
#define MUDA_RANGE_GEN_46() 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45
#define MUDA_RANGE_GEN_47() 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46
#define MUDA_RANGE_GEN_48() 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47
#define MUDA_RANGE_GEN_49() 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48
#define MUDA_RANGE_GEN_50() 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49
#define MUDA_RANGE_GEN_51() 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50
#define MUDA_RANGE_GEN_52() 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51
#define MUDA_RANGE_GEN_53() 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52
#define MUDA_RANGE_GEN_54() 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53
#define MUDA_RANGE_GEN_55() 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54
#define MUDA_RANGE_GEN_56() 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55
#define MUDA_RANGE_GEN_57() 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56
#define MUDA_RANGE_GEN_58() 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57
#define MUDA_RANGE_GEN_59() 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58
#define MUDA_RANGE_GEN_60() 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59
#define MUDA_RANGE_GEN_61() 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60
#define MUDA_RANGE_GEN_62() 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61
#define MUDA_RANGE_GEN_63() 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62
#define MUDA_RANGE_GEN_64() 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63
#define MUDA_RANGE_GEN(N) MUDA_RANGE_GEN_##N()
#define MUDA_RANGE(N) MUDA_RANGE_GEN(N)

#define MUDA_STRINGIFY_IMPL(x) #x
#define MUDA_STRINGIFY(x) MUDA_STRINGIFY_IMPL(x)

#define MUDA_AT_IMPL0(x, ...) x
#define MUDA_AT_IMPL1(x, ...) MUDA_AT_IMPL0(__VA_ARGS__)
#define MUDA_AT_IMPL2(x, ...) MUDA_AT_IMPL1(__VA_ARGS__)
#define MUDA_AT_IMPL3(x, ...) MUDA_AT_IMPL2(__VA_ARGS__)
#define MUDA_AT_IMPL4(x, ...) MUDA_AT_IMPL3(__VA_ARGS__)
#define MUDA_AT_IMPL5(x, ...) MUDA_AT_IMPL4(__VA_ARGS__)
#define MUDA_AT_IMPL6(x, ...) MUDA_AT_IMPL5(__VA_ARGS__)
#define MUDA_AT_IMPL7(x, ...) MUDA_AT_IMPL6(__VA_ARGS__)
#define MUDA_AT_IMPL8(x, ...) MUDA_AT_IMPL7(__VA_ARGS__)
#define MUDA_AT_IMPL9(x, ...) MUDA_AT_IMPL8(__VA_ARGS__)
#define MUDA_AT_IMPL10(x, ...) MUDA_AT_IMPL9(__VA_ARGS__)
#define MUDA_AT_IMPL11(x, ...) MUDA_AT_IMPL10(__VA_ARGS__)
#define MUDA_AT_IMPL12(x, ...) MUDA_AT_IMPL11(__VA_ARGS__)
#define MUDA_AT_IMPL13(x, ...) MUDA_AT_IMPL12(__VA_ARGS__)
#define MUDA_AT_IMPL14(x, ...) MUDA_AT_IMPL13(__VA_ARGS__)
#define MUDA_AT_IMPL15(x, ...) MUDA_AT_IMPL14(__VA_ARGS__)
#define MUDA_AT_IMPL16(x, ...) MUDA_AT_IMPL15(__VA_ARGS__)
#define MUDA_AT_IMPL17(x, ...) MUDA_AT_IMPL16(__VA_ARGS__)
#define MUDA_AT_IMPL18(x, ...) MUDA_AT_IMPL17(__VA_ARGS__)
#define MUDA_AT_IMPL19(x, ...) MUDA_AT_IMPL18(__VA_ARGS__)
#define MUDA_AT_IMPL20(x, ...) MUDA_AT_IMPL19(__VA_ARGS__)
#define MUDA_AT_IMPL21(x, ...) MUDA_AT_IMPL20(__VA_ARGS__)
#define MUDA_AT_IMPL22(x, ...) MUDA_AT_IMPL21(__VA_ARGS__)
#define MUDA_AT_IMPL23(x, ...) MUDA_AT_IMPL22(__VA_ARGS__)
#define MUDA_AT_IMPL24(x, ...) MUDA_AT_IMPL23(__VA_ARGS__)
#define MUDA_AT_IMPL25(x, ...) MUDA_AT_IMPL24(__VA_ARGS__)
#define MUDA_AT_IMPL26(x, ...) MUDA_AT_IMPL25(__VA_ARGS__)
#define MUDA_AT_IMPL27(x, ...) MUDA_AT_IMPL26(__VA_ARGS__)
#define MUDA_AT_IMPL28(x, ...) MUDA_AT_IMPL27(__VA_ARGS__)
#define MUDA_AT_IMPL29(x, ...) MUDA_AT_IMPL28(__VA_ARGS__)
#define MUDA_AT_IMPL30(x, ...) MUDA_AT_IMPL29(__VA_ARGS__)
#define MUDA_AT_IMPL31(x, ...) MUDA_AT_IMPL30(__VA_ARGS__)
#define MUDA_AT_IMPL32(x, ...) MUDA_AT_IMPL31(__VA_ARGS__)
#define MUDA_AT_IMPL33(x, ...) MUDA_AT_IMPL32(__VA_ARGS__)
#define MUDA_AT_IMPL34(x, ...) MUDA_AT_IMPL33(__VA_ARGS__)
#define MUDA_AT_IMPL35(x, ...) MUDA_AT_IMPL34(__VA_ARGS__)
#define MUDA_AT_IMPL36(x, ...) MUDA_AT_IMPL35(__VA_ARGS__)
#define MUDA_AT_IMPL37(x, ...) MUDA_AT_IMPL36(__VA_ARGS__)
#define MUDA_AT_IMPL38(x, ...) MUDA_AT_IMPL37(__VA_ARGS__)
#define MUDA_AT_IMPL39(x, ...) MUDA_AT_IMPL38(__VA_ARGS__)
#define MUDA_AT_IMPL40(x, ...) MUDA_AT_IMPL39(__VA_ARGS__)
#define MUDA_AT_IMPL41(x, ...) MUDA_AT_IMPL40(__VA_ARGS__)
#define MUDA_AT_IMPL42(x, ...) MUDA_AT_IMPL41(__VA_ARGS__)
#define MUDA_AT_IMPL43(x, ...) MUDA_AT_IMPL42(__VA_ARGS__)
#define MUDA_AT_IMPL44(x, ...) MUDA_AT_IMPL43(__VA_ARGS__)
#define MUDA_AT_IMPL45(x, ...) MUDA_AT_IMPL44(__VA_ARGS__)
#define MUDA_AT_IMPL46(x, ...) MUDA_AT_IMPL45(__VA_ARGS__)
#define MUDA_AT_IMPL47(x, ...) MUDA_AT_IMPL46(__VA_ARGS__)
#define MUDA_AT_IMPL48(x, ...) MUDA_AT_IMPL47(__VA_ARGS__)
#define MUDA_AT_IMPL49(x, ...) MUDA_AT_IMPL48(__VA_ARGS__)
#define MUDA_AT_IMPL50(x, ...) MUDA_AT_IMPL49(__VA_ARGS__)
#define MUDA_AT_IMPL51(x, ...) MUDA_AT_IMPL50(__VA_ARGS__)
#define MUDA_AT_IMPL52(x, ...) MUDA_AT_IMPL51(__VA_ARGS__)
#define MUDA_AT_IMPL53(x, ...) MUDA_AT_IMPL52(__VA_ARGS__)
#define MUDA_AT_IMPL54(x, ...) MUDA_AT_IMPL53(__VA_ARGS__)
#define MUDA_AT_IMPL55(x, ...) MUDA_AT_IMPL54(__VA_ARGS__)
#define MUDA_AT_IMPL56(x, ...) MUDA_AT_IMPL55(__VA_ARGS__)
#define MUDA_AT_IMPL57(x, ...) MUDA_AT_IMPL56(__VA_ARGS__)
#define MUDA_AT_IMPL58(x, ...) MUDA_AT_IMPL57(__VA_ARGS__)
#define MUDA_AT_IMPL59(x, ...) MUDA_AT_IMPL58(__VA_ARGS__)
#define MUDA_AT_IMPL60(x, ...) MUDA_AT_IMPL59(__VA_ARGS__)
#define MUDA_AT_IMPL61(x, ...) MUDA_AT_IMPL60(__VA_ARGS__)
#define MUDA_AT_IMPL62(x, ...) MUDA_AT_IMPL61(__VA_ARGS__)
#define MUDA_AT_IMPL63(x, ...) MUDA_AT_IMPL62(__VA_ARGS__)
#define MUDA_AT(index, ...) MUDA_AT_IMPL##index __VA_ARGS__

