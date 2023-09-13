

#pragma once

#include <cstdint>

void test_asum(uint32_t power);

// template <typename T>
void test_asum0(const float *in, float *out, std::size_t elems);
void test_asum1(const float *in, float *out, std::size_t elems);
void test_asum3(const float *in, float *out, std::size_t elems);
void test_asum8(const float *in, float *out, std::size_t elems);
