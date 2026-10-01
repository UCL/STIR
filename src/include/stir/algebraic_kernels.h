#pragma once

#include "stir/common.h"

#include <cstddef>

START_NAMESPACE_STIR

void add_assign(float* dst,
               const float* src,
               size_t N);

void sub_assign(float* dst,
               const float* src,
               size_t N);               

void mult_assign(float* dst,
               const float* src,
               size_t N);               

void div_assign(float* dst,
               const float* src,
               size_t N);  
               
void CUDAxapyb(float *dst, 
               const float *x,
               const float *y,
               float a,
               float b, 
               const size_t N);                

END_NAMESPACE_STIR
