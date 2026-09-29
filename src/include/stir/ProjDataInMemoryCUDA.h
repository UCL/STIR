#pragma once

#include "stir/common.h"

#include <cstddef>

START_NAMESPACE_STIR

void AddAssign(float* dst,
               const float* src,
               size_t N);

void SubAssign(float* dst,
               const float* src,
               size_t N);               

void MultAssign(float* dst,
               const float* src,
               size_t N);               

void DivAssign(float* dst,
               const float* src,
               size_t N);  
               
void CUDAxapyb(float *dst, 
               const float *x,
               const float *y,
               float a,
               float b, 
               const size_t N);                

END_NAMESPACE_STIR
