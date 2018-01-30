/*
  Copyright (C) 2016, UCL
  This file is part of STIR.
  
  This file is free software; you can redistribute it and/or modify
  it under the terms of the GNU Lesser General Public License as published by
  the Free Software Foundation; either version 2.1 of the License, or
  (at your option) any later version.
  
  This file is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU Lesser General Public License for more details.
  
  See STIR/LICENSE.txt for details
*/

#ifndef __stir_STIR_MATH_H__
#define __stir_STIR_MATH_H__

#ifndef STIR_NO_NAMESPACES
using std::cerr;
using std::cout;
using std::endl;
using std::fstream;
using std::unary_function;
using std::transform;
using std::max;
using std::min;
using std::string;
using std::vector;
#endif

USING_NAMESPACE_STIR

// Nikos Efthimiou: Minimal header file to be able to use this class from other
// places in the code.

// a function object that takes a power of a float, and then multiplies with a float, and finally adds a float
class pow_times_add: public unary_function<float,float>
{
public:
  pow_times_add(const float add_scalar, const float mult_scalar, const float power,
        const float min_threshold, const float max_threshold)
    : add(add_scalar), mult(mult_scalar), power(power),
      min_threshold(min_threshold), max_threshold(max_threshold)
  {}

  float operator()(float const arg) const
  {
    const float value = min(max(arg, min_threshold), max_threshold);
    return add+mult*(power==1?value : pow(value,power));
  }
private:
  const float add;
  const float mult;
  const float power;
  const float min_threshold;
  const float max_threshold;
};

#endif

