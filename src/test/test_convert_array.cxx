//
//
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- 2008, Hammersmith Imanet Ltd
    Copyright (C) 2025 University College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0 AND License-ref-PARAPET-license

    See STIR/LICENSE.txt for details
*/
/*!
  \file
  \ingroup test
  \brief tests for the stir::convert_array functions

  \author Kris Thielemans
  \author PARAPET project

*/

#include "stir/Array.h"
#include "stir/convert_array.h"
#include "stir/NumericInfo.h"
#include "stir/IndexRange3D.h"
#include "stir/RunTests.h"
#include "stir/stream.h"
#include <vector>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <type_traits>
using std::cerr;
using std::endl;

START_NAMESPACE_STIR

//! tests  convert_array functionality
template <typename indexT>
class convert_array_Tests : public RunTests
{
public:
  void run_tests() override;
};

template <typename indexT>
void
convert_array_Tests<indexT>::run_tests()
{

  // 1D
  {
    Array<1, float, indexT> tf1(1, 20);
    tf1.fill(100.F);

    Array<1, short, indexT> ti1(1, 20);
    ti1.fill(100);

    {
      // float -> short with a preferred scale factor
      float scale_factor = float(1);
      Array<1, short, indexT> ti2 = convert_array(scale_factor, tf1, NumericInfo<short>());

      check(scale_factor == float(1), "test convert_array float->short 1D");
      check_if_equal(ti1, ti2, "test convert_array float->short 1D");
    }

    {
      // float -> short with automatic scale factor
      float scale_factor = 0;
      Array<1, short, indexT> ti2 = convert_array(scale_factor, tf1, NumericInfo<short>());

      check(fabs(NumericInfo<short>().max_value() / 1.01 / ti2[1] - 1) < 1E-4);
      for (indexT i = 1; i <= 20; i++)
        ti2[i] = short(double(ti2[i]) * scale_factor);
      check(ti1 == ti2);
    }

    tf1 *= 1E20F;
    {
      // float -> short with a preferred scale factor that needs to be adjusted
      float scale_factor = 1;
      Array<1, short, indexT> ti2 = convert_array(scale_factor, tf1, NumericInfo<short>());

      check(fabs(NumericInfo<short>().max_value() / 1.01 / ti2[1] - 1) < 1E-4);
      for (indexT i = 1; i <= 20; i++)
        check(fabs(double(ti2[i]) * scale_factor / tf1[i] - 1) < 1E-4);
    }

    {
      // short -> float with a scale factor = 1
      float scale_factor = 1;
      Array<1, float, indexT> tf2 = convert_array(scale_factor, ti1, NumericInfo<float>());
      Array<1, short, indexT> ti2(1, 20);

      check(scale_factor == float(1));
      check(tf2[1] == 100.F);
      for (indexT i = 1; i <= 20; i++)
        ti2[i] = short(double(tf2[i]) * scale_factor);
      check(ti1 == ti2);
    }

    {
      // short -> float with a preferred scale factor = .01
      float scale_factor = .01F;
      Array<1, float, indexT> tf2 = convert_array(scale_factor, ti1, NumericInfo<float>());
      Array<1, short, indexT> ti2(1, 20);

      check(scale_factor == float(.01));
      // TODO double->short
      for (indexT i = 1; i <= 20; i++)
        ti2[i] = short(double(tf2[i]) * scale_factor + 0.5);
      check(ti1 == ti2);
    }

    tf1.fill(-3.2F);
    ti1.fill(-3);
    {
      // positive float -> unsigned short with a preferred scale factor
      float scale_factor = 1;
      Array<1, short, indexT> ti2 = convert_array(scale_factor, tf1, NumericInfo<short>());

      check(scale_factor == float(1));
      check(ti1 == ti2);
    }

    {
      Array<1, unsigned short, indexT> ti3(1, 20);
      ti3.fill(0);

      // negative float -> unsigned short with a preferred scale factor
      float scale_factor = 1;
      Array<1, unsigned short, indexT> ti2 = convert_array(scale_factor, tf1, NumericInfo<unsigned short>());

      check(scale_factor == float(1));
      check(ti3 == ti2);
    }
  }
  //   3D

  {
    const auto min_indices = make_coordinate<indexT>(1, 1, std::is_signed_v<indexT> ? -2 : 0);
    Array<3, float, indexT> tf1(IndexRange<3, indexT>(min_indices, make_coordinate<indexT>(30, 182, 182)));
    tf1.fill(100.F);

    Array<3, short, indexT> ti1(tf1.get_index_range());
    ti1.fill(100);

    {
      // float -> short with a preferred scale factor
      float scale_factor = float(1);
      Array<3, short, indexT> ti2 = convert_array(scale_factor, tf1, NumericInfo<short>());

      check(scale_factor == float(1));
      check(ti1 == ti2);
    }

    {
      // float -> short with automatic scale factor
      float scale_factor = 0;
      Array<3, short, indexT> ti2 = convert_array(scale_factor, tf1, NumericInfo<short>());
#ifndef DO_TIMING_ONLY
      check(fabs(NumericInfo<short>().max_value() / 1.01 / (*ti2.begin_all()) - 1) < 1E-4);
      const auto iter_end = ti2.end_all();
      for (auto iter = ti2.begin_all(); iter != iter_end; ++iter)
        *iter = short(double((*iter)) * scale_factor);
      check(ti1 == ti2);
#endif
    }

    tf1 *= 1E20F;
    {
      // float -> short with a preferred scale factor that needs to be adjusted
      float scale_factor = 1;
      Array<3, short, indexT> ti2 = convert_array(scale_factor, tf1, NumericInfo<short>());

#ifndef DO_TIMING_ONLY
      check(fabs(NumericInfo<short>().max_value() / 1.01 / (*ti2.begin_all()) - 1) < 1E-4);
      auto iter_ti2 = ti2.begin_all();
      const auto iter_ti2_end = ti2.end_all();
      auto iter_tf1 = tf1.begin_all();
      for (; iter_ti2 != iter_ti2_end; ++iter_ti2, ++iter_tf1)
        check(fabs(double(*iter_ti2) * scale_factor / *iter_tf1 - 1) < 1E-4);
#endif
    }
  }
  // tests on convert_range
  {
    std::vector<signed char> vin(10, 2);
    std::vector<int> vout(10);
    float scale_factor = 0;
    convert_range(vout.begin(), scale_factor, vin.begin(), vin.end());
    {
      std::vector<int>::const_iterator iter_out = vout.begin();
      std::vector<signed char>::const_iterator iter_in = vin.begin();
      for (; iter_out != vout.end(); ++iter_in, ++iter_out)
        check(fabs(double(*iter_out) * scale_factor / *iter_in - 1) < 1E-4, "convert_range signed char->int");
    }
  }
  // equal type
  {
    std::vector<int> vin(10, 2);
    std::vector<int> vout(10);
    float scale_factor = 3;
    convert_range(vout.begin(), scale_factor, vin.begin(), vin.end());
    {
      check_if_equal(scale_factor, 1.F, "scale_factor should be 1 when using equal types");
      std::vector<int>::const_iterator iter_out = vout.begin();
      std::vector<int>::const_iterator iter_in = vin.begin();
      for (; iter_out != vout.end(); ++iter_in, ++iter_out)
        check(fabs(double(*iter_out) * scale_factor / *iter_in - 1) < 1E-4, "convert_range equal types");
    }
  }
}

END_NAMESPACE_STIR

USING_NAMESPACE_STIR

int
main()
{
  {
    cerr << "Testing convert_array with int as indices\n";
    cerr << "=========================================\n";
    convert_array_Tests<int> tests;
    tests.run_tests();
    if (!tests.is_everything_ok())
      return tests.main_return_value();
  }
  {
    cerr << "Testing convert_array with short int as indices\n";
    cerr << "=========================================\n";
    convert_array_Tests<short int> tests;
    tests.run_tests();
    if (!tests.is_everything_ok())
      return tests.main_return_value();
  }
  {
    cerr << "Testing convert_array with long long as indices\n";
    cerr << "=========================================\n";
    convert_array_Tests<long long> tests;
    tests.run_tests();
    if (!tests.is_everything_ok())
      return tests.main_return_value();
  }
  {
    cerr << "Testing convert_array with unsigned int as indices\n";
    cerr << "=========================================\n";
    convert_array_Tests<unsigned int> tests;
    tests.run_tests();
    return tests.main_return_value();
  }
}
