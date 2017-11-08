/*
    Copyright (C) 2017 University College London

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
/*!
  \file 
  \ingroup test
  \ingroup buildblock
 
  \brief tests for the stir::GeneralisedPoissonNoiseGenerator class

  \author Kris Thielemans
*/

#include "stir/RunTests.h"
#include "stir/Array.h"
#include "stir/GeneralisedPoissonNoiseGenerator.h"
#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics.hpp>
#include <boost/format.hpp>
#include <algorithm>
#include <iostream>

START_NAMESPACE_STIR


/*!
  \brief Tests GeneralisedPoissonNoiseGenerator functionality
  \ingroup test
  Currently contains only simple tests to check mean and variance.
*/
class GeneralisedPoissonNoiseGeneratorTests : public RunTests
{
private:
  void
  run_one_test(const int size, const float mu, const float scaling_factor, const bool preserve_mean);
    
public:
  void run_tests();
};

void
GeneralisedPoissonNoiseGeneratorTests::
run_one_test(const int size, const float mu, const float scaling_factor, const bool preserve_mean)
{
  Array<1,float> input(size);
  Array<1,float> output(size);
  input.fill(mu);

  GeneralisedPoissonNoiseGenerator generator(scaling_factor, preserve_mean);

  generator.generate_random(output, input);
    
  using namespace boost::accumulators;
    
  // The accumulator set which will calculate the properties for us:    
  accumulator_set< float, features< tag::variance, tag::mean > > acc;

  // Use std::for_each to accumulate the statistical properties:
  acc = std::for_each( output.begin(), output.end(), acc );
  set_tolerance(.1);

  const float actual_mean = preserve_mean? mu : mu*scaling_factor;
  const float actual_variance = preserve_mean? mu/scaling_factor : actual_mean;

  boost::format formatter("size %1%, mu %2%, scaling_factor %3%, preserve_mean %4%");
  formatter % size % mu % scaling_factor % preserve_mean;
    
  check_if_equal(mean(acc), actual_mean, "test mean with " + formatter.str());
  check_if_equal(variance(acc), actual_variance, "test variance with " + formatter.str());
}

void
GeneralisedPoissonNoiseGeneratorTests::run_tests()
{

  std::cerr << "Testing GeneralisedPoissonNoiseGenerator\n";

  run_one_test(1000, 100.0F, 1.0F, true);
  run_one_test(1000, 100.0F, 3.0F, true);
  run_one_test(1000, 100.0F, 3.0F, false);

  run_one_test(1000, 4.2F, 1.0F, true);
  run_one_test(1000, 4.2F, 3.0F, true);
  run_one_test(1000, 4.2F, 3.0F, false);
}

END_NAMESPACE_STIR

USING_NAMESPACE_STIR

int main()
{
  GeneralisedPoissonNoiseGeneratorTests tests;
  tests.run_tests();
  return tests.main_return_value();
}
