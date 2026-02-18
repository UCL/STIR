/*
    Copyright (C) 2017 University College London

    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

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
#include "stir/format.h"
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
  void run_one_test(const int size, const float mu, const float scaling_factor, const bool preserve_mean);

public:
  void run_tests() override;
};

void
GeneralisedPoissonNoiseGeneratorTests::run_one_test(const int size,
                                                    const float mu,
                                                    const float scaling_factor,
                                                    const bool preserve_mean)
{
  Array<1, float> input(size);
  Array<1, float> output(size);
  input.fill(mu);

  GeneralisedPoissonNoiseGenerator generator(scaling_factor, preserve_mean);

  generator.generate_random(output, input);

  using namespace boost::accumulators;

  // The accumulator set which will calculate the properties for us:
  accumulator_set<float, features<tag::variance, tag::mean>> acc;

  // Use std::for_each to accumulate the statistical properties:
  acc = std::for_each(output.begin(), output.end(), acc);
  set_tolerance(.1);

  const float actual_mean = preserve_mean ? mu : mu * scaling_factor;
  const float actual_variance = preserve_mean ? mu / scaling_factor : actual_mean;

  std::string formatted = format("size {}, mu {}, scaling_factor {}, preserve_mean {}", size, mu, scaling_factor, preserve_mean);
  check_if_equal(mean(acc), actual_mean, "test mean with " + formatted);
  check_if_equal(variance(acc), actual_variance, "test variance with " + formatted);
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

int
main()
{
  GeneralisedPoissonNoiseGeneratorTests tests;
  tests.run_tests();
  return tests.main_return_value();
}
