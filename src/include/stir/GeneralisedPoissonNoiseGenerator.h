/*
    Copyright (C) 2017, University College London
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
  \ingroup buildblock 
  \brief Declares stir::GeneralisedPoissonNoiseGenerator
  \author Kris Thielemans
*/

#include "stir/ProjData.h"

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/variate_generator.hpp>
#include <algorithm>
#include <boost/bind.hpp>

START_NAMESPACE_STIR


/*!
  \ingroup buildblock
  \brief Generates noise realisations according to Poisson statistics but allowing for scaling

  A \c scaling_factor is used to multiply the input data before generating
  the Poisson random number. This means that a \c scaling_factor larger than 1
  will result in less noisy data.

  If \c preserve_mean=\c false,, the mean of the output data will
  be equal to <tt>scaling_factor*mean_of_input</tt>, otherwise it
  will be equal to mean_of_input, but then the output is no longer Poisson
  distributed.
*/
class GeneralisedPoissonNoiseGenerator
{
  // try boost::mt19937 or boost::ecuyer1988 instead of boost::minstd_rand
  typedef boost::mt19937 base_generator_type;
  typedef base_generator_type::result_type poisson_result_type;

 public:
  //! Constructor intialises the andom number generator with a fixed seed
  GeneralisedPoissonNoiseGenerator(const float scaling_factor = 1.0F, const bool preserve_mean = false);

  //! The seed value for the random number generator
  void seed(unsigned int);

  //! generate a random number according to a distribution with mean mu
  float generate_random(const float mu);
     
  template <int num_dimensions, class elemTout, class elemTin>
    void generate_random(Array<num_dimensions, elemTout>& array_out,
                         const Array<num_dimensions, elemTin>& array_in)
    {
      std::transform(array_in.begin_all(), array_in.end_all(),
                     array_out.begin_all(),
                     boost::bind(generate_scaled_poisson_random, _1, this->scaling_factor, this->preserve_mean));
    }

  void
    generate_random(ProjData& output_projdata, 
                    const ProjData& input_projdata);

 private:
  static base_generator_type generator;
  const float scaling_factor;
  const bool preserve_mean;

  static unsigned int generate_poisson_random(const float mu);
  static float generate_scaled_poisson_random(const float mu, const float scaling_factor, const bool preserve_mean);

};

END_NAMESPACE_STIR

