/*
    Copyright (C) 2000 - 2004, Hammersmith Imanet Ltd
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
  \brief Implements stir::GeneralisedPoissonNoiseGenerator
  \author Kris Thielemans
  \author Sanida Mustafovic
*/

#include "stir/GeneralisedPoissonNoiseGenerator.h"
#include "stir/SegmentByView.h"
#include "stir/Succeeded.h"
#include "stir/round.h"

#include <boost/random/uniform_01.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/variate_generator.hpp>

START_NAMESPACE_STIR

GeneralisedPoissonNoiseGenerator::base_generator_type GeneralisedPoissonNoiseGenerator::generator;

GeneralisedPoissonNoiseGenerator::
GeneralisedPoissonNoiseGenerator(const float scaling_factor,
                                 const bool preserve_mean)
  : scaling_factor(scaling_factor),
    preserve_mean(preserve_mean)
{
  this->seed(43u);
}

void
GeneralisedPoissonNoiseGenerator::
seed(unsigned int value)
{
  if (value==unsigned(0))
   error("Seed value has to be non-zero");
  this->generator.seed(static_cast<poisson_result_type>(value));
}

// function that generates a Poisson noise realisation, i.e. without
// using the scaling_factor
unsigned int
GeneralisedPoissonNoiseGenerator::
generate_poisson_random(const float mu)
{  
  static boost::uniform_01<base_generator_type> random01(generator);
  // normal distribution with mean=0 and sigma=1
  static boost::normal_distribution<double> normal_distrib01(0., 1.);

  // check if mu is large. If so, use the normal distribution
  // note: the threshold must be such that exp(threshold) is still a floating point number
  if (mu > 60.F)
  {
    // get random number of normal distribution of mean=mu and sigma=sqrt(mu)

    // get random with mean=0, sigma=1 and use scaling with sqrt(mu) and addition of mu
    // this has the advantage that we don't have to construct a normal_distrib
    // object every time. This will speed things up, especially because the
    // normal_distribution is implemented using with a polar method that calls
    // generator::operator() twice only on 'odd'- number of invocations
    const double random=normal_distrib01(random01)*sqrt(mu) + mu;

    return static_cast<unsigned>(random<=0 ? 0 : round(random));
  }
  else
  {
    double u = random01();
  
    // prevent problems of n growing too large (or even to infinity) 
    // when u is very close to 1
    if (u>1-1.E-6)
      u = 1-1.E-6;
  
    const double upper = exp(mu)*u;
    double accum = 1.;
    double term = 1.; 
    unsigned int n = 1;
  
    while(accum <upper)
    {
      accum += (term *= mu/n); 
      n++;
    }
    
    return (n - 1);
  }
}

float
GeneralisedPoissonNoiseGenerator::
generate_scaled_poisson_random(const float mu, const float scaling_factor, const bool preserve_mean)
{
  const unsigned int random_poisson = generate_poisson_random(mu*scaling_factor);
  return
    preserve_mean
    ? random_poisson / scaling_factor
    : static_cast<float>(random_poisson);
}

float
GeneralisedPoissonNoiseGenerator::
generate_random(const float mu)
{
  return
    generate_scaled_poisson_random(mu, scaling_factor, preserve_mean);
}


void 
GeneralisedPoissonNoiseGenerator::
generate_random(ProjData& output_projdata, 
                const ProjData& input_projdata)
{  
  for (int seg= input_projdata.get_min_segment_num(); 
       seg<=input_projdata.get_max_segment_num();
       seg++)  
    {
	  for (int timing_pos_num = input_projdata.get_min_tof_pos_num();
			timing_pos_num <= input_projdata.get_max_tof_pos_num();
			++timing_pos_num)
	    {
		  SegmentByView<float> seg_output= 
            output_projdata.get_empty_segment_by_view(seg,false, timing_pos_num);

          this->generate_random(seg_output, input_projdata.get_segment_by_view(seg, timing_pos_num));
          if (output_projdata.set_segment(seg_output) == Succeeded::no)
            error("Problem writing to projection data");
        }
    }
}

END_NAMESPACE_STIR

