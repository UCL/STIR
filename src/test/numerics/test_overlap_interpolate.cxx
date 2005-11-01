// $Id$
/*
    Copyright (C) 2005- $Date$, Hammersmith Imanet Ltd
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
  \ingroup numerics_test

  \brief Test program for stir::overlap_interpolate

  \author Kris Thielemans
  $Date$
  $Revision$
*/
#include "stir/Array.h"
#include "stir/make_array.h"
#include "stir/RunTests.h"
#include "stir/numerics/overlap_interpolate.h"
#include "stir/stream.h"
#include "stir/Succeeded.h"
#include "boost/lambda/lambda.hpp"
#include <iostream>
#include <algorithm>
#include <sstream>

START_NAMESPACE_STIR


/*!
  \brief Test class for stir::overlap_interpolate
  \ingroup numerics_test

*/
class overlap_interpolateTests : public RunTests
{
public:
  void run_tests();
private:
#ifndef STIR_OVERLAP_NORMALISATION
  template <class ValueT, class BoundaryT>
  static 
  void
  divide_by_out_size(
		     ValueT& outvalues,
		     const BoundaryT& outboundaries)
  {
    typename ValueT::iterator iter = outvalues.begin();
    typename BoundaryT::const_iterator bound_iter = outboundaries.begin();
    for (; iter != outvalues.end(); ++iter, ++bound_iter)
      {
	*iter /= (*(bound_iter+1)) - (*bound_iter);
      }
  }
#endif

  template <class ValueT, class BoundaryT>
  Succeeded
  test_case(const ValueT& invalues,
	    const BoundaryT& inboundaries,
	    const ValueT& outvalues,
	    const BoundaryT& outboundaries,
	    const char * const description)
  {
    ValueT my_outvalues = outvalues;
    overlap_interpolate(my_outvalues.begin(), my_outvalues.end(),
			outboundaries.begin(), outboundaries.end(),
			invalues.begin(), invalues.end(),
			inboundaries.begin(), inboundaries.end());
#ifndef STIR_OVERLAP_NORMALISATION
    divide_by_out_size(my_outvalues, outboundaries);
#endif

    const bool ret = check_if_equal(my_outvalues, outvalues,description);
    if (!ret)
      std::cerr << "\nres: " << my_outvalues << "should be " << outvalues;
    return 
      ret  ? Succeeded::yes : Succeeded::no;
  }

  template <class ValueT>
  Succeeded
  uniform_test_case(const ValueT& invalues,
		    const float zoom, const float offset,
		    const ValueT& outvalues,
		    const char * const description)
  {
    using namespace boost::lambda;

    ValueT my_outvalues = outvalues;
    overlap_interpolate(my_outvalues,
			invalues, zoom, offset);
#ifndef STIR_OVERLAP_NORMALISATION
    std::for_each(my_outvalues.begin(), my_outvalues.end(), _1 *= zoom);
#endif
    const bool ret = check_if_equal(my_outvalues, outvalues,description);
    if (!ret)
      std::cerr << "\nres: " << my_outvalues << "should be " << outvalues << '\n';
    return 
      ret  ? Succeeded::yes : Succeeded::no;
  }

  Succeeded
  test_case_1d(const char * const input, 
	       const char * const description)
  {
    std::istringstream s(input);
    VectorWithOffset<float> invalues, inboundaries, outvalues, outboundaries;
    s >> invalues >> inboundaries >> outvalues >> outboundaries;
    return test_case(invalues, inboundaries, outvalues, outboundaries, description);
  }


  Succeeded
  uniform_test_case_1d(const char * const input, 
		       const char * const description)
  {
    std::istringstream s(input);
    VectorWithOffset<float> invalues, outvalues;
    int instartindex, outstartindex, outendindex;
    float zoom, offset;
    char out[10000];

    s >> invalues 
      >> instartindex >> outstartindex >> outendindex
      >> zoom >> offset
      >> outvalues;
    invalues.set_min_index(instartindex);
    VectorWithOffset<float> inboundaries(invalues.get_min_index(), invalues.get_max_index()+1);
    for (int i=inboundaries.get_min_index(); i<=inboundaries.get_max_index(); ++i)
	  inboundaries[i]=i-.5F;
    outvalues.set_min_index(outstartindex);
    sprintf(out, "%s: inconsistent index sizes. Check test program", description);
    if (!check_if_equal(outvalues.get_max_index(), outendindex-1, out))
	return Succeeded::no;
    VectorWithOffset<float> outboundaries(outstartindex, outendindex);
    for (int i=outboundaries.get_min_index(); i<=outboundaries.get_max_index(); ++i)
      outboundaries[i]=(i-.5F)/zoom+offset;

    sprintf(out, "%s: test general overlap_interpolate", description);
    if (test_case(invalues, inboundaries, outvalues, outboundaries, out) == Succeeded::no)
      return Succeeded::no;

    sprintf(out, "%s: test uniform overlap_interpolate", description);
    if (uniform_test_case(invalues, zoom, offset, outvalues, out) == Succeeded::no)
      return Succeeded::no;
    return Succeeded::yes;
  }

};

void
overlap_interpolateTests::run_tests()
{
  std::cerr << "Tests for overlap_interpolate\n"
	    << "Everythings is fine if the program runs without any output." << std::endl;
  
  set_tolerance(.0005);
  test_case_1d("{1,2.1,-3,4}"
	       "{2,3,4,5,6}"
	       "{1,2.1,-3,4}"
	       "{2,3,4,5,6}",
	       "equal boundaries");

  test_case_1d("{1.1,2,3,4.5}"
	       "{2,3,4,5,7}"
	       "{0,1.1,2,3,4.5}"
	       "{-1,2,3,4,5,7}",
	       "equal boundaries, but longer at start");

  test_case_1d("{1.1,2,3,4.5}"
	       "{2,3,4,5,7}"
	       "{1.1,2,3,4.5,0}"
	       "{2,3,4,5,7,10}",
	       "equal boundaries, but longer at end");

  test_case_1d("{1.1,2,3,4.5}"
	       "{2,3,4,5,7}"
	       "{0,1.1,1.1,2,2,3,3,4.5,4.5,4.5,4.5}"
	       "{-1,2,2.5,3,3.3,4,4.8,5,5.1,5.2,5.6,7}",
	       "multiple out boxes in each in box");

  test_case_1d("{1,5,2,16,7,4,2,4}"
	       "{-2,5,6,6.5,7,86,101,110,130}"
	       "{0.,0.2,1.,1.,1.,5.8,7.,7.,7.,6.53448,1.28421,0.,0.}"
	       "{-30,-4,-1.5,1,1.5,4,9,13,18,37,95,190,210,250}",
	       "arbitrary sizes case 1");
  test_case_1d("{-1,2.5,2,1.6,7,-4,4}"
	       "{-2.1,5.2,6.3,6.5,7.6,7.8,11,11.7}"
	       "{-0.516667,-0.0416667,1.6,1.55556}"
	       "{-5,1,7,7.1,8}",
	       "arbitrary sizes case 2");
  test_case_1d("{-1,2.56,2,11.6,7.3,-4,4}"
	       "{-2.1,5.2,6.3,6.5,7.6,7.8,11,11.7}"
	       "{-0.0333333,-1.,-1.,-1.,-0.4304,4.10182,9.58333,-4.,-4.,-2.,0.,0.}"
	       "{-5,-2,0,1,3.1,5.6,6.7,7.9,8.3,8.9,11.7,13,15}",
	       "arbitrary sizes case 3");
  test_case_1d("{0.594221,0.932554,0.930552,0.479434,0.44984,0.426074,0.574378,\
0.893291}"
	       "{0,0.0253037,0.889616,1.52744,2.47386,3.11161,3.56033,4.02598,4.94703}"
	       "{0.923744,0.930552,0.509265,0.457499,0.44984}"
	       "{0,0.995915,1.46923,2.34955,2.82987,2.96136}",
	       "random case 1");
  test_case_1d("{0.43649,0.236381,0.978282,0.537264,0.503936,0.305829,0.498848,\
0.0874235}"
	       "{0,0.0778616,0.809312,1.41487,1.47699,1.69054,1.78416,2.4433,2.86767}"
	       "{0.256752,0.925738,0.493403,0.387417,0.0874235,0.00196432,0.,0.,0.}"
	       "{0,0.764837,1.3928,2.13089,2.55935,2.85087,3.59851,3.85628,4.15325,5.00828}",\

	       "random case 2");
  test_case_1d("{0.511261,0.27949,0.759702,0.351098,0.205432,0.780642,0.672278,\
0.273237}"
	       "{0,0.473981,0.649066,1.25922,1.31891,1.69927,2.21521,2.40101,2.69586}"
	       "{0.491445,0.577761,0.641911,0.672278,0.672278,0.29108,0.}"
	       "{0,0.75239,1.53024,2.28759,2.29091,2.29566,2.81574,3.27612}",
	       "random case 3");
  test_case_1d("{0.148291,0.493487,0.240592,0.700675,0.797193,0.288055,0.459951,\
0.0283963,0.523956}"
	       "{0,0.814074,1.09894,1.51718,1.98145,2.41516,3.18408,3.41653,3.58595,4.26728}\
"
	       "{0.209939,0.298013,0.559381,0.74738,0.40299,0.288055,0.288055}"
	       "{0,0.991066,1.46617,1.63226,2.30884,2.77983,2.79455,2.81236}",
	       "random case 4");
  test_case_1d("{0.183092,0.230392,0.314051,0.220611,0.895037}"
	       "{0,0.770442,1.0561,1.75275,1.83371,2.31929}"
	       "{0.18838,0.246126,0.314051,0.633862,0.}"
	       "{0,0.867418,1.0998,1.74705,2.4637,3.09868}",
	       "random case 5");
  test_case_1d("{0.0629649,0.965917,0.72559,0.15987,0.89687,0.289338,0.254605,0.\
145144}"
	       "{0,0.879067,0.985312,1.00953,1.84062,2.49907,2.71028,2.96405,3.50949}"
	       "{0.140638,0.410539,0.178698,0.861756,0.217232,0.0726656,0.,0.}"
	       "{0,0.961802,1.09205,1.86024,2.53826,3.26768,3.75068,3.80222,3.84527}",
	       "random case 6");
  test_case_1d("{0.666452,0.517083,0.32595,0.883177,0.769582,0.227745,0.0713446,\
0.738033}"
	       "{0,0.890515,1.01201,1.05915,1.96609,2.19815,3.10844,3.9018,4.2633}"
	       "{0.666452,0.626473,0.687369}"
	       "{0,0.270258,1.0503,1.07547}",
	       "random case 7");
  test_case_1d("{0.683484,0.540841,0.297046,0.973624,0.640437,0.874389,0.779963,\
0.647674}"
	       "{0,0.75726,0.862067,1.41428,1.99061,2.00984,2.22413,2.65485,3.18405}"
	       "{0.683484,0.57681,0.557915,0.902686,0.701251,0.075033,0.,0.,0.}"
	       "{0,0.112286,1.09452,1.61495,2.35078,3.10157,3.81354,4.55392,5.2646,5.3319}",
	       "random case 8");
  uniform_test_case_1d("{1,5,2,16,7,4,2,4}"
		       "-2 -2 6 1 0"
		       "{1.,5.,2.,16.,7.,4.,2.,4.}",
		       "uniform equal in and out");
  uniform_test_case_1d("{1,5,2,16,7,4,2,4}"
		       "-2 -2 3 1 0"
		       "{1.,5.,2.,16.,7.}",
		       "uniform equal in and out, but short on right");
  uniform_test_case_1d("{1,5,2,16,7,4,2,4}"
		       "-2 0 6 1 0"
		       "{2.,16.,7.,4.,2.,4.}",
		       "uniform equal in and out, but short on left");
  uniform_test_case_1d("{1,5,2,16,7,4,2,4}"
		       "-2 0 10 1 0"
		       "{2.,16.,7.,4.,2.,4.,0.,0.,0.,0.}",
		       "uniform equal in and out, but short on left and long on right");
  uniform_test_case_1d("{1,5,2,16,7,4,2,4}"
		       "-1 -3 5 1 0"
		       "{0.,0.,1.,5.,2.,16.,7.,4.}",
		       "uniform equal in and out, but long on left and short on right");
  uniform_test_case_1d("{1,5,2,16,7,4,2,4}"
		       "-2 0 10 1 -2"
		       "{1.,5.,2.,16.,7.,4.,2.,4.,0.,0.}",
		       "uniform equal in and out, offset only");
  uniform_test_case_1d("{1,5,2,16,7,4,2,4}"
		       "-2 -1 10 0.3 -1"
		       "{0.,3.2,7.6,1.5,0.,0.,0.,0.,0.,0.,0.}",
		       "uniform test case 1");
  uniform_test_case_1d("{1,5,2,16,7,4,2,4}"
		       "2 1 10 0.3 1.3"
		       "{7.88,3.42,0.,0.,0.,0.,0.,0.,0.}",
		       "uniform test case 2");
  uniform_test_case_1d("{1,5,2,16,7,4,2,4}"
		       "-2 -3 10 2.5 -2.2"
		       "{0.,0.,0.25,1.,1.,4.,5.,4.25,2.,2.,12.5,16.,13.75}",
		       "uniform test case 3");
}


END_NAMESPACE_STIR

USING_NAMESPACE_STIR

int main()
{
  overlap_interpolateTests tests;
  tests.run_tests();
  return tests.main_return_value();
}
