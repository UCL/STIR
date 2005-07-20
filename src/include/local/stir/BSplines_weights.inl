//
// $Id$
//
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
  \ingroup numerics_buildblock
  \brief Implementation of the (cubic) B-Splines Interpolation 

  \author Charalampos Tsoumpas
  \author Kris Thielemans

  $Date$
  $Revision$
*/

using namespace std;

START_NAMESPACE_STIR
namespace BSpline {

template <typename pos_type>
pos_type 
BSplines_weight(const pos_type relative_position) 
{	
	const pos_type abs_relative_position = fabs(relative_position);
	assert(abs_relative_position>=0);
	if (abs_relative_position<1)		
	  return 2./3. + (0.5*abs_relative_position-1)*abs_relative_position*abs_relative_position;
	if (abs_relative_position>=2)
	  return 0;
	const pos_type tmp=2-abs_relative_position;
	return tmp*tmp*tmp/6;
	
}

template <typename pos_type>
pos_type 
BSplines_1st_der_weight(const pos_type relative_position) 
{
	const pos_type abs_relative_position = fabs(relative_position);
	if (abs_relative_position>=2)
		return 0;
	int sign = relative_position>0?1:-1;
	if (abs_relative_position>=1)		
		return -0.5*sign*(abs_relative_position-2)*(abs_relative_position-2);
	return sign*abs_relative_position*(1.5*abs_relative_position-2.);	
}

template <typename pos_type>
pos_type 
BSplines_weights(const pos_type relative_position, const BSplineType spline_type) 
{
//	double relative_position = rel_position;
	switch(spline_type)
	{
	case cubic:
		return BSplines_weight(relative_position);	
	case near_n:
	{		
		if (fabs(relative_position)<0.5)
			return 1;		
		if (fabs(relative_position)==0.5)
			return 0.5;		
		else return 0;
	}
	case linear:
		return 	std::max(0.,-1 + relative_position) - 
		      2*std::max(0.,relative_position) + 
			  std::max(0.,1 + relative_position);
	case quadratic:
		return 	-pow(std::max(0.,-1.5 + relative_position),2)/2. + 
		(3*pow(std::max(0.,-0.5 + relative_position),2))/2. - 
		(3*pow(std::max(0.,0.5 + relative_position),2))/2. + 
		pow(std::max(0.,1.5 + relative_position),2)/2.;		
	case quintic:
		return 	
		(pow(std::max(0.,-3 + relative_position),5) -  6*pow(std::max(0.,-2 + relative_position),5) +
	      15*pow(std::max(0.,-1 + relative_position),5) - 20*pow(std::max(0.,relative_position),5) + 
		  15*pow(std::max(0.,1 + relative_position),5) -   6*pow(std::max(0.,2 + relative_position),5) + 
		     pow(std::max(0.,3 + relative_position),5))/ 120. ;		
	case oMoms:
		return oMoms_weight(relative_position);	
	default:
		cerr << "Not implemented b-spline type" << endl;
		return -1000000;
	}
}

template <typename pos_type>
pos_type 
oMoms_weight(const pos_type relative_position) 
{
	const pos_type abs_relative_position = fabs(relative_position);
	assert(abs_relative_position>=0);
	if (abs_relative_position>=2)
		return 0;
	if (abs_relative_position>=1)		
		return 29./21. + abs_relative_position*(-85./42. + 
		                 abs_relative_position*(1.-abs_relative_position/6)) ;		
	else 
		return 13./21. + abs_relative_position*(1./14. + 
		                 abs_relative_position*(0.5*abs_relative_position-1.)) ;
}

} // end BSpline namespace

END_NAMESPACE_STIR

