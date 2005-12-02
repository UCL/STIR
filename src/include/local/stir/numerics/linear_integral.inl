//
// $Id$
//
/*
    Copyright (C) 2004 - $Date$, Hammersmith Imanet Ltd
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

  \file
  \ingroup numerics

  \brief Implementations of inline functions of linear_integral

  \author Charalampos Tsoumpas

  $Date$
  $Revision$
*/

START_NAMESPACE_STIR

float 
linear_integral(std::vector<float> f , std::vector<float> t , int approx)
{
	std::vector<float>::const_iterator cur_iter_f=f.begin();
 	std::vector<float>::const_iterator cur_iter_t=t.begin();
	const unsigned int f_size=f.size();
	const unsigned int t_size=t.size();
	float integral_result=0;
	assert(f_size==t_size);
	if(f_size!=t_size) 
	warning("The linear integral requires equal size of the two input vectors!!!");

	const unsigned int imax=f_size;
	if (approx==0)
	  //Rectangular Formula:
	  // If not at the borders apply: (t_next-t_previous)*0.5*f
	  // If at the borders apply: (t2-t1)*0.5*f, (tN-TN_previous)*0.5*f
	  { 
	    integral_result=f[0]*(t[1]-t[0])*0.5F;
	    for (unsigned int i=1;i<imax-1;++i)
 	      integral_result += f[i]*(t[i+1]-t[i-1])*0.5F;
	    integral_result += f[f_size-1]*(t[t_size-1]-t[t_size-2])*0.5F;
	  }
	if (approx==1)//trapezoidal
	  // Simply apply the formula: (f_next+f)*(t_next-t)*0.5
	  {	
	    for (unsigned int i=0 ; i<imax-1 ; ++i)
	      integral_result += (f[i]+f[i+1])*(t[i+1]-t[i])*0.5F;
	  }
	return integral_result;
}
float
linear_integral(std::vector<float> f , std::vector<float> t)
{
return 
  linear_integral(f , t, 1);
}

END_NAMESPACE_STIR
