//
// $Id$
//

/*!
\file
\ingroup utilities 
\brief compare images to see if they are identical, allowing for small differences

\author Matthew Jacobson
\author Kris Thielemans
\author PARAPET project
\author Charalampos Tsoumpas: Add the tolerance as input
$Date$
$Revision$

This utility compares two images. They are deemed identical if
their maximum absolute relative difference is less than a hard-coded tolerance 
value.
Diagnostic output is written to stdout, and the return value indicates
if the files are identical or not.
*/
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000 - 2009-06-22, Hammersmith Imanet Ltd
    Copyright (C) 2011-12-01 - $Date$, Kris Thielemans
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
/* Modification History:
  
  KT 12/09/2001 added rim_truncation option
*/

#include "stir/DiscretisedDensity.h"
#include "stir/ArrayFunction.h"
#include "stir/recon_array_functions.h"
#include "stir/IO/read_from_file.h"
#include "stir/is_null_ptr.h"
#include <numeric>
#include <stdlib.h>

#ifndef STIR_NO_NAMESPACES
using std::cerr;
using std::cout;
using std::endl;
#endif



//********************** main



USING_NAMESPACE_STIR


int main(int argc, char *argv[])
{
  if(argc<3 || argc>7)
  {
    cerr << "Usage: \n" << argv[0] << "\n\t"
	 << "[-r rimsize] \n\t"
	 << "[-t tolerance] \n\t"
	 << "old_image new_image \n\t"
	 << "'rimsize' has to be a nonnegative integer.\n\t"
	 << "'tolerance' is by default .0005 \n\t"
	 << "When the -r option is used, the (radial) rim of the\n\t"
	 << "images will be set to 0, for 'rimsize' pixels.\n";
    return(EXIT_FAILURE);
  }
  // skip program name
  --argc;
  ++argv;
  int rim_truncation_image = -1;
  float tolerance = .0005F ;

  // first process command line options

  while (argc>0 && argv[0][0]=='-')
    {
      if (strcmp(argv[0], "-r")==0)
	{
	  if (argc<2)
 	    { cerr << "Option '-r' expects a nonnegative (integer) argument\n"; exit(EXIT_FAILURE); }
	      rim_truncation_image = atoi(argv[1]);
	      argc-=2; argv+=2;	    
	} 
      if (strcmp(argv[0], "-t")==0)
	{
	  if (argc<2)
	    { cerr << "Option '-t' expects a (float) argument\n"; exit(EXIT_FAILURE); }
	  tolerance = static_cast<float>(atof(argv[1]));
	  argc-=2; argv+=2;
	}      
    }

  shared_ptr< DiscretisedDensity<3,float> >  
    first_operand(read_from_file<DiscretisedDensity<3,float> >(argv[0]));

  if (is_null_ptr(first_operand))
    { cerr << "Could not read first file\n"; exit(EXIT_FAILURE); }

  shared_ptr< DiscretisedDensity<3,float> >  
    second_operand(read_from_file<DiscretisedDensity<3,float> >(argv[1]));
  if (is_null_ptr(second_operand))
    { cerr << "Could not read 2nd file\n"; exit(EXIT_FAILURE); }

  // check if images are compatible
  {
    string explanation;
    if (!first_operand->has_same_characteristics(*second_operand, 
						 explanation))
      {
	warning("input images do not have the same characteristics.\n%s",
		explanation.c_str());
	return EXIT_FAILURE;
      }
  }

  if (rim_truncation_image>=0)
  {
    truncate_rim(*first_operand, rim_truncation_image);
    truncate_rim(*second_operand, rim_truncation_image);
  }

  float reference_max=first_operand->find_max();
  float reference_min=first_operand->find_min();

  float amplitude=fabs(reference_max)>fabs(reference_min)?
    fabs(reference_max):fabs(reference_min);

  *first_operand -= *second_operand;
  const float max_error=first_operand->find_max();
  const float min_error=first_operand->find_min();
  in_place_abs(*first_operand);
  const float max_abs_error=first_operand->find_max();

  const bool same=(max_abs_error/amplitude<=tolerance);

  cout << "\nMaximum absolute error = "<<max_abs_error
       << "\nMaximum in (1st - 2nd) = "<<max_error
       << "\nMinimum in (1st - 2nd) = "<<min_error<<endl;
  cout <<"Error relative to sup-norm of first image = "<<(max_abs_error/amplitude)*100<<" %"<<endl;

  cout<<"\nImage arrays ";

  if(same)
  {
    cout << (max_abs_error == 0 ? "are " : "deemed ")
         << "identical\n";
  }
  else 
  {
    cout<<"deemed different\n";
  }
  cout << "(tolerance used: " << tolerance*100 << " %)\n\n";
  return same?EXIT_SUCCESS:EXIT_FAILURE;

} //end main
