//
// $Id$
//
/*!

  \file
  \ingroup test

  \brief Interactive test program for overlap_interpolate

  \author Kris Thielemans
  \author PARAPET project

  $Date$
  $Revision$
*/
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- $Date$, Hammersmith Imanet Ltd
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

#include "stir/numerics/overlap_interpolate.h"
#include "stir/Array.h"
#include "stir/IndexRange2D.h"
#include "stir/utilities.h"
#include "stir/stream.h"

#ifndef STIR_NO_NAMESPACES
using std::cout;
using std::endl;
#endif

USING_NAMESPACE_STIR

int 
main(int argc, char *argv[])
{
  if (ask ("1D", false))
  {   
    typedef Array<1,float> array; 
    
    array in(-9,9);
    // initialise with some data
    for (int i=-9; i<=9; i++)
    in[i] = .5*square(i-6)+3.*i-30;
    
    
    do
    {
      const float zoom = 
	ask_num("Zoom ", 0., 100., 1.);
      const float offset = 
	ask_num("Offset ", -100., 100., 0.);
      const float min2 = (in.get_min_index()-.5 - offset)*zoom;
      const float max2 = (in.get_max_index()+.5 - offset)*zoom;
      cout << "Range of non-zero 'out' coordinates : (" 
	<< min2 << ", " << max2 << ")" << endl;
      const int out_min = 
	ask_num("Start index ", (int)(min2-20), (int)(max2+20), (int)min2);
      const int out_max = 
	ask_num("End   index ", out_min, (int)(max2+20), (int)max2);
      
      array out(out_min, out_max);
      // fill with some junk to see if it sets it to 0
      out.fill(111111111.F);
      
      overlap_interpolate(out, in, zoom, offset, true);
      cout << "in:" << in;
      cout << "out:" << out;

      cout << "old sum : " << in.sum() << ", new sum : " << out.sum() << endl;

      {
	Array<1,float> in_coords(in.get_min_index(), in.get_max_index()+1);
	for (int i=in_coords.get_min_index(); i<=in_coords.get_max_index(); ++i)
	  in_coords[i]=i-.5F;
	Array<1,float> out_coords(out.get_min_index(), out.get_max_index()+1);
	for (int i=out_coords.get_min_index(); i<=out_coords.get_max_index(); ++i)
	  out_coords[i]=(i-.5F)/zoom+offset;
	array new_out(out.get_index_range());
	overlap_interpolate(new_out.begin(), new_out.end(),
			    out_coords.begin(), out_coords.end(),
			    in.begin(), in.end(),
			    in_coords.begin(), in_coords.end()
			    );
	cout << new_out;
	new_out -= out;
	cout << "diff:\n" << new_out;
      }
    } while (ask("More ?", true));
  }
  
  
  if (ask ("2D", true))
  {
    
    typedef Array<2,float> array; 
    
    array in(IndexRange2D(-4,4,3,6));
    // initialise with arbitrary data
    for (int i=-4; i<=4; i++)
      for (int j=3; j<=6; j++)
	in[i][j] = .5*square(i-6)+3.*i-30+j;
      
      do
      {
	const float zoom = 
	  ask_num("Zoom ", 0., 100., 1.);
	const float offset = 
	  ask_num("Offset ", -100., 100., 0.);
	const float min2 = (in.get_min_index()-.5 - offset)*zoom;
	const float max2 = (in.get_max_index()+.5 - offset)*zoom;
	cout << "Range of non-zero 'out' coordinates : (" 
	  << min2 << ", " << max2 << ")" << endl;
	const int out_min = 
	  ask_num("Start index ", (int)(min2-20), (int)(max2+20), (int)min2);
	const int out_max = 
	  ask_num("End   index ", out_min, (int)(max2+20), (int)max2);
	
	array out(IndexRange2D(out_min, out_max, 3,6));
	
	
	// fill with some junk to see if it sets it to 0
	out.fill(111111111.F);
	
	overlap_interpolate(out, in, zoom, offset, true);
        cout << out;

	cout << "old sum : " << in.sum() << ", new sum : " << out.sum() << endl;
      {
	Array<1,float> in_coords(in.get_min_index(), in.get_max_index()+1);
	for (int i=in_coords.get_min_index(); i<=in_coords.get_max_index(); ++i)
	  in_coords[i]=i-.5F;
	Array<1,float> out_coords(out.get_min_index(), out.get_max_index()+1);
	for (int i=out_coords.get_min_index(); i<=out_coords.get_max_index(); ++i)
	  out_coords[i]=(i-.5F)/zoom+offset;
	array new_out(out.get_index_range());
	overlap_interpolate(new_out.begin(), new_out.end(),
			    out_coords.begin(), out_coords.end(),
			    in.begin(), in.end(),
			    in_coords.begin(), in_coords.end()
			    );
	cout << new_out;
	new_out -= out;
	cout << "diff:\n" << new_out;
      }
	
      } while (ask("More ?", true));
  }
  
 
  return EXIT_SUCCESS;
}
