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
    Copyright (C) 2000- $Date$, IRSL
    See STIR/LICENSE.txt for details
*/

#include "stir/interpolate.h"
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
      cout << out;

      cout << "old sum : " << in.sum() << ", new sum : " << out.sum() << endl;
      
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
	
      } while (ask("More ?", true));
  }
  
 
  return EXIT_SUCCESS;
}
