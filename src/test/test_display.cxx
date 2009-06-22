// $Id$
/*!

  \file
  \ingroup test

  \brief Very simple test program for stir::display()

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
#include "stir/VectorWithOffset.h"
#include "stir/Array.h"
#include "stir/IndexRange3D.h"

#include "stir/display.h"
#include "stir/utilities.h" // for ask_... facilities

USING_NAMESPACE_STIR

int
main()
{
  cerr << "Tests display with a few very simple bitmaps.\n"
       << "You should see 10 bitmaps (4th and 5th brighter) twice, and then a single bitmap\n";

  typedef float test_type;

  // provide a test example. This could easily be changed in reading 
  // something from file
  // note: sizes are prime numbers to avoid having accidental matches 
  // with 'word-boundaries' etc. This is especailly an issue 
  // when using X windows.
  Array<3,test_type> t(IndexRange3D(10,87,123));

  VectorWithOffset<float> scale_factors(10);
  scale_factors.fill(1.F);
  // make images 3 and 4 stand out
  scale_factors[3] = 1.3F;
  scale_factors[4] = 1.5F;
  for (int i=0; i<t.get_max_index(); i++)
    for (int j=0; j<t[i].get_max_index(); j++)
      for (int k=0; k<t[i][j].get_max_index(); k++)
	t[i][j][k] = test_type(100*sin((i+1.)*j*k/100000. * _PI * 2));
 
  // from here do the real work

  /* Default values */
  double maxi = t[0].find_max() * scale_factors[0];
  for (int i=t.get_min_index(); i<=t.get_max_index(); i++)
    maxi = std::max(maxi, (double)t[i].find_max() * scale_factors[i]);

  int scale = 0;
  int max_mode = 0;


  if (!ask("Accept defaults ?",true))
  { 

    scale = ask_num("Enlargement (0 means maximum possible)",0,999,scale);
    max_mode=ask_num(
 "Choose : Same scaling for all images (maximum self determined) (0)\
\n         All images scaled with given maximum                  (1)\
\n         Independent scaling of the images                     (2)",0,2,max_mode);
    switch (max_mode)
    { case 0 :                          /* maximum self determined      */
        break;
      case 1 :                          /* given maximum                */
        maxi =
         ask_num("Maximum value (will correspond to highest color value)",
                0.0,1e30,maxi);
        break;
      case 2 :                          /* independent                  */
        maxi = 0.0;
        break;
    }
  }
  VectorWithOffset<char *> text(t.get_min_index(), t.get_max_index());
  for (int i=t.get_min_index(); i<= t.get_max_index(); i++)
    {
      text[i] = new char [15];
      sprintf(text[i], "image %d", i);      
    }

  display(t, scale_factors, text, maxi, "Test display 3D all args", scale);
  display(t,t.find_max()/2,"Test display 3D 3 args, half the colour scale" );

  for (int i=t.get_min_index(); i<= t.get_max_index(); i++)
    delete[] text[i];
  

  display(*t.begin(), "Test display 2D, 2 args");
  return EXIT_SUCCESS;
}
