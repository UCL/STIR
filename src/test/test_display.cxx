// $Id$: $Date$
/*!

  \file
  \ingroup test

  \brief Very simple test programme for display()

  \author Kris Thielemans
  \author PARAPET project

  \date $Date$

  \version $Revision$
*/
#include "VectorWithOffset.h"
#include "Array.h"
#include "IndexRange3D.h"

#include "display.h"
#include "utilities.h" // for ask_... facilities

USING_NAMESPACE_TOMO

int
main()
{
  cerr << "Tests display with a few very simple bitmaps.\n"
       << "You should see 10 bitmaps (4th and 5th brighter) twice, and then a single bitmap\n";

  typedef float test_type;

  // provide a test example. This could easily be changed in reading 
  // something from file
  Array<3,test_type> t(IndexRange3D(10,100,120));

  VectorWithOffset<float> scale_factors(10);
  scale_factors.fill(1.F);
  // make images 3 and 4 stand out
  scale_factors[3] = 1.3F;
  scale_factors[4] = 1.5F;
  for (int i=0; i<10; i++)
    for (int j=0; j<100; j++)
      for (int k=0; k<120; k++)
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
