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

$Date$
$Revision$

This utility compares two images. They are deemed identical if
their maximum absolute relative difference is less than a hard-coded tolerance 
value.
Diagnostic output is written to stdout, and the return value indicates
if the files are identical or not.
*/
/* Modification History:
  
  KT 12/09/2001 added rim_truncation option
*/

#include "DiscretisedDensity.h"
#include "ArrayFunction.h"
#include "recon_array_functions.h"

#include <numeric>
#inlcude <stdlib.h>

#ifndef TOMO_NO_NAMESPACES
using std::cerr;
using std::cout;
using std::endl;
#endif



//********************** main



USING_NAMESPACE_TOMO


int main(int argc, char *argv[])
{

  const float tolerance=0.0005F;
  int rim_truncation_image = -1;
  if(argc!=3 && argc!=5 || (argc==5 && strcmp(argv[1],"-r")!=0))
  {
    cerr << "Usage: \n" << argv[0] << " [-r rimsize] old_image new_image\n"
	 << "\t 'rimsize' has to be a nonnegative integer.\n"
	 << "\t When the -r option is used, the (radial) rim of the\n"
	 << "\t images will be set to 0, for 'rimsize' pixels.\n";
    return(EXIT_FAILURE);
  }

  if (argc==5)
    {
      // argv[1] already checked above
      rim_truncation_image = atoi(argv[2]);
      argv+=2;
    }

  shared_ptr< DiscretisedDensity<3,float> >  first_operand= 
    DiscretisedDensity<3,float>::read_from_file(argv[1]);

  shared_ptr< DiscretisedDensity<3,float> >  second_operand= 
    DiscretisedDensity<3,float>::read_from_file(argv[2]);

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
  in_place_abs(*first_operand);
  const float max_error=first_operand->find_max();

  const bool same=(max_error/amplitude<=tolerance);

  cout<<endl<<"Maximum absolute error = "<<max_error<<endl;
  cout<<"Error relative to sup-norm of first image = "<<(max_error/amplitude)*100<<" %"<<endl;

  cout<<"\nImage arrays ";

  if(same)
  {
    cout << (max_error == 0 ? "are " : "deemed ")
         << "identical\n";
  }
  else 
  {
    cout<<"deemed different\n";
  }
  cout << "(tolerance used: " << tolerance*100 << " %)\n\n";
  return same?EXIT_SUCCESS:EXIT_FAILURE;

} //end main
