//
// $Id$
//

/*!
\file
\ingroup utilities 
\brief compare images

\author Matthew Jacobson
\author Kris Thielemans
\author PARAPET project

\date    $Date$
\version $Revision$

This utility compares two images. They are deemed identical if
their maximum absolute difference is less than a hard-coded tolerance value.
Diagnostic output is written to stdout, and the return value indicates
if the files are identical or not.
*/

#include "DiscretisedDensity.h"
#include "ArrayFunction.h"

#include <numeric>

#ifndef TOMO_NO_NAMESPACES
using std::cerr;
using std::cout;
using std::endl;
#endif



//********************** main



USING_NAMESPACE_TOMO


int main(int argc, char *argv[])
{

  const float tolerance=0.0001F;
  float max_error;

  if(argc<3)
  {
    cerr<< "Usage:" << argv[0] << "old_image new_image\n";
    exit(EXIT_FAILURE);
  }

  shared_ptr< DiscretisedDensity<3,float> >  first_operand= 
    DiscretisedDensity<3,float>::read_from_file(argv[1]);

  shared_ptr< DiscretisedDensity<3,float> >  second_operand= 
    DiscretisedDensity<3,float>::read_from_file(argv[2]);

  float reference_max=first_operand->find_max();
  float reference_min=first_operand->find_min();

  float amplitude=fabs(reference_max)>fabs(reference_min)?
    fabs(reference_max):fabs(reference_min);

  *first_operand -= *second_operand;
  in_place_abs(*first_operand);
  max_error=first_operand->find_max();

  bool same=(max_error/amplitude<=tolerance)?true:false;

  cout<<endl<<"Maximum absolute error = "<<max_error<<endl;
  cout<<"Error relative to sup-norm of first array = "<<(max_error/amplitude)*100<<" %"<<endl;

  cout<<"Image arrays deemed ";

  if(same)
    cout<<"identical";
  else cout<<"different";
  cout<<endl;

  return same?EXIT_SUCCESS:EXIT_FAILURE;

} //end main
