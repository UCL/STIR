
//
//
// $Id$
//
//
#include "recon_buildblock/ConjugateSpaceAlgorithmParameters.h"

//
//MJ 01/02/2000 added
//small for now
//

//ConjugateSpaceAlgorithmParameters::ConjugateSpaceAlgorithmParameters()
//  :LogLikelihoodBasedAlgorithmParameters()
//{
//}

void ConjugateSpaceAlgorithmParameters::ask_parameters()
{

  LogLikelihoodBasedAlgorithmParameters::ask_parameters();

 


}


bool ConjugateSpaceAlgorithmParameters::post_processing()
{
  if (LogLikelihoodBasedAlgorithmParameters::post_processing())
    return true;

  return false;
}



// KT&MJ 230899 changed return-type to string
string ConjugateSpaceAlgorithmParameters::parameter_info() const
{
  

  // TODO dangerous for out-of-range, but 'old-style' ostrstream seems to need this
  char str[10000];
  ostrstream s(str, 10000);


  s << LogLikelihoodBasedAlgorithmParameters::parameter_info();
    



 
  s<<ends;

  return s.str();
  
}


