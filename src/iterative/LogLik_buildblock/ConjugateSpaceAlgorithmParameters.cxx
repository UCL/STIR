
//
//
// $Id$
//
/*!
  \file
  \ingroup LogLikBased_buildblock

  \brief non-inline implementations for ConjugateSpaceAlgorithmParameters

  \author Matthew Jacobson
  \author Kris Thielemans
  \author PARAPET project
  
  \date $Date$

  \version $Revision$
*/
#include "LogLikBased/ConjugateSpaceAlgorithmParameters.h"

#ifndef TOMO_NO_NAMESPACES
using std::ends;
#endif

START_NAMESPACE_TOMO

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



string ConjugateSpaceAlgorithmParameters::parameter_info() const
{
  
  
  // TODO dangerous for out-of-range, but 'old-style' ostrstream seems to need this
  char str[10000];
  ostrstream s(str, 10000);
		
  s << LogLikelihoodBasedAlgorithmParameters::parameter_info();  
  s<<ends;
  
  return s.str();
  
}

END_NAMESPACE_TOMO
