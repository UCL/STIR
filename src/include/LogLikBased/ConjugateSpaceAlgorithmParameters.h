//
// $Id$
//

#ifndef __ConjugateSpaceAlgorithmParameters_h__
#define __ConjugateSpaceAlgorithmParameters_h__



/*!
  \file 
 \ingroup LogLikBased_buildblock
 
  \brief declares the ConjugateSpaceAlgorithmParameters class

  \author Matthew Jacobson
  \author PARAPET project

  \date    $Date$
  \version $Revision$
*/


#include "LogLikBased/LogLikelihoodBasedAlgorithmParameters.h"

START_NAMESPACE_TOMO

/*!
 \brief base parameter class for algorithms based on conjugate space methods
  \ingroup LogLikBased_buildblock

*/


class ConjugateSpaceAlgorithmParameters: public LogLikelihoodBasedAlgorithmParameters
{



public:

  //ConjugateSpaceAlgorithmParameters();

  //! lists the parameter values
  virtual string parameter_info() const;

  //! prompts the user to enter parameter values manually
  virtual void ask_parameters();

  //! a proportionality factor for the step size
  double step_size_factor;

protected:

  //! used to check acceptable parameter ranges, etc...  
  virtual bool post_processing();


};



END_NAMESPACE_TOMO


#endif

// __ConjugateSpaceAlgorithmParameters_h__
