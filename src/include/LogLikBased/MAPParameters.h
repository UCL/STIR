//
// $Id$
//

#ifndef __MAPParameters_h__
#define __MAPParameters_h__


/*!
  \file 
  \ingroup LogLikBased_buildblock
 
  \brief declares the MAPParameters class

  \author Claire Labbe
  \author Matthew Jacobson
  \author Kris Thielemans
  \author PARAPET project

  \date    $Date$
  \version $Revision$

*/

#include "LogLikBased/LogLikelihoodBasedAlgorithmParameters.h"

START_NAMESPACE_TOMO


/*! 
 \ingroup LogLikBased_buildblock
 \brief base parameter class for MAP based algorithms 


 */


class MAPParameters: public LogLikelihoodBasedAlgorithmParameters
{

 public:

  //! lists the parameter values
  virtual string parameter_info() const 
    {return LogLikelihoodBasedAlgorithmParameters::parameter_info();}

  //! constructor
  MAPParameters(){MAP_model="";}

  //! the type of MAP model
  string MAP_model; 
    
 protected:

  //! used to check acceptable parameter ranges, etc...
  virtual bool post_processing()
     {return LogLikelihoodBasedAlgorithmParameters::post_processing();}
  


};


END_NAMESPACE_TOMO


#endif // __MAPParameters_h__
