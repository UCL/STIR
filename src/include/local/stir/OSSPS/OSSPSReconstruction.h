//
// $Id: 
//
/*!
  \file
  \ingroup reconstructors
  \ingroup OSSPS
  \brief Declaration of class OSSPSReconstruction

  \author Sanida Mustafovic
  \author Kris Thielemans
  
  $Date: 
  $Revision: 
*/
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- $Date$, IRSL
    See STIR/LICENSE.txt for details
*/

#ifndef __stir_OSSPS_OSSPSReconstruction_h__
#define __stir_OSSPS_OSSPSReconstruction_h__

//#include "stir/LogLikBased/MAPBasedReconstruction.h"
#include "stir/LogLikBased/LogLikelihoodBasedReconstruction.h"
#include "local/stir/OSSPS/OSSPSParameters.h"

START_NAMESPACE_STIR

/*!
  \brief Implementation of the  relaxed Ordered Subsets Separable
  Paraboloidal Surrogate ( OSSPS)
  
  It is based on the OSMAPOSL structure.

  When no prior info is specified, this reduces to 'standard' ML.
  
  Note that all this assumes 'balanced subsets', i.e. 
  
    \f[\sum_{b \in \rm{subset}} p_{bv} = 
       \sum_b p_{bv} \over \rm{numsubsets} \f]

  \warning This class should be the last in a Reconstruction hierarchy.
*/
class OSSPSReconstruction: public LogLikelihoodBasedReconstruction
{
public:

  //! Construct the object given all the parameters
  explicit 
    OSSPSReconstruction(const OSSPSParameters&);

  /*!
  \brief Constructor, initialises everything from parameter file, or (when
  parameter_filename == "") by calling ask_parameters().
  */
  explicit 
    OSSPSReconstruction(const string& parameter_filename="");

  //! accessor for the external parameters
  OSSPSParameters& get_parameters(){return parameters;}

  //! accessor for the external parameters
  const OSSPSParameters& get_parameters() const 
    {return parameters;}

  //! gives method information
  virtual string method_info() const;

  //! lists the parameters
  virtual string parameter_info() 
    {return parameters.parameter_info();}
 
private:

  //! pointer to the precomputed denominator 
  shared_ptr<DiscretisedDensity<3,float> > precomputed_denominator_ptr;


  //! the external parameters
  OSSPSParameters parameters;

  //! a workaround for compilers not supporting covariant return types
  virtual ReconstructionParameters& params(){return parameters;}

  //! a workaround for compilers not supporting covariant return types
  virtual const ReconstructionParameters& params() const {return parameters;}

  //! operations prior to the iterations
  virtual void recon_set_up(shared_ptr <DiscretisedDensity<3,float> > const& target_image_ptr);
 
  //! the principal operations for updating the image iterates at each iteration
  virtual void update_image_estimate(DiscretisedDensity<3,float> &current_image_estimate);


};

END_NAMESPACE_STIR

#endif

// __OSSPSReconstruction_h__

