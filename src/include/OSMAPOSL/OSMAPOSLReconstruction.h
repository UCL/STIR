//
// $Id$
//
/*!

  \file

  \brief Declaration of class OSMAPOSLReconstruction

  \author Matthew Jacobson
  \author Kris Thielemans
  \author PARAPET project

  \date $Date$

  \version $Revision$
*/

#ifndef __OSMAPOSLReconstruction_h__
#define __OSMAPOSLReconstruction_h__

#include "LogLikBased/MAPBasedReconstruction.h"
#include "OSMAPOSL/OSMAPOSLParameters.h"

START_NAMESPACE_TOMO

/*!
  \brief Implementation of the Ordered Subsets version of Green's 
  MAP One Step Late algorithm.

  When no prior info is specified, this reduces to 'standard' OSEM.

  TODO more doc

  \warning This class should be the last in a Reconstruction hierarchy.
*/
class OSMAPOSLReconstruction: public MAPBasedReconstruction
{
public:

  //! Construct the object given all the parameters
  explicit 
    OSMAPOSLReconstruction(const OSMAPOSLParameters&);

  /*!
  \brief Constructor, initialises everything from parameter file, or (when
  parameter_filename == "") by calling ask_parameters().
  */
  explicit 
    OSMAPOSLReconstruction(const string& parameter_filename="");

  //! virtual destructor
  virtual ~OSMAPOSLReconstruction(){};

  //! accessor for the external parameters
  OSMAPOSLParameters& get_parameters(){return parameters;}

  //! accessor for the external parameters
  const OSMAPOSLParameters& get_parameters() const 
    {return parameters;}

  //! gives method information
  virtual string method_info() const;

  //! lists the parameters
  virtual string parameter_info() const
    {return parameters.parameter_info();}
 
private:

  //! the external parameters
  OSMAPOSLParameters parameters;

  //! a workaround for compilers not supporting covariant return types
  virtual ReconstructionParameters& params(){return parameters;}

  //! a workaround for compilers not supporting covariant return types
  virtual const ReconstructionParameters& params() const {return parameters;}

  //! operations prior to the iterations
  virtual void recon_set_up(shared_ptr <DiscretisedDensity<3,float> > const& target_image_ptr);
 
  //! the principal operations for updating the image iterates at each iteration
  virtual void update_image_estimate(DiscretisedDensity<3,float> &current_image_estimate);

  //! operations for the end of the iteration
  virtual void end_of_iteration_processing(DiscretisedDensity<3,float> &current_image_estimate)
	{
	  LogLikelihoodBasedReconstruction::end_of_iteration_processing(current_image_estimate);
	}

};

END_NAMESPACE_TOMO

#endif

// __OSMAPOSLReconstruction_h__

