//
// $Id$
//


#ifndef __OSMAPOSLReconstruction_h__
#define __OSMAPOSLReconstruction_h__

#include "recon_buildblock/MAPBasedReconstruction.h"
#include "OSEM/OSMAPOSLParameters.h"

class OSMAPOSLReconstruction: public MAPBasedReconstruction
{
 
   
 public:

  OSMAPOSLParameters& get_parameters(){return parameters;}
  const OSMAPOSLParameters& get_parameters() const 
    {return parameters;}

  virtual string method_info() const;
  virtual string parameter_info() const
    {return parameters.parameter_info();}
 
  // Ultimately the argument will be a PETStudy or the like
  explicit OSMAPOSLReconstruction(char* parameter_filename="")
    {
     
      LogLikelihoodBasedReconstruction_ctor(parameter_filename);
      cerr<<parameters.parameter_info();
    }

  virtual ~OSMAPOSLReconstruction(){LogLikelihoodBasedReconstruction_dtor();}

 protected:

  virtual void recon_set_up(PETImageOfVolume &target_image);
  virtual void update_image_estimate(PETImageOfVolume &current_image_estimate);
  virtual void end_of_iteration_processing(PETImageOfVolume &current_image_estimate)
    {
      loglikelihood_common_end_of_iteration_processing(current_image_estimate);
    }



 
 private:

  virtual ReconstructionParameters& params(){return parameters;}

  OSMAPOSLParameters parameters;

};

#endif

// __OSMAPOSLReconstruction_h__
