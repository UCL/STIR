//
//
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000 - 2007-10-08, Hammersmith Imanet Ltd
    Copyright (C) 2012-06-05 - 2012, Kris Thielemans
    This file is part of STIR.

    This file is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2.0 of the License, or
    (at your option) any later version.

    This file is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    See STIR/LICENSE.txt for details
*/
/*!
  \file
  \ingroup OSMAPOSL
  \brief Declaration of class stir::OSMAPOSLReconstruction

  \author Matthew Jacobson
  \author Kris Thielemans
  \author PARAPET project

*/

#ifndef __stir_OSMAPOSL_OSMAPOSLReconstruction_h__
#define __stir_OSMAPOSL_OSMAPOSLReconstruction_h__

#include "stir/recon_buildblock/IterativeReconstruction.h"
#include "stir/RegisteredParsingObject.h"

START_NAMESPACE_STIR

template <typename TargetT> 
class PoissonLogLikelihoodWithLinearModelForMean;

/*! \ingroup OSMAPOSL
  \brief Implementation of the Ordered Subsets version of Green's 
  MAP One Step Late algorithm.
  
  See Jacobson et al, PMB for a description of the implementation.

  When no prior info is specified, this reduces to 'standard' OSEM.

  Two different forms of the prior are implemented. 
  (For background, see Mustavic&Thielemans, proc. IEEE MIC 2001).

  When MAP_model == "additive" this implements the standard form of OSL
  (with a small modification for allowing subsets):
  \code
   lambda_new = lambda / ((p_v + beta*prior_gradient)/ num_subsets) *
                   sum_subset backproj(measured/forwproj(lambda))
   \endcode
   with \f$p_v = sum_b p_{bv}\f$.   
   actually, we restrict 1 + beta*prior_gradient/p_v between .1 and 10

  On the other hand, when MAP_model == "multiplicative" it implements
  \code
   lambda_new = lambda / (p_v*(1 + beta*prior_gradient)/ num_subsets) *
                  sum_subset backproj(measured/forwproj(lambda))
  \endcode
   with \f$p_v = sum_b p_{bv}\f$.
   actually, we restrict 1 + beta*prior_gradient between .1 and 10.
  

  Note that all this assumes 'balanced subsets', i.e. 
  
    \f[\sum_{b \in \rm{subset}} p_{bv} = 
       { \sum_b p_{bv} \over \rm{numsubsets} } \f]

  \warning This class should be the last in a Reconstruction hierarchy.
*/
template <typename TargetT>
class OSMAPOSLReconstruction:
        public
            RegisteredParsingObject<
                OSMAPOSLReconstruction <TargetT > ,
                    Reconstruction < TargetT >,
                    IterativeReconstruction < TargetT >
                 >
//public IterativeReconstruction<TargetT >
{
 private:
  typedef RegisteredParsingObject<
    OSMAPOSLReconstruction <TargetT > ,
        Reconstruction < TargetT >,
        IterativeReconstruction < TargetT >
     > base_type;
public:

    //! Name which will be used when parsing a ProjectorByBinPair object
    static const char * const registered_name;

  //! Default constructor (calling set_defaults())
  OSMAPOSLReconstruction();
  /*!
  \brief Constructor, initialises everything from parameter file, or (when
  parameter_filename == "") by calling ask_parameters().
  */
  explicit 
    OSMAPOSLReconstruction(const std::string& parameter_filename);

  //! accessor for the external parameters
  OSMAPOSLReconstruction& get_parameters(){return *this; }

  //! accessor for the external parameters
  const OSMAPOSLReconstruction& get_parameters() const 
    {return *this;}

  //! gives method information
  virtual std::string method_info() const;

  //! Return current objective function
  /* Overloading IterativeReconstruction::get_objective_function()
     with a return-type specifying it'll always be a Poisson log likelihood.
  */
  PoissonLogLikelihoodWithLinearModelForMean<TargetT > const&
    get_objective_function() const;

  /*! \name Functions to set parameters
    This can be used as alternative to the parsing mechanism.
   \warning Be careful with setting shared pointers. If you modify the objects in 
   one place, all objects that use the shared pointer will be affected.
  */
  //@{
  //! subiteration interval at which to apply inter-update filters 
  void set_inter_update_filter_interval(const int);

  //! inter-update filter object
  void set_inter_update_filter_ptr(const shared_ptr<DataProcessor<TargetT > > &);

  //! restrict updates (larger relative updates will be thresholded)
  void set_maximum_relative_change(const double);

  //! restrict updates (smaller relative updates will be thresholded)
  void set_minimum_relative_change(const double);
  
  //! boolean value to determine if the update images have to be written to disk
  void set_write_update_image(const int);

  //! should be either additive or multiplicative
  void set_MAP_model(const std::string&); 
  //@}

  //! prompts the user to enter parameter values manually
  virtual void ask_parameters();

 protected:

  //! determines whether non-positive values in the initial image will be set to small positive ones
  bool enforce_initial_positivity;

  //! determines wether voxels outside a circular FOV will be set to 0 or not
  /*! Currently this circular FOV is slightly smaller than the actual image size (1 pixel at each end or so).
      \deprecated
  */
  bool do_rim_truncation;

  //! subiteration interval at which to apply inter-update filters 
  int inter_update_filter_interval;

  //! inter-update filter object
  shared_ptr<DataProcessor<TargetT > > inter_update_filter_ptr;

  // KT 17/08/2000 3 new parameters

  //! restrict updates (larger relative updates will be thresholded)
  double maximum_relative_change;

  //! restrict updates (smaller relative updates will be thresholded)
  double minimum_relative_change;
  
  //! boolean value to determine if the update images have to be written to disk
  int write_update_image;

  //! should be either additive or multiplicative
  std::string MAP_model; 

  virtual void set_defaults();
  virtual void initialise_keymap();

  //! used to check acceptable parameter ranges, etc...
  virtual bool post_processing();

 
private:
  friend void do_sensitivity(const char * const par_filename);

  //! operations prior to the iterations
  virtual Succeeded set_up(shared_ptr <TargetT > const& target_image_ptr);
 
  //! the principal operations for updating the image iterates at each iteration
  virtual void update_estimate (TargetT& current_image_estimate);

  PoissonLogLikelihoodWithLinearModelForMean<TargetT >&
    objective_function();

  PoissonLogLikelihoodWithLinearModelForMean<TargetT > const&
    objective_function() const;
};

END_NAMESPACE_STIR

#endif

// __OSMAPOSLReconstruction_h__

