//
// $Id$
//
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- $Date$, Hammersmith Imanet Ltd

    This file is part of STIR.

    This file is free software; you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation; either version 2.1 of the License, or
    (at your option) any later version.

    This file is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.

    See STIR/LICENSE.txt for details
*/
/*! 
  \file 
  \ingroup FBP3DRP
  \brief Declaration of class stir::FBP3DRPReconstruction
  \author Claire LABBE
  \author Kris Thielemans
  \author PARAPET project
  $Date$
  $Revision$
*/

#ifndef __stir_analytic_FBP3DRP_FBP3DRPRECONSTRUCTION_H__
#define __stir_analytic_FBP3DRP_FBP3DRPRECONSTRUCTION_H__



#include "stir/recon_buildblock/AnalyticReconstruction.h"
#include "stir/ProjDataInfoCylindrical.h"
#include "stir/recon_buildblock/ForwardProjectorByBin.h"
#include "stir/recon_buildblock/BackProjectorByBin.h"
#include "stir/analytic/FBP3DRP/ColsherFilter.h"
#include "stir/ArcCorrection.h"
#include "stir/shared_ptr.h"

START_NAMESPACE_STIR

template <typename elemT> class RelatedViewgrams;
template <typename elemT> class Sinogram;
template <typename elemT> class SegmentBySinogram;
template <typename elemT> class VoxelsOnCartesianGrid;
template <int num_dimensions, typename elemT> class DiscretisedDensity;
class Succeeded;

/* KT 180899 forget about PETAnalyticReconstruction for the moment
 TODO Derive from PETAnalyticReconstruction when it makes any sense
 */
/*!
  \ingroup FBP3DRP
  \class FBP3DRPReconstruction
  \brief This class contains the implementation of the FBP3DRP algorithm.

  This class implements the 3DRP algorithm (Kinahan and Rogers) as a specific
  case of a 3D FBP reconstruction algorithm. 

  Some care is taken to achieve
  a fairly general implementation. For instance, the number of sinograms
  in the oblique segments is arbitrary (i.e. does not have to be related
  to what you would get from a cylindrical PET scanner). Also, scale
  factors are inserted such that the reconstructed image has (almost)
  the same scale independent of the number of segments that is used.

  Nevertheless, this is an analytic algorithm, and it implements a discrete
  version of a continuous inversion formula. This will work best (but of 
  course slowest) when the number of segments is large. 

  This implementation is specific for data using sampling corresponding
  to cylindrical PET scanners.
  This dependency essentially only occurs in the backprojection where a
  Jacobian is necessary in this case, and where the number of ring differences
  in each segment is taken into account. It would be not too difficult
  to make a version that works on e.g. spherical sampling.

  \par About zooming (rescaling + offset):
  1) The 2D FBP process  works at full resolution, i.e on the original 
     number of bins, and with a pixel size equal to the bin size.
  2) For the process of oblique sinograms:
	  - Forward projection works at full resolution i.e forward projection works
	  from images without zooming and complete missing projection data on normal sinograms
	  - Colsher filter is then applied on complete data
	  - 3D backprojection then puts this data into an image with 
	  appropriate voxel sizes, i.e. it is up to the backprojector to perform
	  the zooming.
	  - So, no zooming is needed on the final image.
     

*/
class FBP3DRPReconstruction: public AnalyticReconstruction
{
  typedef AnalyticReconstruction base_type;
public:


  //! Default constructor (calls set_defaults())
  FBP3DRPReconstruction (); 

  /*!
    \brief Constructor, initialises everything from parameter file, or (when
    parameter_filename == "") by calling ask_parameters().
  */
  explicit 
    FBP3DRPReconstruction(const string& parameter_filename);

  // explicitly implement destructor (NOT inline) to avoid funny problems with
  // the shared_ptr<DiscretisedDensity<3,float> > destructor with gcc.
  // Otherwise, gcc will complain if you didn't include DiscretisedDensity.h
  // because it doesn't know ~DiscretisedDensity.
  ~FBP3DRPReconstruction();

//! This method returns the type of the reconstruction algorithm during the reconstruction, here it is FBP3DRP
   virtual string method_info() const;

protected:
/*!
  \brief Implementation of the reconstruction
  
  This method implements the reconstruction by giving as input the emission sinogram data corrected for attenuation,
  scatter, dead time,
  and returns the output reconstructed image
*/

   virtual Succeeded actual_reconstruct(shared_ptr<DiscretisedDensity<3,float> > const&);
    
//! Best fit of forward projected sinograms
    void do_best_fit(const Sinogram<float> &sino_measured, const Sinogram<float> &sino_calculated);


//!  2D FBP implementation.
    void do_2D_reconstruction();
    
//!  Save image data.
    void do_save_img(const char *file, const VoxelsOnCartesianGrid<float> &data) const;

//!  Read image estimated from 2D FBP 
    void do_read_image2D();

    
//!  3D reconstruction implementation.       
  void do_3D_Reconstruction(  VoxelsOnCartesianGrid<float> &image);

//!  Arc-correction viewgrams
  void do_arc_correction(RelatedViewgrams<float> & viewgrams) const;
   
//!  Growing 8 viewgrams in both ring and bin directions.
    void do_grow3D_viewgram(RelatedViewgrams<float> & viewgrams, 
                            int rmin, int rmax);
//!  3D forward projection implentation by view.
    void do_forward_project_view(RelatedViewgrams<float> & viewgrams,
                                 int rmin, int rmax,
                                 int orig_min_ring, int orig_max_ring) const; 
//!  Apply Colsher filter to 8 viewgrams.
    void do_colsher_filter_view( RelatedViewgrams<float> & viewgrams);
//!  3D backprojection implentation for 8 viewgrams.
    void do_3D_backprojection_view(RelatedViewgrams<float> const & viewgrams,
                                   VoxelsOnCartesianGrid<float> &image,
                                   int rmin, int rmax);
//!  Saving CPU timing and values of reconstruction parameters into a log file.    
    void do_log_file(const VoxelsOnCartesianGrid<float> &image);




    virtual void do_byview_initialise(const VoxelsOnCartesianGrid<float>& image) const
    {};

    virtual void do_byview_finalise(VoxelsOnCartesianGrid<float>& image) {};
public:
      // KT 230899 this has to be public to let the Para stuff access it (sadly)

    virtual void do_process_viewgrams(
                                  RelatedViewgrams<float> & viewgrams,
                                  int rmin, int rmax,
                                  int orig_min_ring, int orig_max_ring,
                                  VoxelsOnCartesianGrid<float> &image);

    // parameters stuff
 public:

   
  void ask_parameters();

 
protected:

  //! Switch to display intermediate images, 0,1,2
  int display_level;

  //! Switch to save files after each segment, 0 or 1
  int save_intermediate_files;
   
  //! Filename of image used in the reprojection step (default is empty)
  /*! If the filename is empty, FBP is used (with filter parameters as 
    specified further). 

    \warning This image must have the correct scale. That is, if you use the 
    forward projector on it, you get sinograms of the same scale as the 
    input sinograms. There is NO check on this.
  */  
  string image_for_reprojection_filename;

  //! Number of segments to combine with SSRB before calling FBP
  /*! default -1 will use SSRB only when the data are not yet axially compressed */
  int num_segments_to_combine;

  //! Transaxial extension for FFT
  int PadS; 
  //! Axial extension for FFT
  int PadZ;
  //! Ramp filter: Alpha value
  double alpha_ramp;
  //! Ramp filter: Cut off frequency
  double fc_ramp;  

  //! Alpha parameter for Colsher filter in axial direction
  double alpha_colsher_axial;
  //! Cut-off frequency for Colsher filter in axial direction
  double fc_colsher_axial;
  //! Alpha parameter for Colsher filter in planar direction
  double alpha_colsher_planar; 
  //! Cut-off frequency for Colsher filter in planar direction
  double fc_colsher_planar; 
  //! Define Colsher at larger size than used for filtering, axial direction
  int colsher_stretch_factor_axial;
  //! Define Colsher at larger size than used for filtering, planar direction
  int colsher_stretch_factor_planar;
 

  //! =1 => apply additional fitting procedure to forward projected data (DISABLED)
  int fit_projections; 
private:

  virtual void set_defaults();
  virtual void initialise_keymap();


  //! access to input proj_data_info cast to cylindrical type
  const ProjDataInfoCylindrical& input_proj_data_info_cyl() const;
  //! Size info for the projection data with missing data filled in
  shared_ptr<ProjDataInfo> proj_data_info_with_missing_data_sptr;

  shared_ptr<DiscretisedDensity<3,float> > image_estimate_density_ptr;
  // convenience access functions to the above member
  inline VoxelsOnCartesianGrid<float>&  estimated_image();
  inline const VoxelsOnCartesianGrid<float>&  estimated_image() const;

  shared_ptr<ForwardProjectorByBin> forward_projector_sptr; 
  shared_ptr<BackProjectorByBin> back_projector_sptr;
#ifndef NRFFT
  ColsherFilter colsher_filter;
#endif
  float alpha_fit;
  float beta_fit;
  
  shared_ptr<ArcCorrection> arc_correction_sptr;
};

END_NAMESPACE_STIR

#endif

