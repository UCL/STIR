//
// $Id$
//

/*! 
  \file 
  \brief Class for serial FBP3DRP reconstruction
  \author Claire LABBE
  \author Kris Thielemans
  \author PARAPET project
  $Date$
  $Revision$
*/
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- $Date$, IRSL
    See STIR/LICENSE.txt for details
*/

#ifndef __FBP3DRPRECONSTRUCTION_H__
#define __FBP3DRPRECONSTRUCTION_H__



#include "stir/recon_buildblock/Reconstruction.h"
#include "local/stir/FBP3DRP/FBP3DRPParameters.h"
#include "stir/ProjDataInfoCylindricalArcCorr.h"


START_NAMESPACE_STIR

template <typename elemT> class RelatedViewgrams;
template <typename elemT> class Sinogram;
template <typename elemT> class SegmentBySinogram;
template <typename elemT> class VoxelsOnCartesianGrid;
template <int num_dimensions, typename elemT> class DiscretisedDensity;
template <typename T> class shared_ptr;

/* KT 180899 forget about PETAnalyticReconstruction for the moment
 TODO Derive from PETAnalyticReconstruction when it makes any sense
 */
/*!
  \class FBP3DRPReconstruction
  \brief This class contains the implementation of the serial FBP3DRP reconstruction,
  derived from PETReconstruction and  FBP3DRPParameters.

        
  ABout zooming (rescaling + offset),
         1) The 2D FBP process  works at full resolution
            i.e on the original number of bins , then after 2D backprojection,
            a zooming is done on the image
          2) For the process of oblique sinograms,
	  - Forward projection works at full resolution i.e forward projection works
	  from images without zooming and complete missing projection data on normal sinograms
	  - Colher filter is then applied on complete data
	  - Then, at this, zooming is applied on sinograms after Colhser filtering
	  so that 3D backprojection will work faster if final size is smaller than the original number of bins
	  - At the final stage, no zooming is needed on the final image as zooming is already done on sinograms
     

*/
class FBP3DRPReconstruction: public Reconstruction, public FBP3DRPParameters 
{

public:


//! This constructor takes as input parameters, the list of parameters needed for the FBP3DRP reconstruction 
   explicit 
     FBP3DRPReconstruction (const FBP3DRPParameters& parameters); 

     /*!
  \brief Constructor, initialises everything from parameter file, or (when
  parameter_filename == "") by calling ask_parameters().
  */
  explicit 
    FBP3DRPReconstruction(const string& parameter_filename="");

  // explicitly implement destructor (NOT inline) to avoid funny problems with
  // the shared_ptr<DiscretisedDensity<3,float> > destructor with gcc.
  // Otherwise, gcc will complain if you didn't include DiscretisedDensity.h
  // because it doesn't know ~DiscretisedDensity.
  ~FBP3DRPReconstruction();

//! This method returns all the parameters used for the reconstruction algorithm
   virtual string parameter_info();
/*!
  \brief Implementation of the reconstruction
  
  This method implements the reconstruction by giving as input the emission sinogram data corrected for attenuation,
  scatter, dead time (see detail in D3.2 section Correction Info in emission, transmission...),
  and returns the output reconstructed image
*/

   virtual Succeeded reconstruct(shared_ptr<DiscretisedDensity<3,float> > const&);

   //! Reconstruction that gets target_image info from the parameters
   /*! sadly have to repeat that here, as Reconstruct::reconstruct() gets hidden by the 
       1-argument version.
       */
   virtual Succeeded reconstruct();

//! This method returns the type of the reconstruction algorithm during the reconstruction, here it is FBP3DRP
    virtual string method_info() const
        { return("FBP3DRP"); }


protected:


    
//! Best fit of forward projected sinograms
    void do_best_fit(const Sinogram<float> &sino_measured, const Sinogram<float> &sino_calculated);


//!  2D FBP implementation.
    void do_2D_reconstruction();
    
//!  Save image data.
    void do_save_img(const char *file, const VoxelsOnCartesianGrid<float> &data);

//!  Read image estimated from 2D FBP 
    void do_read_image2D();

    
//!  3D reconstruction implementation.       
  void do_3D_Reconstruction(  VoxelsOnCartesianGrid<float> &image);
   
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

private:
  virtual ReconstructionParameters& params();
  virtual const ReconstructionParameters& params() const;

  ProjDataInfoCylindricalArcCorr proj_data_info_cyl;
  shared_ptr<DiscretisedDensity<3,float> > image_estimate_density_ptr;
  // convenience access functions to the above member
  inline VoxelsOnCartesianGrid<float>&  estimated_image();
  inline const VoxelsOnCartesianGrid<float>&  estimated_image() const;

  float alpha_fit;
  float beta_fit;
};

END_NAMESPACE_STIR

#endif

