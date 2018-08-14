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
  \brief Declaration of class stir::KOSMAPOSLReconstruction

  \author Matthew Jacobson
  \author Kris Thielemans
  \author PARAPET project

*/

#ifndef __stir_KOSMAPOSL_KOSMAPOSLReconstruction_h__
#define __stir_KOSMAPOSL_KOSMAPOSLReconstruction_h__

#include "stir/recon_buildblock/IterativeReconstruction.h"
#include "stir/RegisteredParsingObject.h"
#include "stir/OSMAPOSL/OSMAPOSLReconstruction.h"

START_NAMESPACE_STIR

template <typename TargetT> 
class PoissonLogLikelihoodWithLinearModelForMean;

/*! \ingroup KOSMAPOSL
  \brief An reconstructor class appropriate for PET emission data

  This class implements the iterative algorithm obtained using the Kernel method (KEM) and Hybrid kernel method (HKEM).
  This implementation corresponds to the one presented by Deidda D et al, ``Hybrid PET-MR list-mode kernelized expectation
  maximization reconstruction for quantitative PET images of the carotid arteries", IEEE MIC Atlanta, 2017. However, this allows
  also sinogram-based reconstruction. Each voxel value of the image, \f$ \boldsymbol{\lambda}\f$, can be represented as a
  linear combination using the kernel method.  If we have an image with prior information, we can construct for each voxel
  \f$ j \f$ of the PET image a feature vector, $\f \boldsymbol{v}_j \f$, using the prior information. The voxel value,
  \f$\lambda_j\f$, can then be described using the kernel matrix



  \f[
   \lambda_j=  \sum_{l=1}^L \alpha_l k_{jl}
  \f]

  where \f$k_{jl}\f$ is the \f$jl^{th}\f$ kernel element of the matrix, \f$\boldsymbol{K}\f$.
  The resulting algorithm with OSEM, for example, is the following:

  \f[
  \alpha^{(n+1)}_j =  \frac{ \alpha^{(n)}_j }{\sum_{m} k^{(n)}_{jm} \sum_i p_{mi}} \sum_{m}k^{(n)}_{jm}\sum_i p_{mi}\frac{ y_i }{\sum_{q} p_{iq} \sum_l k^{(n)}_{ql}\alpha^{(n)}_l  + s_i}
  \f[

  where the  element, $\f jl \f$, of the kernel can be written as:

  \f[
    k^{(n)}_{jl} = k_m(\boldsymbol{v}_j,\boldsymbol{v}_l) \cdot k_p(\boldsymbol{z}^{(n)}_j,\boldsymbol{z}^{(n)}_l);
  \f]

  with

  \f[
   k_m(\boldsymbol{v}_j,\boldsymbol{v}_l) = \exp \left(\tiny - \frac{\|  \boldsymbol{v}_j-\boldsymbol{v}_l \|^2}{2 \sigma_m^2} \right) \exp \left(- \frac{\tiny \|  \boldsymbol{x}_j-\boldsymbol{x}_l \|^2}{ \tiny 2 \sigma_{dm}^2} \right)
  \f]

  being the MR component of the kernel and

  \f[
   k_p(\boldsymbol{z}^{(n)}_j,\boldsymbol{z}^{(n)}_l) = \exp \left(\tiny - \frac{\|  \boldsymbol{z}^{(n)}_j-\boldsymbol{z}^{(n)}_l \|^2}{2 \sigma_p^2} \right) \exp \left(\tiny - \frac{\|  \boldsymbol{x}_j-\boldsymbol{x}_l \|^2}{ \tiny{2 \sigma_{dp}^2}} \right)
  \f]

  is the part coming from the PET iterative update. Here, the Gaussian kernel functions have been modulated by the distance between voxels in the image space.

  \par Parameters for parsing

  \verbatim
  KOSMAPOSL Parameters:=

  hybrid:=1
  sigma m:= 1                                ;is the parameter $\f \sigma_{m} \f$;
  sigma p:=1                                 ;is the parameter $\f \sigma_{p} \f$;
  sigma dm:=1                                ;is the parameter $\f \sigma_{dm} \f$;
  sigma dp:=1                                ;is the parameter $\f \sigma_{dp} \f$;
  number of neighbours:= 3                   ;is the cubic root of the number of voxels in the neighbourhood;
  anatomical image filename:=filename        ;is the filename of the anatomical image;
  number of non-zero feature elements:=1     ;is the number of non zero elements in the feature vector;
  only_2D:=0                                 ;=1 if you want to reconstruct 2D images;


  End KOSMAPOSL Parameters :=
  \endverbatim
*/

template <typename TargetT>
class KOSMAPOSLReconstruction:
        public
            RegisteredParsingObject<
        KOSMAPOSLReconstruction <TargetT > ,
        Reconstruction < TargetT >,
        OSMAPOSLReconstruction < TargetT >
         >

//public IterativeReconstruction<TargetT >
{
 private:
  typedef RegisteredParsingObject<
    KOSMAPOSLReconstruction <TargetT > ,
    Reconstruction < TargetT >,
    OSMAPOSLReconstruction < TargetT >
     >


    base_type;
public:

    //! Name which will be used when parsing a KOSMAPOSLReconstruction object
    static const char * const registered_name;

  //! Default constructor (calling set_defaults())
  KOSMAPOSLReconstruction();
  /*!
  \brief Constructor, initialises everything from parameter file, or (when
  parameter_filename == "") by calling ask_parameters().
  */
  explicit 
    KOSMAPOSLReconstruction(const std::string& parameter_filename);

  //! accessor for the external parameters
  KOSMAPOSLReconstruction& get_parameters(){return *this; }

  //! accessor for the external parameters
  const KOSMAPOSLReconstruction& get_parameters() const
    {return *this;}

  //kernel
  const std::string get_anatomical_filename() const;
  const int get_num_neighbours() const;
  const int get_num_non_zero_feat() const;
  const double get_sigma_m() const;
  double get_sigma_p();
  double get_sigma_dp();
  double get_sigma_dm();
  const bool get_only_2D() const;
  bool get_hybrid();

   shared_ptr<TargetT>& get_kpnorm_sptr();
   shared_ptr<TargetT>& get_kmnorm_sptr();
   shared_ptr<TargetT>& get_anatomical_sptr();

    /*! \name Functions to set parameters
    This can be used as alternative to the parsing mechanism.
   \warning Be careful with setting shared pointers. If you modify the objects in 
   one place, all objects that use the shared pointer will be affected.
  */

  void set_kpnorm_sptr(shared_ptr<TargetT>&);
  void set_kmnorm_sptr(shared_ptr<TargetT>&);
  void set_anatomical_sptr(shared_ptr<TargetT>&);


  //! boolean value to determine if the update images have to be written to disk
  void set_write_update_image(const int);


  //! prompts the user to enter parameter values manually
  virtual void ask_parameters();

 protected:
  //! Filename with input projection data
  std::string input_filename,kernelised_output_filename_prefix;
  std::string current_kimage_filename;
  std::string sens_filenames;

  //! Anatomical image filename
  std::string anatomical_image_filename;
  shared_ptr<TargetT> anatomical_sptr;
  shared_ptr<TargetT> kpnorm_sptr,kmnorm_sptr;
 //kernel parameters
  int num_neighbours,num_non_zero_feat,num_elem_neighbourhood,num_voxels,dimz,dimy,dimx;
  double sigma_m;
  bool only_2D;
  bool hybrid;
  double sigma_p;
  double sigma_dp, sigma_dm;

  //! boolean value to determine if the update images have to be written to disk
  int write_update_image;

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

  int subiteration_counter;
  double anatomical_sd;
  mutable Array<3,float> distance;
  /*! Create a matrix containing the norm of the difference between two feature vectors, \f$ \|  \boldsymbol{z}^{(n)}_j-\boldsymbol{z}^{(n)}_l \| \f$. */
  /*! This is done for the PET image which keeps changing*/
    void  calculate_norm_matrix(TargetT &normp,
                                const int &dimf_row,
                                const int &dimf_col,
                                const TargetT& pet,
                                Array<3,float> distance);

  /*! Create a matrix similarly to calculate_norm_matrix() but this is done for the anatomical image, */
  /*! which does not  change over iteration.*/
    void  calculate_norm_const_matrix(TargetT &normm,
                                const int &dimf_row,
                                const int &dimf_col);

  /*! Estimate the SD of the anatomical image to be used as normalisation for the feature vector */
    void estimate_stand_dev_for_anatomical_image(double &SD);

  /*! Compute for each voxel, jl, of the PET image the linear combination between the coefficient \f$ \alpha_{jl} \f$ and the kernel matrix \f$ k_{jl} \f$\f$ */
  /*! The information is stored in the image, kImage */
  void full_compute_kernelised_image(TargetT& kernelised_image_out,
                                     const TargetT& image_to_kernelise,
                                     const TargetT& current_alpha_estimate);

   /*! Similar to compute_kernelised_image() but this is the special case when the feature vectors contains only one non-zero element. */
   /*! The computation becomes faster because we do not need to create norm matrixes*/
  void compact_compute_kernelised_image(TargetT& kernelised_image_out,
                                        const TargetT& image_to_kernelise,
                                        const TargetT& current_alpha_estimate);

  /*! choose between compact_compute_kernelised_image() and  full_compute_kernelised_image()*/
  void compute_kernelised_image(TargetT& kernelised_image_out,
                                const TargetT& image_to_kernelise,
                                const TargetT& current_alpha_estimate);
};

END_NAMESPACE_STIR

#endif

// __KOSMAPOSLReconstruction_h__
