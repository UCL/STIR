//
//
/*
    Copyright (C) 2018 Commonwealth Scientific and Industrial Research Organisation
    Copyright (C) 2018-2019 University of Leeds
    Copyright (C) 2019 University College of London
    Copyright (C) 2019 National Physical Laboratory

    This file is part of STIR.

    This file is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    This file is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    See STIR/LICENSE.txt for details
*/

/*!
  \file
  \ingroup KOSMAPOSL
  \ingroup reconstructors
  \brief  implementation of the stir::KOSMAPOSLReconstruction class

  \author Daniel Deidda
  \author Ashley Gillman
  \author Palak Wadhwa
  \author Kris Thielemans
      
*/

#include "stir/KOSMAPOSL/KOSMAPOSLReconstruction.h"
#include "stir/OSMAPOSL/OSMAPOSLReconstruction.h"
#include "stir/recon_buildblock/PoissonLogLikelihoodWithLinearModelForMean.h"
#include "stir/DiscretisedDensity.h"
//#include "stir/LogLikBased/common.h"
#include "stir/ThresholdMinToSmallPositiveValueDataProcessor.h"
#include "stir/ChainedDataProcessor.h"
#include "stir/Succeeded.h"
#include "stir/numerics/divide.h"
#include "stir/thresholding.h"
#include "stir/is_null_ptr.h"
#include "stir/NumericInfo.h"
#include "stir/utilities.h"
// for get_symmetries_ptr()
#include "stir/DataSymmetriesForViewSegmentNumbers.h"
#include "stir/ViewSegmentNumbers.h"
#include "stir/stream.h"
#include "stir/info.h"
#include "stir/VoxelsOnCartesianGrid.h"

//#include "stir/modelling/ParametricDiscretisedDensity.h"
//#include "stir/modelling/KineticParameters.h"

#include <memory>
#include <iostream>
#ifdef BOOST_NO_STRINGSTREAM
#include <strstream.h>
#else
#include <sstream>
#endif

#include "stir/unique_ptr.h"
#include <algorithm>
using std::min;
using std::max;
#ifndef STIR_NO_NAMESPACES
using std::cerr;
using std::endl;
#endif
#include "stir/IndexRange3D.h"
#include "stir/IO/read_from_file.h"
#include "stir/IO/write_to_file.h"

START_NAMESPACE_STIR


// worker functions


namespace { // priave namespace for internal functions

inline unsigned int ravel_index(int x, int y, int z,
                         int min_x, int min_y, int min_z,
                         int max_x, int max_y, int max_z) {
  unsigned int ravelled_index=
          (z-min_z)*(max_x-min_x +1)*(max_y-min_y +1)
          + (y-min_y)*(max_x-min_x +1)
          + (x-min_x);
        return ravelled_index;
      }


inline double gaussian_kernel_already_sq(double distance_sq) {
  // std::cout << "gaussian_kernel(" << distance_sq << ", " << sigma << ")" << std::endl;
  return exp(-distance_sq );
}

inline void
precalculate_patch_euclidean_distances(Array<3, float>& distance, int num_neighbours, bool only_2D,
                                       const CartesianCoordinate3D<float>& grid_spacing) {
  int min_dx, max_dx, min_dy, max_dy, min_dz, max_dz;

  if (only_2D) {
    min_dz = max_dz = 0;
  }
  else {
    min_dz = -(num_neighbours-1)/2;
    max_dz = (num_neighbours-1)/2;
  }
  min_dy = -(num_neighbours-1)/2;
  max_dy = (num_neighbours-1)/2;
  min_dx = -(num_neighbours-1)/2;
  max_dx = (num_neighbours-1)/2;

   distance =
    Array<3,float>(IndexRange3D(min_dz, max_dz, min_dy, max_dy, min_dx, max_dx));

  for (int z=min_dz; z<=max_dz; ++z) {
    for (int y=min_dy; y<=max_dy; ++y) {
      for (int x=min_dx; x<=max_dx; ++x) {
        distance[z][y][x] =
          sqrt(square(x * grid_spacing.x())
                + square(y * grid_spacing.y())
                + square(z * grid_spacing.z()))/grid_spacing.x();
      }
    }
  }
}}

template <typename TargetT>
const char * const
KOSMAPOSLReconstruction <TargetT> ::registered_name =
  "KOSMAPOSL";

//*********** parameters ***********

template <typename TargetT>
void
KOSMAPOSLReconstruction<TargetT>::
set_defaults()
{
  base_type::set_defaults();

  this->num_neighbours=3;
  this->num_non_zero_feat=1;
  this->sigma_m=1;
  this->sigma_p=1;
  this->sigma_dp=1;
  this->sigma_dm=1;
  this->only_2D = 0;
  this->kernelised_output_filename_prefix="";
  this->hybrid=0;
}

template <typename TargetT>
void
KOSMAPOSLReconstruction<TargetT>::
initialise_keymap()
{
  base_type::initialise_keymap();
  this->parser.add_start_key("KOSMAPOSLParameters");
  this->parser.add_stop_key("End KOSMAPOSLParameters");

//  this->parser.add_key("anatomical image filename",&this->anatomical_image_filenames);
  this->parser.add_key("number of neighbours",&this->num_neighbours);
  this->parser.add_key("number of non-zero feature elements",&this->num_non_zero_feat);
  this->parser.add_key("sigma_m",&this->sigma_m);
  this->parser.add_key("sigma_p",&this->sigma_p);
  this->parser.add_key("sigma_dp",&this->sigma_dp);
  this->parser.add_key("sigma_dm",&this->sigma_dm);
  this->parser.add_key("only_2D",&this->only_2D);
  this->parser.add_key("hybrid",&this->hybrid);
  this->parser.add_key("anatomical image filenames", &anatomical_image_filenames);
  this->parser.add_key("kernelised output filename prefix",&this->kernelised_output_filename_prefix);



}


template <typename TargetT>
void KOSMAPOSLReconstruction<TargetT>::
ask_parameters()
{
 OSMAPOSLReconstruction<TargetT>::ask_parameters();


}


template <typename TargetT>
bool KOSMAPOSLReconstruction<TargetT>::
post_processing()
{
  if (base_type::post_processing())
    return true;

  this->subiteration_counter=0;
  this->anatomical_sd=0;

  if (this->anatomical_image_filenames.size()>1){
      error("At the moment you can only use one anatomical image %s");
      return false;
  }


  if(!this->only_2D){
     this->num_elem_neighbourhood=this->num_neighbours*this->num_neighbours*this->num_neighbours ;}
  else{
     this->num_elem_neighbourhood=this->num_neighbours*this->num_neighbours ;
      }

  if (!this->anatomical_image_filenames.empty()){
      this->anatomical_prior_sptr= (read_from_file<TargetT>(anatomical_image_filenames[0]));

    set_anatomical_prior_sptr (this->anatomical_prior_sptr);
    info(boost::format("Reading anatomical data '%1%'")
         % anatomical_image_filenames[0]  );
  }
    if (is_null_ptr(this->anatomical_prior_sptr))
        {
            error("Failed to read anatomical file %s", anatomical_image_filenames[0].c_str());
            return false;
        }
    this->anatomical_sd=estimate_stand_dev_for_anatomical_image();

  info(boost::format("SD from anatomical image calculated = '%1%'")
       % this->anatomical_sd);

  const DiscretisedDensityOnCartesianGrid<3,float>* current_anatomical_cast =
      dynamic_cast< const DiscretisedDensityOnCartesianGrid<3,float> *>
        (this->get_anatomical_prior_sptr ().get());

  // TODO - which spacing to use? Need both?
  const CartesianCoordinate3D<float>& grid_spacing =
      current_anatomical_cast->get_grid_spacing();
  precalculate_patch_euclidean_distances(distance,num_neighbours, only_2D, grid_spacing);

    if(num_non_zero_feat>1){
      this->kpnorm_sptr = shared_ptr<TargetT>(this->anatomical_prior_sptr->get_empty_copy ());
      this->kmnorm_sptr = shared_ptr<TargetT>(this->anatomical_prior_sptr->get_empty_copy ());

      this->kpnorm_sptr->resize(IndexRange3D(0,0,0,this->num_voxels-1,0,this->num_elem_neighbourhood-1));
      this->kmnorm_sptr->resize(IndexRange3D(0,0,0,this->num_voxels-1,0,this->num_elem_neighbourhood-1));

    int dimf_col = this->num_non_zero_feat-1;
    int dimf_row=this->num_voxels;

    calculate_norm_const_matrix(*this->kmnorm_sptr,
                                dimf_row,
                                dimf_col);

    info(boost::format("Kernel from anatomical image calculated "));
    }
  return false;
}

//*********** other functions ***********



template <typename TargetT>
KOSMAPOSLReconstruction<TargetT>::
KOSMAPOSLReconstruction()
{  
  set_defaults();
}

template <typename TargetT>
KOSMAPOSLReconstruction<TargetT>::
KOSMAPOSLReconstruction(const std::string& parameter_filename)
{  
  this->initialise(parameter_filename);
  info(this->parameter_info());
}


template <typename TargetT>
Succeeded 
KOSMAPOSLReconstruction<TargetT>::
set_up(shared_ptr <TargetT > const& target_image_ptr)
{

 base_type::set_up(target_image_ptr) == Succeeded::no;

  return Succeeded::yes;
}

/***************************************************************
  get_ functions
***************************************************************/

template <typename TargetT>
const std::string
KOSMAPOSLReconstruction<TargetT>::
get_anatomical_image_filenames() const
{ return this->anatomical_image_filenames[0]; }

template <typename TargetT>
const int
KOSMAPOSLReconstruction<TargetT>::
get_num_neighbours() const
{ return this->num_neighbours; }

template <typename TargetT>
const int
KOSMAPOSLReconstruction<TargetT>::
get_num_non_zero_feat() const
{ return this->num_non_zero_feat; }

template <typename TargetT>
const double
KOSMAPOSLReconstruction<TargetT>::
get_sigma_m() const
{ return this->sigma_m; }

template <typename TargetT>
const double
KOSMAPOSLReconstruction<TargetT>::
get_sigma_p() const
{ return this->sigma_p; }

template <typename TargetT>
const double
KOSMAPOSLReconstruction<TargetT>::
get_sigma_dp() const
{ return this->sigma_dp; }

template <typename TargetT>
const double
KOSMAPOSLReconstruction<TargetT>::
get_sigma_dm() const
{ return this->sigma_dm; }

template <typename TargetT>
const bool
KOSMAPOSLReconstruction<TargetT>::
get_only_2D() const
{ return this->only_2D; }

template <typename TargetT>
const bool
KOSMAPOSLReconstruction<TargetT>::
get_hybrid() const
{ return this->hybrid; }

//template <typename TargetT >
//shared_ptr<TargetT> &KOSMAPOSLReconstruction<TargetT>::get_kpnorm_sptr()
//{ return this->kpnorm_sptr; }

//template <typename TargetT >
//shared_ptr<TargetT> &KOSMAPOSLReconstruction<TargetT>::get_kmnorm_sptr()
//{ return this->kmnorm_sptr; }

template <typename TargetT>
shared_ptr<const TargetT> KOSMAPOSLReconstruction<TargetT>::get_anatomical_prior_sptr() const
{ return this->anatomical_prior_sptr; }


/***************************************************************
  set_ functions
***************************************************************/

template<typename TargetT>
void
KOSMAPOSLReconstruction<TargetT>::
set_anatomical_prior_sptr (shared_ptr<TargetT> arg)
{
  this->anatomical_prior_sptr = arg;
}

template <typename TargetT>
void
KOSMAPOSLReconstruction<TargetT>::
set_anatomical_image_filenames(const std::string& arg)
{
  this->anatomical_image_filenames[0]  = arg;
}

template <typename TargetT>
void
KOSMAPOSLReconstruction<TargetT>::
set_num_neighbours(const int arg)
{
    this->num_neighbours = arg;
}

template <typename TargetT>
void
KOSMAPOSLReconstruction<TargetT>::
set_num_non_zero_feat(const int arg)
{
    this->num_non_zero_feat = arg;
}

template <typename TargetT>
void
KOSMAPOSLReconstruction<TargetT>::
set_sigma_m(const double arg)
{
    this->sigma_m = arg;
}

template <typename TargetT>
void
KOSMAPOSLReconstruction<TargetT>::
set_sigma_p(const double arg)
{
    this->sigma_p = arg;
}

template <typename TargetT>
void
KOSMAPOSLReconstruction<TargetT>::
set_sigma_dp(const double arg)
{
    this->sigma_dp = arg;
}

template <typename TargetT>
void
KOSMAPOSLReconstruction<TargetT>::
set_sigma_dm(const double arg)
{
    this->sigma_dm = arg;
}

template <typename TargetT>
void
KOSMAPOSLReconstruction<TargetT>::
set_only_2D(const bool arg)
{
    this->only_2D = arg;
}

template <typename TargetT>
void
KOSMAPOSLReconstruction<TargetT>::
set_hybrid(const bool arg)
{
    this->hybrid = arg;
}

/***************************************************************/
// Here start the definition of few functions that calculate the SD of the anatomical image, a norm matrix and
// finally the Kernelised image

template<typename TargetT>
void KOSMAPOSLReconstruction<TargetT>::
calculate_norm_matrix(TargetT &normp,
                      const int dimf_row,
                      const int dimf_col,
                      const TargetT& emission)
{
//  The following is the 2D matrix containing the feature vector for each voxel of the image "emission"
  Array<2,float> fp;
//  The following are the indexes obtained when reshaping a 3D matrix to a 1D vector and they depend
//  on x y and z, and dx dy and dz respectively
  int l=0,m=0;

  fp = Array<2,float>(IndexRange2D(0,dimf_row,0,dimf_col));

  const int min_z = emission.get_min_index();
  const int max_z = emission.get_max_index();
  this->dimz=max_z-min_z+1;

//The following loop extracts the feature vector related to each voxel in the "emission" image and save it in "fp"
  for (int z=min_z; z<=max_z; z++)
    {
      const int min_dz = max(distance.get_min_index(), min_z-z);
      const int max_dz = min(distance.get_max_index(), max_z-z);

      const int min_y = emission[z].get_min_index();
      const int max_y = emission[z].get_max_index();
      this->dimy=max_y-min_y+1;
      for (int y=min_y;y<= max_y;y++)
        {
          const int min_dy = max(distance[0].get_min_index(), min_y-y);
          const int max_dy = min(distance[0].get_max_index(), max_y-y);

          const int min_x = emission[z][y].get_min_index();
          const int max_x = emission[z][y].get_max_index();
          this->dimx=max_x-min_x+1;


          for (int x=min_x;x<= max_x;x++)
            {
              const int min_dx = max(distance[0][0].get_min_index(), min_x-x);
              const int max_dx = min(distance[0][0].get_max_index(), max_x-x);


              l = (z-min_z)*(max_x-min_x +1)*(max_y-min_y +1)
                + (y-min_y)*(max_x-min_x +1) + (x-min_x);

              //here a matrix with the feature vectors is created
              for (int dz=min_dz;dz<=max_dz;++dz)
                for (int dy=min_dy;dy<=max_dy;++dy)
                  for (int dx=min_dx;dx<=max_dx;++dx)
                    {
                      m = (dz)*(max_dx-min_dx +1)*(max_dy-min_dy +1)
                        + (dy)*(max_dx-min_dx +1)
                        + (dx);
                      int c = m;

                      if(m<0){
                        c = m+this->num_elem_neighbourhood ;
                      } else {
                        c=m;
                      }

                      if (   z+dz > max_z || y+dy> max_y || x+dx > max_x
                          || z+dz < min_z || y+dy< min_y || x+dx < min_x
                          || m > this->num_non_zero_feat-1 || m <0) {
                        continue;
                      }
                      else{
                        fp[l][c] = (emission[z+dz][y+dy][x+dx]) ;
                      }
                    }
            }
        }
    }

  // the norms of the difference between feature vectors related to the
  // same neighbourhood are calculated now
  int p=0,o=0;

  for (int q=0; q<=dimf_row-1; ++q){
    for (int n=-(this->num_neighbours-1)/2*(!this->only_2D);
         n<=(this->num_neighbours-1)/2*(!this->only_2D);
         ++n)
      for (int k=-(this->num_neighbours-1)/2;
           k<=(this->num_neighbours-1)/2;
           ++k)
        for (int j=-(this->num_neighbours-1)/2;
             j<=(this->num_neighbours-1)/2;
             ++j)
          for (int i=0; i<=dimf_col; ++i)
            {

              p = j
                + k*(this->num_neighbours)
                + n*(this->num_neighbours)*(this->num_neighbours)
                + (this->num_elem_neighbourhood-1)/2;

              if (q%dimx==0 && (j+k*this->dimx+n*dimx*dimy)>=(dimx-1))
                {
                  if (j+k*this->dimx+n*dimx*dimy
                      >= dimx+(this->num_neighbours-1)/2) {
                    continue;
                  }

                  o=q+j+k*this->dimx+n*dimx*dimy+1;
                }

              else{
                o=q+j+k*this->dimx+n*dimx*dimy;
              }

              if(o>=dimf_row-1 || o<0 || i<0|| i>this->num_non_zero_feat-1
                 || q>=dimf_row-1 || q<0){
                continue;
              }
              normp[0][q][p] += square(fp[q][i]-fp[o][i]);
            }
      }
}

template<typename TargetT>
void KOSMAPOSLReconstruction<TargetT>::
calculate_norm_const_matrix(TargetT &normm,
                            const int dimf_row,
                            const int dimf_col)
{

    calculate_norm_matrix(normm,dimf_row,dimf_col,*this->anatomical_prior_sptr);

}
template<typename TargetT>
double KOSMAPOSLReconstruction<TargetT>::
estimate_stand_dev_for_anatomical_image()
{
    double kmean=0;
    double kStand_dev=0;
    double dim_z=0;
    int nv=0;
    const int min_z = (*anatomical_prior_sptr).get_min_index();
    const int max_z = (*anatomical_prior_sptr).get_max_index();

     dim_z = max_z -min_z+1;

        for (int z=min_z; z<=max_z; z++)
          {

            const int min_y = (*anatomical_prior_sptr)[z].get_min_index();
            const int max_y = (*anatomical_prior_sptr)[z].get_max_index();
            double dim_y=0;

            dim_y = max_y -min_y+1;

              for (int y=min_y;y<= max_y;y++)
                {

                  const int min_x = (*anatomical_prior_sptr)[z][y].get_min_index();
                  const int max_x = (*anatomical_prior_sptr)[z][y].get_max_index();
                  double dim_x=0;

                  dim_x = max_x -min_x +1;

                   this->num_voxels = dim_z*dim_y*dim_x;

                    for (int x=min_x;x<= max_x;x++)
                    {
                        if((*anatomical_prior_sptr)[z][y][x]>=0 && (*anatomical_prior_sptr)[z][y][x]<=1000000){
                        kmean += (*anatomical_prior_sptr)[z][y][x];
                        nv+=1;}
                        else{
                            error("The anatomical image might contain nan, negatives or infinitive");
                            break;}
                    }
                }
            }
                      kmean=kmean / nv;

                      for (int z=min_z; z<=max_z; z++)
                        {


                          const int min_y = (*anatomical_prior_sptr)[z].get_min_index();
                          const int max_y = (*anatomical_prior_sptr)[z].get_max_index();

                            for (int y=min_y;y<= max_y;y++)
                              {

                                const int min_x = (*anatomical_prior_sptr)[z][y].get_min_index();
                                const int max_x = (*anatomical_prior_sptr)[z][y].get_max_index();

                                for (int x=min_x;x<= max_x;x++)
                                  {
                                    if((*anatomical_prior_sptr)[z][y][x]>=0 && (*anatomical_prior_sptr)[z][y][x]<=1000000){
                                        kStand_dev += square((*anatomical_prior_sptr)[z][y][x] - kmean);}
                                    else{continue;}
                                  }
                               }
                       }

       return sqrt(kStand_dev / (nv-1));
}

template<typename TargetT>
void KOSMAPOSLReconstruction<TargetT>::compute_kernelised_image(
                         TargetT& kernelised_image_out,
                         const TargetT& image_to_kernelise,
                         const TargetT& current_alpha_estimate)
{
    if(!current_alpha_estimate.has_same_characteristics(*this->anatomical_prior_sptr))
        error("anatomical and emission image have different sizes! Make sure they are the same");
    bool use_compact_implementation = this->num_non_zero_feat == 1;

    // Something very weird happens here if I do not get_empty_copy()
    // KImage elements will be all nan

    unique_ptr<TargetT> kImage_uptr(current_alpha_estimate.get_empty_copy());

    if (!use_compact_implementation && this->get_hybrid()) {
      // Going to need the full emission regional normalised differences
      int dimf_row = this->num_voxels;
      int dimf_col = this->num_non_zero_feat-1;
      calculate_norm_matrix(*this->kpnorm_sptr, dimf_row, dimf_col,
                          current_alpha_estimate);
    }


      //     calculate kernelised image
    int min_z, max_z, min_y, max_y, min_x, max_x;

    min_z = current_alpha_estimate.get_min_index();
    max_z = current_alpha_estimate.get_max_index();
    min_y = current_alpha_estimate[min_z].get_min_index();
    max_y = current_alpha_estimate[min_z].get_max_index();
    min_x = current_alpha_estimate[min_z][min_y].get_min_index();
    max_x = current_alpha_estimate[min_z][min_y].get_max_index();

        // Iterate over the image

        for (int z=min_z; z<=max_z; z++) {
          const int min_dz = max(distance.get_min_index(), min_z-z);
          const int max_dz = min(distance.get_max_index(), max_z-z);

          for (int y=min_y; y<= max_y; y++) {
            const int min_dy = max(distance[0].get_min_index(), min_y-y);
            const int max_dy = min(distance[0].get_max_index(), max_y-y);

            for (int x=min_x; x<= max_x; x++) {

              // std::cout << "c" << std::endl;
              const int min_dx = max(distance[0][0].get_min_index(), min_x-x);
              const int max_dx = min(distance[0][0].get_max_index(), max_x-x);

              // Iterate over the kernel patch, centered at the current voxel

              double kernel_sum = 0;
              for (int dz=min_dz; dz<=max_dz; ++dz) {
                for (int dy=min_dy; dy<=max_dy; ++dy) {
                  for (int dx=min_dx; dx<=max_dx; ++dx) {

                      const int current_ravelled_idx
                        = ravel_index(x, y, z, min_x, min_y, min_z, max_x, max_y, max_z);
                      const int delta_ravelled_idx
                        = ravel_index(dx, dy, dz, min_dx, min_dy, min_dz, max_dx, max_dy, max_dz);

//                     std::cout << "d " <<z<<" "<<y<<" "<<x<< std::endl;
                    // Calculate the emission kernel
                    double emission_kernel;
                    if (get_hybrid()) {
                      if(current_alpha_estimate[z][y][x]==0){
                        continue;
                      }

                      emission_kernel = calc_emission_kernel(current_alpha_estimate[z][y][x],
                                                   current_alpha_estimate[z+dz][y+dy][x+dx],
                                                   distance[dz][dy][dx],
                                                   use_compact_implementation,
                                                   current_ravelled_idx,
                                                   delta_ravelled_idx);
                                    }
                                    else {
                                      emission_kernel = 1;
                                    }
                    // Calculate the anatomical kernel
                       const double anatomical_kernel = calc_anatomical_kernel((*anatomical_prior_sptr)[z][y][x],
                                                                               (*anatomical_prior_sptr)[z+dz][y+dy][x+dx],
                                                                               distance[dz][dy][dx],
                                                                               use_compact_implementation,
                                                                               current_ravelled_idx,
                                                                               delta_ravelled_idx);

                       const double kernel = anatomical_kernel * emission_kernel;

                       kernelised_image_out[z][y][x]
                       += kernel * image_to_kernelise[z+dz][y+dy][x+dx];
                       kernel_sum += kernel;
                                    }
                                 }
                              }

              if (current_alpha_estimate[z][y][x] == 0) {
                continue;

              }


                      kernelised_image_out[z][y][x] /= kernel_sum;
                    }
                  }
                }
}

template <typename TargetT>
double
KOSMAPOSLReconstruction<TargetT>::
calc_emission_kernel(const double current_alpha_estimate_zyx,
                const double current_alpha_estimate_zyx_dr,
                const double distance_dzdydx,
                const bool use_compact_implementation,
                const int l,
                const int m) {

  const double emission_kernel =
    use_compact_implementation
    ? calc_kernel_compact(current_alpha_estimate_zyx-
                                    current_alpha_estimate_zyx_dr,
                                    sigma_p*sigma_p,
                                    sigma_dp*sigma_dp,
                                    distance_dzdydx*distance_dzdydx,
                                    current_alpha_estimate_zyx*current_alpha_estimate_zyx)
    : calc_kernel_from_precalculated((*kmnorm_sptr)[0][l][m],
                                    sigma_p*sigma_p,
                                    sigma_dp*sigma_dp,
                                    distance_dzdydx*distance_dzdydx,
                                    current_alpha_estimate_zyx*current_alpha_estimate_zyx);

   return emission_kernel;
}

template <typename TargetT>
double
KOSMAPOSLReconstruction<TargetT>::
calc_kernel_from_precalculated(const double precalculated_norm_zxy,
                               const double sq_sigma_int,
                               const double sq_sigma_dist,
                               const double sq_distance_dzdydx,
                               const double sq_precalc_denom) {

  const double norm_distance_sq
    = precalculated_norm_zxy/ sq_precalc_denom/sq_sigma_int+
          sq_distance_dzdydx/sq_sigma_dist/2;

  return gaussian_kernel_already_sq(norm_distance_sq);
}

template <typename TargetT>
double
KOSMAPOSLReconstruction<TargetT>::
calc_anatomical_kernel(const double anatomical_prior_zyx,
                       const double anatomical_prior_zyx_dr,
                       const double distance_dzdydx,
                       const bool use_compact_implementation,
                       const int l,
                       const int m) {

  const double anatomical_kernel =
    use_compact_implementation
    ? calc_kernel_compact(anatomical_prior_zyx-
                                    anatomical_prior_zyx_dr,
                                    sigma_m*sigma_m,
                                    sigma_dm*sigma_dm,
                                    distance_dzdydx*distance_dzdydx,
                                    anatomical_sd*anatomical_sd)
    : calc_kernel_from_precalculated((*kmnorm_sptr)[0][l][m],
                                     sigma_m*sigma_m,
                                     sigma_dm*sigma_dm,
                                     distance_dzdydx*distance_dzdydx,
                                     anatomical_sd*anatomical_sd);

  return anatomical_kernel;
}

template <typename TargetT>
double
KOSMAPOSLReconstruction<TargetT>::
calc_kernel_compact(const double prior_image_zyx_diff,
                              const double sq_sigma_int,
                              const double sq_sigma_dist,
                              const double sq_distance_dzdydx,
                              const double sq_precalc_denom) {

    const double norm_distance_sq
      = ((prior_image_zyx_diff)/sq_precalc_denom/sq_sigma_int)*
        ((prior_image_zyx_diff)/2)+
        sq_distance_dzdydx/sq_sigma_dist/2;

    return gaussian_kernel_already_sq(norm_distance_sq);
}

template <typename TargetT>
void 
KOSMAPOSLReconstruction<TargetT>::
update_estimate(TargetT &current_alpha_coefficent_image)
{
  // TODO should use something like iterator_traits to figure out the 
  // type instead of hard-wiring float
  static const float small_num = 0.000001F;
#ifndef PARALLEL
  //CPUTimer subset_timer;
  //subset_timer.start();
#else // PARALLEL
  PTimer timerSubset;
  timerSubset.Start();
#endif // PARALLEL
  
  // TODO make member parameter to avoid reallocation all the time
  unique_ptr< TargetT > multiplicative_update_image_ptr
    (current_alpha_coefficent_image.get_empty_copy());

  const int subset_num=this->get_subset_num();  
  info(boost::format("Now processing subset #: %1%") % subset_num);

  unique_ptr< TargetT > current_update_image_ptr(current_alpha_coefficent_image.get_empty_copy());
  compute_kernelised_image (*current_update_image_ptr, current_alpha_coefficent_image, current_alpha_coefficent_image);



  base_type::compute_sub_gradient_without_penalty_plus_sensitivity (*multiplicative_update_image_ptr,
                                                          *current_update_image_ptr,
                                                          subset_num); 

  //apply kernel to the multiplicative update
  unique_ptr< TargetT > kmultiplicative_update_ptr((*multiplicative_update_image_ptr).get_empty_copy());
  compute_kernelised_image (*kmultiplicative_update_ptr, *multiplicative_update_image_ptr, current_alpha_coefficent_image);

  // divide by subset sensitivity  
  {
    const TargetT& sensitivity =
      base_type::get_subset_sensitivity(subset_num);

  unique_ptr< TargetT > ksens_ptr(sensitivity.get_empty_copy());
  compute_kernelised_image (*ksens_ptr, sensitivity, current_alpha_coefficent_image);

     int count = 0;
    
    //std::cerr <<this->MAP_model << std::endl;
    
  if (this->objective_function_sptr->prior_is_zero())
    {
      divide(kmultiplicative_update_ptr->begin_all(),
             kmultiplicative_update_ptr->end_all(),
             (*ksens_ptr).begin_all(),
             small_num);
        
    }
    else
    {
      unique_ptr< TargetT > denominator_ptr
        (current_alpha_coefficent_image.get_empty_copy());
      
      
      this->objective_function_sptr->
        get_prior_ptr()->compute_gradient(*denominator_ptr, current_alpha_coefficent_image);
      
      typename TargetT::full_iterator denominator_iter = denominator_ptr->begin_all();
      const typename TargetT::full_iterator denominator_end = denominator_ptr->end_all();
      typename TargetT::const_full_iterator sensitivity_iter = (*ksens_ptr).begin_all();

      if(this->MAP_model =="additive" )
      {
        // lambda_new = lambda / (p_v + beta*prior_gradient/ num_subsets) *
        //                   sum_subset backproj(measured/forwproj(lambda))
        // with p_v = sum_{b in subset} p_bv
        // actually, we restrict 1 + beta*prior_gradient/num_subsets/p_v between .1 and 10
        while (denominator_iter != denominator_end)
          {
            *denominator_iter = *denominator_iter/this->get_num_subsets() + (*sensitivity_iter);
            // bound denominator between (*sensitivity_iter)/10 and (*sensitivity_iter)*10
            *denominator_iter =
                std::max(std::min(*denominator_iter, (*sensitivity_iter)*10),(*sensitivity_iter)/10);
            ++denominator_iter;
            ++sensitivity_iter;
          }
      }
      else
      {
        if(this->MAP_model =="multiplicative" )
        {
          // multiplicative form
          // lambda_new = lambda / (p_v*(1 + beta*prior_gradient)) *
          //                   sum_subset backproj(measured/forwproj(lambda))
          // with p_v = sum_{b in subset} p_bv
          // actually, we restrict 1 + beta*prior_gradient between .1 and 10
        while (denominator_iter != denominator_end)
          {
            *denominator_iter += 1;
            // bound denominator between 1/10 and 1*10
            // TODO code will fail if *denominator_iter is not a float
            *denominator_iter =
                std::max(std::min(*denominator_iter, 10.F),1/10.F);
            *denominator_iter *= (*sensitivity_iter);
            ++denominator_iter;
            ++sensitivity_iter;
          }
        }
      }         
      divide(kmultiplicative_update_ptr->begin_all(),
             kmultiplicative_update_ptr->end_all(),
             denominator_ptr->begin_all(),
             small_num);
    }
    
    info(boost::format("Number of (cancelled) singularities in Sensitivity division: %1%") % count);
  }
  
    
  if(this->inter_update_filter_interval>0 &&
     !is_null_ptr(this->inter_update_filter_ptr) &&
     !(this->subiteration_num%this->inter_update_filter_interval))
  {
    info("Applying inter-update filter");
    this->inter_update_filter_ptr->apply(current_alpha_coefficent_image);
  }
  
  // KT 17/08/2000 limit update
  // TODO move below thresholding?
  if (this->write_update_image && !this->_disable_output)
  {
    // allocate space for the filename assuming that
    // we never have more than 10^49 subiterations ...
    char * fname = new char[this->output_filename_prefix.size() + 60];
    sprintf(fname, "%s_update_%d", this->output_filename_prefix.c_str(), this->subiteration_num);
    
    // Write it to file
    this->output_file_format_ptr->
      write_to_file(fname, *kmultiplicative_update_ptr);
    delete[] fname;
  }
  
  if (this->subiteration_num != 1)
    {
      const float current_min =
        *std::min_element(kmultiplicative_update_ptr->begin_all(),
                          kmultiplicative_update_ptr->end_all());
      const float current_max = 
        *std::max_element(kmultiplicative_update_ptr->begin_all(),
                          kmultiplicative_update_ptr->end_all());
      const float new_min = 
        static_cast<float>(this->minimum_relative_change);
      const float new_max = 
        static_cast<float>(this->maximum_relative_change);
      info(boost::format("Update image old min,max: %1%, %2%, new min,max %3%, %4%") % current_min % current_max % (min(current_min, new_min)) % (max(current_max, new_max)));

      threshold_upper_lower(kmultiplicative_update_ptr->begin_all(),
                            kmultiplicative_update_ptr->end_all(),
                            new_min, new_max);      
    }  

  //current_alpha_coefficent_image *= *kmultiplicative_update_ptr;
  {
    typename TargetT::const_full_iterator multiplicative_update_image_iter = kmultiplicative_update_ptr->begin_all_const();
    const typename TargetT::const_full_iterator end_multiplicative_update_image_iter = kmultiplicative_update_ptr->end_all_const();
    typename TargetT::full_iterator current_alpha_coefficent_image_iter = current_alpha_coefficent_image.begin_all();
    while (multiplicative_update_image_iter!=end_multiplicative_update_image_iter) 
      { 
        *current_alpha_coefficent_image_iter *= (*multiplicative_update_image_iter);
        ++current_alpha_coefficent_image_iter; ++multiplicative_update_image_iter;
      }


    unique_ptr<TargetT> kcurrent_ptr(current_alpha_coefficent_image.get_empty_copy());

    // compute the emission image from the alpha coefficient image
    compute_kernelised_image (*kcurrent_ptr, current_alpha_coefficent_image,current_alpha_coefficent_image);

    //Write the emission image estimate:
    subiteration_counter++;
    if((subiteration_counter)%this->save_interval==0){

        char itC[10];
        sprintf (itC, "%d", subiteration_counter);
        std::string it=itC;
        std::string us="_";
        std::string k=".hv";
        this->current_kimage_filename =this->kernelised_output_filename_prefix+us+it+k;

        write_to_file(this->current_kimage_filename,*kcurrent_ptr); }
  }
  
#ifndef PARALLEL
  //cerr << "Subset : " << subset_timer.value() << "secs " <<endl;
#else // PARALLEL
  timerSubset.Stop();
  info(boost::format("Subset: %1%secs") % timerSubset.GetTime());
#endif
  
}

template class KOSMAPOSLReconstruction<DiscretisedDensity<3,float> >;
//template class KOSMAPOSLReconstruction<ParametricVoxelsOnCartesianGrid >;


END_NAMESPACE_STIR


