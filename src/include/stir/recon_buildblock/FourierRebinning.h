//
// $Id$
//

/*! 
  \file 
  \brief Class for FORE Reconstruction 
  \ingroup recon_buildblock
  \author Claire LABBE
  \author Kris Thielemans
  \author Oliver Nix
  \author PARAPET project
  $Date$
  $Revision$
*/
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2003 - 2005, Hammersmith Imanet Ltd
    Copyright (C) 2004 - 2005 DKFZ Heidelberg, Germany
    Copyright (C) 2011-07-01 - $Date$, Kris Thielemans

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

#ifndef __stir_recon_buildblock_FourierRebinning_H__
#define __stir_recon_buildblock_FourierRebinning_H__

#include "stir/recon_buildblock/ProjDataRebinning.h"
#include "stir/RegisteredParsingObject.h"
#include <complex>


START_NAMESPACE_STIR
template <typename elemT> class SegmentByView;
template <typename elemT> class SegmentBySinogram;
//template <typename elemT> class Sinogram;
template <int num_dimensions, typename elemT> class Array;
class Succeeded;

/*
  \class PETCount_rebinned
  \brief Class for rebinned elements counter 
  \ingroup recon_buildblock
*/
class PETCount_rebinned
{
       
  public:
//! Total rebinned elements
    int total;
//! Total missed rebinned elements
    int miss;
//! Total SSRB rebinned elements
    int ssrb;

#ifdef PARALLEL
    friend PMessage& operator<<(PMessage&, PETCount_rebinned&);
    friend PMessage& operator>>(PMessage&, PETCount_rebinned&);

    PETCount_rebinned & operator+= (const PETCount_rebinned &rebin)
        {
            total += rebin.total;
            miss  += rebin.miss;
            ssrb += rebin.ssrb;
            return *this;
        }
#endif
// Default constructor by initialising all the elements conter to null
    explicit PETCount_rebinned(int total_v=0, int miss_v =0, int ssrb_v = 0)
        :total(total_v), miss(miss_v), ssrb(ssrb_v)
        {}
        
    ; 
};

/*!
  \class FourierRebinning
  \ingroup recon_buildblock
  \brief Class for Serial FORE Reconstruction.


  The digital implementation of the rebinning is done as follows:

  a) Initialise the 2D Fourier transform of all rebinned sinograms Pr(w,k);<BR>
  b) Process sequentially each pair of oblique sinograms pij and pji for i,j (= 0..2*num_rings-2) as:<BR>
	- merge pij and pji to get a sinogram sampled over 2p;<BR>
	- calculates the 2D FFT Pij(w,k) of the merged sinogram;<BR>
	- assign each frequency component (w,k) to the rebinned sinogram of the slice lying closest axially to 
	z - (tk/w): Pr(w,k) = Pr(w,k) + Pij(w,k), where r is the nearest integer to (i+j) - k(i-j)/wR;<BR>
  c) Normalise Pr(w,k) for the variable number of contributions to each w, k, r ;<BR>
  d) Calculate the 2D inverse FFT of each Pr(w,k) to get the rebinned sinogram Pr(s,f);

  As FORE is based on a high frequency approximation, it is necessary to handle separately low and high frequencies. 
  This is done by subdividing the (w,k) plane into three sub regions defined by two parameters 
  in Fourier space, w (the continuous frequency corresponding to the radial coordinates s) and k 
  (the integer Fourier index corresponding to the azimuthal angle f), and by applying in each region 
  a different method to estimated the rebinned sinogram.

  KTTODO ^A in the next sentence?

  The rebinned data are represented by  in spatial space with |s|\<=R, 0<=f\<p, |z|\<=L/2, 
  where L is the length of the axial FOV and R the ring radius.
  In the high frequencies (region 1), the rebinned data are estimated using Fourier rebinning.
  In the second high frequency region (region 2), the consistency condition is not satisfied 
  and hence all the rebinned data are forced to 0: Pr(w,k, z) = 0, when |k/w|\>=R; |w|\>wlim or |k|\>klim.
  Finally, in the low-frequency (region 3), Fourier rebinning is not applicable. 
  Therefore the rebinned data are estimated using only the oblique sinograms with 
  a small value of d : dlim. Owing to the small value of d, the axial shift can be 
  neglected as in the SSRB approximation.
*/

class FourierRebinning : public   RegisteredParsingObject<
                                  FourierRebinning,
                                  ProjDataRebinning,
                                  ProjDataRebinning>
{
 private:
  typedef ProjDataRebinning base_type;
 public:
  //! Name which will be used when parsing a ProjDataRebinning object
  static const char * const registered_name; 
 
  protected:
//! Smallest angular freq. index minimum 2 ( 1 is zero frequency)
    int kmin;                
//! Smallest transax. freq. index minimum 2 ( 1 is zero frequency)
    int wmin;
//! Delta max for small omega limiting delta for SSRB for small freq.
    int deltamin;            
//! kc index for consistency
    int kc;                  
//! fore_debug_level. Setting it to >0 will produce some debug information 
    int fore_debug_level; 
        
 public:
//! default constructor calls set_defaults();
    FourierRebinning();

//! This method returns the type of the algorithm for the rebinning
    string method_info() const
        { return("FORE"); }


//! This method creates a stack of 2D rebinned sinograms from the whole 3D data set (i.e. the ProjData data) and saves it.
   Succeeded rebin();

//! A set of get and set utility functions to access the rebinning parameters     
  inline void set_kmin(int km){kmin = km;}
  inline void set_wmin(int wm){wmin = wm;}
  inline void set_deltamin(int dm){deltamin = dm;}
  inline void set_kc(int kcc) {kc = kcc;}
  inline void set_fore_debug_level(int fdebug){fore_debug_level = fdebug;}
  
  inline int get_kmin(){return kmin;}
  inline int get_wmin(){return wmin;}
  inline int get_deltamin(){return deltamin;}
  inline int get_kc() {return kc;}
  inline int get_fore_debug_level(){return fore_debug_level;}     
 
 private:

/*! 
  \brief Fourier rebinning

  This method takes as input the 3D data set (Array3D) in Fourier space of one sinogram
  for a given delta as the data dimension are (1,fft_size,nviews_pow2), the scanner informations
  and returns the updated stack of 2D rebinned sinograms still in Fourier space,
  the updated weigthing factors as well as  the new rebinned elements counter.

 
*/
    void rebinning(Array<3,std::complex<float> > &FT_rebinned_data, Array<3,float> &Weights_for_FT_rebinned_data,
       PETCount_rebinned &num_rebinned, const Array<2,std::complex<float> > &FT_current_sinogram, const float z, 
       const float average_ring_difference_in_segment, const int num_views_pow2, const int num_tang_poss_pow2,
       const float half_distance_between_rings, const float sampling_distance_in_s, const float radial_sampling_freq_w,
       const float R_field_of_view_mm, const float ratio_ring_spacing_to_ring_radius);

/*!
  \brief This method takes as input the real 3D data set
  (in which the number of views have been extended to a number of power of 2)
  and  returns the rebinned sinograms in Fourier space, their weighting factors
  as well as the counter rebinned elements

  \b Rebinning <BR>
  Assign each frequency component (w,k) to the rebinned sinogram of the slice lying closest axially to
  z - (tk/w) with t=((ring0 -ring1)*ring_spacing/(2*R) with R=ring_radius, 
  Pm(w,k) = Pm(w,k) + Pij(w,k) (i=ring0 and j=ring1), and m is the nearest integer to (i+j) -k(i-j)/(Rw)).
*/

    void do_rebinning(Array<3,std::complex<float> > &FT_rebinned_data, Array<3,float> &Weights_for_FT_rebinned_data,
                      PETCount_rebinned &count_rebinned, const SegmentBySinogram<float> &segment, const int num_tang_poss_pow2,
                      const int num_views_pow2, const int num_planes, const float average_ring_difference_in_segment,
                      const float half_distance_between_rings, const float sampling_distance_in_s, 
                      const float radial_sampling_freq_w, const float R_field_of_view_mm,
                      const float ratio_ring_spacing_to_ring_radius);

/*!
  This method takes as input the information of the 3D data set (ProjData)
  as well as the reconstruction parameters and prints out the informations of reconstruction
  parameters implemented in FORE as well as the CPU and relative timing of differents process
  (i.e I/O, rebinning, backprojection, matrix handling).
  fore_debug_level must be set to a integer value > 0
*/
    void do_log_file();

//! This is a function to display the current counter of all rebinned elements 
    void do_display_count(PETCount_rebinned &num_rebinned_total);


//! This is a function to adjust the number of views of a segment to the next power of 2
    void do_adjust_nb_views_to_pow2(SegmentBySinogram<float> &segment) ;

//! This function checks if the steering and input paramters for FORE are inside the possible range of parameters
    Succeeded fore_check_parameters(int num_tang_poss_pow2, int num_views_pow2, int max_segment_num_to_process);

    
 protected:
  virtual bool post_processing();  
  // TODO virtual void set_defaults();
  virtual void initialise_keymap();

};

END_NAMESPACE_STIR
#endif
