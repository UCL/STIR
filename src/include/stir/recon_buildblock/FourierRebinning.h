//
// $Id$
//

/*! 
  \file 
  \brief Class for FORE Reconstruction 
  \ingroup recon_buildblock
  \author Claire LABBE
  \author Kris Thielemans
  \author PARAPET project
  $Date$
  $Revision$
*/
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- $Date$, Hammersmith Imanet
    See STIR/LICENSE.txt for details
*/

#ifndef __stir_FORE_FourierRebinning_H__
#define __stir_FORE_FourierRebinning_H__


#include "local/stir/recon_buildblock/ProjDataRebinning.h"
#include "stir/RegisteredParsingObject.h"


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
  This is done by subdividing the (w,k) plane into three sub regions defining by two parameters 
  in Fourier space, w (the continuous frequency corresponding to the radial coordinates s) and k 
  (the integer Fourier index corresponding to the azimuthal angle f), and by applying in each region 
  a different method to estimated the rebinned sinogram.

  The rebinned data are represented by  in spatial space with |s|<=R, 0<=f<p, |z|<=L/2, 
  where L is the length of the axial FOV and R the ring radius.
  In the high frequencies (region 1), the rebinned data are estimated using Fourier rebinning.
  In the second high frequency region (region 2), the consistency condition is not satisfied 
  and hence all the rebinned data are forced to 0: Pr(w,k, z) = 0, when |k/w|>=R; |w|>wlim or |k|>klim.
  Finally, in the low-frequency (region 3), Fourier rebinning is not applicable. 
  Therefore the rebinned data are estimated using only the oblique sinograms with 
  a small value of d : dlim. Owing to the small value of d, the axial shift can be 
  neglected as in the SSRB approximation.
*/
class FourierRebinning 
: 
public 
  RegisteredParsingObject<
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
//CL10/03/00 Remove these parameters as not useful
        //   int LUT;                 

//    int store;               
    
 public:
    //! default constructor calls set_defaults();
    FourierRebinning();

//! This method returns the type of the algorithm for the rebinning
    string method_info() const
        { return("FORE"); }


    //! This method creates a stack of 2D rebinned sinograms from the whole 3D data set (i.e. the ProjData data) and saves it.
    Succeeded rebin();
 private:
/*! 
  \brief Fourier rebinning

  This method takes as input the 3D data set (Array3D) in Fourier space of one sinogram
  for a given delta as the data dimension are (1,fft_size,nviews_pow2), the scanner informations
  and returns the updated stack of 2D rebinned sinograms still in Fourier space,
  the updated weigthing factors as well as  the new rebinned elements counter.

  \param  data  3D float array containing the 3D data in which the number of views is equal to the next power of two. The dimensions of this array are (1,fft_size,nviews_pow2)
  \param FTdata	(out) 3D float array containing the stack of 2D rebinned data in Fourier space. The dimensions of this array are (2*num_rings()-1, nviews_pow2, fft_size);
  \param Weights (out) weight values in Fourier space or normalisation factors which represents the variable number of contributions for each w, k, r.  The dimensions of this array are  (2*num_rings()-1, nviews_pow2/2, fft_size)
  \param z  Plane number to be processsed
  \param delta	 Ring difference number to be processed
  \param scan_info  Scanner informations
  \param num_rebinned  Counter for the number of total, missing or SSRB elements (see more details in section II.8)
*/
    void rebinning(const Array<3,float> &data, 
		   Array<3,float> &FTdata, Array<3,float> &Weights,
                   float z, float delta,  const ProjDataInfo &, 
		   PETCount_rebinned &num_rebinned);

/*!
  \brief This method takes as input the real 3D data set
  (in which the number of views have been extended  to a number of power of 2)
  and  returns the rebinned sinograms in Fourier space, their weighting factors
  as well as the counter rebinned elements

  \b 3D \b FFT <BR>
  The FFT for plane=plane is now in data and speq. Speq does only contain components at the Nyquist frequency, and
  these are set to zero in the filter function anyway. The remaining values are stored in data in wrap-around order,
  see Num Rec in C page 527.<BR>
  Now let us rebin these data into the direct FTed sino array FTdata[.][.][plane] :<BR>
  Positive w is the second index of data from 1 to fft_size/2+1; Positive k is the third index of data from 1 to nviewdpot-1,
  in steps of two for the real part, the corresponding imaginary part being stored with indices 2 to nviews_pow2.

  \b Rebinning <BR>
  Assign each frequency component (w,k) to the rebinned sinogram of the slice lying closest axially to
  z - (tk/w) with t=((ring0 -ring1)*ring_spacing/(2*R) with R=ring_radius, 
  Pm(w,k) = Pm(w,k) + Pij(w,k) (i=ring0 and j=ring1), and m is the nearest integer to (i+j) -k(i-j)/(Rw)).
*/
    void do_rebinning( SegmentBySinogram<float>&segment,
                       PETCount_rebinned &num_rebinned_total,
                       Array<3,float> &FTdata,
                       Array<3,float> &weight);

//! This method takes as input the segment or the 3D sinograms and returns the new segment on where views have been added by num_views_to_add. This method is general used fro direct planes.
    //    void do_mashing(SegmentBySinogram<float> &direct_sinos);
    
/*!
  This method takes as input the information of the 3D data set (ProjData)
  as well as the reconstruction parameters and prints out the informations of reconstruction
  parameters implemented in FORE as well as the CPU and relative timing of differents process
  (i.e I/O, rebinning, backprojection, matrix handling).
*/
    void do_log_file();


//CL10/03/00 Remove these lines as not useful
#if 0
//  This is a function to save on fly the look-up-table
void do_LUT(char lutfilename[80], Array<3,float> &weight);

//  This is a function to store the look-up-table (weighting factors)
void do_storeLUT(char lutfilename[80], Array<3,float> &weight);
#endif

//! This is a function to display the current counter of all rebinned elements 
void do_display_count(PETCount_rebinned &num_rebinned_total);


//! This is a function to adjust the number of views of a segment to the next power of 2
void do_adjust_nb_views_to_pow2(SegmentBySinogram<float> &segment) ;

 protected:
  virtual bool post_processing();  
  // TODO virtual void set_defaults();
  virtual void initialise_keymap();

};

END_NAMESPACE_STIR
#endif
