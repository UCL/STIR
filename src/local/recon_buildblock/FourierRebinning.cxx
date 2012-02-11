//
// $Id$
//

/*!  
  \file 
  \brief FORE kernel 
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


#include "stir/recon_buildblock/FourierRebinning.h"
#include "stir/Scanner.h"
#include "stir/ProjDataInfoCylindrical.h"
#include "stir/ProjDataInterfile.h"
#include "stir/SegmentBySinogram.h"
#include "stir/Bin.h"             
#include "stir/IndexRange3D.h"
#include "stir/IndexRange2D.h"
#include "stir/Succeeded.h"
#include "stir/round.h"
#include "stir/display.h"
#include <iostream>
#include <fstream>
#include <numeric>
#include <time.h>
#include <complex>
#include "stir/numerics/fourier.h"
#include "stir/interpolate.h"

#define POSITIVE_Z_SHIFT -1
#define NEGATIVE_Z_SHIFT 1
#define CHANGE_Z_SHIFT 2

#ifndef STIR_NO_NAMESPACES
using std::ios;
using std::ofstream;
#endif


START_NAMESPACE_STIR


const char * const 
FourierRebinning::registered_name = "FORE";
 
void
FourierRebinning::
initialise_keymap()
{
  base_type::initialise_keymap();
  parser.add_start_key("FORE Parameters");
  parser.add_stop_key("End FORE Parameters");
  parser.add_key("Smallest angular frequency", &kmin);
  parser.add_key("Smallest transaxial frequency",&wmin);
  parser.add_key("Delta max for small omega", &deltamin);
  parser.add_key("Index for consistency", &kc);
  parser.add_key("FORE debug level", &fore_debug_level);

}
 

bool 
FourierRebinning::
post_processing()
{
  if (base_type::post_processing() == true)
    return true;
  // TODO check other parameterssegment
  return false;
}  

void
FourierRebinning::
set_defaults()
{
  base_type::set_defaults();
//CON There is probably nothing like a set of FORE parameters which make sense for all scanners.
//CON Therefore default it to illegal values such that the application will terminate if the user
//CON does not set them to values which make sense via the parameter file or the set functions.   
  kmin = -1;
  wmin = -1;
  deltamin = -1;
  kc = -1;
  fore_debug_level = 0;
}

FourierRebinning::
FourierRebinning()
{
  set_defaults();
}


Succeeded
FourierRebinning::
rebin()
{

  start_timers();
  CPUTimer timer;
  timer.start();
  
  //CON return value 
  Succeeded success = Succeeded::yes;
    
  //CL Find the number of views and tangential positions power of two
  int num_views_pow2;
  for ( num_views_pow2 = 1; num_views_pow2 < 2*proj_data_sptr->get_num_views() && num_views_pow2 < (1<<15); num_views_pow2*=2);
  int num_tang_poss_pow2;
  for ( num_tang_poss_pow2 = 1; num_tang_poss_pow2 < proj_data_sptr->get_num_tangential_poss() && num_tang_poss_pow2 < (1<<15); num_tang_poss_pow2*=2);
  
  //CL Initialise the 2D Fourier transform of all rebinned sinograms P(w,k)=0
   const int num_planes = proj_data_sptr->get_proj_data_info_ptr()->get_scanner_ptr()->get_num_rings()*2-1;

  Array<3,std::complex<float> > FT_rebinned_data(IndexRange3D(0, num_planes-1, 0, num_views_pow2-1, 0, num_tang_poss_pow2-1));
  Array<3,float> Weights_for_FT_rebinned_data(IndexRange3D(0, num_planes-1, 0,num_views_pow2-1, 0,num_tang_poss_pow2-1));
  //CON some statistics
  PETCount_rebinned num_rebinned(0,0,0);

  //CON Create the output (rebinned projection data) data structure and set the properties of the rebinned sinograms
  shared_ptr<ProjData> rebinned_proj_data_sptr;
  //CON initialise the new projection data properties by copying the properties from the input projection data.
  shared_ptr<ProjDataInfo> rebinned_proj_data_info_sptr = proj_data_sptr->get_proj_data_info_ptr()->clone();
  //CON Adapt the properties that will be modified by the rebinning.
  rebinned_proj_data_info_sptr->set_num_views(num_views_pow2/2);
  //CON After rebinning we have of course only "direct" sinograms left e.q only segment 0 exists 
  rebinned_proj_data_info_sptr->reduce_segment_range(0,0);
  //CON maximal ring difference a LOR in the largest segment that is going to be rebinned 
  const int max_delta = dynamic_cast<ProjDataInfoCylindrical const&>
  (*proj_data_sptr->get_proj_data_info_ptr()).get_max_ring_difference(max_segment_num_to_process);
  //CON The maximum/minimum ring difference covered by LORs written to the rebinned sinogram changed to the maximum ring 
  //CON difference covered by the largest segment that has been rebinned.         
  dynamic_cast<ProjDataInfoCylindrical&>(*rebinned_proj_data_info_sptr).set_min_ring_difference(-max_delta, 0);
  dynamic_cast<ProjDataInfoCylindrical&>(*rebinned_proj_data_info_sptr).set_max_ring_difference(max_delta, 0);
  //CON minimal and maximal axial position number. As ususal we start with axial position 0 in segement 0   
  rebinned_proj_data_info_sptr->set_min_axial_pos_num(0, 0);
  //CON use get_num_rings to determine the maximal axial pos num. (due to comment by KT) 
  rebinned_proj_data_info_sptr->set_max_axial_pos_num(proj_data_sptr->get_num_axial_poss(0)-1,0);
  //CON create the output (interfile) file to where the rebinned data will be written. 
  rebinned_proj_data_sptr = new ProjDataInterfile (rebinned_proj_data_info_sptr,output_filename_prefix,ios::out);
  //CON get scanner related parameters needed for the rebinning kernel.
  //CON create a scanner object. The scanner type is identified from the projection data info. 
  const Scanner* scanner = rebinned_proj_data_sptr->get_proj_data_info_ptr()->get_scanner_ptr();
  const float half_distance_between_rings = scanner->get_ring_spacing()/2.; 
  const float sampling_distance_in_s = rebinned_proj_data_info_sptr->get_sampling_in_s(Bin(0,0,0,0));
  const float radial_sampling_freq_w = 2.*_PI/sampling_distance_in_s/num_tang_poss_pow2;
  //CON D = #bins * binsize, R = D / 2
  const float R_field_of_view_mm = ((int) (rebinned_proj_data_info_sptr->get_num_tangential_poss() / 2) - 1)*sampling_distance_in_s;
  const float scanner_space_between_rings = scanner->get_ring_spacing();
  const float scanner_ring_radius = scanner->get_ring_radius();
  const float ratio_ring_spacing_to_ring_radius = scanner_space_between_rings / scanner_ring_radius;

  //CON Check that the user defineable FORE parameters are inside a possible range of values
  if(fore_check_parameters(num_tang_poss_pow2,num_views_pow2,max_segment_num_to_process) != Succeeded::yes){
    cerr << "FORE Rebinning :: Setup failed " << endl;
    cerr << "FORE Rebinning :: Abort " << endl; 
   };
  
  //CON Loop over all positive segments. Negative segments (those with negative (opposite) ring differences
  //CON will be merged with the positive segment 180 degree sinograms to form a 360 degree segment.  
   for (int seg_num=0; seg_num <=max_segment_num_to_process ; seg_num++){
                   
     cout <<"FORE Rebinning :: Processing segment No " << seg_num << " *" <<endl;

     // TODO at present, the processing is done by segment. However, it's fairly easy to
     // change this to by sinogram (the rebinning call below will do everything 
     // as a loop over axial_pos anyway).
     // This would save some memory overhead, and increase potential for parallelisation later.
     // (The parallelised version of FORE in PARAPET was organised this way).

     //CON get one (positive) segment 
     SegmentBySinogram<float> segment = proj_data_sptr->get_segment_by_sinogram(seg_num);

     //CON Retrieve some segment dependent properties needed for the rebinning kernel
     const ProjDataInfoCylindrical& proj_data_info_cylindrical = dynamic_cast<const ProjDataInfoCylindrical&>(*segment.get_proj_data_info_ptr());
     const float average_ring_difference_in_segment = proj_data_info_cylindrical.get_average_ring_difference(segment.get_segment_num());
   
    
     //CL Form a 360 degree sinogram by merging two 180 degree segments  with opposite ring difference
     //CL to get a new segment sampled over 2*pi (where 0 < view < pi)
     //CON See DeFrise paper (exact and approximate rebinning algorithms for 3D PET data), Sec IV,C (p153)
     //KT TODO this is currently not a good idea, as all ProjDataInfo classes assume that
     //KT views go from 0 to Pi.
     //CON Get the corresponding (negative) segment with the same absolute but opposite obliqueness   
     const SegmentBySinogram<float> segment_neg = proj_data_sptr->get_segment_by_sinogram(-seg_num);
     //CON Expand the (positive) segment such that the two segments can be merged
     segment.grow(IndexRange3D(segment.get_min_axial_pos_num(), segment.get_max_axial_pos_num(),
            0,2*segment.get_num_views()-1, segment.get_min_tangential_pos_num(), segment.get_max_tangential_pos_num()
            ));
  
    
     //CON merge the two segments to form "360 degrees" sinograms   
     const int min_tangential_pos_num = std::max(segment_neg.get_min_tangential_pos_num(),
                                                 -segment.get_max_tangential_pos_num());


     const int max_tangential_pos_num = std::min(segment_neg.get_max_tangential_pos_num(),
                                                -segment.get_min_tangential_pos_num());


     for (int ring = segment.get_min_axial_pos_num(); ring <= segment.get_max_axial_pos_num(); ring++)
       for (int view = segment_neg.get_min_view_num(); view <= segment_neg.get_max_view_num(); view++)
         for( int tangential_pos_num = min_tangential_pos_num; tangential_pos_num<=max_tangential_pos_num; tangential_pos_num++)   
         segment[ring][view+segment_neg.get_num_views()][tangential_pos_num] = segment_neg[ring][view][-tangential_pos_num];
      
    // CON in debug mode visualize the segment	     
      if(fore_debug_level>=2)  
	{
	  char s[100];
	  sprintf(s, "(extended) segment by sinogram %d",segment.get_segment_num());
	  display(segment, segment.find_max(), s);
	}
	
    //CON the sinogramm dimensions need to have a dimension which is a power of 2 (required by the FFT algorithm) 
    //CON for s (radial coordinate) pad the sinogramm with zeros to form a larger array. 
    //CON the phi (azimuthal cordinate (view)) coordinate is periodic. The samples need to be interpolated to the
    //CON to the new matrix size. Do this by linear interpolation.             
    //CON -> DeFrise p. 153 Sec IV.C
    do_adjust_nb_views_to_pow2(segment);

    
    //CON The sinogramm data is now in the required format and ready for rebinning.       
    //CON The rebinned data is stored in a 3 dimensional array of complex numbers (FT_rebinned_data).
    //CON FT_rebinned_data[plane][w(FT of s)][k(FT of phi)] 
    //CON Weight has the same dimensions. It stores normalisation factors (floats)
    //CON to take into account the variable number of contributions to each frequency.     
    do_rebinning(FT_rebinned_data, Weights_for_FT_rebinned_data, num_rebinned, segment,
                 num_tang_poss_pow2, num_views_pow2, num_planes, average_ring_difference_in_segment,
                 half_distance_between_rings, sampling_distance_in_s, radial_sampling_freq_w, R_field_of_view_mm,
	         ratio_ring_spacing_to_ring_radius);
  
 }  //CON end loop over segments.


  //CON Some statistics 
  cout << endl << "FORE Rebinning :: Total rebinning count: " << endl;
  do_display_count(num_rebinned);

  cout << "FORE Rebinning :: Inverse FFT the rebinned sinograms " << endl;
  //CL now finally fill in the new sinogram s
  SegmentBySinogram<float> sino2D_rebinned = rebinned_proj_data_sptr->get_empty_segment_by_sinogram(0);

  
  for (int plane=FT_rebinned_data.get_min_index();plane <= FT_rebinned_data.get_max_index(); plane++){
   
   if(plane%10==0) cout << "FORE Rebinning :: Inv FFT rebinned z-position (slice) = " << plane << endl;
 
  //CON Create a temporary 2D array of complex numbers to store the rebinned and summed fourier coefficients for one slice.
  //CON This data is then inverse FFTd and copied to a sinogram data structure. 
  //CON Strictly seen this temporary data structure is no longer necessary because the inv. FFT could now be 
  //CON be done on FT_rebinned_data itself. Since one has to anyway access the full FTdata matrix to apply 
  //CON the rebinning weights
  //CON before the inv. FFT can be applied this is not much overhead and it can be left like it was done when still 
  //CON using the numerical receipies FFT code.       
   Array<2, std::complex<float> > FT_rebinned_sinogram(IndexRange2D(0,num_tang_poss_pow2-1,0,num_views_pow2/2));
  //CON fourier_for_real_data will resize the array to its appropriate dimensions
   Array<2,float> rebinned_sinogram(IndexRange2D(0,1,0,1));
 
  //CON Normalise the rebinned sinograms by applying the weight factors
  //CON See DeFrise IV.D p154.   
 for (int j = 0; j < num_tang_poss_pow2; j++) {    
   for (int i = 0; i <= num_views_pow2/2; i++) {   
     const float Actual_Weight = (Weights_for_FT_rebinned_data[plane][i][j] == 0) ? 0 : 
       1./(Weights_for_FT_rebinned_data[plane][i][j]);
     FT_rebinned_sinogram[j][i] = FT_rebinned_data[plane][i][j]* Actual_Weight;
     
   }
 }
       
 if(fore_debug_level>=3)  
    {
      char s[100];
      Array<2,float> real(FT_rebinned_sinogram.get_index_range());
      for (int i = 0; i < num_views_pow2; i++) 
	for (int j = 0; j <= num_tang_poss_pow2/2; j++) 
          real[i][j] = FT_rebinned_sinogram[i][j].real();
      sprintf(s, "real part of FT of rebinned (extended) sinogram %d",plane);
      display(real, s, real.find_max());
      for (int i = 0; i < num_views_pow2; i++) 
	for (int j = 0; j <= num_tang_poss_pow2/2; j++) 
          real[i][j] = FT_rebinned_sinogram[i][j].imag();
      sprintf(s, "imag part of FT of rebinned (extended) sinogram %d",plane);
      display(real, s, real.find_max());
    }
  
  //CON inverse FFT the rebinned sinograms  
    rebinned_sinogram = inverse_fourier_for_real_data(FT_rebinned_sinogram); 

   //CL Keep only one half of data [o.._PI]
    for (int i=0;i<(int)(num_views_pow2/2);i++) 
     for (int j=0;j<num_tang_poss_pow2;j++)
        if ((j+sino2D_rebinned.get_min_tangential_pos_num())<=sino2D_rebinned.get_max_tangential_pos_num()) 
          sino2D_rebinned[plane][i][j+sino2D_rebinned.get_min_tangential_pos_num()]=rebinned_sinogram[j][i];
           
 } //CON end loop over planes
      

    cout << "FORE Rebinning :: 2D Rebinned sinograms => Min = " << sino2D_rebinned.find_min()
    << " Max = " << sino2D_rebinned.find_max()
    << " Sum = " << sino2D_rebinned.sum() << endl;
        
   //CON finally write the rebinned sinograms to file 
    const Succeeded success_this_sino =
          rebinned_proj_data_sptr->set_segment(sino2D_rebinned);
  
      
    if (success == Succeeded::yes && success_this_sino == Succeeded::no)
                                     success = Succeeded::no;
  stop_timers();
//CON presently not very useful. Maybe one could define a vriable fore_debug_level and   
//CON only write in case of debugging
   if(fore_debug_level>0)  
    do_log_file();
      
  return success;

}



void 
FourierRebinning::
do_rebinning(Array<3,std::complex<float> > &FT_rebinned_data, Array<3,float> &Weights_for_FT_rebinned_data,
             PETCount_rebinned &count_rebinned, 
             const SegmentBySinogram<float> &segment, const int num_tang_poss_pow2,
             const int num_views_pow2, const int num_planes, const float average_ring_difference_in_segment,
             const float half_distance_between_rings, const float sampling_distance_in_s, 
             const float radial_sampling_freq_w, const float R_field_of_view_mm,
             const float ratio_ring_spacing_to_ring_radius)
 
 
 {
   const int local_rebinned = count_rebinned.total;
   const int local_miss= count_rebinned.miss;
   const int local_ssrb= count_rebinned.ssrb;

//CON Loop over all slices, FFT the sinograms and call the actual rebinning kernel.
   for (int axial_pos_num = segment.get_min_axial_pos_num(); axial_pos_num <= segment.get_max_axial_pos_num() ;axial_pos_num++)   
     {

      if(axial_pos_num%10 == 0)  cout << "FORE Rebinning z (slice) = " << axial_pos_num << endl;   
      Array<2,float> current_sinogram(IndexRange2D(0,num_tang_poss_pow2-1,0,num_views_pow2-1));
  
  //CL Calculate the 2D FFT of P(w,k) of the merged segment
  //CON copy the sinogram data of slice axial_pos_num from the segment array to slicedata
  //CON the sinogram is flipped. This will taken account for in the rebinning, where the assignment of the FFT
  //CON coefficients are assigned opposite.
     for (int j = 0; j < segment.get_num_tangential_poss(); j++) 
      for (int i = 0; i < num_views_pow2; i++) 
        current_sinogram[j][i] = segment[axial_pos_num][i][j + segment.get_min_tangential_pos_num()];
       
  //CON FFT slicedata
   const Array<2,std::complex<float> > FT_current_sinogram = fourier_for_real_data(current_sinogram);

  //CON determine the axial position of the middle of the LOR in mm relative to Bin(segment=0,view=0,axial_pos=0,tang_pos=0)  
    const ProjDataInfo& proj_data_info = *segment.get_proj_data_info_ptr();
    const float z_in_mm = proj_data_info.get_m(Bin(segment.get_segment_num(),0,axial_pos_num,0)) - proj_data_info.get_m(Bin(0,0,0,0));

  //CON Call the rebinning kernel.                                                             
    rebinning(FT_rebinned_data,Weights_for_FT_rebinned_data,count_rebinned,FT_current_sinogram,
              z_in_mm, average_ring_difference_in_segment, num_views_pow2,
              num_tang_poss_pow2,half_distance_between_rings,sampling_distance_in_s,radial_sampling_freq_w,
              R_field_of_view_mm,ratio_ring_spacing_to_ring_radius);

 }//CL End of loop of axial_pos_num
     
    if(fore_debug_level > 0){
      cout << "      Total rebinned: " <<  count_rebinned.total - local_rebinned<< endl;
      cout << "      Total missed: " << count_rebinned.miss - local_miss<< endl;
      cout << "      Total rebinned SSRB: " << count_rebinned.ssrb - local_ssrb << endl;
   }

}




void 
FourierRebinning::
rebinning(Array<3,std::complex<float> > &FT_rebinned_data, Array<3,float> &Weights_for_FT_rebinned_data,
          PETCount_rebinned &num_rebinned, const Array<2,std::complex<float> > &FT_current_sinogram,
	  const float z_in_mm, const float delta, 
          const int num_views_pow2, const int num_tang_poss_pow2, const float half_distance_between_rings, 
	  const float sampling_distance_in_s, const float radial_sampling_freq_w, const float R_field_of_view_mm, 
          const float ratio_ring_spacing_to_ring_radius)
{

 
  //CON prevent rebinning to non existing z-positions (sinograms)
  const int maxplane = FT_rebinned_data.get_max_index();
  //CON determine z position (sino identifier)
  const int z = round(z_in_mm/half_distance_between_rings);
  // TODO replace call to error() by warning() and returning Succeeded::no
  if(fabs(static_cast<float>(z_in_mm/half_distance_between_rings - z)) > .0001)
       error("FORE rebinning :: rebinning kernel expected integer z coordinate but found a non integer value %g\n", z_in_mm);
          
  //CL t is the tangent of the angle theta between the LOR and the transaxial plane
  const float   t = delta * ratio_ring_spacing_to_ring_radius / 2.F;
   

  //CON The continuous frequency "w" corresponds the radial coordinate "s"
  //CON The integer Fourier index "k" corresponds to the azimuthal angle "view"

  //CON FORE regime (rebinning)
  //CON Iterate over all frequency tuples (w,k) starting from wmin,kmin up to num_tang_poss_pow2/2,num_views_pow2/2

      for (int j = wmin; j <= num_tang_poss_pow2/2;j++) {
        for (int i = kmin; i <= num_views_pow2/2; i++) {

              float w = static_cast<float>(j) * radial_sampling_freq_w;
              float k = static_cast<float>(i);     
            
     //CON FORE consistency criteria . FORE is a good approximation only if w and k are large 
     //CON (FORE is a high frequency approximation).
     //CON For small w and k second order corrections become important.
     //CON FORE is a good approximation if:  abs(w/k) > Romega, k>klim || w>wlim. 
     //CON see DeFrise  III.C, p150.
     //CON i>kc is an additional consistency criteria. The components of a 2D FFT sinogram are 
     //CON negligible when |-k/w| is larger than the 
     //CON the radius of the scanners FOV.These contributions can be forced to zero therefore avoiding
     //CON large z-shifts which results in z-positions outside the axial range of the scanner. 
           if ((k > w * R_field_of_view_mm) && (i > kc)){           
                  continue;
                 }   

         //CON FORE establishes a relation of the 2D FT of an oblique sinogram to the 2D FT of a direct sinogram 
         //CON shifted in z by a frequency dependent value (zshift). 
	 //CON see DeFrise, III.B Formula 28
           float zshift;
           if(w==0) zshift = 0.;
           else zshift = t * k / w; //CL in mm
                      
    //CON first evaluate the positive frequency for a given bin (w,k). The FFT is stored in data such, that in the first 
    //CON two dimensions the positive frequencies are stored. Starting with the zero frequency, followed by the smallest
    //CON positive frequency and so on. The smallest negative frequency is in the last index value.
    //CON Due to the ordering of positive and negative frequencies and the fact that this is the FFT of real data
    //CON the assignment of the positive and negative frequency
    //CON contributions to the rebinned data matrix in F-space can be done here in one pass.
      for(int shift_direction=POSITIVE_Z_SHIFT; shift_direction<=NEGATIVE_Z_SHIFT; shift_direction+=CHANGE_Z_SHIFT){

              int jj = j;
         
             if(shift_direction==NEGATIVE_Z_SHIFT && j > 0)   jj = num_tang_poss_pow2 - j;
                
                //CON new_z_sl is the z-coordinate of the shifted z-position this contribution is assigned to.  	    
                const float new_z_sl = static_cast<float>(z) + shift_direction * zshift/half_distance_between_rings;       
                //CON find the nearest "real" direct sinogram located at z < new_z_sl 
                const int small_z = static_cast<int>(floor(new_z_sl));
                //CON distance between the direct sinogram and the z-position this F-space contribution was assigned to. 
                const float m = new_z_sl - small_z;
                //CON Assign the F-space contributions from the given oblique sinogram to the neighbouring direct sinogram.
                //CON The contribution is weighted by the 
                //CON z-"distance" of the calculated shifted z-position to the neighbouring direct sinogram with z < zshift.  
                if (small_z >= 0 && small_z <= maxplane) {
                       const float OneMinusM = 1.F - m;
                       FT_rebinned_data[small_z][i][jj] +=  (FT_current_sinogram[jj][i] * OneMinusM);
                       Weights_for_FT_rebinned_data[small_z][i][jj] += OneMinusM;
		       num_rebinned.total += 1; 
                } else {
                       num_rebinned.miss += 1;
                }
              //CON same for z > zshift  
                if (small_z >= -1 && small_z < maxplane) {
                    FT_rebinned_data[small_z + 1][i][jj] += (FT_current_sinogram[jj][i] * m);
                    Weights_for_FT_rebinned_data[small_z + 1][i][jj] += m;
		 } 

           }//CON shift_direction       
         }//CON end j 
       }//CON end i
 
//CL Particular cases for small frequencies i.e Small w
//CON Due to this they will only contribute to one direct sinogram.
//CON This is SSRB where each oblique sinogramm is an estimate of the direct sinogram
//CON P(w,k,z,d) = P(w,k,z,0)    
//CON Only sinograms with an obliqueness smaller than deltamin (set by parameter file) are accepted
   const int small_z = z;

  if (delta <= deltamin) {
     //CL First, treat small w's then  k=1..kNyq-1 :        
     //CON Same logic as in FORE, except that no zshift position is calculated. It is assumed that the obliqueness is small
     //CON and therefore there will be only contributions to one direct sinogram and the weights are therefore always 1. 
    
       for (int j = 0; j < wmin; j++){
         for (int i = 0; i <= num_views_pow2/2; i++) {
	 
	       for(int shift_direction=POSITIVE_Z_SHIFT;shift_direction<=NEGATIVE_Z_SHIFT;shift_direction+=CHANGE_Z_SHIFT){

                   int  jj=j;

		   // Take reverse ordering of tangential position in the negative segment into account (?)
                   if(shift_direction==NEGATIVE_Z_SHIFT && j > 0)  
                      jj=num_tang_poss_pow2 - j;                  
      
                    
                    if (small_z >= 0 && small_z <= maxplane ) {      
                 
                        FT_rebinned_data[small_z][i][jj] += FT_current_sinogram[jj][i];
                        Weights_for_FT_rebinned_data[small_z][i][jj] += 1.;  
		       if(j==1) num_rebinned.ssrb += 1; 
                   }

           } // end  shift_direction
         } // end for j
       } // end for i
      

//CL Small k :
//CL Next treat small k's and w=wNyq=(num_tang_poss_pow2 / 2)+1, k=1..klim :
       for (int j = wmin; j <= num_tang_poss_pow2/2; j++) {
         for (int i = 0; i <= kmin; i++) {
          
               for(int shift_direction=POSITIVE_Z_SHIFT;shift_direction<=NEGATIVE_Z_SHIFT;shift_direction+=CHANGE_Z_SHIFT){

                   int  jj=j;

		   // Take reverse ordering of tangential position in the negative segment into account (?)
                    if(shift_direction==NEGATIVE_Z_SHIFT && j > 0)  
                       jj=num_tang_poss_pow2 - j;  
               
                   
                    if (small_z >= 0 && small_z <= maxplane ) {            
                        FT_rebinned_data[small_z][i][jj] += FT_current_sinogram[jj][i];
                        Weights_for_FT_rebinned_data[small_z][i][jj] += 1.; 
		        num_rebinned.ssrb += 1;
                   }

           } // shift_direction
         } // end for j    
       } // end for i
  
    } // end delta < deltamin
  
 }


 
void 
FourierRebinning::
do_log_file()
{//CL Saving time details and write them to log file 
    char file[200];
    sprintf(file,"%s.log",output_filename_prefix.c_str());
    
    std::ofstream logfile(file);
 
    if (logfile.fail() || logfile.bad()) {
      warning("FORE Rebinning :: Error opening log file\n");
    }

    std::time_t now = std::time(NULL);
    logfile << "FORE Rebinning :: Date of the FORE image reconstruction : " << std::asctime(std::localtime(&now))
            << parameter_info()
            << "\n\n CPU Time :\n" << get_CPU_timer_value() << endl;
}


void 
FourierRebinning::
do_display_count(PETCount_rebinned &count)
{// Display rebinning statistics 
    cout << "FORE Rebinning :: Total rebinned: " <<  count.total << endl;
    cout << "                  Total missed: " << count.miss << endl;
                
    if (count.miss != 0)
        cout << "                  (" << 100. * count.miss / (count.miss + count.ssrb)
             << " percent)" << endl;
    cout << "FORE Rebinning :: Total rebinned SSRB: " << count.ssrb << endl;
}


void 
FourierRebinning::
do_adjust_nb_views_to_pow2(SegmentBySinogram<float> &segment) 
{
// Adjustment of the number of views to a power of two
//CON Use the STIR overlap_interpolate method and remove the simlar private implementation (adjust_pow2) here.      
  const int num_views_pow2 = stir::round(pow(2.,((int)ceil(log ((float)segment.get_num_views())/log(2.)))));
    const int offset_for_overlap_interpolate = 0;
      
    if (num_views_pow2 == segment.get_num_views()) 
        return; 

    //CON Create the projection data info ptr for the resized segment
    shared_ptr<ProjDataInfo> out_proj_data_info_sptr = segment.get_proj_data_info_ptr()->clone();
    out_proj_data_info_sptr->set_num_views(num_views_pow2);
    //CON the re-dimensioned segment      
    SegmentBySinogram<float> out_segment = 
      out_proj_data_info_sptr->get_empty_segment_by_sinogram(segment.get_segment_num());

    for (int axial_pos_num = segment.get_min_axial_pos_num(); axial_pos_num <= segment.get_max_axial_pos_num();
             axial_pos_num++) 
    {
        const Sinogram<float> sino2D = segment.get_sinogram(axial_pos_num);
        Sinogram<float> out_sino2D = out_segment.get_sinogram(axial_pos_num);
      
        float extension_factor = static_cast<float>(out_sino2D.get_num_views())/ static_cast<float>(sino2D.get_num_views());
	overlap_interpolate(out_sino2D, sino2D,extension_factor,offset_for_overlap_interpolate,true);
   
      out_segment.set_sinogram(out_sino2D);
    }
    segment = out_segment;
}

// KTTODO use warning() instead of cerr<<
Succeeded FourierRebinning::
fore_check_parameters(int num_tang_poss_pow2, int num_views_pow2, int max_segment_num_to_process){

//CON Check if the parameters given make sense.

 if(deltamin<0) {
   cerr << "FORE initialisation :: The deltamin parameter must be an integer value >= 0 " << endl;
   cerr << "                       Fix this in your FORE parameter section of your steering file " << endl;
   cerr << "                       Abort now !" << endl;
   return Succeeded::no; 

 }


 if(wmin < 0 || kmin < 0){
   cerr << "FORE initialisation :: The parameters wmin and kmin must be >=0 " << endl;
   cerr << "                       Negative frequencies are not allowed. Due to symmetry reasons they are implicitly defined by the positive wmin/kmin cut off values " << endl;
   cerr << "                       Fix this in your FORE parameter section of your steering file " << endl;
   cerr << "                       Abort now !" << endl;
   return Succeeded::no; 
 }


 if(wmin >= num_tang_poss_pow2/2 || kmin >= num_views_pow2/2) {
   cerr << "FORE initialisation :: The parameter wmin or kmin is larger than the highest frequency component computed by the FFT alogorithm " << endl;
   cerr << "                       Choose an value smaller than the largest frequency " << endl;
   cerr << "                       kmin must be smaller than " << num_tang_poss_pow2/2 << " and wmin must be smaller than " << num_views_pow2/2 << endl;
   cerr << "                       Fix this in your FORE parameter section of your steering file " << endl;
   cerr << "                       Abort now ! " << endl;
   return Succeeded::no; 
   }


 if(kc >= num_views_pow2/2) {
   cerr << "FORE initialisation :: Your parameter kc is larger than the highest frequency component in w (FTT of radial coordinate s) " << endl;
   cerr << "                       Choose an value smaller than the largest frequency " << endl;
   cerr << "                       kc must be smaller than " << num_views_pow2/2 << endl;
   cerr << "                       Fix this in your FORE parameter section of your steering file " << endl;
   cerr << "                       Abort now ! " << endl;
  return Succeeded::no; 
  } 


 if(max_segment_num_to_process > proj_data_sptr->get_num_segments()){
   cerr << "FORE initialisation :: Your data set stores " << proj_data_sptr->get_num_segments()/2+1 << " segments " << endl;
   cerr << "                       The maximum number of segments to process variable is larger than that. " << endl;
   cerr << "                       Fix this in your FORE parameter section of your steering file " << endl;
   cerr << "                       Abort now ! " << endl;
  return Succeeded::no; 

   }


 if(max_segment_num_to_process < 0){
   cerr << "FORE initialisation :: The maximum segment number to process must be an integer value >=0 " << endl;
   cerr << "                       Fix this in your FORE parameter section of your steering file " << endl;
   cerr << "                       Abort now ! " << endl;
   return Succeeded::no; 
  }       

 return Succeeded::yes; 
}


END_NAMESPACE_STIR
