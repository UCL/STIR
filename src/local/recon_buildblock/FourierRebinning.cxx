//
// $Id$
//

/*!  
  \file 
  \brief FORE kernel 
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
#include <complex.h>
#include "stir/numerics/fourier.h"


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

}


bool 
FourierRebinning::
post_processing()
{
  if (base_type::post_processing() == true)
    return true;
  // TODO check other parameters
  return false;
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
  //CL Find the number of views and tangential positions power of two
  int num_views_pow2;
  for ( num_views_pow2 = 1; num_views_pow2 < 2*proj_data_sptr->get_num_views() && num_views_pow2 < (1<<15); num_views_pow2*=2);
  int num_tang_poss_pow2;
  for ( num_tang_poss_pow2 = 1; num_tang_poss_pow2 < proj_data_sptr->get_num_tangential_poss() && num_tang_poss_pow2 < (1<<15); num_tang_poss_pow2*=2);
  
  //CL Initialise the 2D Fourier transform of all rebinned sinograms P(w,k)=0
  const int num_planes = proj_data_sptr->get_proj_data_info_ptr()->get_scanner_ptr()->get_num_rings()*2-1;
  Array<3,std::complex<float> > FTdata(IndexRange3D(0, num_planes-1, 0, num_views_pow2-1, 0, num_tang_poss_pow2-1));
  Array<3,float> weight(IndexRange3D(0, num_planes-1, 0,(int)(num_views_pow2-1), 0,num_tang_poss_pow2-1));
  
  //CON some statistics
  PETCount_rebinned num_rebinned(0,0,0);// Initialisation of number of counts for rebinning

  // CON Create the output data structure and set the properties of the rebinned sinograms
  shared_ptr<ProjData> rebinned_proj_data_sptr;
  const int max_delta = dynamic_cast<ProjDataInfoCylindrical const&>
  (*proj_data_sptr->get_proj_data_info_ptr()).get_max_ring_difference(max_segment_num_to_process);
  //CON initialise the new sinogram properties by copying the properties from the original one and adjusting the data content 
  //CON (properties) that have been changed by the rebinning 
  shared_ptr<ProjDataInfo> rebinned_proj_data_info_sptr = proj_data_sptr->get_proj_data_info_ptr()->clone();
  rebinned_proj_data_info_sptr-> set_num_views(num_views_pow2/2);
  //CON After rebinning we have of course only "direct" sinograms left e.q only segment 0 remains 
  rebinned_proj_data_info_sptr-> reduce_segment_range(0,0);
  //CON The maximum/minimum ring difference covered by LORs written to the rebinned sinogram changed to the maximal ring 
  //CON difference covered by the largest segment that has been rebinned.        
  dynamic_cast<ProjDataInfoCylindrical&>(*rebinned_proj_data_info_sptr).set_min_ring_difference(-max_delta, 0);
  dynamic_cast<ProjDataInfoCylindrical&>(*rebinned_proj_data_info_sptr).set_max_ring_difference(max_delta, 0);
  //CON minimal and maximal axial position number. As ususal we start with axial position 0 in segement 0   
  rebinned_proj_data_info_sptr->set_min_axial_pos_num(0, 0);
  //CON use get_num_rings to determine the maximal axial pos num. (due to comment by KT) 
  //CON CL used "proj_data_sptr->get_proj_data_info_ptr()->get_scanner_ptr()->get_num_rings()*2"
  rebinned_proj_data_info_sptr->set_max_axial_pos_num(proj_data_sptr->get_num_axial_poss(0)-1,0);
  rebinned_proj_data_sptr = new ProjDataInterfile (rebinned_proj_data_info_sptr,output_filename_prefix,ios::out);
  //CON get data and segment independent scanner related parameters needed for the rebinning kernel.
  const Scanner* scanner = rebinned_proj_data_sptr->get_proj_data_info_ptr()->get_scanner_ptr();
  const float half_distance_between_rings = scanner->get_ring_spacing()/2.; 
  const float sampling_distance_in_s = rebinned_proj_data_info_sptr->get_sampling_in_s(Bin(0,0,0,0));
  const float radial_sampling_freq_w = 2.*_PI/sampling_distance_in_s/num_tang_poss_pow2;
  const float R_field_of_view_mm = ((int) (rebinned_proj_data_info_sptr->get_num_tangential_poss() / 2) - 1)*sampling_distance_in_s;
  const float scanner_space_between_rings = scanner->get_ring_spacing();
  const float scanner_ring_radius = scanner->get_ring_radius();
  const float ratio_ring_spacing_to_ring_radius = scanner_space_between_rings / scanner_ring_radius;

  //CON Check the validity of the user defineable FORE parameters
  fore_check_parameters(num_tang_poss_pow2,num_views_pow2,max_segment_num_to_process);
  
  
  //CON Loop over all positive segments. Negative segments (those with negative (opposite) ring differences
  //CON will be merged with the positive segment 180 degree sinograms to form a 360 degree segment.  
  for (int seg_num=0; seg_num <=max_segment_num_to_process ; seg_num++){
           
     cout <<"  * Processing segment No " << seg_num << " *" <<endl;
     SegmentBySinogram<float> segment = proj_data_sptr->get_segment_by_sinogram(seg_num);

     //CON Retrieve some segment dependent properties needed for the rebinning kernel
     const ProjDataInfoCylindrical& proj_data_info_cylindrical = dynamic_cast<const ProjDataInfoCylindrical&>(*segment.get_proj_data_info_ptr());
     const float average_ring_difference_in_segment = proj_data_info_cylindrical.get_average_ring_difference(segment.get_segment_num());
   
    
     //CL Form a 360 degree sinogram by merging two 180 degree segments  with opposite ring difference
     //CL to get a new segment sampled over 2*pi (where 0 < view < pi)
     //CON As described in the DeFrise paper, Sec IV,C (p153)
     //KT TODO this is currently not a good idea, as all ProjDataInfo classes assume that
     //KT views go from 0 to Pi.
     const SegmentBySinogram<float> segment_neg = proj_data_sptr->get_segment_by_sinogram(-seg_num);

     segment.grow(IndexRange3D(segment.get_min_axial_pos_num(), segment.get_max_axial_pos_num(),
            0,2*segment.get_num_views()-1, segment.get_min_tangential_pos_num(), segment.get_max_tangential_pos_num()
            ));
         
      for (int ring = segment.get_min_axial_pos_num(); ring <= segment.get_max_axial_pos_num(); ring++){
        for (int view = segment_neg.get_min_view_num(); view <= segment_neg.get_max_view_num(); view++){
            segment[ring][view+segment_neg.get_num_views()] = segment_neg[ring][view];
        }
      }

    //CON the sinogramm dimensions need to have a dimension which is a power of 2 (requirement of the FTT) 
    //CON for s (radial coordinate) pad the sinogramm with zeros to form a larger array. 
    //CON the phi (azimuthal cordinate (view)) coordinate is periodic. The samples need to be interpolated to the
    //CON to the new matrix size. Do this by linear interpolation.             
    do_adjust_nb_views_to_pow2(segment);

    //CON The sinogramm data has now in the required format and is ready to be transformed into frequency space in respect to s and phi.       
    //CON FTdata holds the rebinned sinogram data in the Fourier domain w (FT of s) and k (FT of phi). weight holds the normalisation 
    //CON as obtained by application of the rebinning algorithm in fourier space to unit data P(w,k)=1.     
    do_rebinning(segment, num_tang_poss_pow2, num_views_pow2, num_planes, average_ring_difference_in_segment,
                 half_distance_between_rings, sampling_distance_in_s, radial_sampling_freq_w, R_field_of_view_mm, ratio_ring_spacing_to_ring_radius, num_rebinned, FTdata, weight);
  
 }  //CON end loop over segments.


  //CON Some statistics 
  cout << endl << "  - Total rebinning count: " << endl;
  do_display_count(num_rebinned);
  
  cout << "Inverse FFT the rebinned data " << endl;
  
  //CL now finally fill in the new sinogram s
  SegmentBySinogram<float> sino2D_rebinned = rebinned_proj_data_sptr->get_empty_segment_by_sinogram(0);
   
  for (int plane=FTdata.get_min_index();plane <= FTdata.get_max_index(); plane++){
   
   cout << "Inv FFT rebinned z-position (slice) = " << plane << endl;
 
  //CON Create a temporary complex matrix to store the rebinned and weighted data in F-space of one slice.
  //CON This data is then inverse FFTd and copied to a sinogram data structure. 
  //CON Strictly seen this temporary data structure is no longer necessary because the inv. FFT could now be 
  //CON be done on FTdata itself. But unfortunatly one has to anyway access the full FTdata matrix before the inv. FFT to
  //CON apply the rebinning weights. Therefore it is not much overhead to do this in the "NR" style.       
   Array<2, std::complex<float> > data(IndexRange2D(0,num_tang_poss_pow2-1,0,num_views_pow2-1));
   
  //CON recreate a data and speq Array into which we copy the FORE Rebinned Sinogram still in F-Space of a given plane.
  //CON We need these structures for the inverse FTT. Additionally we normalise the FTT data to take into account the variable
  //CON number of contributions to each frequency by dividing through the weight determined by applying FORE to unit data.
  //CON See DeFrise IV.D p154.       
     for (int j = 0; j < num_tang_poss_pow2; j++) {
       for (int i = 0; i < num_views_pow2; i++) {         
           float Weight = (weight[plane][i][j] == 0) ? 0 :  1./ (weight[plane][i][j]);
                 data[i][j] = FTdata[plane][i][j] * Weight;
       }
      }

  //CON Amplitude of the zero frequency. Global norm factor     
    float norm = data[0][0].real();

    inverse_fourier(data);
 
  //CON Normalisation 
    data = data * (1./(num_tang_poss_pow2*num_views_pow2) );
    if(data.sum().real() != 0) {
       data = data * (norm/data.sum().real());
     } else {
       cerr << "FORE rebinning::   Inv FFT: Sum of plane elements for normalisation is 0 !!! " << endl;
       cerr << "                   Plane will not be inv. FFT " << endl;
     }

            
    //CL Keep only one half of data [o.._PI]
    for (int i=0;i<(int)(num_views_pow2/2);i++) {
      for (int j=0;j<num_tang_poss_pow2;j++){                
        if ((j+sino2D_rebinned.get_min_tangential_pos_num())<=sino2D_rebinned.get_max_tangential_pos_num()) {
               sino2D_rebinned[plane][i][j+sino2D_rebinned.get_min_tangential_pos_num()]=data[j][i].real();
        }
      }
    }
  } // end plane
      

  //CL One more normalization for adjusting the number of views to the original ones
    const float AdjustNumberOfViewNormFactor =((float) proj_data_sptr->get_proj_data_info_ptr()->get_num_views()/(float) sino2D_rebinned.get_num_views());
   
    sino2D_rebinned/= AdjustNumberOfViewNormFactor;
  

    cout << "    2D Rebinned sinograms => Min = " << sino2D_rebinned.find_min()
    << " Max = " << sino2D_rebinned.find_max()
    << " Sum = " << sino2D_rebinned.sum() << endl;

    Succeeded success = Succeeded::yes;
    
    const Succeeded success_this_sino =
          rebinned_proj_data_sptr->set_segment(sino2D_rebinned);
  
      
    if (success == Succeeded::yes && success_this_sino == Succeeded::no)
                                     success = Succeeded::no;
  

  stop_timers();
  do_log_file();
      
  return success;

}



void 
FourierRebinning::
do_rebinning(SegmentBySinogram<float> &segment, int &num_tang_poss_pow2,
       int &num_views_pow2, const int &num_planes, const float &average_ring_difference_in_segment,
       const float &half_distance_between_rings, const float &sampling_distance_in_s, const float &radial_sampling_freq_w, const float &R_field_of_view_mm,
       const float &ratio_ring_spacing_to_ring_radius, PETCount_rebinned &count_rebinned, Array<3,std::complex<float> > &FTdata,
       Array<3,float> &weight)
{
    
                      
    int local_rebinned=0, local_ssrb=0, local_miss=0;
    local_rebinned = count_rebinned.total;
    local_miss= count_rebinned.miss;
    local_ssrb= count_rebinned.ssrb;


for (int axial_pos_num = segment.get_min_axial_pos_num(); axial_pos_num <= segment.get_max_axial_pos_num() ;axial_pos_num++)   {

      cout << "Rebinning z (slice) = " << axial_pos_num << endl;   
      Array<2,std::complex<float> > data(IndexRange2D(0,(int)num_tang_poss_pow2-1,0,(int)num_views_pow2-1));
        
   //CL Calculate the 2D FFT of P(w,k) of the merged segment
   //CON fill data with the sinogram of slice axial_pos_num   
         for (int j = 0; j < segment.get_num_tangential_poss(); j++) {
            for (int i = 0; i < num_views_pow2; i++) {
             data[j][i] = segment[axial_pos_num][i][j + segment.get_min_tangential_pos_num()];
                                                                                                  
           }      
         }
              

    fourier(data);


  //CON determine the axial position of the middle of the LOR in mm relative to Bin(segment=0,view=0,axial_pos=0,tang_pos=0)  
    const ProjDataInfo& proj_data_info = *segment.get_proj_data_info_ptr();
    const float z_in_mm = proj_data_info.get_m(Bin(segment.get_segment_num(),0,axial_pos_num,0)) - proj_data_info.get_m(Bin(0,0,0,0));
                                 
    rebinning(data,FTdata,weight,z_in_mm, average_ring_difference_in_segment, num_views_pow2,
              num_tang_poss_pow2,half_distance_between_rings,sampling_distance_in_s,radial_sampling_freq_w,
              R_field_of_view_mm,ratio_ring_spacing_to_ring_radius,count_rebinned);
   
             

 }//CL End of loop of axial_pos_num
    
        cout << "      Total rebinned: " <<  count_rebinned.total -local_rebinned<< endl;
        cout << "      Total missed: " << count_rebinned.miss - local_miss<< endl;
        cout << "      Total rebinned SSRB: " << count_rebinned.ssrb -local_ssrb << endl;
  
}




void 
FourierRebinning::
rebinning(const Array<2,std::complex<float> > &data, Array<3,std::complex<float> > &FTdata,
          Array<3,float> &Weights, float z_in_mm, float delta, int &num_views_pow2,
          int &num_tang_poss_pow2, const float &half_distance_between_rings, const float &sampling_distance_in_s,
          const float &radial_sampling_freq_w, const float &R_field_of_view_mm, const float &ratio_ring_spacing_to_ring_radius,
          PETCount_rebinned &num_rebinned)
{

  //CON prevent rebinning to non existing z-positions (sinograms)
  const int     maxplane = FTdata.get_max_index();
  //CON determine z position (sino identifier)
  const int z = round(z_in_mm/half_distance_between_rings);
    if(fabs(z_in_mm/half_distance_between_rings - z > .0001))
       error("FORE rebinning :: rebinning kernel expected integer z coordinate but found a non integer value %g\n", z_in_mm);
        
//CL t is the tangent of the angle theta between the LOR and the transaxial plane
  const float   t = delta*ratio_ring_spacing_to_ring_radius/2.;
   

// The continuous frequency "w" corresponds the radial coordinate "s"
// The integer Fourier index "k" corresponds to the azimuthal angle "view"


//CON FORE
//CON Loop over all frequency tuples (w,k) from wmin,kmin 
     for (int j = wmin; j <= num_tang_poss_pow2/2;j++) {
      for (int i = kmin; i <= num_views_pow2/2; i++) {

              float w = (j) * radial_sampling_freq_w;
              float k = (float) (i);     
            
           //CON FORE consistency criteria . FORE is a good approximation only if w and k are large (FORE is a high frequency approximation).
     //CON For small w and k second order corrections become important.
     //CON FORE is a good approximation if:  abs(w/k) > Romega, k>klim || w>wlim. 
     //CON DeFrise  III.C, p150.
     //CON i>kc is an additional consistency criteria. The components of a 2D FFT sinogram are negligible when |-k/w| is larger than the 
     //CON the radius of the scanners FOV.These contributions can be forced to zero therefore avoiding large z-shifts which results in
     //CON z-positions outside the axial range of the scanner. 
      if ((k > w * R_field_of_view_mm) && (i > kc)){           
               continue;
              }   

              
           //CON FORE links relates the 2D FT of an oblique sinogram to the 2D FT of a direct sinigram 
           //CON shifted in z by an frequency dependent offset which is calculated here. DeFrise, III.B Formula 28
            float zshift;
             if(w==0) zshift = 0.;
             else zshift = t * k / w; //CL in mm

             
               int jj = j;       
               int ii = i;

    //CON first evaluate the positive frequency for a given bin (w,k). The FT is stored in data such, that in the first 
    //CON two dimensions the positive frequencies are stored. Starting with the zero frequency, followed by the smallest
    //CON positive frequency and so on. The smallest negative frequency is in the last index value.
    //CON The sign 2 loops forces that the F-space symmetry in i (k) is fulfilled;
    //CON sign loop. Due to the ordering of positive and negative frequencies the assignment of the positive and negative frequency
    //CON contributions to the rebinned data matrix in F-space can be done here in one pass.
    //CON Depending on sign jj addresses either the negative of positive frequencie component.     	   
          for(int sign=-1;sign<=1;sign+=2){
            for(int sign2=-1;sign2<=1;sign2+=2){
              jj=j;ii=i;
                
           if(sign==1 && j > 0){ 
              jj = num_tang_poss_pow2  - j;
              }
           if(sign2==1 && i > 0){ 
              ii = num_views_pow2 - i;
           }

               //CON new_z_sl is the z-coordinate of the shifted z-position this contribution is assigned to.  	    
                float new_z_sl = (float) z + (float) sign * zshift / half_distance_between_rings;
               //CON find the nearest "real" direct sinogram located at z < new_z_sl 
                int small_z = (int) floor(new_z_sl);
               //CON distance between the direct sinogram and the z-position this F-space contribution was assigned to. 
                float m =  new_z_sl - small_z;

               //CON Assign the F-space contributions from the given oblique sinogram to the neighbouring direct sinogram. The contribution is weighted by the 
               //CON z-"distance" of the calculated shifted z-position to the neighbouring direct sinogram with z < zshift.  
               if (small_z >= 0 && small_z <= maxplane) {
                      float OneMinusM = 1.-m;
                      FTdata[small_z][jj][ii] +=  (data[jj][ii] * OneMinusM);
                      Weights[small_z][jj][ii] += OneMinusM;
                      num_rebinned.total += 1;                                                                                                      
             } else {
                      num_rebinned.miss += 1;
             }
              //CON same for z > zshift  
               if (small_z >= -1 && small_z < maxplane) {
                  FTdata[small_z + 1][jj][ii] += (data[jj][ii]* m);
                  Weights[small_z + 1][jj][ii] += m;
                } 

             } //CON end sign2
           }//CON end sign         
         }//CON end i 
       }//CON end j
   

//CL Particular cases for small frequencies i.e Small w
//CON Only sinograms with a small obliqueness
//CON Due to the small obliqueness they will only contribute to one direct sinogram.
//CON This is SSRB where each oblique sinogramm is an estimate of the direct sinogram
//CON P(w,k,z,d) = P(w,k,z,0)    
 int small_z = z;
     
     if (delta <= deltamin) {
      //CL First, treat small w's then  k=1..kNyq-1 :
              
     //CON Same logic as in FORE, except that no zshift position is calculated. It is assumed that z(oblique)=z(direct)
     //CON Therefore all frequencies contribute only to one direct sinogram and all weights are 1. 	      
        for (int j = 0; j < wmin; j++){
          for (int i = 0; i <= num_views_pow2/2; i++) {


               int jj=j;
               int ii=i;
                                                                                                 
                for(int sign=-1;sign<=1;sign+=2){
                   for(int sign2=-1;sign2<=1;sign2+=2){
                    jj=j; ii=i;
    
                    if(sign==1 && j > 0)  
                           jj=num_tang_poss_pow2 - j;
                    if(sign2==1 && i > 0)  
                           ii=num_views_pow2 -i;                      

                    
                    if (small_z >= 0 && small_z <= maxplane ) {      
                       FTdata[small_z][jj][ii] += data[jj][ii];
                       Weights[small_z][jj][ii] += 1.; 
                       if(j==1) num_rebinned.ssrb += 1; 
                   }
                    
               } // end sign
              } // end sign2 
             } // end for i
            } // end for j
      

//CL Small k :
//CL Next treat small k's and w=wNyq=(num_tang_poss_pow2 / 2)+1, k=1..klim :
    
        for (int j = wmin; j <= num_tang_poss_pow2/2; j++) {
            for (int i = 0; i < kmin; i++) {


             int jj=j;
             int ii=i;
        
                for(int sign=-1;sign<=1;sign+=2){
                   for(int sign2=-1;sign2<=1;sign2+=2){
                     jj=j; ii=i;

                     if(sign==1 && j > 0) 
                        jj = num_tang_poss_pow2 - j;
                     if(sign2==1 && i > 0)
                        ii = num_views_pow2 - i;
                       
                   
                    if (small_z >= 0 && small_z <= maxplane ) {            
                       FTdata[small_z][jj][ii] += data[jj][ii];
                       Weights[small_z][jj][ii] += 1.;
                       num_rebinned.ssrb += 1;
                   }
                 
             } // end sign 2
            } // end sign
           } // end for i    
          } // end for j
  
    } // end delta < deltamin

 }


 
void 
FourierRebinning::
do_log_file( )
{//CL Saving time details and write them to log file 
    char file[200];
    sprintf(file,"%s.log",output_filename_prefix.c_str());
    
    ofstream logfile(file);
 
    if (logfile.fail() || logfile.bad()) {
      warning("Error opening log file\n");
    }

    time_t now = time(NULL);
    logfile << "Date of the FORE image reconstruction : " << asctime(localtime(&now))
            << parameter_info()
            << "\n\n CPU Time :\n" << get_CPU_timer_value() << endl;
       
}



void 
FourierRebinning::
do_display_count(PETCount_rebinned &count)
{// Display the number of rebinning 
    cout << "    Total rebinned: " <<  count.total << endl;
    cout << "    Total missed: " << count.miss << endl;
                
    if (count.miss != 0)
        cout << "    (" << 100. * count.miss / (count.miss + count.ssrb)
             << " percent)" << endl;
    cout << "    Total rebinned SSRB: " << count.ssrb << endl;
}




//CL This is a function which adapt the number of views of a sinogram  to a power of two
static void adjust_pow2 (Sinogram<float>& out_sino2D, const Sinogram<float>& sino2D);

void 
FourierRebinning::
do_adjust_nb_views_to_pow2(SegmentBySinogram<float> &segment) 
{// Adjustment of the number of views to a power of two
    cout << "    - Adjustment of the number of views to a power of two... " << endl;
    const int  num_views_pow2 = (int) pow(2.,((int)ceil(log ((float)segment.get_num_views())/log(2.))));
      
    if (num_views_pow2 == segment.get_num_views()) 
        return; 

    shared_ptr<ProjDataInfo> out_proj_data_info_sptr =
      segment.get_proj_data_info_ptr()->clone();
    out_proj_data_info_sptr->set_num_views(num_views_pow2);
          
    SegmentBySinogram<float> out_segment = 
      out_proj_data_info_sptr->
      get_empty_segment_by_sinogram(
      segment.get_segment_num());

    for (int axial_pos_num = segment.get_min_axial_pos_num(); axial_pos_num <= segment.get_max_axial_pos_num(); axial_pos_num++) 
    {
        const Sinogram<float> sino2D = segment.get_sinogram(axial_pos_num);
        Sinogram<float> out_sino2D = out_segment.get_sinogram(axial_pos_num);
        adjust_pow2(out_sino2D, sino2D);
        out_segment.set_sinogram(out_sino2D);
    }
  
    segment = out_segment;

}



static const double epsilon = 1e-10;

void
adjust_pow2 (Sinogram<float>& out_sino2D, const Sinogram<float>& sino2D)
{
    const double factor = ((double)out_sino2D.get_num_views()/ sino2D.get_num_views());

    double pix1=1;
    double pix2= 1/factor;
        

    float V1; // Voxel value located on the  current index
    float V2; // Voxel value located on the next index
 
    double dy=1;
      
    for (int x= out_sino2D.get_min_tangential_pos_num(); x <= out_sino2D.get_max_tangential_pos_num(); x++){
        int y1 =  sino2D.get_min_view_num() ;
        for(int  y2=out_sino2D.get_min_view_num(); y2 <= out_sino2D.get_max_view_num(); y2++){
                       
            double y1d = (double) (y1  - sino2D.get_min_view_num()+1);
                       
            if(y1> sino2D.get_max_view_num())
                continue;
            if(y1== sino2D.get_max_view_num() || y2 == out_sino2D.get_max_view_num() ) {
                V1= sino2D[y1][x]; 
                V2= 0; 
            }else {  
                V1= sino2D[y1][x]; 
                V2= sino2D[y1+1][x];
            }
                    
            if ((y1d*pix1  - y2*pix2) >= -epsilon)
                dy =  1;
            else if(!((y1d*pix1 - y2*pix2) >= -epsilon))
                dy = 1- (y2*pix2 - y1d*pix1)/pix2;;


              out_sino2D[y2+out_sino2D.get_min_view_num()][x] += ((pix2/pix1)*(dy*V1 +(1-dy)*V2)) ;
                  
            if  ((fabs(y1d*pix1 -y2*pix2) < epsilon) || dy!=1)
                y1++;
        }// End of for x2
        
    }// End of case where bin size of original image and final image are not the same
    out_sino2D.set_offset(sino2D.get_min_view_num());

    
}

void FourierRebinning::
fore_check_parameters(int num_tang_poss_pow2, int num_views_pow2, int max_segment_num_to_process){

//CON Check if the parameters given make sense.

 if(deltamin<0) {
   cerr << "FORE initialisation :: The deltamin parameter must be an integer value >= 0 " << endl;
   cerr << "                       Fix this in your FORE parameter section of your steering file " << endl;
   cerr << "                       Abort now !" << endl;
   exit(1);

 }


 if(wmin < 0 || kmin < 0){
   cerr << "FORE initialisation :: The parameters wmin and kmin must be >=0 " << endl;
   cerr << "                       Negative frequencies are not allowed. Due to symmetry reasons they are implicitly defined by the positive wmin/kmin cut off values " << endl;
   cerr << "                       Fix this in your FORE parameter section of your steering file " << endl;
   cerr << "                       Abort now !" << endl;
   exit(1);   
 }


 if(wmin >= num_tang_poss_pow2/2 || kmin >= num_views_pow2/2) {
   cerr << "FORE initialisation :: The parameter wmin or kmin is larger than the highest frequency component computed by the FFT alogorithm " << endl;
   cerr << "                       Choose an value smaller than the largest frequency " << endl;
   cerr << "                       kmin must be smaller than " << num_tang_poss_pow2/2 << " and wmin must be smaller than " << num_views_pow2/2 << endl;
   cerr << "                       Fix this in your FORE parameter section of your steering file " << endl;
   cerr << "                       Abort now ! " << endl;
   exit(1);
   }


 if(kc >= num_views_pow2/2) {
   cerr << "FORE initialisation :: Your parameter kc is larger than the highest frequency component in w (FTT of radial coordinate s) " << endl;
   cerr << "                       Choose an value smaller than the largest frequency " << endl;
   cerr << "                       kc must be smaller than " << num_views_pow2/2 << endl;
   cerr << "                       Fix this in your FORE parameter section of your steering file " << endl;
   cerr << "                       Abort now ! " << endl;
   exit(1);
  } 


 if(max_segment_num_to_process > proj_data_sptr->get_num_segments()){
   cerr << "FORE initialisation :: Your data set stores " << proj_data_sptr->get_num_segments()/2+1 << " segments " << endl;
   cerr << "                       The maximum number of segments to process variable is larger than that. " << endl;
   cerr << "                       Fix this in your FORE parameter section of your steering file " << endl;
   cerr << "                       Abort now ! " << endl;
   exit(1);

   }


 if(max_segment_num_to_process < 0){
   cerr << "FORE initialisation :: The maximum segment number to process must be an integer value >=0 " << endl;
   cerr << "                       Fix this in your FORE parameter section of your steering file " << endl;
   cerr << "                       Abort now ! " << endl;
   exit(1);         
  }       

}

END_NAMESPACE_STIR
