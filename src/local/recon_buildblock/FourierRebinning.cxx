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

#include "local/stir/recon_buildblock/FourierRebinning.h"

#include "stir/Scanner.h"
#include "stir/ProjDataInfoCylindrical.h" 
#include "stir/ProjDataInterfile.h"
#include "local/stir/fft.h"
#include "stir/SegmentBySinogram.h"
//#include "stir/SegmentByView.h"
//#include "stir/Sinogram.h"
#include "stir/Bin.h"
#include "stir/IndexRange3D.h"
#include "stir/IndexRange2D.h"
#include "stir/Succeeded.h"
#include "stir/round.h"
#include <iostream>
#include <fstream>
#include <numeric>
#include <time.h>



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
  //CL10/03/00  Remove them as not useful,
  //   parser.add_key("Load table of look-up table",&LUT);
  //   parser.add_key("Store weights in look-up table", &store);
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


void 
FourierRebinning::
do_rebinning(SegmentBySinogram<float> &segment,
	     PETCount_rebinned &count_rebinned,
	     Array<3,float> &FTdata,
	     Array<3,float> &weight)
{// Kernel of do_rebinning
    
    int num_tang_poss_pow2 = FTdata[1][1].get_length();
    int num_views_pow2 = FTdata[1].get_length();
    Array<2,float> speq(IndexRange2D(1,1,1,2*num_tang_poss_pow2));
    Array<3,float> data(IndexRange3D(1,1,1,num_tang_poss_pow2,1,num_views_pow2));

    int local_rebinned=0, local_ssrb=0, local_miss=0;
    local_rebinned = count_rebinned.total;
    local_miss= count_rebinned.miss;
    local_ssrb= count_rebinned.ssrb;
  
    for (int axial_pos_num = segment.get_min_axial_pos_num();
         axial_pos_num <= segment.get_max_axial_pos_num();
         axial_pos_num++) 
      {
	// Calculate the 2D FFT of P(w,k) of the merged segment
        for (int j = 1; j <= segment.get_num_tangential_poss(); j++)
	  for (int i = 1; i <= num_views_pow2; i++)
	    data[1][j][i] = 
	      segment[axial_pos_num][i-1][j + segment.get_min_tangential_pos_num() - 1];
        
        rlft3(data, speq, 1, num_tang_poss_pow2, num_views_pow2, 1);

	// now assign to z in output

	const ProjDataInfoCylindrical& proj_data_info_cylindrical =
	  dynamic_cast<const ProjDataInfoCylindrical&>
	  (*segment.get_proj_data_info_ptr());

	const float average_ring_difference =
	  proj_data_info_cylindrical.
	  get_average_ring_difference(segment.get_segment_num());
        const int span = 
	  proj_data_info_cylindrical.get_max_ring_difference(segment.get_segment_num()) - 
	  proj_data_info_cylindrical.get_min_ring_difference(segment.get_segment_num()) 
	  + 1;

	// KT TODO check this formula. No idea why (span-1)/2 is subtracted 
        const float z = span>1 ?
            axial_pos_num+max(0.,fabs(average_ring_difference)-(span-1)/2):
            2*axial_pos_num+average_ring_difference;
        
        rebinning(data,
                  FTdata,
                  weight,
                  z,
                  average_ring_difference,
                  *segment.get_proj_data_info_ptr(), count_rebinned);
        
    }// End of loop of axial_pos_num
    {
        cout << "      Total rebinned: " <<  count_rebinned.total -local_rebinned<< endl;
        cout << "      Total missed: " << count_rebinned.miss - local_miss<< endl;
        cout << "      Total rebinned SSRB: " << count_rebinned.ssrb -local_ssrb << endl;
    }
}


Succeeded 
FourierRebinning::
rebin()
{

  start_timers();

  int num_views_pow2;
  for ( num_views_pow2 = 1; num_views_pow2 < 2*proj_data_sptr->get_num_views() && num_views_pow2 < (1<<15); num_views_pow2*=2);

  // Find the number of tangential positions power of two
  int num_tang_poss_pow2;
  for ( num_tang_poss_pow2 = 1; num_tang_poss_pow2 < proj_data_sptr->get_num_tangential_poss() && num_tang_poss_pow2 < (1<<15); num_tang_poss_pow2*=2);
    
  // Initialise the 2D Fourier transform of all rebinned sinograms P(w,k)=0
   
  const int num_planes = 
    proj_data_sptr->get_proj_data_info_ptr()->get_scanner_ptr()->get_num_rings()*2-1;
  Array<3,float> FTdata(IndexRange3D(0, num_planes-1,
				     1, num_views_pow2,
				     1,num_tang_poss_pow2));
  Array<3,float> weight(IndexRange3D(0, num_planes-1,
				     1,(int)(num_views_pow2/2),
				     1,num_tang_poss_pow2));

  PETCount_rebinned num_rebinned(0,0,0);// Initialisation of number of counts for rebinning
   
           
  for (int seg_num=0; seg_num <=max_segment_num_to_process ; seg_num++){
           
    cout <<"  * Processing segment No " << seg_num << " *" <<endl;
            
    SegmentBySinogram<float> segment = proj_data_sptr->get_segment_by_sinogram(seg_num);
           
         
    {// Form a 360 degree sinogram by merging two 180 degree segments  with opposite ring difference
      // to get a new segment sampled over 2*pi (where 0 < view < pi)

      // TODO this is currently not a good idea, as all ProjDataInfo classes assume that
      // views go from 0 to Pi.
      const SegmentBySinogram<float> segment_neg = 
	proj_data_sptr->get_segment_by_sinogram(-seg_num);
      segment.grow(IndexRange3D(segment.get_min_axial_pos_num(),
				segment.get_max_axial_pos_num(),
				0,2*segment.get_num_views()-1,
				segment.get_min_tangential_pos_num(),
				segment.get_max_tangential_pos_num()
				));
  
      for (int ring = segment.get_min_axial_pos_num(); ring <= segment.get_max_axial_pos_num(); ring++){
	for (int view = segment_neg.get_min_view_num(); view <= segment_neg.get_max_view_num(); view++)
	  segment[ring][view+segment_neg.get_num_views()] = 
	    segment_neg[ring][view];
      }
                
    }     
 
    // do_mashing(segment);
           
    do_adjust_nb_views_to_pow2(segment);
           
    do_rebinning(segment, num_rebinned, FTdata, weight);
    cout << endl;
  }
        
   
  {
    cout << endl << "  - Total rebinning count: " << endl;
    do_display_count(num_rebinned);
  }
        
  //CL10/03/00 Remove these lines as not useful
#if 0
  char lutfilename[80];
  sprintf(lutfilename, "%s","LUT.dat");
  
  // If LUT load look-up table from file into array Weights:
  if (LUT)
    do_LUT(lutfilename,weight);

  // If not LUT but Store, store array weights as look-up table on file:
  if (!LUT&&store)
    do_storeLUT(lutfilename, weight);
#endif

  // make output projdata
  shared_ptr<ProjData> rebinned_proj_data_sptr;
  {            
    const int max_delta = 
      dynamic_cast<ProjDataInfoCylindrical const&>
      (*proj_data_sptr->get_proj_data_info_ptr()).
      get_max_ring_difference(max_segment_num_to_process);
    shared_ptr<ProjDataInfo> rebinned_proj_data_info_sptr =
      proj_data_sptr->get_proj_data_info_ptr()->clone();
    rebinned_proj_data_info_sptr->
      set_num_views(num_views_pow2/2);
    rebinned_proj_data_info_sptr->
      reduce_segment_range(0,0);
    dynamic_cast<ProjDataInfoCylindrical&>
      (*rebinned_proj_data_info_sptr).
      set_min_ring_difference(-max_delta, 0);
    dynamic_cast<ProjDataInfoCylindrical&>
      (*rebinned_proj_data_info_sptr).
      set_max_ring_difference(max_delta, 0);
    rebinned_proj_data_info_sptr->
      set_min_axial_pos_num(0, 0);
    // TODO do not use num_rings but something with num_axial_poss ?
    rebinned_proj_data_info_sptr->
      set_max_axial_pos_num(proj_data_sptr->get_proj_data_info_ptr()->
			    get_scanner_ptr()->get_num_rings()*2 - 2,
			    0);
    rebinned_proj_data_sptr = 
      new ProjDataInterfile (rebinned_proj_data_info_sptr,
			     output_filename_prefix,
			     ios::out);
  }

  // now finally fill in the new sinograms

  SegmentBySinogram<float> sino2D_rebinned =
    rebinned_proj_data_sptr->get_empty_segment_by_sinogram(0);
  {
    Array<3,float> data(IndexRange3D(1,1,1,num_tang_poss_pow2,1,num_views_pow2));
    Array<2,float> speq(IndexRange2D(1,1,1,2*num_tang_poss_pow2));

    for (int plane =FTdata.get_min_index();plane <= FTdata.get_max_index(); plane++){ 
      // divide by weights
      for (int j = 1; j <= num_tang_poss_pow2; j++)
	for (int i = 1; i <= (int) ( num_views_pow2/ 2); i++) {
	  const float Weight = (weight[plane][i][j] == 0) ? 0 :  1./ (weight[plane][i][j]);
	  data[1][j][2 * i - 1] = FTdata[plane][2 * i - 1][j]* Weight;
	  data[1][j][2 * i] = FTdata[plane][2 * i][j]* Weight; 
	}
      // Inverse 3D FFT
      speq.fill(0); 
      float norm = data[1][1][1];
            
      rlft3(data, speq, 1, num_tang_poss_pow2, num_views_pow2,-1);
      data*=(2./(num_tang_poss_pow2*num_views_pow2));
      data*=(norm/data.sum());
            
      // Keep only one half of data [o.._PI]
      for (int i=1;i<=(int)(num_views_pow2/2);i++) 
	for (int j=1;j<=num_tang_poss_pow2;j++)
	  if ((j+sino2D_rebinned.get_min_tangential_pos_num()-1)<=sino2D_rebinned.get_max_tangential_pos_num())
	    sino2D_rebinned[plane][i-1][j+ sino2D_rebinned.get_min_tangential_pos_num()-1]=data[1][j][i];
    }
  }
  {
    // One more normalization for adjusting the number of views to the original ones
    const float factor =((float) proj_data_sptr->get_proj_data_info_ptr()->get_num_views()/(float) sino2D_rebinned.get_num_views());
    cerr << "  - Multiply by a factor for normalization = " << factor << endl;
    sino2D_rebinned/= factor;
  }

  Succeeded success = Succeeded::yes;
  {// Saving rebinned sinograms
      
    cout << "    2D Rebinned sinograms => Min = " << sino2D_rebinned.find_min()
	 << " Max = " << sino2D_rebinned.find_max()
	 << " Sum = " << sino2D_rebinned.sum() << endl;

    const Succeeded success_this_sino = 
      rebinned_proj_data_sptr->set_segment(sino2D_rebinned);
    if (success == Succeeded::yes &&
	success_this_sino == Succeeded::no)
      success = Succeeded::no;
  }
 
  stop_timers();        
  // Display timing details and write them into a log file
  do_log_file();
 
  return success;
}

void 
FourierRebinning::
rebinning(const Array<3,float> &data,
	  Array<3,float> &FTdata,
	  Array<3,float> &Weights,
	  float z_as_float,
	  float delta,
	  const ProjDataInfo &proj_data_info,
	  PETCount_rebinned &num_rebinned)
{// Fourier rebinning kernel
  const Scanner& scanner = *proj_data_info.get_scanner_ptr();

  const int z = round(z_as_float);
  if (fabs(z-z_as_float) >  scanner.get_ring_spacing()/100)
    error("Fourier rebinning expected integer z coordinate but found %g\n",
	  z_as_float);

  const float bin_size =
    proj_data_info.get_sampling_in_s(Bin(0,0,0,0));
  const int     num_views_pow2 = FTdata[0].get_length();
  const float   d_sl = scanner.get_ring_spacing()*0.5; 
  const int     num_tang_poss_pow2 = FTdata[0][0].get_max_index();
  const int     maxplane = FTdata.get_max_index();
  const float   delta_w = 2.*_PI/bin_size/num_tang_poss_pow2;
  const float   Rmm = ((int) (proj_data_info.get_num_tangential_poss() / 2) - 1)*bin_size;

// t is the tangent of the angle theta between the LOR and the transaxial plane
    const float   t = delta*scanner.get_ring_spacing()/scanner.get_ring_radius() /2.;
    
// The continuous frequency "w" corresponds the radial coordinate "s"
// The integer Fourier index "k" corresponds to the azimuthal angle "view"
     
    
    
    for (int j = wmin; j <= (int) (num_tang_poss_pow2 / 2)+1; j++)// wherej=num_tang_poss_pow2/2+1 is the Nyquist frequency
        for (int i = kmin; i <= (int) (num_views_pow2 / 2); i++) {
            float w = (j - 1) * delta_w;
            float k = (float) (i - 1);
            if ((k > w * Rmm) && (i > kc))
                continue;	// improve consistency
            float zshift = t * k / w; // in mm

            int jj=j;// Positive frequency
            for(int sign=-1;sign<=1;sign+=2){
                if(sign==1)// Negative frequency
                    jj= num_tang_poss_pow2 + 2 - j;
                float new_z_sl = z +sign* zshift / d_sl;
                int small_z = (int) floor(new_z_sl);
                float m = new_z_sl - small_z;
                if (small_z >= 0 && small_z <= maxplane) {
                    FTdata[small_z][2 * i - 1][jj] += (1. - m) * data[1][jj][2 * i - 1];	// Real 
                    FTdata[small_z][2 * i][jj] += (1. - m) * data[1][jj][2 * i];	// Imag 
                    num_rebinned.total += 1;
// CL 10/03/00 Remove it as not useful
//                   if (!LUT)
                    Weights[small_z][i][jj] += 1. - m;
                } else
                    num_rebinned.miss += 1;
                if (small_z >= -1 && small_z < maxplane) {
                    FTdata[small_z + 1][2 * i - 1][jj] += m * data[1][jj][2 * i - 1];  // Real 
                    FTdata[small_z + 1][2 * i][jj] += m * data[1][jj][2 * i];  // Imag 
// CL 10/03/00 Remove it as not useful
//                    if (!LUT)
                    Weights[small_z + 1][i][jj] += m;
                }
            }//End of for (sign=-1...)
        }

// Particular case for small frequencies i.e Small w
    
    int small_z = z; 
    if (delta <= deltamin) {
            // First, treat small w's then  k=1..kNyq-1 :
        for (int j = 1; j < wmin; j++)
            for (int i = 1; i <= (int) (num_views_pow2 / 2); i++) {
                int jj=j;
                for(int sign=-1;sign<=1;sign+=2){
                    if(sign==1 && j>1)
                        jj=num_tang_poss_pow2 + 2 - j;
                    FTdata[small_z][2 * i - 1][jj] += data[1][jj][2 * i - 1];	// Real
                    FTdata[small_z][2 * i][jj] += data[1][jj][2 * i];	// Imag
// CL 10/03/00 Remove it as not useful
//                    if (!LUT)
                    Weights[small_z][i][jj] += 1.;
                    if(j==1)
                        num_rebinned.ssrb += 1;//CL 2110 Remove rebin_ssrb
                }
            }

// Small k :
// Next treat small k's and w=wNyq=(num_tang_poss_pow2 / 2)+1, k=1..klim :
        for (int j = wmin; j <= (int) (num_tang_poss_pow2 / 2)+1; j++)
            for (int i = 1; i < kmin; i++) {
                int jj=j;
                for(int sign=-1;sign<=1;sign+=2){
                    if(sign==1)
                        jj=num_tang_poss_pow2 + 2 - j;
                    FTdata[small_z][2 * i - 1][jj] += data[1][jj][2 * i - 1];	// Real
                    FTdata[small_z][2 * i][jj] += data[1][jj][2 * i];	// Imag
// CL 10/03/00 Remove it as not useful
//                    if (!LUT)
                    Weights[small_z][i][jj] += 1.;

                    num_rebinned.ssrb += 1;
                }
            }
    }
}


void 
FourierRebinning::
do_log_file( )
{// Saving time details and write them to log file 
    char file[200];
    sprintf(file,"%s.log",output_filename_prefix.c_str());
    
    ofstream logfile(file);
 
    if (logfile.fail() || logfile.bad()) {
      warning("Error opening log file\n");
    }

    time_t now = time(NULL);
    logfile << "Date of the FORE image reconstruction : " << asctime(localtime(&now))
            << parameter_info()
            << "\n\n CPU Time :\n"
	    << get_CPU_timer_value()
	    << endl;
       
}


 //CL10/03/00 remove these functions as not useful
#if 0
void 
FourierRebinning::
do_LUT(char lutfilename[80], Array<3,float> &weight)
{
 
    cout << "The relative weights of the frequency components are read from "
         <<  endl << "the look-up table : " <<  lutfilename << endl;
                    
		
    fstream lutfile;
    lutfile.open(lutfilename, ios::in | ios::nocreate);
                          
    if (lutfile.fail() || lutfile.bad()) {
        warning("Error opening file\n");
    }
    weight.read_data(lutfile);
		
}


void 
FourierRebinning::
do_storeLUT(char lutfilename[80], Array<3,float> &weight)
{

    cout << "The relative weights of the frequency components are stored "
         <<  endl << " in the file of the look-up table : " <<  lutfilename << endl;

    ofstream lutfile(lutfilename);
    lutfile.open(lutfilename, ios::out );
    if (lutfile.fail() || lutfile.bad()) {
        error("Error opening file\n");
    }
    weight.write_data(lutfile);
}
#endif

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


//  This is a function which adapt the number of views of a sinogram to a power of two
static void adjust_pow2 (Sinogram<float>& out_sino2D, const Sinogram<float>& sino2D);

void 
FourierRebinning::
do_adjust_nb_views_to_pow2(SegmentBySinogram<float> &segment) 
{// Adjustment of the number of views to a power of two
    cout << "    - Adjustment of the number of views to a power of two... " << endl;
    const int  num_views_pow2 = (int) pow(2,((int)ceil(log(segment.get_num_views())/log(2))));
      
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

    // KT TODO this looks like overlap_interpolate.
    // we probably can just call the existing function.
  
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
                   
            out_sino2D[y2+out_sino2D.get_min_view_num()-1 ][x] += ((pix2/pix1)*(dy*V1 +(1-dy)*V2)) ;
                
            if  ((fabs(y1d*pix1 -y2*pix2) < epsilon) || dy!=1)
                y1++;
        }// End of for x2
        
    }// End of case where bin size of original image and final image are not the same
    out_sino2D.set_offset(sino2D.get_min_view_num());

}

END_NAMESPACE_STIR
