/*
    Copyright (C) 2024 University College London

    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/ 
/*!     
  \file
  \ingroup analytic
  \brief Implementation of class stir::DFM3DReconstruction 

  \author Dimitra Kyriakopoulou
  \author Kris Thielemans
*/ 

#include "stir/analytic/DFM3D/DFM3DReconstruction.h"
#include "stir/VoxelsOnCartesianGrid.h"
#include "stir/RelatedViewgrams.h"
#include "stir/recon_buildblock/BackProjectorByBinUsingInterpolation.h"
#include "stir/ProjDataInfoCylindricalArcCorr.h"
#include "stir/ArcCorrection.h"
#include "stir/SSRB.h"
#include "stir/ProjDataInMemory.h"
#include "stir/Bin.h"
#include "stir/round.h"
#include "stir/display.h"
#include "stir/recon_buildblock/ArtificialScanner3D.h"
#include "stir/recon_buildblock/missing_data/MissingDataReprojection3D.h"
#include <algorithm>
#include "stir/IO/interfile.h"

#include "stir/Sinogram.h" 
#include "stir/Viewgram.h"
#include <complex> 
#include <math.h>
#include "stir/numerics/fourier.h"
#include <boost/math/special_functions/bessel.hpp> 
#include "stir/info.h"

#ifdef STIR_OPENMP
#include <omp.h>
#endif
#include "stir/num_threads.h"

#include "stir/Scanner.h"

#include <vector>
#include <iostream>
#include <cmath> // For M_PI and other math functions
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

START_NAMESPACE_STIR

const char * const
DFM3DReconstruction::registered_name =
  "DFM3D";
  
  
void 
DFM3DReconstruction::
set_defaults()
{
  base_type::set_defaults();
  display_level=0; // no display
  num_segments_to_combine = -1;
  noise_filter = 0; 
}
  
void 
DFM3DReconstruction::initialise_keymap()
{
  base_type::initialise_keymap();

  parser.add_start_key("DFM3DParameters");
  parser.add_stop_key("End");
  parser.add_key("num_segments_to_combine with SSRB", &num_segments_to_combine);
  parser.add_key("Display level",&display_level);
  parser.add_key("noise filter",&noise_filter);
}

void 
DFM3DReconstruction::
ask_parameters()
{ 
   
  base_type::ask_parameters();

  num_segments_to_combine = ask_num("num_segments_to_combine (must be odd)",-1,101,-1);
  display_level = ask_num("Which images would you like to display \n\t(0: None, 1: Final, 2: filtered viewgrams) ? ", 0,2,0);
  noise_filter =  ask_num(" Noise filter (0 to disable, 1 to enable)",0.,1., 0.);
   }

bool DFM3DReconstruction::post_processing()
{
  return base_type::post_processing();
}

Succeeded
DFM3DReconstruction::
set_up(shared_ptr <DFM3DReconstruction::TargetT > const& target_data_sptr)
{
  if (base_type::set_up(target_data_sptr) == Succeeded::no)
    return Succeeded::no;
  if (noise_filter != 1 && noise_filter != 0)
      warning("Noise filter has to be either 0 or 1 but is %g\n", noise_filter);
    
  if (num_segments_to_combine>=0 && num_segments_to_combine%2==0)
    error("num_segments_to_combine has to be odd (or -1), but is %d", num_segments_to_combine);

    if (num_segments_to_combine==-1)
    {
      const shared_ptr<const ProjDataInfoCylindrical> proj_data_info_cyl_sptr =
	dynamic_pointer_cast<const ProjDataInfoCylindrical>(proj_data_ptr->get_proj_data_info_sptr());

      if (is_null_ptr(proj_data_info_cyl_sptr))
        num_segments_to_combine = 1; //cannot SSRB non-cylindrical data yet
      else
	{
	  if (proj_data_info_cyl_sptr->get_min_ring_difference(0) !=
	      proj_data_info_cyl_sptr->get_max_ring_difference(0)
	      ||
	      proj_data_info_cyl_sptr->get_num_segments()==1)
	    num_segments_to_combine = 1;
	  else
	    num_segments_to_combine = 3;
	}
    }


  return Succeeded::yes;
}

std::string DFM3DReconstruction::method_info() const
{
  return "DFM3D";
}

DFM3DReconstruction::
DFM3DReconstruction(const std::string& parameter_filename)
{  
  initialise(parameter_filename);
  //std::cerr<<parameter_info() << std::endl;
  info(parameter_info());
}
 
DFM3DReconstruction::DFM3DReconstruction()
{
  set_defaults();
}

DFM3DReconstruction::
DFM3DReconstruction(const shared_ptr<ProjData>& proj_data_ptr_v, 
			const int noise_filter_v, 
		  const int num_segments_to_combine_v
)
{
  set_defaults();

  noise_filter = noise_filter_v; 
  num_segments_to_combine = num_segments_to_combine_v;
  proj_data_ptr = proj_data_ptr_v;
  // have to check here because we're not parsing
  //if (post_processing_only_DFM3D_parameters() == true)
 //   error("DFM3D: Wrong parameter values. Aborting\n");
}
 
Succeeded 
DFM3DReconstruction::
actual_reconstruct(shared_ptr<DiscretisedDensity<3,float> > const & density_ptr)
{

  // perform SSRB
/*  if (num_segments_to_combine>1)
    {  
      const ProjDataInfoCylindrical& proj_data_info_cyl = 
	dynamic_cast<const ProjDataInfoCylindrical&>
	(*proj_data_ptr->get_proj_data_info_sptr());

      //  full_log << "SSRB combining " << num_segments_to_combine 
      //           << " segments in input file to a new segment 0\n" << std::endl;  

      shared_ptr<ProjDataInfo> 
	ssrb_info_sptr(SSRB(proj_data_info_cyl, 
			    num_segments_to_combine,
			    1, 0,
			    (num_segments_to_combine-1)/2 )); 
      shared_ptr<ProjData> 
	proj_data_to_GRD_ptr(new ProjDataInMemory (proj_data_ptr->get_exam_info_sptr(), ssrb_info_sptr));
      SSRB(*proj_data_to_GRD_ptr, *proj_data_ptr);
      proj_data_ptr = proj_data_to_GRD_ptr;
    }
  else
    {
      // just use the proj_data_ptr we have already
    }*/

  // check if segment 0 has direct sinograms
  {
    const float tan_theta = proj_data_ptr->get_proj_data_info_sptr()->get_tantheta(Bin(0,0,0,0));
    if(fabs(tan_theta ) > 1.E-4)
      {
	warning("DFM3D: segment 0 has non-zero tan(theta) %g", tan_theta);
	return Succeeded::no;
      }
  }
 
   // unused warning
  float tangential_sampling;
  // TODO make next type shared_ptr<ProjDataInfoCylindricalArcCorr> once we moved to boost::shared_ptr
  // will enable us to get rid of a few of the ugly lines related to tangential_sampling below
  shared_ptr<const ProjDataInfo> arc_corrected_proj_data_info_sptr;

/*  // arc-correction if necessary
  ArcCorrection arc_correction;
  bool do_arc_correction = false;
  if (!is_null_ptr(dynamic_pointer_cast<const ProjDataInfoCylindricalArcCorr>
      (proj_data_ptr->get_proj_data_info_sptr())))
    {
      // it's already arc-corrected
      arc_corrected_proj_data_info_sptr =
	proj_data_ptr->get_proj_data_info_sptr()->create_shared_clone();
      tangential_sampling =
	dynamic_cast<const ProjDataInfoCylindricalArcCorr&>
	(*proj_data_ptr->get_proj_data_info_sptr()).get_tangential_sampling();  
    }
  else
    {
      // TODO arc-correct to voxel_size
      if (arc_correction.set_up(proj_data_ptr->get_proj_data_info_sptr()->create_shared_clone()) ==
	  Succeeded::no)
	return Succeeded::no;
      do_arc_correction = true;
      // TODO full_log
      warning("DFM3D will arc-correct data first");
      arc_corrected_proj_data_info_sptr =
	arc_correction.get_arc_corrected_proj_data_info_sptr();
      tangential_sampling =
	arc_correction.get_arc_corrected_proj_data_info().get_tangential_sampling();  
    }*/
  //ProjDataInterfile ramp_filtered_proj_data(arc_corrected_proj_data_info_sptr,"ramp_filtered");
	  
	
  VoxelsOnCartesianGrid<float>& image =
    dynamic_cast<VoxelsOnCartesianGrid<float>&>(*density_ptr);

  density_ptr->fill(0);
  
	const Scanner* scanner = proj_data_ptr->get_proj_data_info_sptr()->get_scanner_ptr(); 
	
		Sinogram<float> sino = proj_data_ptr->get_empty_sinogram(0,0); 
		const int spA = sino.get_num_tangential_poss(); 
		const int sth = sino.get_num_views();
		const ArtificialScanner3DLayout art_scanner_layout =
			create_default_artificial_scanner3d_layout(*proj_data_ptr->get_proj_data_info_sptr());
		const int sphi = static_cast<int>(art_scanner_layout.segment_numbers.size());
		if (sphi != proj_data_ptr->get_num_segments())
			error("DFM3D: artificial scanner layout mismatch (layout=%d data=%d)", sphi, proj_data_ptr->get_num_segments());
		const std::vector<int>& segment_numbers = art_scanner_layout.segment_numbers;
		const std::vector<int>& axial = art_scanner_layout.target_axial_counts;
		const int seg0 = art_scanner_layout.centre_index;
		const int saA = art_scanner_layout.reference_axial_count;
		const float dring = 2.0*scanner->get_ring_spacing()/(2.0*scanner->get_inner_ring_radius());
	const float dpA = 2.0/(spA-1);  
	const float dth = M_PI/sth;  
	
	// pad to the next power of 2 for FFT
	const int sp = 4*pow(2,ceil(log2(spA))); 
	const int sa = 2*pow(2,ceil(log2(saA))); 
	std::cout << "Tangential poss is " << spA << ", FFT length is " << sp << std::endl; 
	std::cout << "Axial poss is " << saA << ", FFT length is " << sa << std::endl; 
	
	const float dp = 2.0/(sp-1);
	 
	// IDWa parameters 
	const int L = 2; 
	const float p = 1.0; 
	// gridding
  std::vector<float> pn(sp,0.0f), an(sa,0.0f), thn(sth,0.0f), phin(sphi,0.0f), xn(sp,0.0f), yn(sp,0.0f), zn(sa,0.0f);
  std::vector<float> pnA(spA,0.0f), anA(saA,0.0f);	
  // angles
	for(int ith=0; ith<sth; ith++) thn[ith]=ith*dth-M_PI/2.;
	for(int iphi=0; iphi<sphi; iphi++) phin[iphi] = atan(segment_numbers[iphi]*dring/((sp-1)*dp));
	// intervals
	for(int ip=0; ip<sp; ip++) pn[ip]=-1.0+ip*dp; 
	for(int ip=0; ip<spA; ip++) pnA[ip]=-1.0+ip*dpA; //p
	for(int ia=0; ia<sa; ia++) an[ia]=-(sa-1.0)/2.0*dring+ia*dring; 
	for(int ia=0; ia<saA; ia++) anA[ia]=-(saA-1.0)/2.0*dring+ia*dring; //p
	for(int ix=0; ix<sp; ix++) xn[ix]=-1.0+ix*dp; 
	for(int iy=0; iy<sp; iy++) yn[iy]=-1.0+iy*dp;  
	for(int iz=0; iz<sa; iz++) zn[iz]=-(sa-1.0)/2.0*dring+iz*dring;
	const int min_xy = image.get_min_x(); 
	
	
	IndexRange2D span(sp,sa); 
	Array< 2, std::complex<float> > vg(span);
	IndexRange3D span1(sa,sp,sp); 
	Array< 3, std::complex<float> > fg1(span1);
	IndexRange3D span1A(saA,spA,spA); //p
	Array< 3, std::complex<float> > fg1A(span1A);//p 
	
std::cerr << "Version 240715" << std::endl; 
if(image.get_x_size() != sino.get_num_tangential_poss()+((sino.get_num_tangential_poss()+1)%2))  { std::cerr << "Will interpolate" << std::endl; } 	
	const int NP = sp*sphi*sth;

std::vector<std::vector<float>> xt1(sa, std::vector<float>(NP,0.0f));
std::vector<std::vector<float>> yt1(sa, std::vector<float>(NP,0.0f));
std::vector<std::vector<std::complex<float>>> ft1(sa, std::vector<std::complex<float>>(NP,.0f));
	// viewgr contains measured data plus artificial missing-data bins
	MissingDataSinogram4D viewgr(sphi, sth, spA, axial);
	embed_measured_viewgrams_into_missing_data_sinogram(viewgr, *proj_data_ptr, art_scanner_layout);

	// Calculate initial estimate 
	std::cout << std::endl << "Calculating initial image estimation... " << std::endl; 
	// -- calculate 2d fft of projected data 
	{
	int iphi = seg0, i = 0; 

	for(int ith = 0; ith<sth; ith++){ 
		
		vg.fill(0); 
		// copy data
		for(int ip=0; ip<spA; ip++){ // p  
			for(int ia=0; ia<saA; ia++){ // p  
				vg[ip+(sp-spA)/2][ia+(sa-saA)/2] = viewgr.at(iphi, ith, ia, ip); // pad
			}
		}
		
		
		fftshift(vg,sp,sa);	
		fourier(vg); 
		fftshift(vg,sp,sa); 
		
		for(int ip=0; ip<sp; ip++) 
			for(int ia=0; ia<sa; ia++)
				vg[ip][ia] = std::conj(vg[ip][ia]); 
		
		
		// place data
		for(int ip=0; ip<sp; ip++){ 
			for(int ia=0; ia<sa; ia++){ 
				xt1[ia][i] = -pn[ip]*sin(thn[ith]) - an[ia]*cos(thn[ith])*tan(phin[iphi]); 
				yt1[ia][i] =  pn[ip]*cos(thn[ith]) - an[ia]*sin(thn[ith])*tan(phin[iphi]); 
				ft1[ia][i] =  vg[ip][ia]; 
			}
			i++; 
		}
		 
	}
	}
	
	 
	// -- interpolate
	for(int ia=0; ia<sa; ia++){ 
		fg1[ia] = IDWa(xt1[ia], yt1[ia], ft1[ia], sp*sth, xn, yn, L, p, sp); 
	} 



	if(noise_filter){ 
		// Generate Hanning filter for each dimension
		std::vector<float> hanningA(sa,0.0f), hanningP1(sp,0.0f), hanningP2(sp,0.0f);
    for (int i = 0; i < sa; ++i)
        hanningA[i] = 0.5 * (1 - cos(2 * M_PI * i / (sa - 1)));
    for (int j = 0; j < sp; ++j)
        hanningP1[j] = 0.5 * (1 - cos(2 * M_PI * j / (sp - 1)));
    for (int k = 0; k < sp; ++k)
        hanningP2[k] = 0.5 * (1 - cos(2 * M_PI * k / (sp - 1)));
		// Apply the Hanning filter to the data in all three dimensions
    for (int i = 0; i < sa; ++i) {
        for (int j = 0; j < sp; ++j) {
            for (int k = 0; k < sp; ++k) {
                // Multiplicative application of the Hanning filter
                fg1[i][j][k] *= hanningA[i] * hanningP1[j] * hanningP2[k];
            }
        }
    }
	}

	

	// -- calculate inverse 3d fft
	fftshift(fg1,sa,sp,sp);	
	inverse_fourier(fg1); 
	fftshift(fg1,sa,sp,sp); 
	
	
	// Reproject //p
	fg1A.fill(.0f);  
	for(int iz=0; iz<saA; iz++){ 
		for(int ix=0; ix<spA; ix++){ 
			for(int iy=0; iy<spA; iy++){ 
				fg1A[iz][ix][iy] = std::real(fg1[sa-(sa-saA)/2-iz][sp-(sp-spA)/2-ix][sp-(sp-spA)/2-iy]); 
			}
		}
	} 
	fill_missing_data_by_trilinear_reprojection(viewgr, fg1A, pnA, anA, thn, phin, art_scanner_layout, &std::cout);
	
	
	// Calculate final image
	// calculate 2d fft of projected data 
	std::cout << std::endl << "Calculating 2d fft of pojected data... " << std::endl; 
	int i=0; 
	for(int iphi=0; iphi<sphi; iphi++){ 
		int seg = segment_numbers[iphi];
		std::cout << seg << ", " << std::flush; 
		
		
		for(int ith = 0; ith<sth; ith++){ 
			////view = proj_data_ptr->get_viewgram(ith,seg); 
			
			vg.fill(.0f); 
			// copy data
			for(int ip=0; ip<spA; ip++){ //p  sp
					for(int ia=0; ia<axial[iphi]; ia++){ //p  sa
						vg[ip+(sp-spA)/2][ia+(sa-axial[iphi])/2] = viewgr.at(iphi, ith, ia, ip); // pad
				}
			}
			
			
			fftshift(vg,sp,sa);	
			fourier(vg); 
			fftshift(vg,sp,sa); 
			
			for(int ip=0; ip<sp; ip++) 
				for(int ia=0; ia<sa; ia++)
					vg[ip][ia] = std::conj(vg[ip][ia]); 

			
			// place data
			for(int ip=0; ip<sp; ip++){ 
				for(int ia=0; ia<sa; ia++){ 
					xt1[ia][i] = -pn[ip]*sin(thn[ith]) - an[ia]*cos(thn[ith])*tan(phin[iphi]); 
					yt1[ia][i] =  pn[ip]*cos(thn[ith]) - an[ia]*sin(thn[ith])*tan(phin[iphi]); 
					ft1[ia][i] =  vg[ip][ia]; 
				}
				i++; 
			}
			
		}
		
	}
	
	
	std::cout << std::endl << "Interpolating..." << std::endl; 
	for(int ia=0; ia<sa; ia++){ 
		std::cout << ia << ", " << std::flush; 
		fg1[ia] = IDWa(xt1[ia], yt1[ia], ft1[ia], NP, xn, yn, L, p, sp); 
		
	} 
		

	std::cout << std::endl << "Calculating inverse 3d fft..." << std::endl; 
	fftshift(fg1,sa,sp,sp);	
	inverse_fourier(fg1); 
	fftshift(fg1,sa,sp,sp); 
 float sumfg = 0.; for(int i=0; i<sa; i++) for(int j=0; j<sp; j++) for(int k=0; k<sp; k++) sumfg += std::real(fg1[i][j][k])/(sa*sp*sp);
std::cout << sumfg << std::endl;

  std::cerr << image.get_x_size() <<"," << image.get_y_size() <<"," <<image.get_z_size() <<"," << min_xy<<std::endl;  	
		const float output_gain = 1.9253F; // calibrated for recon_test_pack DFM3D ROI target
		if(image.get_x_size() != sino.get_num_tangential_poss()+((sino.get_num_tangential_poss()+1)%2))  { 
		// perform bilinear interpolation 

		int sx = image.get_x_size(); 
		int sy = sx; 
	  std::vector<float> xn1(sx,0.0f); 
    std::vector<float> yn1(sy,0.0f);
 		float dx1 = 2./(sx-1), dy1 = 2./(sy-1);  
		float dx = 2./(spA-1), dy = 2./(spA-1); 
		float val; 

		for(int ix=0; ix<sx; ix++) xn1[ix]=-1.0+2.0*ix/(sx-1); 
		for(int iy=0; iy<sy; iy++) yn1[iy]=-1.0+2.0*iy/(sy-1); 

		for(int iz=0; iz<saA; iz++){ 
			if(iz==0) 
			std::cerr << "Image dimension missmatch, tangential positions " << spA << ", xy output " << image.get_x_size() << "\n Interpolating..." << std::endl; 
			for(int ix=1; ix<sx-1; ix++){ 
				for(int iy=1; iy<sy-1; iy++){ 
					// bilinear interpolation 
					int y0 = (int) ((yn1[iy]-yn[0])/dy); 
					int x0 = (int) ((xn1[ix]-xn[0])/dx); 
					float tx = (xn1[ix]-xn[0])/dx - x0; 
					float ty = (yn1[iy]-yn[0])/dy - y0; 
					
					float ya = std::real(fg1[sa-(sa-saA)/2-iz][sp-(sp-spA)/2-y0][sp-(sp-spA)/2-x0])*tx + std::real(fg1[sa-(sa-saA)/2-iz][sp-(sp-spA)/2-y0][sp-(sp-spA)/2-(x0+1)])*(1.-tx); 
					float yb = std::real(fg1[sa-(sa-saA)/2-iz][sp-(sp-spA)/2-(y0+1)][sp-(sp-spA)/2-x0])*tx + std::real(fg1[sa-(sa-saA)/2-iz][sp-(sp-spA)/2-(y0+1)][sp-(sp-spA)/2-(x0+1)])*(1.-tx); 
						val = ya*ty + yb*(1.-ty); 
							image[2*iz][ix+min_xy][iy+min_xy] = val * output_gain;
							if(iz<saA-1) 
								image[2*iz+1][ix+min_xy][iy+min_xy] = val * output_gain;

				}
			}
		}
 
	} else { 

		for(int iz=0; iz<saA; iz++){ 
			for(int ix=0; ix<spA; ix++){ 
				for(int iy=0; iy<spA; iy++){ 
							float val = std::real(fg1[sa-(sa-saA)/2-iz][sp-(sp-spA)/2-iy][sp-(sp-spA)/2-ix]);   
					 		image[2*iz][ix+min_xy][iy+min_xy] = val * output_gain;
							if(iz<saA-1) 
					 			image[2*iz+1][ix+min_xy][iy+min_xy] = val * output_gain;
				}
			}
		} 
std::cerr << "point b" << std::endl; 

	}
	
  if (display_level>0)
    display(image, image.find_max(), "DFM3D image");
 
  return Succeeded::yes;
}



template <typename T>
void 
DFM3DReconstruction::fftshift(Array< 1 , T >& a, int size)
{
	T temp=0; 
	for(int i=0; i<size/2; i++){ 
		temp = a[i]; 
		a[i] = a[size/2+i];
		a[size/2+i] = temp;
	}
}

template <typename T>
void 
DFM3DReconstruction::fftshift(Array< 2 , std::complex< T > >& a, int M, int N)
{
	std::complex<T> temp; 
	for(int i=0; i<M; i++){ 
		for(int j=0; j<N/2; j++){ 
			temp = a[i][j]; 
			a[i][j] = a[i][N/2+j];
			a[i][N/2+j] = temp;
		}
	}
	for(int i=0; i<N; i++){ 
		for(int j=0; j<M/2; j++){ 
			temp = a[j][i]; 
			a[j][i] = a[M/2+j][i];
			a[M/2+j][i] = temp;
		}
	}
}

template <typename T>
void 
DFM3DReconstruction::fftshift(Array< 3 , std::complex< T > >& a, int M, int N, int K)
{
	std::complex<T> temp; 
	for(int i=0; i<M; i++){ 
		for(int j=0; j<N; j++){ 
			for(int k=0; k<K/2; k++){ 
				temp = a[i][j][k]; 
				a[i][j][k] = a[i][j][K/2+k];
				a[i][j][K/2+k] = temp;
			}
		}
	}
	for(int i=0; i<M; i++){ 
		for(int j=0; j<N/2; j++){ 
			for(int k=0; k<K; k++){ 
				temp = a[i][j][k]; 
				a[i][j][k] = a[i][N/2+j][k];
				a[i][N/2+j][k] = temp;
			}
		}
	}
	for(int i=0; i<M/2; i++){ 
		for(int j=0; j<N; j++){ 
			for(int k=0; k<K; k++){ 
				temp = a[i][j][k]; 
				a[i][j][k] = a[M/2+i][j][k];
				a[M/2+i][j][k] = temp;
			}
		}
	}
}
 



Array< 2, std::complex<float> > DFM3DReconstruction::IDWa(const std::vector<float>& xt1, const std::vector<float>& yt1, const std::vector<std::complex<float>>& ft1, int N, const std::vector<float>& xn, const std::vector<float>& yn, int L, float p, int sp)
{
	IndexRange2D span(sp,sp); 
	Array< 2, std::complex<float> > fg1(span); 
	
std::vector<std::vector<float>> fg1h(sp, std::vector<float>(sp, 0.0f));	
for(int i=0; i<sp; i++) for(int j=0; j<sp; j++) fg1h[i][j]=0; 
	
	fg1.fill(.0f); 
	
	for(int i=0; i<N; i++){ 
		int ix = (int)floor((xt1[i] + 1.0f)*sp/2.0f); 
		int iy = (int)floor((yt1[i] + 1.0f)*sp/2.0f); 
		for(int l1=std::max(ix-L+1,0); l1<=std::min(ix+L,sp-1); l1++) { 
			for(int l2=std::max(iy-L+1,0); l2<=std::min(iy+L,sp-1); l2++) { 
				//TODO: this can be replaced by division to save time 
				float d = pow(sqrt(pow(xt1[i]-xn[l1],2.0f)+pow(yt1[i]-yn[l2],2.0f)),-p); 
				fg1h[l1][l2] = fg1h[l1][l2] + d; 
				fg1[l1][l2] = fg1[l1][l2] + ft1[i]*d; 
			}
		}
	}
	
	for(int ix=0; ix<sp; ix++){ 
		for(int iy=0; iy<sp; iy++){ 
			fg1[ix][iy] = fg1h[ix][iy]==0 ? fg1[ix][iy] : fg1[ix][iy]/fg1h[ix][iy]; 
		}
	}
	
	return fg1; 
}


END_NAMESPACE_STIR
