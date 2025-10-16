/*
    Copyright (C) 2025 University College London

    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0 AND License-ref-PARAPET-license

    See STIR/LICENSE.txt for details
*/
/*!
  \file
  \ingroup DDSR2D 
  \brief Implementation of class stir::DDSR2DReconstruction

  \author Dimitra Kyriakopoulou
  \author Kris Thielemans
 
*/

#include "stir/analytic/DDSR2D/DDSR2DReconstruction.h"
#include "stir/VoxelsOnCartesianGrid.h"
#include "stir/RelatedViewgrams.h"
#include "stir/recon_buildblock/BackProjectorByBinUsingInterpolation.h"
#include "stir/Bin.h"
#include "stir/round.h"
#include "stir/display.h"
#include <algorithm>
#include "stir/IO/interfile.h"
#include "stir/info.h"

#include "stir/Sinogram.h" 
#include "stir/Viewgram.h"
#include <complex> 
#include <math.h>
#include "stir/numerics/fourier.h"
#include "stir/IO/read_from_file.h"
#include <boost/format.hpp>

#ifdef STIR_OPENMP
#include <omp.h>
#endif
#include "stir/num_threads.h"

#include <vector>
#include <iostream>
#include <cmath> // For M_PI and other math functions
#ifndef M_PI
#  define M_PI 3.14159265358979323846
#endif

START_NAMESPACE_STIR

const char * const
DDSR2DReconstruction::registered_name =
  "DDSR2D";

void 
DDSR2DReconstruction::
set_defaults()
{
  base_type::set_defaults();
  display_level=0; // no display 
  
  attenuation_map_filename = ""; 
  noise_filter = -1;
  noise_filter2 = -1;
}

void 
DDSR2DReconstruction::initialise_keymap()
{
  base_type::initialise_keymap();

  parser.add_start_key("DDSR2DParameters");
  parser.add_stop_key("End");
  parser.add_key("Display level",&display_level);

  parser.add_key("attenuation map file",&attenuation_map_filename);
  parser.add_key("noise filter",&noise_filter);
  parser.add_key("noise filter 2",&noise_filter2);
}

void 
DDSR2DReconstruction::
ask_parameters()
{ 
   
  base_type::ask_parameters();

  attenuation_map_filename = ask_string("filename of attenuation map:");
  noise_filter = ask_num("noise filter (-1 to disable)",0,1,-1);
  noise_filter2 = ask_num("noise filter 2 (-1 to disable)",0,1,-1); 

  display_level = ask_num("Which images would you like to display \n\t(0: None, 1: Final, 2: filtered viewgrams) ? ", 0,2,0);

#if 0
    // do not ask the user for the projectors to prevent them entering
    // silly things
  do 
    {
      back_projector_sptr =
	BackProjectorByBin::ask_type_and_parameters();
    }
#endif

}

bool DDSR2DReconstruction::post_processing()
{
  return base_type::post_processing();
}

Succeeded
DDSR2DReconstruction::
set_up(shared_ptr <DDSR2DReconstruction::TargetT > const& target_data_sptr)
{
  if (base_type::set_up(target_data_sptr) == Succeeded::no)
    return Succeeded::no;

	if (attenuation_map_filename.length() == 0) 
     error(boost::format("You need to specify an attenuation map file"));

	if (noise_filter > 1)
     error(boost::format("Noise filter has to be between 0 and 1 but is %g") % noise_filter);    

	if (noise_filter2 > 1)
     error(boost::format("Noise filter 2 has to be between 0 and 1 but is %g") % noise_filter2);    
	
	// TODO improve this, drop "stir/IO/read_from_file.h" if possible
	atten_data_ptr= read_from_file<DiscretisedDensity<3,float> >(attenuation_map_filename);
    {
      // --- fail-fast consistency checks ---
      const auto& tgt =
          dynamic_cast<const VoxelsOnCartesianGrid<float>&>(*target_data_sptr); // output image grid
      const auto& att =
          dynamic_cast<const VoxelsOnCartesianGrid<float>&>(*atten_data_ptr);   // attenuation image grid

      // Same grid (sizes, voxel sizes, origin, orientation). Also compares z-size.
      if (!tgt.has_same_characteristics(att))
        error("DDSR2D: target and attenuation images must have identical grid characteristics.");

      // z-size relation used by the  by the current implementation:
     // each sinogram plane iz produces two identical output slices (indices 2*iz and 2*iz+1),
     // hence nz_img must equal 2 × (# axial sinogram planes).
      const int nz_img  = tgt.get_z_size();                     // nz_img := # z-slices in output image
      const int nz_sino = proj_data_ptr->get_num_axial_poss(0); // nz_sino := # axial sinogram planes (segment 0)

      if (nz_img != 2 * nz_sino)
        error(boost::format("DDSR2D: expected output (and attenuation) z-size = 2 x #axial sinogram planes (got %1% vs 2x%2%)")
            % nz_img % nz_sino);

    }

	
  return Succeeded::yes;
}

std::string DDSR2DReconstruction::method_info() const
{
  return "DDSR2D";
}

DDSR2DReconstruction::
DDSR2DReconstruction(const std::string& parameter_filename)
{  
  initialise(parameter_filename);
  //std::cerr<<parameter_info() << std::endl;
  info(boost::format("%1%") % parameter_info());
}

DDSR2DReconstruction::DDSR2DReconstruction()
{
  set_defaults();
}

DDSR2DReconstruction::
DDSR2DReconstruction(const shared_ptr<ProjData>& proj_data_ptr_v,  
			const shared_ptr<DiscretisedDensity<3,float> >& atten_data_ptr_v,
			const double noise_filter_v,  
			const double noise_filter2_v		    
)
{
  set_defaults();

  
  proj_data_ptr = proj_data_ptr_v;
  atten_data_ptr = atten_data_ptr_v; 
  noise_filter = noise_filter_v; 
  noise_filter2 = noise_filter2_v; 
  // have to check here because we're not parsing
//  if (post_processing_only_DDSR2D_parameters() == true)
//    error("DDSR2D: Wrong parameter values. Aborting\n");
} 

Succeeded 
DDSR2DReconstruction::
actual_reconstruct(shared_ptr<DiscretisedDensity<3,float> > const & density_ptr)
{


  // check if segment 0 has direct sinograms
  {
    const float tan_theta = proj_data_ptr->get_proj_data_info_sptr()->get_tantheta(Bin(0,0,0,0));
    if(fabs(tan_theta ) > 1.E-4)
      {
	warning("DDSR2D: segment 0 has non-zero tan(theta) %g", tan_theta);
	return Succeeded::no;
      }
  }

  
  VoxelsOnCartesianGrid<float>& image =
    dynamic_cast<VoxelsOnCartesianGrid<float>&>(*density_ptr);
  VoxelsOnCartesianGrid<float>& atten =
    dynamic_cast<VoxelsOnCartesianGrid<float>&>(*atten_data_ptr);
  
  density_ptr->fill(0);
  
	// SPECT does not have oblique projections
	Sinogram<float> sino = proj_data_ptr->get_empty_sinogram(0,0); 
	
	// Some constants 
	const int sp = sino.get_num_tangential_poss(), sth = sino.get_num_views(); 
	float dp = 2.0/(sp-1); // dp := normalised detector spacing along tangential axis
	int p_cutoff = 0; // p_cutoff := frequency-domain low-pass cutoff (from noise_filter)
	if(noise_filter <= 0) { 
		p_cutoff = 0; 
	} else { 
		noise_filter = noise_filter > 1 ? 1 : noise_filter; 
		p_cutoff = floor(floor(sp/2.0)*noise_filter); 
	}
  int q_cutoff = 0; // q_cutoff := secondary frequency-domain cutoff (from noise_filter2)
//float noise_filter2=-1;
  if(noise_filter2 <= 0) { 
		q_cutoff = 0; 
	} else { 
		noise_filter2 = noise_filter2 > .5 ? .5 : noise_filter2; 
		q_cutoff = floor(floor(sp/2.0)*noise_filter2); 
	}
 
	Array<1, std::complex<float> > W(sp), temp2(sp); // W := cosine taper window in frequency; temp2 := temp spectrum buffer
	Array<1, float> g(sp); // g := projection samples at current view
	W.fill(0); 
	for(int ip=sp/2-p_cutoff; ip<sp/2; ip++) { //fftshift
		W[ip+sp/2] = 1/2.0*(1 + cos(1.0*M_PI*(ip+1-floor(sp/2))/p_cutoff));
	}
	for(int ip=sp/2; ip<sp/2+p_cutoff; ip++) { //fftshift
		W[ip-sp/2] = 1/2.0*(1 + cos(1.0*M_PI*(ip+1-floor(sp/2))/p_cutoff));
	}
	
	
	// Generate partitions 
	std::vector<float> pn(sp,0.0f); // thn := view angles grid (radians)
  std::vector<float> thn(sth,0.0f); // pn := normalised detector coordinate grid
  std::vector<float> xn(sp,0.0f); // xn := normalised image x-grid
  std::vector<float> yn(sp,0.0f); // yn := normalised image y-grid
	for(int ith=0; ith<sth; ith++) thn[ith] = ith*2.0*M_PI/sth;
	for(int ip=0; ip<sp; ip++) pn[ip] = -1.0+2.0*ip/(sp-1);
	for(int ix=0; ix<sp; ix++) xn[ix] = -1.0+2.0*ix/(sp-1);
	for(int iy=0; iy<sp; iy++) yn[iy] = -1.0+2.0*iy/(sp-1);

	
	
	// Assume that image and attenuation dimensions are consistent 
	// Assume z output size is consistent with attenuation 
	
	//const int min_tang = sino.get_min_tangential_pos_num(); 
	const int min_xy = atten.get_min_x(); // min_xy := min index of atten/image grid in x (assumed square)
	const int min_xy_img = image.get_min_x(); // min_xy_img := min index of output image grid in x
	
	// pad to the next power of 2 for FFT
	const int pad = pow(2,ceil(log2(2*sp))); // pad := FFT length (next power of two for 2*sp)
	std::cout << "Tangential poss is " << sp << ", FFT length is " << pad << std::endl; 
	
	//float A_hlb[sp], hc[sp], hs[sp], h1[sp], h2[sp], m[sp];
	std::vector<float> A_hlb(sp,0.0f); // A_hlb := Hilbert transform of A (imag part of spectrum)
  std::vector<float> hc(sp,0.0f); // hc := cos(A_hlb) per ray
  std::vector<float> hs(sp,0.0f); // hs := sin(A_hlb) per ray
  std::vector<float> h1(sp,0.0f); // h1 := exp(A)*g*cos(A_hlb)
  std::vector<float> h2(sp,0.0f); // h2 := exp(A)*g*sin(A_hlb)
  std::vector<float> m(sp,0.0f); // m := attenuation-compensated combination
  Array<1,std::complex<float> > temp(pad), K_fft(pad); // temp := complex FFT buffer; K_fft := Hilbert kernel (1,2,...,2,1)
	K_fft[0] = 1.0; K_fft[pad/2] = 1.0; 
	for(int ip=1; ip<pad/2; ip++) K_fft[ip] = 2.0;
	
	//float expDbtM[sp][sp], M[sp][sp], s, p; 
	float s, p; // s := radial coordinate; p := tangential coordinate (in rotated frame)
  std::vector<std::vector<float>> expDbtM(sp, std::vector<float>(sp,0.0f)); // expDbtM := exp(dbt)*m buffer over path index s and detector p
  std::vector<std::vector<float>> M(sp, std::vector<float>(sp,0.0f)); // M := d/dp[exp(dbt)*m] (finite differences)
	//float dbt[sp][sp], A[sp], x, y, f1, f2, as[sp][sp]; 
  float x, y, f1, f2;  // x,y := rotated image coords; f1,f2 := bilinear weights
  std::vector<std::vector<float>> dbt(sp, std::vector<float>(sp,0.0f));  // dbt := path integral of attenuation along s (per p)
  std::vector<std::vector<float>> as(sp, std::vector<float>(sp,0.0f)); // as := sampled attenuation along path; intermediate for dbt
  std::vector<float> A(sp); // A := 0.5*dbt at s=0 (per p)
	int i, j; 
	
	for(int iz=proj_data_ptr->get_min_axial_pos_num(0); iz<=proj_data_ptr->get_max_axial_pos_num(0)-1; iz++) {
	std::vector<std::vector<float>> img(sp, std::vector<float>(sp,0.0f));  
	for(int i=0; i<sp; i++){ 
			for(int j=0; j<sp; j++){ 
				img[i][j] = 0; 
		  }
		}	

	atten[iz] = atten[iz]*sth/40.0f;
	
	std::cout << "Reconstructing slice " << (iz+1) << " of " << proj_data_ptr->get_num_axial_poss(0) << "..." << std::endl; 
	sino = proj_data_ptr->get_sinogram(iz, 0);
	
	for(int ith=0; ith<sth; ith++){ 
		// Perform noise filter
		g = sino[ith]; 
		g.set_offset(0); 
		if(p_cutoff > 0 || q_cutoff>0) { 
			for(int ip=0; ip<sp/2; ip++) temp2[ip+sp/2] = g[ip]; //fftshift
			for(int ip=sp/2; ip<sp; ip++) temp2[ip-sp/2] = g[ip]; 
			fourier_1d(temp2,1); 
			if(p_cutoff > 0) temp2=temp2*W; 

     if(q_cutoff > 0){
    // Apply filter in frequency domain
    for (int i = 0; i < sp; ++i) {
        float frequency = (i < sp / 2) ? i / (float)sp : (i - sp) / (float)sp;
        if (abs(frequency) > q_cutoff) {
            temp2[i] = 0; // Attenuate frequencies beyond the cutoff
        }
    }
  }


			inverse_fourier_1d(temp2); 
			for(int ip=0; ip<sp/2; ip++) g[ip+sp/2] = std::real(temp2[ip]); //fftshift
			for(int ip=sp/2; ip<sp; ip++) g[ip-sp/2] = std::real(temp2[ip]);
		} 
		
		// Evaluate divergent beam tranform
		for(int ip=0; ip<sp; ip++){ 
			for(int is=0; is<sp; is++){ 
				as[is][ip] = 0; 
				x = pn[is]*cos(thn[ith]) - pn[ip]*sin(thn[ith]); 
				y = pn[is]*sin(thn[ith]) + pn[ip]*cos(thn[ith]); 
				if(pow(x,2)+pow(y,2)>1) continue; 
				
				i = floor((sp-1)*(x+1)/2.0); 
				if(x==1) i = sp-2;
				j = floor((sp-1)*(y+1)/2.0); 
				if(y==1) j = sp-2;
				
				f1 = ((xn[i+1]-x)*atten[iz][i+min_xy][j+min_xy] + (x-xn[i])*atten[iz][i+1+min_xy][j+min_xy])/dp; 
				f2 = ((xn[i+1]-x)*atten[iz][i+min_xy][j+1+min_xy] + (x-xn[i])*atten[iz][i+1+min_xy][j+1+min_xy])/dp; 
				as[is][ip] = ((yn[j+1]-y)*f1 + (y-yn[j])*f2)/dp;
			}
			dbt[sp-1][ip] = 0;
			for(int is=sp-2; is>=0; is--){ 
				dbt[is][ip] = dbt[is+1][ip] + dp/2.0*(as[is][ip]+as[is+1][ip]);
			}
		}
		
		
		// Evaluate radon transforms 
		for(int ip=0; ip<sp; ip++){ 
			A[ip] = 0.5*dbt[0][ip]; 
		}
		
		
		// Evaluate hilbert transform of A(p)
		temp.fill(0); // this has to be vastly improved 
		for(int ip=0; ip<sp; ip++) temp[ip] = A[ip]; 
		fourier_1d(temp,1);
		temp *= K_fft; 
		inverse_fourier_1d(temp); 
		for(int ip=0; ip<sp; ip++) A_hlb[ip] = -std::imag(temp[ip]); 
		
		
		// calculate h_s, h_c
		for(int ip=0; ip<sp; ip++){ 
			hc[ip] = cos(A_hlb[ip]);
			hs[ip] = sin(A_hlb[ip]);
		}
		
		
		// calculate modified projections
		for(int ip=0; ip<sp; ip++){ 
			h1[ip] = hc[ip]*exp(A[ip])*g[ip];
			h2[ip] = hs[ip]*exp(A[ip])*g[ip]; 
		}
		
		
		temp.fill(0); 
		for(int ip=0; ip<sp; ip++) temp[ip] = h1[ip]; 
		fourier_1d(temp,1);
		temp *= K_fft; 
		inverse_fourier_1d(temp); 
		for(int ip=0; ip<sp; ip++) h1[ip] = -std::imag(temp[ip]); 
		
		temp.fill(0); 
		for(int ip=0; ip<sp; ip++) temp[ip] = h2[ip]; 
		fourier_1d(temp,1);
		temp *= K_fft; 
		inverse_fourier_1d(temp); 
		for(int ip=0; ip<sp; ip++) h2[ip] = -std::imag(temp[ip]); 
		
		
		for(int ip=0; ip<sp; ip++){ 
			m[ip] = exp(-A[ip])*(hc[ip]*h1[ip]+hs[ip]*hs[ip]);
		}
		
		
		// Differentiation 
		for(int is=0; is<sp; is++)
			for(int ip=0; ip<sp; ip++)
				expDbtM[is][ip] = exp(dbt[is][ip])*m[ip]; 
			
		for(int is=0; is<sp; is++){ 
			// 4-th order central finite difference 
			for(int ip=2; ip<sp-2; ip++){ 
				M[is][ip] = 1.0/(10.0*dp)*(
					1.0*( -expDbtM[is][ip+2] + expDbtM[is][ip-2]) + 
					8.0*( +expDbtM[is][ip+1] - expDbtM[is][ip-1]) );
			}
			// 4-th orted forward finite difference
			for(int ip=0; ip<2; ip++){ 
				M[is][ip] = 1.0/dp*( -25.0/12.0*expDbtM[is][ip] + 
					4.0*expDbtM[is][ip+1] - 3.0*expDbtM[is][ip+2] + 
					4.0/3.0*expDbtM[is][ip+3] - 1.0/4.0*expDbtM[is][ip+4] ); 
			}
			// 4-th order backward finite difference
			for(int ip=sp-2; ip<sp; ip++){ 
				M[is][ip] = 1.0/dp*( 25.0/12.0*expDbtM[is][ip] + 
					-4.0*expDbtM[is][ip-1] + 3.0*expDbtM[is][ip-2] + 
					-4.0/3.0*expDbtM[is][ip-3] + 1.0/4.0*expDbtM[is][ip-4] ); 
			}
		}
		 
		// Backprojection 
		for(int ix=0; ix<sp; ix++){ 
			for(int iy=0; iy<sp; iy++){ 
				if(pow(xn[ix],2)+pow(yn[iy],2)>1) continue; 
				s = xn[ix]*cos(thn[ith]) + yn[iy]*sin(thn[ith]); 
				p = -xn[ix]*sin(thn[ith]) + yn[iy]*cos(thn[ith]);
				
				i = floor((sp-1)*(s+1)/2.0); 
				if(s==1) i = sp-2;
				j = floor((sp-1)*(p+1)/2.0); 
				if(p==1) j = sp-2;
				
				f1 = ((pn[i+1]-s)*M[i][j] + (s-pn[i])*M[i+1][j])/dp; 
				f2 = ((pn[i+1]-s)*M[i][j+1] + (s-pn[i])*M[i+1][j+1])/dp; 
				img[ix][iy] += ((pn[j+1]-p)*f1 + (p-pn[j])*f2)/dp;
				
			}
		}

		
		

		
	}
	
if(image.get_x_size() != sp) { 
		// perform bilinear interpolation 
		if(iz==0) 
			std::cerr << "Image dimension mismatch, tangential positions " << sp << ", xy output " << image.get_x_size() << "\n Interpolating..." << std::endl; 
		int sx = image.get_x_size(); 
		int sy = sx; 
		//float xn1[sx], yn1[sy]; 
  std::vector<float> xn1(sx,0.0f), yn1(sy,0.0f);
		float dx = 2./(sp-1), dy = 2./(sp-1); 
		float val; 

		for(int ix=0; ix<sx; ix++) xn1[ix]=-1.0+2.0*ix/(sx-1); 
		for(int iy=0; iy<sy; iy++) yn1[iy]=-1.0+2.0*iy/(sy-1); 

		for(int ix=1; ix<sx-1; ix++) { 
			for(int iy=1; iy<sy-1; iy++) { 
				if(pow(xn1[ix],2)+pow(yn1[iy],2)>1) 
					val = 0.;
				else {					
					// bilinear interpolation 
					int y0 = (int) ((yn1[iy]-yn[0])/dy); 
					int x0 = (int) ((xn1[ix]-xn[0])/dx); 
					float tx = (xn1[ix]-xn[0])/dx - x0; 
					float ty = (yn1[iy]-yn[0])/dy - y0; 
					
					float ya = img[y0][x0]*tx + img[y0][x0+1]*(1.-tx); 
					float yb = img[y0+1][x0]*tx + img[y0+1][x0+1]*(1.-tx); 
					val = ya*ty + yb*(1.-ty); 
				}
				image[2*iz][ix+min_xy_img][sy-1-iy+min_xy_img] = val/sp/sp*3.362; 
				image[2*iz+1][ix+min_xy_img][sy-1-iy+min_xy_img] = val/sp/sp*3.362; 
			} 
		}

	} else { 	
		for(int ix=1; ix<=sp-1; ix++) { 
				for(int iy=1; iy<=sp-1; iy++) { 
					if(pow(xn[ix],2)+pow(yn[iy],2)>1) 
						image[iz][ix+min_xy_img][sp-1-iy+min_xy_img] = 0; 
					else
						image[2*iz][ix+min_xy_img][sp-1-iy+min_xy_img] = img[iy][ix]/sp/sp*3.362;   
						image[2*iz+1][ix+min_xy_img][sp-1-iy+min_xy_img] = img[iy][ix]/sp/sp*3.362; 
				}
			}
	}	
	}
	
	
  if (display_level>0)
    display(image, image.find_max(), "DDSR image");

  return Succeeded::yes;
}

 

END_NAMESPACE_STIR
