#include "stir/analytic/SRT2D/SRT2DReconstruction.h"
#include "stir/VoxelsOnCartesianGrid.h"
#include "stir/ProjDataInfoCylindricalArcCorr.h" 
#include "stir/ArcCorrection.h"
#include "stir/SSRB.h"
#include "stir/ProjDataInMemory.h"
#include "stir/Array.h"
#include <vector>
#include "stir/Sinogram.h" 
#include "stir/Viewgram.h" 
#include <math.h>

using std::cerr;
using std::endl;

START_NAMESPACE_STIR


void 
SRT2DReconstruction::
set_defaults()
{
  base_type::set_defaults();
  thres_restr_bound=-pow(10,6); 
  num_segments_to_combine = -1;
}

void 
SRT2DReconstruction::initialise_keymap()
{
  base_type::initialise_keymap();

  parser.add_start_key("SRT2DParameters");
  parser.add_stop_key("End");
  parser.add_key("num_segments_to_combine with SSRB", &num_segments_to_combine);
  parser.add_key("threshold for restriction within boundary", &thres_restr_bound);
  parser.add_key("threshold_per slice for restriction within boundary", &thres_restr_bound_vector);
}

void 
SRT2DReconstruction::
ask_parameters()
{ 
  base_type::ask_parameters();
  num_segments_to_combine = ask_num("num_segments_to_combine (must be odd)",-1,101,-1);
  thres_restr_bound=ask_num("threshold for restriction within boundary",-pow(10,6),pow(10,6),-pow(10,6));

}

bool SRT2DReconstruction::post_processing()
{
   if (base_type::post_processing())
    return true;
  if (num_segments_to_combine>=0 && num_segments_to_combine%2==0)
    {
      warning("num_segments_to_combine has to be odd (or -1), but is %d\n", num_segments_to_combine);
      return true;
    }

  if (num_segments_to_combine==-1)
    {
      const ProjDataInfoCylindrical * proj_data_info_cyl_ptr =
	dynamic_cast<const ProjDataInfoCylindrical *>(proj_data_ptr->get_proj_data_info_ptr());

      if (proj_data_info_cyl_ptr==0)
        num_segments_to_combine = 1; //cannot SSRB non-cylindrical data yet
      else
	{
	  if (proj_data_info_cyl_ptr->get_min_ring_difference(0) != 
	      proj_data_info_cyl_ptr->get_max_ring_difference(0)
	      ||
	      proj_data_info_cyl_ptr->get_num_segments()==1)
	    num_segments_to_combine = 1;
	  else
	    num_segments_to_combine = 3;
	}
    }
  return false;
}



string SRT2DReconstruction::method_info() const
{
  return "SRT2D";
}

SRT2DReconstruction::
SRT2DReconstruction(const string& parameter_filename)
{  
  initialise(parameter_filename); 
  cerr<<parameter_info() << endl; 
}

SRT2DReconstruction::SRT2DReconstruction()
{
  set_defaults();
}

SRT2DReconstruction::
SRT2DReconstruction(const shared_ptr<ProjData>& proj_data_ptr_v, const float thres_restr_bound_v)
{ 
  set_defaults();
  proj_data_ptr = proj_data_ptr_v;
  thres_restr_bound=thres_restr_bound_v;
}

Succeeded 
SRT2DReconstruction::
actual_reconstruct(shared_ptr<DiscretisedDensity<3,float> > const & density_ptr)
{

	// perform SSRB
	if (num_segments_to_combine>1) {  
		const ProjDataInfoCylindrical& proj_data_info_cyl =
			dynamic_cast<const ProjDataInfoCylindrical&> (*proj_data_ptr->get_proj_data_info_ptr());

		//  full_log << "SSRB combining " << num_segments_to_combine 
		//           << " segments in input file to a new segment 0\n" << endl; 

		shared_ptr<ProjDataInfo> 
			ssrb_info_sptr(SSRB(proj_data_info_cyl, num_segments_to_combine,1, 0, (num_segments_to_combine-1)/2 ));
		shared_ptr<ProjData> 
			proj_data_to_FBP_ptr(new ProjDataInMemory (proj_data_ptr->get_exam_info_sptr(), ssrb_info_sptr));
		SSRB(*proj_data_to_FBP_ptr, *proj_data_ptr);
		proj_data_ptr = proj_data_to_FBP_ptr;
	} else {
		// just use the proj_data_ptr we have already
	}

	// check if segment 0 has direct sinograms
	{
		const float tan_theta = proj_data_ptr->get_proj_data_info_ptr()->get_tantheta(Bin(0,0,0,0));
		if(fabs(tan_theta ) > 1.E-4) {
			warning("SRT2D: segment 0 has non-zero tan(theta) %g", tan_theta);
			return Succeeded::no;
		}
	}

	/*float tangential_sampling;*/
	// TODO make next type shared_ptr<ProjDataInfoCylindricalArcCorr> once we moved to boost::shared_ptr
	// will enable us to get rid of a few of the ugly lines related to tangential_sampling below
	shared_ptr<ProjDataInfo> arc_corrected_proj_data_info_sptr;

	// arc-correction if necessary
	ArcCorrection arc_correction;
	bool do_arc_correction = false;
	if (dynamic_cast<const ProjDataInfoCylindricalArcCorr*> (proj_data_ptr->get_proj_data_info_ptr()) != 0) {
		// it's already arc-corrected
		arc_corrected_proj_data_info_sptr = proj_data_ptr->get_proj_data_info_ptr()->create_shared_clone();
		/*tangential_sampling = 
			dynamic_cast<const ProjDataInfoCylindricalArcCorr&> 
				(*proj_data_ptr->get_proj_data_info_ptr()).get_tangential_sampling();*/  
	} else {
		// TODO arc-correct to voxel_size
		if (arc_correction.set_up(proj_data_ptr->get_proj_data_info_ptr()->create_shared_clone()) == Succeeded::no)
			return Succeeded::no;
		do_arc_correction = true;
		// TODO full_log
		warning("FBP2D will arc-correct data first");
		arc_corrected_proj_data_info_sptr = arc_correction.get_arc_corrected_proj_data_info_sptr();
		/*tangential_sampling = arc_correction.get_arc_corrected_proj_data_info().get_tangential_sampling();*/
	}

	VoxelsOnCartesianGrid<float>& image = dynamic_cast<VoxelsOnCartesianGrid<float>&>(*density_ptr);	
	Sinogram<float> sino = proj_data_ptr->get_empty_sinogram(0,0); 
	Viewgram<float> view = proj_data_ptr->get_empty_viewgram(0,0); 
	Viewgram<float> view1 = proj_data_ptr->get_empty_viewgram(0,0); 
	/*cerr << "ax_min = " << proj_data_ptr->get_min_axial_pos_num(0) << 
	", ax_max = " << proj_data_ptr->get_max_axial_pos_num(0) << 
	", img_min = " << image.get_min_y() << 
	", img_max = " << image.get_max_y() << 
	", img_siz = " << image.get_y_size() << 
	 endl; */
	
	const int sp = proj_data_ptr->get_num_tangential_poss(); 
	const int sth = proj_data_ptr->get_num_views();
	const int sa = proj_data_ptr->get_num_axial_poss(0);

	const int sx = image.get_x_size();
	const int sy = image.get_y_size();
	const int sx2 = ceil(sx/2.0), sy2 = ceil(sy/2.0); 

	//The rest of the variables used by the program.
	int ia, image_pos;
	int ith, jth, ip, ix1, ix2; 

	float x, aux, dh[8], z[8]; //ff, pp2
	
	const int image_min_x = image.get_min_x();
	const int image_min_y = image.get_min_y();

	float th[sth], p[sp], p_ud[sp], x1[sx], x2[sy]; //hx[sth]
	float f[sth][sp], ddf[sth][sp];
	float f_ud[sth][sp], ddf_ud[sth][sp];
	float f1[sth][sp], ddf1[sth][sp];
	float f1_ud[sth][sp], ddf1_ud[sth][sp];
	
	float lg[sp], termC[sth]; 
	const float dp6 = 6.0/4.0*2.0/(sp-1.0);
	
	//Some constants.
	//pp2= -1.0/(4*M_PI*M_PI); 
	
	// Put theta and p in arrays.
	for(ith=0; ith<sth; ith++) 
		th[ith]=ith*M_PI/sth; 
	
	for(ip=0; ip<sp; ip++) 
		p[ip]=-1.0+2.0*ip/(sp-1);
	for(ip=0; ip<sp; ip++) 
		p_ud[sp-ip-1]=p[ip];
	
	// Put x1 and x2 in arrays.
	for(ix1=0; ix1<sx; ix1++) 
		x1[ix1]=-1.0+2.0*ix1/(sx-1); 
	
	for(ix2=0; ix2<sy; ix2++) 
		x2[ix2]=-1.0+2.0*ix2/(sy-1);
	
	// Calculate constants 
	//dp = p[1]-p[0];
	//dp6 = 6.0/4.0*dp;

	for(ia=0; ia<sa; ia++) { 
		for(ix1=0; ix1<sx; ix1++){ 
			for(ix2=0; ix2<sy; ix2++){  
				image_pos= image.get_min_z() + 2*(ia - proj_data_ptr->get_min_axial_pos_num(0));
				
				image[image_pos][image_min_x +sx-ix1-1][image_min_y +ix2] = 0; 
			}
		}
	}
	
	// Starting calculations per view
	// 2D algorithm only   
	// special case of ith=0 
	view = proj_data_ptr->get_viewgram(0, 0);
	if (do_arc_correction) { 
		  view = arc_correction.do_arc_correction(view);
	}
	for(ia=0; ia<sa; ia++){
		for(ip=0; ip<sp; ip++) {
			f[ia][ip] = view[view.get_min_axial_pos_num() + ia][view.get_min_tangential_pos_num() + ip];
		}
		spline(p,f[ia],sp,ddf[ia]);
	}
	for(ia=0; ia<sa; ia++) { 
		termC[ia] = (ddf[ia][0]*(3*p[1]-p[0]) + ddf[ia][sp-1]*(p[sp-1]-3.0*p[sp-2]))/4.0;
		for (ip=0; ip<sp; ip++) {
			termC[ia] += dp6*ddf[ia][ip];
		}
	}
	for(ix1=0; ix1<sx2; ix1++){
		for(ix2=0; ix2<sy2; ix2++){  
			aux=sqrt(1.0-x2[ix2]*x2[ix2]);
			if(fabs(x2[ix2]) >= 1.0 || fabs(x1[ix1]) >= aux){ 
				continue;
			}		
			x=x2[ix2]*cos(th[ith])-x1[ix1]*sin(th[ith]); 
			for (ip=0; ip<sp; ip++) {
				lg[ip] = log(fabs(x-p[ip])); 
			}
			for(ia=0; ia<sa; ia++){
					image[2*ia][image_min_x +sx-ix1-1][image_min_y +ix2] 
							= -hilbert_der(x, f[ia], ddf[ia], p, sp, lg, termC[ia])/(M_PI*sth*(sp-1)); 
			}
		}
	}
	// general case, ith=1...sth-1
	for(ith=1; ith<sth; ith++){
		//image_pos= image.get_min_z() + 2*(ia - proj_data_ptr->get_min_axial_pos_num(0));
		//std::cerr << "\nView " << ith << " of " << sth << std::endl;
		
		// Loading related viewgrams 
		view = proj_data_ptr->get_viewgram(ith, 0);
		view1 = proj_data_ptr->get_viewgram(sth-ith, 0);
		if (do_arc_correction) { 
		  view = arc_correction.do_arc_correction(view);
		  view1 = arc_correction.do_arc_correction(view1);
		}
		
		for(ia=0; ia<sa; ia++){
			for(ip=0; ip<sp; ip++) {
				f[ia][ip] = view[view.get_min_axial_pos_num() + ia][view.get_min_tangential_pos_num() + ip];
				f_ud[ia][sp-ip-1] = f[ia][ip]; 
				f1[ia][ip] = view1[view.get_min_axial_pos_num() + ia][view.get_min_tangential_pos_num() + ip];
				f1_ud[ia][sp-ip-1] = f1[ia][ip]; 
		    }
		}
		
		// Calculation of second derivative by use of function spline
		for(ia=0; ia<sa; ia++){ 
			spline(p,f[ia],sp,ddf[ia]);
			spline(p,f1[ia],sp,ddf1[ia]);
			for(ip=0; ip<sp; ip++) { 
				ddf_ud[ia][sp-ip-1] = ddf[ia][ip]; 
				ddf1_ud[ia][sp-ip-1] = ddf1[ia][ip]; 
			}
		}
		

		for(ia=0; ia<sa; ia++) { 
			termC[ia] = (ddf[ia][0]*(3*p[1]-p[0]) + ddf[ia][sp-1]*(p[sp-1]-3.0*p[sp-2]))/4.0;
			for (ip=0; ip<sp; ip++) {
				termC[ia] += dp6*ddf[ia][ip];
			}
		}
		
		//Starting the calculation of ff(x1,x2).
		for(ix1=0; ix1<sx2; ix1++){
			for(ix2=0; ix2<sy2; ix2++){  
				// If x1,x2 off range put ff(x1,x2)=0
				aux=sqrt(1.0-x2[ix2]*x2[ix2]);
				if(fabs(x2[ix2]) >= 1.0 || fabs(x1[ix1]) >= aux){ 
					continue;
				}
				
				// Computation of h_rho			
				x=x2[ix2]*cos(th[ith])-x1[ix1]*sin(th[ith]); 
				for (ip=0; ip<sp; ip++) {
					lg[ip] = log(fabs(x-p[ip])); 
				}
				
				for(ia=0; ia<sa; ia++){
					//image_pos= image.get_min_z() + 2*(ia - proj_data_ptr->get_min_axial_pos_num(0));
					image_pos= 2*ia;
					
					image[image_pos][image_min_x +sx-ix1-1][image_min_y +ix2] 
							+= -hilbert_der(x, f[ia], ddf[ia], p, sp, lg, termC[ia])/(M_PI*sth*(sp-1)); // bot-left
					if(ix2<sy2){ 
						image[image_pos][image_min_x +sx-ix1-1][image_min_y +sy-ix2-1] 
							+= -hilbert_der(x, f1[ia], ddf1[ia], p, sp, lg, termC[ia])/(M_PI*sth*(sp-1)); // bot-right
					}
					if(ix1<sx2){ 
						image[image_pos][image_min_x +ix1][image_min_y +ix2] 
								-= -hilbert_der(-x, f1_ud[ia], ddf1_ud[ia], p_ud, sp, lg, termC[ia])/(M_PI*sth*(sp-1));// top-left
					} 
					if(ix1<sx2&&ix2<sy2){ 
						image[image_pos][image_min_x +ix1][image_min_y +sy-ix2-1] 
								-= -hilbert_der(-x, f_ud[ia], ddf_ud[ia], p_ud, sp, lg, termC[ia])/(M_PI*sth*(sp-1)); // top-right
					}
					/*if(ith<ceil(sth/2.0)){ 
						jth = (int)(ceil(sth/2.0))-ith; 
						dh[4] -= hilbert_der(-x, f_ud[jth], ddf_ud[jth], p_ud, sp, lg, termC[jth]);
						dh[5] += hilbert_der(x, f[sth-jth], ddf[sth-jth], p, sp, lg, termC[jth]);
						dh[6] -= hilbert_der(-x, f_ud[sth-jth], ddf_ud[sth-jth], p_ud, sp, lg, termC[jth]);
						dh[7] += hilbert_der(x, f[jth], ddf[jth], p, sp, lg, termC[jth]);
					} else if(ith>ceil(sth/2.0)) { 
						jth = (int)ceil(3*sth/2)-ith;
						dh[4] += hilbert_der(x, f[jth], ddf[jth], p, sp, lg, termC[jth]);
						dh[5] -= hilbert_der(-x, f_ud[sth-jth], ddf_ud[sth-jth], p_ud, sp, lg, termC[jth]);
						dh[6] += hilbert_der(x, f[sth-jth], ddf[sth-jth], p, sp, lg, termC[jth]);
						dh[7] -= hilbert_der(-x, f_ud[jth], ddf_ud[jth], p_ud, sp, lg, termC[jth]);
					}*/
					
				}
				
				// Ending the calculation of ff(x1,x2)
				//ff = pp2*integ(M_PI,sth,hx); 
				//ff = -dh/(M_PI*sth*(sp-1));
				/*dh[0] = -dh[0]/(M_PI*sth*(sp-1));
				dh[1] = -dh[1]/(M_PI*sth*(sp-1));
				dh[2] = -dh[2]/(M_PI*sth*(sp-1));
				dh[3] = -dh[3]/(M_PI*sth*(sp-1));
				
				dh[4] = -dh[4]/(M_PI*sth*(sp-1));
				dh[5] = -dh[5]/(M_PI*sth*(sp-1));
				dh[6] = -dh[6]/(M_PI*sth*(sp-1));
				dh[7] = -dh[7]/(M_PI*sth*(sp-1));*/
				
				//image[image_pos][image_min_x +sx-ix1-1][image_min_y + ix2] = ff; 
				/*image[image_pos][image_min_x +sx-ix1-1][image_min_y +ix2] = dh[0]; 
				image[image_pos][image_min_x +sx-ix1-1][image_min_y +sy-ix2-1] = dh[1]; 
				image[image_pos][image_min_x +ix1][image_min_y +ix2] = dh[2]; 
				image[image_pos][image_min_x +ix1][image_min_y +sy-ix2-1] = dh[3]; 
				
				image[image_pos][image_min_x +sx-ix2-1][image_min_y +ix1] = dh[4]; 
				image[image_pos][image_min_x +ix2][image_min_y +ix1] = dh[5]; 
				image[image_pos][image_min_x +sx-ix2-1][image_min_y +sx-ix1-1] = dh[6]; 
				image[image_pos][image_min_x +ix2][image_min_y +sx-ix1-1] = dh[7];*/
			}  
		} 
	}
	//image = -image/(M_PI*sth*(sp-1)); 

	return Succeeded::yes;
}
 
float SRT2DReconstruction::hilbert_der(float x, float f[], float ddf[], float p[], int sp, float lg[], float termC) {

	float term, trm0, termd0, trm1, termd; 
	float d, d_div_6, minus_half_div_d;
	
	d = p[1]-p[0]; 
	d_div_6 = d/6.0;
	minus_half_div_d = -0.5/d;

	term = 0.5*(ddf[sp-2] - ddf[0])*x + termC;

	term += ((f[sp-1]-f[sp-2])/d + 
			ddf[sp-2]*(d_div_6 + minus_half_div_d*(p[sp-1]-x)*(p[sp-1]-x)) + 
			ddf[sp-1]*(-d_div_6 - minus_half_div_d*(p[sp-2]-x)*(p[sp-2]-x)))*lg[sp-1];

	trm0 = d_div_6 + minus_half_div_d*(p[1]-x)*(p[1]-x); 
	termd0 = (f[1]-f[0])/d + ddf[0]*trm0 + ddf[1]*(-d_div_6 - minus_half_div_d*(p[0]-x)*(p[0]-x));

	term -= termd0 * lg[0];  

	for (int ip=0; ip<sp-2; ip++) {    
		trm1 = d_div_6 + minus_half_div_d*(p[ip+2]-x)*(p[ip+2]-x);
		termd =  (f[ip+2]-f[ip+1])/d  + ddf[ip+1]*trm1 - ddf[ip+2]*trm0; 
		term += (termd0-termd) * lg[ip+1]; 
		termd0 = termd;
		trm0 = trm1; 
	}
	
	return term; 
}

void SRT2DReconstruction::spline(float x[],float y[],int n, float y2[]) {
	// function for nanural qubic spline.
	int i, k;
	float p, qn, sig, un;
	float u[n];
	y2[0]=0.0;
	u[0]=0.0;
	for(i=1; i<n-1; i++) {
		sig=(x[i]-x[i-1])/(x[i+1]-x[i-1]);
		p=sig*y2[i-1]+2.0;
		y2[i]=(sig-1.0)/p;
		u[i]=(6.0*((y[i+1]-y[i])/(x[i+1]-x[i])-(y[i]-y[i-1])/(x[i]-x[i-1]))/(x[i+1]-x[i-1])-sig*u[i-1])/p;
	}
	qn=0.0;
	un=0.0;
	y2[n-1]=(un-qn*u[n-2])/(qn*y2[n-2]+1.0);
	for(k=n-2; k>=0; k--)
		y2[k]=y2[k]*y2[k+1]+u[k];
	return;
}


float SRT2DReconstruction::integ(float dist, int max, float ff[]) {
	// function for the calculation of integrals (closed formula).
	int k, intg;
	intg=ff[0];
	for(k=1; k<max; k++) {
		intg+=ff[k];
	}
	return intg*dist/max;
}



END_NAMESPACE_STIR