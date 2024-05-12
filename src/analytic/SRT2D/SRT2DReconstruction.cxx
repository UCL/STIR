//author Dimitra Kyriakopoulou

#include "stir/analytic/SRT2D/SRT2DReconstruction.h"
#include "stir/VoxelsOnCartesianGrid.h"
#include "stir/RelatedViewgrams.h"
#include "stir/recon_buildblock/BackProjectorByBinUsingInterpolation.h"
#include "stir/ProjDataInfoCylindricalArcCorr.h" 
#include "stir/ArcCorrection.h"
#include "stir/SSRB.h"
#include "stir/ProjDataInMemory.h"
// #include "stir/ProjDataInterfile.h"
#include "stir/Bin.h"
#include "stir/round.h"
#include "stir/display.h" 
#include <algorithm>
#include "stir/IO/interfile.h"  
#include "stir/info.h" 
#include <boost/format.hpp>   
 
using std::cerr;  
using std::endl; 
#ifdef STIR_OPENMP
#include <omp.h>
#endif
#include "stir/num_threads.h"

#include <cmath>// For M_PI and other math functions
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include <vector>
#include <algorithm> 

START_NAMESPACE_STIR

const char * const
SRT2DReconstruction::registered_name =
  "SRT2D";

void 
SRT2DReconstruction::
set_defaults()
{
  base_type::set_defaults();
  thres_restr_bound=-pow(10,6); 
  num_segments_to_combine = -1; 
  zoom=1.0;
  filter_wiener=1; 
  filter_median=0; 
  filter_gamma=1; 
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
  parser.add_key("zoom", &zoom);
  parser.add_key("wiener filter", &filter_wiener);
  parser.add_key("median filter", &filter_median);
  parser.add_key("gamma filter", &filter_gamma);
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
  return base_type::post_processing();
}

Succeeded
SRT2DReconstruction::
set_up(shared_ptr <SRT2DReconstruction::TargetT > const& target_data_sptr)
{
  if (base_type::set_up(target_data_sptr) == Succeeded::no)
    return Succeeded::no;

  if (num_segments_to_combine>=0 && num_segments_to_combine%2==0)
    error(boost::format("num_segments_to_combine has to be odd (or -1), but is %d") % num_segments_to_combine);

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

  //if (is_null_ptr(back_projector_sptr))
  //  error("Back projector not set.");

  return Succeeded::yes;
}

std::string SRT2DReconstruction::method_info() const
{
  return "SRT2D";
}

SRT2DReconstruction::
SRT2DReconstruction(const std::string& parameter_filename)
{  
  initialise(parameter_filename); 
  //std::cerr<<parameter_info() << endl; 
	info(boost::format("%1%") % parameter_info());
}

SRT2DReconstruction::SRT2DReconstruction()
{
  set_defaults();
}

SRT2DReconstruction::
SRT2DReconstruction(const shared_ptr<ProjData>& proj_data_ptr_v, const float thres_restr_bound_v,
		      const int num_segments_to_combine_v, const float zoom_v, const int filter_wiener_v, 
			  const int filter_median_v, const int filter_gamma_v)
{ 
  set_defaults();
  proj_data_ptr = proj_data_ptr_v;
  thres_restr_bound=thres_restr_bound_v;
	num_segments_to_combine = num_segments_to_combine_v;
  zoom=zoom_v;
  filter_wiener=filter_wiener_v; 
  filter_median=filter_median_v; 
  filter_gamma=filter_gamma_v; 
}

Succeeded 
SRT2DReconstruction::
actual_reconstruct(shared_ptr<DiscretisedDensity<3,float> > const & density_ptr)
{

// In case of 3D data, use only direct sinograms 
	// perform SSRB 
  if (num_segments_to_combine>1)
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
			proj_data_to_FBP_ptr(new ProjDataInMemory (proj_data_ptr->get_exam_info_sptr(), ssrb_info_sptr));
		SSRB(*proj_data_to_FBP_ptr, *proj_data_ptr);
		proj_data_ptr = proj_data_to_FBP_ptr;
    }
  else
    {
		// just use the proj_data_ptr we have already
	}

	// check if segment 0 has direct sinograms
	{
    const float tan_theta = proj_data_ptr->get_proj_data_info_sptr()->get_tantheta(Bin(0,0,0,0));
    if(fabs(tan_theta ) > 1.E-4)
      {
	warning("SRT2D: segment 0 has non-zero tan(theta) %g", tan_theta);
			return Succeeded::no;
		}
	}

  float tangential_sampling;
	// TODO make next type shared_ptr<ProjDataInfoCylindricalArcCorr> once we moved to boost::shared_ptr
	// will enable us to get rid of a few of the ugly lines related to tangential_sampling below
  shared_ptr<const ProjDataInfo> arc_corrected_proj_data_info_sptr;

	// arc-correction if necessary
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
		warning("FBP2D will arc-correct data first");
      arc_corrected_proj_data_info_sptr =
	arc_correction.get_arc_corrected_proj_data_info_sptr();
      tangential_sampling =
	arc_correction.get_arc_corrected_proj_data_info().get_tangential_sampling();  
	}

	VoxelsOnCartesianGrid<float>& image = dynamic_cast<VoxelsOnCartesianGrid<float>&>(*density_ptr);	
	Sinogram<float> sino = proj_data_ptr->get_empty_sinogram(0,0); 
	Viewgram<float> view = proj_data_ptr->get_empty_viewgram(0,0); 
	Viewgram<float> view1 = proj_data_ptr->get_empty_viewgram(0,0); 
	Viewgram<float> view_th = proj_data_ptr->get_empty_viewgram(0,0); 
	Viewgram<float> view1_th = proj_data_ptr->get_empty_viewgram(0,0); 
	/*cerr << "ax_min = " << proj_data_ptr->get_min_axial_pos_num(0) << 
	 *	", ax_max = " << proj_data_ptr->get_max_axial_pos_num(0) << 
	 *	", img_min = " << image.get_min_y() << 
	 *	", img_max = " << image.get_max_y() << 
	 *	", img_siz = " << image.get_y_size() << 
	 *	 endl; */	

    // Retrieve runtime-dependent sizes
   const int sp = proj_data_ptr->get_num_tangential_poss();
   const int sth = proj_data_ptr->get_num_views();
   const int sa = proj_data_ptr->get_num_axial_poss(0);
   const int sx = image.get_x_size();
   const int sy = image.get_y_size();
   const int sx2 = ceil(sx / 2.0), sy2 = ceil(sy/2.0);
   const int sth2 = ceil(sth / 2.0);
   const float c_int = -1/(M_PI*sth*(sp-1)); 
	
	//The rest of the variables used by the program.
	int ia, image_pos;
	int ith, jth, ip, ix1, ix2, k1, k2; 
	
	float x, aux; 
	
	const int image_min_x = image.get_min_x();
	const int image_min_y = image.get_min_y();

    // Dynamically allocated vectors
  //Old static array declarations: float th[sth], p[sp], p_ud[sp], x1[sx],  x2[sy];//hx[sth]
  // New dynamic declarations using std::vector
  std::vector<float> th(sth), p(sp), p_ud(sp), x1(sx), x2(sy);

  //Old static array declarations: float f[sth][sp], ddf[sth][sp];
  // New dynamic declaration using std::vector
  std::vector<std::vector<float>> f(sth, std::vector<float>(sp,0.0f));
  std::vector<std::vector<float>> ddf(sth, std::vector<float>(sp,0.0f));

  // Old declarations: float f_ud[sth][sp], ddf_ud[sth][sp]; 
  // New declarations using std::vector of std::vector
  std::vector<std::vector<float>> f_ud(sth, std::vector<float>(sp,0.0f));
  std::vector<std::vector<float>> ddf_ud(sth, std::vector<float>(sp,0.0f));

	//float f_ud[sth][sp], ddf_ud[sth][sp];
	//float f1[sth][sp], ddf1[sth][sp];
	//float f1_ud[sth][sp], ddf1_ud[sth][sp];
	
	//float f_th[sth][sp], ddf_th[sth][sp];
	//float f_th_ud[sth][sp], ddf_th_ud[sth][sp];
	//float f1_th[sth][sp], ddf1_th[sth][sp];
	//float f1_th_ud[sth][sp], ddf1_th_ud[sth][sp];
	
	// Convert these arrays to std::vector<std::vector<float>>
	std::vector<std::vector<float>> f1(sth, std::vector<float>(sp,0.0f));
	std::vector<std::vector<float>> ddf1(sth, std::vector<float>(sp,0.0f));

	std::vector<std::vector<float>> f1_ud(sth, std::vector<float>(sp,0.0f));
	std::vector<std::vector<float>> ddf1_ud(sth, std::vector<float>(sp,0.0f));


	std::vector<std::vector<float>> f_th(sth, std::vector<float>(sp,0.0f));
	std::vector<std::vector<float>> ddf_th(sth, std::vector<float>(sp,0.0f));

	std::vector<std::vector<float>> f_th_ud(sth, std::vector<float>(sp,0.0f));
	std::vector<std::vector<float>> ddf_th_ud(sth, std::vector<float>(sp,0.0f));

	std::vector<std::vector<float>> f1_th(sth, std::vector<float>(sp,0.0f));
	std::vector<std::vector<float>> ddf1_th(sth, std::vector<float>(sp,0.0f));

	std::vector<std::vector<float>> f1_th_ud(sth, std::vector<float>(sp,0.0f));
	std::vector<std::vector<float>> ddf1_th_ud(sth, std::vector<float>(sp,0.0f));


  //	float lg[sp], termC[sth], termC_th[sth]; 
  std::vector<float> lg(sp), termC(sth), termC_th(sth);
	const float dp6 = 6.0/4.0*2.0/(sp-1.0);
	
	//Some constants.
	//pp2= -1.0/(4*M_PI*M_PI); 
	
	
	#ifdef STIR_OPENMP
	if (getenv("OMP_NUM_THREADS")==NULL) {
		omp_set_num_threads(omp_get_num_procs());
		if (omp_get_num_procs()==1) 
			warning("Using OpenMP with #processors=1 produces parallel overhead. You should compile without using USE_OPENMP=TRUE.");
		cerr<<"Using OpenMP-version of SRT2D with thread-count = processor-count (="<<omp_get_num_procs()<<")."<<endl;
	} else {
		cerr<<"Using OpenMP-version of SRT2D with "<<getenv("OMP_NUM_THREADS")<<" threads on "<<omp_get_num_procs()<<" processors."<<endl;
		if (atoi(getenv("OMP_NUM_THREADS"))==1) 
			warning("Using OpenMP with OMP_NUM_THREADS=1 produces parallel overhead. Use more threads or compile without using USE_OPENMP=TRUE.");
	}
	//cerr<<"Define number of threads by setting OMP_NUM_THREADS environment variable, i.e. \"export OMP_NUM_THREADS=<num_threads>\""<<endl;
	//shared_ptr<DiscretisedDensity<3,float> > empty_density_ptr(density_ptr->clone());
	#endif
	
	
	
	// Put theta and p in arrays.
	for(ith=0; ith<sth; ith++) 
		th[ith]=ith*M_PI/sth; 
	
	for(ip=0; ip<sp; ip++) 
		p[ip]=-1.0+2.0*ip/(sp-1);
	for(ip=0; ip<sp; ip++) 
		p_ud[sp-ip-1]=p[ip];
	
	// Put x1 and x2 in arrays.
	
	//rr = 1.0*sx/((sp+1)*zoom); 
	cerr << "sp = " << sp << endl; 
	//-- Creation of the grid
	for(k1=0; k1<sx; k1++)
		//x1[k1]=-1.0*sx/(sp+1)+2.0*sx/(sp+1)*k1/(sx-1); 
		x1[k1] = -1.0*sx/((sp+1)*zoom) + k1*2.0*sx/((sp+1)*zoom)/(sx-1);
	//x1[k1] = -1.0*sx/((sp+1)*zoom) + k1*2.0*((sp+1)*zoom)
	////x1[k1]=-1.0+2.0*k1/(sx-1); 
	for(k2=0; k2<sx; k2++) 
		//x2[k2]=-1.0*sx/(sp+1)+2.0*sx/(sp+1)*k2/(sx-1); 
		x2[k2] = -1.0*sx/((sp+1)*zoom) + k2*2.0*sx/((sp+1)*zoom)/(sx-1);
	////x2[k2]=-1.0+2.0*k2/(sx-1); 
	
	
	/*	for(ix1=0; ix1<sx; ix1++) 
	 *		x1[ix1]=-1.0+2.0*ix1/(sx-1); 
	 *	
	 *	for(ix2=0; ix2<sy; ix2++) 
	 *		x2[ix2]=-1.0+2.0*ix2/(sy-1);*/
	
	// Calculate constants 
	//dp = p[1]-p[0];
	//dp6 = 6.0/4.0*dp;
	
/*	for(ia=0; ia<sa; ia++) { 
		for(ix1=0; ix1<sx; ix1++){ 
			for(ix2=0; ix2<sy; ix2++){  
				image_pos= image.get_min_z() + 2*(ia - proj_data_ptr->get_min_axial_pos_num(0));
				
				image[image_pos][image_min_x +sx-ix1-1][image_min_y +ix2] = 0; 
			}
		}
	}*/
	
	// Starting calculations per view
	// 2D algorithm only  
	
	// -----
	// special case of ith=0 
	// -----
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
		/*	for (ip=0; ip<sp; ip++) {
				lg[ip] = log(fabs(x-p[ip])); 
if(fabs(p[ip]-x)<2e-6) lg[ip] = 0.; 			
}*/
for (ip=0; ip<sp; ip++) {
    double val = fabs(x - p[ip]);
    lg[ip] = val < 2e-6 ? 0. : std::log(val);  // Using std::log to specify the namespace
}

			for(ia=0; ia<sa; ia++){
				image[ia][image_min_x +sx-ix1-1][image_min_y +ix2] 
				= -hilbert_der(x, f[ia], ddf[ia], p, sp, lg, termC[ia])/(M_PI*sth*(sp-1)); //2*
			}
		}
	}
	// -----
	// general case, ith=1...sth-1
	// -----
	#ifdef STIR_OPENMP
	#pragma omp parallel \
	shared(view, view1, view_th, view1_th,do_arc_correction, arc_correction, p,th,x1,x2,image) \
	private(jth,ia, ip, f, f_ud,f1, f1_ud, ddf,f_th, f_th_ud, f1_th, f1_th_ud, ddf1, ddf_th, ddf1_th, \
	ddf_ud, ddf1_ud, ddf_th_ud, ddf1_th_ud, termC, termC_th, ix2, ix1, aux, x, lg, image_pos)
//	#pragma omp for schedule(auto)  nowait
	#pragma omp for schedule(dynamic)  nowait
	#endif
	for(ith=1; ith<sth; ith++){
		//image_pos= image.get_min_z() + 2*(ia - proj_data_ptr->get_min_axial_pos_num(0));
		//std::cerr << "\nView " << ith << " of " << sth << std::endl;
		
		if(ith<sth2 ){ 
			jth = sth2-ith; 
		} else if(ith>sth2) { 
			jth = (int)ceil(3*sth/2.0)-ith; // MARK integer division
		} else {
			jth = sth2;
		}
		
		// Loading related viewgrams 
		#ifdef STIR_OPENMP
		#pragma omp critical
		#endif
		{
			view = proj_data_ptr->get_viewgram(ith, 0);
			view1 = proj_data_ptr->get_viewgram(sth-ith, 0);
			view_th = proj_data_ptr->get_viewgram(jth, 0);
			view1_th = proj_data_ptr->get_viewgram(sth-jth, 0);
			if (do_arc_correction) { 
				view = arc_correction.do_arc_correction(view);
				view1 = arc_correction.do_arc_correction(view1);
				view_th = arc_correction.do_arc_correction(view_th);
				view1_th = arc_correction.do_arc_correction(view1_th);
			}
			
			for(ia=0; ia<sa; ia++){
				for(ip=0; ip<sp; ip++) {
					f[ia][ip] = view[view.get_min_axial_pos_num() + ia][view.get_min_tangential_pos_num() + ip];
					f_ud[ia][sp-ip-1] = f[ia][ip]; 
					f1[ia][ip] = view1[view.get_min_axial_pos_num() + ia][view.get_min_tangential_pos_num() + ip];
					f1_ud[ia][sp-ip-1] = f1[ia][ip]; 
					
					f_th[ia][ip] = view_th[view.get_min_axial_pos_num() + ia][view.get_min_tangential_pos_num() + ip];
					f_th_ud[ia][sp-ip-1] = f_th[ia][ip]; 
					f1_th[ia][ip] = view1_th[view.get_min_axial_pos_num() + ia][view.get_min_tangential_pos_num() + ip];
					f1_th_ud[ia][sp-ip-1] = f1_th[ia][ip]; 
				}
			}
		}
		// Calculation of second derivative by use of function spline
		for(ia=0; ia<sa; ia++){ 
			spline(p,f[ia],sp,ddf[ia]);
			spline(p,f1[ia],sp,ddf1[ia]);
			spline(p,f_th[ia],sp,ddf_th[ia]);
			spline(p,f1_th[ia],sp,ddf1_th[ia]);
			for(ip=0; ip<sp; ip++) { 
				ddf_ud[ia][sp-ip-1] = ddf[ia][ip]; 
				ddf1_ud[ia][sp-ip-1] = ddf1[ia][ip]; 
				ddf_th_ud[ia][sp-ip-1] = ddf_th[ia][ip]; 
				ddf1_th_ud[ia][sp-ip-1] = ddf1_th[ia][ip]; 
			}
		}
		
		
		for(ia=0; ia<sa; ia++) { 
			termC[ia] = (ddf[ia][0]*(3*p[1]-p[0]) + ddf[ia][sp-1]*(p[sp-1]-3.0*p[sp-2]))/4.0;
			for (ip=0; ip<sp; ip++) {
				termC[ia] += dp6*ddf[ia][ip];
			}
			termC_th[ia] = (ddf_th[ia][0]*(3*p[1]-p[0]) + ddf_th[ia][sp-1]*(p[sp-1]-3.0*p[sp-2]))/4.0;
			for (ip=0; ip<sp; ip++) {
				termC_th[ia] += dp6*ddf_th[ia][ip];
			}
		}
		
		
		//Starting the calculation of ff(x1,x2).
		for(ix1=0; ix1<sx2; ix1++){
			for(ix2=0; ix2<=ix1; ix2++){  
				// If x1,x2 off range put ff(x1,x2)=0
				aux=sqrt(1.0-x2[ix2]*x2[ix2]);
				if(fabs(x2[ix2]) >= 1.0 || fabs(x1[ix1]) >= aux){ 
					continue;
				}
				
				// Computation of h_rho			
				x=x2[ix2]*cos(th[ith])-x1[ix1]*sin(th[ith]); 
				/*	for (ip=0; ip<sp; ip++) {
				lg[ip] = log(fabs(x-p[ip])); 
				if(fabs(p[ip]-x)<2e-6) lg[ip] = 0.; 	 		
				}*/
				for (ip=0; ip<sp; ip++) {
				double val = fabs(x - p[ip]);
				lg[ip] = val < 2e-6 ? 0. : std::log(val);  // Using std::log to specify the namespace
				}
				
				for(ia=0; ia<sa; ia++){
					image_pos= ia;//2*ia;
					
					image[image_pos][image_min_x +sx-ix1-1][image_min_y +ix2] 
					+= hilbert_der(x, f[ia], ddf[ia], p, sp, lg, termC[ia])*c_int; // bot-left
					if(ix2<sy2-1){ 
						image[image_pos][image_min_x +sx-ix1-1][image_min_y +sy-ix2-1] 
						+= hilbert_der(x, f1[ia], ddf1[ia], p, sp, lg, termC[ia])*c_int; // bot-right
					}
					if(ix1<sx2-1){ 
						image[image_pos][image_min_x +ix1][image_min_y +ix2] 
						-= hilbert_der(-x, f1_ud[ia], ddf1_ud[ia], p_ud, sp, lg, termC[ia])*c_int;// top-left
					} 
					if((ix1<sx2-1)&&(ix2<sy2-1)){ 
						image[image_pos][image_min_x +ix1][image_min_y +sy-ix2-1] 
						-= hilbert_der(-x, f_ud[ia], ddf_ud[ia], p_ud, sp, lg, termC[ia])*c_int; // top-right
					}
					
					if(ith<=sth2 && ix1!=ix2){ 
						image[image_pos][image_min_x +sx-ix2-1][image_min_y +ix1] 
						-= hilbert_der(-x, f_th_ud[ia], ddf_th_ud[ia], p_ud, sp, lg, termC_th[ia])*c_int; // bot-left
						if(ix2<sy2-1){ 
							image[image_pos][image_min_x +ix2][image_min_y +ix1]
							+= hilbert_der(x, f1_th[ia], ddf1_th[ia], p, sp, lg, termC_th[ia])*c_int; // bot-right
						}
						if(ix1<sx2-1){ 
							image[image_pos][image_min_x +sx-ix2-1][image_min_y +sx-ix1-1]
							-= hilbert_der(-x, f1_th_ud[ia], ddf1_th_ud[ia], p_ud, sp, lg, termC_th[ia])*c_int;// top-left
						} 
						if((ix1<sx2-1)&&(ix2<sy2-1)){ 
							image[image_pos][image_min_x +ix2][image_min_y +sx-ix1-1]
							+= hilbert_der(x, f_th[ia], ddf_th[ia], p, sp, lg, termC_th[ia])*c_int; // top-right
						}
					} else if(ith>sth2 && ix1!=ix2) { 
						image[image_pos][image_min_x +sx-ix2-1][image_min_y +ix1] 
						+= hilbert_der(x, f_th[ia], ddf_th[ia], p, sp, lg, termC_th[ia])*c_int; // bot-left
						if(ix2<sy2-1){ 
							image[image_pos][image_min_x +ix2][image_min_y +ix1]
							-= hilbert_der(-x, f1_th_ud[ia], ddf1_th_ud[ia], p_ud, sp, lg, termC_th[ia])*c_int; // bot-right
						}
						if(ix1<sx2-1){ 
							image[image_pos][image_min_x +sx-ix2-1][image_min_y +sx-ix1-1]
							+= hilbert_der(x, f1_th[ia], ddf1_th[ia], p, sp, lg, termC_th[ia])*c_int;// top-left
						} 
						if((ix1<sx2-1)&&(ix2<sy2-1)){ 
							image[image_pos][image_min_x +ix2][image_min_y +sx-ix1-1]
							-= hilbert_der(-x, f_th_ud[ia], ddf_th_ud[ia], p_ud, sp, lg, termC_th[ia])*c_int; // top-right
						}
					} 
					
					
				}
				
			}  
		} 
	}
	
	// apply Wiener filter
	if(filter_wiener!=0) 
		wiener(image, sx, sy, sa); 
	// apply median filter 
	if(filter_median!=0)
		median(image, sx, sy, sa); 
	// adjust gamma 
	if(filter_gamma!=0)
		gamma(image, sx, sy, sa); 
	
	
	return Succeeded::yes;
}


void SRT2DReconstruction::wiener(VoxelsOnCartesianGrid<float>& image, int sx, int sy, int sa) { 
	
	const int min_x = image.get_min_x();
	const int min_y = image.get_min_y();
	const int ws = 9; 
	    
	for(int ia=0; ia<sa; ia++) { 
		//float localMean[sx][sy], localVar[sx][sy], 
    std::vector<std::vector<float>> localMean(sx, std::vector<float>(sy, 0.0f));
    std::vector<std::vector<float>> localVar(sx, std::vector<float>(sy, 0.0f));
  
   float noise=0.; 
		
		for(int i=0+1; i<sx-1; i++){ 
			for(int j=0+1; j<sy-1; j++) { 
				localMean[i][j] = 0; localVar[i][j] = 0; 
				
				for(int k=-1; k<=1; k++) 
					for(int l=-1; l<=1; l++) 
						localMean[i][j] += image[ia][min_x+i+k][min_y+j+l]*1.; 
				localMean[i][j] /= ws; 
				
				for(int k=-1; k<=1; k++) 
					for(int l=-1; l<=1; l++) 
				//		localVar[i][j] += std::pow(image[ia][min_x+i+k][min_y+j+l], 2)*1.; 
				//localVar[i][j] = localVar[i][j]/ws - std::pow(localMean[i][j], 2); 
        //Corrected version:
				//localVar[i][j] += std::pow(static_cast<double>(image[ia][min_x+i+k][min_y+j+l]), 2.0); 
				//localVar[i][j] = localVar[i][j]/ws - std::pow(static_cast<double>(localMean[i][j]), 2.0);

				localVar[i][j] += image[ia][min_x+i+k][min_y+j+l] * image[ia][min_x+i+k] [min_y+j+l]; 
			  localVar[i][j] = localVar[i][j]/ws - localMean[i][j] * localMean[i][j];

				noise += localVar[i][j]; 
			}
		}
		noise /= sx*sy; 
		
		for(int i=0+1; i<sx-1; i++)  
			for(int j=0+1; j<sy-1; j++) 
				image[ia][min_x+i][min_y+j] = (image[ia][min_x+i][min_y+j] - localMean[i][j])/std::max(localVar[i][j], noise)* \
								std::max(localVar[i][j] - noise, 0.f) + localMean[i][j]; 
	}
	return; 
}

void SRT2DReconstruction::median(VoxelsOnCartesianGrid<float>& image, int sx, int sy, int sa) { 
	
	const int min_x = image.get_min_x();
	const int min_y = image.get_min_y();
	const int filter_size = 3;
	const int offset = filter_size/2;
	const int len = 4; 
	//double neighbors[9]; 
	std::vector<double> neighbors(filter_size * filter_size, 0);

	for(int ia=0; ia<sa; ia++) { 
		for(int i=0; i<9; i++) 
			neighbors[i] = 0; 
		
		for(int i=0; i<sx; i++) { 
			for(int j=0; j<sy; j++) { 
				if(i==0 || i==sx-1 || j==0 || j==sy-1) 
					continue; 
				for(int k=-offset; k<=offset; k++) { 
					for(int l=-offset; l<=offset; l++) { 
					//	neighbors[(k+offset)*filter_size + l+offset] = image[ia][min_x + i + (k+i<sx?k:0)][min_y + j + (j+l<sy?l:0)];
//          neighbors[(k + offset) * filter_size + (l + offset)] = image[ia][min_x + i + std::clamp(k + i, 0, sx - 1)][min_y + j + std::clamp(l + j, 0, sy - 1)];
neighbors[(k+offset)*filter_size + l+offset] = image[ia][min_x + i + (k+i<sx?k:0)][min_y + j + (j+l<sy?l:0)];			
		}
				}
				//std::sort(std::begin(neighbors), std::end(neighbors));
        std::sort(neighbors.begin(), neighbors.end());
				image[ia][min_x+i][min_y+j] = neighbors[len];
			}
		}
	}
	return;  

}

void SRT2DReconstruction::gamma(VoxelsOnCartesianGrid<float>& image, int sx, int sy, int sa) { 
	
	const int min_x = image.get_min_x();
	const int min_y = image.get_min_y();
	float targetAverage = .25; // Desired average pixel value
	
	for(int ia=0; ia<sa; ia++) { 
		
		// normalize image 
		float min_val = INFINITY, max_val = -INFINITY; 
		for(int i=0; i<sx; i++) { 
			for(int j=0; j<sy; j++) { 
				min_val = std::min(image[ia][min_x+i][min_y+j], min_val); 
				max_val = std::max(image[ia][min_x+i][min_y+j], max_val); 
			}
		}
		for(int i=0; i<sx; i++) 
			for(int j=0; j<sy; j++) 
				image[ia][min_x+i][min_y+j] = (image[ia][min_x+i][min_y+j]-min_val)/(max_val-min_val); 
		
		// averagePixelValue = mean(img(abs(img)>.1));
		int count = 0; 
		float averagePixelValue = 0.; 
		for(int i=0; i<sx; i++) { 
			for(int j=0; j<sy; j++) { 
				if(std::abs(image[ia][min_x+i][min_y+j])>0.1) { 
					count++; 
					averagePixelValue+=image[ia][min_x+i][min_y+j]; 
				}
			}
		}
		averagePixelValue /= count; 
		
		float gamma_val = 1.; 
		if(averagePixelValue>0.)
			gamma_val = std::log(targetAverage) / std::log(averagePixelValue);
		//img = img.^gamma; 
		for(int i=0; i<sx; i++) 
			for(int j=0; j<sy; j++) 
				image[ia][min_x+i][min_y+j] = std::abs(image[ia][min_x+i][min_y+j])>1e-6 ? std::pow(image[ia][min_x+i][min_y+j],gamma_val) : image[ia][min_x+i][min_y+j]; 
		
		// denormalize image
		for(int i=0; i<sx; i++) 
			for(int j=0; j<sy; j++) 
				image[ia][min_x+i][min_y+j] = image[ia][min_x+i][min_y+j]*(max_val-min_val)+min_val; 
	} 
	return; 
}

//float SRT2DReconstruction::hilbert_der(float x, float f[], float ddf[], float p[], int sp, float lg[], float termC) {
float SRT2DReconstruction::hilbert_der(float x, const std::vector<float>& f, const std::vector<float>& ddf, const std::vector<float>& p, int sp, const std::vector<float>& lg, float termC) {
	
	float term, trm0, termd0; 
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
		float trm1 = d_div_6 + minus_half_div_d*(p[ip+2]-x)*(p[ip+2]-x);
		float termd =  (f[ip+2]-f[ip+1])/d  + ddf[ip+1]*trm1 - ddf[ip+2]*trm0; 
		term += (termd0-termd) * lg[ip+1]; 
		termd0 = termd;
		trm0 = trm1; 
	}
	
	return term; 
}


//void SRT2DReconstruction::spline(float x[],float y[],int n, float y2[]) {
void SRT2DReconstruction::spline(const std::vector<float>& x, const std::vector<float>& y, int n, std::vector<float>& y2) {
	// function for nanural qubic spline.
	int i, k;
	float qn, un;
//	float u[n]; 
    std::vector<float> u(n);
	y2[0]=0.0; 
	u[0]=0.0;
	for(i=1; i<n-1; i++) {
		float sig=(x[i]-x[i-1])/(x[i+1]-x[i-1]);
		float p=sig*y2[i-1]+2.0;
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



//float SRT2DReconstruction::integ(float dist, int max, float ff[]) {
float SRT2DReconstruction::integ(float dist, int max, const std::vector<float>& ff) {
	// function for the calculation of integrals (closed formula).
	int k, intg;
	intg=ff[0];
	for(k=1; k<max; k++) {
		intg+=ff[k];
	}
	return intg*dist/max;
}



END_NAMESPACE_STIR
