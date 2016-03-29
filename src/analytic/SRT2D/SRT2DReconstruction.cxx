/*
/author Dimitra Kyriakopoulou
*/

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
 
#ifdef STIR_OPENMP
#include <omp.h>
#endif

using std::cerr; 
using std::endl;
 
START_NAMESPACE_STIR


void 
SRT2DReconstruction::
set_defaults()
{
  base_type::set_defaults();
  attenuation_filename=""; 
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
  parser.add_key("attenuation filename", &attenuation_filename); 
}

void 
SRT2DReconstruction::
ask_parameters()
{ 
  base_type::ask_parameters();
  num_segments_to_combine = ask_num("num_segments_to_combine (must be odd)",-1,101,-1);
  thres_restr_bound=ask_num("threshold for restriction within boundary",-pow(10,6),pow(10,6),-pow(10,6));
  attenuation_filename=ask_string("attenuation filename"); 
}

bool SRT2DReconstruction::post_processing()
{
   if (base_type::post_processing())
    return true;
   
   if (attenuation_filename.length() == 0) { 
		warning("You need to specify an attenuation file\n"); return true; 
	}
	atten_data_ptr= ProjData::read_from_file(attenuation_filename);
	
	
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
  if (num_segments_to_combine>1)
    {  
      const ProjDataInfoCylindrical& proj_data_info_cyl =
	dynamic_cast<const ProjDataInfoCylindrical&>
	(*proj_data_ptr->get_proj_data_info_ptr());

      //  full_log << "SSRB combining " << num_segments_to_combine 
      //           << " segments in input file to a new segment 0\n" << endl; 

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
    const float tan_theta = proj_data_ptr->get_proj_data_info_ptr()->get_tantheta(Bin(0,0,0,0));
    if(fabs(tan_theta ) > 1.E-4)
      {
	warning("SRT2D: segment 0 has non-zero tan(theta) %g", tan_theta);
	return Succeeded::no;
      }
  }

  //float tangential_sampling;
  // TODO make next type shared_ptr<ProjDataInfoCylindricalArcCorr> once we moved to boost::shared_ptr
  // will enable us to get rid of a few of the ugly lines related to tangential_sampling below
  shared_ptr<ProjDataInfo> arc_corrected_proj_data_info_sptr;

  // arc-correction if necessary
  ArcCorrection arc_correction;
  bool do_arc_correction = false;
  if (dynamic_cast<const ProjDataInfoCylindricalArcCorr*>
      (proj_data_ptr->get_proj_data_info_ptr()) != 0)
    {
      // it's already arc-corrected
      arc_corrected_proj_data_info_sptr =
	proj_data_ptr->get_proj_data_info_ptr()->create_shared_clone();
    //  tangential_sampling =
	//dynamic_cast<const ProjDataInfoCylindricalArcCorr&>
	//(*proj_data_ptr->get_proj_data_info_ptr()).get_tangential_sampling();  
    }
  else
    {
      // TODO arc-correct to voxel_size
      if (arc_correction.set_up(proj_data_ptr->get_proj_data_info_ptr()->create_shared_clone()) ==
	  Succeeded::no)
	return Succeeded::no;
      do_arc_correction = true;
      // TODO full_log
      warning("FBP2D will arc-correct data first");
      arc_corrected_proj_data_info_sptr =
	arc_correction.get_arc_corrected_proj_data_info_sptr();
	// tangential_sampling =
	//	arc_correction.get_arc_corrected_proj_data_info().get_tangential_sampling();  
    }

	VoxelsOnCartesianGrid<float>& image = dynamic_cast<VoxelsOnCartesianGrid<float>&>(*density_ptr);	
	Sinogram<float> sino = proj_data_ptr->get_empty_sinogram(0,0); 
	Viewgram<float> view = proj_data_ptr->get_empty_viewgram(0,0); 
	Viewgram<float> view_atten = atten_data_ptr->get_empty_viewgram(0,0); 

	const int sp = proj_data_ptr->get_num_tangential_poss(); 
	const int sth = proj_data_ptr->get_num_views();
	const int sa = proj_data_ptr->get_num_axial_poss(0);
	
	
	const int sx = image.get_x_size();
	const int sy = image.get_y_size();
	/*int sz = image.get_z_size(); */

//c ----------------------------------------------
//c The rest of the variables used by the program.
//c ----------------------------------------------
	int i,j,k1,k2;

	float aux,a,b,f_node; 

	const int image_min_x = image.get_min_x();
	const int image_min_y = image.get_min_y();
	
	float th[sth], p[sp], x1[sx], x2[sy]; 
	float g[sth][sp], ddg[sth][sp];

	const int Nt = 8, Nmul = sth/Nt; 
	float lg[sp], hilb[sth][sp], fcpe[sth][sp], fspe[sth][sp], fc[sth][sp], fs[sth][sp], ddfc[sth][sp], ddfs[sth][sp], dh1[Nt], dh2[Nt], f[sth][sp], ddf[sth][sp], t[Nt]; 
	float rho, h, fcme_fin, fsme_fin, fc_fin, fs_fin, fcpe_fin, fspe_fin, hc_fin, hs_fin, I, Ft1, Ft2, rho1, rho2, tau, tau1, tau2, rx1, rx2; 
	float gx, w, F; 
	
	float rx1x2th[sa][sx][sy], lg1_cache[Nt/2][sp-1], lg2_cache[Nt/2][sp-1]; 

	float f_cache[sa][Nt/2][sp], ddf_cache[sa][Nt/2][sp]; 
	float f1_cache[sa][Nt/2][sp], ddf1_cache[sa][Nt/2][sp]; 

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
	
//c --------------------------
//c Put theta and p in arrays.
//c --------------------------
	for(i=0; i<sth; i++) 
		th[i]=i*2*M_PI/sth; 
	for(int it=0; it<Nt; it++) 
		t[it]=it*2*M_PI/Nt; 
	for(j=0; j<sp; j++) 
		p[j]=-1.0+2.0*j/(sp-1);
	
//c ------------------------
//c Put x1 and x2 in arrays.
//c ------------------------
	for(k1=0; k1<sx; k1++) 
		x1[k1]=-1.0+2.0*k1/(sx-1); 
	for(k2=0; k2<sx; k2++) 
		x2[k2]=-1.0+2.0*k2/(sx-1);
	
	
	for(int it=0; it<Nt/2; it++) { 
		view_atten = atten_data_ptr->get_viewgram(Nmul*it, 0);      
		if (do_arc_correction) {
		  view_atten = arc_correction.do_arc_correction(view_atten);
		}
		//float mv = 1e-4; 
		for(int ia=0; ia<sa; ia++) {
			for(int ip=0; ip<sp; ip++){ 	 
				f_cache[ia][it][ip] = view_atten[view_atten.get_min_axial_pos_num() + ia][view_atten.get_min_tangential_pos_num() + ip];
				//mv = f_cache[ia][it][ip] > mv ? f_cache[ia][it][ip] : mv; 
			}
		}
		//for(int ia=0; ia<sa; ia++) {
		//	for(int ip=0; ip<sp; ip++){ 	
		//		f_cache[ia][it][ip] *= 0.093/mv; 
		//	}
		//}
		for(int ia=0; ia<sa; ia++){
			spline(p,f_cache[ia][it],sp,ddf_cache[ia][it]);
		}
		
		for(int ia=0; ia<sa; ia++) {
			for(int ip=0; ip<sp; ip++)  {
				f1_cache[ia][it][sp-ip-1] = f_cache[ia][it][ip]; 
			}
		}
		for(int ia=0; ia<sa; ia++) {
			for(int ip=0; ip<sp; ip++)  {
				ddf1_cache[ia][it][sp-ip-1] = ddf_cache[ia][it][ip]; 
			}
		}
	}
	
	
	//-- Starting calculations per view
	// 2D algorithm only
	#ifdef STIR_OPENMP
	#pragma omp parallel \
	shared(view,do_arc_correction,arc_correction,p,th,x1,x2,f_cache,ddf_cache,image) \
	private(g,f,ddg,ddf,hilb,fcpe,fspe,fc,fs,ddfc,ddfs,aux,rho,lg,\
	tau,a,b,tau1,tau2,w,rho1,rho2,lg1_cache,lg2_cache,f_node,h,fcme_fin,\
	fsme_fin,fcpe_fin,fspe_fin,gx,fc_fin,fs_fin,hc_fin,hs_fin,rx1x2th,\
	dh1,dh2,Ft1,Ft2,F,I,rx1,rx2)
	#pragma omp for schedule(auto)  nowait
	#endif
	for(int ith=0; ith<sth; ith++){

		//image_pos= 2*ia;
		std::cerr << "\nView " << ith << " of " <<  sth << endl; 
		//if(ia!=55) continue; 

		//-- Loading the viewgram  
		#ifdef STIR_OPENMP
		#pragma omp critical
		#endif
		{
			view = proj_data_ptr->get_viewgram(ith, 0);   
			view_atten = atten_data_ptr->get_viewgram(ith, 0);      
			if (do_arc_correction) {
				view = arc_correction.do_arc_correction(view);
				view_atten = arc_correction.do_arc_correction(view_atten); 
			}
			//float mv = 1e-4; 
			for(int ia=0; ia<sa; ia++) {
				for(int ip=0; ip<sp; ip++){ 	 
					g[ia][ip] = view[view.get_min_axial_pos_num() + ia][view.get_min_tangential_pos_num() + ip];
					f[ia][ip] = view_atten[view_atten.get_min_axial_pos_num() + ia][view_atten.get_min_tangential_pos_num() + ip];
					
					//f[ia][ip] = g[ia][ip]; 
					//mv = f[ia][ip] > mv ? f[ia][ip] : mv; 
				}
			}
			//for(int ia=0; ia<sa; ia++) {
			//	for(int ip=0; ip<sp; ip++){ 	
			//		f[ia][ip] = 0.093*f[ia][ip]/mv; 
			//	}
			//}
		}
		//-- Calculation of second derivative by use of function spline
		for(int ia=0; ia<sa; ia++){
			spline(p,g[ia],sp,ddg[ia]);
			spline(p,f[ia],sp,ddf[ia]);
		}

		//---- calculate h(rho,theta) for all rho, theta 
		for(int ia=0; ia<sa; ia++) { 
			for(int ip=0; ip<sp; ip++) { 
				hilb[ia][ip] = hilbert_node(p[ip], f[ia], ddf[ia], p, sp, f[ia][ip]); 

				fcpe[ia][ip] = exp(0.5*f[ia][ip])*cos(hilb[ia][ip]/(2*M_PI));
				fspe[ia][ip] = exp(0.5*f[ia][ip])*sin(hilb[ia][ip]/(2*M_PI));

				fc[ia][ip] = fcpe[ia][ip]*g[ia][ip]; //USE CORRECT EMISSION
				fs[ia][ip] = fspe[ia][ip]*g[ia][ip]; //USE CORRECT EMISSION
				
			}
			//-- calculate ddfc, ddfs for all \rho, \theta
			spline(p, fc[ia], sp, ddfc[ia]);
			spline(p, fs[ia], sp, ddfs[ia]);
		}

		//---- calculate r(x1, x2, theta) 
		//std::cerr << "\ncalculating r(x1,x2,th) ...\n"; 
		////test
		for(int ix1=0; ix1<sx; ix1++) { 
			//std::cerr << ix1 << ", "; 
			for(int ix2=0; ix2<sy; ix2++) { 
				aux=sqrt(1.0-x2[ix2]*x2[ix2]);
				if(fabs(x2[ix2]) >= 1.0 || fabs(x1[ix1]) >= aux) continue;
				
				rho=x2[ix2]*cos(th[ith])-x1[ix1]*sin(th[ith]);
				
				int i = floor((rho+1)*(sp-1)/2);
				float p1 = p[i]; 
				float p2 = p[i+1];
				float A=(p2-rho)/(p2-p1); 
				float B=1-A; 
				float C=1.0/6*(A*A*A-A)*(p2-p1)*(p2-p1);
				float D=1.0/6*(B*B*B-B)*(p2-p1)*(p2-p1);
				
				for(int ip=0; ip<sp; ip++) { 
					lg[ip] = log(fabs(p[ip]-rho)); 
				}
				
				// calculate I
				tau=x2[ix2]*sin(th[ith])+x1[ix1]*cos(th[ith]);
				if (tau>0) {
					a = tau;
					b = sqrt(1-rho*rho);
				} else {
					a = -sqrt(1-rho*rho);
					b = tau;
				}
				
				tau1 = a + (b-a)*(1.0/2-sqrt(3.0)/6);
				tau2 = a + (b-a)*(1.0/2+sqrt(3.0)/6);
				w = 1.0/2*(b-a);
				
				for(int it=0; it<Nt/2; it++){ 
					rho1 = tau1*sin(th[ith]-t[it]) + rho*cos(th[ith]-t[it]);
					rho2 = tau2*sin(th[ith]-t[it]) + rho*cos(th[ith]-t[it]);
					
					for (int ip=0; ip<sp-1;ip++) {
						lg1_cache[it][ip] = log( fabs( (p[ip+1]-rho1)/(p[ip]-rho1) ) ); 
						lg2_cache[it][ip] = log( fabs( (p[ip+1]-rho2)/(p[ip]-rho2) ) ); 
					}
				}
				
				for(int ia=0; ia<sa; ia++) { 

					f_node = A*f[ia][i]+B*f[ia][i+1]+C*ddf[ia][i]+D*ddf[ia][i+1];
					
					// calculate fcme, fsme, fc, fs, hc, hs
					
					h = hilbert(rho, f[ia], ddf[ia], p, sp, lg); 
					fcme_fin = exp(-0.5*f_node)*cos(h/(2*M_PI));
					fsme_fin = exp(-0.5*f_node)*sin(h/(2*M_PI));

					fcpe_fin = exp(0.5*f_node)*cos(h/(2*M_PI));
					fspe_fin = exp(0.5*f_node)*sin(h/(2*M_PI));

					gx = splint(p, g[ia], ddg[ia], sp, rho);
					
					fc_fin = fcpe_fin*gx;
					fs_fin = fspe_fin*gx;
					//fc_fin = fcpe_fin*fx1x2th[ix1][ix2][ith]; //USE CORRECT EMISSION
					//fs_fin = fspe_fin*fx1x2th[ix1][ix2][ith]; //USE CORRECT EMISSION

					hc_fin = hilbert(rho, fc[ia], ddfc[ia], p, sp, lg); 
					hs_fin = hilbert(rho, fs[ia], ddfs[ia], p, sp, lg); 
					
					
					rx1x2th[ia][ix1][ix2] = fcme_fin*(1.0/M_PI*hc_fin + 2.0*fs_fin) + fsme_fin*(1.0/M_PI*hs_fin-2.0*fc_fin);
					 
					
					// calculate I
					for(int it=0; it<Nt/2; it++){ 
						rho1 = tau1*sin(th[ith]-t[it]) + rho*cos(th[ith]-t[it]);
						rho2 = tau2*sin(th[ith]-t[it]) + rho*cos(th[ith]-t[it]); 				
						//dh1[it] = hilbert_derivative(rho1,f[8*it],ddf[8*it],p,sp);
						//dh2[it] = hilbert_derivative(rho2,f[8*it],ddf[8*it],p,sp);
						////hilbert_der_double(rho1,f[Nmul*it],ddf[Nmul*it],f[Nmul*(it+Nt/2)],ddf[Nmul*(it+Nt/2)],p,sp,&dh1[it],&dh1[it+Nt/2]);
						////hilbert_der_double(rho2,f[Nmul*it],ddf[Nmul*it],f[Nmul*(it+Nt/2)],ddf[Nmul*(it+Nt/2)],p,sp,&dh2[it],&dh2[it+Nt/2]);
						//hilbert_der_double(rho1,f_cache[ia][it],ddf_cache[ia][it],f_cache[ia][it+Nt/2],ddf_cache[ia][it+Nt/2],p,sp,&dh1[it],&dh1[it+Nt/2],lg1_cache[it]);
						//hilbert_der_double(rho2,f_cache[ia][it],ddf_cache[ia][it],f_cache[ia][it+Nt/2],ddf_cache[ia][it+Nt/2],p,sp,&dh2[it],&dh2[it+Nt/2],lg2_cache[it]);
						hilbert_der_double(rho1,f_cache[ia][it],ddf_cache[ia][it],f1_cache[ia][it+Nt/2],ddf1_cache[ia][it+Nt/2],p,sp,&dh1[it],&dh1[it+Nt/2],lg1_cache[it]);
						hilbert_der_double(rho2,f_cache[ia][it],ddf_cache[ia][it],f1_cache[ia][it+Nt/2],ddf1_cache[ia][it+Nt/2],p,sp,&dh2[it],&dh2[it+Nt/2],lg2_cache[it]);
						//dh1[it] = dh1[it+Nt/2] = dh2[it] = dh2[it+Nt/2] = 0; 
					}
					
					Ft1 = -1.0/(4.0*M_PI*M_PI)*integ(2*M_PI, Nt, dh1);
					Ft2 = -1.0/(4.0*M_PI*M_PI)*integ(2*M_PI, Nt, dh2);
					F = w*Ft1 + w*Ft2; 
					
					if(tau>0) { 
						I = exp(F); 
					} else { 
						I = exp(f_node - F); 
					}

					// calculate r
					rx1x2th[ia][ix1][ix2] = I * rx1x2th[ia][ix1][ix2]; 
					
				}
			}
		}

		//---- calculate g(x1, x2) 
		//std::cerr << "\ncalculating g(x1,x2) ...\n"; 
		for(int ia=0; ia<sa; ia++) {
			for(int ix1=0; ix1<sx; ix1++) { 
				//std::cerr << ix1 << ", "; 
				for(int ix2=0; ix2<sy; ix2++) { 
					aux=sqrt(1.0-x2[ix2]*x2[ix2]);
					if(fabs(x2[ix2]) >= 1.0 || fabs(x1[ix1]) >= aux) { 
						continue;
					}

					if(x1[ix1]<0) { 
						rx1 = (-3.0*rx1x2th[ia][ix1][ix2] + 4.0*rx1x2th[ia][ix1+1][ix2] - rx1x2th[ia][ix1+2][ix2])/(2.0*(2.0/(sx-1))); 
					} else { 
						rx1 = (3.0*rx1x2th[ia][ix1][ix2] - 4.0*rx1x2th[ia][ix1-1][ix2] + rx1x2th[ia][ix1-2][ix2])/(2.0*(2.0/(sx-1))); 
					}

					if(x2[ix2]<0) { 
						rx2 = (-3.0*rx1x2th[ia][ix1][ix2] + 4.0*rx1x2th[ia][ix1][ix2+1] - rx1x2th[ia][ix1][ix2+2])/(2.0*(2.0/(sy-1))); 
					} else { 
						rx2 = (3.0*rx1x2th[ia][ix1][ix2] - 4.0*rx1x2th[ia][ix1][ix2-1] + rx1x2th[ia][ix1][ix2-2])/(2.0*(2.0/(sy-1))); 
					}
					
					#ifdef STIR_OPENMP
					#pragma omp critical
					#endif 
					{
						image[2*ia][image_min_x +sx-ix1-1][image_min_y + ix2] += 1.0/(4.0*M_PI)*(rx1*sin(th[ith]) - rx2*cos(th[ith]))*(2.0*M_PI/sth);
					}
				}
		
			}
		}
	} //slice

	return Succeeded::yes;
}

float SRT2DReconstruction::hilbert_node(float x, float f[], float ddf[], float p[], int sp, float fn) 
{
	float dh; 
	
	dh = 0; 
	for (int i=0; i<sp-1;i++) {
		dh = dh - f[i] + f[i+1] 
			+ 1.0/36*(4*p[i]*p[i]-5*p[i]*p[i+1]-5*p[i+1]*p[i+1]-3*(p[i]-5*p[i+1])*x-6*x*x)*ddf[i]   
			+ 1.0/36*(5*p[i]*p[i]+5*p[i]*p[i+1]-4*p[i+1]*p[i+1]-3*(5*p[i]-p[i+1])*x+6*x*x)*ddf[i+1]; 
	} 

	if (fabs(x) == 1) {
		dh = 2/(sp-1)*dh;  
	} else {
		dh = fn*log((1-x)/(1+x)) + 2/(sp-1)*dh; 
	}
	
	return dh; 
}

float SRT2DReconstruction::hilbert(float x, float f[], float ddf[], float p[], int sp, float lg[])
{
	float dh, Di, Di1; 
	int i; 
	 
	i=0; 
	Di = -1.0/(p[i]-p[i+1])* (  
			(p[i+1]-x)*f[i] - (p[i]-x)*f[i+1]  
			-1.0/6*(p[i]-x)*(p[i+1]-x)*( (p[i]-2*p[i+1]+x)*ddf[i] + (2*p[i]-p[i+1]-x)*ddf[i+1] ) ); 
	dh = - f[i] + f[i+1]  
			+ 1.0/36*(4*p[i]*p[i]-5*p[i]*p[i+1]-5*p[i+1]*p[i+1]-3*(p[i]-5*p[i+1])*x-6*x*x)*ddf[i]  
			+ 1.0/36*(5*p[i]*p[i]+5*p[i]*p[i+1]-4*p[i+1]*p[i+1]-3*(5*p[i]-p[i+1])*x+6*x*x)*ddf[i+1]
			-Di*lg[i]; 

	for (i=1;i<sp-2;i++){

		Di1=-1.0/(p[i]-p[i+1])* (  
				(p[i+1]-x)*f[i] - (p[i]-x)*f[i+1]  
				-1.0/6*(p[i]-x)*(p[i+1]-x)*( (p[i]-2*p[i+1]+x)*ddf[i] + (2*p[i]-p[i+1]-x)*ddf[i+1] ) ); 
			
		dh = dh - f[i] + f[i+1]  
				+ 1.0/36*(4*p[i]*p[i]-5*p[i]*p[i+1]-5*p[i+1]*p[i+1]-3*(p[i]-5*p[i+1])*x-6*x*x)*ddf[i]  
				+ 1.0/36*(5*p[i]*p[i]+5*p[i]*p[i+1]-4*p[i+1]*p[i+1]-3*(5*p[i]-p[i+1])*x+6*x*x)*ddf[i+1] 
				+ (Di-Di1)*lg[i+1];
			
		Di = Di1; 
	}

	i=sp-2; 
	Di = -1.0/(p[i]-p[i+1])* (  
			(p[i+1]-x)*f[i] - (p[i]-x)*f[i+1]  
			-1.0/6*(p[i]-x)*(p[i+1]-x)*( (p[i]-2*p[i+1]+x)*ddf[i] + (2*p[i]-p[i+1]-x)*ddf[i+1] ) ); 
	dh = dh - f[i] + f[i+1]  
			+ 1.0/36*(4*p[i]*p[i]-5*p[i]*p[i+1]-5*p[i+1]*p[i+1]-3*(p[i]-5*p[i+1])*x-6*x*x)*ddf[i]  
			+ 1.0/36*(5*p[i]*p[i]+5*p[i]*p[i+1]-4*p[i+1]*p[i+1]-3*(5*p[i]-p[i+1])*x+6*x*x)*ddf[i+1]  
			+ Di*lg[sp-1];   
		
	dh = 2.0/(sp-1)*dh; 
	
	return dh; 
}

void SRT2DReconstruction::hilbert_der_double(float x, float f[], float ddf[], float f1[], float ddf1[], float p[], int sp, float *dhp, float *dh1p, float lg[]){

	float dh, dh1, dp; 
	dh = 0; dh1 = 0; dp = p[1]-p[2]; 
	for (int i=0; i<sp-1;i++) {
		//lg = log( fabs( (p[i+1]-x)/(p[i]-x) ) ); 
		dh = dh + f[i]/(p[i]-x) - f[i+1]/(p[i+1]-x) 
			- 1.0/4*(p[i]-3*p[i+1]+2*x)*ddf[i]  
			- 1.0/4*(3*p[i]-p[i+1]-2*x)*ddf[i+1]  
			+ ( (f[i]-f[i+1])/dp  
				- 1.0/6*(p[i]-p[i+1]-(3*(p[i+1]-x)*(p[i+1]-x))/dp)*ddf[i] 
				+ 1.0/6*(p[i]-p[i+1]-(3*(p[i]-x)*(p[i]-x))/dp)*ddf[i+1] )  
			*lg[i];  
		dh1 = dh1 + f1[i]/(p[i]-x) - f1[i+1]/(p[i+1]-x) 
			- 1.0/4*(p[i]-3*p[i+1]+2*x)*ddf1[i]  
			- 1.0/4*(3*p[i]-p[i+1]-2*x)*ddf1[i+1]  
			+ ( (f1[i]-f1[i+1])/dp  
				- 1.0/6*(p[i]-p[i+1]-(3*(p[i+1]-x)*(p[i+1]-x))/dp)*ddf1[i] 
				+ 1.0/6*(p[i]-p[i+1]-(3*(p[i]-x)*(p[i]-x))/dp)*ddf1[i+1] )  
			*lg[i];  
	} 
	dh = 2.0/(sp-1)*dh; 
	dh1 = 2.0/(sp-1)*dh1;
	*dhp = dh; 
	*dh1p = dh1; 
}

float SRT2DReconstruction::hilbert_derivative(float x, float f[], float ddf[], float p[], int sp)
{
	float dh,dp; 
	
	dh = 0; dp = p[0]-p[1]; 
	for (int i=0; i<sp-1; i++) {
		dh = dh + f[i]/(p[i]-x) - f[i+1]/(p[i+1]-x) 
			- 1.0/4*(p[i]-3*p[i+1]+2*x)*ddf[i]  
			- 1.0/4*(3*p[i]-p[i+1]-2*x)*ddf[i+1]  
			+ ( (f[i]-f[i+1])/dp  
				- 1.0/6*(p[i]-p[i+1]-(3*(p[i+1]-x)*(p[i+1]-x))/dp)*ddf[i] 
				+ 1.0/6*(p[i]-p[i+1]-(3*(p[i]-x)*(p[i]-x))/dp)*ddf[i+1] )  
			*log( fabs( (p[i+1]-x)/(p[i]-x) ) ); 

	}
	dh = 2.0/(sp-1)*dh; 
	return dh; 
}

float SRT2DReconstruction::splint(float xa[], float ya[], float y2a[], int n, float x)
{
	int  klo, khi, k; 
	float h, a, b, y; 
	
	klo = 1;
	khi = n;
	while(khi-klo > 1) { 
		k = floor((khi+klo)/2.0);
		//k = (khi+klo)/2; 
		if(xa[k] > x) { 
			khi = k;
		} else {
			klo = k;
		}
	}

	h = xa[khi]-xa[klo];
	/* if(h == 0) { 
		error('bad xa input in splint');
	} */ 
	a = (xa[khi]-x)/h;
	b = (x-xa[klo])/h;
	y = a*ya[klo] + b*ya[khi] + ((a*a*a-a)*y2a[klo]+(b*b*b-b)*y2a[khi])*(h*h)/6.0;

	return y; 
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
