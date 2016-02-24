/* 
\author Dimitra Kyriakopoulou
\author Dr Kris Thielemans

Initial version June 2012, 1st updated version (4-point symmetry included) November 2012, 2nd updated version (restriction within object boundary) January 2013, 3rd updated version (8-point symmetry included) July 2013  
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

bool
SRT2DReconstruction::
post_processing()
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

  float tangential_sampling;
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
      tangential_sampling =
	dynamic_cast<const ProjDataInfoCylindricalArcCorr&>
	(*proj_data_ptr->get_proj_data_info_ptr()).get_tangential_sampling();  
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
      tangential_sampling =
	arc_correction.get_arc_corrected_proj_data_info().get_tangential_sampling();  
    }
  
//-- Declaration of variables
	int i,j,slc,k,k1,k2,l;
	int cns;

	float pp2,aux,x,trm0, trm1, lg, pcon, rr;
	float f, f1, fB, f1B, f_, f1_, fB_, f1B_;
	float term,term1,termB,term1B,termd01B,termd1B,termd0B,termdB,termd0, termd01,termd, termd1;
	float term_, term1_, termB_, term1B_, termd0_, termd01_, termd0B_, termd01B_, termd_, termd1_, termdB_,termd1B_; 
	float d, d6, d_div_6, minus_d_div_6, half_div_d, minus_half_div_d, po_d, minus_half_div_d_po_d;

	float X,X2,X3,X4,X5,X6,X7,X8;	
	int index1, index2, index3, index4,index5, index6, index7, index8;
	int calc1, calc2, calc3, calc4, calc5, calc6, calc7, calc8;

//-- Creation of required STIR objects
	VoxelsOnCartesianGrid<float>& image = dynamic_cast<VoxelsOnCartesianGrid<float>&>(*density_ptr);
	Sinogram<float> sino = proj_data_ptr->get_sinogram((int)((proj_data_ptr->get_min_axial_pos_num(0) + proj_data_ptr->get_max_axial_pos_num(0))/2), 0);

	//if (fabs(tangential_sampling - image.get_voxel_size().x())> .001)
	  //error("SRT2D currently needs voxel-size equal to tangential sampling (i.e. zoom 1)");
	
//-- Program variables defined by use of STIR object functions
	int sx = image.get_x_size(), a=ceil(sx/2.0); 

	int imsx = image.get_min_x(); 

	const int sp =  arc_corrected_proj_data_info_sptr->get_num_tangential_poss(); 
	const int sth = sino.get_num_views();

//-- Declaration of arrays 
	float p[sp], func[sp], drcoef[sp];
	float hx[8], z[8], dercoef_rat[sth];

	Array<1, float> th(sth), x1(sx), x2(sx), termC(sth);
	Array<2,float> fhat_2D (IndexRange2D(sth, sp)), dercoef(IndexRange2D(sth,sp)), fh_rat(IndexRange2D(sth,sp-1)),fh1_rat(IndexRange2D(sth,sp-1));

//-- Declaration of constants
	pp2= -1.0/(4*M_PI*M_PI);
	pcon=pp2*M_PI/sth;

//-- Creation of theta and p arrays 
	for(i=0; i<sth; i++) 
		th[i]=i*M_PI/sth; 
	for(j=0; j<sp; j++) 
		p[j]=-1.0+2.0*j/(sp-1);

//-- Declaration of constants that depend on the values of array p
	d = p[1]-p[0]; 
	d6 = 6.0/4.0*d; 
	d_div_6=d/6.0;
	minus_d_div_6=-d/6.0;
	half_div_d=0.5/d;
	minus_half_div_d=-0.5/d;
	po_d=2.0*d*d;
	minus_half_div_d_po_d = minus_half_div_d*po_d;

  rr = 1.0*sx/(sp+1); 
//-- Creation of the grid
	for(k1=0; k1<sx; k1++)
		x1[k1]=-1.0*sx/(sp+1)+2.0*sx/(sp+1)*k1/(sx-1); 
	//x1[k1]=-91.0/129.0+2.0*91.0/129.0*k1/(sx-1); 
	//x1[k1]=-1.0+2.0*k1/(sx-1); 
	for(k2=0; k2<sx; k2++) 
		x2[k2]=-1.0*sx/(sp+1)+2.0*sx/(sp+1)*k2/(sx-1); 
	//x2[k2]=-1.0+2.0*k2/(sx-1); 

//-- Starting calculations per slice
	// 2D algorithm only
	const int segment_num = 0;      
	for(slc=proj_data_ptr->get_min_axial_pos_num(segment_num); slc<=proj_data_ptr->get_max_axial_pos_num(segment_num); slc++){
	  std::cerr << "\nSlice " << slc << std::endl;
	  
		cns=image.get_min_z() - proj_data_ptr->get_min_axial_pos_num(segment_num) + slc;

//-- Loading the sinograms  
		sino = proj_data_ptr->get_sinogram(slc, segment_num);      
		if (do_arc_correction)
		  sino =
		    arc_correction.do_arc_correction(sino);

		for(k=0; k<sp; k++)
			for(i=0; i<sth; i++){ 	 
				if(sino[sino.get_min_view_num() + i][sino.get_min_tangential_pos_num() + k] > 0)
				fhat_2D[i][k] = sino[sino.get_min_view_num() + i][sino.get_min_tangential_pos_num()  + k];
			else
					fhat_2D[i][k] = 0; 
		      	}

//-- Calculation of second derivative by use of function spline
		for(i=0; i<sth; i++){
			for(k=0; k<sp; k++)
				func[k]=fhat_2D[i][k];
			spline(p,func,sp,drcoef);
			for(k=0; k<sp; k++)  
				dercoef[i][k]=drcoef[k]; 
		}

//-- Calculation of termC 
		for(i=0; i<sth; i++) {
			termC[i] = (dercoef[i][0]*(3.0*p[1]-p[0]) + dercoef[i][sp-1]*(p[sp-1]-3.0*p[sp-2]))/4.0;
			for(int k=1; k<sp-1; k++){ 
				termC[i] += d6*dercoef[i][k];
			}
		}

//-- Auxiliary functions for speeding up purposes 
		for(i=1; i<sth; i++) {
			for(int k=0; k<sp-1; k++){ 
				fh_rat[i][k]=(fhat_2D[i][k+1]-fhat_2D[i][k])/d;
				fh1_rat[i][k]=(fhat_2D[sth-i][k+1]-fhat_2D[sth-i][k])/d;
			}
		}

		for(i=1; i<sth; i++) 
			dercoef_rat[i]=0.5*(dercoef[i][0]-dercoef[i][sp-1]);
	

//----- Starting the calculation of f(x1,x2) -----
		for(k1=0; k1<=a; k1++){
		  std::cerr << " k1 " << k1;
	  
			for(k2=0; k2<=k1; k2++){ 
				//aux=sqrt(1.0-x2[k2]*x2[k2]);
				aux=sqrt(rr*rr-x2[k2]*x2[k2]);
				//if(fabs(x2[k2]) >= 1.0 || fabs(x1[k1]) >= aux){
				if(fabs(x2[k2]) >= rr || fabs(x1[k1]) >= aux){  
					image[cns][imsx + k1][imsx + k2] = 0; 
					image[cns][imsx + k1][imsx + sx - k2-1] = 0;
					image[cns][imsx + sx - k1-1][imsx + k2] = 0; 
					image[cns][imsx + sx - k1-1][imsx + sx - k2-1] = 0;
				
					image[cns][imsx + k2][imsx + k1] = 0; 
					image[cns][imsx + sx - k2-1][imsx + k1] = 0;
					image[cns][imsx + k2][imsx + sx - k1-1] = 0; 
					image[cns][imsx + sx - k2-1][imsx + sx - k1-1] = 0;
				
					continue;
				} 


				if(!thres_restr_bound_vector.empty())
					thres_restr_bound=thres_restr_bound_vector[proj_data_ptr->get_max_segment_num()+slc]; 


//---- RESTRICTION TO OBJECT BOUNDARY  
 				calc1=1, calc2=1, calc3=1, calc4=1, calc5=1, calc6=1, calc7=1, calc8=1;

				for(l=0; l<sth && (calc1 || calc2 || calc3 || calc4 || calc5 || calc6 || calc7 || calc8); l++){  

					X=x2[k2]*cos(th[l])-x1[k1]*sin(th[l]); 
					index1=(int)floor((X-p[0])/(p[1]-p[0])); 
				
					X2=x2[sx-k2-1]*cos(th[l])-x1[k1]*sin(th[l]);
					index2=(int)floor((X2-p[0])/(p[1]-p[0]));

					X3=x2[k2]*cos(th[l])-x1[sx-k1-1]*sin(th[l]);
					index3=(int)floor((X3-p[0])/(p[1]-p[0]));

					X4=x2[sx-k2-1]*cos(th[l])-x1[sx-k1-1]*sin(th[l]);
					index4=(int)floor((X4-p[0])/(p[1]-p[0]));

				 	X5=x2[k1]*cos(th[l])-x1[k2]*sin(th[l]); 
					index5=(int)floor((X5-p[0])/(p[1]-p[0])); 
				
					X6=x2[k1]*cos(th[l])-x1[sx-k2-1]*sin(th[l]); 
					index6=(int)floor((X6-p[0])/(p[1]-p[0])); 
				
					X7=x2[sx-k1-1]*cos(th[l])-x1[k2]*sin(th[l]); 
					index7=(int)floor((X7-p[0])/(p[1]-p[0])); 
				
					X8=x2[sx-k1-1]*cos(th[l])-x1[sx-k2-1]*sin(th[l]); 
					index8=(int)floor((X8-p[0])/(p[1]-p[0])); 

					if(fhat_2D[l][index1] <=thres_restr_bound) {
						calc1 = 0; 
						image[cns][imsx+k1][imsx+k2] = 0; 
					}
					if(fhat_2D[l][index2] <=thres_restr_bound) { 
						calc2 = 0; 
						image[cns][imsx+k1][imsx+sx - k2-1] = 0; 
					}
					if(fhat_2D[l][index3] <=thres_restr_bound) { 
						calc3 = 0; 
						image[cns][imsx+sx-k1-1][imsx+k2] = 0; 
					}
					if(fhat_2D[l][index4] <=thres_restr_bound) { 
						calc4 = 0; 
						image[cns][imsx+sx-k1-1][imsx+sx - k2-1] = 0; 
					}

					if(fhat_2D[l][index5] <=thres_restr_bound) { 
						calc5 = 0; 
						image[cns][imsx + k2][imsx + k1] = 0; 
					}

					if(fhat_2D[l][index6] <=thres_restr_bound) { 
						calc6 = 0; 
						image[cns][imsx + sx - k2-1][imsx + k1] = 0; 
					}

					if(fhat_2D[l][index7] <=thres_restr_bound) { 
						calc7 = 0; 
						image[cns][imsx + k2][imsx + sx - k1-1] = 0; 
					}

					if(fhat_2D[l][index8] <=thres_restr_bound) { 
						calc8 = 0; 
						image[cns][imsx + sx - k2-1][imsx + sx - k1-1] = 0; 
					}

				}

				if(l < sth) {
					continue; 
				}
//Completion of restriction to boundary object takes place during the saving of the image



//----- COMPUTATION OF NON-SYMMETRIC RHO'S: theta = 0 ----- 

				z[0]=x2[k2]*cos(th[0])-x1[k1]*sin(th[0]); 			// Point with (x1,x2) index: (k1,k2)
	
				z[1]=x2[sx-k2-1]*cos(th[0])-x1[k1]*sin(th[0]);		// Point with (x1,x2) index: (k1, sx-k2+1)

				z[2]=x2[k2]*cos(th[0])-x1[sx-k1-1]*sin(th[0]);      // Point with (x1,x2) index: (sx-k1+1, k2) 

				z[3]=x2[sx-k2-1]*cos(th[0])-x1[sx-k1-1]*sin(th[0]); // Point with (x1,x2) index: (sx-k1+1, k2) 

				z[4]=x2[k1]*cos(th[0])-x1[k2]*sin(th[0]); 			// Point with (x1,x2) index: (k2, k1)

				z[6]=x2[k1]*cos(th[0])-x1[sx-k2-1]*sin(th[0]);		// Point with (x1,x2) index: (sx-k2+1, k1)

				z[5]=x2[sx-k1-1]*cos(th[0])-x1[k2]*sin(th[0]); 		// Point with (x1,x2) index: (k2,sx-k1-1)
		
				z[7]=x2[sx-k1-1]*cos(th[0])-x1[sx-k2-1]*sin(th[0]);	// Point with (x1,x2) index: (sx-k2+1, sx-k1-1)


				for (int w=0;w<8;w++){

					term = 0.5*(dercoef[0][sp-1] - dercoef[0][0])*z[w] + termC[0];

					if(p[sp-1] != z[w]) lg = log(p[sp-1]-z[w]);
					else lg = 0;
	//-- D_{sp-2}(x)
					term = term + ((fhat_2D[0][sp-1]-fhat_2D[0][sp-2])/d + dercoef[0][sp-2]*(d_div_6 +minus_half_div_d*pow(p[sp-1]-z[w],2)) + dercoef[0][sp-1]*(minus_d_div_6 +half_div_d*pow(p[sp-2]-z[w],2)))*lg ;

	//-- D_0(x)  
					trm0 = d_div_6 + minus_half_div_d*pow(p[1]-z[w],2); 
					termd0 = (fhat_2D[0][1]-fhat_2D[0][0])/d + dercoef[0][0]*trm0 + dercoef[0][1]*(minus_d_div_6 +half_div_d*pow(p[0]-z[w],2)) ;

					if(z[w] != p[0]) lg = log(z[w]-p[0]);  
					else lg = 0;  
					term = term - termd0 * lg; 
			   
					for (k=0; k<sp-2; k++){
						trm1 = d_div_6 +minus_half_div_d*pow(p[k+2]-z[w],2);
						termd =  (fhat_2D[0][k+2]-fhat_2D[0][k+1])/d  + dercoef[0][k+1]*trm1 - dercoef[0][k+2]*trm0 ; 
						if (z[w] != p[k+1]){ 
							lg = log(fabs(z[w]-p[k+1])); 
							term = term + (termd0-termd) * lg; 
						}
						termd0 = termd;
						trm0 = trm1; 
					}  

					hx[w]=term;
				}			

			
	
//----- COMPUTATION OF SYMMETRIC RHO'S -----
// The values of 8 points (x1(k1),x2(k2)) are calculated at the same iteration, in order to take advandage of the fact that they share the same logarithmic value, in particular the points corresponding to indices: (k1,k2), (k1,sx-k2-1), (sx-k1-1,k2), (sx-k1-1,sx-k2-1), (k2,k1), (sx-k2-1,k1), (k2,sx-k1-1), (sx-k2-1,sx-k1-1).   


//----computation of h_rho(j) for j=sth/2 . Used for (k2,k1), (sx-k2-1,k1), (k2,sx-k1-1), (sx-k2-1,sx-k1-1)-----

				x=x2[k2]*cos(th[0])-x1[k1]*sin(th[0]);  
				j = (int)(ceil(sth/2.0));				

				term_ = 0.5*(dercoef[j][0] - dercoef[j][sp-1])*x + termC[j];
				term1_ = 0.5*(dercoef[sth-j][0] - dercoef[sth-j][sp-1])*x + termC[j];

				termB_ = 0.5*(dercoef[sth-j][sp-1] - dercoef[sth-j][0])*x + termC[j];
				term1B_ = 0.5*(dercoef[j][sp-1] - dercoef[j][0])*x + termC[j];

				if(p[sp-1] != x) lg = log(p[sp-1]-x);
				else lg = 0;
//-- D_0(x) 
				term_ = term_ + (-fh_rat[j][0] + dercoef[j][1]*(d_div_6 + minus_half_div_d*pow(p[0]+x,2)) + dercoef[j][0]*(minus_d_div_6 + half_div_d*pow(p[1]+x,2)))*lg ;
				term1_ = term1_ + (-fh1_rat[j][0] + dercoef[sth-j][1]*(d_div_6 +minus_half_div_d*pow(p[0]+x,2)) + dercoef[sth-j][0]*(minus_d_div_6 + half_div_d*pow(p[1]+x,2)))*lg ;

 //-- D_{sp-1}(x) 
				termB_ = termB_ + (fh1_rat[j][sp-2] + dercoef[sth-j][sp-2]*(d_div_6 + minus_half_div_d*pow(p[sp-1]-x,2)) + dercoef[sth-j][sp-1]*(minus_d_div_6 + half_div_d*pow(p[sp-2]-x,2)))*lg ;  
				term1B_ = term1B_ + (fh_rat[j][sp-2] + dercoef[j][sp-2]*(d_div_6 + minus_half_div_d*pow(p[sp-1]-x,2)) + dercoef[j][sp-1]*(minus_d_div_6 + half_div_d*pow(p[sp-2]-x,2)))*lg ;    

				trm0 = d_div_6 + minus_half_div_d*pow(p[1]-x,2); 

				termd0_ = -fh_rat[j][sp-2] + dercoef[j][sp-1]*trm0 + dercoef[j][sp-2]*(minus_d_div_6 + half_div_d*pow(p[sp-1]+x,2));
				termd01_ = fh1_rat[j][sp-2] + dercoef[sth-j][sp-1]*trm0 + dercoef[sth-j][sp-2]*(minus_d_div_6 + half_div_d*pow(p[sp-1]+x,2));

				termd0B_ = fh1_rat[j][0] + dercoef[sth-j][0]*trm0 + dercoef[sth-j][1]*(minus_d_div_6 + half_div_d*pow(p[0]-x,2)) ;
				termd01B_ = fh_rat[j][0] + dercoef[j][0]*trm0 + dercoef[j][1]*(minus_d_div_6 + half_div_d*pow(p[0]-x,2)) ;

				if(x != p[0]) lg = log(x-p[0]); // |x-p[0]|==|-x-p[sp-1]|
				else lg = 0; 
			
				term_ = term_ - termd0_ * lg;  
				term1_ = term1_ - termd01_ * lg;  

				termB_ = termB_ - termd0B_ * lg; 
				term1B_ = term1B_ - termd01B_ * lg; 

				for(k=0; k<sp-2; k++) {
				  	trm1 = d_div_6 + minus_half_div_d*pow(p[k+2]-x,2);
					          
					termd_ = ( -fh_rat[j][sp-k-3]  + dercoef[j][sp-k-2]*trm1 - dercoef[j][sp-k-3]*trm0 ); 
					termd1_ = ( -fh1_rat[j][sp-k-3] + dercoef[sth-j][sp-k-2]*trm1 - dercoef[sth-j][sp-k-3]*trm0);  

					termdB_ = ( fh1_rat[j][k+1] + dercoef[sth-j][k+1]*trm1 - dercoef[sth-j][k+2]*trm0 ); 
					termd1B_ = ( fh_rat[j][k+1] + dercoef[j][k+1]*trm1 - dercoef[j][k+2]*trm0 ); 

					if(x != p[k+1]) {  
						lg = log(fabs(x-p[k+1]));

					term_ = term_ + (termd0_-termd_) * lg; 
					term1_ = term1_ + (termd01_-termd1_) * lg;  

					termB_ = termB_ + (termd0B_-termdB_) * lg; 
					term1B_ = term1B_ + (termd01B_-termd1B_) * lg; 
					}

					trm0 = trm1; 

					termd0_ = termd_;
					termd01_ = termd1_;

					termd0B_ = termdB_;
					termd01B_ = termd1B_;
				}  

				hx[4]+= term_;
				hx[5]+= term1_;

				hx[6]+= termB_; 
				hx[7]+= term1B_; 
			

//---- Computation of first half of h_rho for theta=1 to theta=ceil(sth/2)-1 -----
				for(i=1; i<ceil(sth/2.0); i++) {
					x=x2[k2]*cos(th[i])-x1[k1]*sin(th[i]);  
					j = (int)(ceil(sth/2.0))-i;				

					term = 0.5*(dercoef[i][sp-1] - dercoef[i][0])*x + termC[i];
					term1 = 0.5*(dercoef[sth-i][sp-1] - dercoef[sth-i][0])*x + termC[i];

					termB = 0.5*(dercoef[sth-i][0] - dercoef[sth-i][sp-1])*x + termC[i];
					term1B = 0.5*(dercoef[i][0] - dercoef[i][sp-1])*x + termC[i];

					term_ = 0.5*(dercoef[j][0] - dercoef[j][sp-1])*x + termC[j];
					term1_ = 0.5*(dercoef[sth-j][0] - dercoef[sth-j][sp-1])*x + termC[j];

					termB_ = 0.5*(dercoef[sth-j][sp-1] - dercoef[sth-j][0])*x + termC[j];
					term1B_ = 0.5*(dercoef[j][sp-1] - dercoef[j][0])*x + termC[j];

					
					if(p[sp-1] != x) lg = log(p[sp-1]-x);
					else lg = 0;

					term = term + (fh_rat[i][sp-2] + dercoef[i][sp-2]*(d_div_6 + minus_half_div_d*pow(p[sp-1]-x,2)) + dercoef[i][sp-1]*(minus_d_div_6 + half_div_d*pow(p[sp-2]-x,2)))*lg ;
					term1 = term1 + (fh1_rat[i][sp-2] + dercoef[sth-i][sp-2]*(d_div_6 +minus_half_div_d*pow(p[sp-1]-x,2)) + dercoef[sth-i][sp-1]*(minus_d_div_6 + half_div_d*pow(p[sp-2]-x,2)))*lg ;

					termB = termB + (-fh1_rat[i][0] + dercoef[sth-i][1]*(d_div_6 + minus_half_div_d*pow(p[0]+x,2)) + dercoef[sth-i][0]*(minus_d_div_6 + half_div_d*pow(p[1]+x,2)))*lg ;  
		term1B = term1B + (-fh_rat[i][0] + dercoef[i][1]*(d_div_6 + minus_half_div_d*pow(p[0]+x,2)) + dercoef[i][0]*(minus_d_div_6 + half_div_d*pow(p[1]+x,2)))*lg ;    

					term_ = term_ + (-fh_rat[j][0] + dercoef[j][1]*(d_div_6 + minus_half_div_d*pow(p[0]+x,2)) + dercoef[j][0]*(minus_d_div_6 + half_div_d*pow(p[1]+x,2)))*lg ;
					term1_ = term1_ + (-fh1_rat[j][0] + dercoef[sth-j][1]*(d_div_6 +minus_half_div_d*pow(p[0]+x,2)) + dercoef[sth-j][0]*(minus_d_div_6 + half_div_d*pow(p[1]+x,2)))*lg ;

					termB_ = termB_ + (fh1_rat[j][sp-2] + dercoef[sth-j][sp-2]*(d_div_6 + minus_half_div_d*pow(p[sp-1]-x,2)) + dercoef[sth-j][sp-1]*(minus_d_div_6 + half_div_d*pow(p[sp-2]-x,2)))*lg ;  
					term1B_ = term1B_ + (fh_rat[j][sp-2] + dercoef[j][sp-2]*(d_div_6 + minus_half_div_d*pow(p[sp-1]-x,2)) + dercoef[j][sp-1]*(minus_d_div_6 + half_div_d*pow(p[sp-2]-x,2)))*lg ;    

					trm0 = d_div_6 + minus_half_div_d*pow(p[1]-x,2); 

					termd0 = fh_rat[i][0] + dercoef[i][0]*trm0 + dercoef[i][1]*(minus_d_div_6 + half_div_d*pow(p[0]-x,2));
					termd01 = fh1_rat[i][0] + dercoef[sth-i][0]*trm0 + dercoef[sth-i][1]*(minus_d_div_6 + half_div_d*pow(p[0]-x,2));

					termd0B = -fh1_rat[i][sp-2] + dercoef[sth-i][sp-1]*trm0 + dercoef[sth-i][sp-2]*(minus_d_div_6 + half_div_d*pow(p[sp-1]+x,2)) ;
					termd01B = -fh_rat[i][sp-2] + dercoef[i][sp-1]*trm0 + dercoef[i][sp-2]*(minus_d_div_6 + half_div_d*pow(p[sp-1]+x,2)); 

					termd0_ = -fh_rat[j][sp-2] + dercoef[j][sp-1]*trm0 + dercoef[j][sp-2]*(minus_d_div_6 + half_div_d*pow(p[sp-1]+x,2));
					termd01_ = -fh1_rat[j][sp-2] + dercoef[sth-j][sp-1]*trm0 + dercoef[sth-j][sp-2]*(minus_d_div_6 + half_div_d*pow(p[sp-1]+x,2));

					termd0B_ = fh1_rat[j][0] + dercoef[sth-j][0]*trm0 + dercoef[sth-j][1]*(minus_d_div_6 + half_div_d*pow(p[0]-x,2)) ;
					termd01B_ = fh_rat[j][0] + dercoef[j][0]*trm0 + dercoef[j][1]*(minus_d_div_6 + half_div_d*pow(p[0]-x,2)) ;

					if(x != p[0]) lg = log(x-p[0]);  
					else lg = 0; 
				
					term = term - termd0 * lg;  
					term1 = term1 - termd01 * lg;  

					termB = termB - termd0B * lg; 
					term1B = term1B - termd01B * lg; 

					term_ = term_ - termd0_ * lg;  
					term1_ = term1_ - termd01_ * lg;  

					termB_ = termB_ - termd0B_ * lg; 
					term1B_ = term1B_ - termd01B_ * lg; 

					for(k=0; k<sp-2; k++) {
				  		trm1 = d_div_6 + minus_half_div_d*pow(p[k+2]-x,2);

						termd = ( fh_rat[i][k+1]  + dercoef[i][k+1]*trm1 - dercoef[i][k+2]*trm0 ); 
						termd1 = ( fh1_rat[i][k+1] + dercoef[sth-i][k+1]*trm1 - dercoef[sth-i][k+2]*trm0);  

						termdB = ( -fh1_rat[i][sp-k-3] + dercoef[sth-i][sp-k-2]*trm1 - dercoef[sth-i][sp-k-3]*trm0 ); 
						termd1B = ( -fh_rat[i][sp-k-3] + dercoef[i][sp-k-2]*trm1 - dercoef[i][sp-k-3]*trm0 ); 

						termd_ = ( -fh_rat[j][sp-k-3]  + dercoef[j][sp-k-2]*trm1 - dercoef[j][sp-k-3]*trm0 ); 
						termd1_ = ( -fh1_rat[j][sp-k-3] + dercoef[sth-j][sp-k-2]*trm1 - dercoef[sth-j][sp-k-3]*trm0);  

						termdB_ = ( fh1_rat[j][k+1] + dercoef[sth-j][k+1]*trm1 - dercoef[sth-j][k+2]*trm0 ); 
						termd1B_ = ( fh_rat[j][k+1] + dercoef[j][k+1]*trm1 - dercoef[j][k+2]*trm0 ); 

						if(x != p[k+1]) {  
							lg = log(fabs(x-p[k+1]));
							term = term + (termd0-termd) * lg; 
							term1 = term1 + (termd01-termd1) * lg;  

							termB = termB + (termd0B-termdB) * lg; 
							term1B = term1B + (termd01B-termd1B) * lg; 

							term_ = term_ + (termd0_-termd_) * lg; 
							term1_ = term1_ + (termd01_-termd1_) * lg;  

							termB_ = termB_ + (termd0B_-termdB_) * lg; 
							term1B_ = term1B_ + (termd01B_-termd1B_) * lg; 
						}

						termd0 = termd;
						termd01 = termd1;

						termd0B = termdB;
						termd01B = termd1B;

						trm0 = trm1; 

						termd0_ = termd_;
						termd01_ = termd1_;

						termd0B_ = termdB_;
						termd01B_ = termd1B_;

					}  

					hx[0]+=term ;	
					hx[1]+= term1;

					hx[2]+= termB; 
					hx[3]+= term1B; 

					hx[4]+= term_;
					hx[5]+= term1_;

					hx[6]+= termB_; 
					hx[7]+= term1B_; 
				}



//----computation of h_rho(j) for j=sth . Used for (k1,k2), (sx-k1-1,k2), (k1,sx-k2-1), (sx-k1-1,sx-k2-1)-----

				i=ceil(sth/2.0); 
				x=x2[k2]*cos(th[i])-x1[k1]*sin(th[i]); 
				j = (int)ceil(3*sth/2)-i-0; 

				term = 0.5*(dercoef[i][sp-1] - dercoef[i][0])*x + termC[i];
				term1 = 0.5*(dercoef[sth-i][sp-1] - dercoef[sth-i][0])*x + termC[i];

				termB = 0.5*(dercoef[sth-i][0] - dercoef[sth-i][sp-1])*x + termC[i];
				term1B = 0.5*(dercoef[i][0] - dercoef[i][sp-1])*x + termC[i];
			
				if(p[sp-1] != x) lg = log(p[sp-1]-x);
				else lg = 0;
			
				term = term + (fh_rat[i][sp-2] + dercoef[i][sp-2]*(d_div_6 + minus_half_div_d*pow(p[sp-1]-x,2)) + dercoef[i][sp-1]*(minus_d_div_6 + half_div_d*pow(p[sp-2]-x,2)))*lg ;
				term1 = term1 + (fh1_rat[i][sp-2]+ dercoef[sth-i][sp-2]*(d_div_6 +minus_half_div_d*pow(p[sp-1]-x,2)) + dercoef[sth-i][sp-1]*(minus_d_div_6 + half_div_d*pow(p[sp-2]-x,2)))*lg ;

				termB = termB + (-fh1_rat[i][0]+ dercoef[sth-i][1]*(d_div_6 + minus_half_div_d*pow(p[0]+x,2)) + dercoef[sth-i][0]*(minus_d_div_6 + half_div_d*pow(p[1]+x,2)))*lg ;  
				term1B = term1B + (-fh_rat[i][0] + dercoef[i][1]*(d_div_6 + minus_half_div_d*pow(p[0]+x,2)) + dercoef[i][0]*(minus_d_div_6 + half_div_d*pow(p[1]+x,2)))*lg ;    
		        

				trm0 = d_div_6 + minus_half_div_d*pow(p[1]-x,2); 

				termd0 = fh_rat[i][0] + dercoef[i][0]*trm0 + dercoef[i][1]*(minus_d_div_6 + half_div_d*pow(p[0]-x,2));
				termd01 = fh1_rat[i][0] + dercoef[sth-i][0]*trm0 + dercoef[sth-i][1]*(minus_d_div_6 + half_div_d*pow(p[0]-x,2));

				termd0B = -fh1_rat[i][sp-2] + dercoef[sth-i][sp-1]*trm0 + dercoef[sth-i][sp-2]*(minus_d_div_6 + half_div_d*pow(p[sp-1]+x,2)) ;
				termd01B = -fh_rat[i][sp-2] + dercoef[i][sp-1]*trm0 + dercoef[i][sp-2]*(minus_d_div_6 + half_div_d*pow(p[sp-1]+x,2)) ;

				if(x != p[0]) lg = log(x-p[0]);  
				else lg = 0; 
			
				term = term - termd0 * lg;  
				term1 = term1 - termd01 * lg;  

				termB = termB - termd0B * lg; 
				term1B = term1B - termd01B * lg; 

				for(k=0; k<sp-2; k++) {

					trm1 = d_div_6 + minus_half_div_d*pow(p[k+2]-x,2);
		             
					termd =  fh_rat[i][k+1]  + dercoef[i][k+1]*trm1 - dercoef[i][k+2]*trm0 ; 
					termd1 =  fh1_rat[i][k+1] + dercoef[sth-i][k+1]*trm1 - dercoef[sth-i][k+2]*trm0;  

					termdB =  -fh1_rat[i][sp-k-3] + dercoef[sth-i][sp-k-2]*trm1 - dercoef[sth-i][sp-k-3]*trm0 ; 
					termd1B =  -fh_rat[i][sp-k-3] + dercoef[i][sp-k-2]*trm1 - dercoef[i][sp-k-3]*trm0 ; 

					if(x != p[k+1]) {  
						lg = log(fabs(x-p[k+1]));
						term = term + (termd0-termd) * lg; 
						term1 = term1 + (termd01-termd1) * lg;  

						termB = termB + (termd0B-termdB) * lg; 
						term1B = term1B + (termd01B-termd1B) * lg; 
					}

					termd0 = termd;
					termd01 = termd1;

					termd0B = termdB;
					termd01B = termd1B;

					trm0 = trm1; 

				} 

				hx[0]+= term;
				hx[1]+= term1;

				hx[2]+= termB; 
				hx[3]+= term1B; 



//----- Computation of second half of h_rho for theta=ceil(sth/2)+1 to theta=sth-1 -----

				for(i=ceil(sth/2.0)+1; i<sth; i++) {
					x=x2[k2]*cos(th[i])-x1[k1]*sin(th[i]); 
					j = (int)ceil(3*sth/2)-i-0; 

					term = 0.5*(dercoef[i][sp-1] - dercoef[i][0])*x + termC[i];
					term1 = 0.5*(dercoef[sth-i][sp-1] - dercoef[sth-i][0])*x + termC[i];

					termB = 0.5*(dercoef[sth-i][0] - dercoef[sth-i][sp-1])*x + termC[i];
					term1B = 0.5*(dercoef[i][0] - dercoef[i][sp-1])*x + termC[i];

					term_ = 0.5*(dercoef[j][sp-1] - dercoef[j][0])*x + termC[j];
					term1_ = 0.5*(dercoef[sth-j][sp-1] - dercoef[sth-j][0])*x + termC[j];

					termB_ = 0.5*(dercoef[sth-j][0] - dercoef[sth-j][sp-1])*x + termC[j];
					term1B_ = 0.5*(dercoef[j][0] - dercoef[j][sp-1])*x + termC[j];
				
					if(p[sp-1] != x) lg = log(p[sp-1]-x);
					else lg = 0;

					term = term + (fh_rat[i][sp-2] + dercoef[i][sp-2]*(d_div_6 + minus_half_div_d*pow(p[sp-1]-x,2)) + dercoef[i][sp-1]*(minus_d_div_6 + half_div_d*pow(p[sp-2]-x,2)))*lg ;
					term1 = term1 + (fh1_rat[i][sp-2]+ dercoef[sth-i][sp-2]*(d_div_6 +minus_half_div_d*pow(p[sp-1]-x,2)) + dercoef[sth-i][sp-1]*(minus_d_div_6 + half_div_d*pow(p[sp-2]-x,2)))*lg ;

					termB = termB + (-fh1_rat[i][0]+ dercoef[sth-i][1]*(d_div_6 + minus_half_div_d*pow(p[0]+x,2)) + dercoef[sth-i][0]*(minus_d_div_6 + half_div_d*pow(p[1]+x,2)))*lg ;  
					term1B = term1B + (-fh_rat[i][0] + dercoef[i][1]*(d_div_6 + minus_half_div_d*pow(p[0]+x,2)) + dercoef[i][0]*(minus_d_div_6 + half_div_d*pow(p[1]+x,2)))*lg ;    

					term_ = term_ + (fh_rat[j][sp-2] + dercoef[j][sp-2]*(d_div_6 + minus_half_div_d*pow(p[sp-1]-x,2)) + dercoef[j][sp-1]*(minus_d_div_6 + half_div_d*pow(p[sp-2]-x,2)))*lg ;
					term1_ = term1_ + (fh1_rat[j][sp-2] + dercoef[sth-j][sp-2]*(d_div_6 +minus_half_div_d*pow(p[sp-1]-x,2)) + dercoef[sth-j][sp-1]*(minus_d_div_6 + half_div_d*pow(p[sp-2]-x,2)))*lg ;

					termB_ = termB_ + (-fh1_rat[j][0] + dercoef[sth-j][1]*(d_div_6 + minus_half_div_d*pow(p[0]+x,2)) + dercoef[sth-j][0]*(minus_d_div_6 + half_div_d*pow(p[1]+x,2)))*lg ;  
					term1B_ = term1B_ + (-fh_rat[j][0]+ dercoef[j][1]*(d_div_6 + minus_half_div_d*pow(p[0]+x,2)) + dercoef[j][0]*(minus_d_div_6 + half_div_d*pow(p[1]+x,2)))*lg ;    
		            

					trm0 = d_div_6 + minus_half_div_d*pow(p[1]-x,2); 
				
					termd0 = fh_rat[i][0] + dercoef[i][0]*trm0 + dercoef[i][1]*(minus_d_div_6 + half_div_d*pow(p[0]-x,2));
					termd01 = fh1_rat[i][0] + dercoef[sth-i][0]*trm0 + dercoef[sth-i][1]*(minus_d_div_6 + half_div_d*pow(p[0]-x,2));

					termd0B = -fh1_rat[i][sp-2] + dercoef[sth-i][sp-1]*trm0 + dercoef[sth-i][sp-2]*(minus_d_div_6 + half_div_d*pow(p[sp-1]+x,2)) ;
					termd01B = -fh_rat[i][sp-2] + dercoef[i][sp-1]*trm0 + dercoef[i][sp-2]*(minus_d_div_6 + half_div_d*pow(p[sp-1]+x,2)) ;

					termd0_ = fh_rat[j][0] + dercoef[j][0]*trm0 + dercoef[j][1]*(minus_d_div_6 + half_div_d*pow(p[0]-x,2));
					termd01_ = fh1_rat[j][0] + dercoef[sth-j][0]*trm0 + dercoef[sth-j][1]*(minus_d_div_6 + half_div_d*pow(p[0]-x,2));

					termd0B_ = -fh1_rat[j][sp-2] + dercoef[sth-j][sp-1]*trm0 + dercoef[sth-j][sp-2]*(minus_d_div_6 + half_div_d*pow(p[sp-1]+x,2)) ;
					termd01B_ = -fh_rat[j][sp-2] + dercoef[j][sp-1]*trm0 + dercoef[j][sp-2]*(minus_d_div_6 + half_div_d*pow(p[sp-1]+x,2)) ;

					if(x != p[0]) lg = log(x-p[0]);  
					else lg = 0; 
				
					term = term - termd0 * lg;  
					term1 = term1 - termd01 * lg;  

					termB = termB - termd0B * lg; 
					term1B = term1B - termd01B * lg; 

					term_ = term_ - termd0_ * lg;  
					term1_ = term1_ - termd01_ * lg;  

					termB_ = termB_ - termd0B_ * lg; 
					term1B_ = term1B_ - termd01B_ * lg;

					for(k=0; k<sp-2; k++) {

						trm1 = d_div_6 + minus_half_div_d*pow(p[k+2]-x,2);
		                 
						termd = ( fh_rat[i][k+1]  + dercoef[i][k+1]*trm1 - dercoef[i][k+2]*trm0 ); 
						termd1 = ( fh1_rat[i][k+1] + dercoef[sth-i][k+1]*trm1 - dercoef[sth-i][k+2]*trm0);  

						termdB = ( -fh1_rat[i][sp-k-3] + dercoef[sth-i][sp-k-2]*trm1 - dercoef[sth-i][sp-k-3]*trm0 ); 
						termd1B = ( -fh_rat[i][sp-k-3] + dercoef[i][sp-k-2]*trm1 - dercoef[i][sp-k-3]*trm0 ); 

						termd_ = ( fh_rat[j][k+1]  + dercoef[j][k+1]*trm1 - dercoef[j][k+2]*trm0); 
						termd1_ = ( fh1_rat[j][k+1] + dercoef[sth-j][k+1]*trm1 - dercoef[sth-j][k+2]*trm0);  

						termdB_ = ( -fh1_rat[j][sp-k-3] + dercoef[sth-j][sp-k-2]*trm1 - dercoef[sth-j][sp-k-3]*trm0); 
						termd1B_ = ( -fh_rat[j][sp-k-3] + dercoef[j][sp-k-2]*trm1 - dercoef[j][sp-k-3]*trm0); 

						if(x != p[k+1]) {  
							lg = log(fabs(x-p[k+1]));

							term = term + (termd0-termd) * lg; 
							term1 = term1 + (termd01-termd1) * lg;  

							termB = termB + (termd0B-termdB) * lg; 
							term1B = term1B + (termd01B-termd1B) * lg; 

							term_ = term_ + (termd0_-termd_) * lg; 
							term1_ = term1_ + (termd01_-termd1_) * lg;  

							termB_ = termB_ + (termd0B_-termdB_) * lg; 
							term1B_ = term1B_ + (termd01B_-termd1B_) * lg; 
						}

						termd0 = termd;
						termd01 = termd1;

						termd0B = termdB;
						termd01B = termd1B;

						trm0 = trm1; 

						termd0_ = termd_;
						termd01_ = termd1_;

						termd0B_ = termdB_;
						termd01B_ = termd1B_;

					} 
	
					hx[0]+= term;
					hx[1]+= term1;

					hx[2]+= termB; 
					hx[3]+= term1B; 

					hx[4]+= term_;
					hx[5]+= term1_;

					hx[6]+= termB_; 
					hx[7]+= term1B_; 
				}

			 
//--Integration 
				f=2*pcon*hx[0]*2/(sp-1);		
				f1=2*pcon*hx[1]*2/(sp-1); 
				fB=2*pcon*hx[2]*2/(sp-1); 
				f1B=2*pcon*hx[3]*2/(sp-1); 
				f_=2*pcon*hx[4]*2/(sp-1);
				f1_=2*pcon*hx[5]*2/(sp-1); 
				fB_=2*pcon*hx[6]*2/(sp-1);
				f1B_=2*pcon*hx[7]*2/(sp-1);   


//--Saving f
//Positivity constraint and completion of restriction to boundary object
/*				image[cns][imsx +sx-k1-1][imsx + k2] = f>0 && calc1==1 ? f : 0;  
				image[cns][imsx +sx-k1-1][imsx + sx - k2-1] = f1>0 && calc2==1 ? f1 : 0; 

				image[cns][imsx +k1][imsx + k2] = fB>0 && calc3==1 ? fB : 0; 
				image[cns][imsx +k1][imsx + sx - k2-1] = f1B>0 && calc4==1 ? f1B : 0; 

				image[cns][imsx +sx-k2-1][imsx + k1] = f_>0 && calc5==1 ? f_ : 0; 
				image[cns][imsx +k2][imsx + k1] = fB_>0 && calc6==1 ? fB_ : 0; 

				image[cns][imsx +sx-k2-1][imsx + sx - k1-1] = f1_>0 && calc7==1 ? f1_ : 0; 
				image[cns][imsx +k2][imsx + sx - k1-1] = f1B_>0 && calc8==1 ? f1B_ : 0; 
*/
			 	image[cns][imsx +sx-k1-1][imsx + k2] = (calc1==1 ? f : 0);  
				image[cns][imsx +sx-k1-1][imsx + sx - k2-1] = (calc2==1 ? f1 : 0); 

				image[cns][imsx +k1][imsx + k2] = (calc3==1 ? fB : 0); 
				image[cns][imsx +k1][imsx + sx - k2-1] = (calc4==1 ? f1B : 0); 

				image[cns][imsx +sx-k2-1][imsx + k1] = (calc5==1 ? f_ : 0); 
				image[cns][imsx +k2][imsx + k1] = (calc6==1 ? fB_ : 0); 

				image[cns][imsx +sx-k2-1][imsx + sx - k1-1] = (calc7==1 ? f1_ : 0); 
				image[cns][imsx +k2][imsx + sx - k1-1] = (calc8==1 ? f1B_ : 0); 			
			}  
		}
	}	

	return Succeeded::yes;
} 



//Second derivative estimation (see Numerical Recipes)			
inline void SRT2DReconstruction::spline(float x[],float y[],int n, float y2[]) {
	int i, k;
	float p, qn, sig, un;
	float u[n];
	float yp1, ypn; 

	yp1 = 0; 
	ypn = 0;

	y2[0]=-0.5;      
	u[0]=(3.0/(x[1]-x[0]))*((y[1]-y[0])/(x[1]-x[0])-yp1);
	
	for(i=1; i<n-1; i++) {
		sig=(x[i]-x[i-1])/(x[i+1]-x[i-1]);
		p=sig*y2[i-1]+2.0;
		y2[i]=(sig-1.0)/p;
		u[i]=(6.0*((y[i+1]-y[i])/(x[i+1]-x[i])-(y[i]-y[i-1])/(x[i]-x[i-1]))/(x[i+1]-x[i-1])-sig*u[i-1])/p;
	}

	qn=0.5; 
    un=(3.0/(x[n-1]-x[n-2]))*(ypn-(y[n-1]-y[n-2])/(x[n-1]-x[n-2]));

	y2[n-1]=(un-qn*u[n-2])/(qn*y2[n-2]+1.0);
	for(k=n-2; k>=0; k--)
		y2[k]=y2[k]*y2[k+1]+u[k];
	return;
}
			
END_NAMESPACE_STIR

