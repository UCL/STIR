//
// $Id$
//
/*!
  \file
  \ingroup scatter
  \brief Inline implementations for class SingleScatter

  \author Kris Thielemans
  \author Charalampos Tsoumpas

  $Date$
  $Revision$
*/
/*
    Copyright (C) 2000- $Date$, Hammersmith Imanet Ltd
    See STIR/LICENSE.txt for details
*/




//using namespace std;


START_NAMESPACE_STIR

template<int num_dimensions>
inline 
BasicCoordinate<num_dimensions,float> convert_int_to_float(const BasicCoordinate<num_dimensions,int>& cint)
	{	  
	  BasicCoordinate<num_dimensions,float> cfloat;

	  for(int i=1;i<=num_dimensions;++i)
		  cfloat[i]=(float)cint[i];
	  return cfloat;
	}

inline
float dif_cross_section(const float cos_theta, float energy)
{ 
  const float sin_theta2= 1-cos_theta*cos_theta ;
  const float P= 1/(1+(energy/511.0)*(1-cos_theta));
  return(  Re*Re/2*P* (1 - P*sin_theta2 + P*P) );
}
inline
float dif_cross_section(const CartesianCoordinate3D<float>& scatter_point,
	  const CartesianCoordinate3D<float>& detector_coord_A,
	  const CartesianCoordinate3D<float>& detector_coord_B, float energy)
{
  /* the scatter angle theta  should be 0 when the 2 detectors 
     are at opposite sides of the scatter point. So, we need
	 to set cos_theta equal to the following (with a minus sign!)
	 */
  return
	  dif_cross_section(-cos_angle(detector_coord_A - scatter_point,
		                                  detector_coord_B - scatter_point), energy);
}
inline
float
energy_after_scatter(const float cos_theta, const float energy)
{
   return energy/(1+(energy/511.0)*(1-cos_theta));
}

inline
float
energy_after_scatter_511keV(const float cos_theta)
{
   return 511/(2-cos_theta);
}
inline
float dif_cross_section_511keV(const float cos_theta)
{
  const float sin_theta2= 1-cos_theta*cos_theta ;
  const float P= 1/(2-cos_theta);
  return(  Re*Re/2*P* (1 - P*sin_theta2 + P*P) );
}

inline
float dif_cross_section_511keV(const CartesianCoordinate3D<float>& scatter_point,
	  const CartesianCoordinate3D<float>& detector_coord_A,
	  const CartesianCoordinate3D<float>& detector_coord_B)
{
  /* the scatter angle theta  should be 0 when the 2 detectors 
     are at opposite sides of the scatter point. So, we need
	 to set cos_theta equal to the following (with a minus sign!)
	 */
  return
	  dif_cross_section_511keV(-cos_angle(detector_coord_A - scatter_point,
		                                  detector_coord_B - scatter_point)); 
}

inline
float total_cross_section(float energy)
{
  const float a= energy/511.0;
  const float l= log(1.0+2.0*a);
  const double sigma0= 6.65E-25;

  return( 0.75*sigma0  * ( (1.0+a)/(a*a)*( 2.0*(1.0+a)/(1.0+2.0*a)- l/a ) + l/(2.0*a) - (1.0+3.0*a)/(1.0+2.0*a)/(1.0+2.0*a) ) );
}

/*!	\ingroup scatter
   \brief detection efficiency for a given energy window
   
   This function provides a simple model of the interaction of radiation
	with the detector .
	
	\param low, high Discriminator bounds of the detector in keV
	\param energy	 Energy of incident photon in keV
	\param resolution Energy resolution of the detector b(between 0 and 1) at the reference energy
	\param reference_energy  Energy where the FWHM is given by \c resolution
			  
	The energy spectrum is assumed to be Gaussian. The FWHM is assumed to 
	be proportional to sqrt(energy). This is reasonable given the Poisson 
	statistics of scintillation detectors. The proportionality factor is 
	determined by requiring that FWHM(reference_energy)=resolution*reference_energy.

	This formula is the same as the one used by SIMSET for Simple_PET detector.
*/
inline 
float detection_efficiency( const float low, const float high, 
						    const float energy, 
							const float reference_energy, const float resolution )
{
	if (energy<=low || energy >= high)
	   return 0;
	// factor 2.35 is used to convert FWHM to sigma
	const float sigma_times_sqrt2= 
		sqrt(2.*energy*reference_energy)*resolution/2.35;
	
	return
		0.5*( erf((high-energy)/sigma_times_sqrt2) 
		      - erf((low-energy)/sigma_times_sqrt2 ));	
}

/*!	\ingroup scatter
   \brief detection efficiency for BGO  for a given energy window
   */
inline 
float detection_efficiency_BGO( const float low, const float high, 
						        const float energy)
{
	return
		detection_efficiency(low, high, energy, 511, .25);
}

END_NAMESPACE_STIR

