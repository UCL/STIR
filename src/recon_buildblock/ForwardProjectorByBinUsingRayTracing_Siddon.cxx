//
// $Id$
//
/*!

  \file
  \ingroup projection

  \brief implementation of Siddon's algorithm

  \author Claire Labbe
  \author Kris Thielemans
  \author (based originally on C code by Matthias Egger)

  \author PARAPET project

  $Date$
  $Revision$
*/
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- $Date$, Hammersmith Imanet Ltd
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
/*
  Modification history

  KT 25/11/2003 :
  - passed normalisation factor such that we don't have to post-multiply
  - convert Siddon argument to a template argument. This allows the compiler
    to resolve all if (Siddon==...) statements at compiler time, and hence
    speeds up the code.
  KT 1/12/2003 :
  WARNING: somewhat different results are generated since this version
  (see below)
  - almost completely reimplemented in terms of what I did for
   ProjMatrixByBinUsingRayTracing and RayTraceVoxelsOnCartesianGrid.
   This version is now stable under numerical rounding error, and
   gives the same results as the projection matrix version.
   Unfortunately, this means it gives somewhat DIFFERENT results compared
   to the previous version. The difference sits essentially in the end-voxel
   on the LOR. For most/all applications, this difference should be irrelevant.
  - added option restrict_to_cylindrical_FOV
  - make proj_Siddon return true or false to check if any data has been
  forward projected
*/

#include "stir/recon_buildblock/ForwardProjectorByBinUsingRayTracing.h"
#include "stir/ProjDataInfoCylindrical.h"
#include "stir/VoxelsOnCartesianGrid.h"
#include "stir/round.h"
#include <math.h>
#include <algorithm>
#ifndef STIR_NO_NAMESPACE
using std::min;
using std::max;
#endif

START_NAMESPACE_STIR
template <typename T>
static inline int sign(const T& t) 
{
  return t<0 ? -1 : 1;
}

/*!
  This function uses a 3D version of Siddon's algorithm for forward projecting.
  See M. Egger's thesis for details.
  In addition to the symmetries is segment and view, it also uses s,-s symmetry 
  (while being careful when s=0 to avoid self-symmetric cases)

  For historical reasons, 'axial_pos_num' is here called 'ring'. rmin,rmax are
  min_axial_pos_num and max_axial_pos_num.

  See RayTraceVoxelsOnCartesianGrid for a shorter and clearer version of the Siddon algorithm.
  */


// KT 20/06/2001 should now work for non-arccorrected data as well, pass s_in_mm
#ifndef STIR_SIDDON_NO_TEMPLATE
template <int Siddon>
#endif
bool
ForwardProjectorByBinUsingRayTracing::
proj_Siddon(
#ifdef STIR_SIDDON_NO_TEMPLATE
            int Siddon,
#endif
            Array <4,float> & Projptr, const VoxelsOnCartesianGrid<float> &Bild, 
	    const ProjDataInfoCylindrical* proj_data_info_ptr, 
	    const float cphi, const float sphi, const float delta, const 
            float s_in_mm, 
	    const float R, const int rmin, const int rmax, const float offset, 
	    const int num_planes_per_axial_pos,
	    const float axial_pos_to_z_offset,
	    const float norm_factor,
	    const bool restrict_to_cylindrical_FOV)
{
  /*
   * Siddon == 1 => Phiplus90_r0ab 
   * Siddon == 2 => Phiplus90s0_r0ab
   * Siddon == 3 => Symall_r0ab 
   * Siddon == 4 => s0_r0ab
   */

  /* TODO: check in implementation:
     why first voxel +=inc_? in all directions
  */
  /* See RayTraceVoxelsOnCartesianGrid() for a clearer implementation of the
     Siddon algorithm. This doesn't handle the symmetries, so is a lot shorter.

     For a comparison, remember that the current function relies on symmetries
     having been applied first, so the end_point-start_point coordinates
     always have particular signs. In the notation of RayTraceVoxelsOnCartesianGrid():
  */
  const int sign_x=-1;
  const int sign_y=+1;
  const int sign_z=+1;

  // in our current coordinate system, the following constant is always 2
  const int num_planes_per_physical_ring = 2;
  assert(fabs(Bild.get_voxel_size().z() * num_planes_per_physical_ring/ proj_data_info_ptr->get_ring_spacing() -1) < 10E-4);



#ifdef NEWSCALE
  /* KT 16/02/98 removed division by voxel_size.x() to get line integrals in mm
     (or whatever units that voxel_size() is defined)
  */
  const float normalisation_constant = norm_factor;
#else
  const float normalisation_constant = norm_factor / Bild.get_voxel_size().x();
#endif

  // KT 1/12/2003 no longer subtract -1 as determination of first and last voxel is now
  // no longer sensitive to rounding error
  const float fovrad_in_mm   = 
    min((min(Bild.get_max_x(), -Bild.get_min_x()))*Bild.get_voxel_size().x(),
	(min(Bild.get_max_y(), -Bild.get_min_y()))*Bild.get_voxel_size().y()); 

  const CartesianCoordinate3D<float>& voxel_size = Bild.get_voxel_size();

  CartesianCoordinate3D<float> start_point;  
  CartesianCoordinate3D<float> stop_point;
  {
    /* parametrisation of LOR is
         X= s*cphi + a*sphi, 
         Y= s*sphi - a*cphi, 
         // Z= t/costheta+offset_in_z - a*tantheta
         Z= num_planes_per_axial_pos * (rmin + offset) + axial_pos_to_z_offset +
           + num_planes_per_physical_ring * delta/2 * (1 - a / TMP);
	 with TMP = sqrt(R*R - square(s_in_mm));
	 The Z parametrisation can be understood by noting that at a=TMP, the LOR
	 intersects the detector cylinder with radius R.

       find now min_a, max_a such that end-points intersect border of FOV 
    */
    float max_a;
    float min_a;
    
    if (restrict_to_cylindrical_FOV)
    {
      if (fabs(s_in_mm) >= fovrad_in_mm) 
	return false;
      // a has to be such that X^2+Y^2 == fovrad^2      
      max_a = sqrt(square(fovrad_in_mm) - square(s_in_mm));
      min_a = -max_a;
    } // restrict_to_cylindrical_FOV
    else
    {
      // use FOV which is square.
      // note that we use square and not rectangular as otherwise symmetries
      // would take us out of the FOV. TODO
      /*
        a has to be such that 
        |X| <= fovrad_in_mm &&  |Y| <= fovrad_in_mm
      */
      if (fabs(cphi) < 1.E-3 || fabs(sphi) < 1.E-3) 
      {
        if (fovrad_in_mm < fabs(s_in_mm))
          return false;
        max_a = fovrad_in_mm;
        min_a = -fovrad_in_mm;
      }
      else
      {
        max_a = min((fovrad_in_mm*sign(sphi) - s_in_mm*cphi)/sphi,
                    (fovrad_in_mm*sign(cphi) + s_in_mm*sphi)/cphi);
        min_a = max((-fovrad_in_mm*sign(sphi) - s_in_mm*cphi)/sphi,
                    (-fovrad_in_mm*sign(cphi) + s_in_mm*sphi)/cphi);
        if (min_a > max_a - 1.E-3*voxel_size.x())
          return false;
      }
      
    } //!restrict_to_cylindrical_FOV
    const float TMP = sqrt(R*R - square(s_in_mm));
    
    start_point.x() = (s_in_mm*cphi + max_a*sphi)/voxel_size.x();
    start_point.y() = (s_in_mm*sphi - max_a*cphi)/voxel_size.y(); 
    //start_point.z() = (t_in_mm/costheta+offset_in_z - max_a*tantheta)/voxel_size.z();
    start_point.z() = num_planes_per_axial_pos * (rmin + offset) + axial_pos_to_z_offset +
      + num_planes_per_physical_ring * delta/2 * (1 - max_a / TMP);

    stop_point.x() = (s_in_mm*cphi + min_a*sphi)/voxel_size.x();
    stop_point.y() = (s_in_mm*sphi - min_a*cphi)/voxel_size.y(); 
    //stop_point.z() = (t_in_mm/costheta+offset_in_z - min_a*tantheta)/voxel_size.z();
    stop_point.z() =  num_planes_per_axial_pos * (rmin + offset) + axial_pos_to_z_offset +
      + num_planes_per_physical_ring * delta/2 * (1 - min_a / TMP);
  }
  // find voxel which contains the start_point, and go to its 'left' edge
  const float xmin = round(start_point.x()) - sign_x*0.5F;
  const float ymin = round(start_point.y()) - sign_y*0.5F;
  const float zmin = round(start_point.z()) - sign_z*0.5F;
  // find voxel which contains the end_point, and go to its 'right' edge
  const float xmax = round(stop_point.x()) + sign_x*0.5F;
  const float ymax = round(stop_point.y()) + sign_y*0.5F;  
  const float zmax = round(stop_point.z()) + sign_z*0.5F;

  const CartesianCoordinate3D<float> difference = stop_point-start_point;
  assert(difference.x() <= .00001F);
  assert(difference.y() >= -.00001F);
  assert(difference.z() >= -.00001F);

  const float small_difference = 1.E-5F;
  const bool zero_diff_in_x = fabs(difference.x())<=small_difference;
  const bool zero_diff_in_y = fabs(difference.y())<=small_difference;
  const bool zero_diff_in_z = fabs(difference.z())<=small_difference;

  // d12 is distance between the 2 points (times normalisation_const)
  const float d12 = 
    static_cast<float>(norm(difference*Bild.get_voxel_size()) * normalisation_constant);
    // KT 16/02/98 multiply with d12 to get normalisation right from the start
  const float inc_x = (zero_diff_in_x) ? 1000000.F*d12 : d12 / (sign_x*difference.x());
  const float inc_y = (zero_diff_in_y) ? 1000000.F*d12 : d12 / (sign_y*difference.y());
  const float inc_z = (zero_diff_in_z) ? 1000000.F*d12 : d12 / (sign_z*difference.z());


  /* Find the a? values of the intersection points of the LOR with the planes between voxels.
     Note on special handling of rays parallel to one of the planes:
     
     The corresponding a? value would be -infinity. We just set it to
     a value low enough such that the start value of 'a' is not compromised 
     further on.
     Normally
       a? = (?min-start_point.?) * inc_? * sign_?
     Because the start voxel includes the start_point, we have that
       a? <= -inc_?
     As inc_? is set to some large number when the ray is parallel, this is
     a good value for the ray.
  */
  /* intersection point of the LOR with the previous yz-plane : */
  float ax = (zero_diff_in_x) ? -inc_x : (xmin - start_point.x()) * sign_x * inc_x;
  /* intersection point of the LOR with the previous xz-plane : */
  float ay = (zero_diff_in_y) ? -inc_y : (ymin - start_point.y()) * sign_y * inc_y;
  /* The biggest a? value gives the start of the 'a'-row */
  /* intersection point of the LOR with the previous xy-plane : */
  float az = (zero_diff_in_z) ? -inc_z : (zmin - start_point.z()) * sign_z * inc_z;
  float a = max(ax, max(ay, az));

  /* The smallest a? value, gives the end of the 'a'-row */
  /* (Note: doc is copied from RayTraceVoxelsOnCartesianGrid. Would need to be adapted a bit)

     Find a?end for the last intersections with the coordinate planes. 
     amax will then be the smallest of all these a?end.

     If the LOR is parallel to a plane, take care that its a?end is larger than all the others.
     Note that axend <= d12 (difference.x()+1)/difference.x()

     In fact, we will take a?end slightly smaller than the actual last value (i.e. we multiply
     with a factor .9999). This is to avoid rounding errors in the loop below. In this loop,
     we try to detect the end of the LOR by comparing a (which is either ax,ay or az) with
     aend. With exact arithmetic, a? would have been incremented exactly to 
       a?_end_actual = a?start + (?max-?end)*inc_?*sign_?, 
     so we could loop until a==aend_actual. However, because of numerical precision,
     a? might turn out be a tiny bit smaller then a?_end_actual. So, we set aend a 
     (somewhat less tiny) bit smaller than aend_actual, and loop while (a<aend).
  */
  const float axend = (zero_diff_in_x) ? d12*1000000.F : (xmax - start_point.x()) * sign_x * inc_x*.9999F;
  const float ayend = (zero_diff_in_y) ? d12*1000000.F : (ymax - start_point.y()) * sign_y * inc_y*.9999F;
  const float azend = (zero_diff_in_z) ? d12*1000000.F : (zmax - start_point.z()) * sign_z * inc_z*.9999F;

  const float amax = min(axend, min(ayend, azend));

  /* Coordinates of the first Voxel : */
  int X = round(xmin + sign_x* 0.5F);
  int Y = round(ymin + sign_y* 0.5F);
  int Z = round(zmin + sign_z* 0.5F);

  // Find symmetric value in Z by 'mirroring' it around the centre z of the LOR:
  // Z+Q = 2*centre_of_LOR_in_image_coordinates
  int Q;
  {
    // first compute it as floating point (although it has to be an int really)
    const float Qf = (2*num_planes_per_axial_pos*(rmin + offset) 
                      + num_planes_per_physical_ring*delta
	              + 2*axial_pos_to_z_offset
	              - Z); 
    // now use rounding to be safe
    Q = (int)floor(Qf + 0.5);
    assert(fabs(Q-Qf) < 10E-4);
  }
  
  // KT&CL 30/01/98 simplify initialisation (doing a bit too much work sometimes)
  Projptr.fill(0);

  /*
    for (ring0 = rmin; ring0 <= rmax; ring0++) {
    if (Siddon == 2) {
    for (i = 0; i <= 1; i++) {
    Projptr[ring0][i][0][0] = 0.;
    Projptr[ring0][i][0][2] = 0.;
    }
    } else if (Siddon == 4) {
    for (i = 0; i <= 1; i++)
    for (k = 0; k <= 3; k++)
    Projptr[ring0][i][0][k] = 0.;
    } else if (Siddon == 3) {
    for (i = 0; i <= 1; i++)
    for (j = 0; j <= 1; j++)
    for (k = 0; k <= 3; k++)
    Projptr[ring0][i][j][k] = 0.;
    } else if (Siddon == 1) {
    for (i = 0; i <= 1; i++)
    for (j = 0; j <= 1; j++) {
    Projptr[ring0][i][j][0] = 0.;
    Projptr[ring0][i][j][2] = 0.;
    }
    }
    }
  */
  /* Now we go slowly along the LOR */
  if (zero_diff_in_x) ax = axend; else ax += inc_x;
  if (zero_diff_in_y) ay = ayend; else ay += inc_y;
  if (zero_diff_in_z) az = azend; else az += inc_z;

  // these are used to check boundaries on Z
  // We do not use get_min_index() explicitly as presumably a comparison with 0 
  // is just a tiny bit faster...
  // TODO remove this
  const int maxplane = Bild.get_max_index(); 
  assert(Bild.get_min_index() == 0);

  while (a < amax) {
    if (ax < ay)
      if (ax < az) {	/* LOR leaves voxel through yz-plane */
	const float d = ax - a;
	int Zdup = Z;
	int Qdup = Q;
        for (int ring0 = rmin; 
	     ring0 <= rmax; 
	     ring0++, Zdup += num_planes_per_axial_pos, Qdup +=num_planes_per_axial_pos )
	{
	  /* Alle Symetrien ausser s */
	  if (Zdup >= 0 && Zdup <= maxplane) {
	    Projptr[ring0][0][0][0] += d * Bild[Zdup][Y][X];
	    Projptr[ring0][0][0][2] += d * Bild[Zdup][X][-Y];
	    if ((Siddon == 4) || (Siddon == 3)) {
	      Projptr[ring0][1][0][1] += d * Bild[Zdup][X][Y];
	      Projptr[ring0][1][0][3] += d * Bild[Zdup][Y][-X];
	    }
	    if ((Siddon == 1) || (Siddon == 3)) {
	      Projptr[ring0][1][1][0] += d * Bild[Zdup][-Y][-X];
	      Projptr[ring0][1][1][2] += d * Bild[Zdup][-X][Y];
	    }
	    if (Siddon == 3) {
	      Projptr[ring0][0][1][1] += d * Bild[Zdup][-X][-Y];
	      Projptr[ring0][0][1][3] += d * Bild[Zdup][-Y][X];
	    }
	  }
	  if (Qdup >= 0 && Qdup <= maxplane) {
	    if ((Siddon == 4) || (Siddon == 3)) {
	      Projptr[ring0][0][0][1] += d * Bild[Qdup][X][Y];
	      Projptr[ring0][0][0][3] += d * Bild[Qdup][Y][-X];
	    }
	    if ((Siddon == 1) || (Siddon == 3)) {
	      Projptr[ring0][0][1][0] += d * Bild[Qdup][-Y][-X];
	      Projptr[ring0][0][1][2] += d * Bild[Qdup][-X][Y];
	    }
	    if (Siddon == 3) {
	      Projptr[ring0][1][1][1] += d * Bild[Qdup][-X][-Y];
	      Projptr[ring0][1][1][3] += d * Bild[Qdup][-Y][X];
	    }
	    Projptr[ring0][1][0][0] += d * Bild[Qdup][Y][X];
	    Projptr[ring0][1][0][2] += d * Bild[Qdup][X][-Y];
	  }
						
	}
	a = ax ;  ax +=  inc_x;
	X--;
      } else {	/* LOR leaves voxel through xy-plane */
	const float d = az - a;
	int Zdup = Z;
	int Qdup = Q;
        for (int ring0 = rmin; 
	     ring0 <= rmax; 
	     ring0++, Zdup += num_planes_per_axial_pos, Qdup +=num_planes_per_axial_pos )
	{
	  /* all symmetries except in 's'*/
	  if (Zdup >= 0 && Zdup <= maxplane) {
	    Projptr[ring0][0][0][0] += d * Bild[Zdup][Y][X];
	    Projptr[ring0][0][0][2] += d * Bild[Zdup][X][-Y];
	    if ((Siddon == 4) || (Siddon == 3)) {
	      Projptr[ring0][1][0][1] += d * Bild[Zdup][X][Y];
	      Projptr[ring0][1][0][3] += d * Bild[Zdup][Y][-X];
	    }
	    if ((Siddon == 1) || (Siddon == 3)) {
	      Projptr[ring0][1][1][0] += d * Bild[Zdup][-Y][-X];
	      Projptr[ring0][1][1][2] += d * Bild[Zdup][-X][Y];
	    }
	    if (Siddon == 3) {
	      Projptr[ring0][0][1][1] += d * Bild[Zdup][-X][-Y];
	      Projptr[ring0][0][1][3] += d * Bild[Zdup][-Y][X];

	    }
	  }
	  if (Qdup >= 0 && Qdup <= maxplane) {
	    if ((Siddon == 4) || (Siddon == 3)) {
	      Projptr[ring0][0][0][1] += d * Bild[Qdup][X][Y];
	      Projptr[ring0][0][0][3] += d * Bild[Qdup][Y][-X];
	    }
	    if ((Siddon == 1) || (Siddon == 3)) {
	      Projptr[ring0][0][1][0] += d * Bild[Qdup][-Y][-X];
	      Projptr[ring0][0][1][2] += d * Bild[Qdup][-X][Y];
	    }
	    if (Siddon == 3) {
	      Projptr[ring0][1][1][1] += d * Bild[Qdup][-X][-Y];
	      Projptr[ring0][1][1][3] += d * Bild[Qdup][-Y][X];
	    }
	    Projptr[ring0][1][0][0] += d * Bild[Qdup][Y][X];
	    Projptr[ring0][1][0][2] += d * Bild[Qdup][X][-Y];
	  }
						
	}
	a = az ;  az +=  inc_z;
	Z++;
	Q--;
      }
    else if (ay < az) {	/* LOR leaves voxel through xz-plane */
      const float d = ay - a;
      int Zdup = Z;
      int Qdup = Q;
      for (int ring0 = rmin; 
           ring0 <= rmax; 
	   ring0++, Zdup += num_planes_per_axial_pos, Qdup +=num_planes_per_axial_pos )
	   {
	/*   all symmetries except in 's' */
	if (Zdup >= 0 && Zdup <= maxplane) {
	  Projptr[ring0][0][0][0] += d * Bild[Zdup][Y][X];
	  Projptr[ring0][0][0][2] += d * Bild[Zdup][X][-Y];
	  if ((Siddon == 4) || (Siddon == 3)) {
	    Projptr[ring0][1][0][1] += d * Bild[Zdup][X][Y];
	    Projptr[ring0][1][0][3] += d * Bild[Zdup][Y][-X];
	  }
	  if ((Siddon == 1) || (Siddon == 3)) {
	    Projptr[ring0][1][1][0] += d * Bild[Zdup][-Y][-X];
	    Projptr[ring0][1][1][2] += d * Bild[Zdup][-X][Y];
	  }
	  if (Siddon == 3) {
	    Projptr[ring0][0][1][1] += d * Bild[Zdup][-X][-Y];
	    Projptr[ring0][0][1][3] += d * Bild[Zdup][-Y][X];
	  }
	}
	if (Qdup >= 0 && Qdup <= maxplane) {
	  if ((Siddon == 4) || (Siddon == 3)) {
	    Projptr[ring0][0][0][1] += d * Bild[Qdup][X][Y];
	    Projptr[ring0][0][0][3] += d * Bild[Qdup][Y][-X];
	  }
	  if ((Siddon == 1) || (Siddon == 3)) {
	    Projptr[ring0][0][1][0] += d * Bild[Qdup][-Y][-X];
	    Projptr[ring0][0][1][2] += d * Bild[Qdup][-X][Y];
	  }
	  if (Siddon == 3) {
	    Projptr[ring0][1][1][1] += d * Bild[Qdup][-X][-Y];
	    Projptr[ring0][1][1][3] += d * Bild[Qdup][-Y][X];
	  }
	  Projptr[ring0][1][0][0] += d * Bild[Qdup][Y][X];
	  Projptr[ring0][1][0][2] += d * Bild[Qdup][X][-Y];
	}
					
      }
      a = ay;   ay +=  inc_y;
      Y++;
    } else {/* LOR leaves voxel through xy-plane */
      const float d = az - a;
      int Zdup = Z;
      int Qdup = Q;
        for (int ring0 = rmin; 
	     ring0 <= rmax; 
	     ring0++, Zdup += num_planes_per_axial_pos, Qdup +=num_planes_per_axial_pos )
	{
	/*   all symmetries except in 's' */
	if (Zdup >= 0 && Zdup <= maxplane) {
	  Projptr[ring0][0][0][0] += d * Bild[Zdup][Y][X];
	  Projptr[ring0][0][0][2] += d * Bild[Zdup][X][-Y];
	  if ((Siddon == 4) || (Siddon == 3)) {
	    Projptr[ring0][1][0][1] += d * Bild[Zdup][X][Y];
	    Projptr[ring0][1][0][3] += d * Bild[Zdup][Y][-X];
	  }
	  if ((Siddon == 1) || (Siddon == 3)) {
	    Projptr[ring0][1][1][0] += d * Bild[Zdup][-Y][-X];
	    Projptr[ring0][1][1][2] += d * Bild[Zdup][-X][Y];
	  }
	  if (Siddon == 3) {
	    Projptr[ring0][0][1][1] += d * Bild[Zdup][-X][-Y];
	    Projptr[ring0][0][1][3] += d * Bild[Zdup][-Y][X];
	  }
	}
	if (Qdup >= 0 && Qdup <= maxplane) {
	  if ((Siddon == 4) || (Siddon == 3)) {
	    Projptr[ring0][0][0][1] += d * Bild[Qdup][X][Y];
	    Projptr[ring0][0][0][3] += d * Bild[Qdup][Y][-X];
	  }
	  if ((Siddon == 1) || (Siddon == 3)) {
	    Projptr[ring0][0][1][0] += d * Bild[Qdup][-Y][-X];
	    Projptr[ring0][0][1][2] += d * Bild[Qdup][-X][Y];
	  }
	  if (Siddon == 3) {
	    Projptr[ring0][1][1][1] += d * Bild[Qdup][-X][-Y];
	    Projptr[ring0][1][1][3] += d * Bild[Qdup][-Y][X];
	  }
	  Projptr[ring0][1][0][0] += d * Bild[Qdup][Y][X];
	  Projptr[ring0][1][0][2] += d * Bild[Qdup][X][-Y];
	}
					
      }
      a = az ;  az +=  inc_z;
      Z++;
      Q--;
    }

  }		/* Ende while (a<amax) */

  return true;
}


//**************** instantiations

#if !defined(STIR_SIDDON_NO_TEMPLATE)
// this is the normal code, but it doesn't compile with VC 6.0
template 
bool
ForwardProjectorByBinUsingRayTracing::
proj_Siddon<1>(Array<4,float> &Projptr, const VoxelsOnCartesianGrid<float> &, 
	       const ProjDataInfoCylindrical* proj_data_info_ptr, 
	       const float cphi, const float sphi, const float delta, 
	       const float s_in_mm, 
	       const float R, const int min_ax_pos_num, const int max_ax_pos_num, const float offset, 
	       const int num_planes_per_axial_pos,
	       const float axial_pos_to_z_offset,
	       const float norm_factor,
	       const bool restrict_to_cylindrical_FOV);


template
bool
ForwardProjectorByBinUsingRayTracing::
proj_Siddon<2>(Array<4,float> &Projptr, const VoxelsOnCartesianGrid<float> &, 
	       const ProjDataInfoCylindrical* proj_data_info_ptr, 
	       const float cphi, const float sphi, const float delta, 
	       const float s_in_mm, 
	       const float R, const int min_ax_pos_num, const int max_ax_pos_num, const float offset, 
	       const int num_planes_per_axial_pos,
	       const float axial_pos_to_z_offset,
	       const float norm_factor,
	       const bool restrict_to_cylindrical_FOV);


template
bool
ForwardProjectorByBinUsingRayTracing::
proj_Siddon<3>(Array<4,float> &Projptr, const VoxelsOnCartesianGrid<float> &, 
	       const ProjDataInfoCylindrical* proj_data_info_ptr, 
	       const float cphi, const float sphi, const float delta, 
	       const float s_in_mm, 
	       const float R, const int min_ax_pos_num, const int max_ax_pos_num, const float offset, 
	       const int num_planes_per_axial_pos,
	       const float axial_pos_to_z_offset,
	       const float norm_factor,
	       const bool restrict_to_cylindrical_FOV);


template
bool
ForwardProjectorByBinUsingRayTracing::
proj_Siddon<4>(Array<4,float> &Projptr, const VoxelsOnCartesianGrid<float> &, 
	       const ProjDataInfoCylindrical* proj_data_info_ptr, 
	       const float cphi, const float sphi, const float delta, 
	       const float s_in_mm, 
	       const float R, const int min_ax_pos_num, const int max_ax_pos_num, const float offset, 
	       const int num_planes_per_axial_pos,
	       const float axial_pos_to_z_offset,
	       const float norm_factor,
	       const bool restrict_to_cylindrical_FOV);

#endif 
END_NAMESPACE_STIR
