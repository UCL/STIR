//
// $Id$
//
/*!

  \file
  \ingroup recon_buildblock

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
    Copyright (C) 2000- $Date$, IRSL
    See STIR/LICENSE.txt for details
*/
/*
  Modification history

  KT 25/11/2003 :
  - passed normalisation factor such that we don't have to post-multiply
  - convert Siddon argument to a template argument. This allows the compiler
    to resolve all if (Siddon==...) statements at compiler time, and hence
    speeds up the code.
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
template <int Siddon>
void ForwardProjectorByBinUsingRayTracing::
proj_Siddon(Array <4,float> & Projptr, const VoxelsOnCartesianGrid<float> &Bild, 
	    const ProjDataInfoCylindrical* proj_data_info_ptr, 
	    const float cphi, const float sphi, const float delta, const 
            float s_in_mm, 
	    const float R, const int rmin, const int rmax, const float offset, 
	    const int num_planes_per_axial_pos,
	    const float axial_pos_to_z_offset,
	    const float norm_factor)
{
  float           a, a0;
  int             MAXDIM2, X, Y, Z, Q, Zdup, Qdup, k, ix, iy, iz, kmax, ring0;

  /*
   * Siddon == 1 => Phiplus90_r0ab 
   * Siddon == 2 => Phiplus90s0_r0ab
   * Siddon == 3 => Symall_r0ab 
   * Siddon == 4 => s0_r0ab
   */
  //CL&KT 21/12/99 changed ring_unit to num_planes_per_axial_pos
 
  int             xdim = Bild.get_x_size(); //CL 100298  scanner->num_bins;
  // KT 19/05/2000 use other voxel sizes instead of relying on nrings
  int             ydim = Bild.get_y_size();
  int             zdim = Bild.get_z_size();
  float           d_xy = Bild.get_voxel_size().x();
  float           d_sl = Bild.get_voxel_size().z();

  // CL&KT 21/12/99 new
  // in our current coordinate system, the following constant is always 2
  const int num_planes_per_physical_ring = 2;
  assert(fabs(Bild.get_voxel_size().z() * num_planes_per_physical_ring/ proj_data_info_ptr->get_ring_spacing() -1) < 10E-4);


  int maxplane = Bild.get_z_size()-1; // CL 180298 Change the assignement //nrings + nrings - 2;
  MAXDIM2 = (xdim > ydim) ? xdim + 2 : ydim + 2;
  if (zdim + 2 > MAXDIM2)
    MAXDIM2 = zdim + 2;

  VectorWithOffset<float> ax(MAXDIM2);
  VectorWithOffset<float> ay(MAXDIM2);
  VectorWithOffset<float> az(MAXDIM2);
  const float s = s_in_mm / d_xy;  


  // compute intersections with scanner cylinder

  // KT&CL 30/01/98 simplify formulas by not handling s=0 seperately
      // {
  const  float R2pix = R*R/d_xy/d_xy;
  const float TMP = sqrt(R2pix - s * s);
//  }
  const float x2 = s * cphi - sphi * TMP;
  const float x1 = s * cphi + sphi * TMP;
  const float y2 = s * sphi + cphi * TMP;
  const float y1 = s * sphi - cphi * TMP;

  // CL&KT 21/12/99 added axial_pos_to_z_offset
  const float z1 = num_planes_per_axial_pos * (rmin + offset) + axial_pos_to_z_offset;//SPAN
  // CL&KT 21/12/99 introduced num_planes_per_physical_ring
  const float z2 = z1 + num_planes_per_physical_ring * delta;


  // { KT 25/11/2003 no longer local as we need d12 below
    const float dx12 = x1 - x2;
    const float dy12 = y2 - y1;
    const float dz12 = z2 - z1;
#ifdef NEWSCALE
    /* KT 16/02/98 removed division by d_xy to get line integrals in mm
       (or whatever units that bin_size et al are defined)
       */
    // d12 is distance between the 2 points
    const float d12 = sqrt((dx12 * dx12 + dy12 * dy12) * d_xy * d_xy +
	     dz12 * dz12 * d_sl * d_sl) * norm_factor;
#else
    // d12 is distance between the 2 points
    const float d12 = sqrt((dx12 * dx12 + dy12 * dy12) * d_xy * d_xy +
	     dz12 * dz12 * d_sl * d_sl) / d_xy * norm_factor;
#endif
    // KT 16/02/98 multiply with d12 to get normalisation right from the start
    const float Iddx12 = (x1 == x2) ? 1000000.F*d12 : d12 / dx12;
    const float Iddy12 = (y1 == y2) ? 1000000.F*d12 : d12 / dy12;
    const float Iddz12 = (z1 == z2) ? 1000000.F*d12 : d12 / dz12;
    // }

   

  /* Intersection points of LOR and image FOV */
  /* We compute here X1f, Y1f, Z1f, X2f, Y2f, Z2f in voxelcoordinates. */


  // KT 20/06/2001 change calculation of FOV such that even sized images will work
  const float fovrad_in_mm   = 
    min((min(Bild.get_max_x(), -Bild.get_min_x())-1)*Bild.get_voxel_size().x(),
	(min(Bild.get_max_y(), -Bild.get_min_y())-1)*Bild.get_voxel_size().y()); 
  const float fovrad = fovrad_in_mm/d_xy;
  //const int fovradold  = (int) (xdim / 2) - 1;
  //if (Bild.get_x_size()%2==1 && fovradold!=fovrad)
  //  warning("fovrad old %d, new %g\n", fovradold, fovrad);
  const float TMP2 = sqrt(fovrad * fovrad - s * s);
  const float X2f = s * cphi - sphi * TMP2;
  const float X1f = s * cphi + sphi * TMP2;
  const float Y2f = s * sphi + cphi * TMP2;
  const float Y1f = s * sphi - cphi * TMP2;
  // CL&KT 21/12/99 added axial_pos_to_z_offset and num_planes_per_physical_ring
  const float Z1f = num_planes_per_axial_pos * (rmin + offset) 
        + num_planes_per_physical_ring * delta/2 * (1 - TMP2 / TMP) 
	+ axial_pos_to_z_offset;//SPAN
  const float Z2f = num_planes_per_axial_pos * (rmin + offset)
        + num_planes_per_physical_ring * delta/2 * (1 + TMP2 / TMP) 
	+ axial_pos_to_z_offset;//SPAN


  /* intersection points with  intra-voxel planes : */
  const float xmin = (int) (floor(X1f + 0.5)) + 0.5;
  const float ymin = (int) (floor(Y1f - 0.5)) + 0.5;
  const float zmin = (int) (floor(Z1f - 0.5)) + 0.5;
  const float xmax = (int) (floor(X2f - 0.5)) + 0.5;
  const float ymax = (int) (floor(Y2f + 0.5)) + 0.5;
  const float zmax = (int) (floor(Z2f + 0.5)) + 0.5;


  /* intersection point of the LOR with the previous xy-plane  (z smaller) : */
  az[0] = (z1 == z2) ? -1. : (zmin - z1) * Iddz12;
  /* intersection point of the LOR with the previous yz-plane (x smaller) : */
  ax[0] = (x1 == x2) ? -1 : (x1 - xmin) * Iddx12;
  /* intersection point of the LOR with the previous xz-plane (y bigger) : */
  ay[0] = (y1 == y2) ? -1 : (ymin - y1) * Iddy12;


  /* The biggest a?[0] value gives the start of the alpha-row */
  if (az[0] > ax[0])
    if (az[0] > ay[0])	/* LOR enters via xy-plane, z=zmin */
      a0 = az[0];
    else		/* LOR enters via xz-plane, y=ymin */
      a0 = ay[0];
  else if (ax[0] > ay[0])/* LOR enters via yz-plane, x=xmin */
    a0 = ax[0];
  else			/* LOR enters via xz-plane, y=ymin */
    a0 = ay[0];

  /* The smallest a?[0] value, gives the end of the alpha-row */
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
     a? might turn out be a tiny bit smaller then a?_end_actual. So, we set aend a tiny bit 
     smaller than aend_actual.
  */
#if 1
  const float axend = (x2 == x1) ? 1000000.F : (x1 - xmax) * Iddx12/*.9999F*/;
  const float ayend = (y2 == y1) ? 1000000.F : (ymax - y1) * Iddy12/*.9999F*/;
  const float azend = (z2 == z1) ? 1000000.F : (zmax - z1) * Iddz12/*.9999F*/;
#else
  const float axend = (x2 == x1) ? d12*1000000.F : (x1 - xmax) * Iddx12/*.9999F*/;
  const float ayend = (y2 == y1) ? d12*1000000.F : (ymax - y1) * Iddy12/*.9999F*/;
  const float azend = (z2 == z1) ? d12*1000000.F : (zmax - z1) * Iddz12/*.9999F*/;
#endif
#if 0
  const float amax = min(axend, min(ayend, azend));
# if 0
  const float newaxend = (x2 == x1) ? d12*1000000.F : (x1 - xmax) * Iddx12/*.9999F*/;
  const float newayend = (y2 == y1) ? d12*1000000.F : (ymax - y1) * Iddy12/*.9999F*/;
  const float newazend = (z2 == z1) ? d12*1000000.F : (zmax - z1) * Iddz12/*.9999F*/;
  const float newamax = min(newaxend, min(newayend, newazend));
  if (amax != newamax)
    cerr << '\n' << amax << ' ' << newamax << ' ' << newamax-amax;
# endif
#else
  float amax;
  if (azend < axend)
    if (azend < ayend)
      amax = azend;
    else
      amax = ayend;
  else if (axend < ayend)
    amax = axend;
  else
    amax = ayend;
# if 0
  const float newamax = min(axend, min(ayend, azend));
  if (amax != newamax)
    cerr << '\n' << amax << ' ' << newamax << ' ' << newamax-amax;
# endif  
#endif
  /* Computation of ax[], ay[] und az[] : */
  k = 1;
  kmax =(int) (1 + xmin - xmax);
  while (k <= kmax) {
    ax[k] = ax[k - 1] + Iddx12;
    k++;
  }
  k = 1;
  kmax =(int) (1 + ymax - ymin);
  while (k <= kmax) {
    ay[k] = ay[k - 1] + Iddy12;
    k++;
  }
  k = 1;
  kmax = (int) (1 + zmax - zmin);
  while (k <= kmax) {
    az[k] = az[k - 1] + Iddz12;
    k++;
  }

  /* Coordinates of the first Voxel : */
  X = (int) (xmin - 0.5);
  Y = (int) (ymin + 0.5);
  Z = (int) (zmin + 0.5);

  // Find symmetric value in Z by 'mirroring' it around the centre z of the LOR:
  // Z+Q = 2*centre_of_LOR_in_image_coordinates
  // original  Q = (int) (4 * (rmin + offset) + 2 * delta - Z);
  // CL&KT 21/12/99 added axial_pos_to_z_offset and num_planes_per_physical_ring
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
  ix = iy = iz = 1;
  a = a0;
  while (a < amax) {
    if (ax[ix] < ay[iy])
      if (ax[ix] < az[iz]) {	/* LOR leaves voxel through yz-plane */
	const float d = ax[ix] - a;
	Zdup = Z;
	Qdup = Q;
        //CL&KT 21/12/99 used num_planes_per_axial_pos
        for (ring0 = rmin; 
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
	a = ax[ix++];
	X--;
      } else {	/* LOR leaves voxel through xy-plane */
	const float d = az[iz] - a;
	Zdup = Z;
	Qdup = Q;
        //CL&KT 21/12/99 used num_planes_per_axial_pos
        for (ring0 = rmin; 
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
	a = az[iz++];
	Z++;
	Q--;
      }
    else if (ay[iy] < az[iz]) {	/* LOR leaves voxel through xz-plane */
      const float d = ay[iy] - a;
      Zdup = Z;
      Qdup = Q;
      //CL&KT 21/12/99 used num_planes_per_axial_pos
      for (ring0 = rmin; 
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
      a = ay[iy++];
      Y++;
    } else {/* LOR leaves voxel through xy-plane */
      const float d = az[iz] - a;
      Zdup = Z;
      Qdup = Q;
        //CL&KT 21/12/99 used num_planes_per_axial_pos
        for (ring0 = rmin; 
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
      a = az[iz++];
      Z++;
      Q--;
    }

  }		/* Ende while (a<amax) */

	
}


//**************** instantiations

template 
void ForwardProjectorByBinUsingRayTracing::
proj_Siddon<1>(Array<4,float> &Projptr, const VoxelsOnCartesianGrid<float> &, 
	       const ProjDataInfoCylindrical* proj_data_info_ptr, 
	       const float cphi, const float sphi, const float delta, 
	       const float s_in_mm, 
	       const float R, const int min_ax_pos_num, const int max_ax_pos_num, const float offset, 
	       const int num_planes_per_axial_pos,
	       const float axial_pos_to_z_offset,
	       const float norm_factor);


template
void ForwardProjectorByBinUsingRayTracing::
proj_Siddon<2>(Array<4,float> &Projptr, const VoxelsOnCartesianGrid<float> &, 
	       const ProjDataInfoCylindrical* proj_data_info_ptr, 
	       const float cphi, const float sphi, const float delta, 
	       const float s_in_mm, 
	       const float R, const int min_ax_pos_num, const int max_ax_pos_num, const float offset, 
	       const int num_planes_per_axial_pos,
	       const float axial_pos_to_z_offset,
	       const float norm_factor);


template
void ForwardProjectorByBinUsingRayTracing::
proj_Siddon<3>(Array<4,float> &Projptr, const VoxelsOnCartesianGrid<float> &, 
	       const ProjDataInfoCylindrical* proj_data_info_ptr, 
	       const float cphi, const float sphi, const float delta, 
	       const float s_in_mm, 
	       const float R, const int min_ax_pos_num, const int max_ax_pos_num, const float offset, 
	       const int num_planes_per_axial_pos,
	       const float axial_pos_to_z_offset,
	       const float norm_factor);


template
void ForwardProjectorByBinUsingRayTracing::
proj_Siddon<4>(Array<4,float> &Projptr, const VoxelsOnCartesianGrid<float> &, 
	       const ProjDataInfoCylindrical* proj_data_info_ptr, 
	       const float cphi, const float sphi, const float delta, 
	       const float s_in_mm, 
	       const float R, const int min_ax_pos_num, const int max_ax_pos_num, const float offset, 
	       const int num_planes_per_axial_pos,
	       const float axial_pos_to_z_offset,
	       const float norm_factor);

END_NAMESPACE_STIR
