//
//
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- 2008, Hammersmith Imanet Ltd
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
/*!

  \file
  \ingroup projection

  \brief This file defines two private static functions from
  stir::BackProjectorByBinUsingInterpolation.

  \warning This file is #included by BackProjectorByBinUsingInterpolation_linear.cxx 
  and BackProjectorByBinUsingInterpolation_piecewise_linear.cxx. It should NOT be added
  to a Makefile (or project).


  \author Kris Thielemans
  \author Claire Labbe
  \author (based on C version by Matthias Egger)
  \author PARAPET project

*/

// enable this variable if you need to handle very oblique LORs
#define MOREZ
//#define ALTERNATIVE
/* 

  These functions use a 3D version of Cho's algorithm for backprojecting 
  incrementally.
  See M. Egger's thesis for details.

  They use the following symmetries:
  - opposite ring difference
  - views : view, view+90 degrees, 180 degrees - view, 90 degrees - view
  - s,-s symmetry (while being careful when s==0 to avoid self-symmetric 
    cases)

  For historical reasons, 'axial_pos_num' is here called 'ring'.

  Note: encoding used in Proj2424
  Proj2424[n][f][ms][b]
   n  : 1 : negative delta; 0 : positive delta
   f  : 0 : view; 1 : pi/2-view; 2 : pi/2+view; 3: pi-view
   ms : 1 : negative s;     0 : positive s
   b  : 1 : r1,s; 2: r2,s+1; 3: r2,s 4: r1,s+1 
            where r1=ring0, r2=ring0+1 when
	       (n==ms && f==0||2) || (n!=ms && f==1||3)
	    and the role of r1,r2 are reversed in the other cases

  Naming conventions for the incremental variables:
  n==0 && ms==0: A'f'
  n==0 && ms==1: P'f'
  n==1 && ms==0: A'f'n
  n==1 && ms==1: P'f'n

  Symmetries:
  s-> -s          changes dz->1-dz, ds->-ds
  view->pi/2+view changes dz->dz, ds->ds
  view->pi/2-view changes dz->1-dz, ds->ds
  delta->-delta   changes dz->1-dz, ds->ds
  */


#include "stir/ProjDataInfo.h"
#include "stir/VoxelsOnCartesianGrid.h"
#include "stir/ProjDataInfoCylindricalArcCorr.h"
#include "stir/VoxelsOnCartesianGrid.h"
#include "stir/recon_buildblock/BackProjectorByBinUsingInterpolation.h"
#include "stir/round.h"
#include "stir/Succeeded.h"
#include <math.h>
/*
  KT 22/05/98 drastic revision

cosmetic changes:
  - removed lots of local variables which were not used (or are not used 
    anymore)
  - moved most declarations of variables to the place where they are 
    initialised, and add const when possible
  - added grouping {} to make variables local 
  - translated German and Afrikaans comments and added new comments

  - reordered the main loop to avoid repeating the 'voxel-updating' code
  - moved the initialisation code (common to all symmetry cases) into the 
    find_start_values() function

bugs removed:
  - handle self-symmetric case at s==0 properly, this was only correct in
    the 45 degrees case. This removed most of the bright spot in the 
    central pixel.

  - changed the condition to break out of the loop to avoid early stopping, 
    which gave artifacts at the border

  - attempt to take into account that calculations are done with finite precision
    There used to be 'random' misplacements of data into neighboring voxels,
    this happened in the central pixel and at 45 and 135 degrees.
    * use double to prevent rounding errors, necessary for incremental 
      quantities
    * change all comparisons with 0 to comparisons with a small number 
     'epsilon'

increased precision:
  - use double for invariants as well
    (otherwise computations of incremental constants would mean promoting 
    floats to doubles all the time)

speed-up
  - all these changes resulted in an (unexpected) speed-up of about a factor 2
  
  KT&CL 22/12/99 
span works now by introducing offsets
 
  KT&SM 
introduced MOREZ to handle very oblique segments. This handles the case where Z
needs to be incremented more than once to find the next voxel in the beam

  KT 25/09/2001
make sure things are ok for any image size. The main change is
to let find_start_values return Succeeded::no if there's no voxel in 
the FOV for this partiuclar tube. If so, we'll just exit.

TODO: 
solve remaining issues of rounding errors (see backproj2D)

if we never parallelise 'beyond' viewgram level (that is, always execute
the calling back_project() and these functions at the same processor),
there is probably no point in using the Projptr parameter, and we could
use the viewgrams directly. 
Indeed, each element in Projptr is accessed 2 (sometimes 3) times only. The 
compiler probably optimises this to 1 (sometimes 2) accesses only. 
So, it could be more expensive to construct the Projptr and fill it in from the
viewgrams, compared to use the viewgrams directly.
*/

START_NAMESPACE_STIR

static const double epsilon = 1e-10;

/* 
   This declares a local function that finds the first voxel in the beam, and 
   associated ds,dz etc.
   It is implemented at the end of the file.

   image_rad: half the image size (in pixels)
   d_sl  : z-voxel size (in mm)

   It returns Succeeded::no when it couldn't find any voxel in the beam.
   */

// KT 25/09/2001 changed return value
static Succeeded find_start_values(const ProjDataInfoCylindricalArcCorr* proj_data_info_ptr, 
                              const float delta, const double cphi, const double sphi, 
                              const int s, const int ring0,
                              const float image_rad,
			      const double d_sl,
                              int&X1, int&Y1, int& Z1,
                              double& ds, double& dz, double& dzhor, double& dzvert,
                              const float num_planes_per_axial_pos,
			      const float axial_pos_to_z_offset);


inline void 
check_values(const ProjDataInfoCylindricalArcCorr* proj_data_info_ptr, 
	     const float delta, const double cphi, const double sphi, 
	     const int s, const int ring0,
	     const int X1, const int Y1, const int Z1,
	     const double ds, const double dz,
	     const float num_planes_per_axial_pos,
	     const float axial_pos_to_z_offset)
{
#ifdef BPINT_CHECK
  const double d_xy = proj_data_info_ptr->get_tangential_sampling();
    
  const double R2 =square(proj_data_info_ptr->get_ring_radius());
  /* Radius of scanner squared in Pixel^2 */
  const double R2p = R2 / d_xy / d_xy;
  // TODO remove assumption
  const int num_planes_per_physical_ring = 2;

  const double t = s + 0.5;		/* In a beam, not on a ray */
  //t=X1*cphi+Y1*sphi;
  //ds=t-s;

  const double new_ds=X1*cphi+Y1*sphi-s;// Eq 6.13 in Egger thsis
  const double root=sqrt(R2p-t*t);//CL 26/10/98 Put it back the original formula as before it was root=sqrt(R2p-s*s)
  // Eq 6.15 from Egger Thesis
  const double z=( Z1-num_planes_per_physical_ring*delta/2*( (-X1*sphi+Y1*cphi)/root + 1 ) -
                     axial_pos_to_z_offset)/num_planes_per_axial_pos;
  const double new_dz=z-ring0; 

  if (fabs(ds-new_ds)>.005 || fabs(dz - new_dz)>.005)
    {
      warning("Difference ds (%g,%g) dz (%g,%g) at X=%d,Y=%d,Z=%d\n",
	      ds,new_ds,dz,new_dz,X1,Y1,Z1);
    }
#endif //BPINT_CHECK
}

/****************************************************************************

   backproj3D_Cho_view_viewplus90 backprojects 4 beams related by
   symmetry (see bckproj.h)

 *****************************************************************************/

// KT 22/05/98 new, but extracted from backproj3D_ChoView45 
// (which used to be called Backproj_Cho_Symstheta_Phiplus90s0_P)
void
BackProjectorByBinUsingInterpolation::
#if PIECEWISE_INTERPOLATION
piecewise_linear_interpolation_backproj3D_Cho_view_viewplus90
#else
linear_interpolation_backproj3D_Cho_view_viewplus90
#endif
(Array<4, float > const &Projptr,
			       VoxelsOnCartesianGrid<float>& image,                               
                               const ProjDataInfoCylindricalArcCorr* proj_data_info_ptr, 
                               float delta,
                               const double cphi, const double sphi, int s, int ring0,
                               const int num_planes_per_axial_pos,
			       const float axial_pos_to_z_offset)
{
  // KT 04/05/2000 new check
#if PIECEWISE_INTERPOLATION
  if (num_planes_per_axial_pos==1 && delta !=0)
    error("This version of the backprojector cannot be used for span>1. \
Recompile %s with PIECEWISE_INTERPOLATION=0", __FILE__);
#else
#ifdef ALTERNATIVE
  // TODO
  if (num_planes_per_axial_pos==1 && delta !=0)
    error("This version of the backprojector cannot be used for span>1. \
Recompile %s with ALTERNATIVE not #defined", __FILE__);
#endif
#endif

#ifndef MOREZ
  // below test was too conservative. We have to adjust it a bit
#error times sqrt(2)/iets met t
  /* Check if delta is not too large. If the condition below is violated,
     z might have to be incremented more than once to remain in the beam.
     To do this, the code would have to be modified with while() instead of if().
     As this will slow it down, we just abort at the moment.
  */



  if (delta * proj_data_info_ptr_info_cyl_ptr->get_tangential_sampling() / proj_data_info_ptr_info_cyl_ptr->get_ring_radius() > 1)
  {
    error("This backprojector cannot handle such oblique segments: delta = %g. Sorry.\n",delta);
  }
#endif

  // conditions on searching flow
  assert(cphi>=0-.001);
  assert(sphi>=0-.001);
  assert(cphi+sphi>=1-.001);

  double dzvert,dzhor,ds;
  double dsdiag,dzdiag,dz;
  int X,Y,Z,Q;

  const float ring_unit = 1./num_planes_per_axial_pos;
  
  // in our current coordinate system, the following constant is always 2
  // TODO remove assumption
  const int num_planes_per_physical_ring = 2;
  assert(fabs(image.get_voxel_size().z() * num_planes_per_physical_ring/ proj_data_info_ptr->get_ring_spacing() -1) < 10E-4);

  /* FOV radius in voxel units */
  // KT 20/06/2001 change calculation of FOV such that even sized image will work
  const float fovrad_in_mm   = 
    min((min(image.get_max_x(), -image.get_min_x()))*image.get_voxel_size().x(),
	(min(image.get_max_y(), -image.get_min_y()))*image.get_voxel_size().y()); 
  const float image_rad = fovrad_in_mm/image.get_voxel_size().x() - 2;
  //const int image_rad = (int)((image.get_x_size()-1)/2);

  //KT 20/06/2001 allow min_z!=0 in all comparisons below
  const int minplane =  image.get_min_z(); 
  const int maxplane =  image.get_max_z(); 
   

  if (find_start_values(proj_data_info_ptr, 
			delta, cphi, sphi, s, ring0,
			image_rad, image.get_voxel_size().z(),
			X, Y, Z,
			ds, dz, dzhor, dzvert,
			num_planes_per_axial_pos,
			axial_pos_to_z_offset) == Succeeded::no)
    {
      // no voxels in the beam
      return;
    }

    /* 
     The formulas below give the values to update a pixel.
     Things are then complicated by optimising this backprojection
     by 'incrementalising' the formulas.
     // linear (original) interpolation
     double UpA0=Projptr[0][0][0][1]+ds*K1A0+dz*K2A0;
     image[Z][Y][X]+= (UpA0+dsdz*K3A0);

     // new interpolation
     double A0a0 = Projptr[0][0][0][1]+ds*K1A0;
     double A0a1 = Projptr[0][0][0][3]+ds*(K1A0+K3A0);
     double UpA0 = 2*(Projptr[0][0][0][1]+ds*K1A0+dz*K2A0) - (A0a0 + A0a1)/2;
     image[Z][Y][X]+= (dz <= 0.25) ? A0a0 : UpA0+twodsdz*K3A0;

     incremental changes:
     // original interpolation
     UpA0inc=dsinc*K1A0+dzinc*K2A0;
      // new interpolation
     A0a0inc = dsinc*K1A0;
     A0a1inc = dsinc*(K1A0+K3A0);
     UpA0inc = A0a0inc - A0a1inc/2 +2*dzinc*K2A0

  */

  //  K1, K2, K3 invariants

  const double K1A0=Projptr[0][0][0][2]-Projptr[0][0][0][1];
  const double K2A0=Projptr[0][0][0][3]-Projptr[0][0][0][1];
  const double K3A0=Projptr[0][0][0][4]-Projptr[0][0][0][2]-K2A0;
  const double K1A2=Projptr[0][2][0][2]-Projptr[0][2][0][1];
  const double K2A2=Projptr[0][2][0][3]-Projptr[0][2][0][1];
  const double K3A2=Projptr[0][2][0][4]-Projptr[0][2][0][2]-K2A2;

  const double K1A0n=Projptr[1][0][0][2]-Projptr[1][0][0][1];
  const double K2A0n=Projptr[1][0][0][3]-Projptr[1][0][0][1];
  const double K3A0n=Projptr[1][0][0][4]-Projptr[1][0][0][2]-K2A0n;
  const double K1A2n=Projptr[1][2][0][2]-Projptr[1][2][0][1];
  const double K2A2n=Projptr[1][2][0][3]-Projptr[1][2][0][1];
  const double K3A2n=Projptr[1][2][0][4]-Projptr[1][2][0][2]-K2A2n;

  const double K1P0=Projptr[0][0][1][2]-Projptr[0][0][1][1];
  const double K2P0=Projptr[0][0][1][3]-Projptr[0][0][1][1];
  const double K3P0=Projptr[0][0][1][4]-Projptr[0][0][1][2]-K2P0;
  const double K1P2=Projptr[0][2][1][2]-Projptr[0][2][1][1];
  const double K2P2=Projptr[0][2][1][3]-Projptr[0][2][1][1];
  const double K3P2=Projptr[0][2][1][4]-Projptr[0][2][1][2]-K2P2;

  const double K1P0n=Projptr[1][0][1][2]-Projptr[1][0][1][1];
  const double K2P0n=Projptr[1][0][1][3]-Projptr[1][0][1][1];
  const double K3P0n=Projptr[1][0][1][4]-Projptr[1][0][1][2]-K2P0n;
  const double K1P2n=Projptr[1][2][1][2]-Projptr[1][2][1][1];
  const double K2P2n=Projptr[1][2][1][3]-Projptr[1][2][1][1];
  const double K3P2n=Projptr[1][2][1][4]-Projptr[1][2][1][2]-K2P2n;


#if !PIECEWISE_INTERPOLATION

  const double ZplusKorrA0=ring_unit*K2A0;
  const double ZplusKorrA2=ring_unit*K2A2;

  const double ZplusKorrA0n=ring_unit*K2A0n;
  const double ZplusKorrA2n=ring_unit*K2A2n;

  const double ZplusKorrP0=ring_unit*K2P0;
  const double ZplusKorrP2=ring_unit*K2P2;

  const double ZplusKorrP0n=ring_unit*K2P0n;
  const double ZplusKorrP2n=ring_unit*K2P2n;

  // V, H, D invariants

  const double VA0=dzvert*K2A0+sphi*K1A0;
  const double HA0=dzhor*K2A0-cphi*K1A0;
  const double DA0=VA0+HA0;
  const double VZA0=VA0+ring_unit*K2A0;
  const double HZA0=HA0+ring_unit*K2A0;
  const double DZA0=DA0+ring_unit*K2A0;
  const double VA2=dzvert*K2A2+sphi*K1A2;
  const double HA2=dzhor*K2A2-cphi*K1A2;
  const double DA2=VA2+HA2;
  const double VZA2=VA2+ring_unit*K2A2;
  const double HZA2=HA2+ring_unit*K2A2;
  const double DZA2=DA2+ring_unit*K2A2;

  const double VA0n=dzvert*K2A0n+sphi*K1A0n;
  const double HA0n=dzhor*K2A0n-cphi*K1A0n;
  const double DA0n=VA0n+HA0n;
  const double VZA0n=VA0n+ring_unit*K2A0n;
  const double HZA0n=HA0n+ring_unit*K2A0n;
  const double DZA0n=DA0n+ring_unit*K2A0n;
  const double VA2n=dzvert*K2A2n+sphi*K1A2n;
  const double HA2n=dzhor*K2A2n-cphi*K1A2n;
  const double DA2n=VA2n+HA2n;
  const double VZA2n=VA2n+ring_unit*K2A2n;
  const double HZA2n=HA2n+ring_unit*K2A2n;
  const double DZA2n=DA2n+ring_unit*K2A2n;

  const double VP0=dzvert*K2P0+sphi*K1P0;
  const double HP0=dzhor*K2P0-cphi*K1P0;
  const double DP0=VP0+HP0;
  const double VZP0=VP0+ring_unit*K2P0;
  const double HZP0=HP0+ring_unit*K2P0;
  const double DZP0=DP0+ring_unit*K2P0;
  const double VP2=dzvert*K2P2+sphi*K1P2;
  const double HP2=dzhor*K2P2-cphi*K1P2;
  const double DP2=VP2+HP2;
  const double VZP2=VP2+ring_unit*K2P2;
  const double HZP2=HP2+ring_unit*K2P2;
  const double DZP2=DP2+ring_unit*K2P2;

  const double VP0n=dzvert*K2P0n+sphi*K1P0n;
  const double HP0n=dzhor*K2P0n-cphi*K1P0n;
  const double DP0n=VP0n+HP0n;
  const double VZP0n=VP0n+ring_unit*K2P0n;
  const double HZP0n=HP0n+ring_unit*K2P0n;
  const double DZP0n=DP0n+ring_unit*K2P0n;
  const double VP2n=dzvert*K2P2n+sphi*K1P2n;
  const double HP2n=dzhor*K2P2n-cphi*K1P2n;
  const double DP2n=VP2n+HP2n;
  const double VZP2n=VP2n+ring_unit*K2P2n;
  const double HZP2n=HP2n+ring_unit*K2P2n;
  const double DZP2n=DP2n+ring_unit*K2P2n;


  // Initial values of update values (Up)

  double UpA0=Projptr[0][0][0][1]+ds*K1A0+dz*K2A0;
  double UpA2=Projptr[0][2][0][1]+ds*K1A2+dz*K2A2;

  double UpA0n=Projptr[1][0][0][1]+ds*K1A0n+dz*K2A0n;
  double UpA2n=Projptr[1][2][0][1]+ds*K1A2n+dz*K2A2n;

  double UpP0=Projptr[0][0][1][1]+ds*K1P0+dz*K2P0;
  double UpP2=Projptr[0][2][1][1]+ds*K1P2+dz*K2P2;

  double UpP0n=Projptr[1][0][1][1]+ds*K1P0n+dz*K2P0n;
  double UpP2n=Projptr[1][2][1][1]+ds*K1P2n+dz*K2P2n;

#else // PIECEWISE_INTERPOLATION

  const double ZplusKorrA0=K2A0;
  const double ZplusKorrA2=K2A2;

  const double ZplusKorrA0n=K2A0n;
  const double ZplusKorrA2n=K2A2n;

  const double ZplusKorrP0=K2P0;
  const double ZplusKorrP2=K2P2;

  const double ZplusKorrP0n=K2P0n;
  const double ZplusKorrP2n=K2P2n;

  // V, H, D invariants

  const double VA0a0=+sphi*K1A0;
  const double HA0a0=-cphi*K1A0;
  const double DA0a0=VA0a0+HA0a0;
  const double VA2a0=+sphi*K1A2;
  const double HA2a0=-cphi*K1A2;
  const double DA2a0=VA2a0+HA2a0;

  const double VA0na0=+sphi*K1A0n;
  const double HA0na0=-cphi*K1A0n;
  const double DA0na0=VA0na0+HA0na0;
  const double VA2na0=+sphi*K1A2n;
  const double HA2na0=-cphi*K1A2n;
  const double DA2na0=VA2na0+HA2na0;

  const double VP0a0=+sphi*K1P0;
  const double HP0a0=-cphi*K1P0;
  const double DP0a0=VP0a0+HP0a0;
  const double VP2a0=+sphi*K1P2;
  const double HP2a0=-cphi*K1P2;
  const double DP2a0=VP2a0+HP2a0;

  const double VP0na0=+sphi*K1P0n;
  const double HP0na0=-cphi*K1P0n;
  const double DP0na0=VP0na0+HP0na0;
  const double VP2na0=+sphi*K1P2n;
  const double HP2na0=-cphi*K1P2n;
  const double DP2na0=VP2na0+HP2na0;


  const double VA0a1=+sphi*(K1A0+K3A0);
  const double HA0a1=-cphi*(K1A0+K3A0);
  const double DA0a1=VA0a1+HA0a1;
  const double VA2a1=+sphi*(K1A2+K3A2);
  const double HA2a1=-cphi*(K1A2+K3A2);
  const double DA2a1=VA2a1+HA2a1;

  const double VA0na1=+sphi*(K1A0n+K3A0n);
  const double HA0na1=-cphi*(K1A0n+K3A0n);
  const double DA0na1=VA0na1+HA0na1;
  const double VA2na1=+sphi*(K1A2n+K3A2n);
  const double HA2na1=-cphi*(K1A2n+K3A2n);
  const double DA2na1=VA2na1+HA2na1;

  const double VP0a1=+sphi*(K1P0+K3P0);
  const double HP0a1=-cphi*(K1P0+K3P0);
  const double DP0a1=VP0a1+HP0a1;
  const double VP2a1=+sphi*(K1P2+K3P2);
  const double HP2a1=-cphi*(K1P2+K3P2);
  const double DP2a1=VP2a1+HP2a1;

  const double VP0na1=+sphi*(K1P0n+K3P0n);
  const double HP0na1=-cphi*(K1P0n+K3P0n);
  const double DP0na1=VP0na1+HP0na1;
  const double VP2na1=+sphi*(K1P2n+K3P2n);
  const double HP2na1=-cphi*(K1P2n+K3P2n);
  const double DP2na1=VP2na1+HP2na1;


  const double VA0=VA0a0*1.5 - VA0a1/2 + 2*dzvert*K2A0;
  const double HA0=HA0a0*1.5 - HA0a1/2 + 2*dzhor*K2A0;
  const double DA0=VA0+HA0;
  const double VZA0=VA0+K2A0;
  const double HZA0=HA0+K2A0;
  const double DZA0=DA0+K2A0;
  const double VA2=VA2a0*1.5 - VA2a1/2 + 2*dzvert*K2A2;
  const double HA2=HA2a0*1.5 - HA2a1/2 + 2*dzhor*K2A2;
  const double DA2=VA2+HA2;
  const double VZA2=VA2+K2A2;
  const double HZA2=HA2+K2A2;
  const double DZA2=DA2+K2A2;

  const double VA0n=VA0na0*1.5 - VA0na1/2 + 2*dzvert*K2A0n;
  const double HA0n=HA0na0*1.5 - HA0na1/2 + 2*dzhor*K2A0n;
  const double DA0n=VA0n+HA0n;
  const double VZA0n=VA0n+K2A0n;
  const double HZA0n=HA0n+K2A0n;
  const double DZA0n=DA0n+K2A0n;
  const double VA2n=VA2na0*1.5 - VA2na1/2 + 2*dzvert*K2A2n;
  const double HA2n=HA2na0*1.5 - HA2na1/2 + 2*dzhor*K2A2n;
  const double DA2n=VA2n+HA2n;
  const double VZA2n=VA2n+K2A2n;
  const double HZA2n=HA2n+K2A2n;
  const double DZA2n=DA2n+K2A2n;

  const double VP0=VP0a0*1.5 - VP0a1/2 + 2*dzvert*K2P0;
  const double HP0=HP0a0*1.5 - HP0a1/2 + 2*dzhor*K2P0;
  const double DP0=VP0+HP0;
  const double VZP0=VP0+K2P0;
  const double HZP0=HP0+K2P0;
  const double DZP0=DP0+K2P0;
  const double VP2=VP2a0*1.5 - VP2a1/2 + 2*dzvert*K2P2;
  const double HP2=HP2a0*1.5 - HP2a1/2 + 2*dzhor*K2P2;
  const double DP2=VP2+HP2;
  const double VZP2=VP2+K2P2;
  const double HZP2=HP2+K2P2;
  const double DZP2=DP2+K2P2;

  const double VP0n=VP0na0*1.5 - VP0na1/2 + 2*dzvert*K2P0n;
  const double HP0n=HP0na0*1.5 - HP0na1/2 + 2*dzhor*K2P0n;
  const double DP0n=VP0n+HP0n;
  const double VZP0n=VP0n+K2P0n;
  const double HZP0n=HP0n+K2P0n;
  const double DZP0n=DP0n+K2P0n;
  const double VP2n=VP2na0*1.5 - VP2na1/2 + 2*dzvert*K2P2n;
  const double HP2n=HP2na0*1.5 - HP2na1/2 + 2*dzhor*K2P2n;
  const double DP2n=VP2n+HP2n;
  const double VZP2n=VP2n+K2P2n;
  const double HZP2n=HP2n+K2P2n;
  const double DZP2n=DP2n+K2P2n;

  // Initial values of update values (Up)
  double A0a0 = Projptr[0][0][0][1]+ds*K1A0;
  double A0a1 = Projptr[0][0][0][3]+ds*(K1A0+K3A0);
  double A2a0  = Projptr[0][2][0][1]+ds*K1A2;
  double A2a1  = Projptr[0][2][0][3]+ds*(K1A2+K3A2);

  double P0na0  = Projptr[1][0][1][1]+ds*K1P0n;
  double P0na1  = Projptr[1][0][1][3]+ds*(K1P0n+K3P0n);
  double P2na0  = Projptr[1][2][1][1]+ds*K1P2n;
  double P2na1  = Projptr[1][2][1][3]+ds*(K1P2n+K3P2n);

  double A0na0  = Projptr[1][0][0][1]+ds*K1A0n;
  double A0na1  = Projptr[1][0][0][3]+ds*(K1A0n+K3A0n);
  double A2na0  = Projptr[1][2][0][1]+ds*K1A2n;
  double A2na1  = Projptr[1][2][0][3]+ds*(K1A2n+K3A2n);

  double P0a0  = Projptr[0][0][1][1]+ds*K1P0;
  double P0a1  = Projptr[0][0][1][3]+ds*(K1P0+K3P0);
  double P2a0  = Projptr[0][2][1][1]+ds*K1P2;
  double P2a1  = Projptr[0][2][1][3]+ds*(K1P2+K3P2);


  double UpA0 = 2*(Projptr[0][0][0][1]+ds*K1A0+dz*K2A0) - (A0a0 + A0a1)/2;
  double UpA2 = 2*(Projptr[0][2][0][1]+ds*K1A2+dz*K2A2) - (A2a0 + A2a1)/2;

  double UpA0n = 2*(Projptr[1][0][0][1]+ds*K1A0n+dz*K2A0n) - (A0na0 + A0na1)/2;
  double UpA2n = 2*(Projptr[1][2][0][1]+ds*K1A2n+dz*K2A2n) - (A2na0 + A2na1)/2;

  double UpP0 = 2*(Projptr[0][0][1][1]+ds*K1P0+dz*K2P0) - (P0a0 + P0a1)/2;
  double UpP2 = 2*(Projptr[0][2][1][1]+ds*K1P2+dz*K2P2) - (P2a0 + P2a1)/2;

  double UpP0n = 2*(Projptr[1][0][1][1]+ds*K1P0n+dz*K2P0n) - (P0na0 + P0na1)/2;
  double UpP2n = 2*(Projptr[1][2][1][1]+ds*K1P2n+dz*K2P2n) - (P2na0 + P2na1)/2;

#endif // PIECEWISE_INTERPOLATION

  // some extra start values

  // Find symmetric value in Z by 'mirroring' it around the centre z of the LORs:
  // Z+Q = 2*centre_of_LOR_in_image_coordinates
  // Note that we are backprojecting ring0 and ring0+1, so we mirror around ring0+0.5
  // original  Q = (int)  (4*ring0+2*delta+2-Z + 0.5);
  // CL&KT 21/12/99 added axial_pos_to_z_offset and num_planes_per_physical_ring
  {
    // first compute it as floating point (although it has to be an int really)
    const float Qf = (2*num_planes_per_axial_pos*(ring0 + 0.5) 
                      + num_planes_per_physical_ring*delta
	              + 2*axial_pos_to_z_offset
	              - Z); 
    // now use rounding to be safe
    Q = (int)floor(Qf + 0.5);
    assert(fabs(Q-Qf) < 10E-4);
  }

  dzdiag=dzvert+dzhor;
  dsdiag=-cphi+sphi;


  /* KT 13/05/98 changed loop condition, originally a combination of  
     while (X>X2||Y<Y2||Z<Z2) {
     // code which increments Y,Z or decrements X
     if (X<X2||Y>Y2||Z>Z2)
     }
     this had problems for nearly horizontal or vertical lines.
     For example, while Y==Y2+1, some extra iterations had to be done
     on X.
     Now I break out of the while when the voxel goes out of the cylinder,
     which means I don't have to use X2,Y2,Z2 anymore
  */
  do 
    {
      assert(ds >= 0);
      assert(ds <= 1);
      assert(dz >= 0);
      assert(dz <= ring_unit+epsilon);

      // Update voxel values for this X,Y,Z
      // For 1 given (X,Y)-position, there are always 2 voxels along the z-axis 
      // in this beam
      const int Zplus=Z+1;
      const int Qmin=Q-1;
      
#if PIECEWISE_INTERPOLATION
      //KT&MJ 07/08/98 new

      // KT 16/06/98 changed check ds!=0 to fabs(ds)>epsilon for better rounding control
      const bool do_s_symmetry = (s!=0 || fabs(ds) > epsilon);
      const double twodsdz=2*ds*dz;
      const double twodsdz2=2*ds*(dz+0.5);

      if (Z>=minplane&&Z<=maxplane) {
	// original image[Z][Y][X]+=UpA0+dsdz*K3A0
	image[Z][Y][X]+= (dz <= 0.25) ? A0a0 : UpA0+twodsdz*K3A0;
        image[Z][X][-Y]+=(dz <= 0.25) ? A2a0 : UpA2 +twodsdz*K3A2;
	if (do_s_symmetry) {
	  image[Z][-Y][-X]+= (dz <= 0.25) ? P0na0 : UpP0n +twodsdz*K3P0n;
	  image[Z][-X][Y]+=(dz <= 0.25) ? P2na0 : UpP2n +twodsdz*K3P2n;
        }

      }
      if (Zplus>=minplane&&Zplus<=maxplane) {
        image[Zplus][Y][X]+=(dz >= 0.25) ? A0a1 : UpA0+twodsdz2*K3A0 +ZplusKorrA0;
        image[Zplus][X][-Y]+=(dz >= 0.25) ? A2a1 : UpA2+twodsdz2*K3A2 +ZplusKorrA2;
	if (do_s_symmetry) {
	  image[Zplus][-Y][-X]+=(dz >= 0.25) ? P0na1 : UpP0n+twodsdz2*K3P0n +ZplusKorrP0n;
	  image[Zplus][-X][Y]+=(dz >= 0.25) ? P2na1 : UpP2n+twodsdz2*K3P2n +ZplusKorrP2n;
        }

      }
      if (Q>=minplane&&Q<=maxplane) {
        image[Q][Y][X]+=(dz <= 0.25) ? A0na0 : UpA0n +twodsdz*K3A0n;
        image[Q][X][-Y]+=(dz <= 0.25) ? A2na0 : UpA2n +twodsdz*K3A2n;
        if (do_s_symmetry) {
	  image[Q][-Y][-X]+=(dz <= 0.25) ? P0a0 : UpP0 +twodsdz*K3P0;
	  image[Q][-X][Y]+=(dz <= 0.25) ? P2a0 : UpP2 +twodsdz*K3P2;
        }
      }
      if (Qmin>=minplane&&Qmin<=maxplane) {
        image[Qmin][Y][X]+=(dz >= 0.25) ? A0na1 : UpA0n+twodsdz2*K3A0n + ZplusKorrA0n;
        image[Qmin][X][-Y]+=(dz >= 0.25) ? A2na1 : UpA2n+twodsdz2*K3A2n + ZplusKorrA2n;
        if (do_s_symmetry) {
	  image[Qmin][-Y][-X]+=(dz >= 0.25) ? P0a1 : UpP0+twodsdz2*K3P0 + ZplusKorrP0;
	  image[Qmin][-X][Y]+=(dz >= 0.25) ? P2a1 : UpP2+twodsdz2*K3P2 + ZplusKorrP2;
        }
      }
#else // !PIECEWISE_INTERPOLATION
#ifdef ALTERNATIVE 
      // This code and the one after #else are equivalent. However, this version is
      // shorter, while the other version looks 'more optimal'.
      // TODO check which is faster.
      // KT 16/06/98 changed check ds!=0 to fabs(ds)>epsilon for better rounding control
      const bool do_s_symmetry = (s!=0 || fabs(ds) > epsilon);
      const double dsdz=ds*dz;
      const double dsdz2=ds*(dz+ring_unit);

      if (Z>=minplane&&Z<=maxplane) {
        image[Z][Y][X]+=UpA0+dsdz*K3A0;
        image[Z][X][-Y]+=UpA2+dsdz*K3A2;
	if (do_s_symmetry) {
	  image[Z][-Y][-X]+=UpP0n+dsdz*K3P0n;
	  image[Z][-X][Y]+=UpP2n+dsdz*K3P2n;
        }

      }
          //CL SPAN 15/09/98 Add one more condition in order to skip the statement in case of span data
          //as there is only one voxel in the beam in slice unit
      if (Zplus>=minplane&&Zplus<=maxplane && ring_unit==0.5) {
        image[Zplus][Y][X]+=UpA0+dsdz2*K3A0+ZplusKorrA0;
        image[Zplus][X][-Y]+=UpA2+dsdz2*K3A2+ZplusKorrA2;
	if (do_s_symmetry) {
	  image[Zplus][-Y][-X]+=UpP0n+dsdz2*K3P0n+ZplusKorrP0n;
	  image[Zplus][-X][Y]+=UpP2n+dsdz2*K3P2n+ZplusKorrP2n;
        }

      }
      if (Q>=minplane&&Q<=maxplane) {
        image[Q][Y][X]+=UpA0n+dsdz*K3A0n;
        image[Q][X][-Y]+=UpA2n+dsdz*K3A2n;
        if (do_s_symmetry) {
	  image[Q][-Y][-X]+=UpP0+dsdz*K3P0;
	  image[Q][-X][Y]+=UpP2+dsdz*K3P2;
        }
      }
          //CL SPAN 15/09/98 Add one more condition in order to skip the statement in case of span data
          //as there is only one voxel in the beam in slice unit
      if (Qmin>=minplane&&Qmin<=maxplane && ring_unit==0.5) {
        image[Qmin][Y][X]+=UpA0n+dsdz2*K3A0n+ZplusKorrA0n;
        image[Qmin][X][-Y]+=UpA2n+dsdz2*K3A2n+ZplusKorrA2n;
        if (do_s_symmetry) {
	  image[Qmin][-Y][-X]+=UpP0+dsdz2*K3P0+ZplusKorrP0;
	  image[Qmin][-X][Y]+=UpP2+dsdz2*K3P2+ZplusKorrP2;
        }
      }
#else // ALTERNATIVE
     double TMP1,TMP2;

   //   if(Z == maxplane || Q == maxplane || Zplus == maxplane || Qmin == maxplane)
    //      cout << " " << ring0 ;
      
      TMP1=ds*K3A0;
      TMP2=UpA0+dz*TMP1;
      if (Z>=minplane&&Z<=maxplane)
	image[Z][Y][X]+=TMP2;
          //CL SPAN 15/09/98 Add one more condition in order to skip the statement in case of span data
          //as there is only one voxel in the beam in slice unit
       if (Zplus>=minplane&&Zplus<=maxplane && ring_unit==0.5)
	image[Zplus][Y][X]+=TMP2+ring_unit*TMP1+ZplusKorrA0;
      TMP1=ds*K3A2;
      TMP2=UpA2+dz*TMP1;
      if (Z>=minplane&&Z<=maxplane)
	image[Z][X][-Y]+=TMP2;
          //CL SPAN 15/09/98 Add one more condition in order to skip the statement in case of span data
          //as there is only one voxel in the beam in slice unit
       if (Zplus>=minplane&&Zplus<=maxplane && ring_unit==0.5)
	image[Zplus][X][-Y]+=TMP2+ring_unit*TMP1+ZplusKorrA2;
  
      TMP1=ds*K3A0n;
      TMP2=UpA0n+dz*TMP1;
      if (Q>=minplane&&Q<=maxplane)
	image[Q][Y][X]+=TMP2;
          //CL SPAN 15/09/98 Add one more condition in order to skip the statement in case of span data
          //as there is only one voxel in the beam in slice unit
      if (Qmin>=minplane&&Qmin<=maxplane && ring_unit==0.5)
	image[Qmin][Y][X]+=TMP2+ring_unit*TMP1+ZplusKorrA0n;
      TMP1=ds*K3A2n;
      TMP2=UpA2n+dz*TMP1;
      if (Q>=minplane&&Q<=maxplane)
	image[Q][X][-Y]+=TMP2;
          //CL SPAN 15/09/98 Add one more condition in order to skip the statement in case of span data
          //as there is only one voxel in the beam in slice unit
      if (Qmin>=minplane&&Qmin<=maxplane && ring_unit==0.5)
	image[Qmin][X][-Y]+=TMP2+ring_unit*TMP1+ZplusKorrA2n;
  
      // KT 14/05/98 changed X!=-Y to s!=0 || ds!=0 to make it work for view != view45
      // KT 16/06/98 changed check ds!=0 to fabs(ds)>epsilon for better rounding control
      if (s!=0 || fabs(ds) > epsilon) {
	TMP1=ds*K3P0;
	TMP2=UpP0+dz*TMP1;
	if (Q>=minplane&&Q<=maxplane)
	  image[Q][-Y][-X]+=TMP2;
          //CL SPAN 15/09/98 Add one more condition in order to skip the statement in case of span data
          //as there is only one voxel in the beam in slice unit
 	if (Qmin>=minplane&&Qmin<=maxplane && ring_unit==0.5)
	  image[Qmin][-Y][-X]+=TMP2+ring_unit*TMP1+ZplusKorrP0;
	TMP1=ds*K3P2;
	TMP2=UpP2+dz*TMP1;
	if (Q>=minplane&&Q<=maxplane)
	  image[Q][-X][Y]+=TMP2;
          //CL SPAN 15/09/98 Add one more condition in order to skip the statement in case of span data
          //as there is only one voxel in the beam in slice unit
 	if (Qmin>=minplane&&Qmin<=maxplane && ring_unit==0.5)
	  image[Qmin][-X][Y]+=TMP2+ring_unit*TMP1+ZplusKorrP2;
  
	TMP1=ds*K3P0n;
	TMP2=UpP0n+dz*TMP1;
	if (Z>=minplane&&Z<=maxplane)
	  image[Z][-Y][-X]+=TMP2;
          //CL SPAN 15/09/98 Add one more condition in order to skip the statement in case of span data
          //as there is only one voxel in the beam in slice unit
 	if (Zplus>=minplane&&Zplus<=maxplane && ring_unit==0.5)
	  image[Zplus][-Y][-X]+=TMP2+ring_unit*TMP1+ZplusKorrP0n;
	TMP1=ds*K3P2n;
	TMP2=UpP2n+dz*TMP1;
	if (Z>=minplane&&Z<=maxplane)
	  image[Z][-X][Y]+=TMP2;
          //CL SPAN 15/09/98 Add one more condition in order to skip the statement in case of span data
          //as there is only one voxel in the beam in slice unit
 	if (Zplus>=minplane&&Zplus<=maxplane && ring_unit==0.5)
	  image[Zplus][-X][Y]+=TMP2+ring_unit*TMP1+ZplusKorrP2n;
      }
#endif // ALTERNATIVE 
#endif // PIECEWISE_INTERPOLATION

      //  Search for next pixel in the beam

      if (ds>=cphi) {
	/* horizontal*/
	X-=1;
	ds-=cphi;
	dz+=dzhor;
	if (dz<epsilon) {     /* increment Z */
	  Z++; Q--; dz+=ring_unit;
	  UpA0+=HZA0;
	  UpA2+=HZA2;
	  UpA0n+=HZA0n;
	  UpA2n+=HZA2n;
	  UpP0+=HZP0;
	  UpP2+=HZP2;
	  UpP0n+=HZP0n;
	  UpP2n+=HZP2n;
#ifdef MOREZ
          while (dz<epsilon)
          {
	  Z++; Q--; dz+=ring_unit;
    	  UpA0+=HZA0-HA0;
	  UpA2+=HZA2-HA2;
	  UpA0n+=HZA0n-HA0n;
	  UpA2n+=HZA2n-HA2n;
	  UpP0+=HZP0-HP0;
	  UpP2+=HZP2-HP2;
	  UpP0n+=HZP0n-HP0n;
	  UpP2n+=HZP2n-HP2n;
	  }
#endif
        }
    
	else {       /* Z does not change */
	  UpA0+=HA0;
	  UpA2+=HA2;
	  UpA0n+=HA0n;
	  UpA2n+=HA2n;
	  UpP0+=HP0;
	  UpP2+=HP2;
	  UpP0n+=HP0n;
	  UpP2n+=HP2n;
	}
#if PIECEWISE_INTERPOLATION
	A0a0+=HA0a0;
	A2a0+=HA2a0;
	A0na0+=HA0na0;
	A2na0+=HA2na0;
	P0a0+=HP0a0;
	P2a0+=HP2a0;
	P0na0+=HP0na0;
	P2na0+=HP2na0;
	
	A0a1+=HA0a1;
	A2a1+=HA2a1;
	A0na1+=HA0na1;
	A2na1+=HA2na1;
	P0a1+=HP0a1;
	P2a1+=HP2a1;
	P0na1+=HP0na1;
	P2na1+=HP2na1;
#endif // PIECEWISE_INTERPOLATION
      }
      // KT 14/05/98 use < instead of <= (see formula 6.17 in Egger's thesis)
      else if (ds<1-sphi) {
	/*  vertical*/
	Y+=1;
	ds+=sphi;
	dz+=dzvert;
	if (dz<epsilon) {       /* increment Z */
	  Z++; Q--; dz+=ring_unit;
	  UpA0+=VZA0;
	  UpA2+=VZA2;
	  UpA0n+=VZA0n;
	  UpA2n+=VZA2n;
	  UpP0+=VZP0;
	  UpP2+=VZP2;
	  UpP0n+=VZP0n;
	  UpP2n+=VZP2n;

#ifdef MOREZ
          while (dz<epsilon)
          {
	  Z++; Q--; dz+=ring_unit;
	  UpA0+=VZA0-VA0;
	  UpA2+=VZA2-VA2;
	  UpA0n+=VZA0n-VA0n;
	  UpA2n+=VZA2n-VA2n;
	  UpP0+=VZP0-VP0;
	  UpP2+=VZP2-VP2;
	  UpP0n+=VZP0n-VP0n;
	  UpP2n+=VZP2n-VP2n;
        }
#endif
        }
	else {              /* Z does not change  */
	  UpA0+=VA0;
	  UpA2+=VA2;
	  UpA0n+=VA0n;
	  UpA2n+=VA2n;
	  UpP0+=VP0;
	  UpP2+=VP2;
	  UpP0n+=VP0n;
	  UpP2n+=VP2n;
	}
#if PIECEWISE_INTERPOLATION
	A0a0+=VA0a0;
	A2a0+=VA2a0;
	A0na0+=VA0na0;
	A2na0+=VA2na0;
	P0a0+=VP0a0;
	P2a0+=VP2a0;
	P0na0+=VP0na0;
	P2na0+=VP2na0;
	
	A0a1+=VA0a1;
	A2a1+=VA2a1;
	A0na1+=VA0na1;
	A2na1+=VA2na1;
	P0a1+=VP0a1;
	P2a1+=VP2a1;
	P0na1+=VP0na1;
	P2na1+=VP2na1;
#endif // PIECEWISE_INTERPOLATION               
      }
      else {
	/*  diagonal*/
	X-=1;
	Y+=1;
	ds+=dsdiag;
	dz+=dzdiag;
	if (dz<epsilon) {      /* increment Z */
	  Z++; Q--; dz+=ring_unit;
	  UpA0+=DZA0;
	  UpA2+=DZA2;
	  UpA0n+=DZA0n;
	  UpA2n+=DZA2n;
	  UpP0+=DZP0;
	  UpP2+=DZP2;
	  UpP0n+=DZP0n;
	  UpP2n+=DZP2n;
#ifdef MOREZ
          while (dz<epsilon)
          {
	  Z++; Q--; dz+=ring_unit;
	  UpA0+=DZA0-DA0;
	  UpA2+=DZA2-DA2;
	  UpA0n+=DZA0n-DA0n;
	  UpA2n+=DZA2n-DA2n;
	  UpP0+=DZP0-DP0;
	  UpP2+=DZP2-DP2;
	  UpP0n+=DZP0n-DP0n;
	  UpP2n+=DZP2n-DP2n;
        }
#endif
        }
    
	else {       /* Z does not change */
	  UpA0+=DA0;
	  UpA2+=DA2;
	  UpA0n+=DA0n;
	  UpA2n+=DA2n;
	  UpP0+=DP0;
	  UpP2+=DP2;
	  UpP0n+=DP0n;
	  UpP2n+=DP2n;
	}                  
#if PIECEWISE_INTERPOLATION
	A0a0+=DA0a0;
	A2a0+=DA2a0;
	A0na0+=DA0na0;
	A2na0+=DA2na0;
	P0a0+=DP0a0;
	P2a0+=DP2a0;
	P0na0+=DP0na0;
	P2na0+=DP2na0;
	
	A0a1+=DA0a1;
	A2a1+=DA2a1;
	A0na1+=DA0na1;
	A2na1+=DA2na1;
	P0a1+=DP0a1;
	P2a1+=DP2a1;
	P0na1+=DP0na1;
	P2na1+=DP2na1;
#endif // PIECEWISE_INTERPOLATION
      }
      check_values(proj_data_info_ptr, 
		   delta, cphi, sphi, s, ring0,
		   X, Y, Z,
		   ds, dz, 
		   num_planes_per_axial_pos,
		   axial_pos_to_z_offset);
    }
  while ((X*X + Y*Y <= image_rad*image_rad) && (Z<=maxplane || Q>=minplane));

}

/*****************************************************************************

   backproj3D_Cho_view_viewplus90_180minview_90minview backprojects 8 beams 
   related by symmetry (see bckproj.h)

   The structure of this function is essentially the same as the previous one.
   Only we have more variables because we are handling 8 viewgrams here.
 *****************************************************************************/
 
// KT 22/05/98 new, but extracted from backproj3D_ChoViews 
// (which used to be called Backproj_Cho_Symsphitheta_P)

void 
BackProjectorByBinUsingInterpolation::
#if PIECEWISE_INTERPOLATION
piecewise_linear_interpolation_backproj3D_Cho_view_viewplus90_180minview_90minview
#else
linear_interpolation_backproj3D_Cho_view_viewplus90_180minview_90minview
#endif
 (Array<4, float > const& Projptr,
						     VoxelsOnCartesianGrid<float>& image,                                                     
                                                     const ProjDataInfoCylindricalArcCorr* proj_data_info_ptr, 
                                                      float delta,
                                                      const double cphi, const double sphi,
                                                      int s, int ring0,
                                                      const int num_planes_per_axial_pos,
						      const float axial_pos_to_z_offset)
{
  // KT 04/05/2000 new check
#if PIECEWISE_INTERPOLATION
  if (num_planes_per_axial_pos==1 && delta !=0)
    error("This version of the backprojector cannot be used for span>1. \
Recompile %s with PIECEWISE_INTERPOLATION=0", __FILE__);
#else
#ifdef ALTERNATIVE
  // TODO
  if (num_planes_per_axial_pos==1 && delta !=0)
    error("This version of the backprojector cannot be used for span>1. \
Recompile %s with ALTERNATIVE not #defined", __FILE__);
#endif
#endif

#ifndef MOREZ
  // KT 14/04/2000 new check
  /* Check if delta is not too large. If the condition below is violated,
     z might have to be incremented more than once to remain in the beam.
     To do this, the code would have to be modified with while() instead of if().
     As this will slow it down, we just abort at the moment.
  */

 
  if (delta * proj_data_info_ptr_info_cyl_ptr->get_tangential_sampling() /proj_data_info_ptr_info_cyl_ptr->get_ring_radius() > 1)
  {
    error("This backprojector cannot handle such oblique segments: delta = %g. Sorry.\n",delta);
  }
#endif

  // conditions on searching flow
  assert(cphi>=0-.001);
  assert(sphi>=0-.001);
  assert(cphi+sphi>=1-.001);

  const float ring_unit = 1./num_planes_per_axial_pos;
  // CL&KT 21/12/99 new
  // in our current coordinate system, the following constant is always 2
  const int num_planes_per_physical_ring = 2;
  assert(fabs(image.get_voxel_size().z() * num_planes_per_physical_ring/ proj_data_info_ptr->get_ring_spacing() -1) < 10E-4);

  double dzvert,dzhor,ds;
  double dsdiag,dzdiag,dz;
  int X,Y,Z,Q;

  /* FOV radius in voxel units */
  // KT 20/06/2001 change calculation of FOV such that even sized image will work
  const float fovrad_in_mm   = 
    min((min(image.get_max_x(), -image.get_min_x()))*image.get_voxel_size().x(),
	(min(image.get_max_y(), -image.get_min_y()))*image.get_voxel_size().y()); 
  const float image_rad = fovrad_in_mm/image.get_voxel_size().x() - 2;
  //const int image_rad = (int)((image.get_x_size()-1)/2);
  //KT 20/06/2001 allow min_z!=0 in all comparisons below
  const int minplane =  image.get_min_z(); 
  const int maxplane =  image.get_max_z(); 
   

  if(find_start_values(proj_data_info_ptr, 
		       delta, cphi, sphi, s, ring0,
		       image_rad, image.get_voxel_size().z(),
		       X, Y, Z,
		       ds, dz, dzhor, dzvert,                    
		       num_planes_per_axial_pos,
		       axial_pos_to_z_offset) == Succeeded::no)
    {
      // no voxels in the beam
      return;
    }

  // K1, K2, K3 invariants

  const double K1A0=Projptr[0][0][0][2]-Projptr[0][0][0][1];
  const double K2A0=Projptr[0][0][0][3]-Projptr[0][0][0][1];
  const double K3A0=Projptr[0][0][0][4]-Projptr[0][0][0][2]-K2A0;
  const double K1A1=Projptr[0][1][0][2]-Projptr[0][1][0][1];
  const double K2A1=Projptr[0][1][0][3]-Projptr[0][1][0][1];
  const double K3A1=Projptr[0][1][0][4]-Projptr[0][1][0][2]-K2A1;
  const double K1A2=Projptr[0][2][0][2]-Projptr[0][2][0][1];
  const double K2A2=Projptr[0][2][0][3]-Projptr[0][2][0][1];
  const double K3A2=Projptr[0][2][0][4]-Projptr[0][2][0][2]-K2A2;
  const double K1A3=Projptr[0][3][0][2]-Projptr[0][3][0][1];
  const double K2A3=Projptr[0][3][0][3]-Projptr[0][3][0][1];
  const double K3A3=Projptr[0][3][0][4]-Projptr[0][3][0][2]-K2A3;

  const double K1A0n=Projptr[1][0][0][2]-Projptr[1][0][0][1];
  const double K2A0n=Projptr[1][0][0][3]-Projptr[1][0][0][1];
  const double K3A0n=Projptr[1][0][0][4]-Projptr[1][0][0][2]-K2A0n;
  const double K1A1n=Projptr[1][1][0][2]-Projptr[1][1][0][1];
  const double K2A1n=Projptr[1][1][0][3]-Projptr[1][1][0][1];
  const double K3A1n=Projptr[1][1][0][4]-Projptr[1][1][0][2]-K2A1n;
  const double K1A2n=Projptr[1][2][0][2]-Projptr[1][2][0][1];
  const double K2A2n=Projptr[1][2][0][3]-Projptr[1][2][0][1];
  const double K3A2n=Projptr[1][2][0][4]-Projptr[1][2][0][2]-K2A2n;
  const double K1A3n=Projptr[1][3][0][2]-Projptr[1][3][0][1];
  const double K2A3n=Projptr[1][3][0][3]-Projptr[1][3][0][1];
  const double K3A3n=Projptr[1][3][0][4]-Projptr[1][3][0][2]-K2A3n;

  const double K1P0=Projptr[0][0][1][2]-Projptr[0][0][1][1];
  const double K2P0=Projptr[0][0][1][3]-Projptr[0][0][1][1];
  const double K3P0=Projptr[0][0][1][4]-Projptr[0][0][1][2]-K2P0;
  const double K1P1=Projptr[0][1][1][2]-Projptr[0][1][1][1];
  const double K2P1=Projptr[0][1][1][3]-Projptr[0][1][1][1];
  const double K3P1=Projptr[0][1][1][4]-Projptr[0][1][1][2]-K2P1;
  const double K1P2=Projptr[0][2][1][2]-Projptr[0][2][1][1];
  const double K2P2=Projptr[0][2][1][3]-Projptr[0][2][1][1];
  const double K3P2=Projptr[0][2][1][4]-Projptr[0][2][1][2]-K2P2;
  const double K1P3=Projptr[0][3][1][2]-Projptr[0][3][1][1];
  const double K2P3=Projptr[0][3][1][3]-Projptr[0][3][1][1];
  const double K3P3=Projptr[0][3][1][4]-Projptr[0][3][1][2]-K2P3;

  const double K1P0n=Projptr[1][0][1][2]-Projptr[1][0][1][1];
  const double K2P0n=Projptr[1][0][1][3]-Projptr[1][0][1][1];
  const double K3P0n=Projptr[1][0][1][4]-Projptr[1][0][1][2]-K2P0n;
  const double K1P1n=Projptr[1][1][1][2]-Projptr[1][1][1][1];
  const double K2P1n=Projptr[1][1][1][3]-Projptr[1][1][1][1];
  const double K3P1n=Projptr[1][1][1][4]-Projptr[1][1][1][2]-K2P1n;
  const double K1P2n=Projptr[1][2][1][2]-Projptr[1][2][1][1];
  const double K2P2n=Projptr[1][2][1][3]-Projptr[1][2][1][1];
  const double K3P2n=Projptr[1][2][1][4]-Projptr[1][2][1][2]-K2P2n;
  const double K1P3n=Projptr[1][3][1][2]-Projptr[1][3][1][1];
  const double K2P3n=Projptr[1][3][1][3]-Projptr[1][3][1][1];
  const double K3P3n=Projptr[1][3][1][4]-Projptr[1][3][1][2]-K2P3n;

#if !PIECEWISE_INTERPOLATION


  const double ZplusKorrA0=ring_unit*K2A0;
  const double ZplusKorrA1=ring_unit*K2A1;
  const double ZplusKorrA2=ring_unit*K2A2;
  const double ZplusKorrA3=ring_unit*K2A3;

  const double ZplusKorrA0n=ring_unit*K2A0n;
  const double ZplusKorrA1n=ring_unit*K2A1n;
  const double ZplusKorrA2n=ring_unit*K2A2n;
  const double ZplusKorrA3n=ring_unit*K2A3n;

  const double ZplusKorrP0=ring_unit*K2P0;
  const double ZplusKorrP1=ring_unit*K2P1;
  const double ZplusKorrP2=ring_unit*K2P2;
  const double ZplusKorrP3=ring_unit*K2P3;

  const double ZplusKorrP0n=ring_unit*K2P0n;
  const double ZplusKorrP1n=ring_unit*K2P1n;
  const double ZplusKorrP2n=ring_unit*K2P2n;
  const double ZplusKorrP3n=ring_unit*K2P3n;

  //  V, H, D invariants

  const double VA0=dzvert*K2A0+sphi*K1A0;
  const double HA0=dzhor*K2A0-cphi*K1A0;
  const double DA0=VA0+HA0;
  const double VZA0=VA0+ring_unit*K2A0;
  const double HZA0=HA0+ring_unit*K2A0;
  const double DZA0=DA0+ring_unit*K2A0;
  const double VA1=dzvert*K2A1+sphi*K1A1;
  const double HA1=dzhor*K2A1-cphi*K1A1;
  const double DA1=VA1+HA1;
  const double VZA1=VA1+ring_unit*K2A1;
  const double HZA1=HA1+ring_unit*K2A1;
  const double DZA1=DA1+ring_unit*K2A1;
  const double VA2=dzvert*K2A2+sphi*K1A2;
  const double HA2=dzhor*K2A2-cphi*K1A2;
  const double DA2=VA2+HA2;
  const double VZA2=VA2+ring_unit*K2A2;
  const double HZA2=HA2+ring_unit*K2A2;
  const double DZA2=DA2+ring_unit*K2A2;
  const double VA3=dzvert*K2A3+sphi*K1A3;
  const double HA3=dzhor*K2A3-cphi*K1A3;
  const double DA3=VA3+HA3;
  const double VZA3=VA3+ring_unit*K2A3;
  const double HZA3=HA3+ring_unit*K2A3;
  const double DZA3=DA3+ring_unit*K2A3;

  const double VA0n=dzvert*K2A0n+sphi*K1A0n;
  const double HA0n=dzhor*K2A0n-cphi*K1A0n;
  const double DA0n=VA0n+HA0n;
  const double VZA0n=VA0n+ring_unit*K2A0n;
  const double HZA0n=HA0n+ring_unit*K2A0n;
  const double DZA0n=DA0n+ring_unit*K2A0n;
  const double VA1n=dzvert*K2A1n+sphi*K1A1n;
  const double HA1n=dzhor*K2A1n-cphi*K1A1n;
  const double DA1n=VA1n+HA1n;
  const double VZA1n=VA1n+ring_unit*K2A1n;
  const double HZA1n=HA1n+ring_unit*K2A1n;
  const double DZA1n=DA1n+ring_unit*K2A1n;
  const double VA2n=dzvert*K2A2n+sphi*K1A2n;
  const double HA2n=dzhor*K2A2n-cphi*K1A2n;
  const double DA2n=VA2n+HA2n;
  const double VZA2n=VA2n+ring_unit*K2A2n;
  const double HZA2n=HA2n+ring_unit*K2A2n;
  const double DZA2n=DA2n+ring_unit*K2A2n;
  const double VA3n=dzvert*K2A3n+sphi*K1A3n;
  const double HA3n=dzhor*K2A3n-cphi*K1A3n;
  const double DA3n=VA3n+HA3n;
  const double VZA3n=VA3n+ring_unit*K2A3n;
  const double HZA3n=HA3n+ring_unit*K2A3n;
  const double DZA3n=DA3n+ring_unit*K2A3n;

  const double VP0=dzvert*K2P0+sphi*K1P0;
  const double HP0=dzhor*K2P0-cphi*K1P0;
  const double DP0=VP0+HP0;
  const double VZP0=VP0+ring_unit*K2P0;
  const double HZP0=HP0+ring_unit*K2P0;
  const double DZP0=DP0+ring_unit*K2P0;
  const double VP1=dzvert*K2P1+sphi*K1P1;
  const double HP1=dzhor*K2P1-cphi*K1P1;
  const double DP1=VP1+HP1;
  const double VZP1=VP1+ring_unit*K2P1;
  const double HZP1=HP1+ring_unit*K2P1;
  const double DZP1=DP1+ring_unit*K2P1;
  const double VP2=dzvert*K2P2+sphi*K1P2;
  const double HP2=dzhor*K2P2-cphi*K1P2;
  const double DP2=VP2+HP2;
  const double VZP2=VP2+ring_unit*K2P2;
  const double HZP2=HP2+ring_unit*K2P2;
  const double DZP2=DP2+ring_unit*K2P2;
  const double VP3=dzvert*K2P3+sphi*K1P3;
  const double HP3=dzhor*K2P3-cphi*K1P3;
  const double DP3=VP3+HP3;
  const double VZP3=VP3+ring_unit*K2P3;
  const double HZP3=HP3+ring_unit*K2P3;
  const double DZP3=DP3+ring_unit*K2P3;

  const double VP0n=dzvert*K2P0n+sphi*K1P0n;
  const double HP0n=dzhor*K2P0n-cphi*K1P0n;
  const double DP0n=VP0n+HP0n;
  const double VZP0n=VP0n+ring_unit*K2P0n;
  const double HZP0n=HP0n+ring_unit*K2P0n;
  const double DZP0n=DP0n+ring_unit*K2P0n;
  const double VP1n=dzvert*K2P1n+sphi*K1P1n;
  const double HP1n=dzhor*K2P1n-cphi*K1P1n;
  const double DP1n=VP1n+HP1n;
  const double VZP1n=VP1n+ring_unit*K2P1n;
  const double HZP1n=HP1n+ring_unit*K2P1n;
  const double DZP1n=DP1n+ring_unit*K2P1n;
  const double VP2n=dzvert*K2P2n+sphi*K1P2n;
  const double HP2n=dzhor*K2P2n-cphi*K1P2n;
  const double DP2n=VP2n+HP2n;
  const double VZP2n=VP2n+ring_unit*K2P2n;
  const double HZP2n=HP2n+ring_unit*K2P2n;
  const double DZP2n=DP2n+ring_unit*K2P2n;
  const double VP3n=dzvert*K2P3n+sphi*K1P3n;
  const double HP3n=dzhor*K2P3n-cphi*K1P3n;
  const double DP3n=VP3n+HP3n;
  const double VZP3n=VP3n+ring_unit*K2P3n;
  const double HZP3n=HP3n+ring_unit*K2P3n;
  const double DZP3n=DP3n+ring_unit*K2P3n;

 
  // Initial values of update values (Up)

  double UpA0=Projptr[0][0][0][1]+ds*K1A0+dz*K2A0;
  double UpA1=Projptr[0][1][0][1]+ds*K1A1+dz*K2A1;
  double UpA2=Projptr[0][2][0][1]+ds*K1A2+dz*K2A2;
  double UpA3=Projptr[0][3][0][1]+ds*K1A3+dz*K2A3;

  double UpA0n=Projptr[1][0][0][1]+ds*K1A0n+dz*K2A0n;
  double UpA1n=Projptr[1][1][0][1]+ds*K1A1n+dz*K2A1n;
  double UpA2n=Projptr[1][2][0][1]+ds*K1A2n+dz*K2A2n;
  double UpA3n=Projptr[1][3][0][1]+ds*K1A3n+dz*K2A3n;

  double UpP0=Projptr[0][0][1][1]+ds*K1P0+dz*K2P0;
  double UpP1=Projptr[0][1][1][1]+ds*K1P1+dz*K2P1;
  double UpP2=Projptr[0][2][1][1]+ds*K1P2+dz*K2P2;
  double UpP3=Projptr[0][3][1][1]+ds*K1P3+dz*K2P3;

  double UpP0n=Projptr[1][0][1][1]+ds*K1P0n+dz*K2P0n;
  double UpP1n=Projptr[1][1][1][1]+ds*K1P1n+dz*K2P1n;
  double UpP2n=Projptr[1][2][1][1]+ds*K1P2n+dz*K2P2n;
  double UpP3n=Projptr[1][3][1][1]+ds*K1P3n+dz*K2P3n;

#else // PIECEWISE_INTERPOLATION


  const double ZplusKorrA0=K2A0;
  const double ZplusKorrA1=K2A1;
  const double ZplusKorrA2=K2A2;
  const double ZplusKorrA3=K2A3;

  const double ZplusKorrA0n=K2A0n;
  const double ZplusKorrA1n=K2A1n;
  const double ZplusKorrA2n=K2A2n;
  const double ZplusKorrA3n=K2A3n;

  const double ZplusKorrP0=K2P0;
  const double ZplusKorrP1=K2P1;
  const double ZplusKorrP2=K2P2;
  const double ZplusKorrP3=K2P3;

  const double ZplusKorrP0n=K2P0n;
  const double ZplusKorrP1n=K2P1n;
  const double ZplusKorrP2n=K2P2n;
  const double ZplusKorrP3n=K2P3n;

  //  V, H, D invariants

  const double VA0a0=+sphi*K1A0;
  const double HA0a0=-cphi*K1A0;
  const double DA0a0=VA0a0+HA0a0;
  const double VA1a0=+sphi*K1A1;
  const double HA1a0=-cphi*K1A1;
  const double DA1a0=VA1a0+HA1a0;
  const double VA2a0=+sphi*K1A2;
  const double HA2a0=-cphi*K1A2;
  const double DA2a0=VA2a0+HA2a0;
  const double VA3a0=+sphi*K1A3;
  const double HA3a0=-cphi*K1A3;
  const double DA3a0=VA3a0+HA3a0;

  const double VA0na0=+sphi*K1A0n;
  const double HA0na0=-cphi*K1A0n;
  const double DA0na0=VA0na0+HA0na0;
  const double VA1na0=+sphi*K1A1n;
  const double HA1na0=-cphi*K1A1n;
  const double DA1na0=VA1na0+HA1na0;
  const double VA2na0=+sphi*K1A2n;
  const double HA2na0=-cphi*K1A2n;
  const double DA2na0=VA2na0+HA2na0;
  const double VA3na0=+sphi*K1A3n;
  const double HA3na0=-cphi*K1A3n;
  const double DA3na0=VA3na0+HA3na0;

  const double VP0a0=+sphi*K1P0;
  const double HP0a0=-cphi*K1P0;
  const double DP0a0=VP0a0+HP0a0;
  const double VP1a0=+sphi*K1P1;
  const double HP1a0=-cphi*K1P1;
  const double DP1a0=VP1a0+HP1a0;
  const double VP2a0=+sphi*K1P2;
  const double HP2a0=-cphi*K1P2;
  const double DP2a0=VP2a0+HP2a0;
  const double VP3a0=+sphi*K1P3;
  const double HP3a0=-cphi*K1P3;
  const double DP3a0=VP3a0+HP3a0;

  const double VP0na0=+sphi*K1P0n;
  const double HP0na0=-cphi*K1P0n;
  const double DP0na0=VP0na0+HP0na0;
  const double VP1na0=+sphi*K1P1n;
  const double HP1na0=-cphi*K1P1n;
  const double DP1na0=VP1na0+HP1na0;
  const double VP2na0=+sphi*K1P2n;
  const double HP2na0=-cphi*K1P2n;
  const double DP2na0=VP2na0+HP2na0;
  const double VP3na0=+sphi*K1P3n;
  const double HP3na0=-cphi*K1P3n;
  const double DP3na0=VP3na0+HP3na0;


  
 const double VA0a1=+sphi*(K1A0+K3A0);
  const double HA0a1=-cphi*(K1A0+K3A0);
  const double DA0a1=VA0a1+HA0a1;
  const double VA1a1=+sphi*(K1A1+K3A1);
  const double HA1a1=-cphi*(K1A1+K3A1);
  const double DA1a1=VA1a1+HA1a1;
  const double VA2a1=+sphi*(K1A2+K3A2);
  const double HA2a1=-cphi*(K1A2+K3A2);
  const double DA2a1=VA2a1+HA2a1;
  const double VA3a1=+sphi*(K1A3+K3A3);
  const double HA3a1=-cphi*(K1A3+K3A3);
  const double DA3a1=VA3a1+HA3a1;

  const double VA0na1=+sphi*(K1A0n+K3A0n);
  const double HA0na1=-cphi*(K1A0n+K3A0n);
  const double DA0na1=VA0na1+HA0na1;
  const double VA1na1=+sphi*(K1A1n+K3A1n);
  const double HA1na1=-cphi*(K1A1n+K3A1n);
  const double DA1na1=VA1na1+HA1na1;
  const double VA2na1=+sphi*(K1A2n+K3A2n);
  const double HA2na1=-cphi*(K1A2n+K3A2n);
  const double DA2na1=VA2na1+HA2na1;
  const double VA3na1=+sphi*(K1A3n+K3A3n);
  const double HA3na1=-cphi*(K1A3n+K3A3n);
  const double DA3na1=VA3na1+HA3na1;

  const double VP0a1=+sphi*(K1P0+K3P0);
  const double HP0a1=-cphi*(K1P0+K3P0);
  const double DP0a1=VP0a1+HP0a1;
  const double VP1a1=+sphi*(K1P1+K3P1);
  const double HP1a1=-cphi*(K1P1+K3P1);
  const double DP1a1=VP1a1+HP1a1;
  const double VP2a1=+sphi*(K1P2+K3P2);
  const double HP2a1=-cphi*(K1P2+K3P2);
  const double DP2a1=VP2a1+HP2a1;
  const double VP3a1=+sphi*(K1P3+K3P3);
  const double HP3a1=-cphi*(K1P3+K3P3);
  const double DP3a1=VP3a1+HP3a1;

  const double VP0na1=+sphi*(K1P0n+K3P0n);
  const double HP0na1=-cphi*(K1P0n+K3P0n);
  const double DP0na1=VP0na1+HP0na1;
  const double VP1na1=+sphi*(K1P1n+K3P1n);
  const double HP1na1=-cphi*(K1P1n+K3P1n);
  const double DP1na1=VP1na1+HP1na1;
  const double VP2na1=+sphi*(K1P2n+K3P2n);
  const double HP2na1=-cphi*(K1P2n+K3P2n);
  const double DP2na1=VP2na1+HP2na1;
  const double VP3na1=+sphi*(K1P3n+K3P3n);
  const double HP3na1=-cphi*(K1P3n+K3P3n);
  const double DP3na1=VP3na1+HP3na1;


  const double VA0=VA0a0*1.5 - VA0a1/2 + 2*dzvert*K2A0;
  const double HA0=HA0a0*1.5 - HA0a1/2 + 2*dzhor*K2A0;
  const double DA0=VA0+HA0;
  const double VZA0=VA0+K2A0;
  const double HZA0=HA0+K2A0;
  const double DZA0=DA0+K2A0;
  const double VA1=VA1a0*1.5 - VA1a1/2 + 2*dzvert*K2A1;
  const double HA1=HA1a0*1.5 - HA1a1/2 + 2*dzhor*K2A1;
  const double DA1=VA1+HA1;
  const double VZA1=VA1+K2A1;
  const double HZA1=HA1+K2A1;
  const double DZA1=DA1+K2A1;
  const double VA2=VA2a0*1.5 - VA2a1/2 + 2*dzvert*K2A2;
  const double HA2=HA2a0*1.5 - HA2a1/2 + 2*dzhor*K2A2;
  const double DA2=VA2+HA2;
  const double VZA2=VA2+K2A2;
  const double HZA2=HA2+K2A2;
  const double DZA2=DA2+K2A2;
  const double VA3=VA3a0*1.5 - VA3a1/2 + 2*dzvert*K2A3;
  const double HA3=HA3a0*1.5 - HA3a1/2 + 2*dzhor*K2A3;
  const double DA3=VA3+HA3;
  const double VZA3=VA3+K2A3;
  const double HZA3=HA3+K2A3;
  const double DZA3=DA3+K2A3;

  const double VA0n=VA0na0*1.5 - VA0na1/2 + 2*dzvert*K2A0n;
  const double HA0n=HA0na0*1.5 - HA0na1/2 + 2*dzhor*K2A0n;
  const double DA0n=VA0n+HA0n;
  const double VZA0n=VA0n+K2A0n;
  const double HZA0n=HA0n+K2A0n;
  const double DZA0n=DA0n+K2A0n;
  const double VA1n=VA1na0*1.5 - VA1na1/2 + 2*dzvert*K2A1n;
  const double HA1n=HA1na0*1.5 - HA1na1/2 + 2*dzhor*K2A1n;
  const double DA1n=VA1n+HA1n;
  const double VZA1n=VA1n+K2A1n;
  const double HZA1n=HA1n+K2A1n;
  const double DZA1n=DA1n+K2A1n;
  const double VA2n=VA2na0*1.5 - VA2na1/2 + 2*dzvert*K2A2n;
  const double HA2n=HA2na0*1.5 - HA2na1/2 + 2*dzhor*K2A2n;
  const double DA2n=VA2n+HA2n;
  const double VZA2n=VA2n+K2A2n;
  const double HZA2n=HA2n+K2A2n;
  const double DZA2n=DA2n+K2A2n;
  const double VA3n=VA3na0*1.5 - VA3na1/2 + 2*dzvert*K2A3n;
  const double HA3n=HA3na0*1.5 - HA3na1/2 + 2*dzhor*K2A3n;
  const double DA3n=VA3n+HA3n;
  const double VZA3n=VA3n+K2A3n;
  const double HZA3n=HA3n+K2A3n;
  const double DZA3n=DA3n+K2A3n;

  const double VP0=VP0a0*1.5 - VP0a1/2 + 2*dzvert*K2P0;
  const double HP0=HP0a0*1.5 - HP0a1/2 + 2*dzhor*K2P0;
  const double DP0=VP0+HP0;
  const double VZP0=VP0+K2P0;
  const double HZP0=HP0+K2P0;
  const double DZP0=DP0+K2P0;
  const double VP1=VP1a0*1.5 - VP1a1/2 + 2*dzvert*K2P1;
  const double HP1=HP1a0*1.5 - HP1a1/2 + 2*dzhor*K2P1;
  const double DP1=VP1+HP1;
  const double VZP1=VP1+K2P1;
  const double HZP1=HP1+K2P1;
  const double DZP1=DP1+K2P1;
  const double VP2=VP2a0*1.5 - VP2a1/2 + 2*dzvert*K2P2;
  const double HP2=HP2a0*1.5 - HP2a1/2 + 2*dzhor*K2P2;
  const double DP2=VP2+HP2;
  const double VZP2=VP2+K2P2;
  const double HZP2=HP2+K2P2;
  const double DZP2=DP2+K2P2;
  const double VP3=VP3a0*1.5 - VP3a1/2 + 2*dzvert*K2P3;
  const double HP3=HP3a0*1.5 - HP3a1/2 + 2*dzhor*K2P3;
  const double DP3=VP3+HP3;
  const double VZP3=VP3+K2P3;
  const double HZP3=HP3+K2P3;
  const double DZP3=DP3+K2P3;

  const double VP0n=VP0na0*1.5 - VP0na1/2 + 2*dzvert*K2P0n;
  const double HP0n=HP0na0*1.5 - HP0na1/2 + 2*dzhor*K2P0n;
  const double DP0n=VP0n+HP0n;
  const double VZP0n=VP0n+K2P0n;
  const double HZP0n=HP0n+K2P0n;
  const double DZP0n=DP0n+K2P0n;
  const double VP1n=VP1na0*1.5 - VP1na1/2 + 2*dzvert*K2P1n;
  const double HP1n=HP1na0*1.5 - HP1na1/2 + 2*dzhor*K2P1n;
  const double DP1n=VP1n+HP1n;
  const double VZP1n=VP1n+K2P1n;
  const double HZP1n=HP1n+K2P1n;
  const double DZP1n=DP1n+K2P1n;
  const double VP2n=VP2na0*1.5 - VP2na1/2 + 2*dzvert*K2P2n;
  const double HP2n=HP2na0*1.5 - HP2na1/2 + 2*dzhor*K2P2n;
  const double DP2n=VP2n+HP2n;
  const double VZP2n=VP2n+K2P2n;
  const double HZP2n=HP2n+K2P2n;
  const double DZP2n=DP2n+K2P2n;
  const double VP3n=VP3na0*1.5 - VP3na1/2 + 2*dzvert*K2P3n;
  const double HP3n=HP3na0*1.5 - HP3na1/2 + 2*dzhor*K2P3n;
  const double DP3n=VP3n+HP3n;
  const double VZP3n=VP3n+K2P3n;
  const double HZP3n=HP3n+K2P3n;
  const double DZP3n=DP3n+K2P3n;

 
  // Initial values of update values (Up)
  
  double A0a0  = Projptr[0][0][0][1]+ds*K1A0;
  double A0a1  = Projptr[0][0][0][3]+ds*(K1A0+K3A0);
  double A2a0  = Projptr[0][2][0][1]+ds*K1A2;
  double A2a1  = Projptr[0][2][0][3]+ds*(K1A2+K3A2);
  
  double P0na0  = Projptr[1][0][1][1]+ds*K1P0n;
  double P0na1  = Projptr[1][0][1][3]+ds*(K1P0n+K3P0n);
  double P2na0  = Projptr[1][2][1][1]+ds*K1P2n;
  double P2na1  = Projptr[1][2][1][3]+ds*(K1P2n+K3P2n);
  
  double A0na0  = Projptr[1][0][0][1]+ds*K1A0n;
  double A0na1  = Projptr[1][0][0][3]+ds*(K1A0n+K3A0n);
  double A2na0  = Projptr[1][2][0][1]+ds*K1A2n;
  double A2na1  = Projptr[1][2][0][3]+ds*(K1A2n+K3A2n);
  
  double P0a0  = Projptr[0][0][1][1]+ds*K1P0;
  double P0a1  = Projptr[0][0][1][3]+ds*(K1P0+K3P0);
  double P2a0  = Projptr[0][2][1][1]+ds*K1P2;
  double P2a1  = Projptr[0][2][1][3]+ds*(K1P2+K3P2);
  
  double A1a0  = Projptr[0][1][0][1]+ds*K1A1;
  double A1a1  = Projptr[0][1][0][3]+ds*(K1A1+K3A1);
  double A3a0  = Projptr[0][3][0][1]+ds*K1A3;
  double A3a1  = Projptr[0][3][0][3]+ds*(K1A3+K3A3);
  
  double P1na0  = Projptr[1][1][1][1]+ds*K1P1n;
  double P1na1  = Projptr[1][1][1][3]+ds*(K1P1n+K3P1n);
  double P3na0  = Projptr[1][3][1][1]+ds*K1P3n;
  double P3na1  = Projptr[1][3][1][3]+ds*(K1P3n+K3P3n);
  
  double A1na0  = Projptr[1][1][0][1]+ds*K1A1n;
  double A1na1  = Projptr[1][1][0][3]+ds*(K1A1n+K3A1n);
  double A3na0  = Projptr[1][3][0][1]+ds*K1A3n;
  double A3na1  = Projptr[1][3][0][3]+ds*(K1A3n+K3A3n);
  
  double P1a0  = Projptr[0][1][1][1]+ds*K1P1;
  double P1a1  = Projptr[0][1][1][3]+ds*(K1P1+K3P1);
  double P3a0  = Projptr[0][3][1][1]+ds*K1P3;
  double P3a1  = Projptr[0][3][1][3]+ds*(K1P3+K3P3);

  double UpA0 = 2*(Projptr[0][0][0][1]+ds*K1A0+dz*K2A0) - (A0a0 + A0a1)/2;
  double UpA1 = 2*(Projptr[0][1][0][1]+ds*K1A1+dz*K2A1) - (A1a0 + A1a1)/2;
  double UpA2 = 2*(Projptr[0][2][0][1]+ds*K1A2+dz*K2A2) - (A2a0 + A2a1)/2;
  double UpA3 = 2*(Projptr[0][3][0][1]+ds*K1A3+dz*K2A3) - (A3a0 + A3a1)/2;

  double UpA0n = 2*(Projptr[1][0][0][1]+ds*K1A0n+dz*K2A0n) - (A0na0 + A0na1)/2;
  double UpA1n = 2*(Projptr[1][1][0][1]+ds*K1A1n+dz*K2A1n) - (A1na0 + A1na1)/2;
  double UpA2n = 2*(Projptr[1][2][0][1]+ds*K1A2n+dz*K2A2n) - (A2na0 + A2na1)/2;
  double UpA3n = 2*(Projptr[1][3][0][1]+ds*K1A3n+dz*K2A3n) - (A3na0 + A3na1)/2;

  double UpP0 = 2*(Projptr[0][0][1][1]+ds*K1P0+dz*K2P0) - (P0a0 + P0a1)/2;
  double UpP1 = 2*(Projptr[0][1][1][1]+ds*K1P1+dz*K2P1) - (P1a0 + P1a1)/2;
  double UpP2 = 2*(Projptr[0][2][1][1]+ds*K1P2+dz*K2P2) - (P2a0 + P2a1)/2;
  double UpP3 = 2*(Projptr[0][3][1][1]+ds*K1P3+dz*K2P3) - (P3a0 + P3a1)/2;

  double UpP0n = 2*(Projptr[1][0][1][1]+ds*K1P0n+dz*K2P0n) - (P0na0 + P0na1)/2;
  double UpP1n = 2*(Projptr[1][1][1][1]+ds*K1P1n+dz*K2P1n) - (P1na0 + P1na1)/2;
  double UpP2n = 2*(Projptr[1][2][1][1]+ds*K1P2n+dz*K2P2n) - (P2na0 + P2na1)/2;
  double UpP3n = 2*(Projptr[1][3][1][1]+ds*K1P3n+dz*K2P3n) - (P3na0 + P3na1)/2;

#endif

  // some extra start values

  // Find symmetric value in Z by 'mirroring' it around the centre z of the LORs:
  // Z+Q = 2*centre_of_LOR_in_image_coordinates
  // Note that we are backprojecting ring0 and ring0+1, so we mirror around ring0+0.5
  // original  Q = (int)  (4*ring0+2*delta+2-Z + 0.5);
  // CL&KT 21/12/99 added axial_pos_to_z_offset and num_planes_per_physical_ring
  {
    // first compute it as floating point (although it has to be an int really)
    const float Qf = (2*num_planes_per_axial_pos*(ring0 + 0.5) 
                      + num_planes_per_physical_ring*delta
	              + 2*axial_pos_to_z_offset
	              - Z); 
    // now use rounding to be safe
    Q = (int)floor(Qf + 0.5);
    assert(fabs(Q-Qf) < 10E-4);
  }

  dzdiag=dzvert+dzhor;
  dsdiag=-cphi+sphi;


  /* KT 13/05/98 changed loop condition, originally a combination of  
     while (X>X2||Y<Y2||Z<Z2) {
     // code which increments Y,Z or decrements X
     if (X<X2||Y>Y2||Z>Z2)
     }
     this had problems for nearly horizontal or vertical lines.
     For example, while Y==Y2+1, some extra iterations had to be done
     on X.
     Now I break out of the while when the voxel goes out of the cylinder,
     which means I don't have to use X2,Y2,Z2 anymore
  */
  do 
    {
      assert(ds >= 0);
      assert(ds <= 1);
      assert(dz >= 0);
      assert(dz <= ring_unit+epsilon);

      // Update voxel values for this X,Y,Z
      // For 1 given (X,Y)-position, there are always 2 voxels along the z-axis 
      // in this beam. 
      const int Zplus=Z+1;
      const int Qmin=Q-1;

      
#if PIECEWISE_INTERPOLATION
      
      const double twodsdz=2*ds*dz;
      const double twodsdz2=2*ds*(dz+0.5);
      // KT 16/06/98 changed check ds!=0 to fabs(ds)>epsilon for better rounding control
      const bool do_s_symmetry = (s!=0 || fabs(ds) > epsilon);


      if (Z>=minplane&&Z<=maxplane) {
        image[Z][Y][X]+=(dz <= 0.25) ? A0a0 : UpA0 +twodsdz*K3A0;
        image[Z][X][-Y]+=(dz <= 0.25) ? A2a0 : UpA2 +twodsdz*K3A2;
        image[Z][X][Y]+=(dz <= 0.25) ? A1na0 : UpA1n +twodsdz*K3A1n;
        image[Z][Y][-X]+=(dz <= 0.25) ? A3na0 : UpA3n +twodsdz*K3A3n;
        if (do_s_symmetry)
	  {
	    image[Z][-X][-Y]+=(dz <= 0.25) ? P1a0 : UpP1 +twodsdz*K3P1;
	    image[Z][-Y][X]+=(dz <= 0.25) ? P3a0 : UpP3 +twodsdz*K3P3;
	    image[Z][-Y][-X]+=(dz <= 0.25) ? P0na0 : UpP0n +twodsdz*K3P0n;
	    image[Z][-X][Y]+=(dz <= 0.25) ? P2na0 : UpP2n +twodsdz*K3P2n;
	  }
      }
      if (Zplus>=minplane&&Zplus<=maxplane) {
        image[Zplus][Y][X]+=(dz >= 0.25) ? A0a1 : UpA0 +twodsdz2*K3A0+ZplusKorrA0;
        image[Zplus][X][-Y]+=(dz >= 0.25) ? A2a1 : UpA2 +twodsdz2*K3A2+ZplusKorrA2;
        image[Zplus][X][Y]+=(dz >= 0.25) ? A1na1 : UpA1n +twodsdz2*K3A1n+ZplusKorrA1n;
        image[Zplus][Y][-X]+=(dz >= 0.25) ? A3na1 : UpA3n +twodsdz2*K3A3n+ZplusKorrA3n;
        if (do_s_symmetry)
	  {
	    image[Zplus][-X][-Y]+=(dz >= 0.25) ? P1a1 : UpP1 +twodsdz2*K3P1+ZplusKorrP1;
	    image[Zplus][-Y][X]+=(dz >= 0.25) ? P3a1 : UpP3 +twodsdz2*K3P3+ZplusKorrP3;
	    image[Zplus][-Y][-X]+=(dz >= 0.25) ? P0na1 : UpP0n +twodsdz2*K3P0n+ZplusKorrP0n;
	    image[Zplus][-X][Y]+=(dz >= 0.25) ? P2na1 : UpP2n +twodsdz2*K3P2n+ZplusKorrP2n;
	  }
      }
      if (Q>=minplane&&Q<=maxplane) {
        image[Q][Y][-X]+=(dz <= 0.25) ? A3a0 : UpA3 +twodsdz*K3A3;
        image[Q][Y][X]+=(dz <= 0.25) ? A0na0 : UpA0n +twodsdz*K3A0n;
        image[Q][X][-Y]+=(dz <= 0.25) ? A2na0 : UpA2n +twodsdz*K3A2n;
        image[Q][X][Y]+=(dz <= 0.25) ? A1a0 : UpA1 +twodsdz*K3A1;
        if (do_s_symmetry)
	  {
	    image[Q][-Y][-X]+=(dz <= 0.25) ? P0a0 : UpP0 +twodsdz*K3P0;
	    image[Q][-X][Y]+=(dz <= 0.25) ? P2a0 : UpP2 +twodsdz*K3P2;
	    image[Q][-X][-Y]+=(dz <= 0.25) ? P1na0 : UpP1n +twodsdz*K3P1n;
	    image[Q][-Y][X]+=(dz <= 0.25) ? P3na0 : UpP3n +twodsdz*K3P3n;
	  }
      }
      if (Qmin>=minplane&&Qmin<=maxplane) {
        image[Qmin][Y][-X]+=(dz >= 0.25) ? A3a1 : UpA3 +twodsdz2*K3A3+ZplusKorrA3;
        image[Qmin][Y][X]+=(dz >= 0.25) ? A0na1 : UpA0n +twodsdz2*K3A0n+ZplusKorrA0n;
        image[Qmin][X][-Y]+=(dz >= 0.25) ? A2na1 : UpA2n +twodsdz2*K3A2n+ZplusKorrA2n;
        image[Qmin][X][Y]+=(dz >= 0.25) ? A1a1 : UpA1 +twodsdz2*K3A1+ZplusKorrA1;
        if (do_s_symmetry)
	  {
	    image[Qmin][-Y][-X]+=(dz >= 0.25) ? P0a1 : UpP0 +twodsdz2*K3P0+ZplusKorrP0;
	    image[Qmin][-X][Y]+=(dz >= 0.25) ? P2a1 : UpP2 +twodsdz2*K3P2+ZplusKorrP2;
	    image[Qmin][-X][-Y]+=(dz >= 0.25) ? P1na1 : UpP1n +twodsdz2*K3P1n+ZplusKorrP1n;
	    image[Qmin][-Y][X]+=(dz >= 0.25) ? P3na1 : UpP3n +twodsdz2*K3P3n+ZplusKorrP3n;
	  }
      }
#else // !PIECEWISE_INTERPOLATION
#ifdef ALTERNATIVE 
      const double dsdz=ds*dz;
      const double dsdz2=ds*(dz+0.5);
      // KT 16/06/98 changed check ds!=0 to fabs(ds)>epsilon for better rounding control
      const bool do_s_symmetry = (s!=0 || fabs(ds) > epsilon);

      if (Z>=minplane&&Z<=maxplane) {
        image[Z][Y][X]+=UpA0+dsdz*K3A0;
        image[Z][X][-Y]+=UpA2+dsdz*K3A2;
        image[Z][X][Y]+=UpA1n+dsdz*K3A1n;
        image[Z][Y][-X]+=UpA3n+dsdz*K3A3n;
        if (do_s_symmetry)
	  {
	    image[Z][-X][-Y]+=UpP1+dsdz*K3P1;
	    image[Z][-Y][X]+=UpP3+dsdz*K3P3;
	    image[Z][-Y][-X]+=UpP0n+dsdz*K3P0n;
	    image[Z][-X][Y]+=UpP2n+dsdz*K3P2n;
	  }
      }
      if (Zplus>=minplane&&Zplus<=maxplane) {
        image[Zplus][Y][X]+=UpA0+dsdz2*K3A0+ZplusKorrA0;
        image[Zplus][X][-Y]+=UpA2+dsdz2*K3A2+ZplusKorrA2;
        image[Zplus][X][Y]+=UpA1n+dsdz2*K3A1n+ZplusKorrA1n;
        image[Zplus][Y][-X]+=UpA3n+dsdz2*K3A3n+ZplusKorrA3n;
        if (do_s_symmetry)
	  {
	    image[Zplus][-X][-Y]+=UpP1+dsdz2*K3P1+ZplusKorrP1;
	    image[Zplus][-Y][X]+=UpP3+dsdz2*K3P3+ZplusKorrP3;
	    image[Zplus][-Y][-X]+=UpP0n+dsdz2*K3P0n+ZplusKorrP0n;
	    image[Zplus][-X][Y]+=UpP2n+dsdz2*K3P2n+ZplusKorrP2n;
	  }
      }
      if (Q>=minplane&&Q<=maxplane) {
        image[Q][Y][-X]+=UpA3+dsdz*K3A3;
        image[Q][Y][X]+=UpA0n+dsdz*K3A0n;
        image[Q][X][-Y]+=UpA2n+dsdz*K3A2n;
        image[Q][X][Y]+=UpA1+dsdz*K3A1;
        if (do_s_symmetry)
	  {
	    image[Q][-Y][-X]+=UpP0+dsdz*K3P0;
	    image[Q][-X][Y]+=UpP2+dsdz*K3P2;
	    image[Q][-X][-Y]+=UpP1n+dsdz*K3P1n;
	    image[Q][-Y][X]+=UpP3n+dsdz*K3P3n;
	  }
      }
      if (Qmin>=minplane&&Qmin<=maxplane) {
        image[Qmin][Y][-X]+=UpA3+dsdz2*K3A3+ZplusKorrA3;
        image[Qmin][Y][X]+=UpA0n+dsdz2*K3A0n+ZplusKorrA0n;
        image[Qmin][X][-Y]+=UpA2n+dsdz2*K3A2n+ZplusKorrA2n;
        image[Qmin][X][Y]+=UpA1+dsdz2*K3A1+ZplusKorrA1;
        if (do_s_symmetry)
	  {
	    image[Qmin][-Y][-X]+=UpP0+dsdz2*K3P0+ZplusKorrP0;
	    image[Qmin][-X][Y]+=UpP2+dsdz2*K3P2+ZplusKorrP2;
	    image[Qmin][-X][-Y]+=UpP1n+dsdz2*K3P1n+ZplusKorrP1n;
	    image[Qmin][-Y][X]+=UpP3n+dsdz2*K3P3n+ZplusKorrP3n;
	  }
      }
 #else // ALTERNATIVE
      double TMP1,TMP2;
      TMP1=ds*K3A0;
      TMP2=UpA0+dz*TMP1;
      if (Z>=minplane&&Z<=maxplane)
	image[Z][Y][X]+=TMP2;

      
          //CL SPAN 15/09/98 Add one more condition in order to skip the statement in case of span data
          //as there is only one voxel in the beam in slice unit
       if (Zplus>=minplane&&Zplus<=maxplane && ring_unit==0.5)
	image[Zplus][Y][X]+=TMP2+ring_unit*TMP1+ZplusKorrA0;
       
      TMP1=ds*K3A1;
      TMP2=UpA1+dz*TMP1;
      if (Q>=minplane&&Q<=maxplane)
	image[Q][X][Y]+=TMP2;
          //CL SPAN 15/09/98 Add one more condition in order to skip the statement in case of span data
          //as there is only one voxel in the beam in slice unit
       if (Qmin>=minplane&&Qmin<=maxplane && ring_unit==0.5)
	image[Qmin][X][Y]+=TMP2+ring_unit*TMP1+ZplusKorrA1;
  
       
      TMP1=ds*K3A2;
      TMP2=UpA2+dz*TMP1;
      if (Z>=minplane&&Z<=maxplane)
	image[Z][X][-Y]+=TMP2;
          //CL SPAN 15/09/98 Add one more condition in order to skip the statement in case of span data
          //as there is only one voxel in the beam in slice unit
       if (Zplus>=minplane&&Zplus<=maxplane && ring_unit==0.5)
	image[Zplus][X][-Y]+=TMP2+ring_unit*TMP1+ZplusKorrA2;
      TMP1=ds*K3A3;
      TMP2=UpA3+dz*TMP1;
      if (Q>=minplane&&Q<=maxplane)
	image[Q][Y][-X]+=TMP2;
          //CL SPAN 15/09/98 Add one more condition in order to skip the statement in case of span data
          //as there is only one voxel in the beam in slice unit
       if (Qmin>=minplane&&Qmin<=maxplane && ring_unit==0.5)
	image[Qmin][Y][-X]+=TMP2+ring_unit*TMP1+ZplusKorrA3;
  
      TMP1=ds*K3A0n;
      TMP2=UpA0n+dz*TMP1;
      if (Q>=minplane&&Q<=maxplane)
	image[Q][Y][X]+=TMP2;
          //CL SPAN 15/09/98 Add one more condition in order to skip the statement in case of span data
          //as there is only one voxel in the beam in slice unit
       if (Qmin>=minplane&&Qmin<=maxplane && ring_unit==0.5)
	image[Qmin][Y][X]+=TMP2+ring_unit*TMP1+ZplusKorrA0n;
      TMP1=ds*K3A1n;
      TMP2=UpA1n+dz*TMP1;
      if (Z>=minplane&&Z<=maxplane)
	image[Z][X][Y]+=TMP2;
          //CL SPAN 15/09/98 Add one more condition in order to skip the statement in case of span data
          //as there is only one voxel in the beam in slice unit
       if (Zplus>=minplane&&Zplus<=maxplane && ring_unit==0.5)
	image[Zplus][X][Y]+=TMP2+ring_unit*TMP1+ZplusKorrA1n;
      TMP1=ds*K3A2n;
      TMP2=UpA2n+dz*TMP1;
      if (Q>=minplane&&Q<=maxplane)
	image[Q][X][-Y]+=TMP2;
          //CL SPAN 15/09/98 Add one more condition in order to skip the statement in case of span data
          //as there is only one voxel in the beam in slice unit
       if (Qmin>=minplane&&Qmin<=maxplane && ring_unit==0.5)
	image[Qmin][X][-Y]+=TMP2+ring_unit*TMP1+ZplusKorrA2n;
      TMP1=ds*K3A3n;
      TMP2=UpA3n+dz*TMP1;
      if (Z>=minplane&&Z<=maxplane)
	image[Z][Y][-X]+=TMP2;
          //CL SPAN 15/09/98 Add one more condition in order to skip the statement in case of span data
          //as there is only one voxel in the beam in slice unit
       if (Zplus>=minplane&&Zplus<=maxplane && ring_unit==0.5)
	image[Zplus][Y][-X]+=TMP2+ring_unit*TMP1+ZplusKorrA3n;

      // KT 16/06/98 changed check ds!=0 to fabs(ds)>epsilon for better rounding control
      if (s!=0 || fabs(ds) > epsilon)
	{
	  TMP1=ds*K3P0;
	  TMP2=UpP0+dz*TMP1;
	  if (Q>=minplane&&Q<=maxplane)
            image[Q][-Y][-X]+=TMP2;
          //CL SPAN 15/09/98 Add one more condition in order to skip the statement in case of span data
          //as there is only one voxel in the beam in slice unit
 	  if (Qmin>=minplane&&Qmin<=maxplane && ring_unit==0.5)
            image[Qmin][-Y][-X]+=TMP2+ring_unit*TMP1+ZplusKorrP0;
	  TMP1=ds*K3P1;
	  TMP2=UpP1+dz*TMP1;
	  if (Z>=minplane&&Z<=maxplane)
            image[Z][-X][-Y]+=TMP2;
          //CL SPAN 15/09/98 Add one more condition in order to skip the statement in case of span data
          //as there is only one voxel in the beam in slice unit
 	  if (Zplus>=minplane&&Zplus<=maxplane && ring_unit==0.5)
            image[Zplus][-X][-Y]+=TMP2+ring_unit*TMP1+ZplusKorrP1;
	  TMP1=ds*K3P2;
	  TMP2=UpP2+dz*TMP1;
	  if (Q>=minplane&&Q<=maxplane)
            image[Q][-X][Y]+=TMP2;
          //CL SPAN 15/09/98 Add one more condition in order to skip the statement in case of span data
          //as there is only one voxel in the beam in slice unit
 	  if (Qmin>=minplane&&Qmin<=maxplane && ring_unit==0.5)
            image[Qmin][-X][Y]+=TMP2+ring_unit*TMP1+ZplusKorrP2;
	  TMP1=ds*K3P3;
	  TMP2=UpP3+dz*TMP1;
	  if (Z>=minplane&&Z<=maxplane)
            image[Z][-Y][X]+=TMP2;
          //CL SPAN 15/09/98 Add one more condition in order to skip the statement in case of span data
          //as there is only one voxel in the beam in slice unit
 	  if (Zplus>=minplane&&Zplus<=maxplane && ring_unit==0.5)
            image[Zplus][-Y][X]+=TMP2+ring_unit*TMP1+ZplusKorrP3;
  
	  TMP1=ds*K3P0n;
	  TMP2=UpP0n+dz*TMP1;
	  if (Z>=minplane&&Z<=maxplane)
            image[Z][-Y][-X]+=TMP2;
          //CL SPAN 15/09/98 Add one more condition in order to skip the statement in case of span data
          //as there is only one voxel in the beam in slice unit
 	  if (Zplus>=minplane&&Zplus<=maxplane && ring_unit==0.5)
            image[Zplus][-Y][-X]+=TMP2+ring_unit*TMP1+ZplusKorrP0n;
	  TMP1=ds*K3P1n;
	  TMP2=UpP1n+dz*TMP1;
	  if (Q>=minplane&&Q<=maxplane)
            image[Q][-X][-Y]+=TMP2;
          //CL SPAN 15/09/98 Add one more condition in order to skip the statement in case of span data
          //as there is only one voxel in the beam in slice unit
 	  if (Qmin>=minplane&&Qmin<=maxplane && ring_unit==0.5)
            image[Qmin][-X][-Y]+=TMP2+ring_unit*TMP1+ZplusKorrP1n;
	  TMP1=ds*K3P2n;
	  TMP2=UpP2n+dz*TMP1;
	  if (Z>=minplane&&Z<=maxplane)
            image[Z][-X][Y]+=TMP2;
          //CL SPAN 15/09/98 Add one more condition in order to skip the statement in case of span data
          //as there is only one voxel in the beam in slice unit
 	  if (Zplus>=minplane&&Zplus<=maxplane && ring_unit==0.5)
            image[Zplus][-X][Y]+=TMP2+ring_unit*TMP1+ZplusKorrP2n;
	  TMP1=ds*K3P3n;
	  TMP2=UpP3n+dz*TMP1;
	  if (Q>=minplane&&Q<=maxplane)
            image[Q][-Y][X]+=TMP2;
          //CL SPAN 15/09/98 Add one more condition in order to skip the statement in case of span data
          //as there is only one voxel in the beam in slice unit
 	  if (Qmin>=minplane&&Qmin<=maxplane && ring_unit==0.5)
            image[Qmin][-Y][X]+=TMP2+ring_unit*TMP1+ZplusKorrP3n;
	}
    
#endif // ALTERNATIVE 
#endif //PIECEWISE_INTERPOLATION

      //  Search for next pixel in the beam

      if (ds>=cphi) {
	/* horizontal*/
	X-=1;
	ds-=cphi;
	dz+=dzhor;
	if (dz<epsilon) {     /* increment Z */
	  Z++; Q--; dz+=ring_unit;
	  UpA0+=HZA0;
	  UpA1+=HZA1;
	  UpA2+=HZA2;
	  UpA3+=HZA3;
	  UpA0n+=HZA0n;
	  UpA1n+=HZA1n;
	  UpA2n+=HZA2n;
	  UpA3n+=HZA3n;
	  UpP0+=HZP0;
	  UpP1+=HZP1;
	  UpP2+=HZP2;
	  UpP3+=HZP3;
	  UpP0n+=HZP0n;
	  UpP1n+=HZP1n;
	  UpP2n+=HZP2n;
	  UpP3n+=HZP3n;
#ifdef MOREZ
          while (dz<epsilon)
          {
	  Z++; Q--; dz+=ring_unit;
	  UpA0+=HZA0-HA0;
	  UpA1+=HZA1-HA1;
	  UpA2+=HZA2-HA2;
	  UpA3+=HZA3-HA3;
	  UpA0n+=HZA0n-HA0n;
	  UpA1n+=HZA1n-HA1n;
	  UpA2n+=HZA2n-HA2n;
	  UpA3n+=HZA3n-HA3n;
	  UpP0+=HZP0-HP0;
	  UpP1+=HZP1-HP1;
	  UpP2+=HZP2-HP2;
	  UpP3+=HZP3-HP3;
	  UpP0n+=HZP0n-HP0n;
	  UpP1n+=HZP1n-HP1n;
	  UpP2n+=HZP2n-HP2n;
	  UpP3n+=HZP3n-HP3n;
          }
#endif
	}
    
	else {       /* Z does not change */
	  UpA0+=HA0;
	  UpA1+=HA1;
	  UpA2+=HA2;
	  UpA3+=HA3;
	  UpA0n+=HA0n;
	  UpA1n+=HA1n;
	  UpA2n+=HA2n;
	  UpA3n+=HA3n;
	  UpP0+=HP0;
	  UpP1+=HP1;
	  UpP2+=HP2;
	  UpP3+=HP3;
	  UpP0n+=HP0n;
	  UpP1n+=HP1n;
	  UpP2n+=HP2n;
	  UpP3n+=HP3n;
	}
#if PIECEWISE_INTERPOLATION
	A0a0+=HA0a0;
	A1a0+=HA1a0;
	A2a0+=HA2a0;
	A3a0+=HA3a0;
	A0na0+=HA0na0;
	A1na0+=HA1na0;
	A2na0+=HA2na0;
	A3na0+=HA3na0;
	P0a0+=HP0a0;
	P1a0+=HP1a0;
	P2a0+=HP2a0;
	P3a0+=HP3a0;
	P0na0+=HP0na0;
	P1na0+=HP1na0;
	P2na0+=HP2na0;
	P3na0+=HP3na0;

	A0a1+=HA0a1;
	A1a1+=HA1a1;
	A2a1+=HA2a1;
	A3a1+=HA3a1;
	A0na1+=HA0na1;
	A1na1+=HA1na1;
	A2na1+=HA2na1;
	A3na1+=HA3na1;
	P0a1+=HP0a1;
	P1a1+=HP1a1;
	P2a1+=HP2a1;
	P3a1+=HP3a1;
	P0na1+=HP0na1;
	P1na1+=HP1na1;
	P2na1+=HP2na1;
	P3na1+=HP3na1;
#endif // PIECEWISE_INTERPOLATION                
      }
      // KT 14/05/98 use < instead of <= (see formula 6.17 in Egger's thesis)
      else if (ds<1-sphi) {
	/*  vertical*/
	Y+=1;
	ds+=sphi;
	dz+=dzvert;
	if (dz<epsilon) {       /* increment Z */
	  Z++; Q--; dz+=ring_unit;    
	  UpA0+=VZA0;
	  UpA1+=VZA1;
	  UpA2+=VZA2;
	  UpA3+=VZA3;
	  UpA0n+=VZA0n;
	  UpA1n+=VZA1n;
	  UpA2n+=VZA2n;
	  UpA3n+=VZA3n;
	  UpP0+=VZP0;
	  UpP1+=VZP1;
	  UpP2+=VZP2;
	  UpP3+=VZP3;
	  UpP0n+=VZP0n;
	  UpP1n+=VZP1n;
	  UpP2n+=VZP2n;
	  UpP3n+=VZP3n;
#ifdef MOREZ
          while (dz<epsilon)
          {
	  Z++; Q--; dz+=ring_unit;    
	  UpA0+=VZA0-VA0;
	  UpA1+=VZA1-VA1;
	  UpA2+=VZA2-VA2;
	  UpA3+=VZA3-VA3;
	  UpA0n+=VZA0n-VA0n;
	  UpA1n+=VZA1n-VA1n;
	  UpA2n+=VZA2n-VA2n;
	  UpA3n+=VZA3n-VA3n;
	  UpP0+=VZP0-VP0;
	  UpP1+=VZP1-VP1;
	  UpP2+=VZP2-VP2;
	  UpP3+=VZP3-VP3;
	  UpP0n+=VZP0n-VP0n;
	  UpP1n+=VZP1n-VP1n;
	  UpP2n+=VZP2n-VP2n;
	  UpP3n+=VZP3n-VP3n;
          }
#endif
	}
	else {              /* Wenn Z gleich bleibt */
	  UpA0+=VA0;
	  UpA1+=VA1;
	  UpA2+=VA2;
	  UpA3+=VA3;
	  UpA0n+=VA0n;
	  UpA1n+=VA1n;
	  UpA2n+=VA2n;
	  UpA3n+=VA3n;
	  UpP0+=VP0;
	  UpP1+=VP1;
	  UpP2+=VP2;
	  UpP3+=VP3;
	  UpP0n+=VP0n;
	  UpP1n+=VP1n;
	  UpP2n+=VP2n;
	  UpP3n+=VP3n;
	}
#if PIECEWISE_INTERPOLATION
	A0a0+=VA0a0;
	A1a0+=VA1a0;
	A2a0+=VA2a0;
	A3a0+=VA3a0;
	A0na0+=VA0na0;
	A1na0+=VA1na0;
	A2na0+=VA2na0;
	A3na0+=VA3na0;
	P0a0+=VP0a0;
	P1a0+=VP1a0;
	P2a0+=VP2a0;
	P3a0+=VP3a0;
	P0na0+=VP0na0;
	P1na0+=VP1na0;
	P2na0+=VP2na0;
	P3na0+=VP3na0;

	A0a1+=VA0a1;
	A1a1+=VA1a1;
	A2a1+=VA2a1;
	A3a1+=VA3a1;
	A0na1+=VA0na1;
	A1na1+=VA1na1;
	A2na1+=VA2na1;
	A3na1+=VA3na1;
	P0a1+=VP0a1;
	P1a1+=VP1a1;
	P2a1+=VP2a1;
	P3a1+=VP3a1;
	P0na1+=VP0na1;
	P1na1+=VP1na1;
	P2na1+=VP2na1;
	P3na1+=VP3na1;
#endif // PIECEWISE_INTERPOLATION                  
      }
      else {
	/*  diagonal*/
	X-=1;
	Y+=1;
	ds+=dsdiag;
	dz+=dzdiag;
	if (dz<epsilon) {      /* increment Z */
	  Z++; Q--; dz+=ring_unit;
	  UpA0+=DZA0;
	  UpA1+=DZA1;
	  UpA2+=DZA2;
	  UpA3+=DZA3;
	  UpA0n+=DZA0n;
	  UpA1n+=DZA1n;
	  UpA2n+=DZA2n;
	  UpA3n+=DZA3n;
	  UpP0+=DZP0;
	  UpP1+=DZP1;
	  UpP2+=DZP2;
	  UpP3+=DZP3;
	  UpP0n+=DZP0n;
	  UpP1n+=DZP1n;
	  UpP2n+=DZP2n;
	  UpP3n+=DZP3n;
#ifdef MOREZ
          while (dz<epsilon)
          {
	  Z++; Q--; dz+=ring_unit;
	  UpA0+=DZA0-DA0;
	  UpA1+=DZA1-DA1;
	  UpA2+=DZA2-DA2;
	  UpA3+=DZA3-DA3;
	  UpA0n+=DZA0n-DA0n;
	  UpA1n+=DZA1n-DA1n;
	  UpA2n+=DZA2n-DA2n;
	  UpA3n+=DZA3n-DA3n;
	  UpP0+=DZP0-DP0;
	  UpP1+=DZP1-DP1;
	  UpP2+=DZP2-DP2;
	  UpP3+=DZP3-DP3;
	  UpP0n+=DZP0n-DP0n;
	  UpP1n+=DZP1n-DP1n;
	  UpP2n+=DZP2n-DP2n;
	  UpP3n+=DZP3n-DP3n;
          }
#endif
	}
    
	else {       /* Z does not change */
	  UpA0+=DA0;
	  UpA1+=DA1;
	  UpA2+=DA2;
	  UpA3+=DA3;
	  UpA0n+=DA0n;
	  UpA1n+=DA1n;
	  UpA2n+=DA2n;
	  UpA3n+=DA3n;
	  UpP0+=DP0;
	  UpP1+=DP1;
	  UpP2+=DP2;
	  UpP3+=DP3;
	  UpP0n+=DP0n;
	  UpP1n+=DP1n;
	  UpP2n+=DP2n;
	  UpP3n+=DP3n;
	}
#if PIECEWISE_INTERPOLATION
	A0a0+=DA0a0;
	A1a0+=DA1a0;
	A2a0+=DA2a0;
	A3a0+=DA3a0;
	A0na0+=DA0na0;
	A1na0+=DA1na0;
	A2na0+=DA2na0;
	A3na0+=DA3na0;
	P0a0+=DP0a0;
	P1a0+=DP1a0;
	P2a0+=DP2a0;
	P3a0+=DP3a0;
	P0na0+=DP0na0;
	P1na0+=DP1na0;
	P2na0+=DP2na0;
	P3na0+=DP3na0;

	A0a1+=DA0a1;
	A1a1+=DA1a1;
	A2a1+=DA2a1;
	A3a1+=DA3a1;
	A0na1+=DA0na1;
	A1na1+=DA1na1;
	A2na1+=DA2na1;
	A3na1+=DA3na1;
	P0a1+=DP0a1;
	P1a1+=DP1a1;
	P2a1+=DP2a1;
	P3a1+=DP3a1;
	P0na1+=DP0na1;
	P1na1+=DP1na1;
	P2na1+=DP2na1;
	P3na1+=DP3na1;
#endif // PIECEWISE_INTERPOLATION       
                
      }
      check_values(proj_data_info_ptr, 
		   delta, cphi, sphi, s, ring0,
		   X, Y, Z,
		   ds, dz, 
		   num_planes_per_axial_pos,
		   axial_pos_to_z_offset);

    }
  while ((X*X + Y*Y <= image_rad*image_rad) && (Z<=maxplane || Q>=minplane));

}
  

/****************************************************************************************/


static Succeeded
find_start_values(const ProjDataInfoCylindricalArcCorr* proj_data_info_ptr, 
                              const float delta, const double cphi, const double sphi, 
                              const int s, const int ring0,
                              const float image_rad, const double d_sl,
                              int&X1, int&Y1, int& Z1,
                              double& ds, double& dz, double& dzhor, double& dzvert,			      
                              const float num_planes_per_axial_pos,
			      const float axial_pos_to_z_offset)
{
  // use notations from Egger's thesis

     
  const double d_p = proj_data_info_ptr->get_tangential_sampling();
  // d_xy = image.get_voxel_size().x, but we can use the bin_size here as 
  // the routines.work only when these 2 are equal
  const double d_xy = proj_data_info_ptr->get_tangential_sampling();
    
  const double R2 =square(proj_data_info_ptr->get_ring_radius());
  /* Radius of scanner squared in Pixel^2 */
  const double R2p = R2 / d_xy / d_xy;
  
  // TODO REMOVE ASSUMPTION
  const int num_planes_per_physical_ring = 2;
  assert(fabs(d_sl * num_planes_per_physical_ring/proj_data_info_ptr->get_ring_spacing() -1) < 10E-4);
  
  /* KT 16/06/98 
     This code should select a pixel inside the FOV.
     I tried to set rpix=image_rad, and
      X1 = (int)(X1f<0 ? ceil(X1f) : floor(X1f));
      Y1 = (int)(Y1f<0 ? ceil(Y1f) : floor(Y1f));
     Up to this point it is fine. However, it would also require (messy?) 
     modifications of the 'push back into beam' code, so I gave up.
     
     So, now we simply take rpix=image_rad-1. This means that sometimes a
     pixel close to the border is not selected.
     Not that anyone cares...
  */
  int rpix = round(image_rad-1);		 /* Radius of target image in voxel units */
  const double r2 = rpix * rpix * d_xy * d_xy; /* Radius squared of target image in mm^2 */
   
 
//Formula in Eq 6.12 of Egger Matthias PhD
  // First compute X1, Y1
  // KT 16/06/98 added const
  const double t = s + 0.5;		/* In a beam, not on a ray */

  // KT 25/09/2001 replace assert() with return value
  // check if there is any pixel in the beam
  if (t > rpix)
    return Succeeded::no;

  {     
    const double root = sqrt(rpix * rpix - t * t);// Eq 6.12 in EGger Thesis
    const double X1f = t * cphi + sphi * root;	/* Only valid if d_xy = d_p from Eq 6.12 in EGger Thesis  */
    //const double X2f = t * cphi - sphi * root;
    const double Y1f = t * sphi - cphi * root;// Eq 6.12 in EGger Thesis
    //const double Y2f = t * sphi + cphi * root;

    // KTTODO when using even number of bins, this should be floor(X1f) + .5
    X1 = (int) floor(X1f + 0.5);
    //X2 = (int) floor(X2f + 0.5);
    Y1 = (int) floor(Y1f + 0.5);
    //Y2 = (int) floor(Y2f + 0.5);
  }


  // Compute ds = difference between s and s-projection of selected voxel
  // KT 22/05/98 we don't need t later on
  //t=X1*cphi+Y1*sphi;
  //ds=t-s;
  ds=X1*cphi+Y1*sphi-s;// Eq 6.13 in Egger thsis
  // Note that X1f*cphi+Y1f*sphi == t == s + 0.5, so
  // ds == (X1-X1f)*cphi + (Y1-Y1f)*sphi + 0.5, hence
  // -(cphi + sphi)/2 + 0.5 <= ds <= (cphi + sphi)/2 + 0.5

  /* Push voxel back into beam */
  // KT 16/06/98 I now always use the cos(phi) case 
  // For example, when ds<0, adding sin(phi) is not guaranteed to make it positive
  // (where the most obvious example was phi==0, although I did treat that correctly).
  // On the other hand ds + cphi >= (cphi-sphi)/2 + 0.5, which is >=0.5 as
  // 0<=phi<=45 (in fact, it is still >=0 if 45<phi<=90).
  // Similarly, ds - cphi <= -(cphi-sphi)/2 + 0.5 <= 1 for 0<=phi<=90.

  if (ds<epsilon) 
    {
#if 0    
      if(sphi>epsilon)   
	{
	  Y1++; 
	  //Y2--; 
	  //t+=sphi; 
	  ds+=sphi;
	}
      else
#endif
	{
	  X1++; 
	  //X2--; 
	  //t+=cphi;
	  ds+=cphi;
	  // Now 0.5 <= (cphi-sphi)/2 + 0.5 <= ds < cphi+epsilon
	}
    } else if (ds>=1.)
      {
#if 0
	if(sphi>epsilon)   
	  {
	    Y1--; 
	    //Y2++; 
	    //t-=sphi; 
	    ds-=sphi;
	  }
	else
#endif
	  {
	    X1--; 
	    //X2++; 
	    //t-=cphi;
	    ds-=cphi;
	    // Now 0 <= 1-cphi<= ds <= -(cphi-sphi)/2 + 0.5 <= 0.5
	  }
      }
  // At the end of all this, 0 <= ds < 1

  // Now find Z1
  // Note that t=s+.5
  {
    const double t2dp2 = t * t * d_p * d_p;//Eq 6.12 in EGger thesis
    //const double ttheta =(delta * d_sl / sqrt(R2 - t2dp2));//Equivalent to tan(theta) see Eq 6.10 in Egger thesis except that d_r/2  has been replaced to d_sl
    
    if (t2dp2 >= R2 || t2dp2 > r2)
      return Succeeded::no;

    const double root1 = sqrt(R2 - t2dp2);
    const double root2 = sqrt(r2 - t2dp2);    

    // Eq 6.12 in Egger's thesis
    // const double Z1f =  2 * ring0 + 1 + (ttheta * (root1 - root2))/d_sl: 
    // equivalent formula:
    // Z1f =  2 * ring0 + 1 + delta * (1 - root2/root1)
    // We convert this to 'general' sizes of planes w.r.t. rings 
    // KT&CL 22/12/99 inserted offset
    const double Z1f =  
      num_planes_per_axial_pos*(ring0 + 0.5) + axial_pos_to_z_offset 
      + num_planes_per_physical_ring*delta/2 * (1 - root2/root1);
    //const double Z2f = (ring0 + 0.5)/ring_unit + (ttheta * (root1 + root2))/d_sl;
    // TODO possible problem for negative Z1f, use floor() instead
    Z1 =  (int) floor(Z1f);

    //Z2 = (int) Z2f;
  }
  {

    // Compute 'z' (formula 6.15 in Egger's thesis) which is the ring coordinate of the
    // projection of the voxel X1,Y1,Z1
    const double root=sqrt(R2p-t*t);//CL 26/10/98 Put it back the original formula as before it was root=sqrt(R2p-s*s)
    //const double z=ring_unit*( Z1-delta*( (-X1*sphi+Y1*cphi)/root + 1 ) );// Eq 6.15 from Egger Thesis//CL SPAN 14/09/98 2*delta
    // KT&CL 22/12/99 inserted offset
    const double z=( Z1-num_planes_per_physical_ring*delta/2*( (-X1*sphi+Y1*cphi)/root + 1 ) -
                     axial_pos_to_z_offset)/num_planes_per_axial_pos;
    dz=z-ring0; 
    // Using the same formula (z=...) for X1f, Y1f, Z1f gives zf = ring0+ 0.5
    // As the difference between X1f, Y1f, Z1f and X1,Y1,Y2 is at most 1 in every coordinate,
    //   -1/2 - delta/root/2 <= z - zf <= delta/root/2, so
    // -delta/root/2 <= dz <= 0.5 + delta/root/2
    // As delta < Rpix for most scanners, -1/num_planes_per_axial_pos < dz < 1
    // For some scanners though (e.g. HiDAC) this is not true, so we keep on checking if
    // dz is in the appropriate range


    /* Push voxel back into beam */
    // KT 01/06/98 added 'else' here for the case when dz=1/num_planes_per_axial_pos, the first step puts it to 0,
    // the 2nd step shouldn't do anything (dz<0), but because we compare with epsilon, 
    // it got back to .5
    
    //  MOREZ: if ->while
    if (dz>=1./num_planes_per_axial_pos) 
    {
      while (dz>=1./num_planes_per_axial_pos)
      { 
	Z1--; 
	//z-=1/num_planes_per_axial_pos; 
	dz-=1./num_planes_per_axial_pos;
      } 
    }
    else //if 
    { 
      while (dz<epsilon)   
      { 
	Z1++;
	//z+=1/num_planes_per_axial_pos; 
	dz+=1./num_planes_per_axial_pos;
      }
    }

    // TODO we could do a test of either Z or its Q lies in the axial FOV
    // if not, go along the beam. This could save us some updates of the 
    // incremental quantities.

    // KT&CL 22/12/99 changed ring_unit
    dzvert=-delta*cphi/root/num_planes_per_axial_pos;  /* Only valid if d_xy = d_p */
    dzhor=-delta*sphi/root/num_planes_per_axial_pos;
  }

  return Succeeded::yes;
}

END_NAMESPACE_STIR
