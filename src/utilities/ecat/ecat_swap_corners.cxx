//
// $Id$
//
/*
    Copyright (C) CTI PET Inc
    Copyright (C) 2002- $Date$, Hammersmith Imanet Ltd

    This file is part of STIR.

    The dets_to_ve() and compute_swap_lors_mashed() functions are from CTI, and
    come under a somewhat restrictive license.

    The main() function of this file is free software; you can redistribute it and/or modify
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
  \ingroup utilities
  \ingroup ECAT
  \brief A (slightly dangerous) utility to perform so-called corner-swapping
  for ECAT6 projection data.

  \author Christian Michel
  \author CTI PET Inc
  \author Kris Thielemans

  $Date$
  $Revision$

  \par Usage
  \code
   ecat_swap_corners out_name in_name 
  \endcode

  \par What does it do?

  For some historical reason, CTI scanners store 3D sinograms 
  sometimes in a 'corner-swapped' mode. What happens is that some
  corners of the positive and negative segments are interchanged.
  (As a consequence, segment 0 is never affected).<p>

  Below is a summary of what Kris Thielemans understood about
  corner-swapping from various emails with CTI people. However, he
  might have totally misunderstood this, so beware!<p>

  Corner-swapped mode occurs ALWAYS for ECAT6 data straight
  from the scanner. However, data which have been normalised using
  the import_3dscan utility from CTI are already corner-swapped
  correctly. Unfortunately, there is no field in the ECAT6 header
  that allows you to find out which mode it is in.

  For ECAT7 data, the situation is even more confusing. Data acquired
  directly in projection data have to be corner-swapped when the 
  acquisition was in 'volume-mode' (i.e. stored by sinograms), but
  NOT when acquired in 'view-mode' (i.e. stored by view). It seems that
  bkproj_3D_sun follows this convention by assuming that any ECAT7
  projection data stored in 'volume-mode' has to be corner swapped, and
  when it writes projection data in 'view-mode', it does the corner swapping
  for you. 
  So, although there is strictly speaking no field in the ECAT7 header 
  concerning corner swapping, it seems that the storage mode field 
  determines the corner swapping as well.
  <br>
  When the data is acquired in listmode, this changes somewhat.
  Apparently, there is a parameter in the set-up of listmode scans
  that allows you to put the ACS in 'volume-mode' or 'view-mode'. The
  resulting listmode files encode the sinogram coordinates then with
  corner-swapping or without. After the acquisition, the listmode data
  has then to be binned into projection data. It is then up to the
  binning program to take this corner-swapping into account. This is
  easiest to do by generating 'volume-mode' projection data when a 
  'volume-mode'  when the listmode setup was in 'volume-mode', and 
  similar for 'view-mode'.<br>
  If this sounds confusing to you, KT would agree. 
  Here seems to be the best thing to do:<p>
  <i>Do all acquisitions in 'view-mode', set-up your listmode scan
  in 'view-mode', bin the data in 'view-mode'. Forget about 
  corner-swapping.</i><p>
  If you cannot do this, then this utility will corner-swap the 
  projection data for you. 

  \par Who implemented this and how was it tested?

  The actual corner swapping code was supplied by Christian Michel,
  based on code by Larry Byars.<br>
  KT has tested it by performing a very long cylinder scan in 
  'volume-mode' on the ECAT 966, and looking at the delayeds. The oblique
  segments had obvious discontinuities in the efficiency patterns.
  After applyying this utility, these discontinuities appeared.
  
  \warning This utility does not (and cannot) check for you if the
  data has to be corner-swapped or not. So, it can do the wrong thing.

*/
#include "stir/ProjDataInterfile.h"
#include "stir/utilities.h"
#include "stir/SegmentBySinogram.h"
#include "stir/Succeeded.h"
#include <iostream>
#include <algorithm>
#ifndef STIR_NO_NAMESPACES
using std::cerr;
using std::endl;
using std::min;
using std::max;
using std::swap;
#endif

USING_NAMESPACE_STIR

static void dets_to_ve( int da, int db, int *v, int *e, int ndets)
{
	int h,x,y,a,b,te;

	h=ndets/2;
        x=max(da,db);
        y=min(da,db);
	a=((x+y+h+1)%ndets)/2;
	b=a+h;
	te=abs(x-y-h);
	if ((y<a)||(b<x)) te=-te;
	*e=te;
	*v=a;
}

typedef unsigned char byte;

// Magic stuff that computes which bins are in the wrong corner
// A mistery to KT...
static int * 
compute_swap_lors_mashed( int nprojs, int nviews, int nmash, int *nptr)
{
	static byte ring_16[8][3] = {
		{0,5,0}, {0,6,0}, {0,7,1}, {1,6,1},
		{10,15,0}, {9,15,0}, {8,15,2}, {9,14,2}};
	static byte ring_14[8][3] = {
		{0,4,0}, {0,5,0}, {0,6,1}, {1,5,1},
		{9,13,0}, {8,13,0}, {8,12,2}, {7,13,2}};
	static byte ring_12[8][3] = {
		{0,3,0}, {0,4,0}, {0,5,1}, {1,4,1},
		{8,11,0}, {7,11,0}, {7,10,2}, {6,11,2}};
	// KT guess, but it was wrong
	/*static byte ring_18[8][3] = {
		{0,7,0}, {0,8,0}, {0,9,1}, {1,8,1},
		{11,17,0}, {10,17,0}, {10,16,2}, {9,17,2}};
	*/ 
	int ndets, deta, detb, v, e, a, b, *list, i, j, n, m;
	int db = 32, off, nodup;
	byte *fixer = 0;

	if (nviews%2 == 0) fixer = (byte *)ring_16;
	if (nviews%7 == 0) fixer = (byte *)ring_14;
	if (nviews%7 == 0) db = 8*7;
	if (nviews%3 == 0) fixer = (byte *)ring_12;

	if (fixer == 0)
	  error("ecat_swap_corners: num_views not compatible with what this routine can cope with");

	n = 0;
	ndets = nviews*2*nmash;
        cerr<< "swap_corners: guessing "<< ndets << " detectors per ring\n";
	if (nviews%9 == 0) db = ndets/12;
	list = (int*) malloc( db*db*8*sizeof(int));
	for (i=0; i<8; i++)
	{
	  a = fixer[3*i];
	  b = fixer[3*i+1];
	  m = fixer[3*i+2];
	  for (deta=0; deta<db; deta++)
	    for (detb=0; detb<db; detb++)
	    {
		dets_to_ve( a*db+deta, b*db+detb, &v, &e, ndets);
		e += nprojs/2;
		off = (v/nmash)*nprojs+e;
		if (m==1 && v<nviews*nmash/2) continue;
		if (m==2 && v>nviews*nmash/2) continue;
		nodup = 1;
		for (j=0; j<n; j++)
		  if (off == list[j]) nodup=0;
		if (nodup && (e+1>0) && (e<nprojs)) list[n++] = off;
	    }
	}
	*nptr = n;
	return (list);
}

int main(int argc, char **argv)
{
  
  if (argc!=3)
  {
    cerr << "Usage: " << argv[0] << " out_name in_name \n";
    return EXIT_FAILURE;
  }
  
  shared_ptr<ProjData> org_proj_data_ptr = 
    ProjData::read_from_file(argv[2]);
  ProjDataInterfile 
    new_proj_data(org_proj_data_ptr->get_proj_data_info_ptr()->clone(),
                  argv[1]);  

  
  const int num_tang_poss = org_proj_data_ptr->get_num_tangential_poss();
  const int min_tang_pos_num = org_proj_data_ptr->get_min_tangential_pos_num();
  const int min_view_num = org_proj_data_ptr->get_min_view_num();

  const int mash =
    org_proj_data_ptr->get_proj_data_info_ptr()->get_scanner_ptr()->get_num_detectors_per_ring() /
    org_proj_data_ptr->get_num_views() / 2;
  cerr << "Mash factor determined from data is " << mash << endl;
    

  int num_swapped = 0;
  int *swap_lors = 
    compute_swap_lors_mashed(org_proj_data_ptr->get_num_tangential_poss(), 
                             org_proj_data_ptr->get_num_views(), 
                             mash,
                             &num_swapped);

  // first do segment 0 (no swapping)
  new_proj_data.set_segment(org_proj_data_ptr->get_segment_by_sinogram(0));

  // now the rest
  for (int segment_num=1; segment_num<=org_proj_data_ptr->get_max_segment_num(); ++segment_num)
  {
    for (int axial_pos_num=org_proj_data_ptr->get_min_axial_pos_num(segment_num);
         axial_pos_num<=org_proj_data_ptr->get_max_axial_pos_num(segment_num);
         ++axial_pos_num)
    {
      Sinogram<float> sino1=
	org_proj_data_ptr->get_sinogram(axial_pos_num, segment_num, false);
      Sinogram<float> sino2=
	org_proj_data_ptr->get_sinogram(axial_pos_num, -segment_num, false);

      for (int i=0; i<num_swapped; i++)
      {
	int offset = swap_lors[i];
        const int tang_pos_num = offset%num_tang_poss + min_tang_pos_num;
        const int view_num = offset/num_tang_poss + min_view_num;
        swap(sino1[view_num][tang_pos_num], sino2[view_num][tang_pos_num]);
      }
     
      new_proj_data.set_sinogram(sino1);
      new_proj_data.set_sinogram(sino2);
    }
  }
  free (swap_lors);
  return EXIT_SUCCESS;
}
