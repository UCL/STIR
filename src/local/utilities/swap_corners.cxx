#include "stir/ProjDataFromStream.h"
#include "stir/interfile.h"
#include "stir/utilities.h"
#include "stir/SegmentBySinogram.h"
#include <fstream>

#ifndef STIR_NO_NAMESPACES
using std::fstream;
#endif

USING_NAMESPACE_STIR

void dets_to_ve( int da, int db, int *v, int *e, int ndets)
{
	int h,x,y,a,b,te;

	h=ndets/2;
        x=std::max(da,db);
        y=std::min(da,db);
	a=((x+y+h+1)%ndets)/2;
	b=a+h;
	te=abs(x-y-h);
	if ((y<a)||(b<x)) te=-te;
	*e=te;
	*v=a;
}

typedef unsigned char byte;

int * 
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
/*
    Copyright (C) 2000- $Date$, IRSL
    See STIR/LICENSE.txt for details
*/
	int ndets, deta, detb, v, e, a, b, *list, i, j, n, m;
	int db = 32, off, nodup;
	byte *fixer;

	if (nviews%2 == 0) fixer = (byte *)ring_16;
	if (nviews%7 == 0) fixer = (byte *)ring_14;
	if (nviews%7 == 0) db = 8*7;
	if (nviews%3 == 0) fixer = (byte *)ring_12;

	n = 0;
	ndets = nviews*2*nmash;
        std::cerr<< "swap_corners: guessing "<< ndets << " detectors per ring\n";
	// KT guess
	//if (nviews%9 == 0) fixer = (byte *)ring_18;
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
  
  if (argc!=4 && argc!=3)
  {
    std::cerr << "Usage: " << argv[0] << " out_name in_name [mash]\n"
              << "If the mash parameter is omitted, it will be computed from info in the data.\n"
              << "A mash factor of 4 means 4 views have been combined into 1.\n";
    return EXIT_FAILURE;
  }
  int mash = -1;
  if (argc==4)
      mash = atoi(argv[3]);

  
  shared_ptr<ProjData> org_proj_data_ptr = 
    ProjData::read_from_file(argv[2]);
  char out_name[max_filename_length];
  strcpy(out_name, argv[1]);
  add_extension(out_name, ".s");
  fstream *out = new fstream;
  open_write_binary(*out, out_name);
  ProjDataFromStream 
    new_proj_data(org_proj_data_ptr->get_proj_data_info_ptr()->clone(),
                  out);  
  write_basic_interfile_PDFS_header(argv[1], new_proj_data);

  
  const int num_tang_poss = org_proj_data_ptr->get_num_tangential_poss();
  const int min_tang_pos_num = org_proj_data_ptr->get_min_tangential_pos_num();
  const int min_view_num = org_proj_data_ptr->get_min_view_num();

  const int mash_from_data =
    org_proj_data_ptr->get_proj_data_info_ptr()->get_scanner_ptr()->get_num_detectors_per_ring() /
    org_proj_data_ptr->get_num_views() / 2;
  std::cerr << "Mash factor determined from data is " << mash_from_data << std::endl;
  if (mash == -1)  
    mash = mash_from_data;
  else if (mash != mash_from_data)
  {
    std::cerr << "This is different from the parameter on the command line : " << mash<< std::endl;
    if (!ask("Continue with data from command line?",false))
      return EXIT_FAILURE;
  }
    

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
      Sinogram<float> sino1=org_proj_data_ptr->get_sinogram(axial_pos_num, segment_num, false);
      Sinogram<float> sino2=org_proj_data_ptr->get_sinogram(axial_pos_num, -segment_num, false);

      for (int i=0; i<num_swapped; i++)
      {
	int offset = swap_lors[i];
        const int tang_pos_num = offset%num_tang_poss + min_tang_pos_num;
        const int view_num = offset/num_tang_poss + min_view_num;
        std::swap(sino1[view_num][tang_pos_num], sino2[view_num][tang_pos_num]);
        //sino1[view_num][tang_pos_num] = 1; sino2[view_num][tang_pos_num] = 2;
      }
     
      new_proj_data.set_sinogram(sino1);
      new_proj_data.set_sinogram(sino2);
    }
  }
  return EXIT_SUCCESS;
}
