
#include <stdlib.h>
#include "utilities.h"
#include "interfile.h"
#include"ProjDataFromStream.h"
#include "SegmentByView.h"
#include "ProjDataInfoCylindrical.h"

#include <iostream> 
#include <fstream>

#ifndef TOMO_NO_NAMESPACES
using std::iostream;
using std::ofstream;
using std::ifstream;
using std::fstream;
using std::ios;
using std::cerr;
using std::endl;
using std::cout;
#endif


USING_NAMESPACE_TOMO

int generate_poisson_random(const float mu);

void add_poisson(ProjDataFromStream& output_projdata, 
		 const ProjData& input_projdata, 
		 const float scaling_factor);


int
main (int argc,char *argv[])
{
  if(argc<4)
  {
    cout<<"Usage: add_poisson <header file name> (*.hs) scaling_factor\n";
    exit(EXIT_FAILURE);
  }  
  
  const char *const filename = argv[3];
  shared_ptr<ProjData>  in_data = ProjData::read_from_file(argv[1]);
  
  
  const ProjDataInfo* data_info = in_data->get_proj_data_info_ptr();
  string header_name = filename;
  header_name += ".hs"; 
  string filename_san =filename;
  filename_san += ".prj";
  
  ProjDataInfo* new_data_info= data_info->clone();
  
  float scale_factor = 1;
  iostream * sino_stream = new fstream (filename_san.c_str(), ios::out| ios::binary);
  if (!sino_stream->good())
  {
    error("add_poisson: error opening file %s\n",filename_san);
  }
  
  ProjDataFromStream new_data(new_data_info,sino_stream); 
  float scaling_factor = atoi(argv[2]);
  
  add_poisson(new_data,*in_data, scaling_factor);
  cerr << "done" << endl;
  write_basic_interfile_PDFS_header(filename_san, new_data);
  
  
  return EXIT_SUCCESS;
}



int generate_poisson_random(const float mu)
{
  
  double u = static_cast<double>(rand()) / RAND_MAX;
  
  
  // prevent problems if n growing too large (or even to infinity) 
  // when u is very close to 1
  if (u>1-1.E-6)
    u = 1-1.E-6;
  
  const double upper = exp(mu)*u;
  double accum = 1.;
  double term = 1.; 
  int n = 1;
  
  while(accum <upper)
  {
    accum += (term *= mu/n); 
    n++;
  }
  
  
  return (n - 1);
}




void 
add_poisson(ProjDataFromStream& output_projdata, const ProjData& input_projdata, const float scaling_factor)
{
  
  
  for (int seg= input_projdata.get_proj_data_info_ptr()->get_min_segment_num(); 
  seg<=input_projdata.get_proj_data_info_ptr()->get_max_segment_num();seg++)  
  {
    SegmentByView<float> seg_output_set= 
      output_projdata.get_proj_data_info_ptr()->get_empty_segment_by_view(seg);//i);
    seg_output_set.fill(0);
    output_projdata.set_segment(seg_output_set);
  }
  
  
  int i=0;
  SegmentByView<float> seg_input= input_projdata.get_segment_by_view(i);
  SegmentByView<float> seg_output= 
    output_projdata.get_proj_data_info_ptr()->get_empty_segment_by_view(i);
  
  //cerr << "Min View " << seg_input.get_min_view_num() << endl;
  // cerr << "Max View " << seg_input.get_max_view_num() << endl;  
  //cerr << "Min Axial " << seg_input.get_min_axial_pos_num() << endl;
  //cerr << "Max Axial " << seg_input.get_max_axial_pos_num() << endl;
  // cerr << "Min Tang " << seg_input.get_min_tangential_pos_num() << endl;
  // cerr << "Max Tang " << seg_input.get_max_tangential_pos_num() << endl;
  
  
  
  for(int view=seg_input.get_min_view_num();view<seg_input.get_max_view_num();view++)    
    for(int ring=seg_input.get_min_axial_pos_num();ring<seg_input.get_max_axial_pos_num();ring++)
      for(int tang_pos=-5;tang_pos<5;tang_pos++)	
      { 
	cerr << "View " << view << endl;
	cerr << "Ring" << ring << endl;
	cerr << "Tang Poss" << tang_pos << endl;
	
	float bin = seg_input[view][ring][tang_pos];
	cerr << "bin" << bin << endl;
	int random_poisson = generate_poisson_random(bin*scaling_factor);
	cerr << "done random poisson" << endl;
	seg_output[view][ring][tang_pos] = static_cast<float>(random_poisson);	
	
      }           
      output_projdata.set_segment(seg_output);
      cerr << " done in the loop" << endl;
      
      
      
      
}