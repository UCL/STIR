#include "interfile.h"
#include"ProjDataFromStream.h"
#include "SegmentByView.h"

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

void add_poisson(ProjData& output_projdata, 
		 const ProjData& input_projdata, 
		 const float scaling_factor);


int
main (int argc,char *argv[])
{
  if(argc<5)
  {
    cerr<<"Usage: add_poisson <input header file name (*.hs)> scaling_factor <output filename (no extension)> seed-unsigned-int\n";
    exit(EXIT_FAILURE);
  }  
  
  const char *const filename = argv[3];
  const float scaling_factor = atof(argv[2]);
  shared_ptr<ProjData>  in_data = ProjData::read_from_file(argv[1]);
  unsigned int seed = atoi(argv[4]);
  srand(seed);
  
  string filename_san =filename;
  filename_san += ".s";
  
  iostream * sino_stream = new fstream (filename_san.c_str(), ios::out| ios::binary);
  if (!sino_stream->good())
  {
    error("add_poisson: error opening file %s\n",filename_san.c_str());
  }
  
  ProjDataFromStream new_data(in_data->get_proj_data_info_ptr()->clone(),sino_stream); 
  write_basic_interfile_PDFS_header(filename_san, new_data);
  
  add_poisson(new_data,*in_data, scaling_factor);
  
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
add_poisson(ProjData& output_projdata, const ProjData& input_projdata, const float scaling_factor)
{
  
  
  for (int seg= input_projdata.get_min_segment_num(); 
       seg<=input_projdata.get_max_segment_num();
       seg++)  
  {
    SegmentByView<float> seg_input= input_projdata.get_segment_by_view(seg);
    SegmentByView<float> seg_output= 
      output_projdata.get_empty_segment_by_view(seg,false);
  
    cerr << "Segment " << seg << endl;

    for(int view=seg_input.get_min_view_num();view<=seg_input.get_max_view_num();view++)    
      for(int ax_pos=seg_input.get_min_axial_pos_num();ax_pos<=seg_input.get_max_axial_pos_num();ax_pos++)
	//for(int tang_pos=-5;tang_pos<5;tang_pos++)	
      for(int tang_pos=seg_input.get_min_tangential_pos_num();tang_pos<=seg_input.get_max_tangential_pos_num();tang_pos++)
      { 
	/*	cerr << "View " << view << endl;
	cerr << "Ax_Pos" << ax_pos << endl;
	cerr << "Tang Poss" << tang_pos << endl;
	*/
	const float bin = seg_input[view][ax_pos][tang_pos];
	//cerr << "bin" << bin << endl;
	const int random_poisson = generate_poisson_random(bin*scaling_factor);
	//cerr << "done " << random_poisson << endl;
	seg_output[view][ax_pos][tang_pos] = static_cast<float>(random_poisson);		
      }           
      output_projdata.set_segment(seg_output);      
  }
}




