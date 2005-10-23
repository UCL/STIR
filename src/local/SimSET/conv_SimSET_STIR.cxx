/******************************************************************************************************

     conv_SimSET_STIR.c

     This program converts SimSET sinograms in STIR format
     
     Pablo Aguiar
**************************************************************************************************/

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
//#include <util/nrutil.h>
//#include <util/sequen.h>
#include "stir/ProjDataInterfile.h"
#include "stir/ProjDataInfoCylindricalArcCorr.h"
#include "stir/Sinogram.h"
#include "stir/Scanner.h"
#include "stir/shared_ptr.h"
#include "stir/IO/read_data.h"
#include "stir/Succeeded.h"
#define NUMARG 11


int main(int argc,char **argv)
{
  using namespace stir;

int ncol,nfil,slices,segment,nitems,n,ii;
int i_segment,lim_down,lim_up;
char fseq[180],fima[180],tipoin[3];
FILE *file; /*files to read and to write*/

static char *opcions[]={
    "argv[1]  SimSET file\n",
    "argv[2]  SimSET file format\n",
    "argv[3]  Angles in SimSET file\n",
    "argv[4]  Bins in SimSET filele\n",
    "argv[5]  Axial slices in SimSET file\n",
    "argv[6]  Segment\n",
    "argv[7]  FOV_radius in cm as given to simset binning module (max_td)\n",
    "argv[8]  range on Z value in cm as given to simset binning module\n",
    "argv[9]  index of 3d-sinogram in file (0-based)\n"
    "argv[10] STIR file name\n"
    };
if (argc!=NUMARG){
  printf("\n\nConvert SimSET to STIR\n\n");
  printf("Not enough arguments !!! ..\n");
  for (int i=1;i<NUMARG;i++) printf(opcions[i-1]);
  exit(EXIT_FAILURE);
  }

strcpy(fseq,argv[1]);
strncpy(tipoin,argv[2],2);
nfil=atoi(argv[3]);
ncol=atoi(argv[4]);
slices=atoi(argv[5]);
segment=atoi(argv[6]);
 const float FOV_radius = atof(argv[7])*10; // times 10 for mm
 const float scanner_length = atof(argv[8])*10; // times 10 for mm
const int dataset_num = atoi(argv[9]);
strcpy(fima,argv[10]);
nitems=nfil*ncol;



if( (file=fopen(fseq,"rb")) ==NULL){
    printf("\n\nCannot open the simset file\n\n");
    exit(EXIT_FAILURE);
    }
 shared_ptr<Scanner> scanner_sptr;
 {
  scanner_sptr = new Scanner(Scanner::E966);
  const float this_scanner_length = 
    scanner_sptr->get_num_rings() * scanner_sptr->get_ring_spacing();
  if (fabs(this_scanner_length - scanner_length)>1.0)
    {
      scanner_sptr = new Scanner(Scanner::E962);
      const float this_scanner_length = 
	scanner_sptr->get_num_rings() * scanner_sptr->get_ring_spacing();
      if (fabs(this_scanner_length - scanner_length)>1.0)
	{
	  warning("scanner length %g does not match 966 nor 962. Using 962 anyway");
	}
    }
 }
 warning("Selected scanner %s", scanner_sptr->get_name().c_str());

 scanner_sptr->set_num_rings(slices);
 scanner_sptr->set_ring_spacing(scanner_length/slices);
 scanner_sptr->set_num_detectors_per_ring(nfil*2);
 shared_ptr<ProjDataInfo> proj_data_info_sptr =
   ProjDataInfo::ProjDataInfoCTI( scanner_sptr,
				  /*span=*/1, 
				  /*max_delta=*/segment,
				  nfil,
				  ncol,
				  /*arc_corrected =*/ true);
 dynamic_cast<ProjDataInfoCylindricalArcCorr&>(*proj_data_info_sptr).
   set_tangential_sampling(2*FOV_radius/ncol);

 ProjDataInterfile proj_data(proj_data_info_sptr,
			     fima, std::ios::out);
 


if(strncmp(tipoin,"fl",2)==0){
  //seq=vector(0,(slices*slices*nitems)-1);
   //q=vector(0,num_ima_extr*nitems-1);
   }
else 
  {
    error("\nfile format %s not valid. Only fl at present", tipoin);
  }


// skip simset header
  if (fseek(file, 32768 + dataset_num*slices*slices*nitems*4, SEEK_SET) != 0)
    error("Error skipping simset header"); 
  Array<1,float> seq(0,(slices*slices*nitems)-1);
   read_data(file, seq /*, byteorder */);

i_segment=0;
n=0;
while(i_segment<=segment)
    {
      {
        lim_down=0;
        lim_up=lim_down+((slices-i_segment-1)*(slices+1));
      }
      for(n=lim_down;n<=lim_up;) //Extraccion de los sinogramas de una serie !!!
	{

      Sinogram<float> pos_sino = proj_data.get_empty_sinogram((n-lim_down)/(slices+1), i_segment);
      Sinogram<float> neg_sino = proj_data.get_empty_sinogram((n-lim_down)/(slices+1), -i_segment);
      Sinogram<float>::full_iterator pos_sino_iter = pos_sino.begin_all();
      Sinogram<float>::full_iterator neg_sino_iter = neg_sino.begin_all();

    ii=nitems-1;
    const int i_r1r2=(n + i_segment)*nitems;
    const int i_r2r1=(n + i_segment*slices)*nitems;
     
    // get 2 sinograms from simset data
    for(int i=0;i<nitems/2;++i)  
    {
      *pos_sino_iter++ = seq[i_r1r2+ii];
      *neg_sino_iter++ = seq[i_r2r1+ii];
      ii=ii-1;  
    }
    for(int i=nitems/2;i<nitems;++i)   
    {
      *neg_sino_iter++ = seq[i_r1r2+ii];
      *pos_sino_iter++ = seq[i_r2r1+ii];
      ii=ii-1;  
    }
    n=n+slices+1;
    proj_data.set_sinogram(pos_sino);
    proj_data.set_sinogram(neg_sino);
	}
      i_segment=i_segment+1;
    }

//fwrite(q,4,num_ima_extr*nitems,file2);

//free(seq); free(q);

fclose(file);
//fclose(file2);
}












