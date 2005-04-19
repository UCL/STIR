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
#include "stir/ProjDataInfo.h"
#include "stir/Sinogram.h"
#include "stir/Scanner.h"
#include "stir/shared_ptr.h"
#include "stir/IO/read_data.h"
#include "stir/Succeeded.h"
#define NUMARG 9


int main(int argc,char **argv)
{
  using namespace stir;

int i,k,ncol,nfil,slices,segment,nitems,num_ima_extr,n,ii;
int i_segment,diag,lim_down,lim_up,ima_serie,i_write,i_read,sino;
//float *seq,*q,s,qq,stot;
char fseq[180],fima[180],tipoin[3];
FILE *file,*file2; /*files to read and to write*/

static char *opcions[]={
    "argv[1]  SimSET file\n",
    "argv[2]  SimSET file format\n",
    "argv[3]  Angles in SimSET file\n",
    "argv[4]  Bins in SimSET filele\n",
    "argv[5]  Axial slices in SimSET file\n",
    "argv[6]  Segment\n",
    "argv[7]  index of 3d-sinogram in file (0-based)\n"
    "argv[8]  STIR file name\n"
    };
if (argc!=NUMARG){
  printf("\n\nConvert SimSET to STIR\n\n");
  printf("Not enough arguments !!! ..\n");
  for (i=1;i<NUMARG;i++) printf(opcions[i-1]);
  exit(0);
  }

strcpy(fseq,argv[1]);
strncpy(tipoin,argv[2],2);
nfil=atoi(argv[3]);
ncol=atoi(argv[4]);
slices=atoi(argv[5]);
segment=atoi(argv[6]);
const int dataset_num = atoi(argv[7]);
strcpy(fima,argv[8]);
nitems=nfil*ncol;



if( (file=fopen(fseq,"rb")) ==NULL){
    printf("\n\nNO SE PUEDE ABRIR EL ARCHIVO DE LECTURA\n\n");
    exit(0);
    }
/*if( (file2=fopen(fima,"wb")) ==NULL){
    printf("\n\nNO SE PUEDE ABRIR EL ARCHIVO DE ESCRITURA\n\n");
    exit(0);
    }
    num_ima_extr=num_imagenes(slices,segment);
*/
 shared_ptr<Scanner> scanner_sptr = new Scanner(Scanner::E962);
 const float scanner_length = 
   scanner_sptr->get_num_rings() * scanner_sptr->get_ring_spacing();
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
 ProjDataInterfile proj_data(proj_data_info_sptr,
			     fima, std::ios::out);
 


if(strncmp(tipoin,"fl",2)==0){
  //seq=vector(0,(slices*slices*nitems)-1);
   //q=vector(0,num_ima_extr*nitems-1);
   }
else printf("\nFORMATO DE FICHERO NO VALIDO\n\n");
//fread(seq,4,slices*slices*nitems,file);

// skip simset header
  if (fseek(file, 32768 + dataset_num*slices*slices*nitems*4, SEEK_SET) != 0)
    error("Error skipping simset header"); 
  Array<1,float> seq(0,(slices*slices*nitems)-1);
   read_data(file, seq /*, byteorder */);

i_segment=segment*(-1);
k=0;
n=0;
while(i_segment<=segment)
    {
    if(i_segment<0)
        {
        lim_down=i_segment*slices*(-1);
        lim_up=lim_down+((slices+i_segment-1)*(slices+1));
        ima_serie=((lim_up-lim_down)/(slices+1))+1;
        //printf("\n\n  %d  -  %d",lim_down,lim_up);
        //printf("**   %d  imagenes  de esta serie**\n\n",ima_serie);
        }
    else if(i_segment>=0)
        {
        lim_down=i_segment;
        lim_up=lim_down+((slices-i_segment-1)*(slices+1));
        ima_serie=((lim_up-lim_down)/(slices+1))+1;
        //printf("\n\n  %d  -  %d",lim_down,lim_up);
        //printf("**   %d  imagenes de esta serie **\n\n",ima_serie);
        }
    for(n=lim_down;n<=lim_up;) //Extraccion de los sinogramas de una serie !!!
    {
      // TODO potentially use -i_segment

      Sinogram<float> sino = proj_data.get_empty_sinogram((n-lim_down)/(slices+1), i_segment);
      Sinogram<float>::full_iterator sino_iter = sino.begin_all();

    ii=nitems-1;
     
    for(i=0;i<nitems;)   //Extraccion de un sinograma !!!
    {
    i_read=n*nitems;
    //i_write=k*nitems;
    //*(q+i_write+i)=*(seq+i_read+ii);
    *sino_iter = seq[i_read+ii];
    ++sino_iter;
    i=i+1;
    ii=ii-1;  
    }
    k=k+1;
    n=n+slices+1;
    proj_data.set_sinogram(sino);
    }
    i_segment=i_segment+1;
    }

//fwrite(q,4,num_ima_extr*nitems,file2);

//free(seq); free(q);

fclose(file);
//fclose(file2);
}












