//
// $Id$: $Date$
//


#include "pet_common.h" 
#include <iostream> 
#include <fstream>

#include <numeric>
#include "imagedata.h"


#include "display.h"
// KT 13/10/98 include interfile
#include "interfile.h"


#define ZERO_TOL 0.000001
#define ROOF 40000000.0

void display_halo(const PETImageOfVolume& input_image);

void clean_rim(PETImageOfVolume& input_image);
void get_plane(PETImageOfVolume& input_image);
void get_plane_row(PETImageOfVolume& input_image);

main(int argc, char *argv[]){

  if(argc<2) {
    cout<<"Usage: vox <input filename>";
    exit(1);
  }


  //  PETScannerInfo::Scanner_type scanner_type = PETScannerInfo::RPT;
  // PETScannerInfo scanner(scanner_type);
// KT 13/10/98 include interfile

#if 0
  int scanner_num = 0;
  PETScannerInfo scanner;
  char scanner_char[200];


  scanner_num = ask_num("Enter scanner number (0: RPT, 1: 953, 2: 966, 3: GE) ? ", 0,3,0);
  switch( scanner_num )
    {
    case 0:
      scanner = (PETScannerInfo::RPT);
      sprintf(scanner_char,"%s","RPT");
      break;
    case 1:
      scanner = (PETScannerInfo::E953);
      sprintf(scanner_char,"%s","953");   
      break;
    case 2:
      scanner = (PETScannerInfo::E966);
      sprintf(scanner_char,"%s","966"); 
      break;
    case 3:
      scanner = (PETScannerInfo::Advance);
      sprintf(scanner_char,"%s","Advance");   
      break;
    default:
      PETerror("Wrong scanner number\n"); Abort();
    }

 
  if(argc==3){
    scanner.num_bins /= 4;
    scanner.num_views /= 4;
    scanner.num_rings /= 1;
  

    scanner.bin_size = 2* scanner.FOV_radius / scanner.num_bins;
    scanner.ring_spacing = scanner.FOV_axial / scanner.num_rings;
  }



  Point3D origin(0,0,0);
  Point3D voxel_size(scanner.bin_size,
                     scanner.bin_size,
                     scanner.ring_spacing/2); 
  /*
    Point3D voxel_size(scanner.FOV_radius*2/scanner.num_bins,
    scanner.FOV_radius*2/scanner.num_bins,
    scanner.FOV_axial/(2*scanner.num_rings - 1));
  */
  // Create the image
  // two  manipulations are needed now: 
  // -it needs to have an odd number of elements
  // - it needs to be larger than expected because of some overflow in the projectors

  
  int max_bin = (-scanner.num_bins/2) + scanner.num_bins-1;
  if (scanner.num_bins % 2 == 0)
    max_bin++;
   



  PETImageOfVolume 
    input_image(Tensor3D<float>(
				0, 2*scanner.num_rings-2,
				(-scanner.num_bins/2), max_bin,
				(-scanner.num_bins/2), max_bin),
		origin,
		voxel_size);


  // Open file with data
  ifstream input;
  open_read_binary(input, argv[1]);
  input_image.read_data(input);   
#endif // now interfile

  ifstream input(argv[1]);
  if (!input)
    { 
      PETerror("Couldn't open file");
      exit(1);
    }

  PETImageOfVolume 
    input_image = read_interfile_image(input);
   
  int zs,ys,xs, ze,ye,xe,choice;

  zs=input_image.get_min_z();
  ys=input_image.get_min_y();
  xs=input_image.get_min_x(); 
  
  ze=input_image.get_max_z();  
  ye=input_image.get_max_y(); 
  xe=input_image.get_max_x();

 
 

  cerr<<"resx: "<<input_image.get_x_size()<<endl;
  cerr<<"resy: "<<input_image.get_y_size()<<endl;
  cerr<<"resz: "<<input_image.get_z_size()<<endl;
  cerr << "Min and Max in image " << input_image.find_min() 
       << " " << input_image.find_max() << endl;


  int plane=1;


  do{
    cerr<<endl<<"1. Visual"<<endl;
    cerr<<"2. Data total"<<endl;
    cerr<<"3. Data plane-wise"<<endl;
    cerr<<"4. Min/Max for plane"<<endl;
    cerr<<"5. Show halo"<<endl;
    //    cerr<<"6. Central profile "<<endl;
    cerr<<"6. Clean rim "<<endl;
    cerr<<"7. Get plane"<<endl;
    cerr<<"8. Get row"<<endl;
    cerr<<"9. Counts"<<endl;
    cerr<<"10. Quit"<<endl<<endl;
    cerr<<"Input selection: ";
    cin>>choice;
    cerr<<endl;


    switch(choice){

    case 1:
      {  
	// KT 13/10/98 add text to images
	VectorWithOffset<float> 
	  scale_factors(input_image.get_min_index3(), 
			input_image.get_max_index3());
	scale_factors.fill(1.F);
	VectorWithOffset<char *> 
	  text(input_image.get_min_index3(),
	       input_image.get_max_index3());
	for (int i=input_image.get_min_index3(); 
	     i<= input_image.get_max_index3(); 
	     i++)
	  {
	    char *str = new char [15];
	    sprintf(str, "image %d", i);
	    text[i] = str;
	  }
	display(input_image, scale_factors, text, 
		input_image.find_max(), 0, 0);
 
	break;
      }
    case 2:
      {
	for (int z=zs; z<=ze; z++)
	  for (int y=ys; y <= ye; y++)
	    for (int x=xs; x<= xe; x++){
	      cerr<<input_image[z][y][x]<<" ";
	    }

	break;
      }

    case 3:
      {
	plane=1;
	
	while(plane>0 && plane <=ze-zs+1 ){
	  cerr<<endl<<"\nInput plane # (0 to exit):  ";
	  cin>>plane;
	  cerr<<endl<<endl;
	  
	  if(!(plane>0 && plane <=ze-zs+1 )) break;	  	 
	  
	  for (int y=ys; y <= ye; y++)
	    for (int x=xs; x<= xe; x++){
	      cerr<<input_image[zs+plane-1][y][x]<<" ";
	      // pause
	    }
	}
	break;

      }
 
    case 4:
      {
	plane=1;
	
	while(plane>0 && plane <=ze-zs+1 ){
	  cerr<<endl<<"\nInput plane # (0 to exit):  ";
	  cin>>plane;
	  cerr<<endl<<endl;
	  
	  if(!(plane>0 && plane <=ze-zs+1 )) break;
	  cerr << "Min and Max in plane " << input_image[zs+plane-1].find_min() 
	       << " " << input_image[zs+plane-1].find_max() << endl;
	  
	}
	break;
      }


    case 5:
      {
	display_halo(input_image);
	break;
      }

    case 6:
      {
	clean_rim(input_image);
	break;
      }

    case 7:
      {
	get_plane(input_image);
	break;
      }

    case 8:
     {
       get_plane_row(input_image);
       break;
     }

 case 9:
   {

     cerr<<endl<<"The number of counts is: "<<input_image.sum()<<endl;

      break;
   }
    case 10:
      break;
    }
  
  }while(!(choice==10));

  return 0;

}

void display_halo(const PETImageOfVolume& input_image){

  static int flag_halo=0;
  static PETImageOfVolume halo=input_image;

  if (flag_halo==0){
    int zs,ys,xs, ze,ye,xe;

    zs=input_image.get_min_z();
    ys=input_image.get_min_y();
    xs=input_image.get_min_x(); 
  
    ze=input_image.get_max_z();  
    ye=input_image.get_max_y(); 
    xe=input_image.get_max_x();

  
    float zm=(zs+ze)/2.;
    float ym=(ys+ye)/2.;
    float xm=(xs+xe)/2.;
  
    float d;
 
    for (int z=zs; z<=ze; z++)
      for (int y=ys; y <= ye; y++)
	for (int x=xs; x<= xe; x++){


	  d=sqrt(pow(xm-x,2)+pow(ym-y,2));

	  if(halo[z][y][x]>ZERO_TOL && d>(xe-xm) ) 
	    halo[z][y][x]=1.0;   
	  else
	    halo[z][y][x]=0.0;
	
	}
 

    flag_halo=1;
  }

  display(halo, halo.find_max());     

}


void clean_rim(PETImageOfVolume& input_image){

  int zs,ys,xs, ze,ye,xe,plane;

  zs=input_image.get_min_z();
  ys=input_image.get_min_y();
  xs=input_image.get_min_x(); 
  
  ze=input_image.get_max_z();  
  ye=input_image.get_max_y(); 
  xe=input_image.get_max_x();

  float zm=(zs+ze)/2.;
  float ym=(ys+ye)/2.;
  float xm=(xs+xe)/2.;
  
  float d;
 
  for (int z=zs; z<=ze; z++)
    for (int y=ys; y <= ye; y++)
      for (int x=xs; x<= xe; x++){

	d=sqrt(pow(xm-x,2)+pow(ym-y,2));

	if(d>(xe-xm)-4) {
	  input_image[z][y][x]=0.0;   
	}

      }

  //MJ 07/09/98 added output feature

  if(ask_num("Write trimmed image to file ? (0: No, 1: Yes)",0,1,0)){
    char outfile[200];
    cout << endl << "Output filename (without extension) : ";
    cin >> outfile;
    write_basic_interfile( outfile, input_image);

  }


}

void get_plane(PETImageOfVolume& input_image){

  char filename[200];
  int zs,ys,xs, ze,ye,xe;

  zs=input_image.get_min_z();
  ys=input_image.get_min_y();
  xs=input_image.get_min_x(); 
  
  ze=input_image.get_max_z();  
  ye=input_image.get_max_y(); 
  xe=input_image.get_max_x();

  float zm=(zs+ze)/2.;
  float ym=(ys+ye)/2.;
  float xm=(xs+xe)/2.;


  int plane=ask_num("Which plane?",1,ze-zs+1,(int)zm-zs+1);

  cerr<<endl<<"Output to what file:  ";
  cin>>filename;
  cerr<<endl;

  ofstream profile(filename);
      if (!profile)
      { cerr << "Couldn't open " << filename; }

  for (int y=ys; y<= ye; y++)
    for (int x=xs; x<= xe; x++)
      profile<<input_image[zs+plane-1][y][x]<<" ";
      

}

void get_plane_row(PETImageOfVolume& input_image){

  char filename[200];
  int zs,ys,xs, ze,ye,xe;

  zs=input_image.get_min_z();
  ys=input_image.get_min_y();
  xs=input_image.get_min_x(); 
  
  ze=input_image.get_max_z();  
  ye=input_image.get_max_y(); 
  xe=input_image.get_max_x();

  float zm=(zs+ze)/2.;
  float ym=(ys+ye)/2.;
  float xm=(xs+xe)/2.;
  

  cerr<<endl<<"Output to what file:  ";
  cin>>filename;
  cerr<<endl;

  ofstream profile(filename);
  if (!profile)
    { cerr << "Couldn't open " << filename; return; }

  int axdir=ask_num("Which axis direction (z=0,y=1,x=2)?",0,2,2);

 

  if (axdir==0){


    int xcoord=ask_num("X COORDINATE: ",1,xe-xs+1,(int)xm-xs+1);
    int ycoord=ask_num("Y COORDINATE: ",1,ye-ys+1,(int)ym-ys+1);

 
    for (int z=zs; z<= ze; z++)
      profile<<input_image[z][ycoord+ys-1][xcoord+xs-1]<<" ";

  }


  else if (axdir==1){


    int zcoord=ask_num("Z COORDINATE: ",1,ze-zs+1,(int)zm-zs+1);
    int ycoord=ask_num("Y COORDINATE: ",1,ye-ys+1,(int)ym-ys+1);

    for (int x=xs; x<= xe; x++)
      profile<<input_image[zcoord+zs-1][ycoord+ys-1][x]<<" ";



  }

  else{ //axdir=2

    int zcoord=ask_num("Z COORDINATE: ",1,ze-zs+1,(int)zm-zs+1);
    int xcoord=ask_num("X COORDINATE: ",1,xe-xs+1,(int)xm-xs+1);
 
    for (int y=ys; y<= ye; y++)
      profile<<input_image[zcoord+zs-1][y][xcoord+xs-1]<<" ";

  }


}
