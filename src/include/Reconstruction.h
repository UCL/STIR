// $Id$: $Date$
//
#ifndef __RECONSTRUCTION_H__
#define __RECONSTRUCTION_H__

#include "pet_common.h"
#include <string>

#include "sinodata.h"
#include "imagedata.h"
#include "Filter.h"

START_NAMESPACE_TOMO

//	PETSinogramOfVolume s(/* appropriate parameters*/);
//	PETImageVolume v(/* appropriate parameters*/);
//		
//	recon.reconstruct(s,v);


// KT 28/06/98 disabled for now
#if 0
// CL 1/12 These two fonctions should be declared somewherre else
PETSinogramOfVolume CorrectForAttenuation(const PETSinogramOfVolume &sino, const PETSinogramOfVolume &atten)       
{
// Kernel of Attenuation correction
    PETerror("TODO"); Abort();
    return sino;
}

PETSegmentBySinogram CorrectForAttenuation(const PETSegmentBySinogram &sino, const PETSegmentBySinogram &atten)
{
// Kernel of Attenuation correction
    PETerror("TODO"); Abort();
    return sino;
}
#endif

class PETReconstruction
{
 
    
public:
        // KT 02/06/98 made pure virtual
    virtual string method_info();

    virtual string parameter_info()
        { return ""; }
        /* CL 15/10/98 Default constructor*/
    PETReconstruction(){
        cout << "Reconstruction starting..." << endl;
    }
 
  
    virtual void reconstruct(const PETSinogramOfVolume &s, PETImageOfVolume &v)
        {}
  
// KT 28/06/98 disabled for now
#if 0
    virtual void  reconstruct(const PETSinogramOfVolume &s, const PETSinogramOfVolume &a, PETImageOfVolume &v) 
        { 
            reconstruct(CorrectForAttenuation(s, a), v); 
        }
#endif
};

class PETAnalyticReconstruction: public PETReconstruction
{
public:
    int delta_min; /* Lower bound of "obliqueness" */
    int delta_max; /* Upper bound of "obliqueness" */
   const Filter1D<float> &filter; /* Ramp filter */
    
        //CL 15/10/98 New function
    virtual string method_info(){
        return("Analytical algorithm");
    }
  
    virtual string parameter_info();
    
    
    PETAnalyticReconstruction(int min, int max, const Filter1D<float>& f);
    
};

// MS 13/07 comments PETIterativeReconstruction ,
// defined in ART/PETIterativeReconstruction.h with extention
/*
class PETIterativeReconstruction: public PETReconstruction
{
public:
    int max_iterations;
    PETIterativeReconstruction(int max=1);
            //CL 15/10/98 New function
    virtual string method_info();
};

inline PETIterativeReconstruction::PETIterativeReconstruction
(int max) :max_iterations(max)
{    
}
*/


/******************* 2D reconstructions ************/
class PETReconstruction2D
{
public:
        // KT 02/06/98 made pure virtual
    virtual string method_info() = 0;
    virtual string parameter_info()
        { return ""; }
  
    virtual void reconstruct(const PETSinogram &sino2D, PETPlane &image2D)=0;
    virtual void reconstruct(const PETSegment &sino, PETImageOfVolume &image)=0;  
        // KT 28/06/98 disabled for now
#if 0
    virtual void reconstruct(const PETSegment& s, 
                             const PETSegment& a, PETImageOfVolume& v) 
        { reconstruct(CorrectForAttenuation(s, a), v); }
#endif
};





class PETAnalyticReconstruction2D: public PETReconstruction2D
{
protected:
    const Filter1D<float> &filter;
public:
    virtual string parameter_info();
        //CL 15/10/98 ACtivate filter
              PETAnalyticReconstruction2D(const Filter1D<float>& f);    
    
};



class Reconstruct2DFBP : public PETAnalyticReconstruction2D
{

public:
    virtual string method_info()
        { return("2DFBP"); }
 
    Reconstruct2DFBP(const Filter1D<float>& f): PETAnalyticReconstruction2D(f) 
        {
            cout << "  - 2D FBP processing" << endl;
        }
    

    void reconstruct(const PETSinogram &sino2D, PETPlane &image2D);
    void reconstruct(const PETSegment &sino, PETImageOfVolume &image);
};



//08/07 MS add ParaReconstruct2DFBP class & parallel code
#ifdef PARALLEL
#include "Para.h"
#include "parablocks.h"
#include "PTimer.h"
#include "PMsgHandlerReg.h"
#include "PDispatch.h"


extern PEnvironment PEnv;

void 
zoom_segment (PETSegmentBySinogram& segment, 
	      const float zoom, const float Xoffp, const float Yoffp, 
	      const int size, const float itophi);

void Backprojection_2D(const PETSegment &sinos, 
		       PETImageOfVolume &image,
                       const int rmin, 
		       const int rmax);
void Backprojection_2D(const PETSegment &sinos, PETImageOfVolume &image, 
		       const int view, const int rmin, const int rmax);

void
zoom_image(PETImageOfVolume &image,
                       const float zoom,
                       const float Xoff, const float Yoff, 
                       const int new_size )     ;


//MS added this for RampFilter in ParaReconstruct2DFBP 
class PETAnalyticReconstruction2DRampFilter: public PETReconstruction2D
{
protected:
    const Filter1DRamp &filter;
public:
      //  virtual string parameter_info();
     
       
       PETAnalyticReconstruction2DRampFilter(const Filter1DRamp& f)
        : filter(f)
        { } 
};




////////////////////////////////////////////////////
class ParaReconstruct2DFBP   : public PETAnalyticReconstruction2DRampFilter
{

public:

   //friend  PMessage& operator<<(PMessage& msg, ParaReconstruct2DFBP& recon);
   //friend  PMessage& operator>>(PMessage& msg, ParaReconstruct2DFBP& recon);

    virtual string method_info()
        { return("Parallel 2DFBP"); }
    
    ParaReconstruct2DFBP(const Filter1DRamp& f): PETAnalyticReconstruction2DRampFilter(f) 
    {
            cout << "  - Parallel 2D FBP processing" << endl;
    }    

    void reconstruct(const PETSinogram &sino2D, PETPlane &image2D);
    void reconstruct(const PETSegment &sino, PETImageOfVolume &image);

   public :
       static ParaReconstruct2DFBP  *SlaveRecon;  
       static int *Index;
       static int  SlaveIndex,x1,x2,y1,y2;
       static int Estimate_number;
       static Point3D Origine;
       static Point3D VoxelSize;
};


#endif




END_NAMESPACE_TOMO    

#endif
