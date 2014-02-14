/*
*/
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- 2001, IRSL
    See STIR/LICENSE.txt for details
*/

/*!
\file 
  
 \brief  Functions to write data to MathLink
  
 \author Kris Thielemans
 \author PARAPET project
 
  

 \internal
*/

#ifdef STIR_MATHLINK

#include "mathlink.h"
#include <stdio.h>
#include <stdlib.h>

int asknr (char str[], int minv, int maxv,int def)
{
  char ptr[10];
  int nnn,ret;

  while(1)
  { printf ("\n%s [%d:%d D:%d]: ",str,minv,maxv,def);
    fgets(ptr,10,stdin);
    ret=sscanf(ptr,"%d",&nnn);
    if (ret==0 || ret==EOF)
      return def;
    if ((nnn>=minv) && (nnn<=maxv))
      return nnn;
    puts("\nOut of bounds");
  }
}


#if MACINTOSH_MATHLINK
extern int mlmactty_init( char*** argvp);
#endif

void init_and_connectlink( char* linkname);
static void mlerror( MLINK lp);


MLENV ep = (MLENV)0;
MLINK lp = (MLINK)0;



static void mlerror( MLINK lp)
{
	if( MLError( lp)){
		fprintf( stderr, "Error detected by MathLink: %s.\n",
			MLErrorMessage(lp));
	}else{
		fprintf( stderr, "Error detected by this program.\n");
	}
	exit(3);
}


static void deinit( void)
{
	if( ep) MLDeinitialize( ep);
}


static void closelink( void)
{
	if( lp) MLClose( lp);
}


void init_and_connectlink( char* linkname)
{
	long err;
	char arguments[200];
	int num = 0;

    /* test if link is already open */
	if (lp != (MLINK)0)
		return;

	num = asknr("Number to add to the linkname",0,500,0);
	sprintf(arguments, "-linkconnect -linkname \"%s%d\"", linkname, num);

/* TCP connection, slow...   
   num = asknr("TCP IP port to connect to",0,4000,1000);
	sprintf(arguments, "-linkconnect -linkprotocol \"TCP\" -linkname \"%d@petnt1\"", num);
*/
    
#if MACINTOSH_MATHLINK
	MLYieldFunctionObject yielder;
	argc = mlmactty_init( &argv);
#endif

	ep =  MLInitialize( (MLParametersPointer)0);
	if( ep == (MLENV)0) exit(1);
	atexit( deinit);

#if MACINTOSH_MATHLINK
	yielder = MLCreateYieldFunction( ep, NewMLYielderProc( MLDefaultYielder), 0);
#endif

	lp = MLOpenString( ep, arguments, &err);
	if(lp == (MLINK)0) exit(2);
	atexit( closelink);
	
#if MACINTOSH_MATHLINK
	MLSetYieldFunction( lp, yielder);
#endif
    if( MLError( lp) && MLError(lp) != MLECONNECT) 
	  mlerror(lp);
}


#endif /*STIR_MATHLINK */
