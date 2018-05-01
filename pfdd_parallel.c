/*
 *  xtal.c
 *  
 *
 *  Created by marisol on 11/17/08.
 *  Copyright 2008 Purdue. All rights reserved.
 *
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <dfftw_mpi.h>
#include <string.h>

//#define N1 32
//#define N2 32  
//#define N3 32
//#define NS 1        /* 12 number of slip systems for FCC */
//#define ND 3        //number of dimensions
//#define NT 7000     //number of time steps
//#define NSI 20     // for stress incrementing NT*NSI = total time steps
//#define NP 1        /* 4 number of slip planes for FCC*/

int main(int argc, char **argv)
{
  int i, j, k, m, n, ksym, it, is, is2, id, ir, ka, kb, nb, na, ida, idb;
  int na0, na1, na2, nsize, k1, k2, k3, nfreq, ii, itp, b1, b2;
  int isa, isb, psys, iflag, fflag, mflag, cflag, nloops, radius2, oflag;
  int rank, numprocs, rtn_val, ckpt, tslast, channel;
  int lN1, lxs, lnyt, lyst, lsize;
  int index, index2, ia, ib, ic, ig, ie, ih, ij;
  int ic1, ic2, ic3, ic4;
  int N1, N2, N3, NS, ND, NT, NSI, NP;
  unsigned long seed;
  double *fx, *fy, *fz, *slr, *sli, *xi, *xi_sum, *xo, pi;
  double *f, *r, *fcore, *df1core, *df2core, *df3core;
  double L, d1, d2, d3, dt, size, size3, sizegs, xi_ave, xiave, alpha;
  double *dE_core, E_core, setsigma, sigstep;
  double *data_eps, *data_epsd, *data_sigma;
  double c0, c1, c2, c3, c4, a1, a3, isf, usf; /*constants for gamma surface*/
  double b, C11, C12, C44, dslip, CD, obsden, An, Bn, dtt, mu, young, nu, a, ll;
  double *BB, *FF, *DD, S11, S12, S44;
  double T0, TT0, T1, TT1, T2, TT2, T9, TT9, MPI_Wtime();
  double T3, TT3, T4, TT4, T5, TT5, T6, TT6, T7, TT7, T8, TT8, T10, TT10;
  double T0_ave, T1_ave, T2_ave, T3_ave, T4_ave, T5_ave, T10_ave;
  double T6_ave, T7_ave, T8_ave, T9_ave, np;
  double T0a, T1a, T2a, T3a, T4a, T5a, T6a, T7a, T8a, T9a;

  FILE *of2, *of3, *of8, *inf0, *out0, *of11;
  char outfile[100], output[100], outfile3[100], output3[100], outfile8[100], output8[100];
  char scratch[100], title[100], outfile10[100], outold[100], outfile11[100], outckpt[100];

  //read input file
  inf0 = fopen("PFDD_input.dat","r");
  if(inf0 == NULL){
    printf("File 'PFDD_input.dat' could not be opened\n");
    perror("input.dat");
    exit(1);
  }
  else{
    fscanf(inf0, "%d %d %d %d %d %d %d %d %d %d %d %d %d %d %lf %lf %d %lf %lf %d %d %s %s", &N1, &N2, &N3, &NT, &NSI, &NS, &ND, &NP, &iflag, &channel, &nloops, &radius2, &mflag, &cflag, &An, &Bn, &oflag, &setsigma, &sigstep, &ckpt, &tslast, scratch, title);
      }

  double **sigma, ***eps;
  double **avesigma, **avepsd, **avepst, **aveps, **ave_eps, **ave_epst, **ave_sigma, **ave_sigch, **ave_epsd, **avepsts;
  double **xn, **xb, *tau, **tchsig, **tch_sig, **cchsig, **cch_sig;
  
  fftwnd_mpi_plan plan, iplan;
  fftw_complex *data_fftw, *work, *data_real, *temp_data, *data_core, *data_strain, *work_strain;

  //subroutine declarations
  void material(int mflag, double *a, double *mu, double *young, double *c0, double *c1, double *c2, double *c3, double *c4, double *a1, double *a3, double *isf, double *usf, double *nu, double *ll, double *C44, double *C12, double *C11, double *S44, double *S12, double *S11);
  void frec (double *fx, double *fy, double *fz, double, double,double, int, int,int N1, int N2, int N3, int rank);
  void setfcc(double **xn, double **xb, int rank);
  void set2D(double **xn, double **xb, int rank, int oflag);
  void set3D1pl(double **xn, double **xb, int rank, int oflag);
  void set3D2sys(double **xn, double **xb, int rank);
  void resolSS( double **sigma, double *tau, double **xn, double **xb, double dslip, int rank, int ND, int NS);
  void Bmatrix( double *BB, double *fx, double *fy, double *fz, double **xn, double **xb, double ***eps, double , double , double , double, double , double , double, double, int, double nu, int rank, int N1, int N2, int N3, int ND, int NS);
  void Fmatrix(double *FF, double *DD, double *fx, double *fy, double *fz, double ***eps, double d1, double d2, double d3, double C11, double C22, double C44, int lN1, int rank, int N1, int N2, int N3, int ND, int NS); /*needs to be called after function Bmatrix*/
  void initial_sxtal(int rank, int lN1, int lxs, double *xi, double *xo, fftw_complex *data_fftw, int iflag, double *xi_sum, double obsden, double dslip, double b, int N1, int N2, int N3, int ND, int NS, int nloops, int radius2, double nu, char *scratch, char *title, int channel);
  void core_ener(int cflag, int it, int rank, int lxs, double An, int lN1, double c0, double c1, double c2, double c3, double c4, double a1, double a3, double **xn, double **xb, fftw_complex *data_fftw, fftw_complex *data_core, double dslip, double b, double *fcore, double *df1core, double *df2core, double *df3core, double *dE_core, double E_core, int itp, double pi, int N1, int N2, int N3, int ND, int NS, int NP, int NT, int NSI, double Bn, double mu, int nsize, double size, double isf, double usf, char *scratch, char *title, int tslast, int oflag);
  void avestrain(double **avepsd, double **avepst, double ***eps, double *xi, int nsize, double **sigma, double S11, double S12, double S44, double mu, int lN1, double **ave_epsd, int rank, double **avepsts, int N1, int N2, int N3, int ND, int NS);
  void strain(fftw_complex *data_strain, double *data_eps, fftw_complex *data_fftw, double *FF, double d1, double d2, double d3, double size, int it, int itp, double **avepst, int lxs, int lN1, int nsize, fftw_complex *work_strain, int rank, int N1, int N2, int N3, int ND, int NS, int NT, int NSI, char *scratch, char *title, int tslast);
  void stress(double *data_epsd, double *data_sigma, fftw_complex *data_strain, double *xi, double ***eps, double C11, double C12, double C44, int it, int itp, double **avesigma, int lxs, int lN1, int nsize, int rank, double **ave_sigma, double **ave_sigch, double **sigma, double **avepsts, int N1, int N2, int N3, int ND, int NS, int NT, int NSI, char *scratch, char *title, int tslast, double **tchsig, double **tch_sig, double **cchsig, double **cch_sig, int channel); /*called after function strain*/
  void init_genrand(unsigned long s);
  double genrand_real3(void);

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
  if(rank == 0){
    printf("rank %d, numprocs %d\n", rank, numprocs);
  }
  T0 = MPI_Wtime();

  /*create plan and iplan for fftw*/

  plan = fftw3d_mpi_create_plan(MPI_COMM_WORLD, N1, N2, N3, FFTW_FORWARD, FFTW_ESTIMATE);

  iplan = fftw3d_mpi_create_plan(MPI_COMM_WORLD, N1, N2, N3, FFTW_BACKWARD, FFTW_ESTIMATE);

  /* slab decomposition*/

  fftwnd_mpi_local_sizes(plan, &lN1, &lxs, &lnyt, &lyst, &lsize);
  np = numprocs;
  if(rank == 0){
    printf("lN1 %d, lxs %d, lnyt %d, lyst %d, lsize %d\n", lN1, lxs, lnyt, lyst, lsize);
    printf("N1 %d, N2 %d, N3 %d, NT %d, NSI %d, NS %d, ND %d, NP %d, iflag %d, nloops %d, radius2 %d, mflag %d, cflag %d, An %lf, Bn %lf, oflag %d, setsigma %lf, sigstep %lf, ckpt %d, tslast %d\n",N1, N2, N3, NT, NSI, NS, ND, NP, iflag, nloops, radius2, mflag, cflag, An, Bn, oflag, setsigma, sigstep, ckpt, tslast);
  }

  /*malloc data vectors*/
  
  data_fftw = (fftw_complex*) malloc(sizeof(fftw_complex) * lsize*NS);
  data_real = (fftw_complex*) malloc(sizeof(fftw_complex) * lsize*NS);
  temp_data = (fftw_complex*) malloc(sizeof(fftw_complex) * lsize*NS);
  //work = (fftw_complex*) malloc(sizeof(fftw_complex) * lsize*NS);
  work = (fftw_complex*) malloc(sizeof(fftw_complex) * lsize);
  work_strain = (fftw_complex*) malloc(sizeof(fftw_complex) * lsize);
  data_core = (fftw_complex*) malloc(sizeof(fftw_complex) * lsize*NS);
  data_strain = (fftw_complex*) malloc(sizeof(fftw_complex) * lsize*ND*ND);
  xi = (double*) malloc(2*(NS)*(lN1)*(N2)*(N3)*sizeof(double));
  xo = (double*) malloc((NS)*(lN1)*(N2)*(N3)*sizeof(double));
  xi_sum = (double*) malloc(2*(lN1)*(N2)*(N3)*sizeof(double));
  fx = (double*) malloc((lN1)*(N2)*(N3)*sizeof(double));
  fy = (double*) malloc((lN1)*(N2)*(N3)*sizeof(double));
  fz = (double*) malloc((lN1)*(N2)*(N3)*sizeof(double));
  f = (double*) malloc((NS)*(lN1)*(N2)*(N3)*sizeof(double));
  r = (double*) malloc((ND)*sizeof(double));
  BB = (double*) malloc((NS)*(NS)*(lN1)*(N2)*(N3)*sizeof(double));
  FF = (double*) malloc((NS)*lsize*(ND)*(ND)*sizeof(double));
  DD = (double*) malloc((NS)*lsize*(ND)*(ND)*sizeof(double));
  fcore = (double*) malloc((NP)*(lN1)*(N2)*(N3)*sizeof(double));
  df1core = (double*) malloc((NP)*(lN1)*(N2)*(N3)*sizeof(double));
  df2core = (double*) malloc((NP)*(lN1)*(N2)*(N3)*sizeof(double));
  df3core = (double*) malloc((NP)*(lN1)*(N2)*(N3)*sizeof(double));
  dE_core = (double*) malloc((NS)*(lN1)*(N2)*(N3)*sizeof(double));
  data_eps = (double*) malloc(2*((ND)*(ND)*(lN1)*(N2)*(N3))*sizeof(double));
  data_epsd = (double*) malloc(2*((ND)*(ND)*(lN1)*(N2)*(N3))*sizeof(double));
  data_sigma = (double*) malloc(2*((ND)*(ND)*(lN1)*(N2)*(N3))*sizeof(double));
  tau = (double*) malloc((NS)*sizeof(double));

  xn = (double**) malloc((NS)*sizeof(double));
  xb = (double**) malloc((NS)*sizeof(double));

  sigma = (double**) malloc((ND)*sizeof(double));
  avesigma = (double**) malloc((ND)*sizeof(double));
  avepsd = (double**) malloc((ND)*sizeof(double));
  avepst = (double**) malloc((ND)*sizeof(double));
  aveps = (double**) malloc((ND)*sizeof(double));
  ave_eps = (double**) malloc((ND)*sizeof(double));
  ave_epst = (double**) malloc((ND)*sizeof(double));
  ave_sigma = (double**) malloc((ND)*sizeof(double));
  ave_sigch = (double**) malloc((ND)*sizeof(double));
  ave_epsd = (double**) malloc((ND)*sizeof(double));
  avepsts = (double**) malloc((ND)*sizeof(double));
  tchsig = (double**) malloc((ND)*sizeof(double));
  tch_sig = (double**) malloc((ND)*sizeof(double));
  cchsig = (double**) malloc((ND)*sizeof(double));
  cch_sig = (double**) malloc((ND)*sizeof(double));

  eps =  (double***) malloc((NS)*sizeof(double));

  for(b1=0;b1<ND;b1++){
    xn[b1]=(double*) malloc((ND)*sizeof(double));
    xb[b1]=(double*) malloc((ND)*sizeof(double));

    sigma[b1] = (double*) malloc((ND)*sizeof(double));
    avesigma[b1] = (double*) malloc((ND)*sizeof(double));
    avepsd[b1] = (double*) malloc((ND)*sizeof(double));
    avepst[b1] = (double*) malloc((ND)*sizeof(double));
    aveps[b1] = (double*) malloc((ND)*sizeof(double));
    ave_eps[b1] = (double*) malloc((ND)*sizeof(double));
    ave_epst[b1] = (double*) malloc((ND)*sizeof(double));
    ave_sigma[b1] = (double*) malloc((ND)*sizeof(double));
    ave_sigch[b1] = (double*) malloc((ND)*sizeof(double));
    ave_epsd[b1] = (double*) malloc((ND)*sizeof(double));
    avepsts[b1] = (double*) malloc((ND)*sizeof(double));
    tchsig[b1] = (double*) malloc((ND)*sizeof(double));
    tch_sig[b1] = (double*) malloc((ND)*sizeof(double));
    cchsig[b1] = (double*) malloc((ND)*sizeof(double));
    cch_sig[b1] = (double*) malloc((ND)*sizeof(double));

    eps[b1] = (double**) malloc((ND)*sizeof(double));
    for(b2=0;b2<ND;b2++){
      eps[b1][b2] = (double*) malloc((ND)*sizeof(double));
    }
  }

  //define constants
  L = (double)(N3);
  seed = 243578 + lxs;
  pi = 3.141592654;
  init_genrand(seed);  /* seed the random number generator */
  
  /* Material constants*/

  //mflag = 1;
  material(mflag, &a, &mu, &young, &c0, &c1, &c2, &c3, &c4, &a1, &a3, &isf, &usf, &nu, &ll, &C44, &C12, &C11, &S44, &S12, &S11);
  b = (a)/sqrt(2.0); //meters-->a is in meters
  dslip = 1.0; //sqrt(3.0)*sqrt(2.0); /*dslip=d/b and d=2(sqrt(3)a/2) or d = Constant*b*/

  if(rank == 0){
    printf("mflag %d, a %lf, mu %lf, young %lf, nu %lf, ll %lf, C11 %lf, C12 %lf, C44 %lf, S11 %lf, S12 %lf, S44 %lf, isf %lf, usf %lf\n", mflag, a, mu, young, nu, ll, C11, C12, C44, S11, S12, S44, isf, usf);
  }

  nsize = N1*N2*N3;
  //size = alpha*N1*b;/*1.0E-6;*/ //1.0; //meters
  size = N1;
  size3 = size*size*size;
  d1 = 1.0; //size/N1;  //need N1 not lN1 here 
  d2 = 1.0; //size/N2;  //in units of b so frequencies are normalized 
  d3 = 1.0; //size/N3;
  fflag = 1; //determines whether multiple (fflag=1) or single (fflag!=1) ffts are taken, two different ways to the same result.
  //cflag = 1;

  if(rank == 0){
    printf("dslip %lf, size %lf, d1 %lf, d2 %lf, d3 %lf, cflag %d\n", dslip, size, d1, d2, d3, cflag);
  }

  CD = 0.5; //1.0; /*dislocation mobility coefficient*/
  dtt = 1.0; /*time increment*/  
  //An = 0.6/(dslip*b*C44); /*for core energy*/
  //Bn = 0.46;
  //setsigma = 0.004; //set initial stress
  //sigstep = 0.008; //set stress increment
  
  /*Set the slip systems, frequency and BB matrices */	
  
  if(NS == 1 && NP == 1){
    set2D(xn, xb, rank, oflag); //1 slip system edge and screw
  }
  else if(NS == 2 && NP == 2){
    set3D2sys(xn, xb, rank);  //2 slip systems
  }
  else if(NS == 3 && NP == 1){
    set3D1pl(xn, xb, rank, oflag); //3 slip systems on 1 plane edge and screw
  }
  else if(NS == 12 && NP == 4){
    setfcc(xn,xb, rank); //12 slip systems
  }
  else{
    if(rank == 0){
      printf("Direction vectors for the Burger's vectors and slip plane normals have not been included in a subroutine for this number of slip systems or slip planes.\n");
    }
  }

  if(rank == 0){
    printf("Calling Frequency Subroutine.\n"); 
  }

  frec(fx, fy, fz, d1, d2, d3, lN1, lxs, N1, N2, N3, rank);
  
  if(rank == 0){
    printf("Setting interaction matrix.\n"); 
  }

  T8 = MPI_Wtime();
  Bmatrix(BB, fx, fy, fz, xn, xb, eps, d1,  d2 , d3,  C11,  C12,  C44, b, dslip, lN1, nu, rank, N1, N2, N3, ND, NS);
  TT8 = MPI_Wtime() - T8;
  Fmatrix(FF, DD, fx, fy, fz, eps, d1,  d2 , d3,  C11,  C12,  C44, lN1, rank, N1, N2, N3, ND, NS); //after Bmatrix

  /*Initial Data*/

  if(rank == 0){
    printf("Setting Initial Conditions\n"); 
  }

  if(ckpt == 0 || ckpt == 3){
    T3 = MPI_Wtime();
    obsden = 0.1;  //if iflag == 9 this is the dislocation density
    initial_sxtal(rank, lN1, lxs, xi, xo, data_fftw, iflag, xi_sum, obsden, dslip, b, N1, N2, N3, ND, NS, nloops, radius2, nu, scratch, title, channel);
    TT3 = MPI_Wtime() - T3;
  }
  else{
     //read ckpt file
    for(isa=0;isa<NS;isa++){
      strcpy(outfile10,scratch);
      ic = sprintf(outold, "/outckpt_it%08.0d_NS%02.0d_P%03.0d.dat", tslast-1, isa, rank);
      strcat(outfile10, outold);
      for(ib=0; ib<strlen(outfile10); ib++){
	if(outfile10[ib]==' ') outfile10[ib]='0';
      }
      out0 = fopen(outfile10,"r");
      if(out0 == NULL){
	printf("File 'outckpt' could not be opened\n");
	perror("outckpt.dat");
	exit(1);
      }/*if-NULL*/
      else{
	for(i=0;i<lN1;i++)
	  for(j=0;j<N2;j++)
	    for(k=0;k<N3;k++)
	      {
		na0 = 2*(i*N2*N3 + j*N3 + k + isa*lN1*N2*N3);
		na1 = na0+1;
		index = i*N2*N3 + j*N3 + k + isa*lN1*N2*N3;

		fscanf(out0, "%lf %lf %lf", &xi[na0], &xi[na1], &xo[index]);

		data_fftw[index].re = xi[na0];
		data_fftw[index].im = xi[na1];
	      }/*ijk*/
      }/*else*/   
    }/*isa*/
  }/*else*/
  
  /*Time Step Loop */
  
  if(rank == 0){
    printf("Beginning Time Evolution\n"); 
  }

  for(itp=0;itp<NSI;itp++)
    {

      /*Set Applied Stress*/
      
      sigma[0][0]= 0.0;
      sigma[0][1]= 0.0;
      sigma[0][2]= setsigma; //0.0; //(setsigma + itp*sigstep);
      sigma[1][0]= sigma[0][1];
      sigma[1][1]= 0.0;
      sigma[1][2]= 0.0; 
      sigma[2][0]= sigma[0][2];
      sigma[2][1]= sigma[1][2];
      sigma[2][2]= 0.0;

      if(rank == 0){
	printf("set applied stress sigma[0][2] %lf sigma[2][0] %lf\n", sigma[0][2], sigma[2][0]); 
      }

      resolSS(sigma,tau, xn, xb, dslip, rank, ND, NS);

      /*Enter 2nd Time Step Loop*/

      /* data goes to fftw and is repalced by its fft 
	 xi is always in real space and is updated every step*/ 

      T4 = MPI_Wtime();
      TT1 = 0.0;
      TT2 = 0.0;
      TT5 = 0.0;
      TT6 = 0.0;
      TT7 = 0.0;
      TT9 = 0.0;
      TT10 = 0.0;
      
      for(it=0;it<NT;it++)
	{
	  if(rank==0 && fmod((double)(it),200.0) == 0.0){
	    printf("time step, %d %d %d %d\n", it, itp, it+(itp*NT), (it+(itp*NT))+tslast);
	  }
	  
	  /* Initialize values each time step*/
	  
	  for(i=0;i<lN1*N2*N3*2;i++){
	    xi_sum[i] = 0.0;
	  }

	  E_core = 0.0;  //Core Energy for each time step.
	  
	  for(i=0;i<lN1*N2*N3*NS;i++){
	    dE_core[i] = 0.0;
	  }
	  
	  for(i=0;i<lN1*N2*N3*NP;i++){
	    fcore[i] = 0.0;
	    df1core[i] = 0.0;
	    df2core[i] = 0.0;
	    df3core[i] = 0.0;
	  }
	  
	  for(i=0;i<ND;i++)
	    for(j=0;j<ND;j++){
	      avesigma[i][j] = 0.0;
	      avepst[i][j] = 0.0;
	      avepsts[i][j] = 0.0;
	      avepsd[i][j] = 0.0;
	      aveps[i][j] = 0.0;
	      ave_eps[i][j] = 0.0;
	      ave_sigma[i][j] = 0.0;
	      ave_sigch[i][j] = 0.0;
	      ave_epsd[i][j] = 0.0;
	      tchsig[i][j] = 0.0;
	      tch_sig[i][j] = 0.0;
	      cchsig[i][j] = 0.0;
	      cch_sig[i][j] = 0.0;
	    }
      
	  //Calculate Average Strain
	  
	  if(it == NT-1){
	    avestrain(avepsd, avepst, eps, xi, nsize, sigma, S11, S12, S44, mu, lN1, ave_epsd, rank, avepsts, N1, N2, N3, ND, NS);
	  }

	  //Calculate Core Energy

	  core_ener(cflag, it, rank, lxs, An, lN1, c0, c1, c2, c3, c4, a1, a3, xn, xb, data_fftw, data_core, dslip, b, fcore, df1core, df2core, df3core, dE_core, E_core, itp, pi, N1, N2, N3, ND, NS, NP, NT, NSI, Bn, mu, nsize, size, isf, usf, scratch, title, tslast, oflag); 
	  
	  if(rank == 0 && fmod((double)(it),200.0) == 0.0){
	    printf("out of core energy\n"); 
	  }

	   //Take forward FFT

	  if(fflag==1){
	    T1 = MPI_Wtime();
	    for(isa=0;isa<NS;isa++)
	      {
		psys = isa*lN1*N2*N3;
		fftwnd_mpi(plan, 1, data_fftw+psys, work, FFTW_NORMAL_ORDER);  /* Forward FFT (multiple)*/
		//fftwnd_mpi(plan, 1, data_fftw+psys, work, FFTW_TRANSPOSED_ORDER);
	      }
	    TT1 += (MPI_Wtime() - T1);
	  }
	  else{
	    T9 = MPI_Wtime();
	    for(isa=0;isa<NS;isa++){
	      for(ii=0;ii<lsize;ii++){
		index = isa + ii*NS;
		index2 = ii + isa*lsize;
		temp_data[index] = data_fftw[index2];
	      }
	    }
	    TT9 += (MPI_Wtime() - T9);
	    
	    T1 = MPI_Wtime();
	    fftwnd_mpi(plan,NS,temp_data,work,FFTW_NORMAL_ORDER); /*Forward FFT (single)*/
	    TT1 += (MPI_Wtime() - T1);
	    
	    T9 = MPI_Wtime();
	    for(isa=0;isa<NS;isa++){
	      for(ii=0;ii<lsize;ii++){
		index = isa + ii*NS;
		index2 = ii + isa*lsize;
		data_fftw[index2] = temp_data[index];
	      }
	    }
	    TT9 += (MPI_Wtime() - T9);
	  }
	  
	  for (i=0; i<lN1*N2*N3*NS; i++){
	    data_real[i].re = 0;  
	    data_real[i].im = 0;
	  }
	  
	   /*Multiply by Interaction Matrix (B-Matrix)*/
	  //if(rank == 0 && fmod((double)it,100)==0.0){
	  //printf("Multiply by BMatrix\n");
	  //}

	  T5 = MPI_Wtime();
	  for(isa=0;isa<NS;isa++)
	    {
	      for(isb=0;isb<NS;isb++)
		{
		  for(i=0;i<lN1;i++) 
		    for(j=0;j<N2;j++)  
		      for(k=0;k<N3;k++)
			{
			  index  = i*N2*N3 + j*N3 + k + isa*lN1*N2*N3; 
			  index2 = i*N2*N3 + j*N3 + k + isb*lN1*N2*N3; 
			  nb     = i*N2*N3 + j*N3 + k + isa*lN1*N2*N3 + isb*lN1*N2*N3*NS;	
			  data_real[index].re += data_fftw[index2].re * BB[nb];
			  data_real[index].im += data_fftw[index2].im * BB[nb];
			}
		}
	    }
	  TT5 += (MPI_Wtime() -T5);

	  /*Strain & Stress Calculations*/
	  
	  //if(it+(itp*NT) == (NSI*NT)-1){
	  if(fmod((double)(it),20.0) == 0.0 || it == NT-1){
	    //if(it == NT-1){
	    if(rank == 0){

	      /*open output file for stress/strain curve*/

	      strcpy(outfile8, scratch);
	      ic = sprintf(output8, "/outsscurve_it%08.0d_P%03.0d.dat", (it+(itp*NT))+tslast, rank);
	      strcat(outfile8, output8);
	      for(ib=0; ib<strlen(outfile8); ib++){
		if(outfile8[ib]==' ') outfile8[ib]='0';
	      }
	      
	      of8 = fopen(outfile8,"w");
	    }
	    
	    strain(data_strain, data_eps, data_fftw, FF, d1, d2, d3, size, it, itp, avepst, lxs, lN1, nsize, work_strain, rank, N1, N2, N3, ND, NS, NT, NSI, scratch, title, tslast);
	    stress(data_epsd, data_sigma, data_strain, xi, eps, C11, C12, C44, it, itp, avesigma, lxs, lN1, nsize, rank, ave_sigma, ave_sigch, sigma, avepsts, N1, N2, N3, ND, NS, NT, NSI, scratch, title, tslast, tchsig, tch_sig, cchsig, cch_sig, channel);
	    
	    MPI_Barrier(MPI_COMM_WORLD);

	    /*average strain*/

	    for(ida=0;ida<ND;ida++)
	      for (idb=0;idb<ND;idb++)
		{
		  for(i=0;i<lN1;i++)
		    for(j=0;j<N2;j++)
		      for (k=0;k<N3;k++)
			{	    
			  na0 = 2*(k + j*N3 + i*N2*N3 + ida*lN1*N2*N3 + idb*lN1*N2*N3*ND);	
			  aveps[ida][idb] += data_eps[na0]; //aveps only appears here to get total average strain
			}

		  MPI_Reduce(&aveps[ida][idb], &ave_eps[ida][idb], 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

		  if(rank == 0){
		    if(ida == 0 && idb == 2){
		      printf("%lf %lf %lf\n", ave_eps[ida][idb]/nsize, avepst[ida][idb], mu);
		    } 
		    ave_eps[ida][idb] /= nsize;

		    fprintf(of8,"%lf %lf %lf %lf %lf ", ave_eps[ida][idb], avepst[ida][idb], ave_sigma[ida][idb]/mu, ave_sigch[ida][idb], ave_sigch[ida][idb]/mu/*tch_sig[ida][idb]/mu, cch_sig[ida][idb]/mu*/);
		  }
		}
	    if(rank == 0){
	      fprintf(of8, "%lf %lf %d\n", sigma[0][2], sigma[1][2], channel);
	    }
	  }//it
	  
	  //Take inverse FFT

	  if(fflag==1){
	    T2 = MPI_Wtime();
	    for(isa=0;isa<NS;isa++)
	      {
		psys = isa*lN1*N2*N3;
		fftwnd_mpi(iplan, 1, data_real+psys, work, FFTW_NORMAL_ORDER); /* Inverse FFT (multiple)*/
		//fftwnd_mpi(iplan, 1, data_real+psys, work, FFTW_TRANSPOSED_ORDER);
	      }
	    TT2 += (MPI_Wtime() - T2);
	  }
	  else{	
	    T9 = MPI_Wtime();
	    for(isa=0;isa<NS;isa++){
	      for(ii=0;ii<lsize;ii++){
		index = isa + ii*NS;
		index2 = ii + isa*lsize;
		temp_data[index] = data_real[index2];
	      }
	    }
	    TT9 += (MPI_Wtime() - T9);
	    
	    T2 = MPI_Wtime();
	    fftwnd_mpi(iplan,NS,temp_data,work,FFTW_NORMAL_ORDER); /*Inverse FFT (single)*/
	    TT2 += (MPI_Wtime() - T2);
	    
	    T9 = MPI_Wtime();
	    for(isa=0;isa<NS;isa++){
	      for(ii=0;ii<lsize;ii++){
		index = isa + ii*NS;
		index2 = ii + isa*lsize;
		data_real[index2] = temp_data[index];
	      }
	    }
	    TT9 += (MPI_Wtime() - T9);
	  }
	  
	  T6 = MPI_Wtime();
	  for(isa=0;isa<NS;isa++)
	    {
	      /*open output file*/

	      //if(fmod((double)(it),200.0) == 0.0 && it >= 1000){
	      if(fmod((double)(it),20.0) == 0.0 || it+(itp*NT) == (NSI*NT)-1){
		if(ckpt >= 2){
		  //if(it == NT-1){
		  strcpy(outfile,scratch);
		  ic = sprintf(output, "/output_it%08.0d_NS%02.0d_P%03.0d.dat", (it+(itp*NT))+tslast, isa, rank);
		  strcat(outfile, output);
		  for(ib=0; ib<strlen(outfile); ib++){
		    if(outfile[ib]==' ') outfile[ib]='0';
		  }
		}
		else if(ckpt < 2){
		  strcpy(outfile11,scratch);
		  ic = sprintf(outckpt, "/outckpt_it%08.0d_NS%02.0d_P%03.0d.dat", (it+(itp*NT))+tslast, isa, rank);
		  strcat(outfile11, outckpt);
		  for(ib=0; ib<strlen(outfile11); ib++){
		    if(outfile11[ib]==' ') outfile11[ib]='0';
		  }
		}
	      }

	      if(fmod((double)(it),500.0) == 0.0){
		strcpy(outfile3,scratch);
		ih = sprintf(output3, "/outaverage_it%08.0d_NS%02.0d.dat", (it+(itp*NT))+tslast, isa);
		strcat(outfile3, output3);
		for(ij=0; ij<strlen(outfile3); ij++){
		  if(outfile3[ij]==' ') outfile3[ij]='0';
		}
	      }
	      
	      //if(fmod((double)(it),500.0) == 0.0 && it >= 1000){
	      if(fmod((double)(it),20.0) == 0.0 || it+(itp*NT) == (NSI*NT)-1){
		if(ckpt >= 2){
		  of2 = fopen(outfile,"w");
		}
		else if(ckpt < 2){
		  of11 = fopen(outfile11, "w");
		}
	      }
	      if(fmod((double)(it),500.0) == 0.0){ 
		of3 = fopen(outfile3,"w");
		//if(rank == 0 && it+(itp*NT) == 0){
		  //for Ensight XY plot format
		  //fprintf(of3, "1\n"); //integer number of curves-line1		 
		  //fprintf(of3, "Average phase field vs time"); //Chart title-line2
		  //fprintf(of3, "Time increment\n"); //x-axis-line3
		  //fprintf(of3, "Average Phase Field\n"); //y-axis-line4
		  //fprintf(of3, "1\n"); //number of curve segments-line5
		  //fprintf(of3, "%d\n", (it/500)*(itp+1)); //number of points in plot-line6
		//}
	      }
	      //if(rank==0 && fmod((double)(it),500.0) == 0.0 && it >= 1000){
		if(rank ==0 && fmod((double)(it),20.0) == 0.0 || it+(itp*NT) == (NSI*NT)-1 && ckpt >= 2){
		//if(rank == 0 && it == NT-1){
		fprintf(of2,"TITLE=\"%s\"\n", title);
		fprintf(of2,"VARIABLES=\"X\" \"Y\" \"Z\" \"XI_REAL\" \"XI_IMAG\" \"XI_SUM\"\n");
		fprintf(of2,"ZONE   I = %d J = %d K = %d\n", N1, N2, N3);
	      }
	      
	      xi_ave = 0.0;
	      
	      //if(rank==0 && fmod((double)(it),100.0) == 0.0){
	      //printf("GL time step, %d %d %d\n", it, itp, it+(itp*NT));
	      //}

	      for(i=0;i<lN1;i++)
		for(j=0;j<N2;j++)
		  for(k=0;k<N3;k++)
		    {
		      na0 = 2*(i*N2*N3 + j*N3 + k + isa*lN1*N2*N3);
		      index = i*N2*N3 + j*N3 + k + isa*lN1*N2*N3;
		      na = 2*(i*N2*N3 + j*N3 + k);
		      na1 = na0+1;

		      //Ginzburg-Landau Equation for real and imag parts
		      
		      if(xo[index] == 0.0){
			xi[na0] = xi[na0]-((CD*dtt)*(data_real[index].re/(nsize) - tau[isa] + dE_core[index]));
			xi[na1] = xi[na1]-((CD*dtt)*(data_real[index].im/(nsize)));
			xi_sum[na] += xi[na0];
			xi_sum[na+1] += xi[na1];
		      }
		      data_fftw[index].re = xi[na0];
		      data_fftw[index].im = xi[na1];
		      xi_ave += xi[na0];
	      
		      T7 = MPI_Wtime();
		      //if(fmod((double)(it),500.0) == 0.0 && it >= 1000){
		      if(fmod((double)(it),20.0) == 0.0 || it+(itp*NT) == (NSI*NT)-1){
			//if(it == NT-1){
			if(ckpt >= 2){
			  fprintf(of2, "%d %d %d %lf %lf %lf \n",lxs+i, j,k, xi[na0], xi[na1], xi_sum[na]);
			}
			else if(ckpt < 2){
			  fprintf(of11, "%lf %lf %lf \n", xi[na0], xi[na1], xo[index]);
			}
		      }
		      TT7 += (MPI_Wtime() - T7);
		    } /*end i,j,k*/
	      //if(rank == 0){
	      //printf("xi_real %lf, xi_imag %lf\n", xi[0], xi[1]);
	      //}
	      MPI_Reduce(&xi_ave, &xiave, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	      if(rank == 0){
		xiave /= (N1*N2*N3);
		if(fmod((double)(it),500.0) == 0.0 || it+(itp*NT) == (NSI*NT)-1){
		  //fprintf(of3, "%d %d %lf\n", it+(itp*NT), (it+(itp*NT))+tslast, xiave); 
		}
		xiave = 0.0;
	      }
	    } /* end isa */
	  TT6 += (MPI_Wtime() - T6);
	  //if(rank==0 && fmod((double)(it),100.0) == 0.0){
	  //printf("time loop end, %d %d %d %d\n", it, itp, it+(itp*NT),(it+(itp*NT))+tslast );
	  //}
	}  /*end it*/
      TT4 = MPI_Wtime() - T4;
    } /*end itp*/

  /*free malloc'ed memory*/

  free(data_fftw);
  free(work);
  free(data_real);
  free(temp_data);
  free(xi);
  free(xi_sum);
  free(fx);
  free(fy);
  free(fz);
  free(f);
  free(r);
  free(BB);
  free(data_core);
  free(xo);
  free(fcore);
  free(df1core);
  free(df2core);
  free(df3core);
  free(dE_core);
  free(tau);
  
  for(b1=0;b1<ND;b1++){
    for(b2=0;b2<ND;b2++){
      free(eps[b1][b2]);
    }
    free(eps[b1]);

    free(sigma[b1]);
    free(avesigma[b1]);
    free(ave_sigma[b1]);
    free(ave_sigch[b1]);
    free(avepsd[b1]);
    free(avepst[b1]);
    free(ave_epsd[b1]);
    free(avepsts[b1]);
    free(aveps[b1]);
    free(ave_eps[b1]);
    free(ave_epst[b1]);
    free(tchsig[b1]);
    free(tch_sig[b1]);
    free(cchsig[b1]);
    free(cch_sig[b1]);

    free(xn[b1]);
    free(xb[b1]);
  }

  free(xn);
  free(xb);

  free(sigma);
  free(avesigma);
  free(ave_sigma);
  free(ave_sigch);
  free(avepsd);
  free(avepst);
  free(ave_epsd);
  free(avepsts);
  free(aveps);
  free(ave_eps);
  free(ave_epst);
  free(tchsig);
  free(tch_sig);
  free(cchsig);
  free(cch_sig);

  free(eps);

  fftwnd_mpi_destroy_plan(plan);
  fftwnd_mpi_destroy_plan(iplan);
  
  if(ckpt >= 2){
    fclose(of2);
  }
  else if(ckpt < 2){
    fclose(of11);
  }
  fclose(of3);
  if(rank == 0){
    fclose(of8);
  }

  TT0 = MPI_Wtime() - T0;

  MPI_Barrier(MPI_COMM_WORLD);
  
  MPI_Reduce(&TT0, &T0_ave, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(&TT1, &T1_ave, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(&TT2, &T2_ave, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(&TT3, &T3_ave, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(&TT4, &T4_ave, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(&TT5, &T5_ave, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(&TT6, &T6_ave, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(&TT7, &T7_ave, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(&TT8, &T8_ave, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(&TT9, &T9_ave, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  MPI_Barrier(MPI_COMM_WORLD);

  if(rank == 0)
    {
      T0_ave /= np;
      T1_ave /= np;
      T2_ave /= np;
      T3_ave /= np;
      T4_ave /= np;
      T5_ave /= np;
      T6_ave /= np;
      T7_ave /= np;
      T8_ave /= np;
      T9_ave /= np;

      printf("Dimension = %d x3 and NP = %f \n", N1, np);
      printf("Time Summary - Average Times\n");
      printf("Time to create B-matrix = %8.3f seconds\n", T8_ave);
      printf("Time to create initial data = %8.3f seconds\n", T3_ave);
      printf("Time for time stepping loop = %8.3f seconds\n", T4_ave);
      printf("Time to take forward FFT = %8.3f seconds\n", T1_ave);
      printf("Time to multiply data by B-matrix = %8.3f seconds\n", T5_ave);
      printf("Time to take inverse FFT = %8.3f seconds\n", T2_ave);
      printf("Time to re-arrange data for FFTs = %8.3f seconds\n", T9_ave);
      printf("Time to update data = %8.3f seconds\n", T6_ave - T7_ave);
      printf("Time to write output file = %8.3f seconds\n", T7_ave);
      printf("Time for entire code to run = %8.3f seconds\n", T0_ave);
    }

  MPI_Finalize();
  return 0;		
}

/********************************************************************/

  /* subroutines*/

/*********************************************************************/

void initial_sxtal(int rank, int lN1, int lxs, double *xi, double *xo, fftw_complex *data_fftw, int iflag, double *xi_sum, double obsden, double dslip, double b, int N1, int N2, int N3, int ND, int NS, int nloops, int radius2, double nu, char *scratch, char *title, int channel)
{

  /*iflag == 1 -> 1 dislocation loop on 1 slip system
    iflag == 2 -> 2 dislocations on 2 slip systems
    iflag == 3 -> 4 obstacles (set for 128x128x128)
    iflag == 4 -> interface (passivated film), random distribution of obstacles
    iflag == 5 -> interface with columnar obstacle lines (like iflag==4)
    iflag == 6 -> interface (passivated film), random dislocation distribution (like iflag==4)
    iflag == 7 -> infinitely long straight unit dislocation on a 111 plane (1 ss) dislocation line parallel to y-axis -> cflag == 1 (stress/strain verification)
    iflag == 8 -> infinitely long straight unit dislocation on a 111 plane (3 ss) dislocation line parallel to y-axis -> cflag == 2 (paritals-verification)
    iflag == 9 -> single grain (3 grain boundaries), (stress/strain curves)
    iflag == 10 -> interface (passivated film) with columnar grain boundaries generated by matlab code -- grid.m
    iflag == 11 -> grain with a notch to nucleate dislocations (partials crossing grain)*/

  void init_genrand(unsigned long s);
  double genrand_real3(void);

  int is, i, j, k, im, jm, km, ism, ir, ia, ib, indexgb, grainb, num, count;
  int na0, na, na1, index, indexm, *nodes, rtn_val0, rtn_val1, layer;
  double nlx, nly, c0, c1, a, alpha, zeta, eta, d;
  unsigned long seed;
  FILE *of0, *fgrid;
  char infile[100], input[100], c[10];

  seed = 243578 + lxs;
  num = 0.0;
  count = 0.0;
  a = 1.0; //approximate core region in iflag == 7
  d = 2.0;
  zeta = (d/(2.0*(1-nu))); //approximate core region in iflag == 7 PN-model edge
  eta = d/2.0;  //approximate core region in iflag == 7 PN-model screw
  alpha = 1.0;
  layer = (N2-channel)/2; //thickness of passivation layer

  init_genrand(seed);  /* seed the random number generator */  

  if(rank == 0){
    printf("Initial data\n");
    printf("Channel Thickness %d, layer %d\n", channel, layer);
  }

  if(iflag == 10){
    nodes = (int*) malloc(N1*N2*sizeof(int));
    if(rank == 0){
      fgrid = fopen("inputnodes.dat", "r");
      if(fgrid == NULL){
	printf("File 'inputnodes.dat' could not be opened\n");
	perror("inputnodes.dat");
	exit(1);
      }
      else{
	while(fgets(c, 10, fgrid) != NULL){
	  //printf("node is: %d \n",atoi(c));
	  while(num != atoi(c) && num <= N1*N2){
	    num++;
	  }
	  if(num == atoi(c)){
	    nodes[count] = num;
	    //printf("node is: %d, %d %d %d\n", atoi(c), num, count, nodes[count]);
	    count++;
	  }
	}
      }
    }
    rtn_val1 = MPI_Bcast(&count, 1, MPI_INT, 0, MPI_COMM_WORLD);
    rtn_val0 = MPI_Bcast(nodes, count, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    printf("rank %d, rtn_val %d, %d %d %d %d\n", rank, rtn_val0, rtn_val1, count, nodes[0], nodes[count]);

    indexgb = 0;
    while(indexgb <= count){
      grainb = nodes[indexgb];
      printf("node: %d, %d, %d, %d, rank %d\n", grainb, nodes[indexgb], indexgb, count, rank);
      //if(num == atoi(c)){
      jm = (grainb-1)/N2;
      im = grainb-jm*N1-1;
      //printf("%d, im=%d, jm=%d \n", grainb, im, jm);
      if(im < N1 && jm < N2 && im >= lxs+i && im < (lxs+i)+lN1){
	for(ism=0;ism<NS;ism++)
	  for(km=0;km<N3;km++)
	    {
	      indexm = (im-lxs)*N2*N3+jm*N3+km+ism*lN1*N2*N3;
	      xo[indexm] = 1.0;
	    }
	//printf("indexm %d, rank %d, im %d, jm %d lsx %d\n", indexm, rank, im, jm, lxs);
      }
      //}
      indexgb++;
    }
  }

  for(is=0;is<NS;is++)
    {

      /*open input file*/
      strcpy(infile,scratch);
      ia = sprintf(input, "/input_NS%02.0d_P%03.0d.dat",is, rank);
       strcat(infile,input);
      for(ib=0; ib<strlen(infile); ib++){
	if(infile[ib]==' ') infile[ib]='0';
      }
      
      of0 = fopen(infile,"w");
      
      if(rank==0){
	fprintf(of0,"TITLE=\"%s\"\n", title);
	fprintf(of0,"VARIABLES=\"X\" \"Y\" \"Z\" \"XI_REAL\" \"XI_IMAG\" \"XO\"\n");
	fprintf(of0,"ZONE  I = %d J = %d K = %d \n",N1, N2, N3);
	printf("made it to file header print");
      }

      //printf("is %d\n",is);
      for(i=0;i<lN1;i++)	
	for(j=0;j<N2;j++)
	  for(k=0;k<N3;k++)
	    {
	      na0 = 2*(i*N2*N3+j*N3+k+is*lN1*N2*N3);
	      na = 2*(i*N2*N3+j*N3+k);
	      index = i*N2*N3+j*N3+k+is*lN1*N2*N3;
	      na1 = na0+1;
	      xi[na0] = 0.0;      /*(k+1)+(j+1)*10.0+(i+1)*100.0*/
	      xi[na1] = 0.0;
	      xo[index] = 0.0;

	      if(k==N3/2){
		if(iflag == 1){  //1 slip system
		  ir = (lxs+i-N1/2)*(lxs+i-N1/2)+(j-N2/2)*(j-N2/2);
		  if(ir<=N1/2){
		    xi[na0]=1.0;
		    xi_sum[na] = xi_sum[na] + xi[na0];
		  }
		  //write input file
		  fprintf(of0, "%d %d %lf %lf %lf\n", lxs+i, j, xi[na0], xi[na1], xi_sum[na]); 
		} //iflag == 1

       		else if(iflag == 2){  //2 slip system 2 dislocations
		  if(is==0){
		    ir = (lxs+i-N1/4)*(lxs+i-N1/4) + (j-N2/2)*(j-N2/2);  
		  }
		  else{
		    ir = (lxs+i-(3*N1/4))*(lxs+i-(3*N1/4)) + (j-N2/2)*(j-N2/2);
		  }
		  if(ir<=N1/4){
		    xi[na0]=1.0;
		    xi_sum[na] += xi[na0];
		  }
		  //write input file
		  fprintf(of0, "%d %d %lf %lf %lf\n", lxs+i, j, xi[na0], xi[na1], xi_sum[na]);
		} //iflag == 2

		else if(iflag == 3){ //Four obstacles (32x32x32)
		  ir = (lxs+i-N1/2)*(lxs+i-N1/2)+(j-N2/2)*(j-N2/2);
		  if(ir<=N1/2){
		    xi[na0]=1.0;
		    //xi_sum[na] = xi_sum[na] + xi[na0];
		  }  
		  if(lxs+i== 16 /*64*/){
		    if(j== 11/*55*/ || j== 21 /*73*/){
		      xo[index] = 1.0;
		      xi[na0] = 0.0;
		    }
		  }
		  else if(j== 16 /*64*/){
		    if(lxs+i== 11 /*56*/ || lxs+i== 21 /*72*/){
		      xi[na0] = 0.0;
		      xo[index] = 1.0;
		    }
		  }
		  //write input file
		  fprintf(of0, "%d %d %lf %lf %lf\n", lxs+i, j, xi[na0], xi[na1], xi_sum[na]);
		} //iflag == 3	
	      } // k == N3/2 
  
	      if(iflag == 4){ //interface with obstacles
		xo[index] = genrand_real3();
		//printf("%d %d %d %d %lf\n",i, j, k, index, xo[index]);
		if(xo[index] <= obsden){
		  xo[index] = 1.0;
		}
		else{
		  xo[index] = 0.0;
		}
		if(k <= (N3-1)/6 || k >= 5*(N3-1)/6){
		  xo[index] = 1.0;
		}
		/*if(j <= (N2-1)/6 || j >= 5*(N2-1)/6){
		  xo[index] = 1.0;
		  }*/

		fprintf(of0, "%d %d %d %lf %lf %lf\n", lxs+i, j, k, xi[na0], xi[na1], xo[index]);
	      } //iflag == 4

	      if(iflag == 5){ //interface with columnar obstacles
		nlx = (2.0*N1)/((obsden*N1)+1);
		nlx = floor(nlx + 0.5);
		nly = (2.0*N2)/((obsden*N2)+1);
		nly = floor(nly +0.5);
		//printf("%d %d %d %d %lf\n",i, j, k, index, xo[index]);
		if(fmod(lxs+i,nlx) == 0 || fmod(j,nly) == 0){
		  xo[index] = 1.0;
		}
		else if(lxs+i == (N1-1) || j == (N2-1)){
		  xo[index] = 1.0;
		}
		else{
		  xo[index] = 0.0;
		}
		if(k <= (N3-1)/6 || k >= 5*(N3-1)/6){
		  xo[index] = 1.0;
		}

		fprintf(of0, "%d %d %d %lf %lf %lf\n", lxs+i, j, k, xi[na0], xi[na1], xo[index]);

	      } //iflag == 5

	      if(iflag == 6){  //interface with random dislocation distribution
		if(fmod((double)(k),10.0) == 0.0){
		  xi[na0] = genrand_real3();
		  //printf("%d %d %d %d %lf\n",i, j, k, index, xo[index]);
		  if(xi[na0] <= obsden){
		    xi[na0] = 1.0;
		  }
		  else{
		    xi[na0] = 0.0;
		  }
		}
		else{
		  xi[na0] = 0.0;
		}
		if(k <= (N2-1)/6 || k >= 5*(N2-1)/6){
		  xo[index] = 1.0;
		  xi[na0] = 0.0;
		}
		else{
		  xo[index] = 0.0;
		}

		fprintf(of0, "%d %d %d %lf %lf %lf\n", lxs+i, j, k, xi[na0], xi[na1], xo[index]);

	      } //iflag == 6
	      
	      if(iflag == 7){ //infinitely long dislocation 1 slip system dislocation line parallel to y-axis, core region approimately by a = 1. -> cflag == 1 (stress/strain-verification) 
		if((lxs+i) > ((N1/2)-4) && (lxs+i) < ((N1/2)+4) && k == N3/2){
		  //if(k == N3/2){
		  //xi[na0] = 1.0-(1.0/(1+exp(-(a*((lxs+i)-(N1/2))))));
                  //xi[na0] = 0.5+(-(1/pi)*atan(((lxs+i)-(N1/2))/zeta)); //PN-model edge
                  //xi[na0] = 0.5+(-(1/pi)*atan(((lxs+i)-(N1/2))/eta)); //PN-model screw
                  //if((lxs+i) <= N1/2){
		  xi[na0] = 1.0; //step function
		    //}

		}
                //to add grain boundaries, see how they change stress state.
                //if(lxs+i < 5 || j < 5 || k < 5){
		//xi[na0] = 0.0;
		//xo[index] = 1.0;
		//}
		fprintf(of0, "%d %d %d %lf %lf %lf\n", lxs+i, j, k, xi[na0], xi[na1], xo[index]);
	      } //iflag == 7

	      if(iflag == 8){  //infinitely long dislocation 3 slip systems dislocation line parallel to y-axis -> cflag == 2 (partials-verification)
		if(is == 1){

                  //printf("Channel Thickness iflag %d, layer %d\n", channel, layer);
		  //if((lxs+i) > (N1/4)+(1.0/alpha) && (lxs+i) < (3*N1/4)-(1.0/alpha) && k == N3/2){
		  if((lxs+i) > (N1/4) && (lxs+i) < (3*N1/4) && k == N3/2){
		  //xi[na0] = 0.5;
		  //}
                  //if((lxs+i) > (N1/4)+10.0 && (lxs+i) < (3*N1/4)-10.0 && k == N3/2){
		    xi[na0] = 1.0;
		  }
		  //else if((lxs+i) >= (N1/4) && (lxs+i) <= (N1/4)+(1.0/alpha) && k == N3/2){
		  //xi[na0] = (alpha)/2*((lxs+i)-((N1/2)-(1/alpha)));
		  //xi[na0] = (alpha)*((lxs+i)-(N1/4));
		  //}
		  //else if((lxs+i) >= (3*N1/4)-(1.0/alpha) && (lxs+i) <= (3*N1/4) && j == N2/2){
		  //xi[na0] = 1.0 - (alpha)*((lxs+i)-((3*N1/4)-(1.0/alpha)));
		  //}
		  //else{
		  //xi[na0] = 0.0;
		  //}
		}
		else if(is == 0){
		  xi[na0] = 0.0;
		}
		else if(is == 2){
		  xi[na0] = 0.0;
		}
		//if((lxs+i) <= N1/2 && j == 0 && k == N3/2){
		//if(j == 0 || j == N2-1){
		//if(channel != N1){
		//if(j <= layer  || j >= N2-layer){
		//if(j < layer  || j > N2-layer){
		      //if((lxs+i) <= 60  || (lxs+i) >= N1-60){
		      //xi[na0] = 0.0;
		//}
		    //xo[index] = 1.0;
		//}
		//}
		//if(j == (N2/2)-50 || j == (N2/2)+50){
		//xo[index] = 1.0;
		//}
		fprintf(of0, "%d %d %d %lf %lf %lf\n", lxs+i, j, k, xi[na0], xi[na1], xo[index]);
	      } //iflag == 8

	      if(iflag == 9){ //one grain (inpenatrable grain boundary), every 4th plane is active, number of initial loops is variable (stress/strain curves)
		if(fmod((double)(k-2),4.0) == 0.0){
		  if(nloops == 1){
		    ir = (lxs+i-N1/2)*(lxs+i-N1/2)+(j-N2/2)*(j-N2/2);
		    if(ir<=radius2){
		      xi[na0] = 1.0;
		    }
		    else{
		      xi[na0] = 0.0;
		    }
		  }
		  if(nloops >1){
		    for(c0=0;c0<(nloops/2);c0++)
		      for(c1=0;c1<(nloops/2);c1++){
			ir = (lxs+i-((2*c0+1)*N1)/nloops)*(lxs+i-((2*c0+1)*N1)/nloops)+(j-((2*c1+1)*N2)/nloops)*(j-((2*c1+1)*N2)/nloops);
			if(ir<=radius2){
			  xi[na0] = 1.0;
			}
		      }
		  }
		}
		else{
		  xi[na0] = 0.0;
		  xo[index] = 0.0;
		}
		//if((lxs+i) == 0){
		if(lxs+i == 0 || j == 0 || k == 0){
		  xi[na0] = 0.0;
		  xo[index] = 1.0;
		}

		fprintf(of0, "%d %d %d %lf %lf %lf\n", lxs+i, j, k, xi[na0], xi[na1], xo[index]);

	      } //iflag == 9

	      if(iflag == 10){ //passivated films with columnar grains--matlab generated obstacle array
		//if(fmod((double)(k),10.0) == 0.0){
		xi[na0] = genrand_real3();
		//printf("%d %d %d %d %lf\n",i, j, k, index, xo[index]);
		if(xi[na0] <= obsden){
		  xi[na0] = 1.0;
		}
		else{
		  xi[na0] = 0.0;
		}
		//}
		if(k <= (N3-1)/6 || k >= 5*(N3-1)/6){
		  xo[index] = 1.0;
		}
		fprintf(of0, "%d %d %d %lf %lf %lf\n", lxs+i, j, k, xi[na0], xi[na1], xo[index]);
	      } //iflag == 10

	      if(iflag == 11){ //one grain (inpenatrable grain boundary), notch to nucleate dislocation (partials across a grain)
		//if(k != N3/2){
		//xi[na0] = 0.0;
		//xo[index] = 1.0;
		//}
		if(lxs+i < 5 || j < 5 || k < 5){
		  if(is == 1 /*&& k >= 5*/){
		    xi[na0] = 0.0;
		  }
		  xo[index] = 1.0;
		}
		//else if(/*lxs+i == N1/2*/ /*&& j == N2/2*/ /*&&*/ k == N3/2 && is == 1){
		  //if(is == 1 && j >= N2/4 && j <= 3*N2/4){
		  //xi[na0] = 1.0;
		  //if(j == N2/4 || j == 3*N2/4){
		  //xo[index] = 1.0;
		  //}
		  //}
		  //if((lxs+i) <= (N1/32)){
		  //ir = (((lxs+i)-5.0)*((lxs+i)-5.0))+((j-N2/2)*(j-N2/2));
		  //if(ir<=N1/2 /*&& (lxs+i)!= 0*/){
		  //if(ir<=25 /*&& (lxs+i)!= 0*/){
		  //xi[na0] = 1.0;
		  //}
		  //if((lxs+i) == N1/N1){
		  //xo[index] = 1.0;
		  //}
		  //xo[index] = 1.0;
		//}
		if(k == N3/2-1){//second ledge
		  //ir = (((lxs+i)-5.0)*((lxs+i)-5.0))+((j-N2/2)*(j-N2/2));
		  ir = (((lxs+i)-(N1-1))*((lxs+i)-(N1-1)))+((j-N2/2)*(j-N2/2));
		  if(ir<=25){ //second ledge
		    if((lxs+i)== (N1-1) || (lxs+i)== (N1-2)){
		    //if((lxs+i)>=5){
		    ///if((lxs+i)== 5 || (lxs+i)== 6){
		      xo[index] = 1.0;
		      if(is == 1 /*&& k >= 5*/){
			xi[na0] = 1.0;
		      }
		    }
		  }
		}
		if(k == N3/2){
		//ir = (((lxs+i)-(N1-1))*((lxs+i)-(N1-1)))+((j-N2/2)*(j-N2/2));
		  //if(ir<=N1/2){
		  //if(ir<=25){
		//if((lxs+i)== (N1-1) || (lxs+i)== (N1-2)){
		//xo[index] = 1.0;
		//if(is == 1){
		//	xi[na0] = 1.0;
		//}
		//}
		    ///}
		  //ledge
		  ir = (((lxs+i)-5.0)*((lxs+i)-5.0))+((j-N2/2)*(j-N2/2));
		  if(ir<=25){
		    //if((lxs+i)>=5){
		    if((lxs+i)== 5 ||(lxs+i)== 6){
		      xo[index] = 1.0;
		      if(is == 1 /*&& k >= 5*/){
			//if(is == 1 && k == (N3-1)/2 && j >= 7){
		  	xi[na0] = 1.0;
		      }
		    }
		  }
		  //ir = (((lxs+i)-(N1-1))*((lxs+i)-(N1-1)))+((j-N2/2)*(j-N2/2));
		  //if(ir<=25){ //second ledge
		  //if((lxs+i)== N1-1 || (lxs+i)== N1-2){
		  //  xo[index] = 1.0;
		  //  if(is == 1 /*&& k >= 5*/){
		  //if(is == 1 && k == N3/2 && j >= 5){
		  //	xi[na0] = 1.0;
		  //  }
		  //}
		  //}
		}
		fprintf(of0, "%d %d %d %lf %lf %lf\n", lxs+i, j, k, xi[na0], xi[na1], xo[index]);

	      } //iflag == 11

	      data_fftw[index].re = xi[na0];
	      data_fftw[index].im = xi[na1];
	    } //ijk
    } //is
  
  fclose(of0);
  if(iflag == 10){
    free(nodes);
    if(rank == 0){
      fclose(fgrid);
    }
  }

  if(rank == 0){
    printf("Leaving Initial\n");
  }

  
  MPI_Barrier(MPI_COMM_WORLD);

  return;
}

/******************************************************************/

void avestrain(double **avepsd, double **avepst, double ***eps, double *xi, int nsize, double **sigma, double S11, double S12, double S44, double mu, int lN1, double **ave_epsd, int rank, double **avepsts, int N1, int N2, int N3, int ND, int NS)
 {

#define 	DELTA(i, j)   ((i==j)?1:0)

   int i, j, k, l, is, nb, ida, idb;
   double S[ND][ND][ND][ND];

   /*set S matrix*/	
   for (i=0; i<ND; i++)
     for (j=0; j<ND; j++) 
       for (k=0; k<ND; k++) 
	 for (l=0; l<ND; l++) 
	   {
	   S[i][j][k][l] = S44/4 * (DELTA(i,k)*DELTA(j,l)+DELTA(i,l)*DELTA(j,k))+S12*DELTA(i,j)*DELTA(k,l);
	   }
   
   /*calculating average stress free strain, avepsd*/

   //for(is=0;is<NSV;is++)
   for(is=0;is<NS;is++)
     {	
       for (ida=0; ida<ND; ida++)
	 for (idb=0; idb<ND; idb++) 
	   {
	     for(i=0;i<lN1;i++)
	       for(j=0;j<N2;j++)
		 for(k=0;k<N3;k++)
		   {
		     nb = 2*(k + j*N3 + i*N3*N2+ is*lN1*N2*N3);		
		     //if (is < NS){
		     //avepsd[ida][idb]  += eps[is][ida][idb] * xi[nb]/nsize;
		     avepsd[ida][idb]  += eps[is][ida][idb] * xi[nb];
		     //}		
		     //if (is >= NS){
		     //avepsd[ida][idb] += epsv[is][ida][idb] * xi[nb]/nsize;
		     //}
		   }
	     //printf("avepsd[ida][idb] %lf, rank %d, ida %d, idb %d\n", avepsd[ida][idb], rank, ida, idb);

	     MPI_Allreduce(&avepsd[ida][idb], &ave_epsd[ida][idb], 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

	     //printf("ave_epsd[ida][idb] %lf, rank %d, ida %d, idb %d, ave_eps/nisze %lf\n", ave_epsd[ida][idb], rank, ida, idb, ave_epsd[ida][idb]/nsize);
	   
	     ave_epsd[ida][idb] = ave_epsd[ida][idb]/nsize;
	   }
     }

   /*calculating microscopic strain, avepst*/

   for(i=0;i<ND;i++)
     for(j=0;j<ND;j++)
       {
	 for(k=0;k<ND;k++)
	   for(l=0;l<ND;l++)
	     {
	       avepst[i][j] += S[i][j][k][l]*sigma[k][l]*mu;
	       avepsts[i][j] += S[i][j][k][l]*sigma[k][l]*mu;
	       //printf("avepst[i][j] %10.0lf, rank %d, i %d, i %d, S[i][j][k][l] %10.0lf, k %d, l %d, mu %lf, sigma[k][l] %lf\n", avepst[i][j], rank, i, j, S[i][j][k][l], k, l, mu, sigma[k][l]);
	       //avepst[i][j] += S[i][j][k][l]*sigma[k][l];
	     }

	 avepst[i][j] += ave_epsd[i][j];
       }	
   
   return;   
 }

/******************************************************************/

void strain(fftw_complex *data_strain, double *data_eps, fftw_complex *data_fftw, double *FF, double d1, double d2, double d3, double size, int it, int itp, double **avepst, int lxs, int lN1, int nsize, fftw_complex *work_strain, int rank, int N1, int N2, int N3, int ND, int NS, int NT, int NSI, char *scratch, char *title, int tslast)
{
  int i, j, k, l, is, na0, na1, nb, psys, ida, idb, index, index2;
  int na11, na12, na13, na21, na22, na23, na31, na32, na33, ia, ib;
  fftwnd_mpi_plan iiplan;
  FILE *of6;
  char outfile6[100], output6[100];
  
  iiplan = fftw3d_mpi_create_plan(MPI_COMM_WORLD, N1, N2, N3, FFTW_BACKWARD, FFTW_ESTIMATE);

  /*open output file for strain field*/
  //if(it == NT-1){
  if(fmod((double)(it+(itp*NT)), 20.0) == 0.0 || it+(itp*NT) == (NSI*NT)-1){
  for(is=0;is<NS;is++)
    {
      strcpy(outfile6,scratch);
      ia = sprintf(output6, "/outstrain_it%08.0d_P%03.0d.dat", (it+(itp*NT))+tslast, rank);
      strcat(outfile6,output6);
      for(ib=0; ib<strlen(outfile6); ib++){
	if(outfile6[ib]==' ') outfile6[ib]='0';
      }
      
      of6 = fopen(outfile6,"w");
      
      if(rank==0){
	fprintf(of6,"TITLE=\"%s\"\n", title);
	fprintf(of6,"VARIABLES=\"X\" \"Y\" \"Z\" \"EPS11\" \"EPS12\" \"EPS13\" \"EPS21\" \"EPS22\" \"EPS23\" \"EPS31\" \"EPS32\" \"EPS33\"\n");
	//fprintf(of6,"VARIABLES=\"X\" \"Y\" \"EPS11\" \"EPS12\" \"EPS13\" \"EPS21\" \"EPS22\" \"EPS23\" \"EPS31\" \"EPS32\" \"EPS33\"\n");
	fprintf(of6,"ZONE  I = %d J = %d K = %d \n", N1, N2, N3);
	//fprintf(of6,"ZONE  I = %d J = %d \n", N1, N3);
      }
    }
  }
  
  /*initialize*/

  for (i=0; i<lN1*N2*N3*ND*ND; i++)
    {
      data_strain[i].re = 0.0;
      data_strain[i].im = 0.0;
    }

  for (i=0; i<2*lN1*N2*N3*ND*ND; i++)
     {
       data_eps[i] = 0.0;
     }

  /*calculate the total strain */

  //for(is=0;is<NSV;is++)	
  for(is=0;is<NS;is++)
    {	
      for (ida=0; ida<ND; ida++) 
	for (idb=0; idb<ND; idb++)
	  {
	    for(i=0;i<lN1;i++)
	      for(j=0;j<N2;j++)
		for(k=0;k<N3;k++)
		  {
		    index = i*N2*N3 + j*N3 + k + is*lN1*N2*N3;
		    index2 = i*N2*N3 + j*N3 + k + ida*lN1*N2*N3 + idb*lN1*N2*N3*ND; 
		    //if(is<NS){
		    nb = k + j*N3 + i*N2*N3 + is*lN1*N2*N3 + ida*lN1*N2*N3*NS + idb*lN1*N2*N3*NS*ND;
		    //if(rank == 0){
		    //printf("%d %d %d %d %d %d %d %lf \n",is, nb, i, j, k, ida, idb ,FF[nb]);
		    //}
		    data_strain[index2].re += data_fftw[index].re * FF[nb];
		    data_strain[index2].im += data_fftw[index].im * FF[nb];
		    //}
		    //else{
		    //nb = k + j*N3 + i*N2*N3 + (is-NS)*lN1*N2*N3 + ida*lN1*N2*N3*NV + idb*lN1*N2*N3*NV*ND;		
		    //data_strain[na+1] += data_fftw[na1] * FFv[nb];
		    //data_strain[na+2] += data_fftw[na2] * FFv[nb];
		    //}
		  }
	  }
      
    }
	
  for(i=0;i<ND;i++){
    for (j=0;j<ND;j++){
      psys = i*lN1*N2*N3 + j*lN1*N2*N3*ND;
      //psys = i*lN1*N2*N3;
      fftwnd_mpi(iiplan, 1, data_strain+psys, work_strain, FFTW_NORMAL_ORDER); /* Inverse FFT (multiple)*/
    }
  }
  for (i=0; i<lN1*N2*N3*ND*ND; i++)
    {
      data_strain[i].re = data_strain[i].re/(nsize);
      data_strain[i].im = data_strain[i].im/(nsize);
    }

  //add in other two terms in strain
  for(ida=0;ida<ND;ida++)
    for (idb=0;idb<ND;idb++)
      {
	for(i=0;i<lN1;i++)
	  for(j=0;j<N2;j++)
	    for (k=0;k<N3;k++)
	      {	    
		index = k + j*N3 + i*N2*N3 + ida*lN1*N2*N3 + idb*lN1*N2*N3*ND;
		data_strain[index].re += avepst[ida][idb];
	      }
      }
						
  for(ida=0;ida<ND;ida++)
    for (idb=0;idb<ND;idb++)
      {
	for(i=0;i<lN1;i++)
	  for(j=0;j<N2;j++)
	    for (k=0;k<N3;k++)
	      {	    
		na0 = 2*(k + j*N3 + i*N2*N3 + ida*lN1*N2*N3 + idb*lN1*N2*N3*ND);
		na1=na0+1;
		index = k + j*N3 + i*N2*N3 + ida*lN1*N2*N3 + idb*lN1*N2*N3*ND;
		index2 = k + j*N3 + i*N2*N3 + idb*lN1*N2*N3 + ida*lN1*N2*N3*ND;
		//eps_ij = 1/2(dui/dxj + duj/dxi) --> ida=i, idb=j
		data_eps[na0] = (data_strain[index].re + data_strain[index2].re)/2.0;
		data_eps[na1] = (data_strain[index].im + data_strain[index2].im)/2.0;
	      }	
      }	
	
  //if(it+(itp*NT) == (NSI*NT)-1){
  //if(it == NT-1){
  if(fmod((double)(it+(itp*NT)), 20.0) == 0.0 || it+(itp*NT) == (NSI*NT)-1){
    //if(itp == NSI-1){
    //if(rank == 0){
    //printf("printing strains %d %d %d %d\n", it, itp, it+(itp*NT), (it+(itp*NT))+tslast);
    //}
    
    for(i=0;i<lN1;i++)
      for(j=0;j<N2;j++)
	for(k=0;k<N3;k++)
	  {	  
	    na0 = 2*(k + j*N3 + i*N2*N3);  /* +i*N1*N2*N3+j*N1*N2*N3*ND);*/
	    na11 = na0 + 2*(0*lN1*N2*N3+0*lN1*N2*N3*ND);
	    na12 = na0 + 2*(0*lN1*N2*N3+1*lN1*N2*N3*ND);
	    na13 = na0 + 2*(0*lN1*N2*N3+2*lN1*N2*N3*ND);
	    na21 = na0 + 2*(1*lN1*N2*N3+0*lN1*N2*N3*ND);
	    na22 = na0 + 2*(1*lN1*N2*N3+1*lN1*N2*N3*ND);
	    na23 = na0 + 2*(1*lN1*N2*N3+2*lN1*N2*N3*ND);
	    na31 = na0 + 2*(2*lN1*N2*N3+0*lN1*N2*N3*ND);
	    na32 = na0 + 2*(2*lN1*N2*N3+1*lN1*N2*N3*ND);
	    na33 = na0 + 2*(2*lN1*N2*N3+2*lN1*N2*N3*ND);
	    
	    fprintf(of6,"%d %d %d %lf %lf %lf %lf %lf %lf %lf %lf %lf \n", lxs+i, j, k, data_eps[na11], data_eps[na12], data_eps[na13], data_eps[na21], data_eps[na22], data_eps[na23],  data_eps[na31], data_eps[na32], data_eps[na33]);
	    //if(j == N2/2){
	    //fprintf(of6,"%d %d %lf %lf %lf %lf %lf %lf %lf %lf %lf \n", lxs+i, k, data_eps[na11], data_eps[na12], data_eps[na13], data_eps[na21], data_eps[na22], data_eps[na23],  data_eps[na31], data_eps[na32], data_eps[na33]);
	    //}
	    //if(rank == 0){
	    //printf("printing %lf %lf %lf\n", data_eps[na12], data_eps[na22], data_eps[na33]);
	    //}
	  }
  }		
  
  if(rank == 0){
    printf("leaving strain\n");
  }

  fftwnd_mpi_destroy_plan(iiplan);

  return;
}

/******************************************************************/

void stress(double *data_epsd, double *data_sigma, fftw_complex *data_strain, double *xi, double ***eps, double C11, double C12, double C44, int it, int itp, double **avesigma, int lxs, int lN1, int nsize, int rank, double **ave_sigma, double **ave_sigch, double **sigma, double **avepsts, int N1, int N2, int N3, int ND, int NS, int NT, int NSI, char *scratch, char *title, int tslast, double **tchsig, double **tch_sig, double **cchsig, double **cch_sig, int channel)
{
#define DELTA(i, j)   ((i==j)?1:0)
	
  int i, j, k, l, m, ida, idb, na, nb, na0, is, ia, ib, index;
  int na11, na12, na13, na21, na22, na23, na31, na32, na33, layer;
  int tcount[ND][ND], ccount[ND][ND], t_count[ND][ND], c_count[ND][ND];
  double C[ND][ND][ND][ND];
  double mu, xnu, young, ll;
  FILE *of7;
  char outfile7[100], output7[100];

  /* set Cijkl*/ 
  
  mu = C44-(2.0*C44+C12-C11)/5.0;
  ll = C12-(2.0*C44+C12-C11)/5.0;
  young = mu*(3*ll+2*mu)/(ll+mu);
  xnu = young/2.0/mu-1.0;
  layer = (N2-channel)/2;

  //if(rank == 0){
  //printf(" stress mu %lf, ll %lf, young %lf, nu %lf\n", mu, ll, young, xnu);
  //}

  /*open output file*/

  strcpy(outfile7,scratch);
  ia = sprintf(output7, "/outstress_it%08.0d_P%03.0d.dat", (it+(itp*NT))+tslast, rank);
  strcat(outfile7,output7);
  for(ib=0; ib<strlen(outfile7); ib++){
    if(outfile7[ib]==' ') outfile7[ib]='0';
  }
  
  of7 = fopen(outfile7,"w");
  
  if(rank==0){
    fprintf(of7,"TITLE=\"%s\"\n", title);
    fprintf(of7,"VARIABLES=\"X\" \"Y\" \"Z\" \"SIG11\" \"SIG12\" \"SIG13\" \"SIG22\" \"SIG23\" \"SIG33\"\n");
    fprintf(of7,"ZONE  I = %d J = %d K = %d \n", N1, N2, N3);
  }
  
  for (i=0; i<ND; i++) 
    for (j=0; j<ND; j++) 
      for (k=0; k<ND; k++) 
	for (m=0; m<ND; m++) 
	  {
	    C[i][j][k][m] = mu * (DELTA(i,k)*DELTA(j,m)+DELTA(i,m)*DELTA(j,k))+ll*DELTA(i,j)*DELTA(k,m);
	  }

  for (i=0; i<2*lN1*N2*N3*ND*ND; i++)
    {
      data_sigma[i] =0;
      data_epsd[i]=0;
    }

  //for(is=0;is<NSV;is++)
  for(is=0;is<NS;is++)
    {		    
      for (ida=0; ida<ND; ida++)
	for (idb=0; idb<ND; idb++) 
	  {
	    for(i=0;i<lN1;i++)
	      for(j=0;j<N2;j++)
		for(k=0;k<N3;k++)
		  {
		    na = 2*(i*N2*N3 + j*N3 + k + ida*lN1*N2*N3 + idb*lN1*N2*N3*ND);
		    nb = 2*(k + j*N3 + i*N3*N2 + is*lN1*N2*N3);		
		    //if (is<NS) 
		    //{
		    data_epsd[na] += eps[is][ida][idb] * xi[nb];
		    //}		
		    //if (is >=NS){
		    //data_epsd[na+1] += epsv[is][ida][idb] * xi[nb];
		    //}
		  }
	  }
    }
  
  for (ida=0;ida<ND;ida++)
    for (idb=0;idb<ND;idb++)
      {
	tcount[ida][idb] = 0;
	ccount[ida][idb] = 0;
	for(i=0;i<lN1;i++)
	  for(j=0;j<N2;j++)
	    for(k=0;k<N3;k++)
	      {
		na = 2*(i*N2*N3 + j*N3 + k + ida*lN1*N2*N3 + idb*lN1*N2*N3*ND);
		for (m=0;m<ND;m++)
		  for (l=0;l<ND;l++)
		    {
		      na0 = 2*(i*N2*N3 + j*N3 + k + m*lN1*N2*N3 + l*lN1*N2*N3*ND);
		      index = i*N2*N3 + j*N3 + k + m*lN1*N2*N3 + l*lN1*N2*N3*ND;
		      data_sigma[na] += C[ida][idb][m][l]*(data_strain[index].re - data_epsd[na0] - avepsts[m][l]);
		      data_sigma[na+1] = 0.0;
		    }
		data_sigma[na]+=sigma[ida][idb]*mu;
		avesigma[ida][idb] += data_sigma[na];
		//if(channel != 0){
		//if((lxs+i) >= layer && (lxs+i) <= (N2-layer)){
		//if(data_sigma[na] > 0.0){
		//tchsig[ida][idb] += data_sigma[na];
		//tcount[ida][idb] += 1;
		//}
		//else if(data_sigma[na] < 0.0){
		//cchsig[ida][idb] += data_sigma[na];
		//ccount[ida][idb] += 1;
		//}
		//}
		//}
	      }

	MPI_Reduce(&avesigma[ida][idb], &ave_sigma[ida][idb], 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	//if(channel != 0){
	//MPI_Reduce(&tchsig[ida][idb], &tch_sig[ida][idb], 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	//MPI_Reduce(&cchsig[ida][idb], &cch_sig[ida][idb], 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	//MPI_Reduce(&tcount[ida][idb], &t_count[ida][idb], 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
	//MPI_Reduce(&ccount[ida][idb], &c_count[ida][idb], 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
	//}

	if(rank == 0){
	  ave_sigma[ida][idb] = ave_sigma[ida][idb]/nsize;
	  //if(channel != 0){
	  printf("avesigma %lf ida %d idb %d\n", /*tchsig %lf cchsig %lf %d %d\n",*/ave_sigma[ida][idb], ida, idb/*, (tch_sig[ida][idb]/(channel*N2*N3))/mu, (cch_sig[ida][idb]/(channel*N2*N3))/mu, t_count[ida][idb], c_count[ida][idb]*/);
	    //tch_sig[ida][idb] = tch_sig[ida][idb]/(t_count[ida][idb]);
	    //cch_sig[ida][idb] = cch_sig[ida][idb]/(c_count[ida][idb]);
	    //}
	  //printf("avesigma %5.10lf ida %d idb %d asig %5.10lf\n",ave_sigma[ida][idb], ida, idb, avesigma[ida][idb]);
	}
      }

  //if(it+(itp*NT) == (NSI*NT)-1){
  if(itp == NSI-1){
    if(rank == 0){
      printf("printing stress %d %d %d %d\n", it, itp, it+(itp*NT), (it+(itp*NT))+tslast);
    }
    
    for(i=0;i<lN1;i++)
      for(j=0;j<N2;j++)
	for(k=0;k<N3;k++)
	  {	  
	    na0 = 2*(k + j*N3 + i*N2*N3);  /* +i*N1*N2*N3+j*N1*N2*N3*ND);*/
	    na11 = na0 + 2*(0*lN1*N2*N3+0*lN1*N2*N3*ND);
	    na12 = na0 + 2*(0*lN1*N2*N3+1*lN1*N2*N3*ND);
	    na13 = na0 + 2*(0*lN1*N2*N3+2*lN1*N2*N3*ND);
	    na21 = na0 + 2*(1*lN1*N2*N3+0*lN1*N2*N3*ND);
	    na22 = na0 + 2*(1*lN1*N2*N3+1*lN1*N2*N3*ND);
	    na23 = na0 + 2*(1*lN1*N2*N3+2*lN1*N2*N3*ND);
	    na31 = na0 + 2*(2*lN1*N2*N3+0*lN1*N2*N3*ND);
	    na32 = na0 + 2*(2*lN1*N2*N3+1*lN1*N2*N3*ND);
	    na33 = na0 + 2*(2*lN1*N2*N3+2*lN1*N2*N3*ND);

	    //fprintf(of7,"%d %d %d %lf %lf %lf %lf %lf %lf\n", lxs+i, j, k, data_sigma[na11], data_sigma[na12], data_sigma[na13], data_sigma[na22], data_sigma[na23], data_sigma[na33]);
	  }
  }

  if(rank == 0){
    printf("leaving stress\n");
  }	
  
  return;
}

/******************************************************************/

void core_ener(int cflag, int it, int rank, int lxs, double An, int lN1, double c0, double c1, double c2, double c3, double c4, double a1, double a3, double **xn, double **xb, fftw_complex *data_fftw, fftw_complex *data_core, double dslip, double b, double *fcore, double *df1core, double *df2core, double *df3core, double *dE_core, double E_core, int itp, double pi, int N1, int N2, int N3, int ND, int NS, int NP, int NT, int NSI, double Bn, double mu, int nsize, double size, double isf, double usf, char *scratch, char *title, int tslast, int oflag)
{
  /*
    -Core Energy Options
    cflag == 1 -> assume perfect dislocations only, none extended
    cflag == 2 -> models all dislocations as extended, incorporates partials
    cflag == 3 -> extended dislocations sine approximation using intrinsic and unstable stacking fault energies as only parameters -- 1D version
    cflag == 4 -> extended dislocations approximated with only isf and usf--3D version
  */

  int i, j, k, isa, index, index1, index2, index3, indexdx, plane, num;
  int ia, ib, ic, id, ie, ig, ih, ij, ik, il,im,in;
  int counter, marker, indexmin, tag,countSF,cSF,count;
  double *delta, *ddelta, dx, mpidel, *f_core, p, Cn, totAR, sfAR, *disreg, *disregexp, *disregexp2;
  double a, phi2, eta1, eta2, eta3;
  FILE *of4, *of5, *of9, *of10, *of11 /**of6, *of7*/;
  char outfile4[100], output4[100], outfile5[100], output5[100], outfile9[100], output9[100],outfile10[100], output10[100],outfile11[100],output11[100]/*outfile6[100], output6[100], outfile7[100], output7[100]*/;
  MPI_Status status;

  delta = (double*) malloc((NP)*(lN1)*(N2)*(N3)*sizeof(double));
  disreg = (double*) malloc((NP)*(lN1)*(N2)*(N3)*sizeof(double));
  disregexp = (double*) malloc((NP)*(lN1)*(N2)*(N3)*sizeof(double));
  disregexp2 = (double*) malloc((NP)*(lN1)*(N2)*(N3)*sizeof(double));
  //ddelta = (double*) malloc((NP)*(lN1)*(N2)*(N3)*sizeof(double));
  ddelta = (double*) malloc((NP)*(lN1)*sizeof(double));
  f_core = (double*) malloc((NP)*(lN1)*(N2)*(N3)*sizeof(double));

  dx = size/N1;
  tag = 1;
  a = 15.0;
  c0 = c0/(mu);
  c1 = c1/(mu);
  c2 = c2/(mu);
  c3 = c3/(mu);
  c4 = c4/(mu);
  a1 = a1/(mu);
  a3 = a3/(mu);
  isf = isf/(mu*dslip*b);
  An = An/(mu*dslip*b);
  Cn = (usf - (isf/2.0))/(mu*dslip*b);
  p = pi/(sqrt(3.0)*(b/1.0E-10));

  /*Open Output Files*/
  /*if(rank == 0){
    strcpy(outfile7,scratch);
    ih = sprintf(output7, "/outcore_it%05.0d.dat", it);
    strcat(outfile7,output7);

    for(ij=0; ij<strlen(outfile7); ij++){
      if(outfile7[ij]==' ') outfile7[ij]='0';
    }
    of7 = fopen(outfile7,"w");
    }*/

  //if(fmod((double)(it),500.0) == 0.0 || it == NT-1){
  //if((fmod((double)(itp),5.0) == 0.0 && it == NT-1) || it+(itp*NT) == (NSI*NT)-1){
  if(fmod((double)(it+(itp*NT)), 200.0) == 0.0 || it+(itp*NT) == (NSI*NT)-1){
    if(rank==0){
      printf("time step outdelta print, %d %d %d %d\n", it, itp, it+(itp*NT), (it+(itp*NT))+tslast);
    }
   
    strcpy(outfile4,scratch);
    ia = sprintf(output4, "/outdelta_it%05.0d_P%03.0d.dat", it+(itp*NT)+tslast, rank);
    strcat(outfile4,output4);

    for(ib=0; ib<strlen(outfile4); ib++){
      if(outfile4[ib]==' ') outfile4[ib]='0';
    }
    of4 = fopen(outfile4,"w");

    strcpy(outfile5,scratch);
    ic = sprintf(output5, "/outdeltadx_it%05.0d_P%03.0d.dat", it+(itp*NT)+tslast, rank);
    strcat(outfile5,output5);

    for(id=0; id<strlen(outfile5); id++){
      if(outfile5[id]==' ') outfile5[id]='0';
    }
    of5 = fopen(outfile5,"w");

    strcpy(outfile9,scratch);
    ik = sprintf(output9, "/outdelta3D_it%05.0d_P%03.0d.dat", it+(itp*NT)+tslast, rank);
    strcat(outfile9,output9);

    for(il=0; il<strlen(outfile9); il++){
      if(outfile9[il]==' ') outfile9[il]='0';
    }
    of9 = fopen(outfile9,"w");

    strcpy(outfile10,scratch);
    ik = sprintf(output10, "/outdelta3D2_it%05.0d_P%03.0d.dat", it+(itp*NT)+tslast, rank);
    strcat(outfile10,output10);

    for(il=0; il<strlen(outfile10); il++){
      if(outfile10[il]==' ') outfile10[il]='0';
    }
    
    of10 = fopen(outfile10,"w");

    strcpy(outfile11,scratch);
    im = sprintf(output11, "/outdelpro_it%05.0d_P%03.0d.dat", it+(itp*NT)+tslast, rank);
    strcat(outfile11,output11);

   for(in=0; il<strlen(outfile11); in++){
      if(outfile11[in]==' ') outfile11[in]='0';
    }
    
    of11 = fopen(outfile11,"w");

    if(rank == 0){
      fprintf(of4,"TITLE=\"%s, %d\"\n", title, it+(itp*NT)+tslast);
      fprintf(of4,"VARIABLES=\"X\" \"DX\" \"DELTA\" \"DISREG\" \"DISREGEXP\" \"DISREGEXP2\" \n");
      fprintf(of4,"ZONE   I = %d\n", N1);

      fprintf(of5,"TITLE=\"%s, %d\"\n", title, it+(itp*NT)+tslast);
      fprintf(of5,"VARIABLES=\"X\" \"DX\" \"DDELTA\"\n");
      fprintf(of5,"ZONE   I = %d\n", N1);

      fprintf(of9,"TITLE=\"%s, %d\"\n", title, it+(itp*NT)+tslast);
      fprintf(of9,"VARIABLES=\"X\" \"Y\" \"DELTA\" \n");
      fprintf(of9,"ZONE   I = %d J = %d\n", N1, N2);

      fprintf(of10,"TITLE=\"%s\"\n", title);
      fprintf(of10,"VARIABLES=\"X\" \"Y\" \"DELTA\" \n");
      fprintf(of10,"ZONE   I = %d J = %d\n", N1, N2/*, N3*/);

      fprintf(of11,"TITLE=\"%s\"\n", title);
      fprintf(of11,"VARIABLES=\"X\" \"Y\" \"DELTA\" \n");
      fprintf(of11,"ZONE   I = %d K = %d\n", N1, N3);

      //for Ensight XY plot format
      //if(it+(itp*NT) == 0){
      //fprintf(of4, "%d\n",(it/500)+1); //integer number of curves-line1
      //}
      //fprintf(of4, "Delta vs. Distance t = %d\n", it+(itp*NT)); //Chart title-line2
      //fprintf(of4, "Distance (b)\n"); //x-axis-line3
      //fprintf(of4, "Phase Field\n"); //y-axis-line4
      //fprintf(of4, "1\n"); //number of curve segments-line5
      //fprintf(of4, "%d\n", N1); //number of points in plot-line6

      //if(it+(itp+NT) == 0){
      //fprintf(of5, "%d\n",(it/500)+1); //integer number of curves-line1
      //}
      //fprintf(of5, "DDelta vs. Distance t = %d\n", it+(itp*NT)); //Chart title-line2
      //fprintf(of5, "Distance (b)\n"); //x-axis-line3
      //fprintf(of5, "Phase Field\n"); //y-axis-line4
      //fprintf(of5, "1\n"); //number of curve segments-line5
      //fprintf(of5, "%d\n", N1); //number of points in plot-line6
    }
  }/*it*/

  if(cflag == 1){ /*Perfect Dislocations*/
    for(isa=0;isa<NS;isa++)
      {
	for(i=0;i<lN1;i++)
	  for(j=0;j<N2;j++)
	    for(k=0;k<N3;k++)
	      {
		index = i*N2*N3 + j*N3 + k + isa*lN1*N2*N3;
		E_core += An*(sin(pi*data_fftw[index].re)*sin(pi*data_fftw[index].re))/N1;
		dE_core[index] = An*pi*sin(2.0*pi*data_fftw[index].re);
	      }/*ijk*/
      }/*isa*/
    if(rank == 0 && it == NT-1){
      printf("Core Energy for this time step %lf\n", E_core);
    }

    if(fmod((double)(it+(itp*NT)), 200.0) == 0.0 || it+(itp*NT) == (NSI*NT)-1){
   //if(fmod((double)(itp),200.0) == 0.0 || it+(itp*NT) == (NSI*NT)-1){
      for(i=0;i<lN1*N2*N3*NP;i++){
	delta[i] = 0.0;
      }
      for(plane=0;plane<NP;plane++)
	{
	  for(i=0;i<lN1;i++)
	    for(j=0;j<N2;j++)
	      for(k=0;k<N3;k++)
		{
		  index = i*N2*N3 + j*N3 + k + plane*lN1*N2*N3;
		  //indexdx = ((lxs+i)+1)*N2*N3 + j*N3 + k + plane*lN1*N2*N3;
		  index1 = i*N2*N3 + j*N3 + k + 0*lN1*N2*N3 + plane*lN1*N2*N3*3;
		  index2 = i*N2*N3 + j*N3 + k + 1*lN1*N2*N3 + plane*lN1*N2*N3*3;
		  index3 = i*N2*N3 + j*N3 + k + 2*lN1*N2*N3 + plane*lN1*N2*N3*3;
		  
		  delta[index] = /*(1.0/2.0)**/(data_fftw[index1].re*xb[0][0] + data_fftw[index2].re*xb[1][0] + data_fftw[index3].re*xb[2][0])*xb[1][0] + (data_fftw[index1].re*xb[0][1] + data_fftw[index2].re*xb[1][1] + data_fftw[index3].re*xb[2][1])*xb[1][1] + (data_fftw[index1].re*xb[0][2] + data_fftw[index2].re*xb[1][2] + data_fftw[index3].re*xb[2][2])*xb[1][2];

		  if(j == N2/2 && k == N3/2){
		    fprintf(of4, "%d %lf %lf\n", lxs+i, (dx)*(lxs+i), delta[index]);
		    //for Ensight
		    //fprintf(of4, "%lf %lf\n", (dx)*(lxs+i), delta[index]);
		  }
		}/*ijk*/
	}/*plane*/

      for(plane=0;plane<NP;plane++)
	{
	  for(i=0;i<lN1;i++)
	    //for(j=0;j<N2;j++)
	    //for(k=0;k<N3;k++)
	    {
	      //if(j == N2/2 && k == N3/2){
	      j = N2/2;
	      k = N3/2;
	      index = i*N2*N3 + j*N3 + k + plane*lN1*N2*N3;
	      //indexdx = i*N2*N3 + (j+1)*N3 + k + plane*lN1*N2*N3;
	      indexdx = (i+1)*N2*N3 + j*N3 + k + plane*lN1*N2*N3;
	      if((i+1) == lN1){
		indexmin = 0*N2*N3 + j*N3 + k + plane*lN1*N2*N3;
		if(rank != 0){
		  MPI_Send(&delta[indexmin], 1, MPI_DOUBLE, rank-1, tag, MPI_COMM_WORLD);
		}
		if((lxs+(i+1)) == N1){
		  mpidel = delta[index];
		  ddelta[i] = (mpidel - delta[index])/dx;
		  goto skip_point;
		}
		MPI_Recv(&mpidel, 1, MPI_DOUBLE, rank+1, tag, MPI_COMM_WORLD, &status);
		ddelta[i] = (mpidel - delta[index])/dx;
	      }
	      else{
		ddelta[i] = (delta[indexdx] - delta[index])/dx;
	      }

	      //ddelta[index] = (delta[indexdx] - delta[index])/dx;
	      //if(lxs+i == N1/2 && k == N3/2){
	      //fprintf(of5, "%d %lf %lf %d %d %lf %lf %lf\n", j, (dslip/N1)*j, ddelta[index], index, indexdx, delta[index], delta[indexdx], dx);

	    skip_point:
	      fprintf(of5, "%d %lf %lf\n", lxs+i, (dx)*(lxs+i), ddelta[i]);
	      //for Ensight
	      //fprintf(of5, "%lf %lf\n", (dx)*(lxs+i), ddelta[index]);
	    }
	}
    }/*it*/ 
  }/*cflag == 1*/

  else if(cflag == 2){ /*Partial Dislocations - full gamma parameterization*/

    counter = 0;
    countSF = 0;
    marker = 0;

    /*as of now the planes need to be in the correct order initially for this to work
      also make sure to call this subroutine before the FFTW functions are called so that data_fftw = xi[na0] or double check that this is true.
      subroutine calculates core energy for each time step, need to called every time step and initialize every time step.*/

    for(isa=0;isa<NS;isa++)
      {
	for(i=0;i<lN1;i++)
	  for(j=0;j<N2;j++)
	    for(k=0;k<N3;k++)
	      {
		index = i*N2*N3 + j*N3 + k + isa*lN1*N2*N3;
		data_core[index].re = data_fftw[index].re;
		data_core[index].im = data_fftw[index].im;
	      }/*ijk*/
      }/*isa*/
    
    for(plane=0;plane<NP;plane++)
      {
	for(i=0;i<lN1;i++)
	  for(j=0;j<N2;j++)
	    for(k=0;k<N3;k++)
	      {
		index = i*N2*N3 + j*N3 + k + plane*lN1*N2*N3;
		index1 = i*N2*N3 + j*N3 + k + 0*lN1*N2*N3 + plane*lN1*N2*N3*3;
		index2 = i*N2*N3 + j*N3 + k + 1*lN1*N2*N3 + plane*lN1*N2*N3*3;
		index3 = i*N2*N3 + j*N3 + k + 2*lN1*N2*N3 + plane*lN1*N2*N3*3;

		/*no derivatives taken use to calculate E_core*/

		fcore[index] = (c0 + c1*(cos(2.0*pi*(data_core[index1].re-data_core[index2].re)) + cos(2.0*pi*(data_core[index2].re-data_core[index3].re)) + cos(2.0*pi*(data_core[index3].re-data_core[index1].re))) + c2*(cos(2.0*pi*(2.0*data_core[index1].re-data_core[index2].re-data_core[index3].re)) + cos(2.0*pi*(2.0*data_core[index2].re-data_core[index3].re-data_core[index1].re)) + cos(2.0*pi*(2.0*data_core[index3].re-data_core[index1].re-data_core[index2].re))) + c3*(cos(4.0*pi*(data_core[index1].re-data_core[index2].re)) + cos(4.0*pi*(data_core[index2].re-data_core[index3].re)) + cos(4.0*pi*(data_core[index3].re-data_core[index1].re))) + c4*(cos(2.0*pi*(3.0*data_core[index1].re-data_core[index2].re-2.0*data_core[index3].re)) + cos(2.0*pi*(3.0*data_core[index1].re-2.0*data_core[index2].re-data_core[index3].re)) + cos(2*pi*(3.0*data_core[index2].re-data_core[index3].re-2.0*data_core[index1].re)) + cos(2.0*pi*(3.0*data_core[index2].re-2.0*data_core[index3].re-data_core[index1].re)) + cos(2.0*pi*(3.0*data_core[index3].re-data_core[index1].re-2.0*data_core[index2].re)) + cos(2.0*pi*(3.0*data_core[index3].re-2.0*data_core[index1].re-data_core[index2].re))) + a1*(sin(2.0*pi*(data_core[index1].re-data_core[index2].re)) + sin(2.0*pi*(data_core[index2].re-data_core[index3].re)) + sin(2.0*pi*(data_core[index3].re-data_core[index1].re))) + a3*(sin(4.0*pi*(data_core[index1].re-data_core[index2].re)) + sin(4.0*pi*(data_core[index2].re-data_core[index3].re)) + sin(4.0*pi*(data_core[index3].re-data_core[index1].re))))/(dslip*b);

		MPI_Reduce(&fcore[index], &f_core[index], 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		
		if(rank == 0){
		  E_core += Bn*(f_core[index]/nsize);
		}

		/*partial derivative wrt phase field 1*/

		df1core[index] = ((2.0*pi)/(dslip*b))*(c1*(sin(2.0*pi*(data_core[index2].re-data_core[index1].re)) + sin(2.0*pi*(data_core[index3].re-data_core[index1].re))) + c2*(2.0*sin(2.0*pi*(data_core[index2].re+data_core[index3].re-2.0*data_core[index1].re)) + sin(2.0*pi*(2.0*data_core[index2].re-data_core[index3].re-data_core[index1].re)) + sin(2.0*pi*(2.0*data_core[index3].re-data_core[index1].re-data_core[index2].re))) + (2.0*c3)*(sin(4.0*pi*(data_core[index2].re-data_core[index1].re)) + sin(4.0*pi*(data_core[index3].re-data_core[index1].re))) + c4*(3.0*sin(2.0*pi*(data_core[index2].re+2.0*data_core[index3].re-3.0*data_core[index1].re)) + 3.0*sin(2.0*pi*(2.0*data_core[index2].re+data_core[index3].re-3.0*data_core[index1].re)) + 2.0*sin(2.0*pi*(3.0*data_core[index2].re-data_core[index3].re-2.0*data_core[index1].re)) + sin(2.0*pi*(3.0*data_core[index2].re-2.0*data_core[index3].re-data_core[index1].re)) + sin(2.0*pi*(3.0*data_core[index3].re-data_core[index1].re-2.0*data_core[index2].re)) + 2.0*sin(2.0*pi*(3.0*data_core[index3].re-2.0*data_core[index1].re-data_core[index2].re))) + a1*(cos(2.0*pi*(data_core[index1].re-data_core[index2].re)) - cos(2.0*pi*(data_core[index3].re-data_core[index1].re))) + (2.0*a3)*(cos(4.0*pi*(data_core[index1].re-data_core[index2].re)) - cos(4.0*pi*(data_core[index3].re-data_core[index1].re))));

		/*partial derivative wrt phase field 2*/

		df2core[index] = ((2.0*pi)/(dslip*b))*(c1*(sin(2.0*pi*(data_core[index1].re-data_core[index2].re)) + sin(2.0*pi*(data_core[index3].re-data_core[index2].re))) + c2*(sin(2.0*pi*(2.0*data_core[index1].re-data_core[index2].re-data_core[index3].re)) + 2.0*sin(2.0*pi*(data_core[index3].re+data_core[index1].re-2.0*data_core[index2].re)) + sin(2.0*pi*(2.0*data_core[index3].re-data_core[index1].re-data_core[index2].re))) + (2.0*c3)*(sin(4.0*pi*(data_core[index1].re-data_core[index2].re)) + sin(4.0*pi*(data_core[index3].re-data_core[index2].re))) + c4*(sin(2.0*pi*(3.0*data_core[index1].re-data_core[index2].re-2.0*data_core[index3].re)) + 2.0*sin(2.0*pi*(3.0*data_core[index1].re-2.0*data_core[index2].re-data_core[index3].re)) + 3.0*sin(2.0*pi*(data_core[index3].re+2.0*data_core[index1].re-3.0*data_core[index2].re)) + 3.0*sin(2.0*pi*(2.0*data_core[index3].re+data_core[index1].re-3.0*data_core[index2].re)) + 2.0*sin(2.0*pi*(3.0*data_core[index3].re-data_core[index1].re-2.0*data_core[index2].re)) + sin(2.0*pi*(3.0*data_core[index3].re-2.0*data_core[index1].re-data_core[index2].re))) + a1*(cos(2.0*pi*(data_core[index2].re-data_core[index3].re)) - cos(2.0*pi*(data_core[index1].re-data_core[index2].re))) + (2.0*a3)*(cos(4.0*pi*(data_core[index2].re-data_core[index3].re)) - cos(4.0*pi*(data_core[index1].re-data_core[index2].re))));

		/*partial derivative wrt phase field 3*/

		df3core[index] = ((2.0*pi)/(dslip*b))*(c1*(sin(2.0*pi*(data_core[index2].re-data_core[index3].re)) + sin(2.0*pi*(data_core[index1].re-data_core[index3].re))) + c2*(sin(2.0*pi*(2.0*data_core[index1].re-data_core[index2].re-data_core[index3].re)) + sin(2.0*pi*(2.0*data_core[index2].re-data_core[index3].re-data_core[index1].re)) + 2.0*sin(2.0*pi*(data_core[index1].re+data_core[index2].re-2.0*data_core[index3].re))) + (2.0*c3)*(sin(4.0*pi*(data_core[index2].re-data_core[index3].re)) + sin(4.0*pi*(data_core[index1].re-data_core[index3].re))) + c4*(2.0*sin(2.0*pi*(3.0*data_core[index1].re-data_core[index2].re-2.0*data_core[index3].re)) + sin(2.0*pi*(3.0*data_core[index1].re-2.0*data_core[index2].re-data_core[index3].re)) + sin(2.0*pi*(3.0*data_core[index2].re-data_core[index3].re-2.0*data_core[index1].re)) + 2.0*sin(2.0*pi*(3.0*data_core[index2].re-2.0*data_core[index3].re-data_core[index1].re)) + 3.0*sin(2.0*pi*(data_core[index1].re+2.0*data_core[index2].re-3.0*data_core[index3].re)) + 3.0*sin(2.0*pi*(2.0*data_core[index1].re+data_core[index2].re-3.0*data_core[index3].re))) + a1*(cos(2.0*pi*(data_core[index3].re-data_core[index1].re)) - cos(2.0*pi*(data_core[index2].re-data_core[index3].re))) + (2.0*a3)*(cos(4.0*pi*(data_core[index3].re-data_core[index1].re)) - cos(4.0*pi*(data_core[index2].re-data_core[index3].re))));
	      }/*ijk*/
      }/*plane*/

    //if(rank == 0){
    //printf("fcore %lf df1core %lf df2core %lf df3core %lf index %d\n", fcore[index], df1core[index], df2core[index], df3core[index], index);
    //}
    for(plane=0;plane<NP;plane++)
      {
	for(i=0;i<lN1;i++)
	  for(j=0;j<N2;j++)
	    for(k=0;k<N3;k++)
	      {
		index = i*N2*N3 + j*N3 + k + plane*lN1*N2*N3;
		index1 = i*N2*N3 + j*N3 + k + 0*lN1*N2*N3 + plane*lN1*N2*N3*3;
		index2 = i*N2*N3 + j*N3 + k + 1*lN1*N2*N3 + plane*lN1*N2*N3*3;
		index3 = i*N2*N3 + j*N3 + k + 2*lN1*N2*N3 + plane*lN1*N2*N3*3;
		
		dE_core[index1] = Bn*df1core[index];
		dE_core[index2] = Bn*df2core[index];
		dE_core[index3] = Bn*df3core[index];
	      }/*ijk*/
      }/*plane*/

    if(rank == 0 && fmod((double)(it), 200.0) == 0.0 /*it == NT-1*/){
      printf("Core Energy for this time step %lf\n", E_core);
      //fprintf(of7, "%d %lf\n", it, E_core);
    } 

    MPI_Barrier(MPI_COMM_WORLD);

    if(fmod((double)(it+(itp*NT)), 200.0) == 0.0 || it+(itp*NT) == (NSI*NT)-1){
      for(i=0;i<lN1*N2*N3*NP;i++){
	delta[i] = 0.0;
      }
      for(plane=0;plane<NP;plane++)
	{
	  for(i=0;i<lN1;i++)
	    for(j=0;j<N2;j++)
	      for(k=0;k<N3;k++)
		{
		  index = i*N2*N3 + j*N3 + k + plane*lN1*N2*N3;
		  //indexdx = ((lxs+i)+1)*N2*N3 + j*N3 + k + plane*lN1*N2*N3;
		  index1 = i*N2*N3 + j*N3 + k + 0*lN1*N2*N3 + plane*lN1*N2*N3*3;
		  index2 = i*N2*N3 + j*N3 + k + 1*lN1*N2*N3 + plane*lN1*N2*N3*3;
		  index3 = i*N2*N3 + j*N3 + k + 2*lN1*N2*N3 + plane*lN1*N2*N3*3;

		  delta[index] = /*(1.0/2.0)**/(data_core[index1].re*xb[0][0] + data_core[index2].re*xb[1][0] + data_core[index3].re*xb[2][0])*xb[1][0] + (data_core[index1].re*xb[0][1] + data_core[index2].re*xb[1][1] + data_core[index3].re*xb[2][1])*xb[1][1] + (data_core[index1].re*xb[0][2] + data_core[index2].re*xb[1][2] + data_core[index3].re*xb[2][2])*xb[1][2];

		  if(oflag == 0){ //edge orientation
		    disreg[index] = (data_core[index1].re*xb[0][1] + data_core[index2].re*xb[1][1] + data_core[index3].re*xb[2][1]);
		    eta1 = data_core[index2].re - floor(data_core[index2].re);
		    phi2 = data_core[index1].re*xb[0][1] + data_core[index3].re*xb[2][1];
		    eta2 = 3.0*(phi2/3.0 - floor(phi2/3.0));
		    disregexp[index] = exp(-a*(eta1-0.5)*(eta1-0.5))*exp(-a*(eta2-0.5)*(eta2-0.5))+exp(-a*(eta1-1.0)*(eta1-1.0))*exp(-a*(eta2-2.0)*(eta2-2.0))+exp(-a*(eta1*eta1))*exp(-a*(eta2-2.0)*(eta2-2.0));
		  }
		  else{ //screw orientation
		    disreg[index] = (data_core[index1].re*xb[0][0] + data_core[index2].re*xb[1][0] + data_core[index3].re*xb[2][0])*1.0;
		    eta1 = data_core[index2].re - floor(data_core[index2].re);
		    phi2 = data_core[index1].re*xb[0][0] + data_core[index3].re*xb[2][0];
		    eta2 = 3.0*(phi2/3.0 - floor(phi2/3.0));
		    disregexp[index] = exp(-a*(eta1-0.5)*(eta1-0.5))*exp(-a*(eta2-0.5)*(eta2-0.5))+exp(-a*(eta1-1.0)*(eta1-1.0))*exp(-a*(eta2-2.0)*(eta2-2.0))+exp(-a*(eta1*eta1))*exp(-a*(eta2-2.0)*(eta2-2.0));
		  }
		  eta1 = data_core[index2].re - floor(data_core[index2].re);
		  eta3 = sqrt(3.0)/2.0*(data_core[index1].re + data_core[index3].re)- 3.8*floor(sqrt(3.0)/6.0*(data_core[index1].re + data_core[index3].re));
		  disregexp2[index] = exp(-a*(eta1-0.5)*(eta1-0.5))*exp(-a*(eta2-0.5)*(eta2-0.5))+exp(-a*(eta1-1.0)*(eta1-1.0))*exp(-a*(eta2-2.0)*(eta2-2.0))+exp(-a*(eta1*eta1))*exp(-a*(eta2-2.0)*(eta2-2.0));
		  ///if(k <= N3/2-1){
		  ///delta[index] = /*(1.0/2.0)**/(data_core[index1].re*xb[0][0] + data_core[index2].re*xb[1][0] + data_core[index3].re*xb[2][0])*xb[0][0] + (data_core[index1].re*xb[0][1] + data_core[index2].re*xb[1][1] + data_core[index3].re*xb[2][1])*xb[0][1] + (data_core[index1].re*xb[0][2] + data_core[index2].re*xb[1][2] + data_core[index3].re*xb[2][2])*xb[0][2];
		  ///}
		  //else{

		  ///delta[index] = /*(1.0/2.0)**/(data_core[index1].re*xb[0][0] + data_core[index2].re*xb[1][0] + data_core[index3].re*xb[2][0])*xb[2][0] + (data_core[index1].re*xb[0][1] + data_core[index2].re*xb[1][1] + data_core[index3].re*xb[2][1])*xb[2][1] + (data_core[index1].re*xb[0][2] + data_core[index2].re*xb[1][2] + data_core[index3].re*xb[2][2])*xb[2][2];
		  //}

		  //if(lxs+i == N1/2 && k == N3/2){
		  if(j == N2/2 && k == N3/2){
		    fprintf(of4, "%d %lf %lf %lf %lf %lf\n", lxs+i, (dx)*(lxs+i), delta[index], disreg[index], disregexp[index],disregexp2[index]);
		    //for Ensight
		    //fprintf(of4, "%lf %lf\n", (dx)*(lxs+i), delta[index]);
		  }
		  if((fmod((double)(it+(itp*NT)), 200.0) == 0.0) || it+(itp*NT) == (NSI*NT)-1){
		    if(k == N3/2){
		      fprintf(of9, "%d %d %lf\n", lxs+i, j, delta[index]);
		      counter = counter+1;
                      if(delta[index]>=0.1 && delta[index]<1){
			countSF = countSF+1;
		      } 
		    }
		    if(k == (N3/2)-1){
		      fprintf(of10, "%d %d %lf\n", lxs+i, j, delta[index]);
		    }
		    if(j == (N2/2)){
		      fprintf(of11, "%d %d %lf\n", lxs+i, k, delta[index]);
		    }
		  } 
		}/*ijk*/
	}/*plane*/

      for(plane=0;plane<NP;plane++)
	{
	  for(i=0;i<lN1;i++)
	    //for(j=0;j<N2;j++)
	    //for(k=0;k<N3;k++)
	    {
	      //if(j == N2/2 && k == N3/2){
	      j = N2/2;
	      k = N3/2;
	      index = i*N2*N3 + j*N3 + k + plane*lN1*N2*N3;
	      //indexdx = i*N2*N3 + (j+1)*N3 + k + plane*lN1*N2*N3;
	      indexdx = (i+1)*N2*N3 + j*N3 + k + plane*lN1*N2*N3;
	      if((i+1) == lN1){
		indexmin = 0*N2*N3 + j*N3 + k + plane*lN1*N2*N3;
		if(rank != 0){
		  MPI_Send(&delta[indexmin], 1, MPI_DOUBLE, rank-1, tag, MPI_COMM_WORLD);
		}
		if((lxs+(i+1)) == N1){
		  mpidel = delta[index];
		  ddelta[i] = (mpidel - delta[index])/dx;
		  goto skip_point1;
		}
		MPI_Recv(&mpidel, 1, MPI_DOUBLE, rank+1, tag, MPI_COMM_WORLD, &status);
		ddelta[i] = (mpidel - delta[index])/dx;
	      }
	      else{
		ddelta[i] = (delta[indexdx] - delta[index])/dx;
	      }

	      //ddelta[index] = (delta[indexdx] - delta[index])/dx;
	      //if(lxs+i == N1/2 && k == N3/2){
	      //fprintf(of5, "%d %lf %lf %d %d %lf %lf %lf\n", j, (dslip/N1)*j, ddelta[index], index, indexdx, delta[index], delta[indexdx], dx);

	    skip_point1:
	      fprintf(of5, "%d %lf %lf\n", lxs+i, (dx)*(lxs+i), ddelta[i]);
	      //for Ensight
	      //fprintf(of5, "%lf %lf\n", (dx)*(lxs+i), ddelta[index]);
	    }/*ijk*/
	}/*plane*/
    }/*it*/

    MPI_Reduce(&counter, &count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&countSF, &cSF, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if(rank==0){
      if((fmod((double)(it+(itp*NT)),200.0) == 0.0) && it+(itp*NT) != 0 || it+(itp*NT) == (NSI*NT)-1){
	totAR = N1*N2*b*b*1E9*1E9;
	sfAR = (cSF/count)*totAR;
	printf("Total count %d, SF count %d, Total Area %lf, SF Area %lf, Time Step %d \n", count, cSF, totAR,sfAR, it+(itp*NT));
      }
    }    
  }/*cflag == 2*/

  else if(cflag == 3){ /*1D Sine Approx for Partial Dislocations*/

    for(isa=0;isa<NS;isa++)
      {
	for(i=0;i<lN1;i++)
	  for(j=0;j<N2;j++)
	    for(k=0;k<N3;k++)
	      {
		index = i*N2*N3 + j*N3 + k + isa*lN1*N2*N3;
		E_core += (isf*(sin(pi*data_fftw[index].re)*sin(pi*data_fftw[index].re)) + Cn*(sin(2*pi*data_fftw[index].re)*sin(2*pi*data_fftw[index].re)))/N1;
		dE_core[index] = Bn*(isf*pi*sin(2*pi*data_fftw[index].re) + Cn*2*pi*sin(4*pi*data_fftw[index].re));
	      }/*ijk*/
      }/*isa*/

    if(rank == 0 && /*fmod((double)(it),1000.0) == 0.0*/ it == NT-1){
      printf("Core Energy for this time step %lf\n", E_core);
    }

    if(fmod((double)(it+(itp*NT)), 200.0) == 0.0 || it+(itp*NT) == (NSI*NT)-1){
      for(i=0;i<lN1*N2*N3*NP;i++){
	delta[i] = 0.0;
      }
      for(plane=0;plane<NP;plane++)
	{
	  for(i=0;i<lN1;i++)
	    for(j=0;j<N2;j++)
	      for(k=0;k<N3;k++)
		{
		  index = i*N2*N3 + j*N3 + k + plane*lN1*N2*N3;
		  //indexdx = ((lxs+i)+1)*N2*N3 + j*N3 + k + plane*lN1*N2*N3;
		  index1 = i*N2*N3 + j*N3 + k + 0*lN1*N2*N3 + plane*lN1*N2*N3*3;
		  index2 = i*N2*N3 + j*N3 + k + 1*lN1*N2*N3 + plane*lN1*N2*N3*3;
		  index3 = i*N2*N3 + j*N3 + k + 2*lN1*N2*N3 + plane*lN1*N2*N3*3;
		  
		  delta[index] = /*(1.0/2.0)**/(data_fftw[index1].re*xb[0][0] + data_fftw[index2].re*xb[1][0] + data_fftw[index3].re*xb[2][0])*xb[1][0] + (data_fftw[index1].re*xb[0][1] + data_fftw[index2].re*xb[1][1] + data_fftw[index3].re*xb[2][1])*xb[1][1] + (data_fftw[index1].re*xb[0][2] + data_fftw[index2].re*xb[1][2] + data_fftw[index3].re*xb[2][2])*xb[1][2];

		  //if(lxs+i == N1/2 && k == N3/2){
		  if(j == N2/2 && k == N3/2){
		    fprintf(of4, "%d %lf %lf\n", lxs+i, (dx)*(lxs+i), delta[index]);
		    //for Ensight
		    //fprintf(of4, "%lf %lf\n", (dx)*(lxs+i), delta[index]);
		  }
		}/*ijk*/
	}/*plane*/

      for(plane=0;plane<NP;plane++)
	{
	  for(i=0;i<lN1;i++)
	    //for(j=0;j<N2;j++)
	    //for(k=0;k<N3;k++)
	    {
	      //if(j == N2/2 && k == N3/2){
	      j = N2/2;
	      k = N3/2;
	      index = i*N2*N3 + j*N3 + k + plane*lN1*N2*N3;
	      //indexdx = i*N2*N3 + (j+1)*N3 + k + plane*lN1*N2*N3;
	      indexdx = (i+1)*N2*N3 + j*N3 + k + plane*lN1*N2*N3;
	      if((i+1) == lN1){
		indexmin = 0*N2*N3 + j*N3 + k + plane*lN1*N2*N3;
		if(rank != 0){
		  MPI_Send(&delta[indexmin], 1, MPI_DOUBLE, rank-1, tag, MPI_COMM_WORLD);
		}
		if((lxs+(i+1)) == N1){
		  mpidel = delta[index];
		  ddelta[i] = (mpidel - delta[index])/dx;
		  goto skip_point2;
		}
		MPI_Recv(&mpidel, 1, MPI_DOUBLE, rank+1, tag, MPI_COMM_WORLD, &status);
		ddelta[i] = (mpidel - delta[index])/dx;
	      }
	      else{
		ddelta[i] = (delta[indexdx] - delta[index])/dx;
	      }

	      //ddelta[index] = (delta[indexdx] - delta[index])/dx;
	      //if(lxs+i == N1/2 && k == N3/2){
	      //fprintf(of5, "%d %lf %lf %d %d %lf %lf %lf\n", j, (dslip/N1)*j, ddelta[index], index, indexdx, delta[index], delta[indexdx], dx);

	    skip_point2:
	      fprintf(of5, "%d %lf %lf\n", lxs+i, (dx)*(lxs+i), ddelta[i]);
	      //for Ensight
	      //fprintf(of5, "%lf %lf\n", (dx)*(lxs+i), ddelta[index]);
	    }
	}
    }/*it*/ 
  }/*cflag == 3*/

  else if(cflag == 4){ //not working 11/08/10 3D Sine Approx for Partials

    for(isa=0;isa<NS;isa++)
      {
	for(i=0;i<lN1;i++)
	  for(j=0;j<N2;j++)
	    for(k=0;k<N3;k++)
	      {
		index = i*N2*N3 + j*N3 + k + isa*lN1*N2*N3;
		data_core[index].re = data_fftw[index].re;
		data_core[index].im = data_fftw[index].im;
	      }/*ijk*/
      }/*isa*/
    
    for(plane=0;plane<NP;plane++)
      {
	for(i=0;i<lN1;i++)
	  for(j=0;j<N2;j++)
	    for(k=0;k<N3;k++)
	      {
		index = i*N2*N3 + j*N3 + k + plane*lN1*N2*N3;
		index1 = i*N2*N3 + j*N3 + k + 0*lN1*N2*N3 + plane*lN1*N2*N3*3;
		index2 = i*N2*N3 + j*N3 + k + 1*lN1*N2*N3 + plane*lN1*N2*N3*3;
		index3 = i*N2*N3 + j*N3 + k + 2*lN1*N2*N3 + plane*lN1*N2*N3*3;

		/*no derivatives taken use to calculate E_core*/ 
    
		fcore[index] += (isf/2.0)*(sin(pi*(data_core[index3].re - data_core[index1].re))*sin(pi*(data_core[index3].re - data_core[index1].re)) + sin(pi*(data_core[index1].re - data_core[index2].re))*sin(pi*(data_core[index1].re - data_core[index2].re)) + sin(pi*(data_core[index2].re - data_core[index3].re))*sin(pi*(data_core[index2].re - data_core[index3].re))) + An*(sin(2*pi*(data_core[index3].re - data_core[index1].re)+(pi*p))*sin(2*pi*(data_core[index3].re - data_core[index1].re)+(pi*p)) + sin(2*pi*(data_core[index1].re - data_core[index2].re)+(2.0*pi*p))*sin(2*pi*(data_core[index1].re - data_core[index2].re)+(2.0*pi*p)) + sin(2*pi*(data_core[index2].re - data_core[index3].re)+(pi*p))*sin(2*pi*(data_core[index2].re - data_core[index3].re)+(pi*p)))/N1;

		MPI_Reduce(&fcore[index], &f_core[index], 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		
		if(rank == 0){
		  E_core += Bn*(f_core[index]/nsize);
		}
		df1core[index] = ((isf/2.0)*pi*(sin(2.0*pi*(data_core[index1].re-data_core[index2].re)) - sin(2.0*pi*(data_core[index3].re-data_core[index1].re)))) + An*2.0*pi*(sin(2.0*pi*(2.0*(data_core[index1].re-data_core[index2].re))+(2.0*p)) - sin(2.0*pi*(2.0*(data_core[index3].re-data_core[index1].re))+p));

		df2core[index] = ((isf/2.0)*pi*(sin(2.0*pi*(data_core[index2].re-data_core[index3].re)) - sin(2.0*pi*(data_core[index1].re-data_core[index2].re)))) + An*2.0*pi*(sin(2.0*pi*(2.0*(data_core[index2].re-data_core[index3].re)+p)) - sin(2.0*pi*(2.0*(data_core[index1].re-data_core[index2].re)+(2.0*p))));

		df3core[index] = ((isf/2.0)*pi*(sin(2.0*pi*(data_core[index3].re-data_core[index1].re)) - sin(2.0*pi*(data_core[index2].re-data_core[index3].re)))) + An*2.0*pi*(sin(2.0*pi*(2.0*(data_core[index3].re-data_core[index1].re)+p)) - sin(2.0*pi*(2.0*(data_core[index2].re-data_core[index3].re)+p)));
	      }/*ijk*/
      }/*plane*/
    
    for(plane=0;plane<NP;plane++)
      {
	for(i=0;i<lN1;i++)
	  for(j=0;j<N2;j++)
	    for(k=0;k<N3;k++)
	      {
		index = i*N2*N3 + j*N3 + k + plane*lN1*N2*N3;
		index1 = i*N2*N3 + j*N3 + k + 0*lN1*N2*N3 + plane*lN1*N2*N3*3;
		index2 = i*N2*N3 + j*N3 + k + 1*lN1*N2*N3 + plane*lN1*N2*N3*3;
		index3 = i*N2*N3 + j*N3 + k + 2*lN1*N2*N3 + plane*lN1*N2*N3*3;
		
		dE_core[index1] = Bn*df1core[index];
		dE_core[index2] = Bn*df2core[index];
		dE_core[index3] = Bn*df3core[index];
	      }
      }

    if(rank == 0 && /*fmod((double)(it),1000.0) == 0.0*/ it == NT-1){
      printf("Core Energy for this time step %lf\n", E_core);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    //fix right now only prints first point on each processor.?????
    //if(it == NT-1){
    //if(fmod((double)(it),1000.0) == 0.0 || it == NT-1){
    if(/*fmod((double)(itp),5.0) == 0.0 ||*/ it+(itp*NT) == (NSI*NT)-1){
      for(i=0;i<lN1*N2*N3*NP;i++){
	delta[i] = 0.0;
      }
      for(plane=0;plane<NP;plane++)
	{
	  for(i=0;i<lN1;i++)
	    for(j=0;j<N2;j++)
	      for(k=0;k<N3;k++)
		{
		  index = i*N2*N3 + j*N3 + k + plane*lN1*N2*N3;
		  //indexdx = ((lxs+i)+1)*N2*N3 + j*N3 + k + plane*lN1*N2*N3;
		  index1 = i*N2*N3 + j*N3 + k + 0*lN1*N2*N3 + plane*lN1*N2*N3*3;
		  index2 = i*N2*N3 + j*N3 + k + 1*lN1*N2*N3 + plane*lN1*N2*N3*3;
		  index3 = i*N2*N3 + j*N3 + k + 2*lN1*N2*N3 + plane*lN1*N2*N3*3;

		  delta[index] = /*(1.0/2.0)**/(data_core[index1].re*xb[0][0] + data_core[index2].re*xb[1][0] + data_core[index3].re*xb[2][0])*xb[1][0] + (data_core[index1].re*xb[0][1] + data_core[index2].re*xb[1][1] + data_core[index3].re*xb[2][1])*xb[1][1] + (data_core[index1].re*xb[0][2] + data_core[index2].re*xb[1][2] + data_core[index3].re*xb[2][2])*xb[1][2];

		  //if(lxs+i == N1/2 && k == N3/2){
		  if(j == N2/2 && k == N3/2){
		    fprintf(of4, "%d %lf %lf\n", lxs+i, (dx)*(lxs+i), delta[index]);
		    //for Ensight
		    //fprintf(of4, "%lf %lf\n", (dx)*(lxs+i), delta[index]);
		  }
		}/*ijk*/
	}/*plane*/

      for(plane=0;plane<NP;plane++)
	{
	  for(i=0;i<lN1;i++)
	    //for(j=0;j<N2;j++)
	    //for(k=0;k<N3;k++)
	    {
	      //if(j == N2/2 && k == N3/2){
	      j = N2/2;
	      k = N3/2;
	      index = i*N2*N3 + j*N3 + k + plane*lN1*N2*N3;
	      //indexdx = i*N2*N3 + (j+1)*N3 + k + plane*lN1*N2*N3;
	      indexdx = (i+1)*N2*N3 + j*N3 + k + plane*lN1*N2*N3;
	      if((i+1) == lN1){
		indexmin = 0*N2*N3 + j*N3 + k + plane*lN1*N2*N3;
		if(rank != 0){
		  MPI_Send(&delta[indexmin], 1, MPI_DOUBLE, rank-1, tag, MPI_COMM_WORLD);
		}
		if((lxs+(i+1)) == N1){
		  mpidel = delta[index];
		  ddelta[i] = (mpidel - delta[index])/dx;
		  goto skip_point3;
		}
		MPI_Recv(&mpidel, 1, MPI_DOUBLE, rank+1, tag, MPI_COMM_WORLD, &status);
		ddelta[i] = (mpidel - delta[index])/dx;
	      }
	      else{
		ddelta[i] = (delta[indexdx] - delta[index])/dx;
	      }

	      //ddelta[index] = (delta[indexdx] - delta[index])/dx;
	      //if(lxs+i == N1/2 && k == N3/2){
	      //fprintf(of5, "%d %lf %lf %d %d %lf %lf %lf\n", j, (dslip/N1)*j, ddelta[index], index, indexdx, delta[index], delta[indexdx], dx);

	    skip_point3:
	      fprintf(of5, "%d %lf %lf\n", lxs+i, (dx)*(lxs+i), ddelta[i]);
	      //for Ensight
	      //fprintf(of5, "%lf %lf\n", (dx)*(lxs+i), ddelta[index]);
	    }/*ijk*/
	}/*plane*/
    }/*it*/    
  }/*cflag == 4*/

  if(fmod((double)(it+(itp*NT)), 200.0)== 0.0 || it+(itp*NT) == (NSI*NT)-1){
    fclose(of4);
    fclose(of5);
    //fclose(of6); //--cflag == 2 only
  }
  /*if(rank == 0){
    fclose(of7);
    }*/
  //if(cflag == 2 &&  it+(itp*NT) == (NSI*NT)-1){
  if((fmod((double)(it+(itp*NT)), 200.0) == 0.0 && it+(itp*NT) != 0 )|| it+(itp*NT) == (NSI*NT)-1){
    fclose(of9);
    fclose(of10);
    fclose(of11);
  }

  free(delta);
  free(ddelta);
  free(disreg);
  free(disregexp);
  free(f_core);

  MPI_Barrier(MPI_COMM_WORLD);

  return;
}

/******************************************************************/

void Bmatrix (double *BB, double *fx, double *fy, double *fz, double **xn, double **xb, double ***eps, double d1, double d2,double d3, double C11, double C12, double C44, double b, double dslip, int lN1, double nu, int rank, int N1, int N2, int N3, int ND, int NS)
{
  if(rank == 0){
    printf("Set B Matrix \n");
  }

#define 	DELTA(i, j)   ((i==j)?1:0)
	
  int i, j, k, l, m, n, u, v, k1, k2, k3, ka, kb, nv, nb, nfreq;
  int is, js, ks;
  double fkr;
  double C[ND][ND][ND][ND];
  double A[ND][ND][ND][ND];
  double B[NS][NS][lN1][N2][N3];
  double G[ND][ND];
  double fk[ND];
  double xn_temp[NS][ND], xb_temp[NS][ND], eps_temp[NS][ND][ND];
  double xnu, fk2, fk4, fka, fkb, ll, mu, young;
  
  mu = C44-(2.0*C44+C12-C11)/5.0;
  ll = C12-(2.0*C44+C12-C11)/5.0;
  young = mu*(3*ll+2*mu)/(ll+mu);
  xnu = young/2.0/mu-1.0;
  //xnu = nu; //C12/(2.0*(C44+C12));

if(rank == 0){
    printf("Bmatrix mu %lf, ll %lf, young %lf, nu %lf\n", mu, ll, young, xnu);
  }
  
  /* set Cijkl*/
  
  for (i=0; i<ND; i++) {
    for (j=0; j<ND; j++) {
      for (k=0; k<ND; k++) {
	for (m=0; m<ND; m++) {
	  C[i][j][k][m] = C44 * (DELTA(i,k)*DELTA(j,m)+DELTA(i,m)*DELTA(j,k))+C12*DELTA(i,j)*DELTA(k,m);
	  A[i][j][k][m] = 0.0;
	  /*printf("in BB %d %d %d %d %d\n",ND, i,j,k,m);*/
	}
      }
    }
  }

  for (i=0; i<NS; i++) {
    for (j=0; j<ND; j++) {
      xn_temp[i][j]=xn[i][j];
      xb_temp[i][j]=xb[i][j];
    }
  }

  /*set eps */
  
  for (i=0; i<ND; i++) {
    for (j=0; j<ND; j++) {
      for (k=0; k<NS;k++){
	eps_temp[k][i][j] = xb_temp[k][i]*xn_temp[k][j]/dslip;
      }
    }
  }

  for (i=0; i<ND; i++) {
    for (j=0; j<ND; j++) {
      for (k=0; k<NS;k++){
	eps[k][i][j] = eps_temp[k][i][j];
      }
    }
  }

  /* set A, Green function and B matrix*/	
  for(k1=0;k1<lN1;k1++)
    {
      for(k2=0;k2<N2;k2++)
	{
	  for(k3=0;k3<N3;k3++)
	    {
	      nfreq = k3+(k2)*N3+(k1)*N3*N2;
	      fk[0] = fx[nfreq];
	      fk[1] = fy[nfreq];
	      fk[2] = fz[nfreq];
	      fk2 = fk[0]*fk[0]+fk[1]*fk[1]+fk[2]*fk[2];
	      fk4 = fk2*fk2;
	      if(fk2>0)
		{
		  for (m=0; m<ND; m++) {
		    for (n=0; n<ND; n++) {
		      for (u=0; u<ND; u++) {
			for (v=0; v<ND; v++) {
			  A[m][n][u][v] = 0.0;
			  
			  for	(i=0; i<ND; i++) {
			    for (j=0; j<ND; j++) {
			      for (k=0; k<ND; k++) {
				G[k][i] = (2.0 * DELTA(i,k)/fk2-1.0/(1.0-xnu)*fk[i]*fk[k]/fk4)/(2.0*C44);
				for	(l=0; l<ND; l++) {
				  A[m][n][u][v] = A[m][n][u][v] - C[k][l][u][v]*C[i][j][m][n]*G[k][i]*fk[j]*fk[l] ;	
				}	
			      }
			    }
			  }
			  A[m][n][u][v] = A[m][n][u][v]+C[m][n][u][v];
			  /*printf("A %d %d %d %d %f %f %f %f\n",m,n,u,v,fk[0],fk[1],fk[2],A[m][n][u][v]);*/
			}
		      }
		    }
		  }
		  
		} /*if fk2 */										
	      for(ka=0;ka<NS;ka++)
		{
		  for(kb=0;kb<NS;kb++)  
		    {
		      B[ka][kb][k1][k2][k3] = 0.0;
		      for (m=0; m<ND; m++) {
			for (n=0; n<ND; n++) {
			  for (u=0; u<ND; u++) {
			    for (v=0; v<ND; v++) {
			      B[ka][kb][k1][k2][k3]= B[ka][kb][k1][k2][k3] + A[m][n][u][v]*eps_temp[ka][m][n]*eps_temp[kb][u][v];
			    }	
			  }
			}
		      }	
		      
		      nb = nfreq +(ka)*lN1*N2*N3+(kb)*lN1*N2*N3*NS;
		      BB[nb] = B[ka][kb][k1][k2][k3]/mu;
		      /*printf("%lf %lf %lf %lf \n", fx[nfreq], fy[nfreq], fz[nfreq], BB[nb]);*/
		    } /*ka*/
		}/* kb*/
	      
	      
	    }	/*k1*/
	}	/*k2*/	
    }	/*k3*/
  
  
  return;
}

/********************************************************************/

void Fmatrix (double *FF, double *DD, double *fx, double *fy, double *fz, double ***eps, double d1, double d2,double d3, double C11, double C12, double C44, int lN1, int rank, int N1, int N2, int N3, int ND, int NS)
{
  if(rank == 0){
    printf("set F matrix \n");
  }

#define 	DELTA(i, j)   ((i==j)?1:0)
	
  int i, j, k, l, m, n, u, v, k1, k2, k3, ka, nv, nb, nfreq;
  int is, js, ks;
  double fkr;
  double C[ND][ND][ND][ND];
  double F[NS][ND][ND];
  double D[NS][ND][ND];
  double G[ND][ND];
  double fk[ND];
  double xnu, mu, ll, young, fk2, fk4, fka,fkb;
  double A[ND][ND][ND][ND];
  
  mu = C44-(2.0*C44+C12-C11)/5.0;
  ll = C12-(2.0*C44+C12-C11)/5.0;
  young = mu*(3*ll+2*mu)/(ll+mu);
  xnu = young/2.0/mu-1.0;

  if(rank == 0){
    printf("Fmatrix mu %lf, ll %lf, young %lf, nu %lf\n", mu, ll, young, xnu);
  }

  /* set Cijkl*/
  
  for (i=0; i<ND; i++) 
    for (j=0; j<ND; j++) 
      for (k=0; k<ND; k++)
	for (m=0; m<ND; m++) 
	  {
	    C[i][j][k][m] = mu * (DELTA(i,k)*DELTA(j,m)+DELTA(i,m)*DELTA(j,k))+ll*DELTA(i,j)*DELTA(k,m);
	  }  

  /* set Green function and F matrix*/
	
  for(k1=0;k1<lN1;k1++)
      for(k2=0;k2<N2;k2++)
	  for(k3=0;k3<N3;k3++)
	    {
	      nfreq = k3+(k2)*N3+(k1)*N3*N2;
	      fk[0] = fx[nfreq];
	      fk[1] = fy[nfreq];
	      fk[2] = fz[nfreq];
	      fk2 = fk[0]*fk[0]+fk[1]*fk[1]+fk[2]*fk[2];
	      fk4 = fk2*fk2;
	      
	      if(fk2>0)
		{
		  for(m=0; m<ND; m++)
		    for(n=0; n<ND; n++)
		      for(u=0; u<ND; u++)
			for(v=0; v<ND; v++) 
			  {
			    A[m][n][u][v] = 0.0;
			  
			    for(i=0; i<ND; i++)
			      for(j=0; j<ND; j++)
				for (k=0; k<ND; k++) 
				  {
				    G[k][i] = (2.0 * DELTA(i,k)/fk2-1.0/(1.0-xnu)*fk[i]*fk[k]/fk4)/(2.0*mu);
				    for(l=0; l<ND; l++) 
				      {
					A[m][n][u][v] = A[m][n][u][v] - C[k][l][u][v]*C[i][j][m][n]*G[k][i]*fk[j]*fk[l] ;	
				      }	
				  }
			    A[m][n][u][v] = A[m][n][u][v]+C[m][n][u][v];
			  }		  
		} /*if fk2 */		
		
	      for (ka=0; ka<NS; ka++)
		for (i=0; i<ND; i++)
		  for (j=0; j<ND; j++) 
		    {
		      //if (ka<NS)	{F[ka][i][j]= 0.0;}
		      F[ka][i][j]= 0.0;
		      D[ka][i][j]=0.0;
		      //if(ka<NV) {Fv[ka][i][j]=0.0;}			  
		      for(k=0; k<ND; k++)
			for (l=0; l<ND; l++)
			  {
			    for (m=0; m<ND; m++)
			      for (n=0; n<ND; n++) 
				{
				  //if(ka<NV) {
				    //if (ka<NS) {
				  F[ka][i][j] = F[ka][i][j] + C[k][l][m][n]*G[j][k]*eps[ka][m][n]*fk[i]*fk[l] ;		      
				    //}
				    //Fv[ka][i][j] = Fv[ka][i][j] + C[k][l][m][n]*G[j][k]*epsv[ka][m][n]*fk[i]*fk[l] ;
				  //if(rank == 0){
				  //printf(" in F %d  %d %d  %d %d %d %d %lf %lf %lf %lf %lf %lf\n",ka, i,j,k,l,m,n, G[j][k], F[ka][i][j], eps[ka][m][n], C[k][l][m][n], fk[i], fk[l] );
				  //}
				    //}
				}
			    //if(ka<NS)
			    //{
			    D[ka][i][j]=D[ka][i][j]+A[i][j][k][l]*eps[ka][k][l];
			    //}
			    //if(ka>=NS)
			    //{
			    //D[ka][i][j]=D[ka][i][j]+A[i][j][k][l]*epsv[ka-NS][k][l];
			    //}
			  }
		      //if (ka<NS) {
		      nb = nfreq + (ka)*lN1*N2*N3 + i*lN1*N2*N3*NS + j*lN1*N2*N3*NS*ND;
		      FF[nb] = F[ka][i][j];
		      //if(rank == 0){
		      //printf(" in FF %d  %d %d %d %d %d %d %lf \n",ka, nb,k1, k2, k3, i,j,FF[nb]);
		      //}
		      //}
		      //if (ka<NV) {
		      //nb = nfreq +(ka)*N1*N2*N3+i*N1*N2*N3*NV+j*N1*N2*N3*NV*ND;
		      //FFv[nb] = Fv[ka][i][j];
		      //}
		      nb = nfreq + (ka)*lN1*N2*N3 + i*lN1*N2*N3*NS + j*lN1*N2*N3*NS*ND;
		      DD[nb] = D[ka][i][j];
		    }  									 
	    }/*k1,k2,k3*/
  return;
}

/********************************************************************/

void resolSS(double **sigma, double *tau, double **xn, double **xb, double dslip, int rank, int ND, int NS)
{
  int is, i, j, k;
  for (is=0;is<NS;is++){
    tau[is] = 0.0;
    for(i=0;i<ND;i++){
      for(j=0;j<ND;j++){
	tau[is] = tau[is]+(sigma[i][j]*xn[is][j]*xb[is][i])/dslip;	
	/*printf("%d %d %lf %lf %lf\n", i,j,sigma[i][j],xn[is][j], xb[is][i]);*/
      }	
    }
    if(rank == 0){
      printf("Resolved shear stresses\n");
      printf("tau[%d] = %lf\n",is, tau[is]);
    }
  }

  return;
}

/********************************************************************/

void material(int mflag, double *a, double *mu, double *young, double *c0, double *c1, double *c2, double *c3, double *c4, double *a1, double *a3, double *isf, double *usf, double *nu, double *ll, double *C44, double *C12, double *C11, double *S44, double *S12, double *S11)
{
  /*
    -Materials-
    mflag == 1 -> Nickel (PRISM device)
    mflag == 2 -> Aluminum
    mflag == 3 -> Gold
    mflag == 4 -> Copper
    mflag == 5 -> Acetaminophen
    mflag == 6 -> Palladium
    mflag == 7 -> Silver
    mflag == 8 -> Platinum
    mflag == 9 -> Rhodium
    mflag == 10 -> Iridium
    mflag == 11 -> Pd-10%Au

    -Parameters-
    a-lattice parameter in meters
    mu, youngs - shear and Young's moduli in Pascal = N/m^2 = kg/(ms^2)
    c0-c4, a1,a3 - gamma-surface coefficients in J/m^2
    isf, usf - intrinsic and unstable stacking fault energies in J/m^2
  */

  if(mflag == 1){ /*Nickel*/
    //*a = 3.52E-10; 
    //*mu = 75.0E9;
    //*young = 200.0E9;

    //*c0 = 410.024E-3;
    //*c1 = -51.99716E-3;
    //*c2 = -120.555E-3;
    //*c3 = 35.21191E-3;
    //*c4 = 0.594646E-3;
    //*a1 = -66.1927E-3;
    //*a3 = -75.3124E-3;
    //coefficients from Strachan etal MD simulations

    //*isf = 84.718E-3;  //25.0E-3;
    //*usf = 211.688E-3; //270.0E-3;

    //coefficients from DFT(PBE Potential)--Ruifeng Zhang
    *a = 3.52E-10; 
    //*mu = 99.7E9; //Room temp
    //*young = 255.7E9;

    //coefficients from Simmons and Wang @ 80K
    *mu = 130.9E9;
    *young = 331.9E9;

    *c0 = 503.144E-3;
    *c1 = -106.533E-3;
    *c2 = -83.741E-3;
    *c3 = 28.777E-3;
    *c4 = -2.916E-3;
    *a1 = -123.917E-3;
    *a3 = -38.533E-3;

    *isf = 144.5E-3;  //25.0E-3;
    *usf = 289.0E-3; //270.0E-3;
  }
  if(mflag == 2){ /*Aluminum*/

    //Parameters from Shen and Wang, 2004
    //*a = 4.05E-10; //meters
    //*mu = 26.0E9; //Pascal
    //*young = 70.0E9; //Pascal

    //*c0 = 242.5E-3; //J/m^2 
    //*c1 = -51.65E-3; //gamma surface coefficients
    //*c2 = -39.71E-3;
    //*c3 = 13.68E-3;
    //*c4 = -1.572E-3;
    //*a1 = -30.40E-3;
    //*a3 = -13.72E-3;

    //*isf = 141.786E-3;//82.0E-3; //intrinsic stacking fault energy J/m^2
    //*usf = 172.3E-3; //224.03E-3; //385.3457E-3; //unstable stacking fault energy 

    //Parameters from DFT(PBE Potential)--Ruifeng Zhang
    *a = 4.04E-10;
    *mu = 28.4E9;
    *young = 76.0E9;

    *c0 = 149.56346E-3; 
    *c1 = -4.23464E-3;
    *c2 = -74.39331E-3;
    *c3 = 33.84682E-3;
    *c4 = -2.39438E-3;
    *a1 = 67.75268E-3;
    *a3 = -30.88995E-3;    

    *isf = 140.2E-3;
    *usf = 177.0E-3;

    //from EAM potential-comparison with Svendsen
    //*a = 4.05E-10; 
    //*mu = 29.41E9; //26.125E9; 
    //*young = 78.49E9; //70.590E9; 

    //*c0 = 253.87E-3;
    //*c1 = -62.75E-3; 
    //*c2 = -33.91E-3;
    //*c3 = 14.18E-3;
    //*c4 = -0.98E-3;
    //*a1 = -49.54E-3;
    //*a3 = -19.82E-3;
    
    //*isf = 145.4E-3; //37.0E-3;
    //*usf = 144.2E-3; //110.0E-3;
  }
  if(mflag == 3){ /*Gold*/

    //Literature Values ?
    //*a = 4.08E-10;
    //*mu = 27.0E9;
    //*young = 78.0E9;

    //Parameters from DFT(PBE Potential)--Ruifeng Zhang 
    *a = 4.17E-10; 
    *mu = 27.0E9; //18.8E9;
    *young = 78.0E9; //54.0E9;

    *c0 = 161.832E-3;
    *c1 = -40.659E-3; 
    *c2 = -17.320E-3;
    *c3 = 3.197E-3;
    *c4 = 0.472E-3;
    *a1 = -64.718E-3;
    *a3 = -13.282E-3;
    
    *isf = 27.9E-3; //37.0E-3;
    *usf = 66.5E-3; //110.0E-3;
    //from Tadmor and Bernstein 2004

    //from EAM potential-comparison with Svendsen
    //*a = 4.08E-10; 
    //*mu = 34.01E9; //16.06E9; 
    //*young = 95.99E9; //46.79E9; 

    //*c0 = 53.48E-3;
    //*c1 = 56.42E-3; 
    //*c2 = -128.74E-3;
    //*c3 = 62.95E-3;
    //*c4 = -4.03E-3;
    //*a1 = 146.29E-3;
    //*a3 = -64.11E-3;
    
    //*isf = 40.6E-3; //37.0E-3;
    //*usf = 77.2E-3; //110.0E-3;
  }
  if(mflag == 4){ /*Copper*/
    //Literature Values ?
    //*a = 3.61E-10;
    //*mu = 48.0E9;
    //*young = 110.0E9;

    //Parameters from DFT(PBE Potential)--Ruifeng Zhang
    *a = 3.64E-10; //meters
    *mu = 54.5E9; //Pascal 
    *young = 144.4E9; //Pascal

    *c0 = 344.948E-3;
    *c1 = -85.399E-3;
    *c2 = -40.421E-3;
    *c3 = 12.396E-3;
    *c4 = -0.674E-3;
    *a1 = -131.393E-3;
    *a3 = -19.816E-3;
    
    *isf = 38.5E-3;
    *usf = 163.7E-3;
  }
  if(mflag == 5){ /*Acetaminophen*/
    *a = 1.84E-9;
    *mu = 3.2E9;
    *young = 8.0E9;
    
    *c0 = 0.0;
    *c1 = 0.0;
    *c2 = 0.0;
    *c3 = 0.0;
    *c4 = 0.0;
    *a1 = 0.0;
    *a3 = 0.0;
    
    *isf = 0.0;
    *usf = 0.0;
  }
  if(mflag == 6){ /*Palladium*/
    //Parameters from Shen and Wang, 2004
    //*a = 3.890E-10; 
    //*mu = 44.0E9; 
    //*young = 121.0E9;

    //*c0 = 374.1E-3; //J/m^2
    //*c1 = -71.01E-3; //gamma surface coefficients
    //*c2 = -69.62E-3;
    //*c3 = 21.22E-3;
    //*c4 = -2.644E-3;
    //*a1 = -54.80E-3;
    //*a3 = -27.13E-3;

    //*isf = 177.8237E-3;
    //*usf = 255.8E-3;

    //Parameters from DFT(PBE Potential)--Ruifeng Zhang
    *a = 3.950E-10;
    *mu = 45.7E9; 
    *young = 125.8E9;

    *c0 = 308.65990E-3;
    *c1 = -63.66363E-3;
    *c2 = -48.76003E-3;
    *c3 = 12.024463E-3;
    *c4 = -1.11869E-3;
    *a1 = -61.34161E-3;
    *a3 = -23.33224E-3;
   
    *isf = 138.1E-3;
    *usf = 197.9E-3;
  }
  if(mflag == 7){ /*Silver*/
    //Parameters from DFT(PBE Potential)--Ruifeng Zhang
    *a = 4.16E-10;
    *mu = 27.5E9;
    *young = 74.8E9;

    *c0 = 179.573E-3;
    *c1 = -32.837E-3;
    *c2 = -39.578E-3;
    *c3 = 14.700E-3;
    *c4 = -0.989E-3;
    *a1 = -43.791E-3;
    *a3 = -17.254E-3;
    
    *isf = 17.8E-3; //37.0E-3; 
    *usf = 100.4E-3; //110.0E-3;
  }
  if(mflag == 8){ /*Platinum*/
    //Parameters from DFT(PBE Potential)--Ruifeng Zhang
    *a = 3.98E-10;
    *mu = 53.2E9;
    *young = 149.0E9;

    *c0 = 223.27233E-3;
    *c1 = -15.38077E-3;
    *c2 = -75.36747E-3;
    *c3 = 21.70531E-3;
    *c4 = -2.52876E-3;
    *a1 = 58.60391E-3;
    *a3 = -43.19019E-3;
    
    *isf = 253.7E-3; 
    *usf = 257.5E-3;
  }
  if(mflag == 9){ /*Rhodium*/
    //Parameters from DFT(PBE Potential)--Ruifeng Zhang
    *a = 3.84E-10; 
    *mu = 154.1E9;
    *young = 384.9E9;

    *c0 = 556.35590E-3;
    *c1 = -46.42878E-3;
    *c2 = -183.45940E-3;
    *c3 = 55.25737E-3;
    *c4 = -5.04421E-3;
    *a1 = 1.44561E-3;
    *a3 = -72.22364E-3;
    
    *isf = 184.6E-3; 
    *usf = 454.1E-3; 
  }
  if(mflag == 10){ /*Iridium*/
    //Parameters from DFT(PBE Potential)--Ruifeng Zhang
    *a = 3.88E-10;
    *mu = 222.3E9;
    *young = 549.7E9;

    *c0 = 617.61574E-3;
    *c1 = 3.28767E-3;
    *c2 = -283.1158E-3;
    *c3 = 96.12464E-3;
    *c4 = -10.56298E-3;
    *a1 = 157.20734E-3;
    *a3 = -109.4680E-3;
    
    *isf = 324.4E-3;
    *usf = 614.9E-3;
  }
 if(mflag == 11){ /*Pd-10%Au*/
    *a = 3.934E-10;
    *mu = 58.75E9; 
    *young = 160.62E9;

    *c0 = 331.361435E-3;
    *c1 = -78.784553E-3;
    *c2 = -21.920386E-3;
    *c3 = 4.464000E-3;
    *c4 = -7.070715E-3;
    *a1 = -100.61648E-3;
    *a3 = -4.343824E-3;
   
    *isf = 138.1E-3;
    *usf = 197.9E-3;
  }

  *C44 = *mu; //Pa
  *nu = *young/2.0/(*mu)-1.0;
  *C12 = 2.0**nu**C44/(1.0-2.0**nu); //Pa
  *C11 = 2.0**C44+*C12; //Pa
  *ll = *C12;
  
  *S11 = 1.0/(*young);
  *S12 = -*nu/(*young);
  *S44 = 2*(*S11-*S12);

  return;
}

/********************************************************************/

void set2D(double **xn, double **xb, int rank, int oflag)
{
  xn[0][0]= 0.0; //1.0/sqrt(3);
  xn[0][1]= 0.0; //1.0/sqrt(3);
  xn[0][2]= 1.0; //1.0/sqrt(3);
  
  if(oflag == 0){
    xb[0][0]= 1.0; //-1.0/2.0;
    xb[0][1]= 0.0; //1.0/2.0;
    xb[0][2]= 0.0;
  }
  else if(oflag == 1){
    xb[0][0]= 0.0; //-1.0/2.0;
    xb[0][1]= 1.0; //1.0/2.0;
    xb[0][2]= 0.0;
  }
  else{
    printf("Initial configuration specified has not been developed yet"); 
  }
 
  if(rank == 0){
    printf("Burgers vector b = (%lf , %lf , %lf )\n", xb[0][0],xb[0][1], xb[0][2]);	
    printf("Slip plane     n = [%lf,  %lf , %lf ]\n", xn[0][0],xn[0][1], xn[0][2]);
  }

  return;
}

/********************************************************************/

void set3D1pl(double **xn, double **xb, int rank, int oflag)
{

  int i, j, k;
  double s3;
  
  s3 = sqrt(3);
  for (i=0;i<3 ; i++) {
    xn[i][0]= 0.0; //1.0/s3;
    xn[i][1]= 0.0; //1.0/s3;
    xn[i][2]= 1.0; //1.0/s3;
  }
  
  if(oflag == 0){
    xb[0][0]= -0.5; //0.0; //1.0;
    xb[0][1]= -sqrt(3.0)/2.0; //1.0; //0.0;
    xb[0][2]= 0.0; //0.0; //0.0;
    
    xb[1][0]= 1.0; //sqrt(3.0)/2.0; //-0.5 
    xb[1][1]= 0.0; //-0.5; //sqrt(3.0)/2.0;
    xb[1][2]= 0.0; //0.0; //0.0;

    xb[2][0]= -0.5;//-sqrt(3.0)/2.0; //-0.5;
    xb[2][1]= sqrt(3.0)/2.0; //-0.5; //-sqrt(3.0)/2.0;
    xb[2][2]= 0.0; //0.0; //0.0;
  }
  else if(oflag == 1){
    xb[0][0]= sqrt(3.0)/2.0; //0.0; //1.0;
    xb[0][1]= -0.5; //1.0; //0.0;
    xb[0][2]= 0.0; //0.0; //0.0;
    
    xb[1][0]= 0.0; //sqrt(3.0)/2.0; //-0.5 
    xb[1][1]= 1.0; //-0.5; //sqrt(3.0)/2.0;
    xb[1][2]= 0.0; //0.0; //0.0;

    xb[2][0]= -sqrt(3.0)/2.0;//-sqrt(3.0)/2.0; //-0.5;
    xb[2][1]= -0.5; //-0.5; //-sqrt(3.0)/2.0;
    xb[2][2]= 0.0; //0.0; //0.0;
  }
  else{
    printf("Initial configuration specified has not been developed yet"); 
  }
  
  if(rank == 0){
    printf("Burgers vector b1 = (%lf , %lf , %lf )\n", xb[0][0],xb[0][1], xb[0][2]);
    printf("Burgers vector b2 = (%lf , %lf , %lf )\n", xb[1][0],xb[1][1], xb[1][2]);
    printf("Burgers vector b3 = (%lf , %lf , %lf )\n", xb[2][0],xb[2][1], xb[2][2]);	
    printf("Slip plane     n = [%lf,  %lf , %lf ]\n", xn[0][0],xn[0][1], xn[0][2]);
  }

  return;
}

/***********************************************************************/

void set3D2sys(double **xn, double **xb, int rank)
{
  xn[0][0]=1.0/sqrt(3);
  xn[0][1]=1.0/sqrt(3);
  xn[0][2]=1.0/sqrt(3);
  
  xb[0][0]=-1.0/2.0;
  xb[0][1]=1.0/2.0;
  xb[0][2]=0.0;
  
  xn[1][0]=1.0/sqrt(3);
  xn[1][1]=1.0/sqrt(3);
  xn[1][2]=1.0/sqrt(3);
  
  xb[1][0]=0.0;
  xb[1][1]=-1.0/2.0;
  xb[1][2]=1.0/2.0;
  
  if(rank == 0){
    printf("Burgers vector b1 = (%lf , %lf , %lf )\n", xb[0][0],xb[0][1], xb[0][2]);
    printf("Burgers vector b2 = (%lf , %lf , %lf )\n", xb[1][0],xb[1][1], xb[1][2]);
    printf("Slip plane     n  = [%lf,  %lf , %lf ]\n", xn[0][0],xn[0][1], xn[0][2]);
  }

  return;
}

/**********************************************************************/

void setfcc (double **xn, double **xb, int rank)
{
  int i, j, k;
  double s3;
  
  s3 = sqrt(3);
  for (i=0;i<3 ; i++) {
    xn[i][0]= 1.0/s3;
    xn[i][1]= 1.0/s3;
    xn[i][2]= 1.0/s3;
  }
  
  for (i=3;i<6 ; i++) {
    xn[i][0]= -1.0/s3;
    xn[i][1]= 1.0/s3;
    xn[i][2]= 1.0/s3;
  }
  
  for (i=6;i<9 ; i++) {
    xn[i][0]= 1.0/s3;
    xn[i][1]= -1.0/s3;
    xn[i][2]= 1.0/s3;
  }
  for (i=9;i<12 ; i++) {
    xn[i][0]= 1.0/s3;
    xn[i][1]= 1.0/s3;
    xn[i][2]= -1.0/s3;
  }
  xb[0][0] = -1.0;
  xb[0][1] = 1.0;
  xb[0][2] = 0.0;
  
  xb[1][0] = 0.0;
  xb[1][1] = -1.0;
  xb[1][2] = 1.0;
  
  xb[2][0] = 1.0;
  xb[2][1] = 0.0;
  xb[2][2] = -1.0;
  
  xb[3][0] = -1.0;
  xb[3][1] = -1.0;
  xb[3][2] = 0.0;
  
  xb[4][0] = 1.0;
  xb[4][1] = 0.0;
  xb[4][2] = 1.0;
  
  xb[5][0] = 0.0;
  xb[5][1] = -1.0;
  xb[5][2] = 1.0;
  
  xb[6][0] = -1.0;
  xb[6][1] = -1.0;
  xb[6][2] = 0.0;
  
  xb[7][0] = 1.0;
  xb[7][1] = 0.0;
  xb[7][2] = -1.0;
  
  xb[8][0] = 0.0;
  xb[8][1] = -1.0;
  xb[8][2] = -1.0;
  
  xb[9][0] = -1.0;
  xb[9][1] = 1.0;
  xb[9][2] = 0.0;
  
  xb[10][0] = 1.0;
  xb[10][1] = 0.0;
  xb[10][2] = 1.0;
  
  xb[11][0] = 0.0;
  xb[11][1] = -1.0;
  xb[11][2] = -1.0;
  
  if(rank == 0){
    printf("Slip systems\n");
  }
  for (i=0; i<12; i++){
    for (j=0; j<3; j++) {
      xb[i][j] = xb[i][j]/2.0;	
    }
    if(rank == 0){
      printf("b(%d) = (%lf , %lf , %lf )\n", i, xb[i][0],xb[i][1], xb[i][2]);	
      printf("n(%d) = [%lf,  %lf , %lf ]\n", i, xn[i][0],xn[i][1], xn[i][2]);
    }   
  }
return;	
}

/********************************************************************/

void frec( double *fx,double *fy, double *fz, double d1, double d2, double d3,
	   int lN1, int lxs, int N1, int N2, int N3, int rank)
{
  int i,j,k,ksym, nf; 
  
  for(i=0;i<lN1;i++)  //need to pass ln1
    {
      for(j=0;j<N2;j++)
	{
	  for(k=0;k<N3;k++)
	    {
	      nf = k+(j)*N3+(i)*N3*N2;
	      /* frecuency in x */
	      if (lxs+i==0) {  //need to pass lxs
		fx[nf]= 0.0;
	      }
	      if (lxs+i >= 1 && lxs+i < N1/2 ) {
		fx[nf]= (double)(lxs+i)/((double)(N1)/d1);
	      }
	      if (lxs+i >= N1/2) {
		fx[nf]= ((double)(lxs+i)-(double)(N1))/(double)(N1)/d1;	
	      }
	      /* frecuency in y */
	      if (j==0) {
		fy[nf]= 0.0;
	      }
	      if (j >= 1 && j < N2/2 ) {
		fy[nf]= (double)(j)/(double)(N2)/d2;
	      }
	      if (j >= N2/2) {
		fy[nf]= ((double)(j)-(double)(N2))/(double)(N2)/d2;
	      }				
	      /* frecuency in z */
	      if (k==0) {
		fz[nf]= 0.0;
	      }
	      if (k >= 1 && k < N3/2 ) {
		fz[nf]= (double)(k)/(double)(N3)/d3;
	      }	
	      if (k >= N3/2) {
		fz[nf]= ((double)(k)-(double)(N3))/(double)(N3)/d3;
	      }		
	      
	      //if(rank == 0 && i == 0){
	      //if(j < 10 && k <10){
	      //printf("%d %d %d    %lf %lf %lf \n", i, j, k, fx[nf], fy[nf],fz[nf]);  
	      //}  
	      //}
	      
	    }
	}
    }
  return;
}

/********************************************************************/

/*void seteps (double eps[NS][ND][ND], double epsv[NV][ND][ND],double xn[NS][ND], double xb[NS][ND], double dslip)
{
  double xn1[NV][ND], xb1[NV][ND];
  int i, j, k, v;
	
  for (v=0;v<NV;v++){
    for(i=0;i<ND;i++){
      xn1[v][i]=0.0;
      xb1[v][i]=0.0;
    }	
  }
	
  for (v=0;v<NV;v++){
    if(v==0 || v==1 || v==2)
      {xn1[v][0]=1.0;}
    if(v==3 || v==4 || v==5)
      {xn1[v][1]=1.0;}
    if(v==6 || v==7 || v==8)
      {xn1[v][2]=1.0;}
    if(v==0 || v==3 || v==6)
      {xb1[v][0]=1.0;}
    if(v==1 || v==4 || v==7)
      {xb1[v][1]=1.0;}
    if(v==2 || v==5 || v==8)
      {xb1[v][2]=1.0;}
    // printf("%lf %lf %lf %lf %lf %lf \n", xn1[v][0],xn1[v][1],xn1[v][2],xb1[v][0],xb1[v][1],xb1[v][2]);
  }
      
  for (v=0; v<NV;v++){
    printf("strainsys (%d) \n", v);	
    for (i=0; i<ND; i++) {
      for (j=0; j<ND; j++) {
	epsv[v][i][j]= xn1[v][i]*xb1[v][j];	
	printf("%lf ", epsv[v][i][j]);				
      }
      printf("\n");
    }
  }

  /*set eps */
	
  /*for (i=0; i<ND; i++) {
    for (j=0; j<ND; j++) {
      for (k=0; k<NS;k++){
	eps[k][i][j]= xb[k][i]*xn[k][j]/dslip;				
      }
    }
  }
	
  return;
}*/ 

/********************************************************************/

/* 
   A C-program for MT19937, with initialization improved 2002/1/26.
   Coded by Takuji Nishimura and Makoto Matsumoto.

   Before using, initialize the state by using init_genrand(seed)  
   or init_by_array(init_key, key_length).

   Copyright (C) 1997 - 2002, Makoto Matsumoto and Takuji Nishimura,
   All rights reserved.                          

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions
   are met:

     1. Redistributions of source code must retain the above copyright
        notice, this list of conditions and the following disclaimer.

     2. Redistributions in binary form must reproduce the above copyright
        notice, this list of conditions and the following disclaimer in the
        documentation and/or other materials provided with the distribution.

     3. The names of its contributors may not be used to endorse or promote 
        products derived from this software without specific prior written 
        permission.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
   CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


   Any feedback is very welcome.
   http://www.math.sci.hiroshima-u.ac.jp/~m-mat/MT/emt.html
   email: m-mat @ math.sci.hiroshima-u.ac.jp (remove space)
*/

#include <stdio.h>

/* Period parameters */  
#define N 624
#define M 397
#define MATRIX_A 0x9908b0dfUL   /* constant vector a */
#define UPPER_MASK 0x80000000UL /* most significant w-r bits */
#define LOWER_MASK 0x7fffffffUL /* least significant r bits */

static unsigned long mt[N]; /* the array for the state vector  */
static int mti=N+1; /* mti==N+1 means mt[N] is not initialized */

/* initializes mt[N] with a seed */
void init_genrand(unsigned long s)
{
    mt[0]= s & 0xffffffffUL;
    for (mti=1; mti<N; mti++) {
        mt[mti] = 
	    (1812433253UL * (mt[mti-1] ^ (mt[mti-1] >> 30)) + mti); 
        /* See Knuth TAOCP Vol2. 3rd Ed. P.106 for multiplier. */
        /* In the previous versions, MSBs of the seed affect   */
        /* only MSBs of the array mt[].                        */
        /* 2002/01/09 modified by Makoto Matsumoto             */
        mt[mti] &= 0xffffffffUL;
        /* for >32 bit machines */
    }
}

/* initialize by an array with array-length */
/* init_key is the array for initializing keys */
/* key_length is its length */
/* slight change for C++, 2004/2/26 */
void init_by_array(unsigned long init_key[], int key_length)
{
    int i, j, k;
    init_genrand(19650218UL);
    i=1; j=0;
    k = (N>key_length ? N : key_length);
    for (; k; k--) {
        mt[i] = (mt[i] ^ ((mt[i-1] ^ (mt[i-1] >> 30)) * 1664525UL))
          + init_key[j] + j; /* non linear */
        mt[i] &= 0xffffffffUL; /* for WORDSIZE > 32 machines */
        i++; j++;
        if (i>=N) { mt[0] = mt[N-1]; i=1; }
        if (j>=key_length) j=0;
    }
    for (k=N-1; k; k--) {
        mt[i] = (mt[i] ^ ((mt[i-1] ^ (mt[i-1] >> 30)) * 1566083941UL))
          - i; /* non linear */
        mt[i] &= 0xffffffffUL; /* for WORDSIZE > 32 machines */
        i++;
        if (i>=N) { mt[0] = mt[N-1]; i=1; }
    }

    mt[0] = 0x80000000UL; /* MSB is 1; assuring non-zero initial array */ 
}

/* generates a random number on [0,0xffffffff]-interval */
unsigned long genrand_int32(void)
{
    unsigned long y;
    static unsigned long mag01[2]={0x0UL, MATRIX_A};
    /* mag01[x] = x * MATRIX_A  for x=0,1 */

    if (mti >= N) { /* generate N words at one time */
        int kk;

        if (mti == N+1)   /* if init_genrand() has not been called, */
            init_genrand(5489UL); /* a default initial seed is used */

        for (kk=0;kk<N-M;kk++) {
            y = (mt[kk]&UPPER_MASK)|(mt[kk+1]&LOWER_MASK);
            mt[kk] = mt[kk+M] ^ (y >> 1) ^ mag01[y & 0x1UL];
        }
        for (;kk<N-1;kk++) {
            y = (mt[kk]&UPPER_MASK)|(mt[kk+1]&LOWER_MASK);
            mt[kk] = mt[kk+(M-N)] ^ (y >> 1) ^ mag01[y & 0x1UL];
        }
        y = (mt[N-1]&UPPER_MASK)|(mt[0]&LOWER_MASK);
        mt[N-1] = mt[M-1] ^ (y >> 1) ^ mag01[y & 0x1UL];

        mti = 0;
    }
  
    y = mt[mti++];

    /* Tempering */
    y ^= (y >> 11);
    y ^= (y << 7) & 0x9d2c5680UL;
    y ^= (y << 15) & 0xefc60000UL;
    y ^= (y >> 18);

    return y;
}

/* generates a random number on [0,0x7fffffff]-interval */
long genrand_int31(void)
{
    return (long)(genrand_int32()>>1);
}

/* generates a random number on [0,1]-real-interval */
double genrand_real1(void)
{
    return genrand_int32()*(1.0/4294967295.0); 
    /* divided by 2^32-1 */ 
}

/* generates a random number on [0,1)-real-interval */
double genrand_real2(void)
{
    return genrand_int32()*(1.0/4294967296.0); 
    /* divided by 2^32 */
}

/* generates a random number on (0,1)-real-interval */
double genrand_real3(void)
{
    return (((double)genrand_int32()) + 0.5)*(1.0/4294967296.0); 
    /* divided by 2^32 */
}

/* generates a random number on [0,1) with 53-bit resolution*/
double genrand_res53(void) 
{ 
    unsigned long a=genrand_int32()>>5, b=genrand_int32()>>6; 
    return(a*67108864.0+b)*(1.0/9007199254740992.0); 
} 
/* These real versions are due to Isaku Wada, 2002/01/09 added */

/*int main(void)
{
    int i;
    unsigned long init[4]={0x123, 0x234, 0x345, 0x456}, length=4;
    init_by_array(init, length);
    printf("1000 outputs of genrand_int32()\n");
    for (i=0; i<1000; i++) {
      printf("%10lu ", genrand_int32());
      if (i%5==4) printf("\n");
    }
    printf("\n1000 outputs of genrand_real2()\n");
    for (i=0; i<1000; i++) {
      printf("%10.8f ", genrand_real2());
      if (i%5==4) printf("\n");
    }
    return 0;
    }*/
