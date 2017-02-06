// STFM_GPU_OpenMP_float.cpp : Defines the entry point for the console application.
//

//-------pasar a linux, correr unas cuantas iteraciones y realizar fwi sin kernells, comparar
/*
	Programa que resuelve la ecuación de onda escalar en 2D (X-Z,t).
C   Se aplican fronteras absorbentes tipo PML.
C   La solución de la ecuación de onda se hace a través del método de
C   diferencias finitas de orden "no" en espacio y segundo orden en
C   tiempo con condiciones de Dirichlet P=0 en toda la frontera.
*/

// para ejecutar nvcc -o ejecutable.out programa.cu
// ejecutar con ./ejecutable.out
// para verificar tiempo nvprof ./ejecutable.out
//para revisar nucleoscuda es en usr/local/cuda-8.0/samples
//para reconocer device checar pagina 55
//

//-----Para compilar omp se usa nvcc -arch=sm_20 -Xcompiler -fopenmp -Igomp -o ejecutable.out programa.cu
//comentario al azat lol
#include <stdio.h>
#include <stdlib.h>
//#include <conio.h>
//---para conio.h se puede usar #include <curses.h> o <ncurses.h> pero se necesita descargar la libreria
#include <math.h>
#include "pidesist.h"

//#include <cutil.h>
//---para solucionar cutil se puede usar helper_math.h pero no sirve para operadores logicos
#include "Funciones_kernel.cu"
#include "FuncionesOxOz_kernel.cu"
//#include "Funciones_uxp_uzp_kernel.cu"

//#include <windows.h>
//---el windows.h hay que solucionarlo cambiando los comandos
#include <time.h>
//---para remplazar cuda safe call
#include "helper_cuda.h"

#include <cuda_runtime.h>
#include <cuda.h>
//OpenMP
#include <omp.h>

/*
implicit INTEGER (h-n)
implicit float PRECISION (a-g,o-z)
INTEGER,ALLOCATABLE,DIMENSION(:) :: nl,nr,nb,mn,ma,ml,mr

float PRECISION,ALLOCATABLE,DIMENSION(:) :: Tn,Tk,Tt,ricker,d,g
float PRECISION,ALLOCATABLE,DIMENSION(:,:) :: vel,uz,Oz
float PRECISION,ALLOCATABLE,DIMENSION(:,:,:) :: p,ux,Ox
*/



void Ajusta_profundidad(int Na,int no);
void Fuente_Ricker(float *ricker,int nrick,float fmid,float dt);
void Llenar_vect_enteros_pasos(int *mn,int *ma,int no,int Nzz,int Na);
void Estimando_vect_orden(float *g,float *d,int no);
void Modelo_velocidades(float **vel,int Nx,int Nz);
float Estabilidad_numerica(float **vel,int Nx,int Nz,float dt,float dh,float *vmax);
void Modif_vel_Expandir_malla(float **vel,int Nx,int Nz,int Na,int Nxx,int Nzz,float tx);
void Calculo_Ec_Onda(float ***p,float ***Ox,float ***ux,
					 float **uz,float **Oz,float *mvel,float **vel,
					 float *Tn,float *Tk,float *Tt,float *ricker,float *g,float *d,
					 int *mr,int *ml,int *nr,int *nl,int *nb,int *mn,int *ma,
					 float dt,float dh,float vmax,float da,float aa,float tx,float fmid,
					 int ixs,int nixs,int Na,int Nxx,int Nzz,int no,int izs,int ms,int nt,int ibs,int nrick);

void Guardar_Info_Var(float ***p,float ***Ox,float ***ux,float **Oz,float **uz,int Nzz,int Nxx,int Na);
void Pasar_Inf_Var(float ***p,float ***Ox,float ***ux,float **Oz,float **uz,
				   float *p_host,float *Ox_host,float *ux_host,float *Oz_host,float *uz_host,
				   int Nzz,int Nxx,int Na);

//División techo.
int iDivUp(int a, int b){
    return ((a % b) != 0) ? (a / b + 1) : (a / b);
}

cudaArray* cu_array_vel;
cudaArray* cu_array_p1;
cudaArray* cu_array_p2;
cudaArray* cu_array_mn;
cudaArray* cu_array_ma;

//int omp_get_num_procs(void);


int main (int argc, char **argv){

/*	----------------TEMPORALMENTE BLOQUEADO MIENTRAS SE ARREGLA EL SYSTEMTIME
	SYSTEMTIME st_ini,st_fin;
	
	GetSystemTime(&st_ini);
*/
	int num_proc=omp_get_num_procs();
	printf("\nNumero de procesadores: %d\n",num_proc);
	
	omp_set_num_threads(num_proc);//num_proc
    
/*
C----------------------------------------------------------------------
C     Número de orden de las diferencias en espacio
C             no=1  2do orden
C             no=2  4to orden
C             no=3  6to orden
C             no=4  8vo orden
C----------------------------------------------------------------------
*/
      int no=4;//Cambiarlo tambien en el kernel
      int ms=no+1;
/*
C----------------------------------------------------------------------
C     Parámetros generales
C     Unidades: Nx, Nz: [adimensional]
C               dh: [m]
C               dt: [seg]
C                v: [m/s]
C----------------------------------------------------------------------
*/
      int Nx=801;//1096,10000,5000
      int Nz=401;//401,2000
      float dh=25.0;
      int nt=4002;//4002,40
      float dt=0.002f;
/*C----------------------------------------------------------------------
C     Parámetros de frontera absorbente
C----------------------------------------------------------------------
C     Número de elementos absorbentes igual o mayor que el orden "no"*/
      int Na=25;//25,500

//C     Orden del polinomio de absorción
      float da=2;

//C     Parámetros que se cumplen para la transformación S=ak+Q/(aa+iw):
//C     ak>1 y aa>0. Si ak=1 y aa=0 entonces se tiene la PML original
      //float ak=1.0;
      float aa=0.5;
/*
C----------------------------------------------------------------------
C     Parámetros de las fuentes (posicion y frecuencia media)
C----------------------------------------------------------------------
C     Unidades ixs, izs, ibs: [puntos de malla]
C                       nixs: [adimensional]
C                       fmid: [Hz]
C----------------------------------------------------------------------*/
      int ixs=1;//                               !Posición inicial horizontal
	  int izs=3;//                               !Profundidad
      int ibs=1;//                               !Incremento
      int nixs=Nx;//                             !Número de fuentes
      float fmid=25.0;//                           !Frecuencia media
	
/*	C----------------------------------------------------------------------
C     La posicion de los receptores es igual que la de las fuentes
C     sólo ajusta su profundidad
C----------------------------------------------------------------------*/
	ixs=ixs+Na;
	Ajusta_profundidad(Na,no);
/*************************************************************************/
/*	C----------------------------------------------------------------------
C     Fuente tipo Ricker 2D de duración limitada
C----------------------------------------------------------------------*/
	float *ricker;
	int nrick=(int)floor(1/(fmid*dt)+0.5);
	printf("nrick %d\n",nrick);
      //Crear memoria para el vector ricker
	  //ALLOCATE(ricker(nrick))
	ricker=asigna_memoria_vector_f(nrick+1);
	Fuente_Ricker(ricker,nrick,fmid,dt);
	/*for(int i=0;i<nrick;i++)
		printf("%.15E \n",ricker[i]);*/
/*************************************************************************/      
/*C----------------------------------------------------------------------
C     Dimensiones de la malla computacional
C----------------------------------------------------------------------*/
	int Nzz=Nz+Na;                                       //!No. puntos en Z
    int Nxx=Nx+2*Na;                                     //!No. puntos en X
//C----------------------------------------------------------------------
/*C----------------------------------------------------------------------
C     Se llena el vector de enteros de pasos
C----------------------------------------------------------------------*/
	int *mn,*ma;
	//ALLOCATE(mn(Nzz))
	mn=asigna_memoria_vector_int(Nzz+1);
	//ALLOCATE(ma(Na))
	ma=asigna_memoria_vector_int(Na+1);
	Llenar_vect_enteros_pasos(mn,ma,no,Nzz,Na);
	/*printf("\n");
	for(int i=0;i<Nzz;i++)
		printf("%d ",mn[i]);
	printf("\n");
	for(int i=0;i<Na;i++)
		printf("%d ",ma[i]);*/
//C----------------------------------------------------------------------
/*C----------------------------------------------------------------------
C     Se estima el vector de orden
C----------------------------------------------------------------------*/
    //ALLOCATE(g(ms))
	//ALLOCATE(d(no))
	float *g,*d;
	//float g[6],d[6];
    g=asigna_memoria_vector_f(ms+1);
	d=asigna_memoria_vector_f(no+1);
	
	Estimando_vect_orden(g,d,no);
    /*printf("\n");
	for(int i=0;i<ms;i++)
		printf("%.15E ",g[i]);
	printf("\n");
	for(int i=0;i<no;i++)
		printf("%.15E ",d[i]);*/
//C----------------------------------------------------------------------
/*C----------------------------------------------------------------------
C     Modelo de velocidades
C----------------------------------------------------------------------*/
      //ALLOCATE(vel(Nzz,Nxx))
	float *mvel,**vel;
	mvel=asigna_memoria_total_float(Nzz+1,Nxx+1);
	vel=asigna_memoria_matriz_float(mvel,Nzz+1,Nxx+1);
	/*Inicializar Vel*/

	int i,j;
	int CHUNK=20;//Nzz=Nz+Na=401+25
//Esto le da informacion adicional al compilador para que comparta los datos listados entre todos los hilos
#pragma omp parallel shared(Nzz,Nxx,vel) private(i,j)
	{
#pragma omp for schedule(dynamic,CHUNK) nowait
		for(i=0;i<=Nzz;i++){
			for(j=0;j<Nxx;j++){
				vel[i][j]=0.0;
			}
		}
	}
	/*for(int i=1;i<26;i++){
		for(int j=1;j<3;j++){
			printf("%.2E ",vel[j][i]);
		}
		printf("\n");
	}*/
	Modelo_velocidades(vel,Nx,Nz);
	/*for(int i=1;i<26;i++){
		for(int j=1;j<3;j++){
			printf("%.2E ",vel[j][i]);
		}
		printf("\n");
	}*/
	/*for(int i=0;i<10;i++){
		for(int j=0;j<10;j++){
			printf("%f ",vel[j][i]);
		}
		printf("\n");
	}*/
//C----------------------------------------------------------------------
/*C----------------------------------------------------------------------
C     Condición de estabilidad numérica
C----------------------------------------------------------------------*/
	float tx,vmax;
	tx=Estabilidad_numerica(vel,Nx,Nz,dt,dh,&vmax);  
//C----------------------------------------------------------------------
/*C----------------------------------------------------------------------
C     Se hace vel=(dt^2/dx^2)*vel^2 y se expande la malla computacional
C----------------------------------------------------------------------*/
    /*for(int i=1;i<26;i++){
		for(int j=1;j<3;j++){
			printf("%.2E ",vel[j][i]);
		}
		printf("\n");
	}*/
	Modif_vel_Expandir_malla(vel,Nx,Nz,Na,Nxx,Nzz,tx);
	/*for(int i=1;i<26;i++){
		for(int j=1;j<3;j++){
			printf("%.2E ",vel[j][i]);
		}
		printf("\n");
	}*/
	
	//printf("%f",vel[Nzz-1][Nxx-1]);
//C----------------------------------------------------------------------
/*C----------------------------------------------------------------------
C     Despliegue en pantalla de los datos generales de la SOC generada
C----------------------------------------------------------------------*/
      printf("\n******************************************************\n");
      printf("    DATOS DE LA SECCION SISMICA EN TIEMPO GENERADA    \n");
      printf("             No. Trazas: %d\n",Nx);
      printf(" No. Muestras por traza: %d\n",nt-2);
      printf("   Intervalo de muestreo:  %f [seg]\n",dt);
      printf("\n******************************************************\n");
/*C      pause
C----------------------------------------------------------------------*/
      /*ALLOCATE(p(2,Nzz,Nxx))
      ALLOCATE(Ox(2,Nzz,Na))
      ALLOCATE(ux(2,Nzz,Na))
      ALLOCATE(Oz(Na,Nxx))
      ALLOCATE(uz(Na,Nxx))
      ALLOCATE(mr(no))
      ALLOCATE(ml(no))
      ALLOCATE(nr(no))
      ALLOCATE(nl(no))
      ALLOCATE(nb(no))
      ALLOCATE(Tn(Na))
      ALLOCATE(Tk(Na))
      ALLOCATE(Tt(Na))

	  INTEGER,ALLOCATABLE,DIMENSION(:) :: nl,nr,nb,mn,ma,ml,mr
		float PRECISION,ALLOCATABLE,DIMENSION(:) :: Tn,Tk,Tt,ricker,d,g
		float PRECISION,ALLOCATABLE,DIMENSION(:,:) :: vel,uz,Oz
		float PRECISION,ALLOCATABLE,DIMENSION(:,:,:) :: p,ux,Ox
	  */
	float *m1p,**m2p,***p;
	float *m1Ox,**m2Ox,***Ox;
	float *m1ux,**m2ux,***ux;

	/*float *mp1,**p1,*mp2,**p2;
	float *mOx1,**Ox1,*mOx2,**Ox2;
	float *mux1,**ux1,*mux2,**ux2;*/
	float *mOz,**Oz,*muz,**uz;
	int *mr,*ml,*nr,*nl,*nb;
	float *Tn,*Tk,*Tt;

	m1p=asigna_memoria_total_float(3,Nzz+1,Nxx+1);
	m2p=asigna_memoria_matriz_float(m1p,3,Nzz+1,Nxx+1);
	p=asigna_memoria_3D_float(m2p,3,Nzz+1,Nxx+1);
	/*mp1=asigna_memoria_total_float(Nzz+1,Nxx+1);
	p1=asigna_memoria_matriz_float(mp1,Nzz+1,Nxx+1);
	mp2=asigna_memoria_total_float(Nzz+1,Nxx+1);
	p2=asigna_memoria_matriz_float(mp2,Nzz+1,Nxx+1);*/

	m1Ox=asigna_memoria_total_float(3,Nzz+1,Na+1);
	m2Ox=asigna_memoria_matriz_float(m1Ox,3,Nzz+1,Na+1);
	Ox=asigna_memoria_3D_float(m2Ox,3,Nzz+1,Na+1);
	/*mOx1=asigna_memoria_total_float(Nzz+1,Na+1);
	Ox1=asigna_memoria_matriz_float(mOx1,Nzz+1,Na+1);
	mOx2=asigna_memoria_total_float(Nzz+1,Na+1);
	Ox2=asigna_memoria_matriz_float(mOx2,Nzz+1,Na+1);*/

	m1ux=asigna_memoria_total_float(3,Nzz+1,Na+1);
	m2ux=asigna_memoria_matriz_float(m1ux,3,Nzz+1,Na+1);
	ux=asigna_memoria_3D_float(m2ux,3,Nzz+1,Na+1);
	/*mux1=asigna_memoria_total_float(Nzz+1,Na+1);
	ux1=asigna_memoria_matriz_float(mux1,Nzz+1,Na+1);
	mux2=asigna_memoria_total_float(Nzz+1,Na+1);
	ux2=asigna_memoria_matriz_float(mux2,Nzz+1,Na+1);*/

	mOz=asigna_memoria_total_float(Na+1,Nxx+1);
	Oz=asigna_memoria_matriz_float(mOz,Na+1,Nxx+1);
	muz=asigna_memoria_total_float(Na+1,Nxx+1);
	uz=asigna_memoria_matriz_float(muz,Na+1,Nxx+1);
	
	mr=asigna_memoria_vector_int(no+1);
	ml=asigna_memoria_vector_int(no+1);
	nr=asigna_memoria_vector_int(no+1);
	nl=asigna_memoria_vector_int(no+1);
	nb=asigna_memoria_vector_int(no+1);
	
	Tn=asigna_memoria_vector_f(Na+1);
	Tk=asigna_memoria_vector_f(Na+1);
	Tt=asigna_memoria_vector_f(Na+1);


	
	Calculo_Ec_Onda(p,Ox,ux,
					uz,Oz,mvel,vel,
					Tn,Tk,Tt,ricker,g,d,
					mr,ml,nr,nl,nb,mn,ma,
					dt,dh,vmax,da,aa,tx,fmid,
					ixs,nixs,Na,Nxx,Nzz,no,izs,ms,nt,ibs,nrick);
/*	----------------TEMPORALMENTE BLOQUEADO MIENTRAS SE ARREGLA EL SYSTEMTIME
	GetSystemTime(&st_fin);
	printf("\nTime_ini %d : %d : %d : %d",st_ini.wHour,st_ini.wMinute,st_ini.wSecond,st_ini.wMilliseconds);
	printf("\nTime_fin %d : %d : %d : %d\n",st_fin.wHour,st_fin.wMinute,st_fin.wSecond,st_fin.wMilliseconds);
*/
	printf("Vamos Bien...\n");
	getchar();

	//Liberar Memoria
	free(ricker);free(mn);free(ma);free(g);free(d);free(mvel);free(vel);
	free(m1p);free(m2p);free(p);
	free(m1Ox);free(m2Ox);free(Ox);
	free(m1ux);free(m2ux);free(ux);
	free(mOz);free(Oz);free(muz);free(uz);free(mr);free(ml);
	free(nr);free(nl);free(nb);free(Tn);free(Tk);free(Tt);
	/*free(mp1);free(p1);free(mp2);free(p2);
	free(mOx1);free(Ox1);free(mOx2);free(Ox2);
	free(mux1);free(ux1);free(mux2);free(ux2);*/
	
}
void Ajusta_profundidad(int Na,int no){
	if (Na<no){ 
		 printf("Error. Se debe cumplir: Na>no\n");
		 getchar();
		 exit(0);
	}
	if ((no<1)||(4<no)){
		printf("Reajustar valor de no\n");
		getchar();
		exit(0);
	}
}
void Fuente_Ricker(float *ricker,int nrick,float fmid,float dt){
      float pi=4.0f * atan(1.0f);
      float cf=-2.0f * pi*pi*fmid*fmid;
	  float vmax,dl;
	  int i;
#pragma omp parallel shared(nrick,dt,cf,ricker) private(i,vmax,dl)
	{
#pragma omp for schedule(dynamic,2) nowait
		for(i=1;i<=nrick;i++){
		  vmax=(i-1)*dt;
		  dl=vmax*vmax;
		  ricker[i]=vmax*exp(cf*dl);
		}
	}

}
void Llenar_vect_enteros_pasos(int *mn,int *ma,int no,int Nzz,int Na){
	int i,j,k;
	for(i=1;i<=no;i++)
		mn[i]=i-1;
	
	for(j=i;j<=Nzz-Na;j++)
		mn[j]=no;

	k=Na;
	for(i=1;i<=no;i++){
		ma[k]=i-1;
		k=k-1;
	}
	for(j=i;j<=Na;j++){
		ma[k]=no;
		k=k-1;
	}
}
void Estimando_vect_orden(float *g,float *d,int no){
	switch (no){
		case 1:
			d[1]=1.0f/2.0f;
			g[1]=1.0f;
			g[2]=-2.0f;	
		break;
		case 2:
			d[1]=2.0f/3.0f;      d[2]=-1.0f/12.0f;
			g[1]=4.0f/3.0f;      g[2]=-1.0f/12.0f;
			g[3]=-5.0f/2.0f;
		break;
		case 3:
			d[1]=3.0f/4.0f;  d[2]=-3.0f/20.0f;  d[3]=1.0f/60.0f;
			g[1]=3.0f/2.0f;  g[2]=-3.0f/20.0f;  g[3]=1.0f/90.0f;
			g[4]=-49.0f/18.0f;
		break;
		case 4:
			d[1]=4.0f/5.0f; d[2]=-1.0f/5.0f; d[3]=4.0f/105.0f; d[4]=-1.0f/280.0f;
			g[1]=8.0f/5.0f; g[2]=-1.0f/5.0f; g[3]=8.0f/315.0f; g[4]=-1.0f/560.0f;
			g[5]=-205.0f/72.0f;
		break;
	
	
	}
}

void Modelo_velocidades(float **vel,int Nx,int Nz){
	 
	FILE *fp;
	int i,j;
	//open (UNIT=10,FILE='ModVelTabal.VEL',ACTION='Read',STATUS='old')
	if((fp=fopen("VelInt01_2D_IL_75.VEL","rb"))==NULL){
		printf("\n No se puede abrir el archivo...");
		getchar();
		exit(0);
	}
	for(i=1;i<=Nx;i++){
		for(j=1;j<=Nz;j++){
			fscanf(fp,"%f",&vel[j][i]);
		}
	}
	//close(10)	
	fclose(fp);

	/*
	if((fp=fopen("ModVelTabal.VEL","r"))==NULL){
		printf("\n No se puede abrir el archivo...");
		getchar();
		exit(0);
	}
	FILE *fp2;
	if((fp2=fopen("ModVelTabal2.VEL","wb"))==NULL){
		printf("\n No se puede abrir el archivo...");
		getchar();
		exit(0);
	}
	float val;
	while(!feof(fp)){
		fscanf(fp,"%f",&val);
		fprintf(fp2,"%f \n",val);
	} 

	fclose(fp);
	fclose(fp2);
	*/

}
float Estabilidad_numerica(float **vel,int Nx,int Nz,float dt,float dh,float *vmax){
	
	*vmax=-1.0;
	int i,j;
	for(i=1;i<=Nx;i++){
		for(j=1;j<=Nz;j++){
			if(vel[j][i]>*vmax)
				*vmax=vel[j][i];
		}
	}
    float tx=dt/dh;
	float cf=*vmax*sqrt(2.0f)*tx;
	//printf("%f  %E  %f",vmax,tx,cf);
	if (1.0f<cf){
          printf("Inestabilidad detectada 1<vmax*sqrt(2)*dt/dh= %f",cf);
          printf("Incrementa dh o disminuye dt");
          getchar();
		  exit(0);
	}
	return(tx);
}
void Modif_vel_Expandir_malla(float **vel,int Nx,int Nz,int Na,int Nxx,int Nzz,float tx){
	
	int i,j,k;

#pragma omp parallel shared(Nz,Nx,vel,tx) private(i,j)
	{
#pragma omp for schedule(dynamic,20) nowait
		for(i=1;i<=Nz;i++){
			for(j=1;j<=Nx;j++){
				vel[i][j]=(vel[i][j]*vel[i][j])*(tx*tx);
			}
		}
	}
	int H=Nx+Na;
	for(i=H;i>Na;i--){
		for(j=1;j<=Nz;j++){
			vel[j][i]=vel[j][i-Na];
		
		}
		for(k=j;k<=Nzz;k++){
			vel[k][i]=vel[Nz][i-Na];
		}
	}
#pragma omp parallel shared(Na,Nz,Nzz,vel) private(i,j)
	{
#pragma omp for schedule(dynamic,2) nowait
		for(i=1;i<=Na;i++){
			for(j=Nz+1;j<=Nzz;j++){
				vel[j][i]=vel[Nz][i];
			}
		}
	}
	for(i=Nxx;i>H;i--){
		for(j=1;j<=Nz;j++){
			vel[j][i]=vel[j][i-Na];
		}
		for(k=j;k<=Nzz;k++){
			vel[k][i]=vel[Nz][i-Na];
		}
	}
}


void Calculo_Ec_Onda(float ***p,float ***Ox,float ***ux,
					 float **uz,float **Oz,float *mvel,float **vel,
					 float *Tn,float *Tk,float *Tt,float *ricker,float *g,float *d,
					 int *mr,int *ml,int *nr,int *nl,int *nb,int *mn,int *ma,
					 float dt,float dh,float vmax,float da,float aa,float tx,float fmid,
					 int ixs,int nixs,int Na,int Nxx,int Nzz,int no,int izs,int ms,int nt,int ibs,int nrick){
/*C----------------------------------------------------------------------
C     Repetición del cálculo de la solución de la ecuación de onda
C      (se resuelve la ecuación de onda para cada fuente-receptor)
C----------------------------------------------------------------------*/

		float *p_host,*p_host2,*Ox_host,*ux_host,*Oz_host,*uz_host;
		p_host=asigna_memoria_vector_f(2*(Nzz+1)*(Nxx+1));
		p_host2=asigna_memoria_vector_f(Nxx-2*Na);
		Ox_host=asigna_memoria_vector_f(2*(Nzz+1)*(Na+1));
		ux_host=asigna_memoria_vector_f(2*(Nzz+1)*(Na+1));
		Oz_host=asigna_memoria_vector_f((Na+1)*(Nxx+1));
		uz_host=asigna_memoria_vector_f((Na+1)*(Nxx+1));

	




	float tx2=dt/(dh*dh);
    //open(12,file='STFM_Tabal_PML.RES')
	FILE *fp,*fp2;
	if((fp=fopen("../../STFM_Yax_PML_GPU_OpenMP_float.RES","wb"))==NULL){
		printf("\n No se puede abrir el archivo...");
		getchar();
		exit(0);
	}
	
    //int ixo=ixs;
	//int nF=1;
	float dl,pi,cf;
	int i,j,k1,k2,N,M,L;
	//int h,k;
	int i_time;

	//for(nF=1;nF<=nixs;nF++){
		//printf("Grabando fuente-receptor: %d \n",nF);
		/*	!----------------------------------------------------------------
		  !  Coeficientes de fronteras absorbentes
		  !  Qi(l)=Qmax*(l/d)^m
		  !  Qmax=(3*vp*log(1/R))/(2*d)
		  !  vp: velocidad, d: longitud de la PML, R: reflexión teórica
		  !----------------------------------------------------------------	*/
		dl=Na*dh;
		pi=3.0f*vmax/(2.0f*dl);
		cf=pi*log(10000.0f);
		pi=dl;
		for(i=1;i<=Na;i++){  
			Tn[i]=pow((pi/dl),da);//P
			Tn[i]=cf*Tn[i];//P
			pi=pi-dh;//Acum pi=pi-Na*dh
		}
		
		//Creamos memoria en device
		float *Tn_dev,*Tk_dev,*Tt_dev;
//--- cuda safe call ya no es viable
/*		CUDA_SAFE_CALL(cudaMalloc((void **)&Tn_dev,(Na+1)*sizeof(float)));
		CUDA_SAFE_CALL(cudaMalloc((void **)&Tk_dev,(Na+1)*sizeof(float)));
		CUDA_SAFE_CALL(cudaMalloc((void **)&Tt_dev,(Na+1)*sizeof(float)));
*/
		checkCudaErrors(cudaMalloc((void **)&Tn_dev,(Na+1)*sizeof(float)));
		checkCudaErrors(cudaMalloc((void **)&Tk_dev,(Na+1)*sizeof(float)));
		checkCudaErrors(cudaMalloc((void **)&Tt_dev,(Na+1)*sizeof(float)));

		//CUDA_SAFE_CALL(cudaMemcpy(Tn_dev,Tn,(Na+1)*sizeof(float),cudaMemcpyHostToDevice));
		dim3 blockNa(16);
		dim3 gridNa(iDivUp(Na+1,blockNa.x));
		/*Calcular_Tn<<<gridNa,blockNa>>>(Tn_dev,pi,dl,da,cf,Na+1);
		pi=pi-Na*dh;*/


		//CUDA_SAFE_CALL(cudaMemcpy(Tn,Tn_dev,(Na+1)*sizeof(float),cudaMemcpyDeviceToHost));
//---CUIDADO CON EL
		checkCudaErrors(cudaMemcpy(Tn_dev,Tn,(Na+1)*sizeof(float),cudaMemcpyHostToDevice));


		//printf("%f %f %f\n",dl,pi,cf);
		/*!----------------------------------------------------------------
      !  Cálculo de: Tk = Tn + aa  y: Tt=dt*(Tk+1)
      !----------------------------------------------------------------*/
        dl=dh*dh;
		for(i=1;i<=Na;i++){	
			cf=Tn[i] + aa;//No acum...
            Tk[i]=cf*dl;//P
            Tt[i]=(1.0f - dt*cf);//P
		}
		//Calcular_Tk_Tt<<<gridNa,blockNa>>>(Tk_dev,Tt_dev,Tn_dev,aa,dl,dt,Na+1);
		//CUDA_SAFE_CALL(cudaMemcpy(Tk,Tk_dev,(Na+1)*sizeof(float),cudaMemcpyDeviceToHost));
		//CUDA_SAFE_CALL(cudaMemcpy(Tt,Tt_dev,(Na+1)*sizeof(float),cudaMemcpyDeviceToHost));

		checkCudaErrors(cudaMemcpy(Tk_dev,Tk,(Na+1)*sizeof(float),cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(Tt_dev,Tt,(Na+1)*sizeof(float),cudaMemcpyHostToDevice));

		k1=1;
        k2=2;
	
		float *p_dev,*p_dev2,*Ox_dev,*ux_dev;
		size_t size1= 2*(Nzz+1)*(Nxx+1)*sizeof(float);
		checkCudaErrors(cudaMalloc((void **)&p_dev,size1));
		checkCudaErrors(cudaMalloc((void **)&p_dev2,(Nxx-2*Na)*sizeof(float)));
		checkCudaErrors(cudaMalloc((void **)&Ox_dev,2*(Nzz+1)*(Na+1)*sizeof(float)));
		checkCudaErrors(cudaMalloc((void **)&ux_dev,2*(Nzz+1)*(Na+1)*sizeof(float)));
		
		dim3 blockNzzNxx(16,16,2);
		dim3 gridNzzNxx(iDivUp(Nzz+1,blockNzzNxx.x),iDivUp(Nxx+1,blockNzzNxx.y),1);
		Inicializar_p<<<gridNzzNxx,blockNzzNxx>>>(p_dev,(Nzz+1),(Nxx+1),2);
		dim3 blockNzzNa(16,16,2);
		dim3 gridNzzNa(iDivUp(Nzz+1,blockNzzNa.x),iDivUp(Na+1,blockNzzNa.y),1);
		Inicializar_Ox_ux<<<gridNzzNa,blockNzzNa>>>(Ox_dev,ux_dev,(Nzz+1),(Na+1),2);


		cudaChannelFormatDesc description = cudaCreateChannelDesc<float>();
		checkCudaErrors(cudaMallocArray(&cu_array_p1, &description, Nxx+1,Nzz+1));
		checkCudaErrors(cudaMemcpyToArray(cu_array_p1,0,0,p_dev,(Nzz+1)*(Nxx+1)*sizeof(float),cudaMemcpyDeviceToDevice));
		checkCudaErrors(cudaBindTextureToArray(tex_p1,cu_array_p1));
		
		checkCudaErrors(cudaMallocArray(&cu_array_p2, &description, Nxx+1,Nzz+1));
		checkCudaErrors(cudaMemcpyToArray(cu_array_p2,0,0,p_dev+(Nzz+1)*(Nxx+1),(Nzz+1)*(Nxx+1)*sizeof(float),cudaMemcpyDeviceToDevice));
		checkCudaErrors(cudaBindTextureToArray(tex_p2,cu_array_p2));
		


		float *uz_dev,*Oz_dev;
		checkCudaErrors(cudaMalloc((void **)&uz_dev,(Na+1)*(Nxx+1)*sizeof(float)));
		checkCudaErrors(cudaMalloc((void **)&Oz_dev,(Na+1)*(Nxx+1)*sizeof(float)));
		
		dim3 blockNaNxx(16,16);
		dim3 gridNaNxx(iDivUp(Na+1,blockNaNxx.x),iDivUp(Nxx+1,blockNaNxx.y));
		Inicializar_uz_Oz<<<gridNaNxx,blockNaNxx>>>(uz_dev,Oz_dev,(Na+1),(Nxx+1));

		/*
		CUDA_SAFE_CALL(cudaMemcpy(p_host,p_dev,2*(Nzz+1)*(Nxx+1)*sizeof(float),cudaMemcpyDeviceToHost));
		CUDA_SAFE_CALL(cudaMemcpy(Ox_host,Ox_dev,2*(Nzz+1)*(Na+1)*sizeof(float),cudaMemcpyDeviceToHost));
		CUDA_SAFE_CALL(cudaMemcpy(ux_host,ux_dev,2*(Nzz+1)*(Na+1)*sizeof(float),cudaMemcpyDeviceToHost));
		CUDA_SAFE_CALL(cudaMemcpy(Oz_host,Oz_dev,(Na+1)*(Nxx+1)*sizeof(float),cudaMemcpyDeviceToHost));
		CUDA_SAFE_CALL(cudaMemcpy(uz_host,uz_dev,(Na+1)*(Nxx+1)*sizeof(float),cudaMemcpyDeviceToHost));
		
		Pasar_Inf_Var(p,Ox,ux,Oz,uz,p_host,Ox_host,ux_host,Oz_host,uz_host,Nzz,Nxx,Na);
		Guardar_Info_Var(p,Ox,ux,Oz,uz,Nzz,Nxx,Na);
		*/
		
		float *vel_dev;
		checkCudaErrors(cudaMalloc((void **)&vel_dev,(Nzz+1)*(Nxx+1)*sizeof(float)));
		checkCudaErrors(cudaMemcpy(vel_dev,mvel,(Nzz+1)*(Nxx+1)*sizeof(float),cudaMemcpyHostToDevice));

		//cudaChannelFormatDesc description = cudaCreateChannelDesc<float>();
		checkCudaErrors(cudaMallocArray(&cu_array_vel, &description, Nxx+1,Nzz+1));
		checkCudaErrors(cudaMemcpyToArray(cu_array_vel,0,0,mvel,(Nzz+1)*(Nxx+1)*sizeof(float),cudaMemcpyHostToDevice));
		checkCudaErrors(cudaBindTextureToArray(tex_vel,cu_array_vel));
		
		/*CUDA_SAFE_CALL(cudaMemcpy(mvel,vel_dev,(Nzz+1)*(Nxx+1)*sizeof(float),cudaMemcpyDeviceToHost));
		
		if((fp2=fopen("../../vel_parallel1.txt","wb"))==NULL){
			printf("\n No se puede abrir el archivo...");
			getchar();
			exit(0);
		}

		for(int i=0;i<=Nzz;i++){
			for(int j=0;j<Nxx;j++){
				fprintf(fp2,"%lf \n",mvel[i*Nxx+j]);
			}
		}
		fclose(fp2);

		Pasar_Valores_vel<<<gridNzzNxx,blockNzzNxx>>>(vel_dev,Nzz+1,Nxx+1);
		CUDA_SAFE_CALL(cudaMemcpy(mvel,vel_dev,(Nzz+1)*(Nxx+1)*sizeof(float),cudaMemcpyDeviceToHost));

		if((fp2=fopen("../../vel_parallel2.txt","wb"))==NULL){
			printf("\n No se puede abrir el archivo...");
			getchar();
			exit(0);
		}

		for(int i=0;i<=Nzz;i++){
			for(int j=0;j<Nxx;j++){
				fprintf(fp2,"%lf \n",mvel[i*Nxx+j]);
			}
		}
		fclose(fp2);*/

		/*for(i=1;i<=no;i++){
			mr[i]=0;
            ml[i]=0;
			nr[i]=0;
            nl[i]=0;
			nb[i]=0;
		}*/
		i_time=1;
        /*//!  Fuente única
        p[k1][izs][ixo]=ricker[i_time];*/
		/*C----------------------------------------------------------------------
		C     Fuentes múltiples
		C----------------------------------------------------------------------*/
		/*ixo=ixs;
		for(i=1;i<=nixs;i++){//PD
			p[k1][izs][ixo]=ricker[i_time];
			ixo=ixo+ibs;//Posición
		}*/
		float *ricker_dev;
		checkCudaErrors(cudaMalloc((void **)&ricker_dev,(Na+1)*sizeof(float)));
		checkCudaErrors(cudaMemcpy(ricker_dev,ricker,(Na+1)*sizeof(float),cudaMemcpyHostToDevice));

		/*for(i=1;i<=Na;i++)
			printf("%f \n",ricker[i]);*/
		
		/*dim3 block_nixs(nixs+1);
		dim3 grid_1(1);*/
		dim3 block_nixs(16);
		dim3 grid_1(iDivUp(nixs+1,block_nixs.x));
		
		Fuentes_multiples<<<grid_1,block_nixs>>>(p_dev,ricker[i_time],ricker[i_time+1],izs,ixs,ibs,nixs+1,Nzz+1,Nxx+1);
		
		//C----------------------------------------------------------------------
        i_time=2;
		/*//!  Fuente única
        p[k2][izs][ixo]=ricker[i_time];*/
		/*C----------------------------------------------------------------------
		C     Fuentes múltiples
		C----------------------------------------------------------------------*/
		/*ixo=ixs;
		for(i=1;i<=nixs;i++){//PD
			p[k2][izs][ixo]=ricker[i_time];
			ixo=ixo+ibs;//Posición
		}*/
		//C----------------------------------------------------------------------

		int *mn_dev,*ma_dev;
		checkCudaErrors(cudaMalloc((void **)&mn_dev,(Nzz+1)*sizeof(int)));
		checkCudaErrors(cudaMalloc((void **)&ma_dev,(Na+1)*sizeof(int)));
		checkCudaErrors(cudaMemcpy(mn_dev,mn,(Nzz+1)*sizeof(int),cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(ma_dev,ma,(Na+1)*sizeof(int),cudaMemcpyHostToDevice));

		cudaChannelFormatDesc description2 = cudaCreateChannelDesc<int>();
		checkCudaErrors(cudaMallocArray(&cu_array_mn, &description2,Nzz+1,1));
		checkCudaErrors(cudaMemcpyToArray(cu_array_mn,0,0,mn,(Nzz+1)*sizeof(int),cudaMemcpyHostToDevice));
			
		checkCudaErrors(cudaMallocArray(&cu_array_ma, &description2,Na+1,1));
		checkCudaErrors(cudaMemcpyToArray(cu_array_ma,0,0,ma,(Na+1)*sizeof(int),cudaMemcpyHostToDevice));
		
		/*printf("mn 1: \n ");
		for(int i=0;i<Nzz+1;i++){
			printf("%d ",mn[i]);
		}*/

		//CUDA_SAFE_CALL(cudaBindTexture(0,tex_mn,mn_dev,(Nzz+1)*sizeof(int)));
		//CUDA_SAFE_CALL(cudaBindTexture(0,tex_ma,ma_dev,(Na+1)*sizeof(int)));
		checkCudaErrors(cudaBindTextureToArray(tex_mn,cu_array_mn));
		checkCudaErrors(cudaBindTextureToArray(tex_ma,cu_array_ma));

		/*dim3 sizeblockmn(20);
		dim3 gridmn(iDivUp(Nzz+1,sizeblockmn.x));
		Pasar_Valores_vel<<<gridmn,sizeblockmn>>>(mn_dev,Nzz+1);
		CUDA_SAFE_CALL(cudaMemcpy(mn,mn_dev,(Nzz+1)*sizeof(int),cudaMemcpyDeviceToHost));
		*/
		/*printf("mn 2: \n ");
		for(int i=0;i<Nzz+1;i++){
			printf("%d ",mn[i]);
		}*/
		

		/*float *g_dev,*d_dev;
		CUDA_SAFE_CALL(cudaMalloc((void **)&g_dev,(ms+1)*sizeof(float)));
		CUDA_SAFE_CALL(cudaMalloc((void **)&d_dev,(no+1)*sizeof(float)));
		CUDA_SAFE_CALL(cudaMemcpy(g_dev,g,(ms+1)*sizeof(float),cudaMemcpyHostToDevice));
		CUDA_SAFE_CALL(cudaMemcpy(d_dev,d,(no+1)*sizeof(float),cudaMemcpyHostToDevice));*/
		
		/*for(int i=1;i<=ms;i++){
			printf("%f ",g[i]);
		}
		printf("\n");*/
		checkCudaErrors(cudaMemcpyToSymbol(g_dev,g, (ms+1)*sizeof(float),0,cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpyToSymbol(d_dev,d, (no+1)*sizeof(float),0,cudaMemcpyHostToDevice));
		/*cudaMemcpyFromSymbol(g,g_dev,6*sizeof(float),0,cudaMemcpyDeviceToHost); 
		for(int i=1;i<=ms;i++){
			printf("%f ",g[i]);
		}*/

		//!  Se calculan las funciones absorbentes
		//!  Para ux y uz
        N=Nxx;
        M=Nzz;
		
		
		dim3 blockNzz(16);
		dim3 gridNzz(iDivUp(Nzz+1,blockNzz.x));
		dim3 blockNxx(16);
		dim3 gridNxx(iDivUp(Nxx+1,blockNxx.x));
		/*
		Nzz-Na
		M=Na+1;
		N=Nxx-Na;
		*/
		dim3 blockNzzNaMN(16,16);
		dim3 gridNzzNaMN(iDivUp(Nzz-Na+1,blockNzzNaMN.x), iDivUp(Nxx-2*Na,blockNzzNaMN.y));

		dim3 blockNaNa(16,16);
		dim3 gridNaNa(iDivUp(Na+1,blockNaNa.x), iDivUp(Na+1,blockNaNa.y));

		dim3 blockNaNxxNa(16,16);
		dim3 gridNaNxxNa(iDivUp(Na+1,blockNaNxxNa.x), iDivUp(Nxx-2*Na,blockNaNxxNa.y));
		

		dim3 blockNzzNaNa(16,16);
		dim3 gridNzzNaNa(iDivUp(Nzz-Na+1,blockNzzNaNa.x), iDivUp(Na+1,blockNzzNaNa.y));

		//SYSTEMTIME st_ini,st_fin;

		//GetSystemTime(&st_ini);
		
		dim3 blockNaNzz(16,16);
		dim3 gridNaNzz(iDivUp(Na+1,blockNaNzz.x),iDivUp(Nzz+1,blockNaNzz.y));

		Calcular_funciones_absorbentes1<<<gridNaNzz,blockNaNzz>>>(p_dev,Ox_dev,ux_dev,Tn_dev,Tk_dev,
																 //mn_dev,ma_dev,//g_dev,d_dev,
																 tx2,dh,
																 N,no,Na+1,Nzz+1,Nxx+1,ms,k1-1);
		
		/*dim3 blockNaNxx(16,16);
		dim3 gridNaNxx(iDivUp(Na+1,blockNaNzz.x),iDivUp(Nxx+1,blockNaNzz.y));*/
		Calcular_funciones_absorbentes2<<<gridNaNxx,blockNaNxx>>>(p_dev,Oz_dev,uz_dev,Tn_dev,Tk_dev,
																 //mn_dev,ma_dev,//g_dev,d_dev,
																 tx2,dh,
																 M,no,Na+1,Nzz+1,Nxx+1,ms,k1-1);
		
		
		
		/*GetSystemTime(&st_fin);
		printf("\nTime_ini %d : %d",st_ini.wSecond,st_ini.wMilliseconds);
		printf("\nTime_fin %d : %d\n",st_fin.wSecond,st_fin.wMilliseconds);

		getchar();*/

		i_time=3;

		dim3 blockNxxNa(16);
		dim3 gridNxxNa(iDivUp(Nxx-2*Na,blockNxxNa.x));

		/*SYSTEMTIME st_ini,st_fin;
		GetSystemTime(&st_ini);*/
		dim3 gridNaNxxNzz(iDivUp(Na+1,blockNaNzz.x),iDivUp(Nzz+1,blockNaNzz.y));
		if(Nxx>Nzz){
			//gridNaNxxNzz.x=iDivUp(Na+1,blockNaNzz.x);
			gridNaNxxNzz.y=iDivUp(Nxx+1,blockNaNzz.y);
		}
		dim3 gridNuevo1(iDivUp(Na+1,blockNaNzz.x),iDivUp(Nzz+1,blockNaNzz.y));
		if(Nxx-2*Na > Nzz+1)
			gridNuevo1.y=iDivUp(Nxx-2*Na,blockNaNzz.y);

		/*CUDA_SAFE_CALL(cudaMemcpyToArray(cu_array_p1,0,0,p_dev,(Nzz+1)*(Nxx+1)*sizeof(float),cudaMemcpyDeviceToDevice));
		CUDA_SAFE_CALL(cudaBindTextureToArray(tex_p1,cu_array_p1));				
		CUDA_SAFE_CALL(cudaMemcpyToArray(cu_array_p2,0,0,p_dev+(Nzz+1)*(Nxx+1),(Nzz+1)*(Nxx+1)*sizeof(float),cudaMemcpyDeviceToDevice));
		CUDA_SAFE_CALL(cudaBindTextureToArray(tex_p2,cu_array_p2));*/
		//for(L=3;L<=3;L++){
		for(L=3;L<=nt;L++){
			//Para Ox y Oz
            N=Nxx;
            M=Nzz;
			//GetSystemTime(&st_ini);

			/*ParaOx<<<gridNaNzz,blockNaNzz>>>(p_dev,Ox_dev,Tt_dev,
											mn_dev,ma_dev,//g_dev,d_dev,
											tx,dh,
											N,no,Na+1,Nzz+1,Nxx+1,ms,k1-1);
			
			ParaOz<<<gridNaNxx,blockNaNxx>>>(p_dev,Oz_dev,Tt_dev,
											 mn_dev,ma_dev,//g_dev,d_dev,
											 tx,dh,
											 M,no,Na+1,Nzz+1,Nxx+1,ms,k1-1);
			*/
			/*ParaOxOz<<<gridNaNxxNzz,blockNaNzz>>>(p_dev,Ox_dev,Oz_dev,Tt_dev,
											      mn_dev,ma_dev,
											      tx,dh,
											      N,M,no,Na+1,Nzz+1,Nxx+1,ms,k1-1);
			*/
			
			ParaOxOz2<<<gridNaNxxNzz,blockNaNzz>>>(Ox_dev,Oz_dev,Tt_dev,
											      //mn_dev,ma_dev,
											      tx,dh,
											      N,M,no,Na+1,Nzz+1,Nxx+1,ms,k1-1);
			
			/*GetSystemTime(&st_fin);
			printf("\nTime_ini %d : %d",st_ini.wSecond,st_ini.wMilliseconds);
			printf("\nTime_fin %d : %d\n",st_fin.wSecond,st_fin.wMilliseconds);
			getchar();*/

			//!     Se resuelve la ecuación solo en el dominio real
			//!     Para u en n+2
			M=Na+1;
			N=Nxx-Na;
			
			/*CUDA_SAFE_CALL(cudaMemcpyToArray(cu_array_p1,0,0,p_dev,(Nzz+1)*(Nxx+1)*sizeof(float),cudaMemcpyDeviceToDevice));
			CUDA_SAFE_CALL(cudaBindTextureToArray(tex_p1,cu_array_p1));
			CUDA_SAFE_CALL(cudaMemcpyToArray(cu_array_p2,0,0,p_dev+(Nzz+1)*(Nxx+1),(Nzz+1)*(Nxx+1)*sizeof(float),cudaMemcpyDeviceToDevice));
			CUDA_SAFE_CALL(cudaBindTextureToArray(tex_p2,cu_array_p2));*/

			/*EcDominioReal<<<gridNzzNaMN,blockNzzNaMN>>>(p_dev,Oz_dev,Tt_dev,//vel_dev,
											 mn_dev,ma_dev,//g_dev,d_dev,
											 tx,dh,
											 M,N,no,Na+1,Nzz+1,Nxx+1,ms,k1-1,k2-1);
			*/
			
			EcDominioReal2<<<gridNzzNaMN,blockNzzNaMN>>>(p_dev,Oz_dev,Tt_dev,//vel_dev,
											 //mn_dev,ma_dev,//g_dev,d_dev,
											 tx,dh,
											 M,N,no,Na+1,Nzz+1,Nxx+1,ms,k1-1,k2-1);
			
			
			//!     Se resuelve para el dominio ficticio
			//!     Se calculan las funciones absorbentes ux-uz
			//!     para la siguiente iteracion
            N=Nxx;
			//!     para ux y p en i_time=n+2

			
			/*if(k1==1){
				CUDA_SAFE_CALL(cudaMemcpyToArray(cu_array_p1,0,0,p_dev,(Nzz+1)*(Nxx+1)*sizeof(float),cudaMemcpyDeviceToDevice));
				CUDA_SAFE_CALL(cudaBindTextureToArray(tex_p1,cu_array_p1));
			}
			else{
				CUDA_SAFE_CALL(cudaMemcpyToArray(cu_array_p2,0,0,p_dev+(Nzz+1)*(Nxx+1),(Nzz+1)*(Nxx+1)*sizeof(float),cudaMemcpyDeviceToDevice));
				CUDA_SAFE_CALL(cudaBindTextureToArray(tex_p2,cu_array_p2));
			}*/
			
			Para_uxp<<<gridNaNzz,blockNaNzz>>>(p_dev,Ox_dev,ux_dev,Tn_dev,Tk_dev,vel_dev,
											   //mn_dev,ma_dev,//g_dev,d_dev,
											   tx2,dh,
											   N,no,Na+1,Nzz+1,Nxx+1,ms,k1-1,k2-1);
			/*Para_uxp2<<<gridNaNzz,blockNaNzz>>>(p_dev,Ox_dev,ux_dev,Tn_dev,Tk_dev,//vel_dev,
											   //mn_dev,ma_dev,//g_dev,d_dev,
											   tx2,dh,
											   N,no,Na+1,Nzz+1,Nxx+1,ms,k1-1,k2-1);*/
			
			/*
			CUDA_SAFE_CALL(cudaMemcpyToArray(cu_array_p1,0,0,p_dev,(Nzz+1)*(Nxx+1)*sizeof(float),cudaMemcpyDeviceToDevice));
			CUDA_SAFE_CALL(cudaBindTextureToArray(tex_p1,cu_array_p1));
			CUDA_SAFE_CALL(cudaMemcpyToArray(cu_array_p2,0,0,p_dev+(Nzz+1)*(Nxx+1),(Nzz+1)*(Nxx+1)*sizeof(float),cudaMemcpyDeviceToDevice));
			CUDA_SAFE_CALL(cudaBindTextureToArray(tex_p2,cu_array_p2));
			*/
			//!     para uz and p in itime=n+2
            M=Nzz;

			Para_uzp<<<gridNaNa,blockNaNa>>>(p_dev,Oz_dev,uz_dev,Tn_dev,Tk_dev,vel_dev,
											 //mn_dev,ma_dev,//g_dev,d_dev,
											 tx2,dh,
											 M,no,Na+1,Nzz+1,Nxx+1,ms,k1-1,k2-1);
			
			/*Para_uzp2<<<gridNaNa,blockNaNa>>>(p_dev,Oz_dev,uz_dev,Tn_dev,Tk_dev,//vel_dev,
											 //mn_dev,ma_dev,//g_dev,d_dev,
											 tx2,dh,
											 M,no,Na+1,Nzz+1,Nxx+1,ms,k1-1,k2-1);*/
			
			/*Para_uxp_uzp<<<gridNaNzz,blockNaNzz>>>(p_dev,Ox_dev,Oz_dev,ux_dev,uz_dev,Tn_dev,Tk_dev,vel_dev,
						                     mn_dev,ma_dev,
						                     tx2,dh,
						                     N,M,no,Na+1,Nzz+1,Nxx+1,ms,k1-1,k2-1);
			*/
			/*Para_uxp_uzp<<<gridNuevo1,blockNaNzz>>>(p_dev,Ox_dev,Oz_dev,ux_dev,uz_dev,Tn_dev,Tk_dev,vel_dev,
						                     mn_dev,ma_dev,
						                     tx2,dh,
						                     N,M,no,Na+1,Nzz+1,Nxx+1,ms,k1-1,k2-1,Nxx-2*Na);
			*/
			
			/*Enmedio_uz<<<gridNaNxxNa,blockNaNxxNa>>>(p_dev,Oz_dev,uz_dev,Tn_dev,Tk_dev,vel_dev,
											 //mn_dev,ma_dev,//g_dev,d_dev,
											 tx2,dh,
											 M,no,Na+1,Nzz+1,Nxx+1,ms,k1-1,k2-1,Nxx-2*Na);

			*/
			if(k1==1){
				checkCudaErrors(cudaMemcpyToArray(cu_array_p1,0,0,p_dev,(Nzz+1)*(Nxx+1)*sizeof(float),cudaMemcpyDeviceToDevice));
				checkCudaErrors(cudaBindTextureToArray(tex_p1,cu_array_p1));
			}
			else{
				checkCudaErrors(cudaMemcpyToArray(cu_array_p2,0,0,p_dev+(Nzz+1)*(Nxx+1),(Nzz+1)*(Nxx+1)*sizeof(float),cudaMemcpyDeviceToDevice));
				checkCudaErrors(cudaBindTextureToArray(tex_p2,cu_array_p2));
			}			
			Enmedio_uz2<<<gridNaNxxNa,blockNaNxxNa>>>(p_dev,Oz_dev,uz_dev,Tn_dev,Tk_dev,
											 //mn_dev,ma_dev,//g_dev,d_dev,
											 tx2,dh,
											 M,no,Na+1,Nzz+1,Nxx+1,ms,k1-1,k2-1,Nxx-2*Na);


			Final_uzp<<<gridNaNa,blockNaNa>>>(p_dev,Oz_dev,uz_dev,Tn_dev,Tk_dev,vel_dev,
											 //mn_dev,ma_dev,//g_dev,d_dev,
											 tx2,dh,
											 M,no,Na+1,Nzz+1,Nxx+1,ms,k1-1,k2-1);

			
			
			ParteSuperior_ux<<<gridNzzNaNa,blockNzzNaNa>>>(p_dev,Ox_dev,ux_dev,Tn_dev,Tk_dev,vel_dev,
											  // mn_dev,ma_dev,//g_dev,d_dev,
											   tx2,dh,
											   N,no,Na+1,Nzz+1,Nxx+1,ms,k1-1,k2-1);

			
			/*//!     Fuente única
			if(i_time<(1/(fmid*dt))){
				p[k1][izs][ixo]=ricker[i_time];
			}*/
			/*C----------------------------------------------------------------------
			C     Fuentes múltiples
			C----------------------------------------------------------------------*/
			/*if(i_time<nrick){
				//ixo=ixs;
				Fuentes_multiples2<<<grid_1,block_nixs>>>(p_dev,ricker_dev,izs,i_time,ixs,ibs,nixs+1,Nzz+1,Nxx+1,k1-1);
			}*/

			if(i_time<nrick){
				//ixo=ixs;
				Fuentes_multiples3<<<grid_1,block_nixs>>>(p_dev,ricker[i_time],izs,ixs,ibs,nixs+1,Nzz+1,Nxx+1,k1-1);
			}

			if(k1==1){
				checkCudaErrors(cudaMemcpyToArray(cu_array_p1,0,0,p_dev,(Nzz+1)*(Nxx+1)*sizeof(float),cudaMemcpyDeviceToDevice));
				checkCudaErrors(cudaBindTextureToArray(tex_p1,cu_array_p1));
			}
			else{
				checkCudaErrors(cudaMemcpyToArray(cu_array_p2,0,0,p_dev+(Nzz+1)*(Nxx+1),(Nzz+1)*(Nxx+1)*sizeof(float),cudaMemcpyDeviceToDevice));
				checkCudaErrors(cudaBindTextureToArray(tex_p2,cu_array_p2));
			}


			/* !     Se imprime la sección en tiempo sin los puntos de frontera
			!     laterales y considerando únicamente receptores en la
			!     superficie*/
            /*do J=Na+1,Nxx-Na
		         WRITE(12,*)p(k1,1,J)
			end do*/

			Pasar_Valores<<<gridNxxNa,blockNxxNa>>>(p_dev2,p_dev,Nxx-2*Na,Nxx+1,Na+1,Nzz+1,k1-1);
			//CUDA_SAFE_CALL(cudaMemcpy(p_host,p_dev,2*(Nzz+1)*(Nxx+1)*sizeof(float),cudaMemcpyDeviceToHost));
			checkCudaErrors(cudaMemcpy(p_host2,p_dev2,(Nxx-2*Na)*sizeof(float),cudaMemcpyDeviceToHost));
			
			/*for(i=1;i<=Nzz;i++){
				for(j=1;j<=Nxx;j++){
					p[1][i][j]=p_host[j*Nzz+i];
					p[2][i][j]=p_host[Nzz*Nxx+j*Nzz+i];
				}
			}*/

			/*for(j=Na+1;j<=Nxx-Na;j++){
				fprintf(fp,"%f \n",p_host[(k1-1)*(Nzz+1)*(Nxx+1) + (Nxx+1) + j]);
				//printf("%.15f \n",p_host[(k1-1)*(Nzz+1)*(Nxx+1) + (Nxx+1) + j]);
			}*/
			for(j=0;j<(Nxx-2*Na);j++)
				fprintf(fp,"%f \n",p_host2[j]);
			


			/*
			for(j=Na+1;j<=Nxx-Na;j++){
				fprintf(fp,"%f \n",p[k1][1][j]);
			}*/

			//printf("%d  %.15E \n",i_time,p[k1][1][ixo]);//WRITE(12,*)p(k1,1,ixo)
			//fprintf(fp,"%.15E \n",p[k1][1][ixo]);
            N=k1;
            k1=k2;
            k2=N;
			//WRITE(*,*)'Grabando tiempo',i_time
			printf("\nGrabando tiempo: %d",i_time);
            i_time=i_time+1;  

			

		}
		//ixo=ixo+ibs;
	//}	
		
	/*
		CUDA_SAFE_CALL(cudaMemcpy(p_host,p_dev,2*(Nzz+1)*(Nxx+1)*sizeof(float),cudaMemcpyDeviceToHost));
		CUDA_SAFE_CALL(cudaMemcpy(Ox_host,Ox_dev,2*(Nzz+1)*(Na+1)*sizeof(float),cudaMemcpyDeviceToHost));
		CUDA_SAFE_CALL(cudaMemcpy(ux_host,ux_dev,2*(Nzz+1)*(Na+1)*sizeof(float),cudaMemcpyDeviceToHost));
		CUDA_SAFE_CALL(cudaMemcpy(Oz_host,Oz_dev,(Na+1)*(Nxx+1)*sizeof(float),cudaMemcpyDeviceToHost));
		CUDA_SAFE_CALL(cudaMemcpy(uz_host,uz_dev,(Na+1)*(Nxx+1)*sizeof(float),cudaMemcpyDeviceToHost));
		
		Pasar_Inf_Var(p,Ox,ux,Oz,uz,p_host,Ox_host,ux_host,Oz_host,uz_host,Nzz,Nxx,Na);
		Guardar_Info_Var(p,Ox,ux,Oz,uz,Nzz,Nxx,Na);
	*/	
	/*GetSystemTime(&st_fin);
	printf("\nTime_ini %d : %d : %d : %d",st_ini.wHour,st_ini.wMinute,st_ini.wSecond,st_ini.wMilliseconds);
	printf("\nTime_fin %d : %d : %d : %d\n",st_fin.wHour,st_fin.wMinute,st_fin.wSecond,st_fin.wMilliseconds);
	*/

	fclose(fp);

	/*Liberar memoria del device*/
	cudaFree(Tn_dev);
	cudaFree(Tk_dev);
	cudaFree(Tt_dev);

	cudaFree(p_dev);cudaFree(p_dev2);cudaFree(Ox_dev);cudaFree(ux_dev);
	cudaFree(uz_dev);cudaFree(Oz_dev);
	
	cudaFree(vel_dev);

	cudaFree(ricker_dev);
	cudaFree(mn_dev);cudaFree(ma_dev);
	/*cudaFree(mr_dev);cudaFree(ml_dev);
	cudaFree(nr_dev);cudaFree(nl_dev);cudaFree(nb_dev);*/
	cudaFreeArray(cu_array_vel);
	cudaFreeArray(cu_array_p1);
	cudaFreeArray(cu_array_p2);
	cudaFreeArray(cu_array_mn);
	cudaFreeArray(cu_array_ma);

	free(p_host);
	free(p_host2);
	free(Ox_host);free(ux_host);
	free(Oz_host);free(uz_host);

}

void Guardar_Info_Var(float ***p,float ***Ox,float ***ux,float **Oz,float **uz,int Nzz,int Nxx,int Na){
	FILE *fp_p,*fp_Ox,*fp_ux,*fp_Oz,*fp_uz;
	
	if((fp_p=fopen("../../p2.txt","wb"))==NULL){
		printf("\n No se puede abrir el archivo...");
		getchar();
		exit(0);
	}
	if((fp_Ox=fopen("../../Ox2.txt","wb"))==NULL){
		printf("\n No se puede abrir el archivo...");
		getchar();
		exit(0);
	}
	if((fp_ux=fopen("../../ux2.txt","wb"))==NULL){
		printf("\n No se puede abrir el archivo...");
		getchar();
		exit(0);
	}
	if((fp_Oz=fopen("../../Oz2.txt","wb"))==NULL){
		printf("\n No se puede abrir el archivo...");
		getchar();
		exit(0);
	}
	if((fp_uz=fopen("../../uz2.txt","wb"))==NULL){
		printf("\n No se puede abrir el archivo...");
		getchar();
		exit(0);
	}
	int i,j,k;

	for(k=1;k<=2;k++){
		for(i=1;i<=Nzz;i++){
			for(j=1;j<=Nxx;j++){
				fprintf(fp_p,"%f ",p[k][i][j]);

			}
			fprintf(fp_p,"\n");

			for(j=1;j<=Na;j++){
				fprintf(fp_Ox,"%f ",Ox[k][i][j]);
				fprintf(fp_ux,"%f ",ux[k][i][j]);
			}
			fprintf(fp_Ox,"\n");
			fprintf(fp_ux,"\n");

		}
		
		fprintf(fp_p,"\n\n");
		fprintf(fp_p,"Mitad\n\n");
		
		fprintf(fp_Ox,"\n\n");
		fprintf(fp_Ox,"Mitad\n\n");
		fprintf(fp_ux,"\n\n");
		fprintf(fp_ux,"Mitad\n\n");
	}

	for(i=1;i<=Na;i++){
		for(j=1;j<=Nxx;j++){
			fprintf(fp_Oz,"%f ",Oz[i][j]);
			fprintf(fp_uz,"%f ",uz[i][j]);
		}
		fprintf(fp_Oz,"\n");
		fprintf(fp_uz,"\n");
	}

	fclose(fp_p);
	fclose(fp_Ox);
	fclose(fp_ux);
	fclose(fp_Oz);
	fclose(fp_uz);
}

void Pasar_Inf_Var(float ***p,float ***Ox,float ***ux,float **Oz,float **uz,
				   float *p_host,float *Ox_host,float *ux_host,float *Oz_host,float *uz_host,
				   int Nzz,int Nxx,int Na){

	int i,j,k;

	for(k=1;k<=2;k++){
		for(i=1;i<=Nzz;i++){
			for(j=1;j<=Nxx;j++){
				p[k][i][j]=p_host[(k-1)*(Nzz+1)*(Nxx+1) + i*(Nxx+1) + j];
			}
			for(j=1;j<=Na;j++){
				Ox[k][i][j]=Ox_host[(k-1)*(Nzz+1)*(Na+1) + i*(Na+1) + j];
				ux[k][i][j]=ux_host[(k-1)*(Nzz+1)*(Na+1) + i*(Na+1) + j];
			}
		}	
	}
	for(i=1;i<=Na;i++){
		for(j=1;j<=Nxx;j++){
			Oz[i][j]=Oz_host[i*(Nxx+1)+j];
			uz[i][j]=uz_host[i*(Nxx+1)+j];	
		}
	}
}

 
