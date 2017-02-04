/*
Francisco Javier Hernández López.
Noviembre/2006

PROCEDIMIENTOS Y FUNCIONES PARA CREAR, PEDIR DATOS, Y DESPLEGAR UN SISTEMA DE ECUACIONES
*/

/*******************************************************************************
  FUNCION:     **asigna_memoria_matriz
  PARAMETROS:  n (Orden de la matriz de coeficientes).
  RETORNA:     regresa la direción de la matriz dinámica creada.
  DESCRIPCION: Crea una matriz dinámica de tipo double.
********************************************************************************/
double **asigna_memoria_matriz(double *m,int n)
{
        double **mat;
		int i;
		if((mat=(double **)malloc(n*sizeof(double)))==NULL){
			printf("\nNo hay memoria sufiecinte");
			exit(1);
		}
		for(i=0;i<n;i++){
			mat[i]=m+i*n;
		}
        return (mat);
}
/*******************************************************************************
  FUNCION:     **asigna_memoria_matriz_int
  PARAMETROS:  n (Orden de la matriz de coeficientes).
  RETORNA:     regresa la direción de la matriz dinámica creada.
  DESCRIPCION: Crea una matriz dinámica de tipo int.
********************************************************************************/
int **asigna_memoria_matriz_int(int *m,int n1,int n2)
{
        int **mat;
		int i;
		if((mat=(int **)malloc(n1*sizeof(int)))==NULL){
			printf("\nNo hay memoria sufiecinte");
			exit(1);
		}
		for(i=0;i<n1;i++){
			mat[i]=m+i*n2;
		}
        return (mat);
}
/*******************************************************************************
  FUNCION:     **asigna_memoria_matriz_double
  PARAMETROS:  n (Orden de la matriz de coeficientes).
  RETORNA:     regresa la direción de la matriz dinámica creada.
  DESCRIPCION: Crea una matriz dinámica de tipo int.
********************************************************************************/
double **asigna_memoria_matriz_double(double *m,int n1,int n2)
{
        double **mat;
		int i;
		if((mat=(double **)malloc(n1*sizeof(double)))==NULL){
			printf("\nNo hay memoria sufiecinte");
			exit(1);
		}
		for(i=0;i<n1;i++){
			mat[i]=m+i*n2;
		}
        return (mat);
}
float **asigna_memoria_matriz_float(float *m,int n1,int n2)
{
        float **mat;
		int i;
		if((mat=(float **)malloc(n1*sizeof(float)))==NULL){
			printf("\nNo hay memoria sufiecinte");
			exit(1);
		}
		for(i=0;i<n1;i++){
			mat[i]=m+i*n2;
		}
        return (mat);
}
long double **asigna_memoria_matriz_ldouble(long double *m,int n1,int n2)
{
        long double **mat;
		int i;
		if((mat=(long double **)malloc(n1*sizeof(long double)))==NULL){
			printf("\nNo hay memoria sufiecinte");
			exit(1);
		}
		for(i=0;i<n1;i++){
			mat[i]=m+i*n2;
		}
        return (mat);
}
/*******************************************************************************
  FUNCION:     *asigna_memoria_total
  PARAMETROS:  n (Orden de la matriz de coeficientes).
  RETORNA:     regresa la dirección del vector dinámico creado.
  DESCRIPCION: Crea un vector dinámico de tamaño n*n.
********************************************************************************/
double *asigna_memoria_total(int n)
{
	double *m;
	if((m=(double *)malloc(n*n*sizeof(double)))==NULL){
			printf("\nNo hay memoria sufiecinte");
			exit(1);
	}
	return (m);
}
/*******************************************************************************
  FUNCION:     *asigna_memoria_total_int
  PARAMETROS:  n1, n2 (Orden de la matriz de coeficientes).
  RETORNA:     regresa la dirección del vector dinámico creado.
  DESCRIPCION: Crea un vector dinámico de tamaño n1*n2 de tipo int.
********************************************************************************/
int *asigna_memoria_total_int(int n1,int n2)
{
	int *m;
	if((m=(int *)malloc(n1*n2*sizeof(int)))==NULL){
			printf("\nNo hay memoria sufiecinte");
			exit(1);
	}
	return (m);
}
/*******************************************************************************
  FUNCION:     *asigna_memoria_total_double
  PARAMETROS:  n (Orden de la matriz de coeficientes).
  RETORNA:     regresa la dirección del vector dinámico creado.
  DESCRIPCION: Crea un vector dinámico de tamaño n1*n2 de tipo int.
********************************************************************************/
double *asigna_memoria_total_double(int n1,int n2)
{
	double *m;
	if((m=(double *)malloc(n1*n2*sizeof(double)))==NULL){
			printf("\nNo hay memoria sufiecinte");
			exit(1);
	}
	return (m);
}
float *asigna_memoria_total_float(int n1,int n2)
{
	float *m;
	if((m=(float *)malloc(n1*n2*sizeof(float)))==NULL){
			printf("\nNo hay memoria sufiecinte");
			exit(1);
	}
	return (m);
}
long double *asigna_memoria_total_ldouble(int n1,int n2)
{
	long double *m;
	if((m=(long double *)malloc(n1*n2*sizeof(long double)))==NULL){
			printf("\nNo hay memoria sufiecinte");
			exit(1);
	}
	return (m);
}
/*******************************************************************************
  FUNCION:     *asigna_memoria_vector
  PARAMETROS:  n (Orden de la matriz triangular superior).
  RETORNA:     regresa la dirección del vector dinámico creado.
  DESCRIPCION: Crea un vector dinámico de tamaño n.
********************************************************************************/
double *asigna_memoria_vector(int n){
        double *vector;
        if ((vector=(double *)malloc(n*sizeof(double)))==NULL){
			printf("\nNo hay memoria sufiecinte");
			exit(1);
		}
        return vector;
}
float *asigna_memoria_vector_f(int n){
        float *vector;
        if ((vector=(float *)malloc(n*sizeof(float)))==NULL){
			printf("\nNo hay memoria sufiecinte");
			exit(1);
		}
        return vector;
}
long double *asigna_memoria_vector_ld(int n){
        long double *vector;
        if ((vector=(long double *)malloc(n*sizeof(long double)))==NULL){
			printf("\nNo hay memoria sufiecinte");
			exit(1);
		}
        return vector;
}
/*******************************************************************************
  FUNCION:     *asigna_memoria_vector
  PARAMETROS:  n (Orden de la matriz triangular superior).
  RETORNA:     regresa la dirección del vector dinámico creado.
  DESCRIPCION: Crea un vector dinámico de tamaño n.
********************************************************************************/
int *asigna_memoria_vector_int(int n){
        int *vector;
        if ((vector=(int *)malloc(n*sizeof(int)))==NULL){
			printf("\nNo hay memoria sufiecinte");
			exit(1);
		}
        return vector;
}
/*******************************************************************************
  PROCEDIMIENTO: carga_matriz
  PARAMETROS:    **mat (Matriz triangular superior).
                 n (Orden de la matriz triangular superior).
  DESCRIPCION:   Pide los valores de la matriz triangular superior
                 y los asigna a **mat.
********************************************************************************/
void carga_matriz(double **mat,int n)
{
        int i,j;
        for (i=0;i<n;i++){
                for (j=0;j<n;j++){
                        printf("matriz[%d][%d]= ",i+1,j+1);
//                        scanf_s("%lf",&mat[i][j]);
						scanf("%lf",&mat[i][j]);
                }
                printf("\n");
        }
}
/*******************************************************************************
  PROCEDIMIENTO: carga_tindep
  PARAMETROS:    *term (Vector de términos independientes).
                 n (Orden de la matriz triangular superior).
  DESCRIPCION:   Pide los valores del vector de términos independientes
                 y los asigna a *term.
********************************************************************************/
void carga_tindep(double *term,int n)
{
        int i;
        for (i=0;i<n;i++){
                printf("vector[%d]= ",i+1);
//                scanf_s("%lf",&term[i]);
                scanf("%lf",&term[i]);
        }
}
/*******************************************************************************
  PROCEDIMIENTO: imprime_sistema
  PARAMETROS:    **mat (Matriz triangular superior).
                 *term (Vector de términos independientes).
                 n (Orden de la matriz triangular superior).
  DESCRIPCION:   Imprime en pantalla el Sistema de Ecuaciones.
********************************************************************************/
void imprime_sistema(double **mat,double *term,int n)
{
        int i,j;
        for (i=0;i<n;i++){
                printf("\n");
                for (j=0;j<n;j++){
			if (mat[i][j]==-0.0)
                                printf("%.3lf ",0.0);
                        else
                                printf("%.3lf ",mat[i][j]);
                }
                printf(" x%d = ",i+1);
                printf("%.3lf",term[i]);
        }
}
/*******************************************************************************
  FUNCION:     **asigna_memoria_matriz
  PARAMETROS:  n (Orden de la matriz de coeficientes).
  RETORNA:     regresa la direción de la matriz dinámica creada.
  DESCRIPCION: Crea una matriz dinámica.
********************************************************************************/
double **asigna_memoria_mat_trid(double *m,int n1,int n2)
{
        double **mat;
		int i;
		if((mat=(double **)malloc(n1*sizeof(double)))==NULL){
			printf("\nNo hay memoria sufiecinte");
			exit(1);
		}
		for(i=0;i<n1;i++){
			mat[i]=m+i*n2;
		}
        return (mat);
}
/*******************************************************************************
  FUNCION:     *asigna_memoria_total
  PARAMETROS:  n (Orden de la matriz de coeficientes).
  RETORNA:     regresa la dirección del vector dinámico creado.
  DESCRIPCION: Crea un vector dinámico de tamaño n*n.
********************************************************************************/
double *asigna_memoria_trid(int n1,int n2 )
{
	double *m;
	if((m=(double *)malloc(n1*n2*sizeof(double)))==NULL){
			printf("\nNo hay memoria sufiecinte");
			exit(1);
	}
	return (m);
}

/*******************************************************************************
  FUNCION:     ***asigna_memoria_total_double
  PARAMETROS:  n1,n2,n3 (tamaño 3D).
  RETORNA:     regresa la dirección dinámica creada.
  DESCRIPCION: Crea un arreglo dinámico de tamaño n1*n2*n3 de tipo double.
********************************************************************************/
double *asigna_memoria_total_double(int n1,int n2,int n3)
{
	double *m;
	if((m=(double *)malloc(n1*n2*n3*sizeof(double)))==NULL){
			printf("\nNo hay memoria sufiecinte");
			exit(1);
	}
	return (m);
}
double **asigna_memoria_matriz_double(double *m,int n1,int n2,int n3)
{
        double **mat;
		int i;
		if((mat=(double **)malloc(n1*n2*sizeof(double)))==NULL){
			printf("\nNo hay memoria sufiecinte");
			exit(1);
		}
		for(i=0;i<n1*n2;i++){
			mat[i]=m+i*n3;
		}
        return (mat);
}
double ***asigna_memoria_3D_double(double **m,int n1,int n2,int n3)
{
        double ***mat3D;
		int i;
		if((mat3D=(double ***)malloc(n1*sizeof(double)))==NULL){
			printf("\nNo hay memoria sufiecinte");
			exit(1);
		}
		for(i=0;i<n1;i++){
			mat3D[i]=m+i*n2;
		}
		return (mat3D);
}
long double *asigna_memoria_total_ldouble(int n1,int n2,int n3)
{
	long double *m;
	if((m=(long double *)malloc(n1*n2*n3*sizeof(long double)))==NULL){
			printf("\nNo hay memoria sufiecinte");
			exit(1);
	}
	return (m);
}
long double **asigna_memoria_matriz_ldouble(long double *m,int n1,int n2,int n3)
{
        long double **mat;
		int i;
		if((mat=(long double **)malloc(n1*n2*sizeof(long double)))==NULL){
			printf("\nNo hay memoria sufiecinte");
			exit(1);
		}
		for(i=0;i<n1*n2;i++){
			mat[i]=m+i*n3;
		}
        return (mat);
}
long double ***asigna_memoria_3D_ldouble(long double **m,int n1,int n2,int n3)
{
        long double ***mat3D;
		int i;
		if((mat3D=(long double ***)malloc(n1*sizeof(long double)))==NULL){
			printf("\nNo hay memoria sufiecinte");
			exit(1);
		}
		for(i=0;i<n1;i++){
			mat3D[i]=m+i*n2;
		}
		return (mat3D);
}
float *asigna_memoria_total_float(int n1,int n2,int n3)
{
	float *m;
	if((m=(float *)malloc(n1*n2*n3*sizeof(float)))==NULL){
			printf("\nNo hay memoria sufiecinte");
			exit(1);
	}
	return (m);
}
float **asigna_memoria_matriz_float(float *m,int n1,int n2,int n3)
{
        float **mat;
		int i;
		if((mat=(float **)malloc(n1*n2*sizeof(float)))==NULL){
			printf("\nNo hay memoria sufiecinte");
			exit(1);
		}
		for(i=0;i<n1*n2;i++){
			mat[i]=m+i*n3;
		}
        return (mat);
}
float ***asigna_memoria_3D_float(float **m,int n1,int n2,int n3)
{
        float ***mat3D;
		int i;
		if((mat3D=(float ***)malloc(n1*sizeof(float)))==NULL){
			printf("\nNo hay memoria sufiecinte");
			exit(1);
		}
		for(i=0;i<n1;i++){
			mat3D[i]=m+i*n2;
		}
		return (mat3D);
}

