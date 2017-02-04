__constant__ float g_dev[6]; //ms=no+1
__constant__ float d_dev[6]; //no=4
texture<float, 2> tex_vel;
texture<int, 1> tex_mn;
texture<int, 1> tex_ma;

texture<float, 2> tex_p1;
texture<float, 2> tex_p2;


__global__ void Calcular_Tn(float *Tn_dev,float pi,float dl,float da,float cf,int N){

	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx>0 && idx<N){
		Tn_dev[idx]=cf*pow((pi/dl),da);
	}
}

__global__ void Calcular_Tk_Tt(float *Tk_dev,float *Tt_dev,float *Tn_dev,float aa,float dl,float dt,int N){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	float cf;
	if (idx>0 && idx<N){
		cf=Tn_dev[idx] + aa;//No acum...
        Tk_dev[idx]=cf*dl;//P
        Tt_dev[idx]=(1.0f - dt*cf);//P
	}

}

__global__ void Inicializar_p(float *p_dev,int Nzz,int Nxx,int Dep){
	int idx=blockIdx.x * blockDim.x + threadIdx.x;
	int idy=blockIdx.y * blockDim.y + threadIdx.y;
	int idz=blockIdx.z * blockDim.z + threadIdx.z;
	
	if (idx>0 && idx<Nzz && idy>0 && idy<Nxx && idz<Dep){
		p_dev[idz*Nzz*Nxx + idy + idx*Nxx]=0.0;//Error idy*Nzz+idx
	}

}
__global__ void Inicializar_Ox_ux(float *Ox_dev,float *ux_dev,int Nzz,int Na,int Dep){
	int idx=blockIdx.x * blockDim.x + threadIdx.x;
	int idy=blockIdx.y * blockDim.y + threadIdx.y;
	int idz=blockIdx.z * blockDim.z + threadIdx.z;
	
	if (idx>0 && idx<Nzz && idy>0 && idy<Na && idz<Dep){
		Ox_dev[idz*Nzz*Na + idy + idx*Na]=0.0;
		ux_dev[idz*Nzz*Na + idy + idx*Na]=0.0;
	}

}
__global__ void Inicializar_uz_Oz(float *uz_dev,float *Oz_dev,int Na,int Nxx){
	int idx=blockIdx.x * blockDim.x + threadIdx.x;
	int idy=blockIdx.y * blockDim.y + threadIdx.y;
	if (idx>0 && idx<Na && idy>0 && idy<Nxx){
		Oz_dev[idy + idx*Nxx]=0.0;
		uz_dev[idy + idx*Nxx]=0.0;
	}


}


/*__global__ void Fuentes_multiples(float *p_dev,float *ricker_dev,int izs,int i_time,int ixs,int ibs,int N,int Nzz,int Nxx){

	int idx=blockIdx.x * blockDim.x + threadIdx.x;

	__shared__ float val_ricker[2];
	val_ricker[0]=ricker_dev[i_time];
	val_ricker[1]=ricker_dev[i_time+1];
	__syncthreads();

	if(idx>=0 && idx<N-1){
		p_dev[izs*Nxx+(idx+ixs)]=val_ricker[0];
		p_dev[Nzz*Nxx+ izs*Nxx + (idx+ixs)]=val_ricker[1];
	}
}*/
__global__ void Fuentes_multiples(float *p_dev,float ricker_itime1,float ricker_itime2,int izs,int ixs,int ibs,int N,int Nzz,int Nxx){

	int idx=blockIdx.x * blockDim.x + threadIdx.x;

	if(idx>=0 && idx<N-1){
		p_dev[izs*Nxx+(idx+ixs)]=ricker_itime1;
		p_dev[Nzz*Nxx+ izs*Nxx + (idx+ixs)]=ricker_itime2;
	}
}




__global__ void Fuentes_multiples2(float *p_dev,float *ricker_dev,int izs,int i_time,int ixs,int ibs,int N,int Nzz,int Nxx,int k1){

	int idx=blockIdx.x * blockDim.x + threadIdx.x;

	__shared__ float val_ricker[1];
	val_ricker[0]=ricker_dev[i_time];
	__syncthreads();

	if(idx>=0 && idx<N-1){
		p_dev[k1*Nzz*Nxx + izs*Nxx + (idx+ixs)]=val_ricker[0];
	}
}

__global__ void Fuentes_multiples3(float *p_dev,float ricker_itime,int izs,int ixs,int ibs,int N,int Nzz,int Nxx,int k1){

	int idx=blockIdx.x * blockDim.x + threadIdx.x;

	if(idx>=0 && idx<N-1){
		p_dev[k1*Nzz*Nxx + izs*Nxx + (idx+ixs)]=ricker_itime;
	}
}






__global__ void Calcular_vecinos(int *mr_dev,int *ml_dev,int *nr_dev,int *nl_dev,int *nb_dev,
								 int *mn_dev,int *ma_dev,int no,int Na,int N){

	int idx=blockIdx.x * blockDim.x + threadIdx.x;
	int h,k;
	if(idx>0 && idx<Na){
		//para ux en n+1
		for(h=1;h<=mn_dev[idx];h++){
			nl_dev[(idx-1)*no+h]=idx-h;  nr_dev[(idx-1)*no+h]=idx+h;
			ml_dev[(idx-1)*no+h]=N-h;  mr_dev[(idx-1)*no+h]=N+h;	
		}
		for(k=h;k<no;k++){
			nr_dev[(idx-1)*no+k]=idx+k;  ml_dev[(idx-1)*no+k]=N-k;
		}
		for(h=1;h<=ma_dev[idx];h++){
			nb_dev[(idx-1)*no+h]=idx+h;
		}
	}
}
__global__ void Calcular_funciones_absorbentes1(float *p_dev,float *Ox_dev,float *ux_dev,float *Tn_dev,float *Tk_dev,
											   //int *mn_dev,int *ma_dev,//float *g_dev,float *d_dev,
											   float tx2,float dh,
											   int N,int no,int Na,int Nzz,int Nxx,int ms,int k1){

	int idx=blockIdx.x * blockDim.x + threadIdx.x;
	int idy=blockIdx.y * blockDim.y + threadIdx.y;

	float dl,da,pi,cf,ak,aa;
	int h,k;

	if (idx>0 && idx<Na && idy>0 && idy<Nzz){
			int ind_p=k1*Nzz*Nxx+idy*Nxx;
			int ind_Ox=Nzz*Na + idy*Na;
			int ind_ux=idx + idy*Na;
			int ind_ux2=Nzz*Na + idx + idy*Na;
			dl=g_dev[ms]*p_dev[ind_p + idx]; 
			pi=g_dev[ms]*p_dev[ind_p + (Nxx-idx)];
			da=0.0;                  cf=0.0;
			int val_mn=tex1D(tex_mn,idx);
			for(h=1;h<=val_mn;h++){
				dl=dl+g_dev[h]*(p_dev[ind_p + (idx-h)]+p_dev[ind_p + (idx+h)]);
				da=da-d_dev[h]*Ox_dev[(idx-h) + idy*Na]*Tn_dev[(idx-h)];
				pi=pi+g_dev[h]*(p_dev[ind_p + ((Nxx-idx)-h)]+p_dev[ind_p + ((Nxx-idx)+h)]);
				cf=cf+d_dev[h]*Ox_dev[ind_Ox + (idx-h)]*Tn_dev[(idx-h)];	
			}
			for(k=h;k<=no;k++){
				dl=dl+g_dev[k]*p_dev[ind_p + (idx+k)];
				pi=pi+g_dev[k]*p_dev[ind_p + ((Nxx-idx)-k)];
			}
			int val_ma=tex1D(tex_ma,idx);
			for(h=1;h<=val_ma;h++){
				//da=da+d_dev[h]*Ox_dev[(idx+h) + idy*Nxx]*Tn_dev[(idx+h)]; //Error2 es Na en vez de Nxx
				da=da+d_dev[h]*Ox_dev[(idx+h) + idy*Na]*Tn_dev[(idx+h)];
				cf=cf-d_dev[h]*Ox_dev[ind_Ox + (idx+h)]*Tn_dev[(idx+h)];
			}
			aa=dl - da*dh - Tk_dev[idx]*ux_dev[ind_ux];
			ux_dev[ind_ux]=tx2*aa+ux_dev[ind_ux];
			ak=pi - cf*dh - Tk_dev[idx]*ux_dev[ind_ux2];
			ux_dev[ind_ux2]=tx2*ak+ux_dev[ind_ux2];		
	}
}
__global__ void Calcular_funciones_absorbentes2(float *p_dev,float *Oz_dev,float *uz_dev,float *Tn_dev,float *Tk_dev,
											   //int *mn_dev,int *ma_dev,//float *g_dev,float *d_dev,
											   float tx2,float dh,
											   int M,int no,int Na,int Nzz,int Nxx,int ms,int k1){


	int idx=blockIdx.x * blockDim.x + threadIdx.x;
	int idy=blockIdx.y * blockDim.y + threadIdx.y;

	int h,k;

	float pi,cf,aa;

	if (idx>0 && idx<Na && idy>0 && idy<Nxx){
	
		int ind_p=k1*Nzz*Nxx + idy;
		//int ind_Oz=idy;
		pi=g_dev[ms]*p_dev[ind_p + (Nzz-idx)*Nxx];
		cf=0.0;
		int val_mn=tex1D(tex_mn,idx);
		for(h=1;h<=val_mn;h++){
			pi=pi+g_dev[h]*(p_dev[ind_p + ((Nzz-idx)-h)*Nxx]+p_dev[ind_p + ((Nzz-idx)+h)*Nxx]);
			cf=cf+d_dev[h]*Oz_dev[idy + (idx-h)*Nxx]*Tn_dev[(idx-h)];	
			
		}
		for(k=h;k<=no;k++)
			pi=pi+g_dev[k]*p_dev[ind_p + ((Nzz-idx)-k)*Nxx];
		int val_ma=tex1D(tex_ma,idx);
		for(h=1;h<=val_ma;h++)
			cf=cf-d_dev[h]*Oz_dev[idy + (idx+h)*Nxx]*Tn_dev[(idx+h)];
		

		aa=pi - cf*dh - Tk_dev[idx]*uz_dev[idy + idx*Nxx];
		uz_dev[idy + idx*Nxx]=tx2*aa+uz_dev[idy + idx*Nxx];

	}
}

__global__ void ParaOx(float *p_dev,float *Ox_dev,float *Tt_dev,
											   //int *mn_dev,int *ma_dev,//float *g_dev,float *d_dev,
											   float tx,float dh,
											   int N,int no,int Na,int Nzz,int Nxx,int ms,int k1){


	int idx=blockIdx.x * blockDim.x + threadIdx.x;
	int idy=blockIdx.y * blockDim.y + threadIdx.y;

	int h,k;

	float da,cf,ak,aa;
	if (idx>0 && idx<Na && idy>0 && idy<Nzz){
		da=0.0;     cf=0.0;
		int ind_p=k1*Nzz*Nxx + idy*Nxx;
		int ind_Ox1=idx + idy*Na;
		int ind_Ox2=Nzz*Na + idx + idy*Na;

		int val_mn=tex1D(tex_mn,idx);
		for(h=1;h<=val_mn;h++){
			
			da=da+d_dev[h]*(p_dev[ind_p + (idx+h)]-p_dev[ind_p + (idx-h)]);
			cf=cf+d_dev[h]*(p_dev[ind_p + ((Nxx-idx)+h)]-p_dev[ind_p + ((Nxx-idx)-h)]);
		}
		for(k=h;k<=no;k++){
			da=da+d_dev[k]*p_dev[ind_p + (idx+k)];
			cf=cf-d_dev[k]*p_dev[ind_p + ((Nxx-idx)-k)];
		}
		aa=tx*da + Tt_dev[idx]*Ox_dev[ind_Ox1];
		Ox_dev[ind_Ox1]=aa;
        ak=tx*cf + Tt_dev[idx]*Ox_dev[ind_Ox2];
		Ox_dev[ind_Ox2]=ak;

	}
}

__global__ void ParaOz(float *p_dev,float *Oz_dev,float *Tt_dev,
											   //int *mn_dev,int *ma_dev,//float *g_dev,float *d_dev,
											   float tx,float dh,
											   int M,int no,int Na,int Nzz,int Nxx,int ms,int k1){


	int idx=blockIdx.x * blockDim.x + threadIdx.x;
	int idy=blockIdx.y * blockDim.y + threadIdx.y;

	int h,k;

	float cf,aa;

	if (idx>0 && idx<Na && idy>0 && idy<Nxx){
		int ind_p=k1*Nzz*Nxx + idy;
		int ind_Oz=idy + idx*Nxx;
		cf=0.0;
		int val_mn=tex1D(tex_mn,idx);
		for(h=1;h<=val_mn;h++){
			cf=cf+d_dev[h]*(p_dev[ind_p + (Nzz-idx+h)*Nxx] - p_dev[ind_p + (Nzz-idx-h)*Nxx]);
		}
		for(k=h;k<=no;k++){
			cf=cf-d_dev[k]*p_dev[ind_p + (Nzz-idx-k)*Nxx];
		}
		aa=tx*cf + Tt_dev[idx]*Oz_dev[ind_Oz];
		Oz_dev[ind_Oz]=aa;
	}
}

__global__ void EcDominioReal(float *p_dev,float *Oz_dev,float *Tt_dev,//float *vel_dev,
											   //int *mn_dev,int *ma_dev,//float *g_dev,float *d_dev,
											   float tx,float dh,
											   int M,int N,int no,int Na,int Nzz,int Nxx,int ms,int k1,int k2){


	int idx=blockIdx.x * blockDim.x + threadIdx.x;
	int idy=blockIdx.y * blockDim.y + threadIdx.y;

	int h,k;
	float dl,da;

	if (idx>0 && idx<=Nzz-Na && idy>=0 && idy<=N-M){
		idy=idy+M;	
		int ind_p1=k2*Nzz*Nxx;
		int ind_p2=k1*Nzz*Nxx;
		int ind_idxNxx=idx*Nxx;
		int ind_p3=ind_p2 + idy + ind_idxNxx;
		int ind_p4=idy + ind_idxNxx;

		dl=2.0f*g_dev[ms]*p_dev[ind_p1 + idx*Nxx + (idy)];
		int val_mn=tex1D(tex_mn,idx);
		//for(k=1;k<=mn_dev[idx];k++){
		for(k=1;k<=val_mn;k++){
			da=p_dev[ind_p1 + (idy-k) + idx*Nxx]   + p_dev[ind_p1 + (idy+k) + idx*Nxx] +
			   p_dev[ind_p1 + (idy) + (idx-k)*Nxx] + p_dev[ind_p1 + idy + (idx+k)*Nxx];
			dl=dl+g_dev[k]*da;
		}
		for(h=k;h<=no;h++){
			da=p_dev[ind_p1 + (idy-h) + idx*Nxx] + p_dev[ind_p1 + (idy+h) + idx*Nxx] + 
			   p_dev[ind_p1 + idy + (idx+h)*Nxx];
			dl=dl+g_dev[h]*da;
		}
		/*
		p_dev[ind_p2 + ((idy)) + idx*Nxx] = 2.0f*p_dev[ind_p1 + ((idy)) + idx*Nxx] -
			                                p_dev[ind_p2 + ((idy)) + idx*Nxx] + vel_dev[idx*Nxx + (idy)]*dl;
		*/
		//p_dev[ind_p3] = 2.0f*p_dev[ind_p1 + ind_p4]-p_dev[ind_p3] + vel_dev[ind_p4]*dl;

		float vel=tex2D(tex_vel,(float)idy,(float)idx);
		p_dev[ind_p3] =2.0f*p_dev[ind_p1 + ind_p4]-p_dev[ind_p3] + vel*dl;

		/*
		float val1=2*p_dev[ind_p1 + ind_p4];
		float val2=val1-p_dev[ind_p3];
		//float val3=vel_dev[ind_p4]*dl;
		p_dev[ind_p3] = val2;*/
		
	}
}
__global__ void EcDominioReal2(float *p_dev,float *Oz_dev,float *Tt_dev,//float *vel_dev,
											   //int *mn_dev,int *ma_dev,//float *g_dev,float *d_dev,
											   float tx,float dh,
											   int M,int N,int no,int Na,int Nzz,int Nxx,int ms,int k1,int k2){


	int idx=blockIdx.x * blockDim.x + threadIdx.x;
	int idy=blockIdx.y * blockDim.y + threadIdx.y;

	int h,k;
	float dl,da;
	float val_p1,val_p2;
	float val_p2_1;	float val_p2_2;	float val_p2_3;	float val_p2_4;	
	float val_p1_1;	float val_p1_2;	float val_p1_3;	float val_p1_4;

	if (idx>0 && idx<=Nzz-Na && idy>=0 && idy<=N-M){
		idy=idy+M;	
		//int ind_p1=k2*Nzz*Nxx;
		int ind_p2=k1*Nzz*Nxx;
		int ind_idxNxx=idx*Nxx;
		int ind_p3=ind_p2 + idy + ind_idxNxx;
		//int ind_p4=idy + ind_idxNxx;

		float vel=tex2D(tex_vel,idy,idx);
		if(k1==0){
			val_p2=tex2D(tex_p2,idy,idx);
			//dl=2.0f*g_dev[ms]*p_dev[ind_p1 + idx*Nxx + (idy)];
			dl=2.0f*g_dev[ms]*val_p2;
			int val_mn=tex1D(tex_mn,idx);
			//for(k=1;k<=mn_dev[idx];k++){
			for(k=1;k<=val_mn;k++){

				val_p2_1=tex2D(tex_p2,(idy-k),idx);
				val_p2_2=tex2D(tex_p2,(idy+k),idx);
				val_p2_3=tex2D(tex_p2,idy,(idx-k));
				val_p2_4=tex2D(tex_p2,idy,(idx+k));

				da=val_p2_1 + val_p2_2 + val_p2_3 + val_p2_4;
				
				//da=p_dev[ind_p1 + (idy-k) + idx*Nxx]   + p_dev[ind_p1 + (idy+k) + idx*Nxx] +
				  // p_dev[ind_p1 + (idy) + (idx-k)*Nxx] + p_dev[ind_p1 + idy + (idx+k)*Nxx];
				dl=dl+g_dev[k]*da;
			}
			for(h=k;h<=no;h++){
				val_p2_1=tex2D(tex_p2,(idy-h),idx);
				val_p2_2=tex2D(tex_p2,(idy+h),idx);
				val_p2_3=tex2D(tex_p2,idy,(idx+h));
				
				da=val_p2_1 + val_p2_2 + val_p2_3;
				//da=p_dev[ind_p1 + (idy-h) + idx*Nxx] + p_dev[ind_p1 + (idy+h) + idx*Nxx] + 
				  // p_dev[ind_p1 + idy + (idx+h)*Nxx];
				dl=dl+g_dev[h]*da;
			}
			
			val_p1=tex2D(tex_p1,idy,idx);
			//p_dev[ind_p3] =2.0f*p_dev[ind_p1 + ind_p4]-p_dev[ind_p3] + vel*dl;
			p_dev[ind_p3] =2.0f*val_p2 - val_p1 + vel*dl;
		}
		else{
			val_p1=tex2D(tex_p1,idy,idx);
			//dl=2.0f*g_dev[ms]*p_dev[ind_p1 + idx*Nxx + (idy)];
			dl=2.0f*g_dev[ms]*val_p1;
			int val_mn=tex1D(tex_mn,idx);
			//for(k=1;k<=mn_dev[idx];k++){
			for(k=1;k<=val_mn;k++){

				val_p1_1=tex2D(tex_p1,(idy-k),idx);
				val_p1_2=tex2D(tex_p1,(idy+k),idx);
				val_p1_3=tex2D(tex_p1,idy,(idx-k));
				val_p1_4=tex2D(tex_p1,idy,(idx+k));

				da=val_p1_1 + val_p1_2 + val_p1_3 + val_p1_4;
				
				//da=p_dev[ind_p1 + (idy-k) + idx*Nxx]   + p_dev[ind_p1 + (idy+k) + idx*Nxx] +
				  // p_dev[ind_p1 + (idy) + (idx-k)*Nxx] + p_dev[ind_p1 + idy + (idx+k)*Nxx];
				dl=dl+g_dev[k]*da;
			}
			for(h=k;h<=no;h++){
				val_p1_1=tex2D(tex_p1,(idy-h),idx);
				val_p1_2=tex2D(tex_p1,(idy+h),idx);
				val_p1_3=tex2D(tex_p1,idy,(idx+h));
				
				da=val_p1_1 + val_p1_2 + val_p1_3;
				//da=p_dev[ind_p1 + (idy-h) + idx*Nxx] + p_dev[ind_p1 + (idy+h) + idx*Nxx] + 
				  // p_dev[ind_p1 + idy + (idx+h)*Nxx];
				dl=dl+g_dev[h]*da;
			}
			
			val_p2=tex2D(tex_p2,idy,idx);
			//p_dev[ind_p3] =2.0f*p_dev[ind_p1 + ind_p4]-p_dev[ind_p3] + vel*dl;
			p_dev[ind_p3] =2.0f*val_p1 - val_p2 + vel*dl;
		}
	}
}

__global__ void Para_uxp(float *p_dev,float *Ox_dev,float *ux_dev,float *Tn_dev,float *Tk_dev,float *vel_dev,
						  //int *mn_dev,int *ma_dev,//float *g_dev,float *d_dev,
						  float tx2,float dh,
						  int N,int no,int Na,int Nzz,int Nxx,int ms,int k1,int k2){

	int idx=blockIdx.x * blockDim.x + threadIdx.x;
	int idy=blockIdx.y * blockDim.y + threadIdx.y;

	float dl,da,pi,cf,aa;
	int h,k;

	if (idx>0 && idx<Na && idy>0 && idy<Nzz){
			int ind_p=k2*Nzz*Nxx+idy*Nxx;
			int ind_Ox=Nzz*Na+ idy*Na;
			int ind_ux=idx + idy*Na;
			int ind_ux2=Nzz*Na + idx + idy*Na;
			int ind_p2=k1*Nzz*Nxx + idx + idy*Nxx;
			int ind_p3=k1*Nzz*Nxx+idy*Nxx;

			dl=g_dev[ms]*p_dev[ind_p + idx];       
			pi=g_dev[ms]*p_dev[ind_p + (Nxx-idx)];
			da=0.0;                   cf=0.0;
			
			int val_mn=tex1D(tex_mn,idx);
			//for(h=1;h<=mn_dev[idx];h++){
			for(h=1;h<=val_mn;h++){
				dl=dl+g_dev[h]*(p_dev[ind_p + (idx-h)]+p_dev[ind_p + (idx+h)]);
				da=da-d_dev[h]*Ox_dev[(idx-h) + idy*Na]*Tn_dev[(idx-h)];
				pi=pi+g_dev[h]*(p_dev[ind_p + ((Nxx-idx)-h)]+p_dev[ind_p + ((Nxx-idx)+h)]);
				cf=cf+d_dev[h]*Ox_dev[ind_Ox + (idx-h)]*Tn_dev[(idx-h)];	
			}
			for(k=h;k<=no;k++){
				dl=dl+g_dev[k]*p_dev[ind_p + (idx+k)];
				pi=pi+g_dev[k]*p_dev[ind_p + ((Nxx-idx)-k)];
			}
			int val_ma=tex1D(tex_ma,idx);
			for(h=1;h<=val_ma;h++){
				da=da+d_dev[h]*Ox_dev[(idx+h) + idy*Na]*Tn_dev[(idx+h)];
				cf=cf-d_dev[h]*Ox_dev[ind_Ox + (idx+h)]*Tn_dev[(idx+h)];
			}

			/*float vel=tex2D(tex_vel,(float)idx+0.5,(float)idy+0.5);
			float vel2=tex2D(tex_vel,(float)(Nxx-idx)+0.5,(float)idy+0.5);
*/
			aa=dl - da*dh - Tk_dev[idx]*ux_dev[ind_ux];
			ux_dev[ind_ux]=tx2*aa+ux_dev[ind_ux];
			p_dev[ind_p2]=vel_dev[idy*Nxx + idx]*aa+ 2.0f*p_dev[ind_p + idx] - p_dev[ind_p2];
			//p_dev[ind_p2]=vel*aa+ 2.0f*p_dev[ind_p + idx] - p_dev[ind_p2];
			aa=pi - cf*dh - Tk_dev[idx]*ux_dev[ind_ux2];
			ux_dev[ind_ux2]=tx2*aa+ux_dev[ind_ux2];
			p_dev[ind_p3 + (Nxx-idx)]=vel_dev[idy*Nxx + (Nxx-idx)]*aa + 
			//p_dev[ind_p3 + (Nxx-idx)]=vel2*aa + 
				                          2.0f*p_dev[ind_p + (Nxx-idx)] - p_dev[ind_p3 + (Nxx-idx)];

	}
}
__global__ void Para_uxp2(float *p_dev,float *Ox_dev,float *ux_dev,float *Tn_dev,float *Tk_dev,//float *vel_dev,
						  //int *mn_dev,int *ma_dev,//float *g_dev,float *d_dev,
						  float tx2,float dh,
						  int N,int no,int Na,int Nzz,int Nxx,int ms,int k1,int k2){

	int idx=blockIdx.x * blockDim.x + threadIdx.x;
	int idy=blockIdx.y * blockDim.y + threadIdx.y;

	float dl,da,pi,cf,aa;
	int h,k;

	float val_p1,val_p2;
	float val_p2_1;	float val_p2_2;	float val_p2_3;	float val_p2_4;	
	float val_p1_1;	float val_p1_2;	float val_p1_3;	float val_p1_4;

	if (idx>0 && idx<Na && idy>0 && idy<Nzz){
			//int ind_p=k2*Nzz*Nxx+idy*Nxx;
			int ind_Ox=Nzz*Na+ idy*Na;
			int ind_ux=idx + idy*Na;
			int ind_ux2=Nzz*Na + idx + idy*Na;
			int ind_p2=k1*Nzz*Nxx + idx + idy*Nxx;
			int ind_p3=k1*Nzz*Nxx+idy*Nxx;
			
			
			float vel=tex2D(tex_vel,idx,idy);
			float vel2=tex2D(tex_vel,(Nxx-idx),idy);

			if(k1==0){

				val_p2=tex2D(tex_p2,idx,idy);
				val_p2_1=tex2D(tex_p2,(Nxx-idx),idy);

				//dl=g_dev[ms]*p_dev[ind_p + idx];       
				dl=g_dev[ms]*val_p2;       
				//pi=g_dev[ms]*p_dev[ind_p + (Nxx-idx)];
				pi=g_dev[ms]*val_p2_1;
				da=0.0;                   cf=0.0;

				int val_mn=tex1D(tex_mn,idx);
				//for(h=1;h<=mn_dev[idx];h++){
				for(h=1;h<=val_mn;h++){
					val_p2_1=tex2D(tex_p2,(idx-h),idy);	
					val_p2_2=tex2D(tex_p2,(idx+h),idy);	
					val_p2_3=tex2D(tex_p2,((Nxx-idx)-h),idy);	
					val_p2_4=tex2D(tex_p2,((Nxx-idx)+h),idy);	

					dl=dl+g_dev[h]*(val_p2_1+val_p2_2);
					da=da-d_dev[h]*Ox_dev[(idx-h) + idy*Na]*Tn_dev[(idx-h)];
					pi=pi+g_dev[h]*(val_p2_3+val_p2_4);
					cf=cf+d_dev[h]*Ox_dev[ind_Ox + (idx-h)]*Tn_dev[(idx-h)];	
				}
				for(k=h;k<=no;k++){
					val_p2_1=tex2D(tex_p2,(idx+k),idy);	
					val_p2_2=tex2D(tex_p2,((Nxx-idx)-k),idy);	
					dl=dl+g_dev[k]*val_p2_1;
					pi=pi+g_dev[k]*val_p2_2;
				}
				int val_ma=tex1D(tex_ma,idx);
				for(h=1;h<=val_ma;h++){
					da=da+d_dev[h]*Ox_dev[(idx+h) + idy*Na]*Tn_dev[(idx+h)];
					cf=cf-d_dev[h]*Ox_dev[ind_Ox + (idx+h)]*Tn_dev[(idx+h)];
				}
				aa=dl - da*dh - Tk_dev[idx]*ux_dev[ind_ux];
				ux_dev[ind_ux]=tx2*aa+ux_dev[ind_ux];
				
				val_p1=tex2D(tex_p1,idx,idy);
				p_dev[ind_p2]=vel*aa+ 2.0f*val_p2 - val_p1;

				aa=pi - cf*dh - Tk_dev[idx]*ux_dev[ind_ux2];
				ux_dev[ind_ux2]=tx2*aa+ux_dev[ind_ux2];
				
				val_p2_1=tex2D(tex_p2,(Nxx-idx),idy);
				val_p1_1=tex2D(tex_p1,(Nxx-idx),idy);

				p_dev[ind_p3 + (Nxx-idx)]=vel2*aa + 2.0f*val_p2_1 - val_p1_1;	
			}
			else{
				val_p1=tex2D(tex_p1,idx,idy);
				val_p1_1=tex2D(tex_p1,(Nxx-idx),idy);

				//dl=g_dev[ms]*p_dev[ind_p + idx];       
				dl=g_dev[ms]*val_p1;       
				//pi=g_dev[ms]*p_dev[ind_p + (Nxx-idx)];
				pi=g_dev[ms]*val_p1_1;
				da=0.0;                   cf=0.0;

				int val_mn=tex1D(tex_mn,idx);
				//for(h=1;h<=mn_dev[idx];h++){
				for(h=1;h<=val_mn;h++){
					val_p1_1=tex2D(tex_p1,(idx-h),idy);	
					val_p1_2=tex2D(tex_p1,(idx+h),idy);	
					val_p1_3=tex2D(tex_p1,((Nxx-idx)-h),idy);	
					val_p1_4=tex2D(tex_p1,((Nxx-idx)+h),idy);	

					dl=dl+g_dev[h]*(val_p1_1+val_p1_2);
					da=da-d_dev[h]*Ox_dev[(idx-h) + idy*Na]*Tn_dev[(idx-h)];
					pi=pi+g_dev[h]*(val_p1_3+val_p1_4);
					cf=cf+d_dev[h]*Ox_dev[ind_Ox + (idx-h)]*Tn_dev[(idx-h)];	
				}
				for(k=h;k<=no;k++){
					val_p1_1=tex2D(tex_p1,(idx+k),idy);	
					val_p1_2=tex2D(tex_p1,((Nxx-idx)-k),idy);	
					dl=dl+g_dev[k]*val_p1_1;
					pi=pi+g_dev[k]*val_p1_2;
				}
				int val_ma=tex1D(tex_ma,idx);
				for(h=1;h<=val_ma;h++){
					da=da+d_dev[h]*Ox_dev[(idx+h) + idy*Na]*Tn_dev[(idx+h)];
					cf=cf-d_dev[h]*Ox_dev[ind_Ox + (idx+h)]*Tn_dev[(idx+h)];
				}
				aa=dl - da*dh - Tk_dev[idx]*ux_dev[ind_ux];
				ux_dev[ind_ux]=tx2*aa+ux_dev[ind_ux];
				
				val_p2=tex2D(tex_p2,idx,idy);
				p_dev[ind_p2]=vel*aa+ 2.0f*val_p1 - val_p2;

				aa=pi - cf*dh - Tk_dev[idx]*ux_dev[ind_ux2];
				ux_dev[ind_ux2]=tx2*aa+ux_dev[ind_ux2];
				
				val_p1_1=tex2D(tex_p1,(Nxx-idx),idy);
				val_p2_1=tex2D(tex_p2,(Nxx-idx),idy);

				p_dev[ind_p3 + (Nxx-idx)]=vel2*aa + 2.0f*val_p1_1 - val_p2_1;	
			}
	}
}


__global__ void Para_uzp(float *p_dev,float *Oz_dev,float *uz_dev,float *Tn_dev,float *Tk_dev,float *vel_dev,
						 //int *mn_dev,int *ma_dev,//float *g_dev,float *d_dev,
						 float tx2,float dh,
						 int M,int no,int Na,int Nzz,int Nxx,int ms,int k1,int k2){


	int idx=blockIdx.x * blockDim.x + threadIdx.x;
	int idy=blockIdx.y * blockDim.y + threadIdx.y;

	int h,k;

	float pi,cf,aa;

	if (idx>0 && idx<Na && idy>0 && idy<Na){

		int ind_p1=k1*Nzz*Nxx + idy;
		int ind_p2=k2*Nzz*Nxx + idy;
		int ind_Oz=idy;

		pi=g_dev[ms]*p_dev[ind_p2 + (Nzz-idx)*Nxx];
		cf=0.0;
		int val_mn=tex1D(tex_mn,idx);
		for(h=1;h<=val_mn;h++){
		//for(h=1;h<=mn_dev[idx];h++){
			pi=pi+g_dev[h]*(p_dev[ind_p2 + ((Nzz-idx)-h)*Nxx]+p_dev[ind_p2 + ((Nzz-idx)+h)*Nxx]);
			cf=cf+d_dev[h]*Oz_dev[ind_Oz + (idx-h)*Nxx]*Tn_dev[(idx-h)];

		}
		for(k=h;k<=no;k++){
			pi=pi+g_dev[k]*p_dev[ind_p2 + ((Nzz-idx)-k)*Nxx];
		}
		int val_ma=tex1D(tex_ma,idx);
		for(h=1;h<=val_ma;h++){
			cf=cf-d_dev[h]*Oz_dev[ind_Oz + (idx+h)*Nxx]*Tn_dev[(idx+h)];
		}
		aa=pi - cf*dh - Tk_dev[idx]*uz_dev[ind_Oz + idx*Nxx];
		uz_dev[ind_Oz + idx*Nxx]=tx2*aa+uz_dev[ind_Oz + idx*Nxx];

		//float vel=tex2D(tex_vel,(float)idy+0.5,(float)(Nzz-idx)+0.5);
		p_dev[ind_p1 + (Nzz-idx)*Nxx]=p_dev[ind_p1 + (Nzz-idx)*Nxx]+vel_dev[(Nzz-idx)*Nxx + idy]*aa;	
		//p_dev[ind_p1 + (Nzz-idx)*Nxx]=p_dev[ind_p1 + (Nzz-idx)*Nxx]+vel*aa;	
	}
}
__global__ void Para_uzp2(float *p_dev,float *Oz_dev,float *uz_dev,float *Tn_dev,float *Tk_dev,//float *vel_dev,
						 //int *mn_dev,int *ma_dev,//float *g_dev,float *d_dev,
						 float tx2,float dh,
						 int M,int no,int Na,int Nzz,int Nxx,int ms,int k1,int k2){


	int idx=blockIdx.x * blockDim.x + threadIdx.x;
	int idy=blockIdx.y * blockDim.y + threadIdx.y;

	int h,k;

	float pi,cf,aa;

	float val_p1,val_p2;
	float val_p2_1;	float val_p2_2;	
	float val_p1_1;	float val_p1_2;	
	if (idx>0 && idx<Na && idy>0 && idy<Na){
		int ind_p1=k1*Nzz*Nxx + idy;
		//int ind_p2=k2*Nzz*Nxx + idy;
		int ind_Oz=idy;
		float vel=tex2D(tex_vel,idy,(Nzz-idx));

		if(k1==0){
			val_p2=tex2D(tex_p2,idy,(Nzz-idx));
			//pi=g_dev[ms]*p_dev[ind_p2 + (Nzz-idx)*Nxx];
			pi=g_dev[ms]*val_p2;
			cf=0.0;
			int val_mn=tex1D(tex_mn,idx);
			for(h=1;h<=val_mn;h++){
				val_p2_1=tex2D(tex_p2,idy,((Nzz-idx)-h));
				val_p2_2=tex2D(tex_p2,idy,((Nzz-idx)+h));

				pi=pi+g_dev[h]*(val_p2_1+val_p2_2);
				cf=cf+d_dev[h]*Oz_dev[ind_Oz + (idx-h)*Nxx]*Tn_dev[(idx-h)];

			}
			for(k=h;k<=no;k++){
				val_p2_1=tex2D(tex_p2,idy,((Nzz-idx)-k));
				pi=pi+g_dev[k]*val_p2_1;
			}
			int val_ma=tex1D(tex_ma,idx);
			for(h=1;h<=val_ma;h++){
				cf=cf-d_dev[h]*Oz_dev[ind_Oz + (idx+h)*Nxx]*Tn_dev[(idx+h)];
			}
			aa=pi - cf*dh - Tk_dev[idx]*uz_dev[ind_Oz + idx*Nxx];
			uz_dev[ind_Oz + idx*Nxx]=tx2*aa+uz_dev[ind_Oz + idx*Nxx];

			val_p1=tex2D(tex_p1,idy,(Nzz-idx));
			p_dev[ind_p1 + (Nzz-idx)*Nxx]=val_p1+vel*aa;	
		
		}
		else{
			val_p1=tex2D(tex_p1,idy,(Nzz-idx));
			//pi=g_dev[ms]*p_dev[ind_p2 + (Nzz-idx)*Nxx];
			pi=g_dev[ms]*val_p1;
			cf=0.0;
			int val_mn=tex1D(tex_mn,idx);
			for(h=1;h<=val_mn;h++){
				val_p1_1=tex2D(tex_p1,idy,((Nzz-idx)-h));
				val_p1_2=tex2D(tex_p1,idy,((Nzz-idx)+h));

				pi=pi+g_dev[h]*(val_p1_1+val_p1_2);
				cf=cf+d_dev[h]*Oz_dev[ind_Oz + (idx-h)*Nxx]*Tn_dev[(idx-h)];

			}
			for(k=h;k<=no;k++){
				val_p1_1=tex2D(tex_p1,idy,((Nzz-idx)-k));
				pi=pi+g_dev[k]*val_p1_1;
			}
			int val_ma=tex1D(tex_ma,idx);
			for(h=1;h<=val_ma;h++){
				cf=cf-d_dev[h]*Oz_dev[ind_Oz + (idx+h)*Nxx]*Tn_dev[(idx+h)];
			}
			aa=pi - cf*dh - Tk_dev[idx]*uz_dev[ind_Oz + idx*Nxx];
			uz_dev[ind_Oz + idx*Nxx]=tx2*aa+uz_dev[ind_Oz + idx*Nxx];

			val_p2=tex2D(tex_p2,idy,(Nzz-idx));
			p_dev[ind_p1 + (Nzz-idx)*Nxx]=val_p2+vel*aa;	
		
		}
	}
}
__global__ void Enmedio_uz(float *p_dev,float *Oz_dev,float *uz_dev,float *Tn_dev,float *Tk_dev,float *vel_dev,
						// int *mn_dev,int *ma_dev,//float *g_dev,float *d_dev,
						 float tx2,float dh,
						 int M,int no,int Na,int Nzz,int Nxx,int ms,int k1,int k2, int lim2){


	int idx=blockIdx.x * blockDim.x + threadIdx.x;
	int idy=blockIdx.y * blockDim.y + threadIdx.y;

	int h,k;

	float pi,cf,aa,dl,ak;

	if (idx>0 && idx<Na && idy>=0 && idy<lim2){//lim2=(Nxx-1)-2*(Na-1)
		idy=idy+Na;

		int ind_p1=k1*Nzz*Nxx + idy;
		int ind_p2=k2*Nzz*Nxx + idy;
		int ind_p3=k2*Nzz*Nxx;
		int ind_Oz=idy;

		pi=g_dev[ms]*p_dev[ind_p2 + (Nzz-idx)*Nxx];
		dl=pi;
		cf=0.0;
		int val_mn=tex1D(tex_mn,idx);
		for(h=1;h<=val_mn;h++){
			aa=p_dev[ind_p2 + ((Nzz-idx)-h)*Nxx] + p_dev[ind_p2 + ((Nzz-idx)+h)*Nxx];
			ak=p_dev[ind_p3 + (idy+h) + (Nzz-idx)*Nxx] + p_dev[ind_p3 + (idy-h) + (Nzz-idx)*Nxx];
			pi=pi+g_dev[h]*aa;
			dl=dl+g_dev[h]*ak;
			cf=cf+d_dev[h]*Oz_dev[ind_Oz + (idx-h)*Nxx]*Tn_dev[(idx-h)];	
		}
		for(k=h;k<=no;k++){
			pi=pi+g_dev[k]*(p_dev[ind_p2 + ((Nzz-idx)-k)*Nxx]);
			dl=dl+g_dev[k]*(p_dev[ind_p3 + (idy+k) + (Nzz-idx)*Nxx] + p_dev[ind_p3 + (idy-k) + (Nzz-idx)*Nxx]);
		}
		int val_ma=tex1D(tex_ma,idx);
		for(h=1;h<=val_ma;h++){
			cf=cf-d_dev[h]*Oz_dev[ind_Oz + (idx+h)*Nxx]*Tn_dev[(idx+h)];
		}
		aa=pi - cf*dh - Tk_dev[idx]*uz_dev[ind_Oz + idx*Nxx];
		ak=aa+dl;
		uz_dev[ind_Oz + idx*Nxx]=tx2*aa+uz_dev[ind_Oz + idx*Nxx];

		//float vel=tex2D(tex_vel,(float)idy+0.5,(float)(Nzz-idx)+0.5);
		
		p_dev[ind_p1 + (Nzz-idx)*Nxx]=vel_dev[(Nzz-idx)*Nxx + idy]*ak + 2.0f*p_dev[ind_p2 + (Nzz-idx)*Nxx] - 
		//p_dev[ind_p1 + (Nzz-idx)*Nxx]=vel*ak + 2.0f*p_dev[ind_p2 + (Nzz-idx)*Nxx] - 
			                  p_dev[ind_p1 + (Nzz-idx)*Nxx];
		
	}
}

__global__ void Enmedio_uz2(float *p_dev,float *Oz_dev,float *uz_dev,float *Tn_dev,float *Tk_dev,
						 //int *mn_dev,int *ma_dev,//float *g_dev,float *d_dev,
						 float tx2,float dh,
						 int M,int no,int Na,int Nzz,int Nxx,int ms,int k1,int k2, int lim2){


	int idx=blockIdx.x * blockDim.x + threadIdx.x;
	int idy=blockIdx.y * blockDim.y + threadIdx.y;

	int h,k;

	float pi,cf,aa,dl,ak;
	float val_p1,val_p2;
	float val_p2_1;	float val_p2_2;	float val_p2_3;	float val_p2_4;	
	float val_p1_1;	float val_p1_2;	float val_p1_3;	float val_p1_4;

	if (idx>0 && idx<Na && idy>=0 && idy<lim2){//lim2=(Nxx-1)-2*(Na-1)
		idy=idy+Na;

		int ind_p1=k1*Nzz*Nxx + idy;
		//int ind_p2=k2*Nzz*Nxx + idy;
		//int ind_p3=k2*Nzz*Nxx;
		int ind_Oz=idy;
		float vel=tex2D(tex_vel,idy,(Nzz-idx));

		if(k1==0){
			val_p2=tex2D(tex_p2,idy,(Nzz-idx));

			//pi=g_dev[ms]*p_dev[ind_p2 + (Nzz-idx)*Nxx];
			pi=g_dev[ms]*val_p2;
			dl=pi;
			cf=0.0;
			int val_mn=tex1D(tex_mn,idx);
			for(h=1;h<=val_mn;h++){
				val_p2_1=tex2D(tex_p2,idy,((Nzz-idx)-h));
				val_p2_2=tex2D(tex_p2,idy,((Nzz-idx)+h));
				val_p2_3=tex2D(tex_p2,(idy+h),(Nzz-idx));
				val_p2_4=tex2D(tex_p2,(idy-h),(Nzz-idx));
				aa=val_p2_1 + val_p2_2;
				ak=val_p2_3 + val_p2_4;
				pi=pi+g_dev[h]*aa;
				dl=dl+g_dev[h]*ak;
				cf=cf+d_dev[h]*Oz_dev[ind_Oz + (idx-h)*Nxx]*Tn_dev[(idx-h)];	
			}
			for(k=h;k<=no;k++){
				val_p2_1=tex2D(tex_p2,idy,((Nzz-idx)-k));
				val_p2_2=tex2D(tex_p2,(idy+k),(Nzz-idx));
				val_p2_3=tex2D(tex_p2,(idy-k),(Nzz-idx));
				
				pi=pi+g_dev[k]*(val_p2_1);
				dl=dl+g_dev[k]*(val_p2_2 + val_p2_3);
			}
			int val_ma=tex1D(tex_ma,idx);
			for(h=1;h<=val_ma;h++){
				cf=cf-d_dev[h]*Oz_dev[ind_Oz + (idx+h)*Nxx]*Tn_dev[(idx+h)];
			}
			aa=pi - cf*dh - Tk_dev[idx]*uz_dev[ind_Oz + idx*Nxx];
			ak=aa+dl;
			uz_dev[ind_Oz + idx*Nxx]=tx2*aa+uz_dev[ind_Oz + idx*Nxx];

			val_p1=tex2D(tex_p1,idy,(Nzz-idx));
			p_dev[ind_p1 + (Nzz-idx)*Nxx]=vel*ak + 2.0f*val_p2 - val_p1;
		}
		else{
			val_p1=tex2D(tex_p1,idy,(Nzz-idx));

			//pi=g_dev[ms]*p_dev[ind_p2 + (Nzz-idx)*Nxx];
			pi=g_dev[ms]*val_p1;
			dl=pi;
			cf=0.0;
			int val_mn=tex1D(tex_mn,idx);
			for(h=1;h<=val_mn;h++){
				val_p1_1=tex2D(tex_p1,idy,((Nzz-idx)-h));
				val_p1_2=tex2D(tex_p1,idy,((Nzz-idx)+h));
				val_p1_3=tex2D(tex_p1,(idy+h),(Nzz-idx));
				val_p1_4=tex2D(tex_p1,(idy-h),(Nzz-idx));
				aa=val_p1_1 + val_p1_2;
				ak=val_p1_3 + val_p1_4;
				pi=pi+g_dev[h]*aa;
				dl=dl+g_dev[h]*ak;
				cf=cf+d_dev[h]*Oz_dev[ind_Oz + (idx-h)*Nxx]*Tn_dev[(idx-h)];	
			}
			for(k=h;k<=no;k++){
				val_p1_1=tex2D(tex_p1,idy,((Nzz-idx)-k));
				val_p1_2=tex2D(tex_p1,(idy+k),(Nzz-idx));
				val_p1_3=tex2D(tex_p1,(idy-k),(Nzz-idx));
				
				pi=pi+g_dev[k]*(val_p1_1);
				dl=dl+g_dev[k]*(val_p1_2 + val_p1_3);
			}
			int val_ma=tex1D(tex_ma,idx);
			for(h=1;h<=val_ma;h++){
				cf=cf-d_dev[h]*Oz_dev[ind_Oz + (idx+h)*Nxx]*Tn_dev[(idx+h)];
			}
			aa=pi - cf*dh - Tk_dev[idx]*uz_dev[ind_Oz + idx*Nxx];
			ak=aa+dl;
			uz_dev[ind_Oz + idx*Nxx]=tx2*aa+uz_dev[ind_Oz + idx*Nxx];

			val_p2=tex2D(tex_p2,idy,(Nzz-idx));
			p_dev[ind_p1 + (Nzz-idx)*Nxx]=vel*ak + 2.0f*val_p1 - val_p2;
		
		}
	}
}

__global__ void Final_uzp(float *p_dev,float *Oz_dev,float *uz_dev,float *Tn_dev,float *Tk_dev,float *vel_dev,
						 //int *mn_dev,int *ma_dev,//float *g_dev,float *d_dev,
						 float tx2,float dh,
						 int M,int no,int Na,int Nzz,int Nxx,int ms,int k1,int k2){


	int idx=blockIdx.x * blockDim.x + threadIdx.x;
	int idy=blockIdx.y * blockDim.y + threadIdx.y;

	int h,k;

	float pi,cf,aa;

	if (idx>0 && idx<Na && idy>=0 && idy<Na){
		idy=idy+(Nxx-Na+1);

		int ind_p1=k1*Nzz*Nxx + idy;
		int ind_p2=k2*Nzz*Nxx + idy;
		int ind_Oz=idy;

		pi=g_dev[ms]*p_dev[ind_p2 + (Nzz-idx)*Nxx];
		cf=0.0;
		int val_mn=tex1D(tex_mn,idx);
		for(h=1;h<=val_mn;h++){
			pi=pi+g_dev[h]*(p_dev[ind_p2 + ((Nzz-idx)-h)*Nxx]+p_dev[ind_p2 + ((Nzz-idx)+h)*Nxx]);
			cf=cf+d_dev[h]*Oz_dev[ind_Oz + (idx-h)*Nxx]*Tn_dev[(idx-h)];
		}
		for(k=h;k<=no;k++){
			pi=pi+g_dev[k]*p_dev[ind_p2 + ((Nzz-idx)-k)*Nxx];
		}
		int val_ma=tex1D(tex_ma,idx);
		for(h=1;h<=val_ma;h++){
			cf=cf-d_dev[h]*Oz_dev[ind_Oz + (idx+h)*Nxx]*Tn_dev[(idx+h)];
		}	
		aa=pi - cf*dh - Tk_dev[idx]*uz_dev[ind_Oz + idx*Nxx];
		uz_dev[ind_Oz + idx*Nxx]=tx2*aa+uz_dev[ind_Oz + idx*Nxx];
		
		//float vel=tex2D(tex_vel,(float)idy+0.5,(float)(Nzz-idx)+0.5);
		
		p_dev[ind_p1 + (Nzz-idx)*Nxx]=p_dev[ind_p1 + (Nzz-idx)*Nxx]+vel_dev[(Nzz-idx)*Nxx + idy]*aa;	
		//p_dev[ind_p1 + (Nzz-idx)*Nxx]=p_dev[ind_p1 + (Nzz-idx)*Nxx]+vel*aa;	
		//i*Nxx+j

	}
}

__global__ void ParteSuperior_ux(float *p_dev,float *Ox_dev,float *ux_dev,float *Tn_dev,float *Tk_dev,float *vel_dev,
						  //int *mn_dev,int *ma_dev,//float *g_dev,float *d_dev,
						  float tx2,float dh,
						  int N,int no,int Na,int Nzz,int Nxx,int ms,int k1,int k2){

	int idx=blockIdx.x * blockDim.x + threadIdx.x;
	int idy=blockIdx.y * blockDim.y + threadIdx.y;

	float dl,pi;
	int h,k;


	if (idx>0 && idx<=Nzz-Na && idy>0 && idy<Na){

			int ind_p1=k1*Nzz*Nxx + idy + idx*Nxx;
			int ind_p2=k2*Nzz*Nxx + idx*Nxx;
			int ind_p3=k2*Nzz*Nxx + idy;
			int ind_p4=k2*Nzz*Nxx + (Nxx-idy);
			int ind_p5=k1*Nzz*Nxx + (Nxx-idy) + idx*Nxx;

			dl=g_dev[ms]*p_dev[ind_p2 + idy];
			pi=g_dev[ms]*p_dev[ind_p2 + (Nxx-idy)];//Error1
			int val_mn=tex1D(tex_mn,idx);
			for(h=1;h<=val_mn;h++){
				dl=dl+g_dev[h]*(p_dev[ind_p3 + (idx-h)*Nxx]+p_dev[ind_p3 + (idx+h)*Nxx]);
				pi=pi+g_dev[h]*(p_dev[ind_p4 + (idx-h)*Nxx]+p_dev[ind_p4 + (idx+h)*Nxx]);
			}
			for(k=h;k<=no;k++){
				dl=dl+g_dev[k]*p_dev[ind_p3 + (idx+k)*Nxx];
				pi=pi+g_dev[k]*p_dev[ind_p4 + (idx+k)*Nxx];
			}

			/*float vel=tex2D(tex_vel,(float)idy+0.5,(float)idx+0.5);
			float vel2=tex2D(tex_vel,(float)(Nxx-idy)+0.5,(float)idx+0.5);*/
			p_dev[ind_p1]=p_dev[ind_p1]+vel_dev[idx*Nxx + idy]*dl;
			//p_dev[ind_p1]=p_dev[ind_p1]+vel*dl;
			p_dev[ind_p5]=p_dev[ind_p5]+vel_dev[idx*Nxx + (Nxx-idy)]*pi;
			//p_dev[ind_p5]=p_dev[ind_p5]+vel2*pi;
	}
}

__global__ void Pasar_Valores(float *p_dev2,float *p_dev,int N,int Nxx,int Na,int Nzz,int k1){
	
	int idx=blockIdx.x * blockDim.x + threadIdx.x;

	if(idx<N){
		p_dev2[idx]=p_dev[k1*Nzz*Nxx + Nxx + Na+idx];
		//p_host[(k1-1)*(Nzz+1)*(Nxx+1) + (Nxx+1) + j]
	}
}

__global__ void Pasar_Valores_vel(float *vel_dev,int Nzz,int Nxx){
	int idx=blockIdx.x * blockDim.x + threadIdx.x;
	int idy=blockIdx.y * blockDim.y + threadIdx.y;
	if(idx>=0 && idx<Nzz && idy>=0 && idy<Nxx){
		vel_dev[idx*Nxx+idy]=tex2D(tex_vel,(float)idy,(float)idx);
	}
}

__global__ void Pasar_Valores_vel(int *mn_dev,int Nzz){
	int idx=blockIdx.x * blockDim.x + threadIdx.x;
	if(idx<Nzz){
		mn_dev[idx]=tex1D(tex_mn,idx);
	}
}