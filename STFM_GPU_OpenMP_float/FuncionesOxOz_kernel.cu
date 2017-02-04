

__global__ void ParaOxOz(float *p_dev,float *Ox_dev,float *Oz_dev,float *Tt_dev,
											   //int *mn_dev,int *ma_dev,
											   float tx,float dh,
											   int N,int M,int no,int Na,int Nzz,int Nxx,int ms,int k1){

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
__global__ void ParaOxOz2(float *Ox_dev,float *Oz_dev,float *Tt_dev,
											   //int *mn_dev,int *ma_dev,
											   float tx,float dh,
											   int N,int M,int no,int Na,int Nzz,int Nxx,int ms,int k1){

	int idx=blockIdx.x * blockDim.x + threadIdx.x;
	int idy=blockIdx.y * blockDim.y + threadIdx.y;

	int h,k;
	float da,cf,ak,aa;
	float val_p2_1;	float val_p2_2;	float val_p2_3;	float val_p2_4;	
	float val_p1_1;	float val_p1_2;	float val_p1_3;	float val_p1_4;

	if (idx>0 && idx<Na && idy>0 && idy<Nzz){
		da=0.0;     cf=0.0;
		//int ind_p=k1*Nzz*Nxx + idy*Nxx;
		int ind_Ox1=idx + idy*Na;
		int ind_Ox2=Nzz*Na + idx + idy*Na;

		if(k1==0){
			int val_mn=tex1D(tex_mn,idx);
			for(h=1;h<=val_mn;h++){
				val_p1_1=tex2D(tex_p1,(idx+h),idy);
				val_p1_2=tex2D(tex_p1,(idx-h),idy);
				val_p1_3=tex2D(tex_p1,((Nxx-idx)+h),idy);
				val_p1_4=tex2D(tex_p1,((Nxx-idx)-h),idy);	
				
				/*da=da+d_dev[h]*(p_dev[ind_p + (idx+h)]-p_dev[ind_p + (idx-h)]);
				cf=cf+d_dev[h]*(p_dev[ind_p + ((Nxx-idx)+h)]-p_dev[ind_p + ((Nxx-idx)-h)]);*/

				da=da+d_dev[h]*(val_p1_1 - val_p1_2);
				cf=cf+d_dev[h]*(val_p1_3 - val_p1_4);
			}
			for(k=h;k<=no;k++){
				val_p1_1=tex2D(tex_p1,(idx+k),idy);
				val_p1_2=tex2D(tex_p1,((Nxx-idx)-k),idy);

				/*da=da+d_dev[k]*p_dev[ind_p + (idx+k)];
			cf=cf-d_dev[k]*p_dev[ind_p + ((Nxx-idx)-k)];*/
				
				da=da+d_dev[k]*val_p1_1;
				cf=cf-d_dev[k]*val_p1_2;
			}	
			aa=tx*da + Tt_dev[idx]*Ox_dev[ind_Ox1];
			Ox_dev[ind_Ox1]=aa;
			ak=tx*cf + Tt_dev[idx]*Ox_dev[ind_Ox2];
			Ox_dev[ind_Ox2]=ak;	
		}
		else{
			int val_mn=tex1D(tex_mn,idx);
			for(h=1;h<=val_mn;h++){
				val_p2_1=tex2D(tex_p2,(idx+h),idy);
				val_p2_2=tex2D(tex_p2,(idx-h),idy);
				val_p2_3=tex2D(tex_p2,((Nxx-idx)+h),idy);
				val_p2_4=tex2D(tex_p2,((Nxx-idx)-h),idy);	

				da=da+d_dev[h]*(val_p2_1 - val_p2_2);
				cf=cf+d_dev[h]*(val_p2_3 - val_p2_4);
			}
			for(k=h;k<=no;k++){
				val_p2_1=tex2D(tex_p2,(idx+k),idy);
				val_p2_2=tex2D(tex_p2,((Nxx-idx)-k),idy);
				
				da=da+d_dev[k]*val_p2_1;
				cf=cf-d_dev[k]*val_p2_2;
			}	
			aa=tx*da + Tt_dev[idx]*Ox_dev[ind_Ox1];
			Ox_dev[ind_Ox1]=aa;
			ak=tx*cf + Tt_dev[idx]*Ox_dev[ind_Ox2];
			Ox_dev[ind_Ox2]=ak;	
		}
	}
	if (idx>0 && idx<Na && idy>0 && idy<Nxx){
		//int ind_p=k1*Nzz*Nxx + idy;
		int ind_Oz=idy + idx*Nxx;
		cf=0.0;

		if(k1==0){
			int val_mn=tex1D(tex_mn,idx);
			for(h=1;h<=val_mn;h++){
				val_p1_1=tex2D(tex_p1,idy,(Nzz-idx+h));
				val_p1_2=tex2D(tex_p1,idy,(Nzz-idx-h));

				cf=cf+d_dev[h]*(val_p1_1 - val_p1_2);
			}
			for(k=h;k<=no;k++){
				val_p1_1=tex2D(tex_p1,idy,(Nzz-idx-k));
				cf=cf-d_dev[k]*val_p1_1;
			}
			aa=tx*cf + Tt_dev[idx]*Oz_dev[ind_Oz];
			Oz_dev[ind_Oz]=aa;		
		}
		else{
			int val_mn=tex1D(tex_mn,idx);
			for(h=1;h<=val_mn;h++){
				val_p2_1=tex2D(tex_p2,idy,(Nzz-idx+h));
				val_p2_2=tex2D(tex_p2,idy,(Nzz-idx-h));

				cf=cf+d_dev[h]*(val_p2_1 - val_p2_2);
			}
			for(k=h;k<=no;k++){
				val_p2_1=tex2D(tex_p2,idy,(Nzz-idx-k));
				cf=cf-d_dev[k]*val_p2_1;
			}
			aa=tx*cf + Tt_dev[idx]*Oz_dev[ind_Oz];
			Oz_dev[ind_Oz]=aa;		
		}
	}
}
