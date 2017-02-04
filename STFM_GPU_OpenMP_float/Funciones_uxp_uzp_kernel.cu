
__global__ void Para_uxp_uzp(float *p_dev,float *Ox_dev,float *Oz_dev,float *ux_dev,float *uz_dev,float *Tn_dev,float *Tk_dev,float *vel_dev,
						  int *mn_dev,int *ma_dev,//float *g_dev,float *d_dev,
						  float tx2,float dh,
						  int N,int M,int no,int Na,int Nzz,int Nxx,int ms,int k1,int k2/*, int lim2*/){

	int idx=blockIdx.x * blockDim.x + threadIdx.x;
	int idy=blockIdx.y * blockDim.y + threadIdx.y;

	float dl,da,pi,cf,aa,ak;
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

			for(h=1;h<=mn_dev[idx];h++){
				dl=dl+g_dev[h]*(p_dev[ind_p + (idx-h)]+p_dev[ind_p + (idx+h)]);
				da=da-d_dev[h]*Ox_dev[(idx-h) + idy*Na]*Tn_dev[(idx-h)];
				pi=pi+g_dev[h]*(p_dev[ind_p + ((Nxx-idx)-h)]+p_dev[ind_p + ((Nxx-idx)+h)]);
				cf=cf+d_dev[h]*Ox_dev[ind_Ox + (idx-h)]*Tn_dev[(idx-h)];	
			}
			for(k=h;k<=no;k++){
				dl=dl+g_dev[k]*p_dev[ind_p + (idx+k)];
				pi=pi+g_dev[k]*p_dev[ind_p + ((Nxx-idx)-k)];
			}
			for(h=1;h<=ma_dev[idx];h++){
				da=da+d_dev[h]*Ox_dev[(idx+h) + idy*Na]*Tn_dev[(idx+h)];
				cf=cf-d_dev[h]*Ox_dev[ind_Ox + (idx+h)]*Tn_dev[(idx+h)];
			}
			aa=dl - da*dh - Tk_dev[idx]*ux_dev[ind_ux];
			ux_dev[ind_ux]=tx2*aa+ux_dev[ind_ux];
			p_dev[ind_p2]=vel_dev[idy*Nxx + idx]*aa+ 2.0f*p_dev[ind_p + idx] - p_dev[ind_p2];
			aa=pi - cf*dh - Tk_dev[idx]*ux_dev[ind_ux2];
			ux_dev[ind_ux2]=tx2*aa+ux_dev[ind_ux2];
			p_dev[ind_p3 + (Nxx-idx)]=vel_dev[idy*Nxx + (Nxx-idx)]*aa + 
				                          2.0f*p_dev[ind_p + (Nxx-idx)] - p_dev[ind_p3 + (Nxx-idx)];

	}
	if (idx>0 && idx<Na && idy>0 && idy<Na){
		int ind_p1=k1*Nzz*Nxx + idy;
		int ind_p2=k2*Nzz*Nxx + idy;
		int ind_Oz=idy;

		pi=g_dev[ms]*p_dev[ind_p2 + (Nzz-idx)*Nxx];
		cf=0.0;
		for(h=1;h<=mn_dev[idx];h++){
			pi=pi+g_dev[h]*(p_dev[ind_p2 + ((Nzz-idx)-h)*Nxx]+p_dev[ind_p2 + ((Nzz-idx)+h)*Nxx]);
			cf=cf+d_dev[h]*Oz_dev[ind_Oz + (idx-h)*Nxx]*Tn_dev[(idx-h)];

		}
		for(k=h;k<=no;k++){
			pi=pi+g_dev[k]*p_dev[ind_p2 + ((Nzz-idx)-k)*Nxx];
		}
		for(h=1;h<=ma_dev[idx];h++){
			cf=cf-d_dev[h]*Oz_dev[ind_Oz + (idx+h)*Nxx]*Tn_dev[(idx+h)];
		}
		aa=pi - cf*dh - Tk_dev[idx]*uz_dev[ind_Oz + idx*Nxx];
		uz_dev[ind_Oz + idx*Nxx]=tx2*aa+uz_dev[ind_Oz + idx*Nxx];

		p_dev[ind_p1 + (Nzz-idx)*Nxx]=p_dev[ind_p1 + (Nzz-idx)*Nxx]+vel_dev[(Nzz-idx)*Nxx + idy]*aa;	
	}
	/*if (idx>0 && idx<Na && idy>=0 && idy<lim2){//lim2=(Nxx-1)-2*(Na-1)
		idy=idy+Na;

		int ind_p1=k1*Nzz*Nxx + idy;
		int ind_p2=k2*Nzz*Nxx + idy;
		int ind_p3=k2*Nzz*Nxx;
		int ind_Oz=idy;

		pi=g_dev[ms]*p_dev[ind_p2 + (Nzz-idx)*Nxx];
		dl=pi;
		cf=0.0;
		for(h=1;h<=mn_dev[idx];h++){
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
		for(h=1;h<=ma_dev[idx];h++){
			cf=cf-d_dev[h]*Oz_dev[ind_Oz + (idx+h)*Nxx]*Tn_dev[(idx+h)];
		}
		aa=pi - cf*dh - Tk_dev[idx]*uz_dev[ind_Oz + idx*Nxx];
		ak=aa+dl;
		uz_dev[ind_Oz + idx*Nxx]=tx2*aa+uz_dev[ind_Oz + idx*Nxx];

		p_dev[ind_p1 + (Nzz-idx)*Nxx]=vel_dev[(Nzz-idx)*Nxx + idy]*ak + 2.0f*p_dev[ind_p2 + (Nzz-idx)*Nxx] - 
			                  p_dev[ind_p1 + (Nzz-idx)*Nxx];
		
	}*/
	/*if (idx>0 && idx<Na && idy>=0 && idy<Na){
		idy=idy+(Nxx-Na+1);

		int ind_p1=k1*Nzz*Nxx + idy;
		int ind_p2=k2*Nzz*Nxx + idy;
		int ind_Oz=idy;

		pi=g_dev[ms]*p_dev[ind_p2 + (Nzz-idx)*Nxx];
		cf=0.0;
		for(h=1;h<=mn_dev[idx];h++){
			pi=pi+g_dev[h]*(p_dev[ind_p2 + ((Nzz-idx)-h)*Nxx]+p_dev[ind_p2 + ((Nzz-idx)+h)*Nxx]);
			cf=cf+d_dev[h]*Oz_dev[ind_Oz + (idx-h)*Nxx]*Tn_dev[(idx-h)];
		}
		for(k=h;k<=no;k++){
			pi=pi+g_dev[k]*p_dev[ind_p2 + ((Nzz-idx)-k)*Nxx];
		}
		for(h=1;h<=ma_dev[idx];h++){
			cf=cf-d_dev[h]*Oz_dev[ind_Oz + (idx+h)*Nxx]*Tn_dev[(idx+h)];
		}	
		aa=pi - cf*dh - Tk_dev[idx]*uz_dev[ind_Oz + idx*Nxx];
		uz_dev[ind_Oz + idx*Nxx]=tx2*aa+uz_dev[ind_Oz + idx*Nxx];
		p_dev[ind_p1 + (Nzz-idx)*Nxx]=p_dev[ind_p1 + (Nzz-idx)*Nxx]+vel_dev[(Nzz-idx)*Nxx + idy]*aa;	
		//i*Nxx+j

	}*/
}

