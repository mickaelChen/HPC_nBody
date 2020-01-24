#include "direct_computation.h"
#include "omp.h"
#include <immintrin.h>

/* debug */
/*extern my_rank;*/


/*********************************************************************************************
**********************************************************************************************
**********************************************************************************************

   Without Matrices

**********************************************************************************************
**********************************************************************************************
*********************************************************************************************/

/* All the following functions use the mutual interaction principle (i.e. reciprocity). */




/*** For debugging only: we check if the positions are too close for direct computation: ***/
/* #define _CHECK_IF_POSITION_ARE_TOO_CLOSE_ */
#ifdef _CHECK_IF_POSITION_ARE_TOO_CLOSE_
#define CHECK_IF_POSITION_ARE_TOO_CLOSE(pos_x_tgt, pos_y_tgt, pos_z_tgt, pos_x_src, pos_y_src, pos_z_src) { \
  if (position_Are_too_close(pos_x_tgt, pos_y_tgt, pos_z_tgt, pos_x_src, pos_y_src, pos_z_src)){            \
    fprintf(f_output, "In file direct_computation.c: the two position are too close:\n");                   \
    pos_xyz_Display(pos_x_tgt, pos_y_tgt, pos_z_tgt, f_output, low);	\
    fprintf(f_output, "\t and \t");					\
    pos_xyz_Display(pos_x_src, pos_y_src, pos_z_src, f_output, low);	\
    fprintf(f_output, "\n");						\
    FMB_ERROR_BRIEF();							\
  }									\
}
#else 
#define CHECK_IF_POSITION_ARE_TOO_CLOSE(pos_x_tgt, pos_y_tgt, pos_z_tgt, pos_x_src, pos_y_src, pos_z_src) 
#endif 


/*********************************************************************************************
**********************************************************************************************

   bodies_Compute_own_interaction

**********************************************************************************************
*********************************************************************************************/

void bodies_Compute_own_interaction(bodies_t *FMB_RESTRICT p_b){

  bodies_ind_t i,j;
  bodies_ind_t n = bodies_Nb_bodies(p_b);

  FMB_CONST COORDINATES_T *FMB_RESTRICT p_px;
  FMB_CONST COORDINATES_T *FMB_RESTRICT p_py;
  FMB_CONST COORDINATES_T *FMB_RESTRICT p_pz;
  FMB_CONST VALUES_T *FMB_RESTRICT p_val;
  COORDINATES_T pix, piy, piz, pjx, pjy, pjz;
  VALUES_T val_i, val_j;

  COORDINATES_T *FMB_RESTRICT p_fx;
  COORDINATES_T *FMB_RESTRICT p_fy;
  COORDINATES_T *FMB_RESTRICT p_fz;

  REAL_T eps_soft_square = FMB_Info.eps_soft_square;

  p_px = p_b->p_pos_x;
  p_py = p_b->p_pos_y;
  p_pz = p_b->p_pos_z;

  p_val = bodies_Get_p_value(p_b, 0);
  p_fx = p_b->p_fx;
  p_fy = p_b->p_fy;
  p_fz = p_b->p_fz;

  for(i=0; i<n-1; i++){
    pix = p_px[i];
    piy = p_py[i];
    piz = p_pz[i];
    val_i = p_val[i];

    for (j=i+1;	 j<n;	 j++){
      pjx = p_px[j];
      pjy = p_py[j];
      pjz = p_pz[j];
      val_j = p_val[j];

      CHECK_IF_POSITION_ARE_TOO_CLOSE(pix, piy, piz, pjx, pjy, pjz);       
      DIRECT_COMPUTATION_MUTUAL_SOFT(pix, piy, piz,
				     pjx, pjy, pjz,
				     val_i,
				     val_j,
				     p_fx[i], p_fy[i], p_fz[i],
				     p_fx[j], p_fy[j], p_fz[j],
				     pot_i,
				     p_pot[j],
				     eps_soft_square);
    }
  }
}

void bodies_Compute_other_interaction(bodies_t *FMB_RESTRICT p_b, COORDINATES_T *pj_pos_x, COORDINATES_T *pj_pos_y, COORDINATES_T *pj_pos_z,
				      COORDINATES_T *pj_fx, COORDINATES_T *pj_fy, COORDINATES_T *pj_fz, VALUES_T *pj_values)
{
  bodies_ind_t i,j;
  int k;
  bodies_ind_t n = bodies_Nb_bodies(p_b);

  FMB_CONST COORDINATES_T *FMB_RESTRICT p_px;
  FMB_CONST COORDINATES_T *FMB_RESTRICT p_py;
  FMB_CONST COORDINATES_T *FMB_RESTRICT p_pz;
  FMB_CONST VALUES_T *FMB_RESTRICT p_val;

  COORDINATES_T *FMB_RESTRICT p_fx;
  COORDINATES_T *FMB_RESTRICT p_fy;
  COORDINATES_T *FMB_RESTRICT p_fz;


  REAL_T eps = FMB_Info.eps_soft_square;
  float tabeps[8] __attribute__((aligned(32))) = {eps, eps, eps, eps, eps, eps, eps, eps};
  __m256 veps = _mm256_load_ps(tabeps);

  __m256 vpix, vpiy, vpiz;
  __m256 vpjx, vpjy, vpjz;
  __m256 vfix, vfiy, vfiz;
  __m256 vvali;
  __m256 vvalj;

  p_px = p_b->p_pos_x;
  p_py = p_b->p_pos_y;
  p_pz = p_b->p_pos_z;

  p_val = bodies_Get_p_value(p_b, 0);
  p_fx = p_b->p_fx;
  p_fy = p_b->p_fy;
  p_fz = p_b->p_fz;

  for(i=0; i<n; i+=8)
    {
      vpix = _mm256_loadu_ps(p_px + i);
      vpiy = _mm256_loadu_ps(p_py + i);
      vpiz = _mm256_loadu_ps(p_pz + i);
      vvali = _mm256_loadu_ps(p_val+i);
      vfix = _mm256_loadu_ps(p_fx + i);
      vfiy = _mm256_loadu_ps(p_fy + i);
      vfiz = _mm256_loadu_ps(p_fz + i);
      for (j=0; j<n; j++)
	{
	  float tpjx[8] __attribute__((aligned(32)));
	  float tpjy[8] __attribute__((aligned(32)));
	  float tpjz[8] __attribute__((aligned(32)));
	  float tvalj[8] __attribute__((aligned(32)));
	  for (k = 0; k < 8; ++k)
	    {
	      tpjx[k] = pj_pos_x[j];
	      tpjy[k] = pj_pos_y[j];
	      tpjz[k] = pj_pos_z[j];
	      tvalj[k] = pj_values[j];
	    }
	  vpjx = _mm256_load_ps(tpjx);
	  vpjy = _mm256_load_ps(tpjy);
	  vpjz = _mm256_load_ps(tpjz);
	  vvalj = _mm256_load_ps(tvalj);
	  for (k = 0; k < 8; ++k)
	    {
	      CHECK_IF_POSITION_ARE_TOO_CLOSE(p_px[i+k], p_py[i+k], p_pz[i+k], pj_pos_x[j], pj_pos_y[j], pj_pos_z[j]);
	    }
	  DIRECT_COMPUTATION_MUTUAL_SOFT_VEC(vpix, vpiy, vpiz,
					     vpjx, vpjy, vpjz,
					     vvali,
					     vvalj,
					     vfix, vfiy, vfiz,
					     pj_fx[j], pj_fy[j], pj_fz[j],
					     pot_i,
					     p_pot[j],
					     veps);
	}
      _mm256_storeu_ps(p_fx+i, vfix);
      _mm256_storeu_ps(p_fy+i, vfiy);
      _mm256_storeu_ps(p_fz+i, vfiz);
    }
}

void bodies_Compute_other_half_interaction(bodies_t *FMB_RESTRICT p_b, COORDINATES_T *pj_pos_x, COORDINATES_T *pj_pos_y, COORDINATES_T *pj_pos_z,
					   COORDINATES_T *pj_fx, COORDINATES_T *pj_fy, COORDINATES_T *pj_fz, VALUES_T *pj_values, int h)
{

  bodies_ind_t i,j;
  bodies_ind_t n = bodies_Nb_bodies(p_b);

  FMB_CONST COORDINATES_T *FMB_RESTRICT p_px;
  FMB_CONST COORDINATES_T *FMB_RESTRICT p_py;
  FMB_CONST COORDINATES_T *FMB_RESTRICT p_pz;
  FMB_CONST VALUES_T *FMB_RESTRICT p_val;
  COORDINATES_T pix, piy, piz, pjx, pjy, pjz;
  VALUES_T val_i, val_j;

  COORDINATES_T *FMB_RESTRICT p_fx;
  COORDINATES_T *FMB_RESTRICT p_fy;
  COORDINATES_T *FMB_RESTRICT p_fz;
  COORDINATES_T fix, fiy, fiz;

  REAL_T eps_soft_square = FMB_Info.eps_soft_square;

  p_px = p_b->p_pos_x;
  p_py = p_b->p_pos_y;
  p_pz = p_b->p_pos_z;

  p_val = bodies_Get_p_value(p_b, 0);
  p_fx = p_b->p_fx;
  p_fy = p_b->p_fy;
  p_fz = p_b->p_fz;

  for(i=0; i<n; i++){
    pix = p_px[i];
    piy = p_py[i];
    piz = p_pz[i];
    val_i = p_val[i];

    fix = p_fx[i];
    fiy = p_fy[i];
    fiz = p_fz[i];

    for (j=0;	 j<=i;	 j++)
      if (j != i || (h == 0 && j < n/2) || (h == 1 && j >= n/2))
      {
	pjx = pj_pos_x[j];
	pjy = pj_pos_y[j];
	pjz = pj_pos_z[j];
	val_j = pj_values[j];

	CHECK_IF_POSITION_ARE_TOO_CLOSE(pix, piy, piz, pjx, pjy, pjz);
	DIRECT_COMPUTATION_MUTUAL_SOFT(pix, piy, piz,
				       pjx, pjy, pjz,
				       val_i,
				       val_j,
				       fix, fiy, fiz,
				       pj_fx[j], pj_fy[j], pj_fz[j],
				       pot_i,
				       p_pot[j],
				       eps_soft_square);
      }
    p_fx[i] = fix;
    p_fy[i] = fiy;
    p_fz[i] = fiz;   
  }
}
