#include "direct_computation.h"







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


#ifdef _BODIES_SPLIT_DATA_

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
  COORDINATES_T fix, fiy, fiz;

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

    fix = p_fx[i];
    fiy = p_fy[i];
    fiz = p_fz[i];


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
				     fix, fiy, fiz,
				     p_fx[j], p_fy[j], p_fz[j],
				     pot_i,
				     p_pot[j],
				     eps_soft_square);
    }

    p_fx[i] = fix;    
    p_fy[i] = fiy;
    p_fz[i] = fiz;   
  }
}

void bodies_Compute_other_interaction(bodies_t *FMB_RESTRICT p_b, COORDINATES_T *pj_pos_x, COORDINATES_T *pj_pos_y, COORDINATES_T *pj_pos_z,
				      COORDINATES_T *pj_fx, COORDINATES_T *pj_fy, COORDINATES_T *pj_fz, VALUES_T *pj_values)
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
  COORDINATES_T fix, fiy, fiz, fjx, fjy, fjz;

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


    for (j=0; j<n; j++)
      {
	pjx = pj_pos_x[j];
	pjy = pj_pos_y[j];
	pjz = pj_pos_z[j];
	fjx = pj_fx[j];
	fjy = pj_fy[j];
	fjz = pj_fz[j];

	val_j = pj_values[j];

	CHECK_IF_POSITION_ARE_TOO_CLOSE(pix, piy, piz, pjx, pjy, pjz);
	DIRECT_COMPUTATION_MUTUAL_SOFT(pix, piy, piz,
				       pjx, pjy, pjz,
				       val_i,
				       val_j,
				       fix, fiy, fiz,
				       fjx, fjy, fjz,
				       pot_i,
				       p_pot[j],
				       eps_soft_square);
	pj_fx[j] = fjx;    
	pj_fy[j] = fjy;
	pj_fz[j] = fjz;   
      }
    p_fx[i] = fix;
    p_fy[i] = fiy;
    p_fz[i] = fiz;   
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
  COORDINATES_T fix, fiy, fiz, fjx, fjy, fjz;

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
	fjx = pj_fx[j];
	fjy = pj_fy[j];
	fjz = pj_fz[j];

	val_j = pj_values[j];

	CHECK_IF_POSITION_ARE_TOO_CLOSE(pix, piy, piz, pjx, pjy, pjz);
	DIRECT_COMPUTATION_MUTUAL_SOFT(pix, piy, piz,
				       pjx, pjy, pjz,
				       val_i,
				       val_j,
				       fix, fiy, fiz,
				       fjx, fjy, fjz,
				       pot_i,
				       p_pot[j],
				       eps_soft_square);
	pj_fx[j] = fjx;    
	pj_fy[j] = fjy;
	pj_fz[j] = fjz;   
      }
    p_fx[i] = fix;
    p_fy[i] = fiy;
    p_fz[i] = fiz;   
  }
}

#else /* #ifdef _BODIES_SPLIT_DATA_ */

void 
bodies_Compute_own_interaction(bodies_t *FMB_RESTRICT p_b){

  bodies_ind_t i,j;
  bodies_ind_t n = bodies_Nb_bodies(p_b);

  body_t *FMB_RESTRICT p_body_i;
  body_t *FMB_RESTRICT p_body_j;

  COORDINATES_T pix, piy, piz;
  COORDINATES_T pjx, pjy, pjz;
  VALUES_T val_i;
  position_t *p_force_vector_j;
  COORDINATES_T fix, fiy, fiz;

  REAL_T eps_soft_square = FMB_Info.eps_soft_square;

  p_body_i = p_b->p_bodies;

  for(i=0; i<n-1; i++){
    pix = POSITION_GET_X(body_Get_p_position(p_body_i));
    piy = POSITION_GET_Y(body_Get_p_position(p_body_i));
    piz = POSITION_GET_Z(body_Get_p_position(p_body_i));
    val_i = body_Get_value(p_body_i);
    fix = POSITION_GET_X(body_Get_p_force_vector(p_body_i));
    fiy = POSITION_GET_Y(body_Get_p_force_vector(p_body_i));
    fiz = POSITION_GET_Z(body_Get_p_force_vector(p_body_i));

    p_body_j = p_body_i;

    for (j=i+1;	 j<n;	 j++){
      ++p_body_j;
      pjx = POSITION_GET_X(body_Get_p_position(p_body_j));
      pjy = POSITION_GET_Y(body_Get_p_position(p_body_j));
      pjz = POSITION_GET_Z(body_Get_p_position(p_body_j));
      p_force_vector_j = body_Get_p_force_vector(p_body_j);

      CHECK_IF_POSITION_ARE_TOO_CLOSE(pix, piy, piz, pjx, pjy, pjz); 
      DIRECT_COMPUTATION_MUTUAL_SOFT(pix, piy, piz,
				     pjx, pjy, pjz,
				     val_i,
				     body_Get_value(p_body_j),
				     fix, fiy, fiz,
				     POSITION_GET_X(p_force_vector_j), 
				     POSITION_GET_Y(p_force_vector_j), 
				     POSITION_GET_Z(p_force_vector_j), 
				     pot_i,
				     BODY_GET_POTENTIAL(p_body_j),
				     eps_soft_square);
    }

    position_Set_x(body_Get_p_force_vector(p_body_i), fix);
    position_Set_y(body_Get_p_force_vector(p_body_i), fiy);
    position_Set_z(body_Get_p_force_vector(p_body_i), fiz);
    ++p_body_i;
  }
}

void bodies_Compute_other_half_interaction(bodies_t *FMB_RESTRICT p_b, COORDINATES_T *pj_pos_x, COORDINATES_T *pj_pos_y, COORDINATES_T *pj_pos_z,
					   COORDINATES_T *pj_fx, COORDINATES_T *pj_fy, COORDINATES_T *pj_fz, VALUES_T *pj_values, int h)
{

  bodies_ind_t i,j;
  bodies_ind_t n = bodies_Nb_bodies(p_b);

  body_t *FMB_RESTRICT p_body_i;
  body_t *FMB_RESTRICT p_body_j;

  COORDINATES_T pix, piy, piz;
  COORDINATES_T pjx, pjy, pjz;
  VALUES_T val_i;
  position_t *p_force_vector_j;
  COORDINATES_T fix, fiy, fiz;
  COORDINATES_T fjx, fjy, fjz;

  REAL_T eps_soft_square = FMB_Info.eps_soft_square;

  p_body_i = p_b->p_bodies;



  for(i=0; i<n-1; i++){
    pix = POSITION_GET_X(body_Get_p_position(p_body_i));
    piy = POSITION_GET_Y(body_Get_p_position(p_body_i));
    piz = POSITION_GET_Z(body_Get_p_position(p_body_i));
    val_i = body_Get_value(p_body_i);
    fix = POSITION_GET_X(body_Get_p_force_vector(p_body_i));
    fiy = POSITION_GET_Y(body_Get_p_force_vector(p_body_i));
    fiz = POSITION_GET_Z(body_Get_p_force_vector(p_body_i));

    p_body_j = p_body_i;

    for (j=0;	 j<=i;	 j++){
      if (j != i || (h == 0 && j < n/2) || (h == 1 && j >= n/2))
	{
	  ++p_body_j;
	  pjx = pj_pos_x[j];
	  pjy = pj_pos_y[j];
	  pjz = pj_pos_z[j];
	  fjx = pj_fx[j];
	  fjy = pj_fy[j];
	  fjz = pj_fz[j];
      
	  CHECK_IF_POSITION_ARE_TOO_CLOSE(pix, piy, piz, pjx, pjy, pjz); 
	  DIRECT_COMPUTATION_MUTUAL_SOFT(pix, piy, piz,
					 pjx, pjy, pjz,
					 val_i,
					 pj_values[j],
					 fix, fiy, fiz,
					 fjx, fjy, fjz,
					 pot_i,
					 BODY_GET_POTENTIAL(p_body_j),
					 eps_soft_square);
	}

      position_Set_x(body_Get_p_force_vector(p_body_i), fix);
      position_Set_y(body_Get_p_force_vector(p_body_i), fiy);
      position_Set_z(body_Get_p_force_vector(p_body_i), fiz);
      ++p_body_i;
    }
  }
}

#endif /* #ifdef _BODIES_SPLIT_DATA_ */
