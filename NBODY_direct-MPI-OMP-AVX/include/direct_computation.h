#ifndef __DIRECT_COMPUTATION_H__
#define __DIRECT_COMPUTATION_H__

#include "bodies.h"






/*********************************************************************************************
**********************************************************************************************

   Compute the interaction (forces and/or potentials)

**********************************************************************************************
*********************************************************************************************/



/*! If 'mutual' is set to TRUE, we use the mutual interaction principle to set the force vector of 
 all the bodies of *p_b_target AND of *p_b_src (and possibly the potential).
 Otherwise we do not modify the content of *p_b_src. */
void bodies_Compute_direct_interaction(bodies_t *FMB_RESTRICT p_b_target, 
				       bodies_t *FMB_RESTRICT p_b_src,
				       bool mutual);

/*! The mutual interaction principle is always used in 'bodies_Compute_own_interaction()'. */
void bodies_Compute_own_interaction(bodies_t *FMB_RESTRICT p_b);

void bodies_Compute_own_interaction_for_first_ones(bodies_t *FMB_RESTRICT p_b,
					      bodies_ind_t nmax);

void bodies_Compute_other_interaction(bodies_t *FMB_RESTRICT p_b, COORDINATES_T *pj_pos_x, COORDINATES_T *pj_pos_y, COORDINATES_T *pj_pos_z,
				      COORDINATES_T *pj_fx, COORDINATES_T *pj_fy, COORDINATES_T *pj_fz, VALUES_T *pj_values);

void bodies_Compute_other_half_interaction(bodies_t *FMB_RESTRICT p_b, COORDINATES_T *pj_pos_x, COORDINATES_T *pj_pos_y, COORDINATES_T *pj_pos_z,
					   COORDINATES_T *pj_fx, COORDINATES_T *pj_fy, COORDINATES_T *pj_fz, VALUES_T *pj_values, int h);


#endif /* #ifdef __DIRECT_COMPUTATION_H__ */ 
