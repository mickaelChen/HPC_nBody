#ifndef __FORCE_DEFINITION_H__
#define __FORCE_DEFINITION_H__

/*! \file 
  We describe here all the features of the force field between the bodies
  (electrostatic, gravitational ...) */

/*! Universal gravitational constant: */
#define G ((VALUES_T) 6.67259E-11)
/* In 2D: */
/* #define POTENTIAL_SIGN  */
/* #define UNIT_VECTOR_COMPONENT(tgt_comp, src_comp) ((tgt_comp) - (src_comp))   */
/* In 3D: */
#define POTENTIAL_SIGN -
#define UNIT_VECTOR_COMPONENT(tgt_comp, src_comp) ((src_comp) - (tgt_comp))  
#define UNIT_VECTOR_COMPONENT_VEC(tgt_comp, src_comp) _mm256_sub_ps((src_comp), (tgt_comp))  


#define CONSTANT_INTERACTION_FACTOR G 


extern long long unsigned int test;


#include <immintrin.h>







/*********************************************************************************************
**********************************************************************************************
**********************************************************************************************

   MACROS MUTUAL 

**********************************************************************************************
**********************************************************************************************
*********************************************************************************************/


/*********************************************************************************************
**********************************************************************************************

   Forces only

**********************************************************************************************
*********************************************************************************************/


/* With softening parameter. */
/* 23 flops 
 * (remark: there is another version which exposes rsqrt() but leads to 24 flops...). */
#define DIRECT_COMPUTATION_MUTUAL_SOFT(pxt, pyt, pzt,								\
				       pxs, pys, pzs,								\
				       v_target,								\
				       v_src,									\
				       fxt, fyt, fzt,								\
				       fxs, fys, fzs,	        					        \
				       unused1,									\
				       unused2,									\
                                       eps_soft_square){                                                        \
  COORDINATES_T dx = UNIT_VECTOR_COMPONENT(pxt, pxs);								\
  COORDINATES_T dy = UNIT_VECTOR_COMPONENT(pyt, pys);								\
  COORDINATES_T dz = UNIT_VECTOR_COMPONENT(pzt, pzs);								\
  COORDINATES_T inv_square_distance;										\
  COORDINATES_T inv_distance;											\
  COORDINATES_T fx, fy, fz;											\
														\
  /************ Compute the "distance" and the norm of the force: ***************************/			\
  inv_square_distance = 1.0/ (dx*dx + dy*dy + dz*dz + eps_soft_square);	        				\
  inv_distance = FMB_SQRT(inv_square_distance);									\
														\
  /* The "inv_square_distance" will now be equal to: distance^{-3} */						\
  inv_distance *= (v_target) * (v_src);										\
  inv_square_distance *= inv_distance;										\
														\
  /************ Compute the force vector contribution 								\
                and update the force vectors of the target AND the source: ******************/			\
  fx = dx * inv_square_distance;					\
  fy = dy * inv_square_distance;					\
  fz = dz * inv_square_distance;					\
  fxt += fx;								\
  fyt += fy;								\
  fzt += fz;								\
  fxs -= fx;								\
  fys -= fy;								\
  fzs -= fz;								\
  									\
  test++;								\
}

#define DIRECT_COMPUTATION_MUTUAL_SOFT_VEC(pxt, pyt, pzt,		\
					   pxs, pys, pzs,		\
					   v_target,			\
					   v_src,			\
					   fxt, fyt, fzt,		\
					   fxs, fys, fzs,		\
					   unused1,			\
					   unused2,			\
					   eps_soft_square){		\
    __m256 dx = UNIT_VECTOR_COMPONENT_VEC(pxt, pxs);			\
    __m256 dy = UNIT_VECTOR_COMPONENT_VEC(pyt, pys);			\
    __m256 dz = UNIT_VECTOR_COMPONENT_VEC(pzt, pzs);			\
    __m256 inv_square_distance;						\
    __m256 inv_distance;						\
    __m256 fx, fy, fz;							\
    float tab_ones[8] __attribute__((aligned(32))) = {1,1,1,1,1,1,1,1};	\
    __m256 vones = _mm256_load_ps(tab_ones);				\
    int ireduce;							\
    float sumx, sumy, sumz;						\
    /************ Compute the "distance" and the norm of the force: ***************************/ \
    inv_square_distance = _mm256_div_ps(vones, ( _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(dx,dx), _mm256_mul_ps(dy,dy)), \
							       _mm256_add_ps(_mm256_mul_ps(dz,dz), eps_soft_square)))); \
    inv_distance = _mm256_sqrt_ps(inv_square_distance);			\
    									\
    /* The "inv_square_distance" will now be equal to: distance^{-3} */	\
    inv_distance = _mm256_mul_ps(inv_distance, _mm256_mul_ps(v_target, v_src)); \
    inv_square_distance = _mm256_mul_ps(inv_square_distance, inv_distance); \
    									\
    /************ Compute the force vector contribution			\
                and update the force vectors of the target AND the source: ******************/ \
    /* on aurait pu faire une fma en AVX2 */                            \
    fx = _mm256_mul_ps(dx, inv_square_distance);			\
    fy = _mm256_mul_ps(dy, inv_square_distance);			\
    fz = _mm256_mul_ps(dz, inv_square_distance);			\
    fxt = _mm256_add_ps(fxt, fx);					\
    fyt = _mm256_add_ps(fyt, fy);					\
    fzt = _mm256_add_ps(fzt, fz);					\
    sumx = 0;								\
    sumy = 0;								\
    sumz = 0;								\
    for (ireduce = 0; ireduce < 8; ireduce++)				\
      {									\
	sumx += ((float*)&fx)[ireduce];					\
	sumy += ((float*)&fy)[ireduce];					\
	sumz += ((float*)&fz)[ireduce];					\
      }									\
    fxs -= sumx;							\
    fys -= sumy;							\
    fzs -= sumz;							\
    									\
    test++;								\
  }

#endif /* #ifndef FORCE_DEFINITION_H */


