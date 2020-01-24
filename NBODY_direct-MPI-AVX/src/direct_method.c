#include "direct_method.h"
#include "IO.h" 
#include "mpi.h"
#include "omp.h"

/* Here are the initialization of the global variables: */
bodies_t bodies;
char *Direct_data_file;
bool Direct_are_data_bzipped2 = FALSE; 
position_t center;
COORDINATES_T half_side;

extern int nb_proc;
extern int my_rank;

FMB_Info_t FMB_Info;

/* Buffer des Pj et Fj pour les calculs distants */
COORDINATES_T *pj_pos_x;
COORDINATES_T *pj_pos_y;
COORDINATES_T *pj_pos_z;
COORDINATES_T *pj_fx;
COORDINATES_T *pj_fy;
COORDINATES_T *pj_fz;

/* L'ensemble des masses */
VALUES_T *p_allvalues;
/* pointeur vers les masses courantes */
VALUES_T *p_values;

/* See definition in 'FMB.c'. */
void bunzip2_file(const char *filename);
void bzip2_file(const char *filename);

#define SWAP(x1, y1, z1, v1, x2, y2, z2, v2, tmp) tmp=x1; x1=x2; x2=tmp; tmp=y1; y1=y2; y2=tmp; tmp=z1; z1=z2; z2=tmp;





/*********************************************************************************************
**********************************************************************************************

   Direct_method_Init

**********************************************************************************************
*********************************************************************************************/
void Direct_method_Init(){

  /* Checking: */
  if (f_output == NULL){
    FMB_error("'f_output' must be set.\n");
  }


  /************************************ eps_soft_square: **********************************************/
  fprintf(f_output, "Softening parameter: %.1e\n", FMB_Info.eps_soft); 
  FMB_Info.eps_soft_square = FMB_Info.eps_soft * FMB_Info.eps_soft;

  /* Clear 'center' and 'half_side': */
  position_Initialize(&center);
  half_side = (COORDINATES_T) 0.0;

}




/*********************************************************************************************
**********************************************************************************************

   Direct_method_Data

**********************************************************************************************
*********************************************************************************************/

/* Initialisation des buffers Pj */
void Direct_method_Pj_Initialize(bodies_ind_t nb_bodies)
{
  bodies_ind_t k;

  pj_pos_x = FMB_malloc_with_check(nb_bodies * sizeof(COORDINATES_T));
  pj_pos_y = FMB_malloc_with_check(nb_bodies * sizeof(COORDINATES_T));
  pj_pos_z = FMB_malloc_with_check(nb_bodies * sizeof(COORDINATES_T));
  pj_fx = FMB_malloc_with_check(nb_bodies * sizeof(COORDINATES_T));
  pj_fy = FMB_malloc_with_check(nb_bodies * sizeof(COORDINATES_T));
  pj_fz = FMB_malloc_with_check(nb_bodies * sizeof(COORDINATES_T));

  for (k = 0; k < nb_bodies; ++k)
    {
      pj_pos_x[k] = 0;
      pj_pos_y[k] = 0;
      pj_pos_z[k] = 0;
      pj_fx[k] = 0;
      pj_fy[k] = 0;
      pj_fz[k] = 0;
    }
}

/* Free les Pj*/
void Direct_method_Pj_Terminate()
{
  FMB_free(pj_pos_x);
  FMB_free(pj_pos_y);
  FMB_free(pj_pos_z);
  FMB_free(pj_fx);
  FMB_free(pj_fy);
  FMB_free(pj_fz);
}

/* Send des données depuis un buffer d'envoi du processus 0 au processus p*/
void Direct_method_Data_Send(int p, position_t *spd, MPI_Request *req)
{
  MPI_Isend(pj_pos_x, bodies.nb_bodies, MY_MPI_F, p, 1, MPI_COMM_WORLD, req);
  MPI_Isend(pj_pos_y, bodies.nb_bodies, MY_MPI_F, p, 2, MPI_COMM_WORLD, req+1);
  MPI_Isend(pj_pos_z, bodies.nb_bodies, MY_MPI_F, p, 3, MPI_COMM_WORLD, req+2);
  MPI_Isend(spd, bodies.nb_bodies * 3, MY_MPI_F, p, 5, MPI_COMM_WORLD, req+3);
}

/* Initialisation des corps et reception des données */
/* utilisé par les processus autres que 0 */
void Direct_method_Data_InitRecv()
{
  bodies_ind_t nb_bodies;
  bodies_ind_t nb_bodies_total;
  MPI_Request req[4];

  /* Recuperer le nombre de corps local */
  MPI_Bcast(&nb_bodies, 1, MPI_LONG, 0, MPI_COMM_WORLD);
  /* retrouver le nombre de corps total et allouer les masses */
  nb_bodies_total = nb_bodies * nb_proc;
  p_allvalues = FMB_malloc_with_check(nb_bodies_total * sizeof(VALUES_T));

  /* Allocation et initialisation des corps */
  bodies_Initialize(&bodies, nb_bodies);
  bodies.nb_bodies = nb_bodies;
  bodies.size_allocated = nb_bodies;
  bodies.p_values = p_allvalues + nb_bodies * (my_rank - 1);

  /* On place le pointeur des masses */
  p_values = bodies.p_values + nb_bodies;

  /* Reception des corps (position et vitesses) */
  MPI_Irecv(bodies.p_pos_x, nb_bodies, MY_MPI_F, 0, 1, MPI_COMM_WORLD, req);
  MPI_Irecv(bodies.p_pos_y, nb_bodies, MY_MPI_F, 0, 2, MPI_COMM_WORLD, req+1);
  MPI_Irecv(bodies.p_pos_z, nb_bodies, MY_MPI_F, 0, 3, MPI_COMM_WORLD, req+2);
  MPI_Irecv(bodies.p_speed_vectors, nb_bodies * 3, MY_MPI_F, 0, 5, MPI_COMM_WORLD, req+3); /* Utiliser un type MPI_STRUCT si machines non homogènes */
  bodies_ClearFP(&bodies);

  /* Recuperation des masses */
  MPI_Bcast(p_allvalues, nb_bodies_total, MY_MPI_F, 0, MPI_COMM_WORLD);

  /* Initialisation des buffers Pj */
  Direct_method_Pj_Initialize(nb_bodies);

  /* On termine la reception */
  MPI_Wait(req, NULL);
  MPI_Wait(req+1, NULL);
  MPI_Wait(req+2, NULL);
  MPI_Wait(req+3, NULL);
}

/* Lecture et envoie des données */
/* Appelé uniquement par le processus 0 */
void Direct_method_Data(char *data_file){
  bodies_ind_t k;
  bodies_ind_t nb_bodies; 
  int i;
  VALUES_T *current_values;
  void *swap;
  position_t *spd;
  MPI_Request id[4];

  if (INFO_DISPLAY(2)){
    fprintf(f_output, "Opening data file \'%s\' for direct computation... \n", data_file); 
  }

  /* Initialize Input operations: */    
  FMB_IO_InitI(data_file);
  FMB_IO_Scan_header(&nb_bodies, &center, &half_side);

  if (INFO_DISPLAY(1)){
    fprintf(f_output, "Bodies number: ");
    fprintf(f_output, FORMAT_BODIES_IND_T, nb_bodies);
    fprintf(f_output, "\n"); 
    fflush(f_output);
  }

  /* Allocation des masses */
  p_allvalues = FMB_malloc_with_check(nb_bodies * sizeof(VALUES_T));
  /* Positionnement du pointeur des masses courantes */
  p_values = p_allvalues;
  current_values = p_allvalues;

  /* nombre local de corps */
  nb_bodies /= nb_proc;

  /* On envoie le nombre de corps */
  MPI_Bcast(&nb_bodies, 1, MPI_LONG, 0, MPI_COMM_WORLD);

  /* On initialise les corps et les buffers Pj et spd */
  bodies_Initialize(&bodies, nb_bodies);
  Direct_method_Pj_Initialize(nb_bodies);
  spd = FMB_malloc_with_check(nb_bodies * sizeof(position_t));

  /* Pour chaque processus */
  for (i = 1; i <= nb_proc; ++i)
    {
      /* On remplit bodies des nb_bodies corps suivants */
      bodies.nb_bodies = 0;
      bodies.p_values = current_values; /* On place le pointeur de masse au bon endroit du buffer */
      for (k=0; k<nb_bodies; ++k)
	{
	  body_t body_tmp;
	  body_Initialize(&body_tmp);
	  if (FMB_IO_Scan_body(&body_tmp) != 1)
	    FMB_error("In Direct_method_Data(): FMB_IO_Scan_body() failed for body #%i\n", k);
	  /*     if (k<100){ body_Display(&body_tmp, f_output); }  */ 
	  bodies_Add(&bodies, &body_tmp);
	}
      /* On les envoie (en asynchrone) */
      if (i != 1)
	{
	  MPI_Wait(id,NULL); MPI_Wait(id+1,NULL);
	  MPI_Wait(id+2,NULL); MPI_Wait(id+3,NULL);
	}
      if (i != nb_proc)
	{
	  SWAP(bodies.p_pos_x, bodies.p_pos_y, bodies.p_pos_z, bodies.p_speed_vectors,
	       pj_pos_x, pj_pos_y, pj_pos_z, spd, swap);
	  Direct_method_Data_Send(i, spd, id);
	}
      current_values += nb_bodies;
    }
  /* Du coup le processus 0 conserve les derniers corps. Il aurait été malin que ce soit le processus nb_proc - 1 qui fasse la lecture. */

  /* Chaque processus doit récuperer l'ensemble des masses. */
  MPI_Bcast(p_allvalues, nb_bodies * nb_proc, MY_MPI_F, 0, MPI_COMM_WORLD);
  bodies_ClearFP(&bodies);
  FMB_free(spd);
  /* Terminate Input operations: */
  FMB_IO_TerminateI();
}

/*********************************************************************************************
 ********************************************************************************************
 **********************************************************************************************

 Direct_method_Data_bodies

 **********************************************************************************************
 *********************************************************************************************/
/* Same as Direct_method_Data() but we use the position and values
 * of all bodies stored in 'p_b' (instead of the bodies stored
 * in the file "data_file" in Direct_method_Data()). */
void Direct_method_Data_bodies(bodies_t *p_b){
  
  bodies_it_t it;

  bodies_Initialize(&bodies, bodies_Nb_bodies(p_b));

  for (bodies_it_Initialize(&it, p_b);
       bodies_it_Is_valid(&it);
       bodies_it_Go2Next(&it)){
    body_t body_tmp;
    bodies_it_Get_body(&it, &body_tmp);
    bodies_Add(&bodies, &body_tmp);
  }

  bodies_ClearFP(&bodies);

}





/*********************************************************************************************
**********************************************************************************************

   Direct_method_Compute

**********************************************************************************************
*********************************************************************************************/
void Direct_method_Compute_First(){

    /********************* Without reciprocity: *******************************************/
    /* bodies_Compute_own_interaction_no_mutual() is not implemented ... */

    /********************* With reciprocity: **********************************************/
    /* Compute the force and the potential: */
    bodies_Compute_own_interaction(&bodies);        


    /**************** Possible scaling with CONSTANT_INTERACTION_FACTOR: ********************/
    /* We can also use CONSTANT_INTERACTION_FACTOR only for the total potential energy ... */
#ifdef _USE_CONSTANT_INTERACTION_FACTOR_
    bodies_Scale_with_CONSTANT_INTERACTION_FACTOR(&bodies);
#endif /* #ifdef _USE_CONSTANT_INTERACTION_FACTOR_ */


}

/* Computation pour les blocs intermédiaire */
void Direct_method_Compute_Mid()
{
  bodies_Compute_other_interaction(&bodies, pj_pos_x, pj_pos_y, pj_pos_z, pj_fx, pj_fy, pj_fz, p_values);
  p_values += bodies.nb_bodies;
  if (p_values == p_allvalues + bodies.nb_bodies * nb_proc)
    p_values = p_allvalues;

}

/* Computation pour le demi-bloc final */
void Direct_method_Compute_Last()
{
  int rank = (my_rank == 0 ? nb_proc : my_rank);

  bodies_Compute_other_half_interaction(&bodies, pj_pos_x, pj_pos_y, pj_pos_z, pj_fx, pj_fy, pj_fz, p_values, rank <= (nb_proc / 2) ? 0 : 1);
}




/*********************************************************************************************
**********************************************************************************************
************************* Move of the bodies: ************************************************

   Direct_method_Move : Leapfrog integrator ( Kick Drift Kick )  

**********************************************************************************************
*********************************************************************************************/

void KnD_Direct_method_Move(REAL_T dt ){
  /**** Kick N Drift ***/
    
  bodies_it_t it;

 for (bodies_it_Initialize(&it, &bodies);
      bodies_it_Is_valid(&it);
      bodies_it_Go2Next(&it)){
   bodies_Kick_Move(&it,dt);
   bodies_Drift_Move(&it,dt); 
   }
   /*
  bodies_ind_t k;
  
#pragma omp parallel for schedule(static)
  for (k = 0; k < bodies.nb_bodies; ++k)
    {
      bodies.p_speed_vectors[k].dat[0] += bodies.p_fx[k] * (1 / bodies.p_values[k]) * (dt / 2);
      bodies.p_speed_vectors[k].dat[1] += bodies.p_fy[k] * (1 / bodies.p_values[k]) * (dt / 2);
      bodies.p_speed_vectors[k].dat[2] += bodies.p_fz[k] * (1 / bodies.p_values[k]) * (dt / 2);

      bodies.p_pos_x[k] += bodies.p_speed_vectors[k].dat[0] * dt;
      bodies.p_pos_y[k] += bodies.p_speed_vectors[k].dat[1] * dt;
      bodies.p_pos_z[k] += bodies.p_speed_vectors[k].dat[2] * dt;
    }
  */
}

void K_Direct_method_Move(REAL_T dt ){
  /************************* Move of the bodies: ******************************************/
    
  bodies_it_t it;
  for (bodies_it_Initialize(&it, &bodies);
       bodies_it_Is_valid(&it);
       bodies_it_Go2Next(&it)){
    bodies_Kick_Move(&it,dt);
  }
  /*
  bodies_ind_t k;
  
#pragma omp parallel for schedule(static)
  for (k = 0; k < bodies.nb_bodies; ++k)
    {
      //      fprintf(stderr,"c=%d thread %d \n", k, omp_get_thread_num());
      bodies.p_speed_vectors[k].dat[0] += bodies.p_fx[k] * (1 / bodies.p_values[k]) * (dt / 2);
      bodies.p_speed_vectors[k].dat[1] += bodies.p_fy[k] * (1 / bodies.p_values[k]) * (dt / 2);
      bodies.p_speed_vectors[k].dat[2] += bodies.p_fz[k] * (1 / bodies.p_values[k]) * (dt / 2);
    }
  */
}










/*********************************************************************************************
**********************************************************************************************

   Direct_method_Terminate

**********************************************************************************************
*********************************************************************************************/
void Direct_method_Terminate(){

  bodies_Free(&bodies);
  FMB_free(p_allvalues);
  if (Direct_are_data_bzipped2){
    /* We recompress the data file: */
    bzip2_file(Direct_data_file);
  }
  FMB_free(Direct_data_file);

}

void Direct_method_Terminate2(){
  
  bodies_Free(&bodies);
  if (Direct_are_data_bzipped2){
    bzip2_file(Direct_data_file);
  }
  FMB_free(Direct_data_file);
}



























/*********************************************************************************************
**********************************************************************************************

   sum

**********************************************************************************************
*********************************************************************************************/
void Direct_method_Sum(char *results_file,
		       unsigned long step_number_value,
		       bodies_t *p_bodies, 
		       VALUES_T total_potential_energy){

  FILE *f_results;
  position_t force_sum;
  position_t force_sum_total;
  bodies_it_t it;

  f_results = f_output;

  position_Initialize(&force_sum);
  position_Initialize(&force_sum_total);
  for (bodies_it_Initialize(&it, p_bodies);
       bodies_it_Is_valid(&it);
       bodies_it_Go2Next(&it)){ 
    position_Set_x(&force_sum, position_Get_x(&force_sum) + bodies_it_Get_fx(&it));
    position_Set_y(&force_sum, position_Get_y(&force_sum) + bodies_it_Get_fy(&it));
    position_Set_z(&force_sum, position_Get_z(&force_sum) + bodies_it_Get_fz(&it));
  }

  MPI_Reduce(&force_sum, &force_sum_total, 3, MY_MPI_F, MPI_SUM, 0, MPI_COMM_WORLD);

  if (my_rank == 0)
    {
      fprintf(f_results, "Sum (force): ");
      position_Display(&force_sum_total, f_results, high);
      fprintf(f_results, "\n");
    }
}








/*********************************************************************************************
**********************************************************************************************

   save 

**********************************************************************************************
*********************************************************************************************/
void Direct_method_Dump_bodies(char *results_filename,
			       unsigned long step_number_value,
			       bodies_t *p_bodies)
{
  bodies_it_t it;

  /* Initialize Ouput operations: */    
  FMB_IO_InitO(results_filename);
  
  if (FMB_IO_Info.output_format != NEMO_format){
    
    /********** FMB file format: **********/
    if (FMB_IO_Info.output_format == FMB_binary_format){
      FMB_error("Unable to write the 'header' for FMB_binary_format in Direct_method_Dump_bodies(). \n");
    }

    FMB_IO_Print_header(step_number_value, FALSE /* only_position_and_value */,
			bodies_Nb_bodies(p_bodies) * nb_proc, &center, half_side);
    
    for (bodies_it_Initialize(&it, p_bodies);
	 bodies_it_Is_valid(&it);
	 bodies_it_Go2Next(&it)){ 

      FMB_IO_Print_body_from_bodies_it(&it, FALSE /* only_position_and_value */);
    } /* for */
    
  } /* if (FMB_IO_Info.output_format != NEMO_format) */
  else {
    /********** NEMO file format: **********/
    FMB_IO_Print_all_bodies_from_bodies_t(p_bodies);
  } /* else (FMB_IO_Info.output_format != NEMO_format) */

  /* Terminate Output operations: */    
  FMB_IO_TerminateO();

}
