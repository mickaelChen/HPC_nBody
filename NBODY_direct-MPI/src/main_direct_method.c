/* #include <stdio.h> */
/* #include <stdlib.h> */
/* #include <string.h> */
#include <sys/stat.h>

#include "direct_method.h"
#include "IO.h" 
#include "mpi.h"

int nb_proc;
int my_rank;

long long unsigned int test;


/* Buffer des Pj et Fj pour les calculs distants */
extern COORDINATES_T *pj_pos_x;
extern COORDINATES_T *pj_pos_y;
extern COORDINATES_T *pj_pos_z;
extern COORDINATES_T *pj_fx;
extern COORDINATES_T *pj_fy;
extern COORDINATES_T *pj_fz;

/* L'ensemble des masses */
extern VALUES_T *p_allvalues;

/* Les buffers d'envoie et de reception */
COORDINATES_T *pjrecv_pos_x;
COORDINATES_T *pjrecv_pos_y;
COORDINATES_T *pjrecv_pos_z;
COORDINATES_T *pjrecv_fx;
COORDINATES_T *pjrecv_fy;
COORDINATES_T *pjrecv_fz;
COORDINATES_T *pjsend_fx;
COORDINATES_T *pjsend_fy;
COORDINATES_T *pjsend_fz;

/* For FMB_Info.save: */
#define RESULTS_DIR "/tmp/NBODY_direct_results_CHEN-COURTOIS/"
#define RESULTS_FILE "results_"

#define POS(a) (a) >= 0 ? (a) : ((a) + nb_proc)

/*** For timers: ***/
#include <sys/time.h>
double my_gettimeofday(){
  struct timeval tmp_time;
  gettimeofday(&tmp_time, NULL);
  return tmp_time.tv_sec + (tmp_time.tv_usec * 1.0e-6L);
}

/* See definition at the end of this file. */
int parse_command(int argc, 
		  char **argv,
		  char **p_data_file,
		  char **p_results_file);

/* Send de nb_bodies coordonées au processus k.
t=0 pour les Pi
t=1 pour les résultats */
void my_send(COORDINATES_T *x, COORDINATES_T *y, COORDINATES_T *z, int k, int t, MPI_Request * ids)
{
  MPI_Isend(x, bodies.nb_bodies, MY_MPI_F, k, 3*t,   MPI_COMM_WORLD, ids);
  MPI_Isend(y, bodies.nb_bodies, MY_MPI_F, k, 3*t+1, MPI_COMM_WORLD, ids+1);
  MPI_Isend(z, bodies.nb_bodies, MY_MPI_F, k, 3*t+2, MPI_COMM_WORLD, ids+2);
}

/* Send de nb_bodies coordonées au processus k.
t=0 pour les Pi
t=1 pour les résultats */
void my_recv(COORDINATES_T *x, COORDINATES_T *y, COORDINATES_T *z, int k, int t, MPI_Request * idr)
{
  MPI_Irecv(x, bodies.nb_bodies, MY_MPI_F, k, 3*t,   MPI_COMM_WORLD, idr);
  MPI_Irecv(y, bodies.nb_bodies, MY_MPI_F, k, 3*t+1, MPI_COMM_WORLD, idr+1);
  MPI_Irecv(z, bodies.nb_bodies, MY_MPI_F, k, 3*t+2, MPI_COMM_WORLD, idr+2);
}

/* Réinitialise le buffer des résultats */
void clearFj()
{
  bodies_ind_t k;

  for (k = 0; k < bodies.nb_bodies; ++k)
    {
      pj_fx[k] = 0;
      pj_fy[k] = 0;
      pj_fz[k] = 0;
    }  
}

/* Enchange les pointeurs */
#define SWAP(x1, y1, z1, x2, y2, z2, tmp) tmp=x1; x1=x2; x2=tmp; tmp=y1; y1=y2; y2=tmp; tmp=z1; z1=z2; z2=tmp;

/*********************************************************************************************
**********************************************************************************************

   MAIN 

**********************************************************************************************
*********************************************************************************************/

int main(int argc, char **argv){
  long nb_steps = 0;
  REAL_T tstart ,tend , tnow ; 
  tstart = 0 ; 
  tnow = tstart ; 
  tend = 0.001 ; 

  char *data_file = NULL;
  char *results_file = NULL;
  VALUES_T total_potential_energy = 0.0;

  /* Timers: */
  double t_start = 0.0, t_end = 0.0;

  /* itérateurs */
  int i;
  bodies_ind_t k;

  /* l'ensemble des bodies (uniquement processus 0 pour le save) */
  bodies_t bodies_all;

  /* Pour le recouvrement */
  COORDINATES_T *tmp;   /* utilisé par SWAP */
  MPI_Request idps[3];
  MPI_Request idpr[3];
  MPI_Request idfs[3];
  MPI_Request idfr[3];

  /* Initialisation de MPI */
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &nb_proc);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

  f_output = stdout; /* by default */

  if (nb_proc <= 1)
    {
      fprintf(f_output, "Error: Pour un seul processus, utilisez la version sequentielle.\n");
      MPI_Finalize();
      return 0;
    }

  /* Récupération des paramètres */
  parse_command(argc, argv, &data_file, &results_file);

  if (my_rank == 0)
    {
      /* On utilise la structure de tableaux */
#ifndef _BODIES_SPLIT_DATA_
      fprintf(f_output, "Error: Please use _BODIES_SPLIT_DATA_\n");
      MPI_Finalize();
      return -1;
#else
      fprintf(f_output, "Using _BODIES_SPLIT_DATA_\n");
#endif
      
      if (INFO_DISPLAY(1)){
	fprintf(f_output, 
		"*** Compute own interactions of the box defined in \"%s\" ***.\n", 
		data_file);
      }
    }

  Direct_method_Init();

  /* Lecture et envoie des données par le processus 0 */
  /* On découpe les tableaux en nb_proc bloc de lignes. Chaque processus reçoit nb_corps / nb_proc corps. */
  /* Attention: Le processus k recoit les k-ièmes paquet de données et le processus 0 recoit le nb_proc-ième paquet */
  if (my_rank == 0)
      Direct_method_Data(data_file);

  tend=FMB_Info.tend;
  if (my_rank == 0 && INFO_DISPLAY(1))
    { 
      fprintf(f_output, "Start Time : %lf \t End Time : %lf \t dt : %lf \n",tstart, tend, FMB_Info.dt);
      fprintf(f_output, "Number of steps: %lu\n", (unsigned long) ((tend-tstart)/FMB_Info.dt));
    }

  /* Recupération des données par les autres processus */
  if (my_rank != 0)
    Direct_method_Data_InitRecv();

  /* Si --save, le processus 0 nécéssite un buffer pour stocker l'ensemble des corps */
  if (FMB_Info.save)
    {
      bodies_Initialize(&bodies_all, bodies.nb_bodies * (nb_proc+1)); /* +1 pour réordonner les corps facilement */
      bodies_all.nb_bodies = bodies.nb_bodies * nb_proc;
    }

  /* Initialisation des buffers pour la reception et envoie des données pour le recouvrement par le calcul */
  pjrecv_pos_x = FMB_malloc_with_check(bodies.nb_bodies * sizeof(COORDINATES_T));
  pjrecv_pos_y = FMB_malloc_with_check(bodies.nb_bodies * sizeof(COORDINATES_T));
  pjrecv_pos_z = FMB_malloc_with_check(bodies.nb_bodies * sizeof(COORDINATES_T));
  pjrecv_fx = FMB_malloc_with_check(bodies.nb_bodies * sizeof(COORDINATES_T));
  pjrecv_fy = FMB_malloc_with_check(bodies.nb_bodies * sizeof(COORDINATES_T));
  pjrecv_fz = FMB_malloc_with_check(bodies.nb_bodies * sizeof(COORDINATES_T));
  pjsend_fx = FMB_malloc_with_check(bodies.nb_bodies * sizeof(COORDINATES_T));
  pjsend_fy = FMB_malloc_with_check(bodies.nb_bodies * sizeof(COORDINATES_T));
  pjsend_fz = FMB_malloc_with_check(bodies.nb_bodies * sizeof(COORDINATES_T));


  /******************************************************************************************/
  /********************************** Start of the simulation: ******************************/
  /******************************************************************************************/

  /* A chaque pas de temps: */
  while ( tnow-FMB_Info.dt < tend )
    { 
      /******************************************************************/
      test = 0;
      /******************************************************************/
      /********************* Direct method computation: ************************************/
      /*********************Direct metho Move : K-D-K **************************************/ 
      if(tnow!=0)
	{
	  /* Kick and Drift*/
	  KnD_Direct_method_Move(FMB_Info.dt ); 
	}
	  /* Reinitialisation des forces */
      bodies_ClearFP(&bodies);


      /* Start timer: */
      t_start = my_gettimeofday();

      /* On commence l'envoie des nouvelles positions dès qu'elles sont calculées (apres le drift) */
      my_send(bodies.p_pos_x, bodies.p_pos_y, bodies.p_pos_z, POS(my_rank - 1) % nb_proc, 0, idps); //Pi
      my_recv(pj_pos_x, pj_pos_y, pj_pos_z, (my_rank+1) % nb_proc, 0, idpr); //Pj

      /* Computation des blocs de Cij diagonaux (anti-symétriques) */
      Direct_method_Compute_First();

      /* On attends d'avoir reçu les données dans pj pour pouvoir procéder à la suite */
      MPI_Wait(idpr,NULL);
      MPI_Wait(idpr+1,NULL);
      MPI_Wait(idpr+2,NULL);
      
      /* Une itération correspond à un bloc de Cij par processus */
      for (i = 1; i <= (nb_proc-1) / 2 ; ++i)
	{
	  /* On commence directement la transmission des Pj vers le tampon du processus précédent */
	  my_send(pj_pos_x, pj_pos_y, pj_pos_z, POS(my_rank-1) % nb_proc, 0, idps);
	  my_recv(pjrecv_pos_x, pjrecv_pos_y, pjrecv_pos_z, (my_rank+1) % nb_proc, 0, idpr);

	  /* Pendant ce temps... on calcule */
	  clearFj();
	  Direct_method_Compute_Mid(); /* Calcul des blocs non symétriques */

	  /* On attends que les F(j-1) ont été reçus et on les ajoute aux Fi */
	  if (i != 1)
	    {
	      MPI_Wait(idfr,NULL); MPI_Wait(idfr+1,NULL); MPI_Wait(idfr+2,NULL);
	      for (k = 0; k < bodies.nb_bodies; ++k)
		{
		  bodies.p_fx[k] += pjrecv_fx[k];
		  bodies.p_fy[k] += pjrecv_fy[k];
		  bodies.p_fz[k] += pjrecv_fz[k];
		}
	      MPI_Wait(idfs,NULL); MPI_Wait(idfs+1,NULL); MPI_Wait(idfs+2,NULL);
	    }
	  /* On envoie les Fj lorsque l'envoie précédent est terminé */
	  SWAP(pj_fx, pj_fy, pj_fz, pjsend_fx, pjsend_fy, pjsend_fz, tmp);
	  my_send(pjsend_fx, pjsend_fy, pjsend_fz, (my_rank+i) % nb_proc, 1, idfs); // Fj
	  my_recv(pjrecv_fx, pjrecv_fy, pjrecv_fz, POS(my_rank-i) % nb_proc, 1, idfr); // Fj

	  /* On attends la fin de la transmission des Pj et on echange les buffers */
	  MPI_Wait(idpr,NULL); MPI_Wait(idpr+1,NULL); MPI_Wait(idpr+2,NULL);
	  MPI_Wait(idps,NULL); MPI_Wait(idps+1,NULL); MPI_Wait(idps+2,NULL);
	  SWAP(pj_pos_x, pj_pos_y, pj_pos_z, pjrecv_pos_x, pjrecv_pos_y, pjrecv_pos_z, tmp);
	}

      
      if ((nb_proc & 1) == 0) // Si le nombre de processus est paire, il faut calculer des demi-matrices supplémentaires
      	{
      	  /* Calcul */
      	  clearFj();
	  Direct_method_Compute_Last(); /* Calcul du dernier demi-bloc */

      	  /* On attends que les F(j-1) ont été reçus et on les ajoute aux Fi */
      	  if (i!=1)
      	    {
      	      MPI_Wait(idfr,NULL); MPI_Wait(idfr+1,NULL); MPI_Wait(idfr+2,NULL);
      	      for (k = 0; k < bodies.nb_bodies; ++k)
      		{
      		  bodies.p_fx[k] += pjrecv_fx[k];
      		  bodies.p_fy[k] += pjrecv_fy[k];
      		  bodies.p_fz[k] += pjrecv_fz[k];
      		}
      	      MPI_Wait(idfs,NULL); MPI_Wait(idfs+1,NULL); MPI_Wait(idfs+2,NULL);
      	    }
      	  SWAP(pj_fx, pj_fy, pj_fz, pjsend_fx, pjsend_fy, pjsend_fz, tmp);
      	  my_send(pjsend_fx, pjsend_fy, pjsend_fz, (my_rank+i) % nb_proc, 1, idfs); // Fj
      	  my_recv(pjrecv_fx, pjrecv_fy, pjrecv_fz, POS(my_rank - i) % nb_proc, 1, idfr); // Fj
      	}
      /* On envoie les Fj lorsque l'envoie précédent est terminé */
      MPI_Wait(idfr,NULL); MPI_Wait(idfr+1,NULL); MPI_Wait(idfr+2,NULL);
      for (k = 0; k < bodies.nb_bodies; ++k)
      	{
      	  bodies.p_fx[k] += pjrecv_fx[k];
      	  bodies.p_fy[k] += pjrecv_fy[k];
      	  bodies.p_fz[k] += pjrecv_fz[k];
      	}

      /* End timer: */
      t_end = my_gettimeofday();
     
      /* Kick */
      if (tnow !=0)
	K_Direct_method_Move(FMB_Info.dt);

      /****************** Save & display the total time used for this step: *******************/      
      if (my_rank == 0 && INFO_DISPLAY(1))
	{
	  unsigned long long nb_int = NB_OWN_INT(bodies_Nb_bodies(&bodies) * nb_proc);

	  fprintf(f_output, "\n#######################################################################\n");

	  fprintf(f_output, "Time now ( Step number) : %lf (%ld) \n",tnow,nb_steps );
	  fprintf(f_output, "Computation time = %f seconds\n", t_end - t_start);

	  fprintf(f_output, "Interactions computed: %llu\n", nb_int);
	  fprintf(f_output, "  Nb interactions / second: %.3f\n", ((double) nb_int) / (t_end - t_start));

	  fprintf(f_output, "  Gflop/s = %.3f (11.5 flop with mutual) \n",
		  ((((double) nb_int) / (t_end - t_start)) * 11.5 /* 23 flops divided by 2 since mutual */) / (1000000000.0));
	}
      //fprintf(stderr, "Nb interactions calculés par le processus %llu: %d\n", my_rank, test);

      /************************* Save the positions and the forces: ***************************/
      if (FMB_Info.save)
	{
	  /* On récupere tous les corps */
	  MPI_Gather(bodies.p_pos_x, bodies.nb_bodies, MY_MPI_F, bodies_all.p_pos_x, bodies.nb_bodies, MY_MPI_F, 0, MPI_COMM_WORLD);
	  MPI_Gather(bodies.p_pos_y, bodies.nb_bodies, MY_MPI_F, bodies_all.p_pos_y, bodies.nb_bodies, MY_MPI_F, 0, MPI_COMM_WORLD);
	  MPI_Gather(bodies.p_pos_z, bodies.nb_bodies, MY_MPI_F, bodies_all.p_pos_z, bodies.nb_bodies, MY_MPI_F, 0, MPI_COMM_WORLD);
	  MPI_Gather(bodies.p_fx, bodies.nb_bodies, MY_MPI_F, bodies_all.p_fx, bodies.nb_bodies, MY_MPI_F, 0, MPI_COMM_WORLD);
	  MPI_Gather(bodies.p_fy, bodies.nb_bodies, MY_MPI_F, bodies_all.p_fy, bodies.nb_bodies, MY_MPI_F, 0, MPI_COMM_WORLD);
	  MPI_Gather(bodies.p_fz, bodies.nb_bodies, MY_MPI_F, bodies_all.p_fz, bodies.nb_bodies, MY_MPI_F, 0, MPI_COMM_WORLD);
	  MPI_Gather(bodies.p_speed_vectors, 3 * bodies.nb_bodies, MY_MPI_F, bodies_all.p_speed_vectors, 3 * bodies.nb_bodies, MY_MPI_F, 0, MPI_COMM_WORLD);

	  if (my_rank == 0)
	    {
	      bodies_all.p_values = p_allvalues;

	      /* On réordonne les corps car le processus 0 possède les derniers du a la lecture. */
	      for (k = 0; k < bodies.nb_bodies; k++)
		{
		  bodies_all.p_pos_x[bodies_all.nb_bodies + k] = bodies.p_pos_x[k];
		  bodies_all.p_pos_y[bodies_all.nb_bodies + k] = bodies.p_pos_y[k];
		  bodies_all.p_pos_z[bodies_all.nb_bodies + k] = bodies.p_pos_z[k];
		  bodies_all.p_fx[bodies_all.nb_bodies + k] = bodies.p_fx[k];
		  bodies_all.p_fy[bodies_all.nb_bodies + k] = bodies.p_fy[k];
		  bodies_all.p_fz[bodies_all.nb_bodies + k] = bodies.p_fz[k];
		  bodies_all.p_speed_vectors[bodies_all.nb_bodies + k].dat[0] = bodies.p_speed_vectors[k].dat[0];
		  bodies_all.p_speed_vectors[bodies_all.nb_bodies + k].dat[1] = bodies.p_speed_vectors[k].dat[1];
		  bodies_all.p_speed_vectors[bodies_all.nb_bodies + k].dat[2] = bodies.p_speed_vectors[k].dat[2];
		}
	      bodies_all.p_pos_x += bodies.nb_bodies;
	      bodies_all.p_pos_y += bodies.nb_bodies;
	      bodies_all.p_pos_z += bodies.nb_bodies;
	      bodies_all.p_fx += bodies.nb_bodies;
	      bodies_all.p_fy += bodies.nb_bodies;
	      bodies_all.p_fz += bodies.nb_bodies;
	      bodies_all.p_speed_vectors += bodies.nb_bodies;

	      /*
	      for (k = 0; k < bodies_all.nb_bodies; k++)
	      	fprintf(stderr, "%f\t%f\t%f\n", bodies_all.p_pos_x[k], bodies_all.p_pos_y[k], bodies_all.p_pos_z[k]);
	      */

	      if (results_file == NULL)
		{
		  /* The 'results' filename has not been set yet: */
#define TMP_STRING_LENGTH 10
		  char step_number_string[TMP_STRING_LENGTH];
		  int  results_file_length = 0;
	
		  /* Find the relative filename in 'data_file': */
		  char *rel_data_file = strrchr(data_file, '/') + 1 ; /* find last '/' and go to the next character */
	
		  results_file_length = strlen(RESULTS_DIR) + 
		    strlen(RESULTS_FILE) + 
		    strlen(rel_data_file) +  
		    TMP_STRING_LENGTH + 
		    1 /* for '\0' */ ; 
	
		  results_file = (char *) FMB_malloc_with_check(results_file_length * sizeof(char));
	
		  strncpy(step_number_string, "", TMP_STRING_LENGTH);
		  sprintf(step_number_string, "_%lu", nb_steps ); 
	
		  strncpy(results_file, "", results_file_length);
		  strcpy(results_file, RESULTS_DIR); 
		  strcat(results_file, RESULTS_FILE); 
		  strcat(results_file, rel_data_file); 
		  strcat(results_file, step_number_string); 
#undef TMP_STRING_LENGTH
		}

	      /* Create directory RESULTS_DIR: */
	      {	struct stat filestat;
		if (stat (RESULTS_DIR, &filestat) != 0) {
		  /* The directory RESULTS_DIR does not exist, we create it: */
		  mkdir(RESULTS_DIR, 0700); 
		}
	      }
	      Direct_method_Dump_bodies(results_file, nb_steps, &bodies_all);

	      bodies_all.p_pos_x -= bodies.nb_bodies;
	      bodies_all.p_pos_y -= bodies.nb_bodies;
	      bodies_all.p_pos_z -= bodies.nb_bodies;
	      bodies_all.p_fx -= bodies.nb_bodies;
	      bodies_all.p_fy -=  bodies.nb_bodies;
	      bodies_all.p_fz -= bodies.nb_bodies;
	      bodies_all.p_speed_vectors -= bodies.nb_bodies;

	      FMB_free(results_file);
	      results_file= NULL ; 
	    }
	}
      
      
      /************************** Sum of forces and potential: ***************************/
      /* On a seulement besoin des Forces. */
      if (FMB_Info.sum){
	Direct_method_Sum(NULL, nb_steps, &bodies, total_potential_energy);
      }
      tnow+=FMB_Info.dt ; 
      nb_steps ++ ; 
    }  /* while ( tnow-FMB_Info.dt <= tend )  */
  /******************************************************************************************/
  /********************************** End of the simulation: ********************************/
  /******************************************************************************************/

  FMB_free(pjrecv_pos_x);
  FMB_free(pjrecv_pos_y);
  FMB_free(pjrecv_pos_z);
  FMB_free(pjrecv_fx);
  FMB_free(pjrecv_fy);
  FMB_free(pjrecv_fz);
  FMB_free(pjsend_fx);
  FMB_free(pjsend_fy);
  FMB_free(pjsend_fz);

  if (my_rank == 0)
    {
      
      if (FMB_Info.save)
	bodies_Free(&bodies_all);
      
      Direct_method_Terminate();
      Direct_method_Pj_Terminate();
    }
  else
    {
      Direct_method_Terminate2();
    }
  if (my_rank == 0)
    {
      /********************** Close FILE* and free memory before exiting: ***********************/
      if (argc == 3)
	if (fclose(f_output) == EOF)
	  perror("fclose(f_output)");
      FMB_free(data_file);
    }
  MPI_Finalize();

  /****************************************** EXIT ******************************************/
  exit(EXIT_SUCCESS);
}















/*********************************************************************************************
**********************************************************************************************

   usage

**********************************************************************************************
*********************************************************************************************/

void usage(){
  char mes[300] = "";
  
  sprintf(mes, "Usage : a.out [-h] %s [-o output_filename] --in[r]=data_filename %s \n"
	  , "[--soft value]"
	  , ""
	  );
  
  fprintf(stderr, "%s", mes);


  fprintf(stderr, "\nDescription of the short options:\n"); 
/*   fprintf(stderr, "\t -v \t\t\t Display the version.\n"); */
  fprintf(stderr, "\t -h \t\t\t Display this message.\n"); 
  fprintf(stderr, "\t -i 'level' \t\t Info display level (0, 1 or 2).\n");
  fprintf(stderr, "\t -o 'output_filename' \t Otherwise stdout.\n");


  fprintf(stderr, "\nDescription of the long options:\n");
  fprintf(stderr, "\t --in='filename' \t Input data filename.\n");

  fprintf(stderr, "\t --save \t\t Save position, mass, force and/or potential of all particles.\n");

  /* Unused in this code: */
  /*   fprintf(stderr, "\t --out='filename' \t Output data filename for '--save' option.\n"); */

  fprintf(stderr, "\t --sum  \t\t Compute and display the sum of the forces and/or potential over all particles.\n");
  fprintf(stderr, "\t --soft='value' \t Softening parameter.\n");
  fprintf(stderr, "\t --dt='value' \t\t Leapfrog integration timestep \n");
  fprintf(stderr, "\t --tend='value' \t Time to stop integration \n");

  /* We use only NEMO file format in this code: */
  /*   fprintf(stderr, "\t --it='value' \t\t input  data format ('fma' for FMB ASCII, 'fmb' for FMB binary, 'nemo').\n"); */
  /*   fprintf(stderr, "\t --ot='value' \t\t output data format ('fma' for FMB ASCII, 'fmah' for FMB ASCII human readable, 'fmb' for FMB binary, 'nemo').\n"); */

  fprintf(stderr, "\n");
  exit(EXIT_FAILURE);
}










/*********************************************************************************************
**********************************************************************************************

   parse_command

**********************************************************************************************
*********************************************************************************************/


/* Long option codes for 'val' field of struct option. 
 * Ascii codes 65 -> 90 ('A'->'Z') and 97 -> 122 ('a'->'z') 
 * are reserved for short options */
/* Same code as in main.c: */
#define LONGOPT_CODE_SOFT    14
#define LONGOPT_CODE_SAVE    24
#define LONGOPT_CODE_SUM     25
#define LONGOPT_CODE_IT 34
#define LONGOPT_CODE_OT 35
#define LONGOPT_CODE_IN 41
#define LONGOPT_CODE_OUT 43
#define LONGOPT_CODE_DT 49
#define LONGOPT_CODE_TEND 48 

int parse_command(int argc, 
		  char **argv,
		  char **p_data_file,
		  char **p_results_file){
  char options[]="hi:o:";
  int curr_opt;
  /*   opterr = 0; */

  /* Default values: */
  FMB_Info.dt = 0.001;
  FMB_Info.tend = 0.001 ; 
  FMB_Info.eps_soft = 0.0;

  
  struct option longopts[] = {
			      {"soft",
			       required_argument,
			       NULL, 
			       LONGOPT_CODE_SOFT},
			      {"dt",
			       required_argument,
			       NULL, 
			       LONGOPT_CODE_DT},
			      {"tend",
			       required_argument,
			       NULL, 
			       LONGOPT_CODE_TEND},
			      {"save",
			       no_argument,
                               NULL,
			       LONGOPT_CODE_SAVE},
			      {"sum",
			       no_argument,
			       NULL, 
			       LONGOPT_CODE_SUM},
			      {"it",
			       required_argument,
			       NULL, 
			       LONGOPT_CODE_IT},
			      {"ot",
			       required_argument,
			       NULL, 
			       LONGOPT_CODE_OT},
			      {"in",
			       required_argument,
			       NULL, 
			       LONGOPT_CODE_IN},
			      {"out",
			       required_argument,
			       NULL, 
			       LONGOPT_CODE_OUT},
 			      {0}}; /* last element of the array  */
  

  
  if (argc == 1){
    usage();
  }


  /* Default values: see direct_*/
  FMB_Info.eps_soft = 0.0;

  
  curr_opt=getopt_long(argc, argv, options, longopts, NULL);
  while(curr_opt != (int) EOF){

    switch(curr_opt){
    case 'h' : 
      usage();
      break;
    case 'i':
      FMB_IO_Info.info_display_level = atoi(optarg);
      if (FMB_IO_Info.info_display_level != 0 && 
	  FMB_IO_Info.info_display_level != 1 &&
	  FMB_IO_Info.info_display_level != 2 &&
	  FMB_IO_Info.info_display_level != 3){
	FMB_error("Wrong FMB_IO_Info.info_display_level value.\n");
      }
      break;
    case 'o':
      if ((f_output = fopen(optarg, "w")) == NULL){
	perror("fopen(\'output_filename\', \"w\")");
      }	  
      break;
    case '?' : 
      usage();
      break;

      
    case LONGOPT_CODE_SOFT:
      FMB_Info.eps_soft = (REAL_T) atof(optarg);
      break;
    case LONGOPT_CODE_DT:
      FMB_Info.dt =(REAL_T) atof(optarg);
      break;
    case LONGOPT_CODE_TEND:
      FMB_Info.tend =(REAL_T) atof(optarg) ; 
      break;
    case LONGOPT_CODE_SAVE:
      FMB_Info.save = TRUE;
      break;
    case LONGOPT_CODE_SUM:
      FMB_Info.sum  = TRUE;
      break;      
    case LONGOPT_CODE_IT:
      if (strcmp(optarg, "fma") == 0){
	FMB_IO_Info.input_format = FMB_ASCII_format;
      }
      else {
	if (strcmp(optarg, "fmah") == 0){
	  FMB_error("FMB_ASCII_human_format is only for \"output format\", not for \"input format\".\n");
	} 
	else {
	  if (strcmp(optarg, "fmb") == 0){
	    FMB_IO_Info.input_format = FMB_binary_format;
	  }
	  else {
	    if ((strcmp(optarg, "nemo") == 0) || (strcmp(optarg, "NEMO"))){
	      FMB_IO_Info.input_format = NEMO_format;
	    }
	    else {
	      FMB_error("Unknow format for --it option!\n");	    
	    }
	  }
	}
      }
      FMB_IO_Info.input_format_from_cmd_line = TRUE;
      break;
    case LONGOPT_CODE_OT:
      if (strcmp(optarg, "fma") == 0){
	FMB_IO_Info.output_format = FMB_ASCII_format;
      }
      else {
	if (strcmp(optarg, "fmah") == 0){
	  FMB_IO_Info.output_format = FMB_ASCII_human_format;
	}
	else {
	  if (strcmp(optarg, "fmb") == 0){
	    FMB_IO_Info.output_format = FMB_binary_format;
	  }
	  else {
	    if ((strcmp(optarg, "nemo") == 0) || (strcmp(optarg, "NEMO"))){
	      FMB_IO_Info.output_format = NEMO_format;
	    }
	    else {
	      FMB_error("Unknow format for --it option!\n");	    
	    }
	  }
	}
      }
      FMB_IO_Info.output_format_from_cmd_line = TRUE;
      break;
    case LONGOPT_CODE_IN:
      if (*p_data_file != NULL){ FMB_ERROR_BRIEF(); }
      *p_data_file = (char *) FMB_malloc_with_check((strlen(optarg) + 1 /* for '\0' */) * sizeof(char));
      strcpy(*p_data_file, optarg); 
      break;
    case LONGOPT_CODE_OUT:
      if (*p_results_file != NULL){ FMB_ERROR_BRIEF(); }
      *p_results_file = (char *) FMB_malloc_with_check((strlen(optarg) + 1 /* for '\0' */) * sizeof(char));
      strcpy(*p_results_file, optarg); 
      break;
    } /* switch */
    curr_opt=getopt_long(argc, argv, options, longopts, NULL);
  }

  /* Check that an input data filename has been provided: */
  if (*p_data_file == NULL){
    FMB_error("No 'input data filename' provided.\n");
  }

  return 0;
}




















