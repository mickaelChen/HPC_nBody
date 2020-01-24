#---------------------------------------------------------------------------------------#
#
#                                         MAKEFILE
#
#---------------------------------------------------------------------------------------#


################################### High-level options ##################################

#---------------------------------------------------------------------------------------#
# NEMO
#---------------------------------------------------------------------------------------#
# To use NEMO file format (for input/output data files): */
#NO_NEMO=y#n

#---------------------------------------------------------------------------------------#
# ZLIB
#---------------------------------------------------------------------------------------#
# To read/write directly in "*.gz" files (for input/output data files): */
#NO_ZLIB=y#n

#---------------------------------------------------------------------------------------#
# Debugging
#---------------------------------------------------------------------------------------#
#DEBUGGING=y

#---------------------------------------------------------------------------------------#
# Profiling
#---------------------------------------------------------------------------------------#
#PROFILING=y#n

################################ End of high-level options ############################## 


#---------------------------------------------------------------------------------------#
# Compiler options:
#---------------------------------------------------------------------------------------#
CC=mpicc

# For the compiler:
LIB_DIR=
INC_DIR= 
LIBRARIES=-lm

WARNINGS=-Wall #-Winline

# Directories:
NBODY_DIRECT_SRC=src/
NBODY_DIRECT_INC=include/
INC_DIR += -I$(NBODY_DIRECT_INC)
BIN=bin/
OBJ=lib/

### Optimisation:
OPTIM_FLAGS = -O3 -mavx -ffast-math
OMP_FLAGS = -fopenmp -DNB_THREADS=4

#---------------------------------------------------------------------------------------#
# Set parameters 
#---------------------------------------------------------------------------------------#

### local path:
LOCAL_PATH=$(HOME)/local/
INC_DIR += -I$(LOCAL_PATH)include/
LIB_DIR += -L$(LOCAL_PATH)lib/

### NEMO:
ifneq ($(NO_NEMO), y)
 NEMO_PATH=/users/Enseignants/fortin/Public/HPC_fev2015/Projet/NEMO/nemo_cvs
 NEMO_INC=${NEMO_PATH}/include/
 NEMO_LIB=${NEMO_PATH}/lib/
 INC_DIR += -I$(NEMO_INC) -I$(NEMO_LIB) # we also include $(NEMO_LIB) for 'maxsizes.h' 
 LIB_DIR += -L$(NEMO_LIB)
 LIBRARIES+=-lnemo
else
 CODE_FLAGS+=-D_FMB_NO_NEMO_ 
endif 

### ZLIB:
ifneq ($(NO_ZLIB), y)
 LIBRARIES+= -lz
else
 CODE_FLAGS+=-D_FMB_NO_ZLIB_ 
endif 

CFLAGS = $(WARNINGS) 	\
	$(OPTIM_FLAGS) 	\
	$(GNU_FLAGS)

### gprof:
GPROF_FLAG = -pg
ifeq ($(PROFILING), y) 
# CFLAGS = $(GPROF_FLAG) # (overwrite CFLAGS)
 CFLAGS += $(GPROF_FLAG) 
endif 

### Debugging:
ifeq ($(DEBUGGING), y)
 CFLAGS = -g -Wall -fno-inline -D_REENTRANT $(GNU_FLAGS) #-pg #-std=gnu99 	
# CFLAGS := -g $(CFLAGS) 	
endif


#---------------------------------------------------------------------------------------#
# General options:
#---------------------------------------------------------------------------------------#
CODE_FLAGS += -DCODE_DIRECTORY=\"$(PWD)/\"      # Initialization  

# OS
OS_FLAGS = -D__LINUX__
CODE_FLAGS += $(OS_FLAGS)

#CODE_FLAGS += -D_USE_CONSTANT_INTERACTION_FACTOR_ 


#---------------------------------------------------------------------------------------#
# Objects files:
#---------------------------------------------------------------------------------------#
OBJECTS =  	$(OBJ)math_implementation.o	\
	   	$(OBJ)bodies.o 			\
		$(OBJ)memory_allocation.o 	\
		$(OBJ)IO.o 			\
		$(OBJ)direct_method.o		\
		$(OBJ)direct_computation.o


SRCS = $(subst $(OBJ),$(NBODY_DIRECT_SRC),$(OBJECTS:.o=.c))

#---------------------------------------------------------------------------------------#
# Files path:
#---------------------------------------------------------------------------------------#

FILE=cube_16384.nemo
NBPROC=4
NBTHREADS=4
PATHPROJET=.
DATA=/users/Enseignants/fortin/Public/HPC_fev2015/Projet/data/${FILE}

#---------------------------------------------------------------------------------------#
#--------------------------------------- Rules: ----------------------------------------#
#---------------------------------------------------------------------------------------#

all: init .remove.o $(OBJECTS) direct 
	@echo ""
	@echo "     *************** Compilation is over! ***************"
	@echo ""

exec: direct
	export OMP_NUM_THREADS=${NBTHREADS}
	mpirun -n ${NBPROC} -x OMP_NUM_THREADS -hostfile ${PATHPROJET}/hostfile ${PATHPROJET}/bin/NBODY_direct --in=${DATA} --tend=1. --dt=0.1 --soft=0.01 --sum

# What must be done before everything:
init:
	@echo ""
	@echo "Compiling with $(CC)= `which $(CC)` ..."
	@echo ""
ifeq ($(MAKEFILE_LIST),) #see .remove.o
	@echo "Error: MAKEFILE_LIST is undefined by make/gmake. Check that make/gmake version is >= 3.80, otherwise install it!"
	@exit 1
endif

direct: $(BIN)NBODY_direct

doc: 
	doxygen ./doc/Doxyfile 

# For .remove.o, we require the MAKEFILE_LIST feature that was added in GNU make 3.80.  
# Earlier versions do not have it.	
# MAKEFILE_LIST contains all Makefiles used/included.
#
# To force 'all' target when $(MAKEFILE_LIST) is modified: 
MY_MAKEFILE_LIST:=$(MAKEFILE_LIST) # To avoid .depend (included below)
.remove.o: $(MY_MAKEFILE_LIST)
#	@echo "MY_MAKEFILE_LIST: " $(MY_MAKEFILE_LIST)
#	@echo "first word in MY_MAKEFILE_LIST: " $(firstword $(MY_MAKEFILE_LIST))
	$(MAKE) -f $(firstword $(MY_MAKEFILE_LIST)) net.o 
	touch $@


#---------------------------------------------------------------------------------------#
#---------------------------------------- Bin: -----------------------------------------#
#---------------------------------------------------------------------------------------#

$(BIN)NBODY_direct: $(NBODY_DIRECT_SRC)main_direct_method.c $(OBJECTS)
	$(CC) $(CFLAGS) $(OMP_FLAGS) $(CODE_FLAGS) $(INC_DIR) $(LIB_DIR) -o $@ $+ $(LIBRARIES) 


#---------------------------------------------------------------------------------------#
#-------------------------------------- .o files: --------------------------------------#
#---------------------------------------------------------------------------------------#

$(OBJ)%.o: $(NBODY_DIRECT_SRC)%.c $(NBODY_DIRECT_INC)%.h
	$(CC) -c $(CFLAGS) $(OMP_FLAGS) $(CODE_FLAGS) $(INC_DIR) -o $@ $<

.depend: $(SRCS) 
	rm -f .depend 
	for i in $^ ; do \
		OBJ=$$( echo $$i | sed 's?$(NBODY_DIRECT_SRC)?$(OBJ)?;s?.c$$?.o?' ) ; \
		gcc  $(CFLAGS) $(CODE_FLAGS) $(INC_DIR) -MM -MT "$${OBJ}" $$i >> .depend 2>/dev/null ; \
	done 

-include .depend


#---------------------------------------------------------------------------------------#
#----------------------------------- Auxiliary targets: --------------------------------#
#---------------------------------------------------------------------------------------#

net.o: 
	@echo ""
	@echo "*** Remove object files: ***"
	rm -f ${OBJ}*.o ${OBJ}*.a ${OBJ}*.gcda ${OBJ}*.gcno *.o *.gcda *.gcno
	@echo ""

net.bin:
	@echo ""
	@echo "*** Remove binary files: ***"
	rm -f ${BIN}NBODY_direct
	@echo ""

clean: 
	@echo "" 
	@echo "*** Clean: ***"
	rm -f core *~ ${NBODY_DIRECT_SRC}*~ ${NBODY_DIRECT_INC}*~ 
	@echo ""

vclean: net.o net.bin clean 
	rm -f .remove.o .depend 


.PHONY: all init direct doc net.o net.bin clean vclean 

