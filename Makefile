# definisce la macro CPPC
ifndef CPPC
	CPPC=nvcc
endif

ifdef TEST
	HDR=./data/test_header.txt
	DEM=./data/test_dem.txt
	SRC=./data/test_source.txt
	OUT=./test_output_OpenMP
	OUT_SERIAL=./test_output_serial
	STEPS=2
else
	HDR=./data/tessina_header.txt
	DEM=./data/tessina_dem.txt
	SRC=./data/tessina_source.txt
	OUT=./tessina_output_OpenMP
	OUT_SERIAL=./tessina_output_serial
	STEPS=4000
endif

# definisce le macro contenenti i nomei degli eseguibili
# e il numero di thread omp per la versione parallela
NT = 2 # numero di threads OpenMP
EXEC = sciddicaTomp
EXEC_SERIAL = sciddicaTserial

# definisce il target di default, utile in
# caso di invocazione di make senza parametri
default:all

# compila le versioni seriale e OpenMP
all:
#	$(CPPC) sciddicaT.cpp -o $(EXEC) -fopenmp -O3
# SUPPRESS: warning #2464-D: conversion from a string literal to "char *" is deprecated
	$(CPPC) sciddicaT.cu -o $(EXEC_SERIAL) -O3 -Xcudafe --diag_suppress=2464

# esegue la simulazione OpenMP
run_omp:
	OMP_NUM_THREADS=$(NT) ./$(EXEC) $(HDR) $(DEM) $(SRC) $(OUT) $(STEPS) &&  md5sum $(OUT) && cat $(HDR) $(OUT) > $(OUT).qgis && rm $(OUT)

# esegue la simulazione seriale 
run:
	./$(EXEC_SERIAL) $(HDR) $(DEM) $(SRC) $(OUT_SERIAL) $(STEPS) &&  md5sum $(OUT_SERIAL) && cat ./hash.txt ; echo && cat $(HDR) $(OUT_SERIAL) > $(OUT_SERIAL).qgis && rm $(OUT_SERIAL)

# elimina l'eseguibile, file oggetto e file di output
clean:
	rm -f $(EXEC) $(EXEC_SERIAL) *.o *output*

# elimina file oggetto e di output
wipe:
	rm -f *.o *output*

registers:
	$(CPPC) sciddicaT.cu --ptxas-options=-v -o $(EXEC_SERIAL) -O3 -diag-suppress 2464

roofline:
	nvprof --log-file log_oi.csv --csv --metrics flop_count_dp --metrics flop_count_sp --metrics flop_count_hp --metrics gld_transactions --metrics gst_transactions --metrics atomic_transactions --metrics local_load_transactions --metrics local_store_transactions --metrics shared_load_transactions --metrics shared_store_transactions --metrics l2_read_transactions --metrics l2_write_transactions --metrics dram_read_transactions --metrics dram_write_transactions ./$(EXEC_SERIAL) $(HDR) $(DEM) $(SRC) $(OUT_SERIAL) $(STEPS)

time:
	nvprof --log-file log_flop.csv --csv --print-gpu-summary $(EXEC_SERIAL)
