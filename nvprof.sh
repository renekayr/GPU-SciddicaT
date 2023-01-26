#!/bin/sh

nvprof --log-file log_oi-flops.csv --csv --metrics flop_count_dp --metrics flop_count_sp ./sciddicaTserial ./data/tessina_header.txt ./data/tessina_dem.txt ./data/tessina_source.txt ./tessina_output_serial 4000
nvprof --log-file log_oi-dram.csv --csv --metrics dram_read_transactions --metrics dram_write_transactions ./sciddicaTserial ./data/tessina_header.txt ./data/tessina_dem.txt ./data/tessina_source.txt ./tessina_output_serial 4000
