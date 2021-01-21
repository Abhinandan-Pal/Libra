#!/usr/bin/env bash
##########################################
# experiment C: different number of CPUs #
##########################################

LIBRA=$1

#==========#
# symbolic #
#==========#

$1 tests/japanese/japanese.txt tests/japanese/20.py --domain symbolic --min_lower 0 --lower 1 --upper 0 --max_upper 20 --cpu 64 | tee tests/japanese/logs3/japanese-20-symbolic-64cpu.log
$1 tests/japanese/japanese.txt tests/japanese/20.py --domain symbolic --min_lower 0 --lower 1 --upper 0 --max_upper 20 --cpu 32 | tee tests/japanese/logs3/japanese-20-symbolic-32cpu.log
$1 tests/japanese/japanese.txt tests/japanese/20.py --domain symbolic --min_lower 0 --lower 1 --upper 0 --max_upper 20 --cpu 16 | tee tests/japanese/logs3/japanese-20-symbolic-16cpu.log
$1 tests/japanese/japanese.txt tests/japanese/20.py --domain symbolic --min_lower 0 --lower 1 --upper 0 --max_upper 20 --cpu 8 | tee tests/japanese/logs3/japanese-20-symbolic-8cpu.log
$1 tests/japanese/japanese.txt tests/japanese/20.py --domain symbolic --min_lower 0 --lower 1 --upper 0 --max_upper 20 --cpu 4 | tee tests/japanese/logs3/japanese-20-symbolic-4cpu.log