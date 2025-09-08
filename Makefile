.PHONY: build run run-all test

build:
\tdocker build -t egr-cmb .

run-all:
\tdocker run --rm \
\t  -v "$(PWD)/data:/work/data" \
\t  -v "$(PWD)/out_cmb:/work/out_cmb" \
\t  egr-cmb egr-cmb-run-all \
\t    --data /work/data/COM_PowerSpect_CMB-TT-full_R3.01.txt \
\t    --outdir /work/out_cmb

run:
\tdocker run --rm \
\t  -v "$(PWD)/data:/work/data" \
\t  -v "$(PWD)/out_cmb:/work/out_cmb" \
\t  egr-cmb egr-cmb \
\t    --data /work/data/COM_PowerSpect_CMB-TT-full_R3.01.txt \
\t    --outdir /work/out_cmb --plot --ar1 --ar1-calibrate --band-scatter --mcmc

test:
\tpytest -q
