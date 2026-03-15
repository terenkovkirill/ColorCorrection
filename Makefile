PY=python3

REFLECT=reflect_db.reflect.gz
NIKON=nikon.xlsx
D65=CIE_std_illum_D65.csv
CMF=CIE_xyz_1931_2deg.csv

METHOD=ls
OPT_LOSS=lab
OPT_MAXITER=3000
OPT_XATOL=1e-6
OPT_FATOL=1e-6
EXPOSURE=1.0
RIDGE=0.0
SEED=42
DECIMALS=2

COMMON_ARGS=\
	--reflect_gz $(REFLECT) \
	--nikon_xlsx $(NIKON) \
	--cie_d65_csv $(D65) \
	--cie_cmf_csv $(CMF) \
	--opt_loss $(OPT_LOSS) \
	--opt_maxiter $(OPT_MAXITER) \
	--opt_xatol $(OPT_XATOL) \
	--opt_fatol $(OPT_FATOL) \
	--exposure $(EXPOSURE) \
	--ridge $(RIDGE) \
	--seed $(SEED) \
	--decimals $(DECIMALS)

run:
	@$(PY) main.py $(COMMON_ARGS) --method $(METHOD)

run-all:
	@$(PY) main.py $(COMMON_ARGS) --all_methods

run-ls:
	@$(MAKE) --no-print-directory run METHOD=ls

run-lsp:
	@$(MAKE) --no-print-directory run METHOD=ls-p

run-lsrp:
	@$(MAKE) --no-print-directory run METHOD=ls-rp

run-lsopt:
	@$(MAKE) --no-print-directory run METHOD=ls-opt OPT_LOSS=lab

run-lspopt:
	@$(MAKE) --no-print-directory run METHOD=ls-p-opt OPT_LOSS=lab

run-lsrpopt:
	@$(MAKE) --no-print-directory run METHOD=ls-rp-opt OPT_LOSS=lab

run-lsopt-de2000:
	@$(MAKE) --no-print-directory run METHOD=ls-opt OPT_LOSS=de2000

run-lspopt-de2000:
	@$(MAKE) --no-print-directory run METHOD=ls-p-opt OPT_LOSS=de2000

run-lsrpopt-de2000:
	@$(MAKE) --no-print-directory run METHOD=ls-rp-opt OPT_LOSS=de2000

clean:
	@rm -rf __pycache__ *.pyc