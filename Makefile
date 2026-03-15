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
OPT_STARTS=1
OPT_PERTURB=0.05
EXPOSURE=1.0
RIDGE=0.0
AUTO_RIDGE=0
RIDGE_GRID=0,1e-9,1e-8,1e-7,1e-6,1e-5,1e-4
RIDGE_INNER_FOLDS=3
SEED=42
DECIMALS=2
SHOW_SELECTED_RIDGES=0

AUTO_RIDGE_FLAG=$(if $(filter 1 true yes on,$(AUTO_RIDGE)),--auto_ridge,)
SHOW_SELECTED_RIDGES_FLAG=$(if $(filter 1 true yes on,$(SHOW_SELECTED_RIDGES)),--show_selected_ridges,)

COMMON_ARGS=\
	--reflect_gz $(REFLECT) \
	--nikon_xlsx $(NIKON) \
	--cie_d65_csv $(D65) \
	--cie_cmf_csv $(CMF) \
	--opt_loss $(OPT_LOSS) \
	--opt_maxiter $(OPT_MAXITER) \
	--opt_xatol $(OPT_XATOL) \
	--opt_fatol $(OPT_FATOL) \
	--opt_starts $(OPT_STARTS) \
	--opt_perturb $(OPT_PERTURB) \
	--exposure $(EXPOSURE) \
	--ridge $(RIDGE) \
	--ridge_grid $(RIDGE_GRID) \
	--ridge_inner_folds $(RIDGE_INNER_FOLDS) \
	--seed $(SEED) \
	--decimals $(DECIMALS) \
	$(AUTO_RIDGE_FLAG) \
	$(SHOW_SELECTED_RIDGES_FLAG)

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

run-lsrp3:
	@$(MAKE) --no-print-directory run METHOD=ls-rp3

run-lsrp4:
	@$(MAKE) --no-print-directory run METHOD=ls-rp4

run-lsopt:
	@$(MAKE) --no-print-directory run METHOD=ls-opt OPT_LOSS=lab

run-lspopt:
	@$(MAKE) --no-print-directory run METHOD=ls-p-opt OPT_LOSS=lab

run-lsrpopt:
	@$(MAKE) --no-print-directory run METHOD=ls-rp-opt OPT_LOSS=lab

run-lsrp3opt:
	@$(MAKE) --no-print-directory run METHOD=ls-rp3-opt OPT_LOSS=lab

run-lsrp4opt:
	@$(MAKE) --no-print-directory run METHOD=ls-rp4-opt OPT_LOSS=lab

run-lsopt-de2000:
	@$(MAKE) --no-print-directory run METHOD=ls-opt OPT_LOSS=de2000

run-lspopt-de2000:
	@$(MAKE) --no-print-directory run METHOD=ls-p-opt OPT_LOSS=de2000

run-lsrpopt-de2000:
	@$(MAKE) --no-print-directory run METHOD=ls-rp-opt OPT_LOSS=de2000

run-lsrp3opt-de2000:
	@$(MAKE) --no-print-directory run METHOD=ls-rp3-opt OPT_LOSS=de2000

run-lsrp4opt-de2000:
	@$(MAKE) --no-print-directory run METHOD=ls-rp4-opt OPT_LOSS=de2000

clean:
	@rm -rf __pycache__ *.pyc