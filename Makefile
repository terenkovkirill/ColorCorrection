PY=python3

REFLECT=reflect_db.reflect.gz
NIKON=nikon.xlsx
D65=CIE_std_illum_D65.csv
CMF=CIE_xyz_1931_2deg.csv

METHOD=ls
EXPOSURE=1.0
RIDGE=0.0
SEED=42

run:
	$(PY) main.py \
	  --reflect_gz $(REFLECT) \
	  --nikon_xlsx $(NIKON) \
	  --cie_d65_csv $(D65) \
	  --cie_cmf_csv $(CMF) \
	  --method $(METHOD) \
	  --exposure $(EXPOSURE) \
	  --ridge $(RIDGE) \
	  --seed $(SEED)

run-ls:
	$(MAKE) run METHOD=ls

run-lsp:
	$(MAKE) run METHOD=ls-p

run-lsrp:
	$(MAKE) run METHOD=ls-rp

clean:
	rm -rf __pycache__ *.pyc