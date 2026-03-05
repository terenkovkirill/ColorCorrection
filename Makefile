PY=python3

REFLECT=reflect_db.reflect.gz
NIKON=nikon.xlsx
D65=CIE_std_illum_D65.csv
CMF=CIE_xyz_1931_2deg.csv

EXPOSURE=1.0
RIDGE=0.0
SEED=42

run:
	$(PY) main.py \
	  --reflect_gz $(REFLECT) \
	  --nikon_xlsx $(NIKON) \
	  --cie_d65_csv $(D65) \
	  --cie_cmf_csv $(CMF) \
	  --exposure $(EXPOSURE) \
	  --ridge $(RIDGE) \
	  --seed $(SEED)

exp:
	$(PY) main.py \
	  --reflect_gz $(REFLECT) \
	  --nikon_xlsx $(NIKON) \
	  --cie_d65_csv $(D65) \
	  --cie_cmf_csv $(CMF) \
	  --exposure $(EXPOSURE) \
	  --ridge $(RIDGE) \
	  --seed $(SEED)

clean:
	rm -rf __pycache__ *.pyc