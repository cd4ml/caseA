REPO:=git@github.com:davidgortega/mlrunner.git
BUCKET_PATH:=gs://dgodvctest1/data/
GOOGLE_APPLICATION_CREDENTIALS:="gsremote.json"

reset:
	rm -rf data
	rm -rf models
	rm -rf metrics
	rm -rf .dvc
	rm -rf *.dvc
	rm -rf .git
	
init: ## Init repo
	virtualenv .env
	. .env/bin/activate 
	pip install -r requirements.txt 
	pip install dvc[all]

	git init
	git remote add origin git@github.com:davidgortega/mlrunner.git
	
	dvc init
	export GOOGLE_APPLICATION_CREDENTIALS="gsremote.json"
	dvc remote add --local gs_remote gs://dgodvctest1/data/

	echo .env/ >> .gitignore
	echo __pycache__/ >> .gitignore

	mkdir data models metrics
	touch data/.gitkeep 
	touch models/.gitkeep 
	touch metrics/.gitkeep

pipeline: ## Add the pipelines
	# dvc pull 

	dvc run \
		-f preprocess.dvc \
		-d code/preprocess.py \
		-o data/train-images-idx3-ubyte.gz \
		-o data/train-labels-idx1-ubyte.gz \
		-o data/t10k-images-idx3-ubyte.gz \
		-o data/t10k-labels-idx1-ubyte.gz \
		python code/preprocess.py

	dvc run \
		-f train.dvc \
		-d code/train.py \
		-d data/train-images-idx3-ubyte.gz \
		-d data/train-labels-idx1-ubyte.gz \
		-d data/t10k-images-idx3-ubyte.gz \
		-d data/t10k-labels-idx1-ubyte.gz \
		-o models -M metrics/train.json \
		python code/train.py

	dvc run \
		-f eval.dvc \
		-d code/eval.py \
		-d models \
		-M metrics/eval.json \
		python code/eval.py

run:
	# dvc pipeline show --ascii eval.dvc
	dvc repro eval.dvc
	dvc metrics show

push:
	# dvc add
	export GOOGLE_APPLICATION_CREDENTIALS="gsremote.json"
	export
	dvc push -a -r gs_remote
	
	git add .
	git commit -m "$(COMMIT_MESSAGE)"
	git push

metrics:
	dvc metrics show -a
