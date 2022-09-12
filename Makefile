run=python -m

submmit:
	export PYTHONWARNINGS="ignore" && python3 src/make_submission.py $(check_point_name) $(submission_file_name)

train:
	export PYTHONWARNINGS="ignore" && python3 src/train.py $(exp_name) $(use_optim_params)

tune:
	export PYTHONWARNINGS="ignore" && python3 src/hp_tuning.py $(exp_name) $(accelerator)

dirs:
	mkdir src/data && mkdir src/data/check_points && mkdir src/data/processed && mkdir src/data/submissions

create_env_files:
	cp .env.example .env

lint:
	$(run) flake8 ./src
	$(run) pylint ./src

format:
	$(run) isort ./src
	$(run) black ./src
