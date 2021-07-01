
# LRA Benchmark

## Credits

Adapted from https://github.com/mlpen/Nystromformer

## Initial setup

- install the needed dependencies, listed in the `requirements-lra.txt` file. Can be done from the root of the repo with `pip install -r requirements-lra.txt`
- Run the install script, which can take a little while. `bash install.sh`

## Evaluating

To run a LRA experiment, run the following command in `LRA` root folder

```bash
python3 run_tasks.py --attention <attention> --task <task> --config <config_path> --world_size N
```

where `<attention>` can be set to any attention defined in the attached config file, and world size is the number of GPUs on which the benchmark should be run. The best models and log files will be saved `LRA/logs/` folder. More parts can be tested, not just attention, the model configuration can be freely altered in the config file (as long as the dimensions agree).

A list and explanation of all the options is available through

```bash
python run_tasks.py --help
```

## Submitit

A single job can be submitted to a SLURM cluster as follows
```python3 run_with_submitit.py --attentions <your attention> --tasks <your task> --nodes 1 --config code/config.json --world_size <how many ranks> --checkpoint_dir <shared folder where to write the logs and best models>```

Batch submissions are possible, for instance as follows

```bash
python3 batch_submit.py -c code/config.json -ck <your checkpoing and log path> -a lambda
```
