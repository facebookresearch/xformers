# LRA Benchmark

## CREDITS

Adapted from [the Nystromformer authors' repo](https://github.com/mlpen/Nystromformer) and [the original long-range-arena repo](https://github.com/google-research/long-range-arena)

## Initial setup

- install the needed dependencies, listed in the `requirements-lra.txt` file.
Can be done from the root of the repo with `pip install -r requirements-lra.txt`

- Run the install script, which can take a little while. `bash install.sh`

## Evaluating

To __run a LRA experiment__, run the following command in `LRA` root folder

```bash
python3 run_tasks.py --attention <attention> --task <task> --config <config_path> --world_size N
```

where `<attention>` can be set to any attention defined in the attached config file, and world size is the number of GPUs on which the benchmark should be run. The best models and log files will be saved `LRA/logs/` folder. More parts can be tested, not just attention, the model configuration can be freely altered in the config file (as long as the dimensions agree).

A __list and explanation of all the options__ is available via

```bash
python run_tasks.py --help
```

## Submitit

If you have access to a SLURM enabled cluster, you can __submit distributed jobs__ via this script.
```python3 run_with_submitit.py --attentions <your attention> --tasks <your task> --nodes 1 --config code/config.json --world_size <how many ranks> --checkpoint_dir <shared folder where to write the logs and best models>```

__Batch submissions__ are possible, for instance as follows

```bash
python3 batch_submit.py -c code/config.json -ck <your checkpoing and log path> -a lambda
```

Collecting all the results at once can be done with another small script, as follows:

```bash
python3 batch_fetch_results.py -ck <your checkpoing and log path>
```

This will synthetise the logs and show the score matrix for all the tasks and variants in this run.
