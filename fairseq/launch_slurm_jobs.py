""" Create lots of slurm scripts and launch them

High-level:
    1) Define a template script that will be called by sbatch with placeholders.
    2) Loop over the stuff I care about.
        a) Fill in the template.
        b) Write filled-in template to disk.
        c) Call sbatch on script.
        (TODO(Alex): abstract so I don't have to specify the loops everytime)

"""
import os
import subprocess


SLURM_SCRIPT = "tmp.sbatch"

SLURM_TEMPLATE = """
#!/bin/bash

#SBATCH --job-name={job_name}
#SBATCH --output={out_dir}/slurm.out
#SBATCH --error={out_dir}/slurm.err
#SBATCH --mail-user=wangalexc@fb.com
#SBATCH --mail-type=end

#SBATCH --partition=learnfair
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --time=2-00:00:00

./scripts/aw/gen_qg.sh {gen_mdl}
"""

def launch_job():
    with open(SLURM_FILE, 'w') as fh:
        fh.write(SLURM_TEMPLATE.format(**locals()).lstrip())
    print(bash('sbatch ' + SLURM_FILE))

def bash(cmd):
    process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    output = str(output)
    output = output[:-3]
    output = output.lstrip('b').strip('\'').strip('"')
    return output

run_dir = "."
date = "07-18-2019"
#gen_mdls = ["tfmr", "lstmsmall", "lstm", "lstmsmalltied"]
gen_mdls = [ "lstmsmall"]
for gen_mdl in gen_mdls:
    SLURM_FILE = os.path.join(run_dir, SLURM_SCRIPT)
    job_name = f"qg-{gen_mdl}"
    out_dir = f"/checkpoint/wangalexc/fairseq/{date}"
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    with open(SLURM_FILE, 'w') as fh:
        fh.write(SLURM_TEMPLATE.format(**locals()).lstrip())
    print(bash('sbatch ' + SLURM_FILE) + f" ({job_name})")

