#!/bin/bash
#SBATCH -p rocky_gpu                       # Partition (queue)
#SBATCH --job-name=lda_huge                # What to show in squeue
#SBATCH --mail-type=END,FAIL               # When to send mail (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=szhang1@g.harvard.edu  # Where to send mail
#SBATCH --mem=495gb                        # How much memory to request
#SBATCH --nodes=1                          # Minimum number of nodes allocated for this job
#SBATCH --cpus-per-task=64                 # Number of cores requested
#SBATCH --time=02-00:00:00                 # How long to run (days-hrs:min:sec)
#SBATCH --output=topic_lda_huge-%j.txt     # Where to save the output


echo -n "STARTED JOB AT: "
date
echo ""

module load Anaconda3
python topic_lda_huge.py

echo -n "FINISHED JOB AT: "
date
echo ""
