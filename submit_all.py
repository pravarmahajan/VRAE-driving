import csv, subprocess
import time

#parameter_file_full_path = "./job_params_threshold.csv"
parameter_file_full_path = "./job_params.csv"

with open(parameter_file_full_path, "rb") as csvfile:
    reader = csv.reader(csvfile)
    for job in reader:
        qsub_command = """qsub -v NUM_DRIVERS={0},THRESHOLD={1} run.sub""".format(*job)

        #print qsub_command # Uncomment this line when testing to view the qsub command

        # Comment the following 3 lines when testing to prevent jobs from being submitted
        exit_status = subprocess.call(qsub_command, shell=True)
        if exit_status is 1:  # Check to make sure the job submitted
            print "Job {0} failed to submit".format(qsub_command)
print "Done submitting jobs!"
