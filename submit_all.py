import csv, subprocess

parameter_file_full_path = "./job_params.csv"

import ipdb; ipdb.set_trace()
with open(parameter_file_full_path, "rb") as csvfile:
    reader = csv.reader(csvfile)
    for job in reader:
        qsub_command = """qsub -v NUM_DRIVERS={0},LAMDA1={1},LAMDA_L1={2},LAMDA_L2={3} run.sub""".format(*job)

        #print qsub_command # Uncomment this line when testing to view the qsub command

        # Comment the following 3 lines when testing to prevent jobs from being submitted
        exit_status = subprocess.call(qsub_command, shell=True)
        if exit_status is 1:  # Check to make sure the job submitted
            print "Job {0} failed to submit".format(qsub_command)
print "Done submitting jobs!"
