import argparse
import enum
import io
import subprocess

import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--kill", action="store_true", help="Kill the pending jobs")
args = parser.parse_args()


class Status(enum.StrEnum):
    PENDING = "Pending"


command = ["fun", "job", "list", "--my-jobs"]
output = subprocess.run(command, check=True, stdout=subprocess.PIPE)

table = pd.read_csv(io.StringIO(output.stdout.decode("utf8")), sep=r"\s+")

pending = table[table["STATUS"] == Status.PENDING]

print(pending)

if args.kill:
    for job_name in pending["JOB_NAME"]:
        command = f"fun job delete --name {job_name}"
        print(command)
        subprocess.run(command.split(" "), check=True)
print("done")
