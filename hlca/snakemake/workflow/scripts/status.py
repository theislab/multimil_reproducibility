#!/usr/bin/env python3
import argparse
import subprocess
import time

STATE_MAP = {
    "BOOT_FAIL": "failed",
    "CANCELLED": "failed",
    "COMPLETED": "success",
    "CONFIGURING": "running",
    "COMPLETING": "running",
    "DEADLINE": "failed",
    "FAILED": "failed",
    "NODE_FAIL": "failed",
    "OUT_OF_MEMORY": "failed",
    "PENDING": "running",
    "PREEMPTED": "failed",
    "RUNNING": "running",
    "RESIZING": "running",
    "SUSPENDED": "running",
    "TIMEOUT": "failed",
    "UNKNOWN": "running"
}


def fetch_status(batch_id):
    """fetch the status for the batch id"""
    sacct_args = ["sacct", "-j",  batch_id, "-o", "State", "--parsable2",
                  "--noheader"]

    try:
        output = subprocess.check_output(sacct_args).decode("utf-8").strip()
    except Exception:
        # If sacct fails for whatever reason, assume its temporary and return 'running'
        output = 'UNKNOWN'

    # Sometimes, sacct returns nothing, in which case we assume it is temporary
    # and return 'running'
    if not output:
        output = 'UNKNOWN'

    # The first output is the state of the overall job
    # See
    # https://stackoverflow.com/questions/52447602/slurm-sacct-shows-batch-and-extern-job-names
    # for details
    job_status = output.split("\n")[0]

    # If the job was cancelled manually, it will say by who, e.g "CANCELLED by 12345"
    # We only care that it was cancelled
    if job_status.startswith("CANCELLED by"):
        job_status = "CANCELLED"

    # Otherwise, return the status
    try:
        return STATE_MAP[job_status]
    except KeyError:
        raise NotImplementedError(f"Encountered unknown status '{job_status}' "
                                  f"when parsing output:\n'{output}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("batch_id", type=str)
    args = parser.parse_args()

    status = fetch_status(args.batch_id)
    print(status)



# #!/usr/bin/env python3
# import json
# import os
# import re
# import requests
# import subprocess as sp
# import shlex
# import sys
# import time
# import logging
# from CookieCutter import CookieCutter

# logger = logging.getLogger(__name__)

# STATUS_ATTEMPTS = 20
# SIDECAR_VARS = os.environ.get("SNAKEMAKE_CLUSTER_SIDECAR_VARS", None)
# DEBUG = bool(int(os.environ.get("SNAKEMAKE_SLURM_DEBUG", "0")))

# if DEBUG:
#     logging.basicConfig(level=logging.DEBUG)
#     logger.setLevel(logging.DEBUG)


# def get_status_direct(jobid):
#     """Get status directly from sacct/scontrol"""
#     cluster = CookieCutter.get_cluster_option()
#     for i in range(STATUS_ATTEMPTS):
#         try:
#             sacct_res = sp.check_output(shlex.split(f"sacct {cluster} -P -b -j {jobid} -n"))
#             res = {x.split("|")[0]: x.split("|")[1] for x in sacct_res.decode().strip().split("\n")}
#             break
#         except sp.CalledProcessError as e:
#             logger.error("sacct process error")
#             logger.error(e)
#         except IndexError as e:
#             logger.error(e)
#             pass
#         # Try getting job with scontrol instead in case sacct is misconfigured
#         try:
#             sctrl_res = sp.check_output(shlex.split(f"scontrol {cluster} -o show job {jobid}"))
#             m = re.search(r"JobState=(\w+)", sctrl_res.decode())
#             res = {jobid: m.group(1)}
#             break
#         except sp.CalledProcessError as e:
#             logger.error("scontrol process error")
#             logger.error(e)
#             if i >= STATUS_ATTEMPTS - 1:
#                 print("failed")
#                 exit(0)
#             else:
#                 time.sleep(1)

#     return res[jobid] or ""


# def get_status_sidecar(jobid):
#     """Get status from cluster sidecar"""
#     sidecar_vars = json.loads(SIDECAR_VARS)
#     url = "http://localhost:%d/job/status/%s" % (sidecar_vars["server_port"], jobid)
#     headers = {"Authorization": "Bearer %s" % sidecar_vars["server_secret"]}
#     try:
#         resp = requests.get(url, headers=headers)
#         if resp.status_code == 404:
#             return ""  # not found yet
#         logger.debug("sidecar returned: %s" % resp.json())
#         resp.raise_for_status()
#         return resp.json().get("status") or ""
#     except requests.exceptions.ConnectionError as e:
#         logger.warning("slurm-status.py: could not query side car: %s", e)
#         logger.info("slurm-status.py: falling back to direct query")
#         return get_status_direct(jobid)


# jobid = sys.argv[1]

# if SIDECAR_VARS:
#     logger.debug("slurm-status.py: querying sidecar")
#     status = get_status_sidecar(jobid)
# else:
#     logger.debug("slurm-status.py: direct query")
#     status = get_status_direct(jobid)

# logger.debug("job status: %s", repr(status))

# if status == "BOOT_FAIL":
#     print("failed")
# elif status == "OUT_OF_MEMORY":
#     print("failed")
# elif status.startswith("CANCELLED"):
#     print("failed")
# elif status == "COMPLETED":
#     print("success")
# elif status == "DEADLINE":
#     print("failed")
# elif status == "FAILED":
#     print("failed")
# elif status == "NODE_FAIL":
#     print("failed")
# elif status == "PREEMPTED":
#     print("failed")
# elif status == "TIMEOUT":
#     print("failed")
# elif status == "SUSPENDED":
#     print("running")
# else:
#     print("running")