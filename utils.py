#### Those are the functions that make your life easier while coding. Enjoy :)

import subprocess
import os
import threading

def call_subprocess(command: str, params: list, outfile=None, chdir=None):
    # When we want to pipe the result to a text file, then we have to use the outfile option.
    # If the program asks you to specify the output with -o etc. then leave the outfile param None
    if outfile:
        stdout_buffer = open(outfile, "wb", buffering=0)
    else:
        stdout_buffer = subprocess.PIPE

    popen_args = dict(
        args=[command] + params,
        preexec_fn=os.setsid,
        stdin=subprocess.DEVNULL,
        stdout=stdout_buffer,
        stderr=subprocess.PIPE,
        bufsize=0,
        cwd=chdir,
    )
    process = subprocess.Popen(**popen_args)
    stdout, stderr = process.communicate()

    return_code = process.returncode

    if return_code != 0:
        full_command = " ".join(popen_args['args'])
        raise Exception(full_command, stdout, stderr)
    retstdout = stdout.decode() if stdout is not None else None
    return return_code, retstdout



def load_json(json_file):
    import json
    json_file = str(json_file)
    with open(json_file) as fh:
        return json.load(fh)


def dump_json(jdict, jfile):
    import json
    with open(jfile, "w") as jfh:
        json.dump(jdict, jfh, indent=4)


class CustomThreading(threading.Thread):
    """
    This edited threading class is used to catch exceptions
    in the child thread and raise it in the main thread
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.keyword_args = kwargs

    def run(self):

        # Variable that stores the exception, if raised by someFunction
        self.exc = None
        try:
            ## starts running the targeted process
            self.keyword_args["target"]()
        except BaseException as e:
            self.exc = e

    def join(self):
        threading.Thread.join(self)
        # Since join() returns in caller thread
        # we re-raise the caught exception
        # if any was caught
        if self.exc:
            raise self.exc