# Copyright © 2025 Apple Inc.

import argparse
import base64
import ipaddress
import json
import os
import shlex
import shutil
import sys
import tempfile
import threading
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from select import select
from subprocess import PIPE, Popen, run


@dataclass
class Host:
    rank: int
    ssh_hostname: str
    ips: list[str]


def positive_number(x):
    x = int(x)
    if x <= 0:
        raise ValueError("Number should be positive")
    return x


def log(verbose, *args, **kwargs):
    if not verbose:
        return
    print("\033[32m[INFO]", *args, "\033[0m", **kwargs)


def log_warning(*args, **kwargs):
    kwargs["file"] = sys.stderr
    print("\033[33m[WARN]", *args, "\033[0m", **kwargs)


def parse_hostfile(parser, hostfile):
    """Parse the json hostfile that contains both the hostnames to ssh into and
    the ips to communicate over when using the ring backend.

    Example:

        [
            {"ssh": "hostname1", "ips": ["123.123.123.1"]},
            {"ssh": "hostname2", "ips": ["123.123.123.2"]},
            ...
            {"ssh": "hostnameN", "ips": ["123.123.123.N"]},
        ]

    Args:
        hostfile (str): The path to the json file containing the host
            information
    """
    hostfile = Path(hostfile)
    if not hostfile.exists():
        parser.error(f"Hostfile {str(hostfile)} doesn't exist")

    try:
        hosts = []
        with open(hostfile) as f:
            for i, h in enumerate(json.load(f)):
                hosts.append(Host(i, h["ssh"], h.get("ips", [])))
        return hosts
    except e:
        parser.error(f"Failed to parse hostfile {str(hostfile)} ({str(e)})")


def parse_hostlist(parser, hostlist, repeats):
    hosts = []
    for i, h in enumerate(hostlist.split(",")):
        try:
            ipaddress.ip_address(h)
            ips = [h]
        except ValueError:
            ips = []
        for i in range(repeats):
            hosts.append(Host(i, h, ips))
    return hosts


def make_monitor_script(rank, hostfile, cwd, env, command, verbose):
    script = ""

    # Write the PID to a file so we can kill the process if needed
    script += "pidfile=$(mktemp)\n"
    script += "echo $$ >$pidfile\n"
    script += "echo $pidfile\n"

    # Change the working directory if one was requested. Otherwise attempt to
    # change to change to the current one but don't fail if it wasn't possible.
    d = cwd or os.getcwd()
    script += f"if [ -d {shlex.quote(d)} ]; then\n"
    script += f"    cd {shlex.quote(d)}\n"
    if cwd is not None:
        script += "else\n"
        script += f"    echo Failed to change directory to {shlex.quote(d)} 1>&2\n"
        script += f"    exit 1\n"
    script += "fi\n"

    # Add the environment variables that were given to us
    for e in env:
        key, *value = e.split("=", maxsplit=1)
        value = shlex.quote(value[0]) if len(value) > 0 else ""
        if not all(c.isalnum() or c == "_" for c in key):
            log_warning(f"'{e}' is an invalid environment variable so it is ignored")
            continue
        script += f"export {key}={value}\n"

    # Add the environment variables to enable the ring distributed backend
    if hostfile != "":
        script += "tmpfile=$(mktemp)\n"
        script += f"echo {shlex.quote(hostfile)} >$tmpfile\n"
        if verbose:
            script += "export MLX_RING_VERBOSE=1\n"
        script += "export MLX_HOSTFILE=$tmpfile\n"
        script += f"export MLX_RANK={rank}\n"
        script += "\n"

    # Replace the process with the script
    script += shlex.join(["exec", *command])
    script += "\n"

    return script


def launch_ring(parser, hosts, args, command):
    stop = False
    exit_codes = [None] * len(hosts)

    def node_thread(rank, host, hostfile):
        is_local = host == "127.0.0.1"
        script = make_monitor_script(
            rank, hostfile, args.cwd, args.env, command, args.verbose
        )
        script_b64 = base64.b64encode(script.encode()).decode()
        cmd = f'echo "{script_b64}" | base64 -d | /bin/bash'
        if not is_local:
            cmd = f"ssh {host} '{cmd}'"
        p = Popen(
            cmd,
            shell=True,
            stdout=PIPE,
            stderr=PIPE,
        )
        os.set_blocking(p.stdout.fileno(), False)
        os.set_blocking(p.stderr.fileno(), False)

        # Repeat the stdout and stderr to the local machine
        pidfile = ""
        while p.poll() is None:
            rlist, _, _ = select([p.stdout.fileno(), p.stderr.fileno()], [], [], 1.0)
            for fd in rlist:
                is_stdout = fd == p.stdout.fileno()
                outfile = sys.stdout if is_stdout else sys.stderr
                msg = os.read(fd, 8192).decode(errors="ignore")

                # Fetch the PID file first if we haven't already
                if pidfile == "":
                    pidfile, *msg = msg.split("\n", maxsplit=1)
                    msg = msg[0] if msg else ""

                outfile.write(msg)
            if stop:
                p.terminate()
                break
        p.wait()
        exit_codes[rank] = p.returncode

        # Kill the remote program if possible
        cmd = ""
        cmd += f"pid=$(cat {pidfile}); "
        cmd += "if ps -p $pid >/dev/null; then "
        cmd += "    kill $pid; "
        cmd += "    echo 1; "
        cmd += "else "
        cmd += "    echo 0; "
        cmd += "fi; "
        cmd += f"rm {pidfile}"
        if not is_local:
            cmd = f"ssh {host} '{cmd}'"
        c = run(cmd, check=True, shell=True, capture_output=True, text=True)
        if c.stdout.strip() == "1":
            log_warning(f"Node with rank {rank} was killed")
        elif p.returncode != 0:
            log_warning(f"Node with rank {rank} exited with code {p.returncode}")
        else:
            log(args.verbose, f"Node with rank {rank} completed")

    if all(len(h.ips) == 0 for h in hosts):
        parser.error(
            "The ring backend requires IPs to be provided instead of hostnames"
        )

    port = 5000
    ring_hosts = []
    for h in hosts:
        node = []
        for ip in h.ips:
            for i in range(args.connections_per_ip):
                node.append(f"{ip}:{port}")
                port += 1
        ring_hosts.append(node)
    hostfile = json.dumps(ring_hosts) if len(ring_hosts) > 1 else ""

    log(args.verbose, "Running", shlex.join(command))

    threads = []
    for i, h in enumerate(hosts):
        if i + 1 == len(hosts):
            time.sleep(1.0)
        t = threading.Thread(target=node_thread, args=(i, h.ssh_hostname, hostfile))
        t.start()
        threads.append(t)

    while not stop:
        time.sleep(1.0)
        if any(t.is_alive() for t in threads):
            for i, t in enumerate(threads):
                if not t.is_alive():
                    if exit_codes[i] != 0:
                        stop = True
                        break
        else:
            break
    for t in threads:
        t.join()


def launch_mpi(parser, hosts, args, command):
    mpirun = run(["which", "mpirun"], check=True, capture_output=True)
    mpirun = mpirun.stdout.strip().decode()

    # Homebrew libmpi doesn't work with anaconda python out of the box.
    # TODO: Check if we should do this with every mpirun
    if "homebrew" in mpirun:
        dyld = Path(mpirun).parent.parent / "lib"
        args.env = [f"DYLD_LIBRARY_PATH={str(dyld)}"] + args.env

    log(args.verbose, f"Using '{mpirun}'")
    with tempfile.NamedTemporaryFile(mode="w") as f:
        hosts = Counter((h.ssh_hostname for h in hosts))
        for h, n in hosts.items():
            print(f"{h} slots={n}", file=f)
        f.flush()

        cmd = [
            mpirun,
            "--output",
            ":raw",  # do not line buffer output
            "--hostfile",
            f.name,
            *(["-cwd", args.cwd] if args.cwd else []),
            *sum((["-x", e] for e in args.env), []),
            *sum([shlex.split(arg) for arg in args.mpi_arg], []),
            "--",
            *command,
        ]
        log(args.verbose, "Running", " ".join(cmd))
        try:
            run(cmd)
        except KeyboardInterrupt:
            pass


def main():
    parser = argparse.ArgumentParser(description="Launch an MLX distributed program")
    parser.add_argument(
        "--verbose", action="store_true", help="Print debug messages in stdout"
    )
    parser.add_argument(
        "--hosts", default="127.0.0.1", help="A comma separated list of hosts"
    )
    parser.add_argument(
        "--repeat-hosts",
        "-n",
        type=positive_number,
        default=1,
        help="Repeat each host a given number of times",
    )
    parser.add_argument("--hostfile", help="The file containing the hosts")
    parser.add_argument(
        "--backend",
        choices=["ring", "mpi"],
        default="ring",
        help="Which distributed backend to launch",
    )
    parser.add_argument(
        "--env",
        action="append",
        default=[],
        help="Set environment variables for the jobs",
    )
    parser.add_argument(
        "--mpi-arg",
        action="append",
        default=[],
        help="Arguments to pass directly to mpirun",
    )
    parser.add_argument(
        "--connections-per-ip",
        default=1,
        type=int,
        help="How many connections per ip to use for the ring backend",
    )
    parser.add_argument(
        "--cwd", help="Set the working directory on each node to the provided one"
    )
    args, rest = parser.parse_known_args()

    # Try to extract a list of hosts and corresponding ips
    if args.hostfile is not None:
        hosts = parse_hostfile(parser, args.hostfile)
    else:
        hosts = parse_hostlist(parser, args.hosts, args.repeat_hosts)

    # Check if the script is a file and convert it to a full path
    if (script := Path(rest[0])).exists():
        rest[0:1] = [sys.executable, str(script.resolve())]
    elif (command := shutil.which(rest[0])) is not None:
        rest[0] = command
    else:
        raise ValueError(f"Invalid script or command {rest[0]}")

    # Launch
    if args.backend == "ring":
        launch_ring(parser, hosts, args, rest)
    elif args.backend == "mpi":
        launch_mpi(parser, hosts, args, rest)


if __name__ == "__main__":
    main()
