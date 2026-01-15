# Copyright © 2025 Apple Inc.

import argparse
import base64
import json
import os
import shlex
import shutil
import sys
import tempfile
import threading
from collections import Counter
from itertools import chain
from pathlib import Path
from queue import Empty as QueueEmpty
from queue import Queue
from select import select
from subprocess import PIPE, Popen, run

import mlx.core as mx

from .common import log, log_warning, parse_hostfile, parse_hostlist, positive_number


class CommandProcess:
    @property
    def process(self):
        """Return the Popen object that refers to the current command."""
        raise NotImplementedError()

    @property
    def exit_status(self):
        """Return a tuple (returncode, killed) for the command. It should be
        (None, None) while the command is running normally."""
        raise NotImplementedError()

    def preprocess_output(self, data: str, is_stdout=False):
        """Preprocess the output of the command so that extra data can be
        capture or the format changed on the fly."""
        raise NotImplementedError()

    def terminate(self):
        """Terminate or return the exit code."""
        raise NotImplementedError()


class RemoteProcess(CommandProcess):
    def __init__(self, rank, host, python, cwd, files, env, command):
        is_local = host == "127.0.0.1"
        cmd = RemoteProcess.make_launch_script(rank, cwd, files, env, command, is_local)
        if not is_local:
            cmd = f"ssh -tt -o LogLevel=QUIET {host} {shlex.quote(cmd)}"

        self._host = host
        self._pidfile = None
        self._is_local = is_local
        self._process = Popen(
            cmd,
            shell=True,
            executable="/bin/bash",
            stdin=PIPE,
            stdout=PIPE,
            stderr=PIPE,
        )

        self._killed = False

    @property
    def process(self):
        return self._process

    @property
    def exit_status(self):
        return self._process.poll(), self._killed

    def preprocess_output(self, data, is_stdout=False):
        if self._pidfile is None:
            pidfile, *rest = data.split("\n", maxsplit=1)
            self._pidfile = pidfile
            return rest[0] if rest else ""

        return data

    def terminate(self):
        if self._killed:
            return

        self._process.terminate()
        self._process.wait()

        # Kill the remote program if possible
        cmd = RemoteProcess.make_kill_script(self._pidfile)
        if not self._is_local:
            cmd = f"ssh {self._host} {shlex.quote(cmd)}"
        c = run(
            cmd,
            check=True,
            shell=True,
            executable="/bin/bash",
            capture_output=True,
            text=True,
        )

        self._killed = c.stdout.strip() == "1"

    @staticmethod
    def make_launch_script(rank, cwd, files, env, command, is_local):
        script = ""

        # Disable echo
        if not is_local:
            script = "stty -echo; "

        # Write the PID to a file so we can kill the process if needed
        script += "pidfile=$(mktemp); "
        script += "echo $$ > $pidfile; "
        script += 'printf "%s\\n" $pidfile; '

        # Change the working directory if one was requested. Otherwise attempt to
        # change to the current one but don't fail if it wasn't possible.
        d = cwd or os.getcwd()
        script += f"if [[ -d {repr(d)} ]]; then "
        script += f"  cd {repr(d)}; "
        if cwd is not None:
            script += "else "
            script += f" echo 'Failed to change directory to' {repr(d)} >2; "
        script += "fi; "

        # Add the environment variables that were requested
        for e in env:
            key, *value = e.split("=", maxsplit=1)
            value = shlex.quote(value[0]) if len(value) > 0 else ""
            if not all(c.isalnum() or c == "_" for c in key):
                log_warning(
                    f"'{e}' is an invalid environment variable so it is ignored"
                )
                continue
            script += f"export {key}={value}; "

        # Make the temporary files
        for env_name, content in files.items():
            script += "fname=$(mktemp); "
            script += f"echo {shlex.quote(content)} >$fname; "
            script += f"export {env_name}=$fname; "

        # Finally add the rank
        script += f"export MLX_RANK={rank}; "

        # Replace the process with the script
        script += f"cmd=({' '.join(map(shlex.quote, command))}); "
        script += 'exec "${cmd[@]}"'

        return script

    @staticmethod
    def make_kill_script(pidfile):
        script = ""
        script += f"pid=$(cat {pidfile}); "
        script += "if ps -p $pid >/dev/null; then "
        script += "    kill $pid; "
        script += "    echo 1; "
        script += "else "
        script += "    echo 0; "
        script += "fi; "
        script += f"rm {pidfile}"

        return script


def _launch_with_io(command_class, arguments, verbose):
    stop = False
    exit_codes = [(None, None)] * len(arguments)

    def _thread_fn(rank, *args, **kwargs):
        stdin_queue = kwargs.pop("stdin_queue")
        stdout_queue = kwargs.pop("stdout_queue")
        stderr_queue = kwargs.pop("stderr_queue")

        command = command_class(rank, *args, **kwargs)
        p = command.process
        os.set_blocking(p.stdout.fileno(), False)
        os.set_blocking(p.stderr.fileno(), False)
        os.set_blocking(p.stdin.fileno(), False)

        to_read = [p.stdout.fileno(), p.stderr.fileno()]
        to_write = [p.stdin.fileno()]

        stdin_buffer = b""
        while p.poll() is None:
            try:
                stdin_buffer += stdin_queue.get_nowait()
            except QueueEmpty:
                pass
            rlist, wlist, _ = select(to_read, to_write, [], 1.0)
            for fd in rlist:
                is_stdout = fd == p.stdout.fileno()
                msg = os.read(fd, 8192).decode(errors="ignore")
                msg = command.preprocess_output(msg, is_stdout)
                if is_stdout:
                    stdout_queue.put(msg.encode())
                else:
                    stderr_queue.put(msg.encode())
            for fd in wlist:
                if len(stdin_buffer) > 0:
                    n = os.write(fd, stdin_buffer)
                    stdin_buffer = stdin_buffer[n:]
            if stop:
                command.terminate()
                break
        exit_codes[rank] = command.exit_status

        if exit_codes[rank][1]:
            log_warning(f"Node with rank {rank} was killed")
        elif exit_codes[rank][0] != 0:
            log_warning(f"Node with rank {rank} exited with code {exit_codes[rank][0]}")
        else:
            log(verbose, f"Node with rank {rank} completed")

    stdin_queues = []
    stdout_queues = []
    stderr_queues = []
    threads = []
    for i, (args, kwargs) in enumerate(arguments):
        stdin_queues.append(Queue())
        stdout_queues.append(Queue())
        stderr_queues.append(Queue())
        t = threading.Thread(
            target=_thread_fn,
            args=args,
            kwargs=kwargs
            | {
                "stdin_queue": stdin_queues[-1],
                "stdout_queue": stdout_queues[-1],
                "stderr_queue": stderr_queues[-1],
            },
        )
        t.start()
        threads.append(t)

    os.set_blocking(sys.stdin.fileno(), False)
    os.set_blocking(sys.stdout.fileno(), True)
    os.set_blocking(sys.stderr.fileno(), True)
    while not stop or any(not q.empty() for q in chain(stdout_queues, stderr_queues)):
        # Broadcast user input to the jobs
        rlist, _, _ = select([sys.stdin.fileno()], [], [], 0.1)
        for fd in rlist:
            stdin_buffer = os.read(fd, 8192)
            for q in stdin_queues:
                q.put(stdin_buffer)

        # Gather job output
        for q in stdout_queues:
            try:
                while not q.empty():
                    sys.stdout.buffer.write(q.get_nowait())
            except QueueEmpty:
                pass
        for q in stderr_queues:
            try:
                while not q.empty():
                    sys.stderr.buffer.write(q.get_nowait())
            except QueueEmpty:
                pass
        sys.stdout.buffer.flush()
        sys.stderr.buffer.flush()

        # Check if all are running and terminate otherwise
        if any(t.is_alive() for t in threads):
            for i, t in enumerate(threads):
                if not t.is_alive():
                    if exit_codes[i][0] != 0:
                        stop = True
                        break
        else:
            break

    # Wait for the jobs to finish
    for t in threads:
        t.join()

    # Process any remaining outputs
    for q in stdout_queues:
        while not q.empty():
            sys.stdout.buffer.write(q.get())
    for q in stderr_queues:
        while not q.empty():
            sys.stderr.buffer.write(q.get())
    sys.stdout.buffer.flush()
    sys.stderr.buffer.flush()


def launch_ring(parser, hosts, args, command):
    if any(len(h.ips) == 0 for h in hosts):
        parser.error(
            "The ring backend requires IPs to be provided instead of hostnames"
        )

    port = args.starting_port
    ring_hosts = []
    for h in hosts:
        node = []
        for ip in h.ips:
            for i in range(args.connections_per_ip):
                node.append(f"{ip}:{port}")
                port += 1
        ring_hosts.append(node)
    hostfile = json.dumps(ring_hosts) if len(ring_hosts) > 1 else ""

    files = {"MLX_HOSTFILE": hostfile}
    env = args.env
    if args.verbose:
        env.append("MLX_RING_VERBOSE=1")
    cwd = args.cwd

    log(args.verbose, "Running", shlex.join(command))

    _launch_with_io(
        RemoteProcess,
        [
            ((rank, h.ssh_hostname, args.python, cwd, files, env, command), {})
            for rank, h in enumerate(hosts)
        ],
        args.verbose,
    )


def launch_nccl(parser, hosts, args, command):
    if not hosts[0].ips:
        raise ValueError("Rank 0 should have an IP reachable from all other ranks")

    master_host = hosts[0].ips[0]
    master_port = args.nccl_port
    world_size = len(hosts)

    env = args.env
    cwd = args.cwd
    if args.verbose:
        env.append("NCCL_DEBUG=INFO")
    env.append(f"NCCL_HOST_IP={master_host}")
    env.append(f"NCCL_PORT={master_port}")
    env.append(f"MLX_WORLD_SIZE={world_size}")

    log(args.verbose, "Running", shlex.join(command))

    _launch_with_io(
        RemoteProcess,
        [
            (
                (
                    rank,
                    h.ssh_hostname,
                    args.python,
                    cwd,
                    {},
                    env + [f"CUDA_VISIBLE_DEVICES={rank % args.repeat_hosts}"],
                    command,
                ),
                {},
            )
            for rank, h in enumerate(hosts)
        ],
        args.verbose,
    )


def launch_jaccl(parser, hosts, args, command):
    if not hosts[0].ips:
        raise ValueError("Rank 0 should have an IP reachable from all other ranks")

    have_rdmas = all(len(h.rdma) == len(hosts) for h in hosts)
    have_nulls = all(h.rdma[i] is None for i, h in enumerate(hosts))
    if not have_rdmas or not have_nulls:
        raise ValueError("Malformed hostfile for jaccl backend")

    coordinator = hosts[0].ips[0]
    env = args.env
    cwd = args.cwd
    env.append(f"MLX_JACCL_COORDINATOR={coordinator}:{args.starting_port}")
    files = {"MLX_IBV_DEVICES": json.dumps([h.rdma for h in hosts])}

    log(args.verbose, "Running", shlex.join(command))

    _launch_with_io(
        RemoteProcess,
        [
            ((rank, h.ssh_hostname, args.python, cwd, files, env, command), {})
            for rank, h in enumerate(hosts)
        ],
        args.verbose,
    )


def get_mpi_libname():
    try:
        ompi_info = run(["which", "ompi_info"], check=True, capture_output=True)
        ompi_info = ompi_info.stdout.strip().decode()

        if platform.system() == "Darwin":
            otool_output = run(
                ["otool", "-L", ompi_info], check=True, capture_output=True
            )
        else:
            otool_output = run(["ldd", ompi_info], check=True, capture_output=True)
        otool_output = otool_output.stdout.decode()

        # StopIteration if not found
        libmpi_line = next(
            filter(lambda line: "libmpi" in line, otool_output.splitlines())
        )
        return libmpi_line.strip().split()[0].removeprefix("@rpath/")
    except:
        return None


def launch_mpi(parser, hosts, args, command):
    mpirun = run(["which", "mpirun"], check=True, capture_output=True)
    mpirun = mpirun.stdout.strip().decode()

    # Compatibility with homebrew and pip installs
    mpi_libname = get_mpi_libname()
    if mpi_libname is not None:
        dyld = Path(mpirun).parent.parent / "lib"
        args.env = [
            f"DYLD_LIBRARY_PATH={str(dyld)}",
            f"MLX_MPI_LIBNAME={mpi_libname}",
        ] + args.env

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
        "--print-python",
        action="store_true",
        help="Print the path to the current python executable and exit",
    )
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
        choices=["ring", "mpi", "nccl", "jaccl"],
        default="nccl" if mx.cuda.is_available() else "ring",
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
        "--starting-port",
        "-p",
        type=int,
        default=32323,
        help="For the ring backend listen on this port increasing by 1 per rank and IP",
    )
    parser.add_argument(
        "--cwd", help="Set the working directory on each node to the provided one"
    )
    parser.add_argument(
        "--nccl-port",
        type=int,
        default=12345,
        help="The port to use for the NCCL communication (only for nccl backend)",
    )
    parser.add_argument(
        "--no-verify-script",
        action="store_false",
        dest="verify_script",
        help="Do not verify that the script exists",
    )
    parser.add_argument(
        "--python", default=sys.executable, help="Use this python on the remote hosts"
    )

    args, rest = parser.parse_known_args()

    if args.print_python:
        print(args.python)
        return

    if len(rest) == 0:
        parser.error("No script is provided")
    if rest[0] == "--":
        rest.pop(0)

    # Try to extract a list of hosts and corresponding ips
    if args.hostfile is not None:
        hosts = parse_hostfile(parser, args.hostfile)
    else:
        hosts = parse_hostlist(parser, args.hosts, args.repeat_hosts)

    # Check if the script is a file and convert it to a full path
    if (script := Path(rest[0])).exists() and script.is_file():
        rest[0:1] = [args.python, str(script.resolve())]
    elif (command := shutil.which(rest[0])) is not None:
        rest[0] = command
    elif args.verify_script:
        raise ValueError(f"Invalid script or command {rest[0]}")

    # Launch
    if args.backend == "ring":
        launch_ring(parser, hosts, args, rest)
    if args.backend == "mpi":
        launch_mpi(parser, hosts, args, rest)
    if args.backend == "nccl":
        launch_nccl(parser, hosts, args, rest)
    if args.backend == "jaccl":
        launch_jaccl(parser, hosts, args, rest)
