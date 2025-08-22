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
from queue import Empty as QueueEmpty
from queue import Queue
from select import select
from subprocess import PIPE, Popen, run
from typing import Optional

import mlx.core as mx


@dataclass
class Host:
    rank: int
    ssh_hostname: str
    ips: list[str]


@dataclass
class ThunderboltPort:
    iface: str
    uuid: str
    connected_to: Optional[str]


@dataclass
class ThunderboltHost:
    name: str
    ports: list[ThunderboltPort]


def parse_hardware_ports(ports_string):
    ports = {}
    port_name = None
    for l in ports_string.decode("utf-8").split("\n"):
        if l.startswith("Hardware Port:"):
            port_name = l.strip()[15:]
        elif l.startswith("Device:"):
            ports[port_name] = l.strip()[8:]
            port_name = None
    return ports


def get_num_nvidia_gpus():
    result = run(["nvidia-smi", "-L"], capture_output=True, text=True, check=True)
    return len(result.stdout.strip().split("\n"))


def extract_rings(hosts, index):
    def usable_port(i, j, used_ports):
        return (i, j) not in used_ports and hosts[i].ports[j].connected_to is not None

    def dfs(start_node, node, path, visited, used_ports):
        path.append(node)
        visited.add(node)
        for j, p in enumerate(hosts[node].ports):
            if not usable_port(node, j, used_ports):
                continue
            next_node, _ = index[p.connected_to]
            if next_node == start_node:
                yield path[:]
            if next_node not in visited:
                yield from dfs(start_node, next_node, path, visited, used_ports)
        path.pop()
        visited.remove(node)

    # Concretize maps the found cycle to real thunderbolt ports. It also adds
    # those ports to the used set so next cycles can't use them again.
    def concretize(cycle, used_ports):
        concrete_path = []
        for n1, n2 in zip(cycle, cycle[1:] + cycle[:1]):
            for j, p in enumerate(hosts[n1].ports):
                if not usable_port(n1, j, used_ports):
                    continue
                n2_hat, nj = index[p.connected_to]
                if n2 == n2_hat:
                    concrete_path.append(((n1, j), (n2, nj)))
                    used_ports.add((n1, j))
                    used_ports.add((n2, nj))
                    break
            if concrete_path[-1][0][0] != n1:
                raise RuntimeError("Couldn't concretize the cycle")
        return concrete_path

    # Normalize tries to ensure that the cycles have the same direction so we can
    # use them together. We achieve this by selecting the direction such that
    # the smallest rank hosts connect to larger rank hosts.
    def normalize(path):
        small_to_large = sum(1 for p in path if p[0][0] < p[1][0])
        if small_to_large > len(path) - small_to_large:
            return path
        else:
            return [(p[1], p[0]) for p in path]

    rings = []
    used_ports = set()
    for start_node in range(len(hosts)):
        while True:
            ring = []
            for r in dfs(start_node, start_node, [], set(), used_ports):
                if len(r) > len(ring):
                    ring = r
                # Break early since we won't find a bigger ring no matter what
                if len(ring) == len(hosts):
                    break
            if not ring:
                break
            try:
                rings.append(normalize(concretize(ring, used_ports)))
            except RuntimeError:
                if len(rings) > 0:
                    return rings
                raise

    return rings


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


def log_error(*args, **kwargs):
    kwargs["file"] = sys.stderr
    print("\033[31m[ERROR]", *args, "\033[0m", **kwargs)


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
    except Exception as e:
        parser.error(f"Failed to parse hostfile {str(hostfile)} ({str(e)})")


def parse_hostlist(parser, hostlist, repeats):
    hosts = []
    for i, h in enumerate(hostlist.split(",")):
        if h == "":
            raise ValueError("Hostname cannot be empty")
        try:
            ipaddress.ip_address(h)
            ips = [h]
        except ValueError:
            ips = []
        for i in range(repeats):
            hosts.append(Host(i, h, ips))
    return hosts


def make_monitor_script(rank, hostfile, cwd, env, command, verbose):
    # Imports that are used throughout
    script = ""
    script += "import os\n"
    script += "import sys\n"
    script += "import tempfile\n"
    script += "from pathlib import Path\n"

    # Write the PID to a file so we can kill the process if needed
    script += "_, pidfile = tempfile.mkstemp() \n"
    script += "open(pidfile, 'w').write(str(os.getpid()))\n"
    script += "print(pidfile, flush=True)\n"

    # Change the working directory if one was requested. Otherwise attempt to
    # change to the current one but don't fail if it wasn't possible.
    d = cwd or os.getcwd()
    script += f"if Path({repr(d)}).exists():\n"
    script += f"    os.chdir({repr(d)})\n"
    if cwd is not None:
        script += "else:\n"
        script += (
            f"    print('Failed to change directory to', {repr(d)}, file=sys.stderr)\n"
        )
        script += f"    sys.exit(1)\n"

    # Add the environment variables that were given to us
    script += "env = dict(os.environ)\n"
    for e in env:
        key, *value = e.split("=", maxsplit=1)
        value = shlex.quote(value[0]) if len(value) > 0 else ""
        if not all(c.isalnum() or c == "_" for c in key):
            log_warning(f"'{e}' is an invalid environment variable so it is ignored")
            continue
        script += f"env[{repr(key)}] = {repr(value)}\n"

    # Add the environment variables to enable the ring distributed backend
    if hostfile != "":
        script += "_, hostfile = tempfile.mkstemp()\n"
        script += "with open(hostfile, 'w') as f:\n"
        script += f"    f.write({repr(hostfile)})\n"
        if verbose:
            script += "env['MLX_RING_VERBOSE'] = '1'\n"
        script += "env['MLX_HOSTFILE'] = hostfile\n"
        script += f"env['MLX_RANK'] = '{rank}'\n"
        script += "\n"

    # Replace the process with the script
    script += f"command = [{','.join(map(repr, command))}]\n"
    script += "os.execve(command[0], command, env)\n"

    return script


def launch_ring(parser, hosts, args, command):
    stop = False
    exit_codes = [None] * len(hosts)

    def node_thread(rank, host, hostfile, input_queue):
        is_local = host == "127.0.0.1"
        script = make_monitor_script(
            rank, hostfile, args.cwd, args.env, command, args.verbose
        )
        script_b64 = base64.b64encode(script.encode()).decode()
        cmd = f'{sys.executable} -c "import base64; exec(base64.b64decode(\\"{script_b64}\\"));"'
        if not is_local:
            cmd = f"ssh {host} '{cmd}'"
        p = Popen(
            cmd,
            shell=True,
            stdin=PIPE,
            stdout=PIPE,
            stderr=PIPE,
        )
        os.set_blocking(p.stdout.fileno(), False)
        os.set_blocking(p.stderr.fileno(), False)
        os.set_blocking(p.stdin.fileno(), False)

        # Repeat the stdout and stderr to the local machine
        to_read = [p.stdout.fileno(), p.stderr.fileno()]
        to_write = [p.stdin.fileno(), sys.stdout.fileno(), sys.stderr.fileno()]
        pidfile = ""
        stdin_buffer = b""
        stdout_buffer = b""
        stderr_buffer = b""
        while p.poll() is None:
            try:
                stdin_buffer += input_queue.get_nowait()
            except QueueEmpty:
                pass
            rlist, wlist, _ = select(to_read, to_write, [], 1.0)
            for fd in rlist:
                msg = os.read(fd, 8192).decode(errors="ignore")

                # Fetch the PID file first if we haven't already
                if pidfile == "":
                    pidfile, *msg = msg.split("\n", maxsplit=1)
                    msg = msg[0] if msg else ""

                is_stdout = fd == p.stdout.fileno()
                if is_stdout:
                    stdout_buffer += msg.encode()
                else:
                    stderr_buffer += msg.encode()
            for fd in wlist:
                if fd == p.stdin.fileno() and len(stdin_buffer) > 0:
                    n = os.write(fd, stdin_buffer)
                    stdin_buffer = stdin_buffer[n:]
                elif fd == sys.stdout.fileno() and len(stdout_buffer) > 0:
                    n = os.write(fd, stdout_buffer)
                    stdout_buffer = stdout_buffer[n:]
                elif fd == sys.stderr.fileno() and len(stderr_buffer) > 0:
                    n = os.write(fd, stderr_buffer)
                    stderr_buffer = stderr_buffer[n:]
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

    log(args.verbose, "Running", shlex.join(command))

    input_queues = []
    threads = []
    for i, h in enumerate(hosts):
        if i + 1 == len(hosts):
            time.sleep(1.0)
        input_queues.append(Queue())
        t = threading.Thread(
            target=node_thread, args=(i, h.ssh_hostname, hostfile, input_queues[-1])
        )
        t.start()
        threads.append(t)

    os.set_blocking(sys.stdin.fileno(), False)
    while not stop:
        rlist, _, _ = select([sys.stdin.fileno()], [], [], 1.0)
        for fd in rlist:
            stdin_buffer = os.read(fd, 8192)
            for q in input_queues:
                q.put(stdin_buffer)
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


def launch_nccl(parser, hosts, args, command):
    master_host = hosts[0].ips[0]

    if master_host != "127.0.0.1":
        raise ValueError("The NCCL backend only supports localhost for now.")
    master_port = args.nccl_port
    world_size = len(hosts)

    base_env = os.environ.copy()
    base_env.update(
        {
            "NCCL_DEBUG": base_env.get(
                "NCCL_DEBUG", "INFO" if args.verbose else "DEBUG"
            ),
            "NCCL_SOCKET_IFNAME": "lo",  # Use loopback for local communication
            "NCCL_HOST_IP": master_host,
            "NCCL_PORT": str(master_port),
            "MLX_WORLD_SIZE": str(world_size),
        }
    )
    procs = []
    num_gpus = get_num_nvidia_gpus()
    if num_gpus == 0:
        raise RuntimeError("Cannot run NCCL backend with no GPUs.")
    if args.repeat_hosts > num_gpus:
        raise RuntimeError("NCCL requires a separate GPU per process.")

    try:
        for rank in range(world_size):
            env = base_env.copy()
            mlx_rank = str(rank % args.repeat_hosts)
            env["MLX_RANK"] = mlx_rank
            env["CUDA_VISIBLE_DEVICES"] = mlx_rank
            p = Popen(command, env=env)
            procs.append(p)

        for p in procs:
            ret = p.wait()
            if ret != 0:
                raise RuntimeError(f"Rank process exited with {ret}")

    except (RuntimeError, KeyboardInterrupt) as err:
        for p in procs:
            if p.poll() is None:
                try:
                    p.kill()
                except Exception:
                    pass
        raise


def check_ssh_connections(hosts):
    results = [False] * len(hosts)

    def _check(hostname, i):
        result = run(
            [
                "ssh",
                "-o",
                "BatchMode=yes",
                "-o",
                "ConnectTimeout=5",
                hostname,
                "echo",
                "success",
            ],
            stdout=PIPE,
            stderr=PIPE,
        )
        results[i] = result.returncode == 0

    threads = [
        threading.Thread(target=_check, args=(h.ssh_hostname, i))
        for i, h in enumerate(hosts)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    if not all(results):
        log_error("Could not ssh to the following hosts:")
        for i, h in enumerate(hosts):
            if not results[i]:
                log_error("  - ", h.ssh_hostname)
        log_error()
        log_error("Maybe they are not set-up for password-less ssh?")
        sys.exit(1)


def prepare_tb_ring(args, hosts):
    log(
        args.verbose,
        f"Preparing a thunderbolt ring for {', '.join(h.ssh_hostname for h in hosts)}",
    )

    # Check that we can ssh
    check_ssh_connections(hosts)
    if args.auto_setup and args.verbose:
        log_warning(
            "--auto-setup is requested which requires password-less sudo",
            "on the remote hosts",
        )

    # Extract the current connectivity from the remote hosts
    thunderbolt_connections = []
    for h in hosts:
        log(args.verbose, "Getting connectivity from", h.ssh_hostname)
        thunderbolt_connections.append(
            json.loads(
                run(
                    [
                        "ssh",
                        h.ssh_hostname,
                        "system_profiler",
                        "SPThunderboltDataType",
                        "-json",
                    ],
                    capture_output=True,
                ).stdout
            )
        )
    interface_maps = []
    for h in hosts:
        log(args.verbose, "Getting interface names from", h.ssh_hostname)
        interface_maps.append(
            parse_hardware_ports(
                run(
                    [
                        "ssh",
                        h.ssh_hostname,
                        "networksetup",
                        "-listallhardwareports",
                    ],
                    capture_output=True,
                ).stdout
            )
        )

    # Parse the connectivity into some simple dataclasses
    tb_hosts = []
    for c, iface_map in zip(thunderbolt_connections, interface_maps):
        name = ""
        ports = []
        for t in c["SPThunderboltDataType"]:
            name = t["device_name_key"]
            uuid = t["domain_uuid_key"]
            tag = t["receptacle_1_tag"]["receptacle_id_key"]
            if items := t.get("_items", []):
                connected_to = items[0]["domain_uuid_key"]
            else:
                connected_to = None
            iface = iface_map[f"Thunderbolt {tag}"]
            ports.append(ThunderboltPort(iface, uuid, connected_to))
        tb_hosts.append(ThunderboltHost(name, sorted(ports, key=lambda x: x.iface)))

    # Create a reverse index to be able to map uuids to (host, port) quickly
    uuid_reverse_index = {}
    for i, h in enumerate(tb_hosts):
        for j, p in enumerate(h.ports):
            uuid_reverse_index[p.uuid] = (i, j)

    # Find the rings by simply walking and marking visited (host, port) tuples
    # and keeping the largest rings greedily.
    log(args.verbose, "Extracting rings from the parsed connectivity")
    rings = extract_rings(tb_hosts, uuid_reverse_index)

    # Just output a DOT graphical representation of the found rings
    if args.dot:
        names = []
        for i in range(len(tb_hosts)):
            n = ""
            j = i
            while True:
                n += chr(97 + j % 26)
                j //= 26
                if j == 0:
                    break
            names.append(n)

        print("graph G {")
        print("  node [shape=rectangle];")
        for i, h in enumerate(hosts):
            print(f'  {names[i]} [label="{h.ssh_hostname}"];')
        for r in rings:
            for (i, _), (j, _) in r:
                print(f"  {names[i]} -- {names[j]};")
        print("}")
        return

    # Assign IPs to each interface such that the interfaces can communicate
    ips = {}
    pairs = {}
    expecting = set()
    ip0 = 0
    ip1 = 0
    netmask = "255.255.255.252"
    for r in rings:
        for a, b in r:
            ips[a] = f"192.168.{ip0}.{ip1 + 1}"
            ips[b] = f"192.168.{ip0}.{ip1 + 2}"
            pairs[a] = b
            pairs[b] = a
            expecting.add(b)
            ip1 += 4
            if ip1 > 255:
                ip0 += 1
                ip1 = 0
            if ip0 > 255:
                raise ValueError("Ran out of available local IPs for the ring")

    # Create the hostfile
    hostfile = []
    for i, h in enumerate(hosts):
        host = {
            "ssh": h.ssh_hostname,
            "ips": [
                ips[i, j]
                for j, p in enumerate(tb_hosts[i].ports)
                if (i, j) in expecting
            ],
        }
        hostfile.append(host)

    if not args.hostfile_only:
        for i, h in enumerate(hosts):
            command = ""
            command += "sudo ifconfig bridge0 down\n"
            for j, p in enumerate(tb_hosts[i].ports):
                if (i, j) not in ips:
                    continue
                iface = p.iface
                ip = ips[i, j]
                peer = ips[pairs[i, j]]
                command += f"sudo ifconfig {iface} inet {ip} netmask {netmask}\n"
                command += f"sudo route change {peer} -interface {iface}\n"
            if args.auto_setup:
                print(f"Running auto setup for {h.ssh_hostname}")
                command = command.strip().replace("\n", " && ")
                command = ["ssh", h.ssh_hostname, command]
                log(args.verbose, shlex.join(command))
                run(command)
            else:
                msg = f"Setup for {h.ssh_hostname}"
                print(msg)
                print("=" * len(msg))
                print(command)
                input("Enter to continue")
            print()

    if args.output_hostfile:
        with open(args.output_hostfile, "w") as f:
            json.dump(hostfile, f, indent=4)
    else:
        print("Hostfile")
        print("========")
        print(json.dumps(hostfile, indent=4))


def prepare_hostfile(args, hosts):
    log(
        args.verbose,
        f"Preparing an ethernet hostfile for {', '.join(h.ssh_hostname for h in hosts)}",
    )

    # Check that we can ssh
    check_ssh_connections(hosts)

    # Get the ips for each host
    for h in hosts:
        log(args.verbose, "Getting the ip from", h.ssh_hostname)
        h.ips.append(
            run(
                ["ssh", h.ssh_hostname, "ipconfig", "getifaddr", "en0"],
                capture_output=True,
                text=True,
            ).stdout.strip()
        )

    hostfile = []
    for h in hosts:
        hostfile.append(dict(ssh=h.ssh_hostname, ips=h.ips))

    if args.output_hostfile:
        with open(args.output_hostfile, "w") as f:
            json.dump(hostfile, f, indent=4)
    else:
        print("Hostfile")
        print("========")
        print(json.dumps(hostfile, indent=4))


def distributed_config():
    parser = argparse.ArgumentParser(
        description="Configure remote machines for use with MLX distributed"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print debug messages in stdout"
    )
    parser.add_argument(
        "--backend",
        choices=["ring", "mpi", "nccl"],
        default="nccl" if mx.cuda.is_available() else "ring",
        help="Which distributed backend to configure",
    )
    parser.add_argument(
        "--over",
        choices=["thunderbolt", "ethernet"],
        default="thunderbolt",
        help="What type of connectivity to configure",
    )
    parser.add_argument(
        "--hosts", default="127.0.0.1", help="A comma separated list of hosts"
    )
    parser.add_argument("--hostfile", help="The file containing the hosts")
    parser.add_argument(
        "--dot", action="store_true", help="Output the topology in DOT format and exit"
    )
    parser.add_argument(
        "--hostfile-only", action="store_true", help="If set only compute the hostfile"
    )
    parser.add_argument(
        "--output-hostfile", help="If provided, save the hostfile to this path"
    )
    parser.add_argument(
        "--auto-setup",
        action="store_true",
        help="If set we will attempt to automatically configure the machines via ssh",
    )
    args = parser.parse_args()

    if args.backend == "mpi" and args.over == "thunderbolt":
        raise ValueError(
            (
                "The configuration of MPI over thunderbolt is "
                "not supported yet by mlx.distributed_config"
            )
        )

    if args.hostfile is not None:
        hosts = parse_hostfile(parser, args.hostfile)
    else:
        hosts = parse_hostlist(parser, args.hosts, 1)

    if args.over == "thunderbolt":
        prepare_tb_ring(args, hosts)
    else:
        prepare_hostfile(args, hosts)


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
        choices=["ring", "mpi", "nccl"],
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
        default=5000,
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

    args, rest = parser.parse_known_args()

    if args.print_python:
        print(sys.executable)
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
    if (script := Path(rest[0])).exists():
        rest[0:1] = [sys.executable, str(script.resolve())]
    elif (command := shutil.which(rest[0])) is not None:
        rest[0] = command
    else:
        raise ValueError(f"Invalid script or command {rest[0]}")

    # Launch
    if args.backend == "ring":
        launch_ring(parser, hosts, args, rest)
    if args.backend == "mpi":
        launch_mpi(parser, hosts, args, rest)
    if args.backend == "nccl":
        launch_nccl(parser, hosts, args, rest)


if __name__ == "__main__":
    main()
