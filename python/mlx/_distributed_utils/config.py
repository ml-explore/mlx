# Copyright © 2025 Apple Inc.

import argparse
import json
import shlex
import sys
import threading
from collections import defaultdict
from dataclasses import dataclass
from subprocess import DEVNULL, run
from typing import Optional

import mlx.core as mx

from .common import (
    Host,
    OptionalBoolAction,
    log,
    log_error,
    log_warning,
    parse_hostfile,
    parse_hostlist,
)


@dataclass
class SSHInfo:
    can_ssh: bool
    has_sudo: bool

    def __bool__(self):
        return self.can_ssh


@dataclass
class ThunderboltPort:
    iface: str
    uuid: str
    connected_to: Optional[str]


@dataclass
class ThunderboltHost:
    name: str
    ports: list[ThunderboltPort]


def add_ips(hosts, verbose=False):
    # Get the ips for each host
    for h in hosts:
        log(verbose, "Getting the ip from", h.ssh_hostname)
        ip = run(
            ["ssh", h.ssh_hostname, "ipconfig", "getifaddr", "en0"],
            capture_output=True,
            text=True,
        ).stdout.strip()
        if ip != "":
            h.ips.append(ip)
            continue

        ip = run(
            ["ssh", h.ssh_hostname, "ipconfig", "getifaddr", "en1"],
            capture_output=True,
            text=True,
        ).stdout.strip()
        if ip != "":
            h.ips.append(ip)
            continue

        log_warning("Could not extract ip for", h.ssh_hostname)


def check_rdma(hosts, verbose=False):
    # Check whether the hosts are capable of RDMA over thunderbolt
    warn = False
    for h in hosts:
        log(verbose, "Checking that", h.ssh_hostname, "supports RDMA")
        rdma_devs = (
            run(["ssh", h.ssh_hostname, "ibv_devices"], capture_output=True, text=True)
            .stdout.strip()
            .split()
        )
        rdma_devs = [d for d in rdma_devs if d.startswith("rdma_")]
        if not rdma_devs:
            log_warning(h.ssh_hostname, "does not seem to have RDMA enabled")
            warn = True

    if warn:
        log_warning()
        log_warning(
            "Some of the hosts don't have RDMA enabled or they don't support RDMA."
        )
        log_warning()
        log_warning(
            "See https://ml-explore.github.io/mlx/build/html/usage/distributed.html"
        )
        log_warning("for instructions on how to enable RDMA.")


def can_auto_setup(hosts, sshinfo, auto_setup=False):
    has_sudo = all(info.has_sudo for info in sshinfo)
    if not has_sudo and auto_setup:
        log_warning(
            "Automatic setup requested but the following hosts do not have passwordless sudo"
        )
        for h, i in zip(hosts, sshinfo):
            if not i.has_sudo:
                log_warning(" - ", h.ssh_hostname)
    return has_sudo


class IPConfigurator:
    def __init__(self, hosts, tb_hosts, uuid_reverse_index):
        assigned = set()
        ips = defaultdict(list)
        ip0 = 0
        ip1 = 0
        for src_node, h in enumerate(tb_hosts):
            for src_port, p in enumerate(h.ports):
                if not p.connected_to:
                    continue
                if p.connected_to not in uuid_reverse_index:
                    continue
                if (src_node, src_port) in assigned:
                    continue

                dst_node, dst_port = uuid_reverse_index[p.connected_to]

                ip_src = f"192.168.{ip0}.{ip1 + 1}"
                ip_dst = f"192.168.{ip0}.{ip1 + 2}"
                iface_src = p.iface
                iface_dst = tb_hosts[dst_node].ports[dst_port].iface

                ips[src_node, dst_node].append((iface_src, ip_src))
                ips[dst_node, src_node].append((iface_dst, ip_dst))

                assigned.add((src_node, src_port))
                assigned.add((dst_node, dst_port))

                ip1 += 4
                if ip1 > 255:
                    ip0 += 1
                    ip1 = 0
                if ip0 > 255:
                    raise ValueError("Ran out of available local IPs")

        self.ips = ips
        self.hosts = hosts
        self.tb_hosts = tb_hosts

    def setup(self, verbose=False, auto_setup=False):
        netmask = "255.255.255.252"
        for i, (h, th) in enumerate(zip(self.hosts, self.tb_hosts)):
            command = ""
            command += "sudo ifconfig bridge0 down\n"
            for j in range(len(self.hosts)):
                if i == j or (i, j) not in self.ips:
                    continue
                for (iface, ip), (_, peer) in zip(self.ips[i, j], self.ips[j, i]):
                    command += f"sudo ifconfig {iface} inet {ip} netmask {netmask}\n"
                    command += f"sudo route change {peer} -interface {iface}\n"
            if auto_setup:
                print(f"Running auto setup for {h.ssh_hostname}")
                command = command.strip().replace("\n", " ; ")
                command = ["ssh", h.ssh_hostname, command]
                log(verbose, shlex.join(command))
                run(command)
            else:
                msg = f"Setup for {h.ssh_hostname}"
                print(msg)
                print("=" * len(msg))
                print(command)
                input("Enter to continue")
            print()


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


def extract_connectivity(hosts, verbose):
    # Extract the current connectivity from the remote hosts
    thunderbolt_connections = []
    for h in hosts:
        log(verbose, "Getting connectivity from", h.ssh_hostname)
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
        log(verbose, "Getting interface names from", h.ssh_hostname)
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
            uuid = t.get("domain_uuid_key")
            if uuid is None:
                continue
            name = t["device_name_key"]
            tag = t["receptacle_1_tag"]["receptacle_id_key"]
            items = t.get("_items", [])
            connected_items = [item for item in items if "domain_uuid_key" in item]
            connected_to = (
                connected_items[0]["domain_uuid_key"] if connected_items else None
            )
            iface = iface_map[f"Thunderbolt {tag}"]
            ports.append(ThunderboltPort(iface, uuid, connected_to))
        tb_hosts.append(ThunderboltHost(name, sorted(ports, key=lambda x: x.iface)))

    # Create a reverse index to be able to map uuids to (host, port) quickly
    uuid_reverse_index = {}
    for i, h in enumerate(tb_hosts):
        for j, p in enumerate(h.ports):
            uuid_reverse_index[p.uuid] = (i, j)

    return tb_hosts, uuid_reverse_index


def make_connectivity_matrix(tb_hosts, uuid_reverse_index):
    connectivity = []
    for i, h in enumerate(tb_hosts):
        c = [0] * len(tb_hosts)
        for p in h.ports:
            if p.connected_to in uuid_reverse_index:
                j, _ = uuid_reverse_index[p.connected_to]
                c[j] += 1
        connectivity.append(c)
    return connectivity


def tb_connectivity_to_dot(hosts, tb_hosts, uuid_reverse_index):
    # Make ids per node
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
    for i, h in enumerate(tb_hosts):
        for p in h.ports:
            if not p.connected_to:
                continue
            dst = uuid_reverse_index[p.connected_to]
            if dst[0] < i:
                continue
            print(f"  {names[i]} -- {names[dst[0]]}", end="")
            print(f' [label="{p.iface}/{tb_hosts[dst[0]].ports[dst[1]].iface}"]')
    print("}")


def extract_rings(connectivity):
    rings = []
    existing_rings = set()
    num_nodes = len(connectivity)

    def dfs(start_node, node, path, visited):
        path.append(node)
        visited.add(node)
        for j in range(num_nodes):
            if connectivity[node][j] <= 0:
                continue
            if j == start_node:
                yield path[:]
            if j not in visited:
                yield from dfs(start_node, j, path, visited)
        path.pop()
        visited.remove(node)

    for start in range(num_nodes):
        for r in dfs(start, start, [], set()):
            cnt = min(connectivity[r[i]][r[(i + 1) % len(r)]] for i in range(len(r)))
            rkey = tuple(sorted(r))
            if rkey not in existing_rings:
                rings.append((r, cnt))
                existing_rings.add(rkey)

    return sorted(rings, key=lambda x: -len(x[0]))


def check_valid_mesh(hosts, connectivity, strict=True):
    num_nodes = len(connectivity)
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i == j:
                continue
            if connectivity[i][j] <= 0:
                if strict:
                    log_error(
                        f"Incomplete mesh, {hosts[i].ssh_hostname} is not connected to {hosts[j].ssh_hostname}"
                    )
                    log_error()
                    log_error("Try passing --dot to visualize the connectivity")
                    sys.exit(1)
                else:
                    return False
    return True


def check_ssh_connections(hosts):
    results = [None] * len(hosts)

    def _check(hostname, i):
        info = SSHInfo(False, False)
        results[i] = info

        # Check for ssh
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
            stdout=DEVNULL,
            stderr=DEVNULL,
        )
        info.can_ssh = result.returncode == 0
        if not info.can_ssh:
            return

        # Check for sudo
        result = run(
            [
                "ssh",
                "-o",
                "BatchMode=yes",
                "-o",
                "ConnectTimeout=5",
                hostname,
                "sudo",
                "ls",
            ],
            stdout=DEVNULL,
            stderr=DEVNULL,
        )
        info.has_sudo = result.returncode == 0

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

    return results


def prepare_ethernet_hostfile(args, hosts):
    log(args.verbose, f"Preparing an ethernet hostfile")
    add_ips(hosts, args.verbose)

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


def configure_ring(args, hosts, ips, ring, sshinfo):
    log(args.verbose, "Prepare a ring hostfile")
    ring, count = ring
    hostfile = []
    for i, node in enumerate(ring):
        h = hosts[node]
        peer = ring[i - 1]
        hostfile.append(
            {
                "ssh": h.ssh_hostname,
                "ips": [ips.ips[node, peer][c][1] for c in range(count)],
                "rdma": [],
            }
        )

    has_sudo = can_auto_setup(hosts, sshinfo, args.auto_setup)
    ips.setup(verbose=args.verbose, auto_setup=args.auto_setup and has_sudo)

    if args.output_hostfile:
        with open(args.output_hostfile, "w") as f:
            json.dump(hostfile, f, indent=4)
    else:
        print("Hostfile")
        print("========")
        print(json.dumps(hostfile, indent=4))


def configure_jaccl(args, hosts, ips, sshinfo):
    log(args.verbose, "Prepare a jaccl hostfile")
    check_rdma(hosts, args.verbose)
    add_ips(hosts, args.verbose)

    hostfile = []
    for i, h in enumerate(hosts):
        rdma = []
        for j in range(len(hosts)):
            if i == j:
                rdma.append(None)
            else:
                rdma.append(f"rdma_{ips.ips[i, j][0][0]}")
        hostfile.append({"ssh": h.ssh_hostname, "ips": h.ips, "rdma": rdma})

    has_sudo = can_auto_setup(hosts, sshinfo, args.auto_setup)
    ips.setup(verbose=args.verbose, auto_setup=args.auto_setup and has_sudo)

    if args.output_hostfile:
        with open(args.output_hostfile, "w") as f:
            json.dump(hostfile, f, indent=4)
    else:
        print("Hostfile")
        print("========")
        print(json.dumps(hostfile, indent=4))


def prepare_tb_hostfile(args, hosts, sshinfo):
    log(args.verbose, f"Preparing for communication over thunderbolt")
    tb_hosts, uuid_reverse_index = extract_connectivity(hosts, args.verbose)

    if args.dot:
        tb_connectivity_to_dot(hosts, tb_hosts, uuid_reverse_index)
        return

    ips = IPConfigurator(hosts, tb_hosts, uuid_reverse_index)
    connectivity = make_connectivity_matrix(tb_hosts, uuid_reverse_index)

    if args.backend is None:
        rings = extract_rings(connectivity)
        has_mesh = check_valid_mesh(hosts, connectivity, False)
        has_ring = len(rings) > 0 and len(rings[0][0]) == len(hosts)

        if not has_ring and not has_mesh:
            log_error("Neither thunderbolt mesh nor ring found.")
            log_error("Perhaps run with --dot to generate a plot of the connectivity.")
            sys.exit(1)

        elif has_ring:
            configure_ring(args, hosts, ips, rings[0], sshinfo)

        else:
            configure_jaccl(args, hosts, ips, sshinfo)

    elif args.backend == "ring":
        rings = extract_rings(connectivity)
        has_ring = len(rings) > 0 and len(rings[0][0]) == len(hosts)
        if not has_ring:
            log_error("Could not find a full ring.")
            log_error()
            log_error("Try passing --dot to visualize the connectivity")
            if len(rings) > 0:
                log_error("Rings found:")
                for r in rings:
                    log_error(f" - {','.join(hosts[i].ssh_hostname for i in r)}")
            sys.exit(1)
        configure_ring(args, hosts, ips, rings[0], sshinfo)

    elif args.backend == "jaccl":
        check_valid_mesh(hosts, connectivity)
        configure_jaccl(args, hosts, ips, sshinfo)


def main():
    parser = argparse.ArgumentParser(
        description="Configure remote machines for use with MLX distributed"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print debug messages in stdout"
    )
    parser.add_argument(
        "--hosts", default="127.0.0.1", help="A comma separated list of hosts"
    )
    parser.add_argument("--hostfile", help="The file containing the hosts")
    parser.add_argument(
        "--over",
        choices=["thunderbolt", "ethernet"],
        default="thunderbolt",
        help="What type of connectivity to configure",
        required=True,
    )
    parser.add_argument(
        "--output-hostfile", help="If provided, save the hostfile to this path"
    )
    parser.add_argument(
        "--auto-setup",
        "--no-auto-setup",
        action=OptionalBoolAction,
        nargs=0,
        dest="auto_setup",
        default=None,
    )
    parser.add_argument(
        "--dot", action="store_true", help="Output the topology in DOT format and exit"
    )
    parser.add_argument(
        "--backend",
        choices=["ring", "jaccl"],
        default=None,
        help="Which distributed backend to configure",
    )
    args = parser.parse_args()

    if args.hostfile is not None:
        hosts = parse_hostfile(parser, args.hostfile)
    else:
        hosts = parse_hostlist(parser, args.hosts, 1)

    # Check that we can ssh
    log(
        args.verbose,
        f"Checking for ssh access for {', '.join(h.ssh_hostname for h in hosts)}",
    )
    sshinfo = check_ssh_connections(hosts)

    # Prepare a hostfile for communication over ethernet using the ips of the
    # provided hostnames
    if args.over == "ethernet":
        prepare_ethernet_hostfile(args, hosts)

    # Configure the macs for communication over thunderbolt, both via RDMA and IP
    else:
        prepare_tb_hostfile(args, hosts, sshinfo)
