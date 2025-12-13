# Copyright © 2025 Apple Inc.

import argparse
import ipaddress
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class Host:
    rank: int
    ssh_hostname: str
    ips: list[str]
    rdma: list[Optional[str]]


class OptionalBoolAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        if option_string.startswith("--no-"):
            setattr(namespace, self.dest, False)
        else:
            setattr(namespace, self.dest, True)


def positive_number(x):
    x = int(x)
    if x <= 0:
        raise ValueError("Number should be positive")
    return x


def log(verbose, *args, **kwargs):
    if not verbose:
        return
    kwargs["file"] = sys.stderr
    print("\033[32m[INFO]", *args, "\033[0m", **kwargs)


def log_warning(*args, **kwargs):
    kwargs["file"] = sys.stderr
    print("\033[33m[WARN]", *args, "\033[0m", **kwargs)


def log_error(*args, **kwargs):
    kwargs["file"] = sys.stderr
    print("\033[31m[ERROR]", *args, "\033[0m", **kwargs)


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
            hosts.append(Host(i, h, ips, []))
    return hosts


def parse_hostfile(parser, hostfile):
    """Parse the json hostfile that contains both the hostnames to ssh into and
    the ips to communicate over when using the ring backend.

    Example:

        [
            {"ssh": "hostname1", "ips": ["123.123.123.1"], "rdma": [null, "rdma_en2", "rdma_en3"]},
            {"ssh": "hostname2", "ips": ["123.123.123.2"], "rdma": ["rdma_en2", null, "rdma_en3"]},
            ...
            {"ssh": "hostnameN", "ips": ["123.123.123.N"], "rdma": ["rdma_en2", "rdma_en3", null]},
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
                hosts.append(Host(i, h["ssh"], h.get("ips", []), h.get("rdma", [])))
        return hosts
    except Exception as e:
        parser.error(f"Failed to parse hostfile {str(hostfile)} ({str(e)})")
