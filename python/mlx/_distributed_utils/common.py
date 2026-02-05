# Copyright © 2025 Apple Inc.

import argparse
import ipaddress
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union


@dataclass
class Host:
    rank: int
    ssh_hostname: str
    ips: list[str]
    rdma: list[Optional[Union[str, list[str]]]]


@dataclass
class Hostfile:
    hosts: list[Host]
    backend: str = ""
    envs: list[str] = field(default_factory=list)

    def to_json(self):
        return {
            "backend": self.backend,
            "envs": self.envs,
            "hosts": [
                {"ssh": h.ssh_hostname, "ips": h.ips, "rdma": h.rdma}
                for h in self.hosts
            ],
        }

    @classmethod
    def from_file(cls, hostfile):
        """Parse the json hostfile that contains both the hostnames to ssh into and
        the ips to communicate over when using the ring backend. It can also
        contain the backend to be used and environment variables to set when
        launching a distributed job.

        Example:

            {
                "backend": "jaccl",
                "envs": [
                    "MLX_METAL_FAST_SYNCH=1"
                ],
                "hosts": [
                    {"ssh": "hostname1", "ips": ["123.123.123.1"], "rdma": [null, "rdma_en2", "rdma_en3"]},
                    {"ssh": "hostname2", "ips": ["123.123.123.2"], "rdma": ["rdma_en2", null, "rdma_en3"]},
                    ...
                    {"ssh": "hostnameN", "ips": ["123.123.123.N"], "rdma": ["rdma_en2", "rdma_en3", null]},
                ]
            }

        Args:
            hostfile (str): The path to the json file containing the host
                information
        """
        hostfile = Path(hostfile)
        if not hostfile.exists():
            raise ValueError(f"Hostfile {str(hostfile)} doesn't exist")

        try:
            data = json.load(open(hostfile))
            backend = ""
            envs = []
            hosts = []
            if isinstance(data, dict):
                backend = data["backend"]
                envs = data["envs"]
                hosts = data["hosts"]
            elif isinstance(data, list):
                hosts = data

            hosts = [
                Host(i, h["ssh"], h.get("ips", []), h.get("rdma", []))
                for i, h in enumerate(hosts)
            ]

            return cls(hosts, backend, envs)

        except Exception as e:
            raise ValueError(
                f"Failed to parse hostfile {str(hostfile)} ({str(e)})"
            ) from e

    @classmethod
    def from_list(cls, hostlist, repeats=1):
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
        return cls(hosts)


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
