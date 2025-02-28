:orphan:

.. _usage_launch_distributed:

Launching Distributed Programs
==============================

.. currentmodule:: mlx.core.distributed

Installing the MLX python package provides a helper script ``mlx.launch`` that
can be used to run python scripts distributed on several nodes. It allows
launching using either the MPI backend or the ring backend. See the
:doc:`distributed docs <distributed>` for the different backends.

Usage
-----

The minimal usage example of ``mlx.launch`` is simply

.. code:: shell

    mlx.launch --hosts ip1,ip2 my_script.py

or for testing on localhost

.. code:: shell

    mlx.launch -n 2 my_script.py

The ``mlx.launch`` command connects to the provided host and launches the input
script on each host. It monitors each of the launched processes and terminates
the rest if one of them fails unexpectedly or if ``mlx.launch`` is terminated.
It also takes care of forwarding the output of each remote process to stdout
and stderr respectively.

Providing Hosts
^^^^^^^^^^^^^^^^

Hosts can be provided as command line arguments, like above, but the way that
allows to fully define a list of hosts is via a JSON hostfile. The hostfile has
a very simple schema. It is simply a list of objects that define each host via
a hostname to ssh to and a list of IPs to utilize for the communication.

.. code:: json

    [
        {"ssh": "hostname1", "ips": ["123.123.1.1", "123.123.2.1"]},
        {"ssh": "hostname2", "ips": ["123.123.1.2", "123.123.2.2"]}
    ]

You can use ``mlx.distributed_config --over ethernet`` to create a hostfile
with IPs corresponding to the ``en0`` interface.

Setting up Remote Hosts
^^^^^^^^^^^^^^^^^^^^^^^^

In order to be able to launch the script on each host we need to be able to
connect via ssh. Moreover the input script and python binary need to be on each
host and on the same path. A good checklist to debug errors is the following:

* ``ssh hostname`` works without asking for password or host confirmation
* the python binary is available on all hosts at the same path. You can use
  ``mlx.launch --print-python`` to see what that path is.
* the script you want to run is available on all hosts at the same path

.. _mpi_specifics:

MPI Specifics
-------------

One can use MPI by passing ``--backend mpi`` to ``mlx.launch``. In that case,
``mlx.launch`` is a thin wrapper over ``mpirun``. Moreover,

* The IPs in the hostfile are ignored
* The ssh connectivity requirement is stronger as every node needs to be able
  to connect to every other node
* ``mpirun`` needs to be available on every node at the same path

Finally, one can pass arguments to ``mpirun`` using ``--mpi-arg``. For instance
to choose a specific interface for the byte-transfer-layer of MPI we can call
``mlx.launch`` as follows:

.. code:: shell

    mlx.launch --backend mpi --mpi-arg '--mca btl_tcp_if_include en0' --hostfile hosts.json my_script.py


.. _ring_specifics:

Ring Specifics
--------------

The ring backend, which is also the default backend, can be explicitly selected
with the argument ``--backend ring``. The ring backend has some specific
requirements and arguments that are different to MPI:

* The argument ``--hosts`` only accepts IPs and not hostnames. If we need to
  ssh to a hostname that does not correspond to the IP we want to bind to we
  have to provide a hostfile.
* ``--starting-port`` defines the port to bind to on the remote hosts.
  Specifically rank 0 for the first IP will use this port and each subsequent
  IP or rank will add 1 to this port.
* ``--connections-per-ip`` allows us to increase the number of connections
  between neighboring nodes. This corresponds to ``--mca btl_tcp_links 2`` for
  ``mpirun``.
