:orphan:

.. _usage_launch_distributed:

Launching Distributed Programs
==============================

.. currentmodule:: mlx.core.distributed

The MLX python package provides two utilities to help you configure
your Macs for distributed computation and also launch distributed programs on
multiple nodes or with many processes in a single node. These utilities are aptly named

- ``mlx.launch``
- ``mlx.distributed_config``

See the :doc:`distributed docs <distributed>` for an introduction and
getting-started guides to the various backends.

``mlx.distributed_config`` 
---------------------------

Unless you are launching distributed jobs locally for development or multi-gpu
CUDA environments, then you have several Macs that you need to configure for
distributed communication with MLX.

``mlx.distributed_config`` aims to automate the process of configuring the
network interfaces (especially for communication over thunderbolt) and also
creating the hostfile to be used with ``mlx.launch``.

We will analyse 3 cases of using ``mlx.distributed_config``

1. RDMA over thunderbolt using JACCL
2. TCP/IP over thunderbolt using the ring backend
3. TCP/IP over ethernet using the ring backend

JACCL
^^^^^^^

After following :ref:`the steps to enable RDMA <jaccl_section>` you can run the
following command to configure the nodes and create the hostfile.

.. code-block::

   mlx.distributed_config --verbose --backend jaccl \
        --hosts m3-ultra-1,m3-ultra-2,m3-ultra-3,m3-ultra-4 --over thunderbolt \
        --auto-setup --output m3-ultra-jaccl.json

Let's walk through the steps that the script takes to configure the nodes.

1. ssh to all nodes to verify that they are reachable
2. Extract the thunderbolt connectivity. Namely run commands on each node to
   calculate which node is connected to which other node.
3. Verify that we have a valid fully connected mesh
4. Check that RDMA is enabled
5. Extract the ethernet IP from interface en0
6. Disable the thunderbolt bridge and set up peer to peer networks for each
   thunderbolt cable
7. Write the hostfile

Knowing the above steps allows you to manually configure the nodes but also
debug any configuration issue. For instance changing the Ethernet IP to a
different interface directly in the config is possible (as long as it is
reachable from all nodes).

The ``--auto-setup`` argument requires password-less sudo on each node. If it
isn't available then the configuration script will print commands to be run on
each node.

Ring over thunderbolt
^^^^^^^^^^^^^^^^^^^^^

Setting up a ring backend over thunderbolt only requires changing the
``--backend`` from ``jaccl`` to ``ring``.

The steps are very similar with the main difference being that instead of
verifying that the nodes are fully connected, the script attempts to identify a
ring topology (or multiple rings).

Ring over Ethernet
^^^^^^^^^^^^^^^^^^

Configuring the ring backend over ethernet doesn't require setting up network
interface and as such it simply extracts the ``en0`` IP from each node and
writes the hostfile.

Debugging cable connections
^^^^^^^^^^^^^^^^^^^^^^^^^^^

``mlx.distributed_config`` can help you debug the connectivity of your nodes
over thunderbolt by exporting a graph of the connections.

Running

.. code-block::

   mlx.distributed_config --verbose \
        --hosts host1,host2,host3,host4 \
        --over thunderbolt --dot

will export a `GraphViz <https://graphviz.org>`_ representation of the
connections between the nodes which makes it very easy to figure out which
cable is not connected correctly.

See :ref:`the JACCL section <jaccl_section>` for an example.


``mlx.launch``
--------------

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

Importantly, it also broadcasts stdin to each process which enables interactive
programs to work in distributed mode as well as debugging using the interactive
debugger.

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

If you are launching from a node with a completely different setup than the
nodes that the program will run on, you can specify ``--no-verify-script`` so
that ``mlx.launch`` does not attempt to verify that the executable and script
exist locally before launching the distributed job.

.. _ring_specifics:

Ring Specifics
^^^^^^^^^^^^^^

The :ref:`ring <ring_section>` backend, which is also the default
backend, can be explicitly selected with the argument ``--backend ring``. The
ring backend has some specific requirements and arguments that are different to
other backends:

* The argument ``--hosts`` only accepts IPs and not hostnames. If we need to
  ssh to a hostname that does not correspond to the IP we want to bind to we
  have to provide a hostfile.
* ``--starting-port`` defines the port to bind to on the remote hosts.
  Specifically rank 0 for the first IP will use this port and each subsequent
  IP or rank will add 1 to this port.
* ``--connections-per-ip`` allows us to increase the number of connections
  between neighboring nodes. This corresponds to ``--mca btl_tcp_links 2`` for
  ``mpirun``.

.. _jaccl_specifics:

JACCL Specifics
^^^^^^^^^^^^^^^^

The :ref:`JACCL <jaccl_section>` backend can be selected with the argument
``--backend jaccl``. A hostfile is necessary to launch with this backend
because it needs to contain the RDMA devices connecting each node to each other
node.

NCCL Specifics
^^^^^^^^^^^^^^

The :ref:`NCCL <nccl_section>` backend is the default backend for CUDA
environments. When launching from a Mac to a Linux machine with CUDA then the
backend should be selected using ``--backend nccl``.

The ``--repeat-hosts, -n`` argument should be used to launch multi-node and
multi-gpu jobs. For instance

.. code-block::

   mlx.launch --backend nccl --hosts linux-1,linux-2 -n 8 --no-verify-script -- ./my-job.sh

will attempt to launch 16 processes, 8 on each node that will all run
``my-job.sh``.

.. _mpi_specifics:

MPI Specifics
^^^^^^^^^^^^^

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
