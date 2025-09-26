.. _usage_distributed:

Distributed Communication
=========================

.. currentmodule:: mlx.core.distributed

MLX supports distributed communication operations that allow the computational cost
of training or inference to be shared across many physical machines. At the
moment we support two different communication backends:

* `MPI <https://en.wikipedia.org/wiki/Message_Passing_Interface>`_ a
  full-featured and mature distributed communications library
* A **ring** backend of our own that uses native TCP sockets and should be
  faster for thunderbolt connections.

The list of all currently supported operations and their documentation can be
seen in the :ref:`API docs<distributed>`.

.. note::
   Some operations may not be supported or not as fast as they should be.
   We are adding more and tuning the ones we have as we are figuring out the
   best way to do distributed computing on Macs using MLX.

Getting Started
---------------

A distributed program in MLX is as simple as:

.. code:: python

    import mlx.core as mx

    world = mx.distributed.init()
    x = mx.distributed.all_sum(mx.ones(10))
    print(world.rank(), x)

The program above sums the array ``mx.ones(10)`` across all
distributed processes. However, when this script is run with ``python`` only
one process is launched and no distributed communication takes place. Namely,
all operations in ``mx.distributed`` are noops when the distributed group has a
size of one. This property allows us to avoid code that checks if we are in a
distributed setting similar to the one below:

.. code:: python

    import mlx.core as mx

    x = ...
    world = mx.distributed.init()
    # No need for the check we can simply do x = mx.distributed.all_sum(x)
    if world.size() > 1:
        x = mx.distributed.all_sum(x)

Running Distributed Programs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

MLX provides ``mlx.launch`` a helper script to launch distributed programs.
Continuing with our initial example we can run it on localhost with 4 processes using

.. code:: shell

    $ mlx.launch -n 4 my_script.py
    3 array([4, 4, 4, ..., 4, 4, 4], dtype=float32)
    2 array([4, 4, 4, ..., 4, 4, 4], dtype=float32)
    1 array([4, 4, 4, ..., 4, 4, 4], dtype=float32)
    0 array([4, 4, 4, ..., 4, 4, 4], dtype=float32)

We can also run it on some remote hosts by providing their IPs (provided that
the script exists on all hosts and they are reachable by ssh)

.. code:: shell

    $ mlx.launch --hosts ip1,ip2,ip3,ip4 my_script.py
    3 array([4, 4, 4, ..., 4, 4, 4], dtype=float32)
    2 array([4, 4, 4, ..., 4, 4, 4], dtype=float32)
    1 array([4, 4, 4, ..., 4, 4, 4], dtype=float32)
    0 array([4, 4, 4, ..., 4, 4, 4], dtype=float32)

Consult the dedicated :doc:`usage guide<launching_distributed>` for more
information on using ``mlx.launch``.

Selecting Backend
^^^^^^^^^^^^^^^^^

You can select the backend you want to use when calling :func:`init` by passing
one of ``{'any', 'ring', 'mpi'}``. When passing ``any``, MLX will try to
initialize the ``ring`` backend and if it fails the ``mpi`` backend. If they
both fail then a singleton group is created.

.. note::
   After a distributed backend is successfully initialized :func:`init` will
   return **the same backend** if called without arguments or with backend set to
   ``any``.

The following examples aim to clarify the backend initialization logic in MLX:

.. code:: python

    # Case 1: Initialize MPI regardless if it was possible to initialize the ring backend
    world = mx.distributed.init(backend="mpi")
    world2 = mx.distributed.init()  # subsequent calls return the MPI backend!

    # Case 2: Initialize any backend
    world = mx.distributed.init(backend="any")  # equivalent to no arguments
    world2 = mx.distributed.init()  # same as above

    # Case 3: Initialize both backends at the same time
    world_mpi = mx.distributed.init(backend="mpi")
    world_ring = mx.distributed.init(backend="ring")
    world_any = mx.distributed.init()  # same as MPI because it was initialized first!

Training Example
----------------

In this section we will adapt an MLX training loop to support data parallel
distributed training. Namely, we will average the gradients across a set of
hosts before applying them to the model.

Our training loop looks like the following code snippet if we omit the model,
dataset and optimizer initialization.

.. code:: python

    model = ...
    optimizer = ...
    dataset = ...

    def step(model, x, y):
        loss, grads = loss_grad_fn(model, x, y)
        optimizer.update(model, grads)
        return loss

    for x, y in dataset:
        loss = step(model, x, y)
        mx.eval(loss, model.parameters())

All we have to do to average the gradients across machines is perform an
:func:`all_sum` and divide by the size of the :class:`Group`. Namely we
have to :func:`mlx.utils.tree_map` the gradients with following function.

.. code:: python

    def all_avg(x):
        return mx.distributed.all_sum(x) / mx.distributed.init().size()

Putting everything together our training loop step looks as follows with
everything else remaining the same.

.. code:: python

    from mlx.utils import tree_map

    def all_reduce_grads(grads):
        N = mx.distributed.init().size()
        if N == 1:
            return grads
        return tree_map(
            lambda x: mx.distributed.all_sum(x) / N,
            grads
        )

    def step(model, x, y):
        loss, grads = loss_grad_fn(model, x, y)
        grads = all_reduce_grads(grads)  # <--- This line was added
        optimizer.update(model, grads)
        return loss

Utilizing ``nn.average_gradients``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Although the code example above works correctly; it performs one communication
per gradient. It is significantly more efficient to aggregate several gradients
together and perform fewer communication steps.

This is the purpose of :func:`mlx.nn.average_gradients`. The final code looks
almost identical to the example above:

.. code:: python

    model = ...
    optimizer = ...
    dataset = ...

    def step(model, x, y):
        loss, grads = loss_grad_fn(model, x, y)
        grads = mx.nn.average_gradients(grads)  # <---- This line was added
        optimizer.update(model, grads)
        return loss

    for x, y in dataset:
        loss = step(model, x, y)
        mx.eval(loss, model.parameters())


Getting Started with MPI
------------------------

MLX already comes with the ability to "talk" to MPI if it is installed on the
machine. Launching distributed MLX programs that use MPI can be done with
``mpirun`` as expected. However, in the following examples we will be using
``mlx.launch --backend mpi`` which takes care of some nuisances such as setting
absolute paths for the ``mpirun`` executable and the ``libmpi.dyld`` shared
library.

The simplest possible usage is the following which, assuming the minimal
example in the beginning of this page, should result in:

.. code:: shell

    $ mlx.launch --backend mpi -n 2 test.py
    1 array([2, 2, 2, ..., 2, 2, 2], dtype=float32)
    0 array([2, 2, 2, ..., 2, 2, 2], dtype=float32)

The above launches two processes on the same (local) machine and we can see
both standard output streams. The processes send the array of 1s to each other
and compute the sum which is printed. Launching with ``mlx.launch -n 4 ...`` would
print 4 etc.

Installing MPI
^^^^^^^^^^^^^^

MPI can be installed with Homebrew, using the Anaconda package manager or
compiled from source. Most of our testing is done using ``openmpi`` installed
with the Anaconda package manager as follows:

.. code:: shell

    $ conda install conda-forge::openmpi

Installing with Homebrew may require specifying the location of ``libmpi.dyld``
so that MLX can find it and load it at runtime. This can simply be achieved by
passing the ``DYLD_LIBRARY_PATH`` environment variable to ``mpirun`` and it is
done automatically by ``mlx.launch``.

.. code:: shell

    $ mpirun -np 2 -x DYLD_LIBRARY_PATH=/opt/homebrew/lib/ python test.py
    $ # or simply
    $ mlx.launch -n 2 test.py

Setting up Remote Hosts
^^^^^^^^^^^^^^^^^^^^^^^

MPI can automatically connect to remote hosts and set up the communication over
the network if the remote hosts can be accessed via ssh. A good checklist to
debug connectivity issues is the following:

* ``ssh hostname`` works from all machines to all machines without asking for
  password or host confirmation
* ``mpirun`` is accessible on all machines.
* Ensure that the ``hostname`` used by MPI is the one that you have configured
  in the ``.ssh/config`` files on all machines.

Tuning MPI All Reduce
^^^^^^^^^^^^^^^^^^^^^

.. note::

    For faster all reduce consider using the ring backend either with Thunderbolt
    connections or over Ethernet.

Configure MPI to use N tcp connections between each host to improve bandwidth
by passing ``--mca btl_tcp_links N``.

Force MPI to use the most performant network interface by setting ``--mca
btl_tcp_if_include <iface>`` where ``<iface>`` should be the interface you want
to use.

Getting Started with Ring
-------------------------

The ring backend does not depend on any third party library so it is always
available. It uses TCP sockets so the nodes need to be reachable via a network.
As the name suggests the nodes are connected in a ring which means that rank 1
can only communicate with rank 0 and rank 2, rank 2 only with rank 1 and rank 3
and so on and so forth. As a result :func:`send` and :func:`recv` with
arbitrary sender and receiver is not supported in the ring backend.

Defining a Ring
^^^^^^^^^^^^^^^

The easiest way to define and use a ring is via a JSON hostfile and the
``mlx.launch`` :doc:`helper script <launching_distributed>`. For each node one
defines a hostname to ssh into to run commands on this node and one or more IPs
that this node will listen to for connections.

For example the hostfile below defines a 4 node ring. ``hostname1`` will be
rank 0, ``hostname2`` rank 1 etc.

.. code:: json

    [
        {"ssh": "hostname1", "ips": ["123.123.123.1"]},
        {"ssh": "hostname2", "ips": ["123.123.123.2"]},
        {"ssh": "hostname3", "ips": ["123.123.123.3"]},
        {"ssh": "hostname4", "ips": ["123.123.123.4"]}
    ]

Running ``mlx.launch --hostfile ring-4.json my_script.py`` will ssh into each
node, run the script which will listen for connections in each of the provided
IPs. Specifically, ``hostname1`` will connect to ``123.123.123.2`` and accept a
connection from ``123.123.123.4`` and so on and so forth.

Thunderbolt Ring
^^^^^^^^^^^^^^^^

Although the ring backend can have benefits over MPI even for Ethernet, its
main purpose is to use Thunderbolt rings for higher bandwidth communication.
Setting up such thunderbolt rings can be done manually, but is a relatively
tedious process. To simplify this, we provide the utility ``mlx.distributed_config``.

To use ``mlx.distributed_config`` your computers need to be accessible by ssh via
Ethernet or Wi-Fi. Subsequently, connect them via thunderbolt cables and then call the
utility as follows:

.. code:: shell

   mlx.distributed_config --verbose --hosts host1,host2,host3,host4

By default the script will attempt to discover the thunderbolt ring and provide
you with the commands to configure each node as well as the ``hostfile.json``
to use with ``mlx.launch``. If password-less ``sudo`` is available on the nodes
then ``--auto-setup`` can be used to configure them automatically.

To validate your connection without configuring anything
``mlx.distributed_config`` can also plot the ring using DOT format.

.. code:: shell

   mlx.distributed_config --verbose --hosts host1,host2,host3,host4 --dot >ring.dot
   dot -Tpng ring.dot >ring.png
   open ring.png

If you want to go through the process manually, the steps are as follows:

* Disable the thunderbolt bridge interface
* For the cable connecting rank ``i`` to rank ``i + 1`` find the interfaces
  corresponding to that cable in nodes ``i`` and ``i + 1``.
* Set up a unique subnetwork connecting the two nodes for the corresponding
  interfaces. For instance if the cable corresponds to ``en2`` on node ``i``
  and ``en2`` also on node ``i + 1`` then we may assign IPs ``192.168.0.1`` and
  ``192.168.0.2`` respectively to the two nodes. For more details you can see
  the commands prepared by the utility script.
