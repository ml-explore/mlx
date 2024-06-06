.. _usage_distributed:

Distributed Communication
=========================

.. currentmodule:: mlx.core.distributed

MLX utilizes `MPI <https://en.wikipedia.org/wiki/Message_Passing_Interface>`_ to
provide distributed communication operations that allow the computational cost
of training or inference to be shared across many physical machines. You can
see a list of the supported operations in the :ref:`API docs<distributed>`.

.. note::
   A lot of operations may not be supported or not as fast as they should be.
   We are adding more and tuning the ones we have as we are figuring out the
   best way to do distributed computing on Macs using MLX.

Getting Started
---------------

MLX already comes with the ability to "talk" to MPI if it is installed on the
machine. The minimal distributed program in MLX is as simple as:

.. code:: python

    import mlx.core as mx

    world = mx.distributed.init()
    x = mx.distributed.all_sum(mx.ones(10))
    print(world.rank(), x)

The program above sums the array ``mx.ones(10)`` across all
distributed processes. If simply run with ``python``, however, only one
process is launched and no distributed communication takes place.

To launch the program in distributed mode we need to use ``mpirun`` or
``mpiexec`` depending on the MPI installation. The simplest possible way is the
following:

.. code:: shell

    $ mpirun -np 2 python test.py
    1 array([2, 2, 2, ..., 2, 2, 2], dtype=float32)
    0 array([2, 2, 2, ..., 2, 2, 2], dtype=float32)

The above launches two processes on the same (local) machine and we can see
both standard output streams. The processes send the array of 1s to each other
and compute the sum which is printed. Launching with ``mpirun -np 4 ...`` would
print 4 etc.

Installing MPI
---------------

MPI can be installed with Homebrew, using the Anaconda package manager or
compiled from source. Most of our testing is done using ``openmpi`` installed
with the Anaconda package manager as follows:

.. code:: shell

    $ conda install openmpi

Installing with Homebrew may require specifying the location of ``libmpi.dyld``
so that MLX can find it and load it at runtime. This can simply be achieved by
passing the ``DYLD_LIBRARY_PATH`` environment variable to ``mpirun``.

.. code:: shell

    $ mpirun -np 2 -x DYLD_LIBRARY_PATH=/opt/homebrew/lib/ python test.py

Setting up Remote Hosts
-----------------------

MPI can automatically connect to remote hosts and set up the communication over
the network if the remote hosts can be accessed via ssh. A good checklist to
debug connectivity issues is the following:

* ``ssh hostname`` works from all machines to all machines without asking for
  password or host confirmation
* ``mpirun`` is accessible on all machines. You can call ``mpirun`` using its
  full path to force all machines to use a specific path.
* Ensure that the ``hostname`` used by MPI is the one that you have configured
  in the ``.ssh/config`` files on all machines.

.. note::
  For an example hostname ``foo.bar.com`` MPI can use only ``foo`` as
  the hostname passed to ssh if the current hostname matches ``*.bar.com``.

An easy way to pass the host names to MPI is using a host file. A host file
looks like the following, where ``host1`` and ``host2`` should be the fully
qualified domain names or IPs for these hosts.

.. code::

    host1 slots=1
    host2 slots=1

When using MLX, it is very likely that you want to use 1 slot per host, ie one
process per host.  The hostfile also needs to contain the current
host if you want to run on the local host. Passing the host file to
``mpirun`` is simply done using the ``--hostfile`` command line argument.

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
        N = mx.distributed.init()
        if N == 1:
            return grads
        return tree_map(
                lambda x: mx.distributed.all_sum(x) / N,
                grads)

    def step(model, x, y):
        loss, grads = loss_grad_fn(model, x, y)
        grads = all_reduce_grads(grads)  # <--- This line was added
        optimizer.update(model, grads)
        return loss

Tuning All Reduce
-----------------

We are working on improving the performance of all reduce on MLX but for now
the two main things one can do to extract the most out of distributed training with MLX are:

1. Perform a few large reductions instead of many small ones to improve
   bandwidth and latency
2. Pass ``--mca btl_tcp_links 4`` to ``mpirun`` to configure it to use 4 tcp
   connections between each host to improve bandwidth
