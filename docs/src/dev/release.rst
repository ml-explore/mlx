Publishing a New Release
========================

Steps
-----

1. Check the current version on the latest ``main`` branch in ``mlx/version.h``.
2. Push a new tag for current version to the GitHub repository, which would
   automatically trigger release workflow.
3. Wait for the release workflow to **succeed**.
4. Create a new GitHub Release for the latest tag from the `Tags
   <https://github.com/ml-explore/mlx/tags>`__ page.
5. Create a pull request that increases the version in ``mlx/version.h``.

Troubleshooting
---------------

Build or test failure
^^^^^^^^^^^^^^^^^^^^^

If the release workflow fails at the build or test step:

1. Remove the tag from GitHub.
2. Fix the build or test failure.
3. Do the steps again.

See `Common causes of release build failures`_ for possible causes.

All publish jobs fail
^^^^^^^^^^^^^^^^^^^^^

If all the publish jobs fail, remove the tag from GitHub first. A possible cause
is that the credentials are not valid anymore, you must get it fixed before
retrying.

Some publish jobs fail
^^^^^^^^^^^^^^^^^^^^^^

If only some publish jobs fail, find the cause:

- If the cause is a network issue, restart **only** the failed publish jobs.
- If restarting cannot fix the cause (for example, the package is larger than
  the PyPI size limit), create a fix and merge it to the main branch. Then
  `trigger the release workflow manually`_.

Trigger the release workflow manually
-------------------------------------

Use this method only for troubleshooting. It is **not** the recommended way to
publish a release.

1. Open https://github.com/ml-explore/mlx/actions/workflows/release.yml.
2. For a dry run, which only tests that the release build works, keep the
   "Publish to PyPI" option unchecked. Then click "Run workflow".
3. To publish a release, select "Publish to PyPI". Then click "Run workflow".
   Note that it is only possible to publish from main branch.

Common causes of release build failures
---------------------------------------

Release build failures are rare, because pull requests run the full tests.
But the usual CI makes only debug builds, so a release build can fail because:

1. Some code is inside a debug macro guard and behaves differently in release
   builds.
2. A bug, for example a race condition, shows only when the code runs faster in
   release builds.
3. In the CUDA build, the debug build targets only the current architecture,
   while the release build targets all supported architectures, on which CUDA
   code can fail to compile or run. So when changing a CUDA kernel that makes
   use of hardware specific features, also start a dry run of the release build
   to avoid such problems.
4. The release build makes two Python packages: a frontend package with only the
   Python bindings, and a backend package with the headers and the backend- and
   platform-specific binaries. A change to the CMake files or ``setup.py`` can
   break them. See ``.github/actions/build-wheel/action.yml`` for details.
