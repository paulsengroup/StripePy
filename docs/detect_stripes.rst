..
  Copyright (C) 2025 Andrea Raffo <andrea.raffo@ibv.uio.no>
  SPDX-License-Identifier: MIT

Detect architectural stripes
============================

The `stripepy_call_help` command serves as the main component within StripePy, and may take several minutes to complete when processing large files.

Positional arguments
--------------------

The command mandates two positional arguments:
  * ``contact_map``, which specifies the path to the input Hi-C file (in .cool, .mcool, or .hic format).
  * ``resolution``, which represents the resolution (in base pairs) at which the analysis should be run.

For instance, to run the command on a file named ``4DNFI9GMP2J8.mcool`` at a resolution of 10,000 bp, you would use the following command:

.. code-block:: console

  user@dev:/tmp$ stripepy call 4DNFI9GMP2J8.mcool 10000

Running the above command produces a similar output, here truncated for the sake of brevity:

.. code-block:: console

  2025-04-15 08:13:24.639742 [info     ] running StripePy v1.0.0
  2025-04-15 08:13:24.637358 [info     ] [main      ] CONFIG:
  {
    "constrain_heights": false,
    "contact_map": "4DNFI9GMP2J8.mcool",
    "force": false,
    "genomic_belt": 5000000,
    "glob_pers_min": 0.04,
    "loc_pers_min": 0.33,
    "loc_trend_min": 0.25,
    "log_file": null,
    "max_width": 100000,
    "min_chrom_size": 2000000,
    "normalization": null,
    "nproc": 1,
    "output_file": "/tmp/4DNFI9GMP2J8.10000.hdf5",
    "plot_dir": null,
    "resolution": 10000,
    "roi": null,
    "verbosity": "info"
  }
  2025-04-15 08:13:24.637440 [info     ] [main      ] validating file "4DNFI9GMP2J8.mcool" (10000bp)
  2025-04-15 08:13:24.650236 [info     ] [main      ] file "4DNFI9GMP2J8.mcool" successfully validated
  2025-04-15 08:13:24.650445 [info     ] [IO        ] initializing result file "/tmp/4DNFI9GMP2J8.10000.hdf5"
  2025-04-15 08:13:24.672613 [info     ] [chr1 ] [main      ] begin processing
  2025-04-15 08:13:24.672729 [info     ] [chr1 ] [IO        ] fetching interactions using normalization=NONE
  2025-04-15 08:13:25.483686 [info     ] [chr1 ] [IO        ] fetched 6823257 pixels in 810.948ms
  2025-04-15 08:13:25.483913 [info     ] [chr1 ] [step 1    ] data pre-processing
  2025-04-15 08:13:25.483995 [info     ] [chr1 ] [step 1.1  ] focusing on a neighborhood of the main diagonal
  2025-04-15 08:13:25.535171 [info     ] [chr1 ] [step 1.1  ] removed 0.00% of the non-zero entries (0/6823257)
  2025-04-15 08:13:25.535378 [info     ] [chr1 ] [step 1.2  ] applying log-transformation
  2025-04-15 08:13:25.549232 [info     ] [chr1 ] [step 1.3  ] projecting interactions onto [1, 0]
  2025-04-15 08:13:25.553946 [info     ] [chr1 ] [step 1    ] preprocessing took 69.937ms
  2025-04-15 08:13:25.558918 [info     ] [chr1 ] [step 2    ] topological data analysis
  2025-04-15 08:13:25.559059 [info     ] [chr1 ] [step 2.1.0] [LT] computing global 1D pseudo-distribution
  2025-04-15 08:13:25.583652 [info     ] [chr1 ] [step 2.2.0] [LT] detection of persistent maxima and corresponding minima
  2025-04-15 08:13:25.583770 [info     ] [chr1 ] [step 2.2.1] [LT] computing persistence
  2025-04-15 08:13:25.625730 [info     ] [chr1 ] [step 2.2.2] [LT] filtering low persistence values
  2025-04-15 08:13:25.626417 [info     ] [chr1 ] [step 2.2.3] [LT] removing seeds overlapping sparse regions
  2025-04-15 08:13:25.686625 [info     ] [chr1 ] [step 2.2.3] [LT] number of seed sites reduced from 1807 to 1748
  2025-04-15 08:13:25.686795 [info     ] [chr1 ] [step 2.3.1] [LT] generating the list of candidate stripes
  2025-04-15 08:13:25.687662 [info     ] [chr1 ] [step 2.3.1] [LT] identified 1748 candidate stripes
  2025-04-15 08:13:25.687864 [info     ] [chr1 ] [step 2.1.0] [UT] computing global 1D pseudo-distribution
  2025-04-15 08:13:25.713048 [info     ] [chr1 ] [step 2.2.0] [UT] detection of persistent maxima and corresponding minima
  2025-04-15 08:13:25.713154 [info     ] [chr1 ] [step 2.2.1] [UT] computing persistence
  2025-04-15 08:13:25.753436 [info     ] [chr1 ] [step 2.2.2] [UT] filtering low persistence values
  2025-04-15 08:13:25.753932 [info     ] [chr1 ] [step 2.2.3] [UT] removing seeds overlapping sparse regions
  2025-04-15 08:13:25.813509 [info     ] [chr1 ] [step 2.2.3] [UT] number of seed sites reduced from 1698 to 1647
  2025-04-15 08:13:25.813687 [info     ] [chr1 ] [step 2.3.1] [UT] generating the list of candidate stripes
  ...
  2025-04-15 08:14:59.123408 [info     ] [IO        ] finalizing file "/tmp/4DNFI9GMP2J8.10000.hdf5"
  2025-04-15 08:14:59.127303 [info     ] [main      ] DONE!
  2025-04-15 08:14:59.127399 [info     ] [main      ] processed 24 chromosomes in 1m:34.490s


Upon successful completion, the above command will generate a single HDF5 file named ``4DNFI9GMP2J8.10000.hdf5`` in the current working directory.
Note that, if the HDF5 already exists, it will be overwritten if and only if the ``--force`` flag is specified; otherwise, the command will fail to prevent accidental data loss.

Output and logging configuration
--------------------------------

By default, the output HDF5 file is named after the input matrix file with the resolution appended, for example, ``4DNFI9GMP2J8.10000.hdf5``.
However, you have the flexibility to specify an alternative output path and filename for this HDF5 file using the ``--output-file`` option.

Furthermore, it is possible to save the complete log of a run to a file by specifying the path where to store the log file through the ``--log-file`` CLI option.

Stripe detection parameters
---------------------------

Beyond these arguments, `stripepy_call_help` comes with a suite of optional parameters for fine-tuning the stripe detection process. For a full understanding of their meaning, the user is referred to our `paper <https://doi.org/10.1093/bioinformatics/btaf351>`_.

Step 1: pre-processing
^^^^^^^^^^^^^^^^^^^^^^

You can apply a specific ``--normalization`` method when fetching the contact map data from the input file; by default, no normalization is applied.
As found in our experiments, our algorithm performs optimally when no prior balancing is applied (see the `Supplementary Information <https://academic.oup.com/bioinformatics/article/41/6/btaf351/8161567#524807912>`_ from our paper).

The ``--genomic-belt`` option defines a radial band around the main diagonal of the contact map, specified in base pairs, to which the stripe search is confined; its default value is 5 Mbp.

Step 2: line detection
^^^^^^^^^^^^^^^^^^^^^^

The ``--glob-pers-min`` option sets a critical threshold value between 0 and 1 (defaulting to 0.04).
This threshold is instrumental in filtering persistence maxima points for the global pseudo-distribution, which are crucial for identifying initial candidate stripe locations, frequently referred to as "seeds".

Step 3: shape analysis
^^^^^^^^^^^^^^^^^^^^^^

The maximum permissible stripe width can be explicitly controlled using the ``--max-width`` option, which is specified in base pairs and defaults to 100,000 bp.

The height of a stripe is determined by studying a local pseudo-distribution.
The algorithm applies topological persistence to the local pseudo-distribution to identify persistent peaks.
The ``--loc-pers-min`` option acts as a threshold value between 0 and 1 (defaulting to 0.33) used to determine which peaks are persistent with respect to their topological persistence.
The location of the furthest identified peak is then used as a boundary for the stripe.
If no persistent maximum other than the global maximum is found, we threshold the local pseudo-distribution to a minimum value, specified via the option ``--loc-trend-min``, which should be set to a value between 0 and 1 (defaulting to 0.25).
A higher value for this parameter generally results in the detection of shorter stripes.

Step 4: signal analysis
^^^^^^^^^^^^^^^^^^^^^^^

The ``--k-neighbour`` option allows you to define 'k' for the k-neighbours: it represents the number of bins that are considered adjacent to the stripe boundaries on both sides, with a default value of 3.
It is used to compute various signal descriptors, such as the relative change parameter.


Diagnostic plots generation
---------------------------

The command `stripepy_call_help` can generate several diagnostic plots that can be of help to gain more insights into the decisions made by the tool.

To generate the diagnostic plots, pass ``--roi=middle`` and specify the path to a folder where to store the plots using ``--plot-dir``.
The ``--roi`` option requires you to specify a criterion (``start`` or ``middle``) to select a representative region from each chromosome for plot generation.
Concurrently, the ``--plot-dir`` option designates the path to a directory where these output plots will be stored.
It is important to note that the ``--plot-dir`` option is mandatory when ``--roi`` is specified and is otherwise ignored.
If the specified directory does not exist at the time of execution, ``stripepy`` will automatically create it.

Performance options
-------------------

When processing larger Hi-C matrix, StripePy can take advantage of multicore processors.

The maximum number of CPU cores use by StripePy can be changed through option ``--nproc`` (set to 1 core by default).

Whenever possible, we recommend using 4-8 CPU cores. Using more than 8 CPU cores is unlikely to result in significantly better computational performance (that is unless your Hi-C dataset is particularly dense).
