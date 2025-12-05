
..
  Copyright (C) 2025 Andrea Raffo <andrea.raffo@ibv.uio.no>
  SPDX-License-Identifier: MIT

CLI Reference
#############

For an up-to-date list of subcommands and CLI options refer to ``stripepy --help``.

.. _stripepy_help:

Subcommands
-----------

.. code-block:: text


  usage: stripepy {call,download,plot,view} ...
  stripepy is designed to recognize linear patterns in contact maps (.hic, .mcool, .cool) through the geometric reasoning, including topological persistence and quasi-interpolation.
  options:
    -h, --help            show this help message and exit
    --license             Print StripePy's license and return.
    --cite                Print StripePy's reference and return.
    -v, --version         show program's version number and exit
  subcommands:
    {call,download,plot,view}
                          List of available subcommands:
      call                stripepy works in four consecutive steps:
                          • Step 1: Pre-processing
                          • Step 2: Recognition of loci of interest (also called 'seeds')
                          • Step 3: Shape analysis (i.e., width and height estimation)
                          • Step 4: Signal analysis
      download            Helper command to simplify downloading datasets that can be used to test StripePy.
      plot                Generate various static plots useful to visually inspect the output produced by stripepy call.
      view                Fetch stripes from the HDF5 file produced by stripepy call.


.. _stripepy_call_help:

stripepy call
-------------

.. code-block:: text

  usage: stripepy call [-h] [-n NORMALIZATION] [-b GENOMIC_BELT]
                       [--roi {middle,start}] [-o OUTPUT_FILE]
                       [--log-file LOG_FILE] [--plot-dir PLOT_DIR]
                       [--max-width MAX_WIDTH] [--glob-pers-min GLOB_PERS_MIN]
                       [-k K] [--loc-pers-min LOC_PERS_MIN]
                       [--loc-trend-min LOC_TREND_MIN] [-f]
                       [--verbosity {debug,info,warning,error,critical}]
                       [-p NPROC] [--low-memory]
                       [--min-chrom-size MIN_CHROM_SIZE]
                       contact-map resolution
  positional arguments:
    contact-map           Path to a .cool, .mcool, or .hic file for input.
    resolution            Resolution (in bp).
  options:
    -h, --help            show this help message and exit
    -n, --normalization NORMALIZATION
                          Normalization to fetch (default: None).
    -b, --genomic-belt GENOMIC_BELT
                          Radius of the band, centred around the diagonal, where the search is restricted to (in bp, default: 5000000).
    --roi {middle,start}  Criterion used to select a region from each chromosome used to generate diagnostic plots (default: None).
                          Requires --plot-dir.
    -o, --output-file OUTPUT_FILE
                          Path where to store the output HDF5 file.
                          When not specified, the output file will be saved in the current working directory with a named based on the name of input matrix file.
    --log-file LOG_FILE   Path where to store the log file.
    --plot-dir PLOT_DIR   Path where to store the output plots.
                          Required when --roi is specified and ignored otherwise.
                          If the specified folder does not already exist, it will be created.
    --max-width MAX_WIDTH
                          Maximum stripe width, in bp (default: 100000).
    --glob-pers-min GLOB_PERS_MIN
                          Threshold value between 0 and 1 to filter persistence maxima points and identify loci of interest, aka seeds (default: 0.04).
    -k, --k-neighbour K   k for the k-neighbour, i.e., number of bins adjacent to the stripe boundaries on both sides (default: 3).
    --loc-pers-min LOC_PERS_MIN
                          Threshold value between 0 and 1 to find peaks in signal in a horizontal domain while estimating the height of a stripe (default: 0.33).
    --loc-trend-min LOC_TREND_MIN
                          Threshold value between 0 and 1 to estimate the height of a stripe (default: 0.25); the higher this value, the shorter the stripe; it is used to avoid overly long stripes when no persistent maximum besides the global one is found.
    -f, --force           Overwrite existing file(s) (default: False).
    --verbosity {debug,info,warning,error,critical}
                          Set verbosity of output to the console (default: info).
    -p, --nproc NPROC     Maximum number of parallel processes to use (default: 1).
    --low-memory          Minimize memory usage when fetching interactions from files (default: False).
    --min-chrom-size MIN_CHROM_SIZE
                          Minimum size, in bp, for a chromosome to be analysed (default: 2000000).


.. _stripepy_download_help:

stripepy download
-----------------

.. code-block:: text

  usage: stripepy download [-h] [--assembly {hg38,mm10} | --name NAME |
                           --list-only] [--unit-test | --end2end-test]
                           [--include-private] [--max-size MAX_SIZE]
                           [-o OUTPUT_PATH] [-f]
                           [--verbosity {debug,info,warning,error,critical}]
  options:
    -h, --help            show this help message and exit
    --assembly {hg38,mm10}
                          Restrict downloads to the given reference genome assembly.
    --name NAME           Name of the dataset to be downloaded.
                          When not provided, randomly select and download a dataset based on the provided CLI options (if any).
    --list-only           Print the list of available datasets and return (default: False).
    --unit-test           Download the test datasets required by the unit tests.
                          Files will be stored under folder test/data/
                          When specified, all other options are ignored.
                          Existing files will be overwritten.
    --end2end-test        Download the test datasets required by the end2end tests.
                          Files will be stored under folder test/data/
                          When specified, all other options are ignored.
                          Existing files will be overwritten.
    --include-private     Include datasets used for internal testing (default: False).
    --max-size MAX_SIZE   Upper bound for the size of the files to be considered when --name is not provided (default: 512.0).
    -o, --output OUTPUT_PATH
                          Path where to store the downloaded file (default: None).
    -f, --force           Overwrite existing file(s) (default: False).
    --verbosity {debug,info,warning,error,critical}
                          Set verbosity of output to the console (default: info).


.. _stripepy_plot_help:

stripepy plot
-------------

.. code-block:: text

  usage: stripepy plot [-h]
                       {contact-map,cm,pseudodistribution,pd,stripe-hist,hist} ...
  options:
    -h, --help            show this help message and exit
  plot_subcommands:
    {contact-map,cm,pseudodistribution,pd,stripe-hist,hist}
                          List of available subcommands:
      contact-map (cm)    Plot stripes and other features over the Hi-C matrix.
      pseudodistribution (pd)
                          Plot the pseudo-distribution over the given region of interest.
      stripe-hist (hist)  Generate and plot the histograms showing the distribution of the stripe heights and widths.


.. _stripepy_plot_contact_map_help:

stripepy plot contact-map
-------------------------

.. code-block:: text

  usage: stripepy plot contact-map [-h] [--stripepy-hdf5 STRIPEPY_HDF5]
                                   [--relative-change-threshold RELATIVE_CHANGE_THRESHOLD]
                                   [--coefficient-of-variation-threshold COEFFICIENT_OF_VARIATION_THRESHOLD]
                                   [--highlight-seeds | --highlight-stripes]
                                   [--ignore-stripe-heights] [--cmap CMAP]
                                   [--linear-scale | --log-scale]
                                   [--region REGION] [--dpi DPI] [--seed SEED]
                                   [-n NORMALIZATION] [-f]
                                   [--verbosity {debug,info,warning,error,critical}]
                                   contact-map resolution output-name
  positional arguments:
    contact-map           Path to the .cool, .mcool, or .hic file used to call stripes.
    resolution            Resolution (in bp).
    output-name           Path where to store the generated plot.
  options:
    -h, --help            show this help message and exit
    --stripepy-hdf5 STRIPEPY_HDF5
                          Path to the .hdf5 generated by stripepy call.
                          Required when highlighting stripes or seeds.
    --relative-change-threshold RELATIVE_CHANGE_THRESHOLD
                          Cutoff for the relative change (default: None).
                          Only used when highlighting architectural stripes.
                          The relative change is computed as the ratio between the average number of interactions found inside a stripe and the number of interactions in a neighborhood outside of the stripe.
    --coefficient-of-variation-threshold COEFFICIENT_OF_VARIATION_THRESHOLD
                          Cutoff for the coefficient of variation (default: None).
                          Only used when highlighting architectural stripes.
                          The coefficient of variation is computed as the ratio between the standard deviation and the mean
                          of the values inside a stripe. In our case, it is always nonnegative because of the preprocessing step.
    --highlight-seeds     Highlight the stripe seeds (default: False).
    --highlight-stripes   Highlight the architectural stripes (default: False).
    --ignore-stripe-heights
                          Ignore the stripes height (default: False).
                          Has no effect when --highlight-stripes is not specified.
    --cmap CMAP           Color map used to plot Hi-C interactions (default: fruit_punch).
                          Can be any of the color maps supported by matplotlib as well as: fall, fruit_punch, blues, acidblues, and nmeth.
    --linear-scale        Plot interactions in linear scale (default: False).
    --log-scale           Plot interactions in log scale (default: True).
    --region REGION       Genomic region to be plotted (UCSC format). When not specified, a random 2.5Mb region is plotted.
    --dpi DPI             DPI of the generated plot (default: 300; ignored when the output format is a vector graphic).
    --seed SEED           Seed for random number generation (default: 7606490399616306585).
    -n, --normalization NORMALIZATION
                          Normalization to fetch (default: None).
    -f, --force           Overwrite existing file(s) (default: False).
    --verbosity {debug,info,warning,error,critical}
                          Set verbosity of output to the console (default: info).


.. _stripepy_plot_pseudodistribution_help:

stripepy plot pseudodistribution
--------------------------------

.. code-block:: text

  usage: stripepy plot pseudodistribution [-h] [--region REGION] [--dpi DPI]
                                          [--seed SEED] [-n NORMALIZATION] [-f]
                                          [--verbosity {debug,info,warning,error,critical}]
                                          stripepy-hdf5 output-name
  positional arguments:
    stripepy-hdf5         Path to the .hdf5 generated by stripepy call.
    output-name           Path where to store the generated plot.
  options:
    -h, --help            show this help message and exit
    --region REGION       Genomic region to be plotted (UCSC format). When not specified, a random 2.5Mb region is plotted.
    --dpi DPI             DPI of the generated plot (default: 300; ignored when the output format is a vector graphic).
    --seed SEED           Seed for random number generation (default: 7606490399616306585).
    -n, --normalization NORMALIZATION
                          Normalization to fetch (default: None).
    -f, --force           Overwrite existing file(s) (default: False).
    --verbosity {debug,info,warning,error,critical}
                          Set verbosity of output to the console (default: info).


.. _stripepy_plot_stripe_hist_help:

stripepy plot stripe-hist
-------------------------

.. code-block:: text

  usage: stripepy plot stripe-hist [-h] [--region REGION] [--dpi DPI]
                                   [--seed SEED] [-n NORMALIZATION] [-f]
                                   [--verbosity {debug,info,warning,error,critical}]
                                   stripepy-hdf5 output-name
  positional arguments:
    stripepy-hdf5         Path to the .hdf5 generated by stripepy call.
    output-name           Path where to store the generated plot.
  options:
    -h, --help            show this help message and exit
    --region REGION       Genomic region to be plotted (UCSC format). When not specified, data for the entire genome is plotted.
    --dpi DPI             DPI of the generated plot (default: 300; ignored when the output format is a vector graphic).
    --seed SEED           Seed for random number generation (default: 7606490399616306585).
    -n, --normalization NORMALIZATION
                          Normalization to fetch (default: None).
    -f, --force           Overwrite existing file(s) (default: False).
    --verbosity {debug,info,warning,error,critical}
                          Set verbosity of output to the console (default: info).


.. _stripepy_view_help:

stripepy view
-------------

.. code-block:: text

  usage: stripepy view [-h]
                       [--relative-change-threshold RELATIVE_CHANGE_THRESHOLD]
                       [--coefficient-of-variation-threshold COEFFICIENT_OF_VARIATION_THRESHOLD]
                       [--with-biodescriptors] [--with-header]
                       [--transform {None,transpose_to_lt,transpose_to_ut}]
                       [--verbosity {debug,info,warning,error,critical}]
                       h5-file
  positional arguments:
    h5-file               Path to the HDF5 file generated by stripepy call.
  options:
    -h, --help            show this help message and exit
    --relative-change-threshold RELATIVE_CHANGE_THRESHOLD
                          Cutoff for the relative change (default: 5.0).
                          The relative change is computed as the ratio between the average number of interactions
                          found inside a stripe and the number of interactions in a neighborhood outside of the stripe.
    --coefficient-of-variation-threshold COEFFICIENT_OF_VARIATION_THRESHOLD
                          Cutoff for the coefficient of variation (default: None).
                          The coefficient of variation is computed as the ratio between the standard deviation and the mean
                          of the values inside a stripe. In our case, it is always nonnegative because of the preprocessing step.
    --with-biodescriptors
                          Include the stripe biodescriptors in the output.
    --with-header         Include column names in the output.
    --transform {None,transpose_to_lt,transpose_to_ut}
                          Control if and how stripe coordinates should be transformed (default: None).
    --verbosity {debug,info,warning,error,critical}
                          Set verbosity of output to the console (default: info).
