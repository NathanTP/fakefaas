# Function-as-a-Service Sorter
This is a FaaS interface to libsort for performing distributed radix sort.

## Requirements
This implementation requires the 'sharedfs' branch of open-lambda which includes
GPU support as well as shared filesystem mounting.

## Quick Start
First, install the python requirements using anaconda or pip:

  pip install -r requirements.txt

To test locally, you can invoke f.py directly. Make sure your LD\_LIBRARY\_PATH
contains libsort.so (../env.sh helps with this).

  python3 f.py

You should see 'PASS' printed.

You can now install the function using srk (assuming 'srk' is on your path):

  make install

There is currently no test for invoking f.py via SRK, use the Go-based
benchmark included in this repo.

## Protocol
Requests take the form of lists of partRefs to process and an output key to use
in an output DistributedArray. Sorters will output buckets as partitions of
this array. The handlers expect JSON-encoded arguments.

### Common Fields
  - "offset" - The starting bit index to start sorting
  - "width" - The number of radix bits to process
  - "arrType" - The type of distributed array used for exchanging data.
  - "input" - A list of JSON-encoded partRefs. The exact format of these arguments depends on "arrType" (see below).
  - "output" - An identifier to use for storing output. The meaning of this fields depends on "arrType" (see below).

### File Distributed Array
A file distributed array uses the filesystem to exchange data. The system must
ensure that the filesystem is shared between the requestor and the worker
(pylibsort does not handle mounting).

#### Arguments
The arrType field must be set to 'file'

Each element of input is a partRef map with the following fields:
  - "arrayName" - Directory name for this FileDistributedArray. The search path
      for this array depends on how the FaaS system was configured, but is
      assumed to be shared between the host and the FaaS executor.
  - "partID" - The numeric ID of the partition, this will be converted into "arrayPath/p${partID}.dat"
  - "start" - The byte index to start reading the partition from.
  - "nbyte" - The number of bytes to read. May be -1 to read the remainder of the partition (from start)

This worker will store its outputs in the filesystem at the path indicated by
the the 'output' field. As in partRefs, the exact interpretation of the path depends on
system configuration.

#### Shared FS Mounting
Pylibsort uses '/shared' by default for FileDistribArrays, but this can be
overridden by pylibsort.SetDistribMount(). The 'sharedfs' branch of OpenLambda
will mount the shared filesystem at '/shared' for the function. When running
openlambda (e.g. through SRK), you should set the OL\_SHARED\_VOLUME
environment variable to the local directory to use for sharing.
