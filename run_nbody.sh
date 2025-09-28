#!/bin/bash

run=true
run_compare=false
compare_only=false
get_flag_list=false

cd "$(dirname "$0")"
PathToMilkyWayAtHomeClientDirectory="$(pwd)"
echo "Path to milkywayathome_client directory: $PathToMilkyWayAtHomeClientDirectory"

cd build/bin

if $run 
then
    ./milkyway_nbody \
    -f $PathToMilkyWayAtHomeClientDirectory/nbody/sample_workunits/for_developers.lua \
    -o $PathToMilkyWayAtHomeClientDirectory/output/output.out \
    -z $PathToMilkyWayAtHomeClientDirectory/output/output.hist \
    -n 8 -w 1 -P -e 54231651 \
    -i 4.0 1.0 0.2 0.2 12.0 0.2 \
    
fi

if $run_compare
then
    ./milkyway_nbody \
    -f $PathToMilkyWayAtHomeClientDirectory/nbody/sample_workunits/for_developers.lua \
    -o $PathToMilkyWayAtHomeClientDirectory/output/output.out \
    -z $PathToMilkyWayAtHomeClientDirectory/output/output.hist \
    -h $PathToMilkyWayAtHomeClientDirectory/input/input.hist \
    -n 8 -w 1 -P -e 54231651 \
    -p 4.0 1.0 0.2 0.2 12.0 0.2 \

fi

#SMU = 222,288.47 SOLAR MASSES

#OPTIONS:
#-s -> The histogram to compare. Will always compare using emd and cost component
#ADD ADDITIONAL FLAGS TO INCLUDE OTHER COMPARISONS
#-S -> Include beta dispersion in the comparison
#-V -> Include velocity dispersion in the comparison
#-B -> Include beta average in the comparison
#-Q -> Include line of sight velocity average in the comparison
#-D -> Include distance average in the comparison
#-U -> Include proper motion average in the comparison
#-L -> Include momentum in the comparison
#Values input through the histogram, such as EMDRange, will be read from the input histogram given with -h
if $compare_only 
then
    ./milkyway_nbody \
    -h $PathToMilkyWayAtHomeClientDirectory/input/input.hist \
    -s $PathToMilkyWayAtHomeClientDirectory/output/output.hist \

fi

# if you run:

if $get_flag_list
then
    ./milkyway_nbody --help
# it will show you what all the flags mean
fi
