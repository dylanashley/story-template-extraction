#!/bin/bash

DIR=$(dirname "$0")
TASKS_PER_FILE=1

# assert command line arguments valid
if [ "$#" -gt "0" ]
    then
        echo 'usage: ./build.sh'
        exit
    fi

# begin amalgamating all tasks
TASKS_PREFIX='tasks_'
rm "$TASKS_PREFIX"*.sh 2>/dev/null
rm tasks.sh 2>/dev/null

###############################################################################

for SEED in `seq 0 9`; do
    for NUM_TEMPLATES in `seq 1 6`; do
        OUTFILE='valence_'"$NUM_TEMPLATES"'_'"$SEED"'.npy'
        ARGS=("--num-templates=$NUM_TEMPLATES"
              "--outfile=$OUTFILE"
              "--seed='$SEED'"
              "--no-plot")
        echo 'python -O '"$DIR"'/valence.py '"${ARGS[*]}" >> tasks.sh

        OUTFILE='narrative_essence_'"$NUM_TEMPLATES"'_'"$SEED"'.npy'
        ARGS=("--num-templates=$NUM_TEMPLATES"
              "--outfile=$OUTFILE"
              "--seed='$SEED'"
              "--no-plot")
        echo 'python -O '"$DIR"'/narrative_essence.py '"${ARGS[*]}" >> tasks.sh
    done
done

###############################################################################

# split tasks into files
perl -MList::Util=shuffle -e 'print shuffle(<STDIN>);' < tasks.sh > temp.sh
rm tasks.sh 2>/dev/null
split -l $TASKS_PER_FILE -a 3 temp.sh
rm temp.sh
AL=({a..z})
for i in `seq 0 25`; do
    for j in `seq 0 25`; do
        for k in `seq 0 25`; do
        FILE='x'"${AL[i]}${AL[j]}${AL[k]}"
        if [ -f $FILE ]; then
            ID=$((i * 26 * 26 + j * 26 + k))
            ID=${ID##+(0)}
            mv 'x'"${AL[i]}${AL[j]}${AL[k]}" "$TASKS_PREFIX""$ID"'.sh' 2>/dev/null
            chmod +x "$TASKS_PREFIX""$ID"'.sh' 2>/dev/null
        else
            break 3
        fi
        done
    done
done
echo $ID
