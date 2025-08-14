set -e
python convert_dataset.py datasets/demo/rel-amazon-input datasets/demo/test-amazon-output --sampling_rate 0.3 --activity_bias_factor 2.0 --today_date 2016-01-01
python convert_dataset.py datasets/demo/rel-stack-input datasets/demo/test-stack-output --sampling_rate 0.3 --activity_bias_factor 2.0 --today_date 2021-01-01

#for DATASET in rel-amazon rel-avito rel-event rel-f1 rel-hm rel-stack rel-trial; do
for DATASET in demo/test-amazon-output demo/test-stack-output; do
    python -m tab2graph.main preprocess datasets/$DATASET transform datasets/$DATASET-pre-dfs -c configs/transform/pre-dfs-tabpfn.yaml
    #python -m tab2graph.main preprocess datasets/$DATASET-pre-dfs transform datasets/$DATASET-post-dfs-0 -c configs/transform/post-dfs-tabpfn2.yaml
    #for DEPTH in 1 2 3; do
    for DEPTH in 3; do
       python -m tab2graph.main preprocess datasets/$DATASET-pre-dfs dfs datasets/$DATASET-post-dfs -c configs/dfs/dfs-$DEPTH-sql.yaml
       python -m tab2graph.main preprocess datasets/$DATASET-post-dfs transform datasets/$DATASET-post-dfs-$DEPTH -c configs/transform/post-dfs-tabpfn2.yaml
    done
done

find datasets/demo/test-amazon-output-post-dfs-3 -name train.npz | while read f; do g=$(basename $(dirname $f)); mv $f datasets/demo/rel-amazon-input/__dfs__/${g%%-template}.npz; done
find datasets/demo/test-stack-output-post-dfs-3 -name train.npz | while read f; do g=$(basename $(dirname $f)); mv $f datasets/demo/rel-stack-input/__dfs__/${g%%-template}.npz; done
