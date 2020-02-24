for i in {0..0}
do
    CUDA_VISIBLE_DEVICES=1  python lstm-evaluation-2.py  a${i} > ../data/results/results-a${i}.txt &
    CUDA_VISIBLE_DEVICES=1  python lstm-evaluation-2.py  b${i} > ../data/results/results-b${i}.txt &
    CUDA_VISIBLE_DEVICES=1  python lstm-evaluation-2.py  c${i} > ../data/results/results-c${i}.txt
done