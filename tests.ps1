python main.py --kernel poly;
python main.py --kernel tanh;
python main.py --dataset Beauty;
python main.py --dataset Beauty --kernel poly;
python main.py --dataset Beauty --kernel tanh;
python main.py --epoch 1 --dataset Beauty --metrics "recall,hit_ratio,coverage,discount_coverage";