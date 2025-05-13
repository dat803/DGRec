# 5 tests for our different kernels
python main.py `
  --metrics recall,hit_ratio,coverage,IUD,DILAD,ILAD,DCC,FDCC `
  --dataset TaoBao `
  --kernel tanh `
  --DCC_alpha 0.008 `
  --FDCC_alpha 0.008 `
  --DILAD_beta 0.004 `
  --k_list 10,20,100,300

python main.py `
  --metrics recall,hit_ratio,coverage,IUD,DILAD,ILAD,DCC,FDCC `
  --dataset TaoBao `
  --kernel linear `
  --DCC_alpha 0.008 `
  --FDCC_alpha 0.008 `
  --DILAD_beta 0.004 `
  --k_list 10,20,100,300

python main.py `
  --metrics recall,hit_ratio,coverage,IUD,DILAD,ILAD,DCC,FDCC `
  --dataset TaoBao `
  --kernel poly `
  --DCC_alpha 0.008 `
  --FDCC_alpha 0.008 `
  --DILAD_beta 0.004 `
  --k_list 10,20,100,300

python main.py `
  --metrics recall,hit_ratio,coverage,IUD,DILAD,ILAD,DCC,FDCC `
  --dataset TaoBao `
  --kernel radial `
  --DCC_alpha 0.008 `
  --FDCC_alpha 0.008 `
  --DILAD_beta 0.004 `
  --k_list 10,20,100,300

python main.py `
  --metrics recall,hit_ratio,coverage,IUD,DILAD,ILAD,DCC,FDCC `
  --dataset TaoBao `
  --kernel gaussian `
  --DCC_alpha 0.008 `
  --FDCC_alpha 0.008 `
  --DILAD_beta 0.004 `
  --k_list 10,20,100,300

# Tests for gaussian kernel but with, without and inverse popularity (without is default just above),
python main.py `
  --metrics recall,hit_ratio,coverage,IUD,DILAD,ILAD,DCC,FDCC `
  --dataset TaoBao `
  --kernel gaussian `
  --DCC_alpha 0.008 `
  --FDCC_alpha 0.008 `
  --DILAD_beta 0.004 `
  --popularity enabled `
  --k_list 10,20,100,300

python main.py `
  --metrics recall,hit_ratio,coverage,IUD,DILAD,ILAD,DCC,FDCC `
  --dataset TaoBao `
  --kernel gaussian `
  --DCC_alpha 0.008 `
  --FDCC_alpha 0.008 `
  --DILAD_beta 0.004 `
  --popularity inverse `
  --k_list 10,20,100,300