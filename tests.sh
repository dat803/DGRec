python .\main.py --model_path "./best_models/2_baseline 2 - TaoBao_model_dgrec_lr_0.05_embed_size_32_batch_size_2048_weight_decay_8e-08_layers_1_neg_number_4_seed_2022_k_20_sigma_1.0_gamma_2.0_beta_class_0.9.pt" --metrics "recall,hit_ratio,DILAD,ILAD,IUD,DCC,FDCC"
python .\main.py --model_path "./best_models/4_TaoBao_model_dgrec_lr_0.05_embed_size_32_batch_size_2048_weight_decay_8e-08_layers_1_neg_number_4_seed_2022_k_20_sigma_1.0_gamma_2.0_beta_class_0.9_kernel_function_tanh_submodular_function_original.pt" --metrics "recall,hit_ratio,DILAD,ILAD,IUD,DCC,FDCC"
python .\main.py --model_path "./best_models/1_TaoBao_model_dgrec_lr_0.05_embed_size_32_batch_size_2048_weight_decay_8e-08_layers_1_neg_number_4_seed_2022_k_20_sigma_1.0_gamma_2.0_beta_class_0.9_kernel_function_poly_submodular_function_original_popularity_True.pt" --metrics "recall,hit_ratio,DILAD,ILAD,IUD,DCC,FDCC"
python .\main.py --model_path "./best_models/3_TaoBao_model_dgrec_lr_0.05_embed_size_32_batch_size_2048_weight_decay_8e-08_layers_1_neg_number_4_seed_2022_k_20_sigma_1.0_gamma_2.0_beta_class_0.9_kernel_function_linear_submodular_function_original.pt" --metrics "recall,hit_ratio,DILAD,ILAD,IUD,DCC,FDCC"
python .\main.py --popularity True --metrics "recall,hit_ratio,DILAD,ILAD,IUD,DCC,FDCC"

#python main.py --kernel poly;
#python main.py --kernel tanh;
#python main.py --dataset Beauty;
#python main.py --dataset Beauty --kernel poly;
#python main.py --dataset Beauty --kernel tanh;
#python main.py --epoch 1 --dataset Beauty --metrics "recall,hit_ratio,coverage,discount_coverage";