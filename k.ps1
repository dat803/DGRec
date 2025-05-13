$kernels = @("tanh", "linear", "radial", "poly", "gaussian")
$popularities = @("enabled", "inverse", "disabled")

python main.py --metrics recall,hit_ratio,coverage,IUD,DILAD,ILAD,DCC,FDCC --dataset TaoBao --kernel tanh --DCC_alpha 0.008 --FDCC_alpha 0.008 --DILAD_beta 0.004 --popularity enabled --k_list 10,20,100,300 --embedding_provider ".\embedding_providers\TaoBao\TaoBao_m_dgrec_lr_0.05_es_32_bs_2048_wd_8e-08_l_1_nn_4_s_2022_k_20_sigma_1.0_gamma_2.0_coef0_1.0_degree_2.0_bc_0.9_kf_tanh_sf_original_p_enabled_DCCa_0.008_FDCC_a_0.008_DILAD_b_0.004.pt"

foreach ($popularity in $popularities) {
    foreach ($kernel in $kernels) {
        $cmd = @"
python main.py --metrics recall,hit_ratio,coverage,IUD,DILAD,ILAD,DCC,FDCC --dataset TaoBao --kernel $kernel --DCC_alpha 0.008 --FDCC_alpha 0.008 --DILAD_beta 0.004 --popularity $popularity --k_list 10,20,100,300 --embedding_provider ".\embedding_providers\TaoBao\TaoBao_m_dgrec_lr_0.05_es_32_bs_2048_wd_8e-08_l_1_nn_4_s_2022_k_20_sigma_1.0_gamma_2.0_coef0_1.0_degree_2.0_bc_0.9_kf_tanh_sf_original_p_enabled_DCCa_0.008_FDCC_a_0.008_DILAD_b_0.004.pt" --model_path ".\embedding_providers\TaoBao\TaoBao_m_dgrec_lr_0.05_es_32_bs_2048_wd_8e-08_l_1_nn_4_s_2022_k_20_sigma_1.0_gamma_2.0_coef0_1.0_degree_2.0_bc_0.9_kf_${kernel}_sf_original_p_${popularity}_DCCa_0.008_FDCC_a_0.008_DILAD_b_0.004.pt"
"@ 

        Write-Output "Running with kernel=$kernel and popularity=$popularity"
        Invoke-Expression "$cmd"
    }
}


$kernels = @("tanh", "linear", "radial", "poly", "gaussian")
$popularities = @("enabled", "inverse", "disabled")

foreach ($popularity in $popularities) {
    foreach ($kernel in $kernels) {
        $cmd = @"
python main.py --metrics recall,hit_ratio,coverage,IUD,DILAD,ILAD,DCC,FDCC --dataset Beauty --kernel $kernel --DCC_alpha 0.008 --FDCC_alpha 0.008 --DILAD_beta 0.004 --popularity $popularity --k_list 10,20,100,300 --embedding_provider ".\embedding_providers\TaoBao\TaoBao_m_dgrec_lr_0.05_es_32_bs_2048_wd_8e-08_l_1_nn_4_s_2022_k_20_sigma_1.0_gamma_2.0_coef0_1.0_degree_2.0_bc_0.9_kf_tanh_sf_original_p_enabled_DCCa_0.008_FDCC_a_0.008_DILAD_b_0.004.pt"
"@ 

        Write-Output "Running with kernel=$kernel and popularity=$popularity"
        Invoke-Expression "$cmd"
    }
}

