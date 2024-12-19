# Schema Docs

- [1. Property `ReduxOptions > name`](#name)
- [2. Property `ReduxOptions > scale`](#scale)
- [3. Property `ReduxOptions > num_gpu`](#num_gpu)
  - [3.1. Property `ReduxOptions > num_gpu > anyOf > item 0`](#num_gpu_anyOf_i0)
  - [3.2. Property `ReduxOptions > num_gpu > anyOf > item 1`](#num_gpu_anyOf_i1)
- [4. Property `ReduxOptions > path`](#path)
  - [4.1. Property `ReduxOptions > path > experiments_root`](#path_experiments_root)
    - [4.1.1. Property `ReduxOptions > path > experiments_root > anyOf > item 0`](#path_experiments_root_anyOf_i0)
    - [4.1.2. Property `ReduxOptions > path > experiments_root > anyOf > item 1`](#path_experiments_root_anyOf_i1)
  - [4.2. Property `ReduxOptions > path > models`](#path_models)
    - [4.2.1. Property `ReduxOptions > path > models > anyOf > item 0`](#path_models_anyOf_i0)
    - [4.2.2. Property `ReduxOptions > path > models > anyOf > item 1`](#path_models_anyOf_i1)
  - [4.3. Property `ReduxOptions > path > resume_models`](#path_resume_models)
    - [4.3.1. Property `ReduxOptions > path > resume_models > anyOf > item 0`](#path_resume_models_anyOf_i0)
    - [4.3.2. Property `ReduxOptions > path > resume_models > anyOf > item 1`](#path_resume_models_anyOf_i1)
  - [4.4. Property `ReduxOptions > path > training_states`](#path_training_states)
    - [4.4.1. Property `ReduxOptions > path > training_states > anyOf > item 0`](#path_training_states_anyOf_i0)
    - [4.4.2. Property `ReduxOptions > path > training_states > anyOf > item 1`](#path_training_states_anyOf_i1)
  - [4.5. Property `ReduxOptions > path > log`](#path_log)
    - [4.5.1. Property `ReduxOptions > path > log > anyOf > item 0`](#path_log_anyOf_i0)
    - [4.5.2. Property `ReduxOptions > path > log > anyOf > item 1`](#path_log_anyOf_i1)
  - [4.6. Property `ReduxOptions > path > visualization`](#path_visualization)
    - [4.6.1. Property `ReduxOptions > path > visualization > anyOf > item 0`](#path_visualization_anyOf_i0)
    - [4.6.2. Property `ReduxOptions > path > visualization > anyOf > item 1`](#path_visualization_anyOf_i1)
  - [4.7. Property `ReduxOptions > path > results_root`](#path_results_root)
    - [4.7.1. Property `ReduxOptions > path > results_root > anyOf > item 0`](#path_results_root_anyOf_i0)
    - [4.7.2. Property `ReduxOptions > path > results_root > anyOf > item 1`](#path_results_root_anyOf_i1)
  - [4.8. Property `ReduxOptions > path > pretrain_network_g`](#path_pretrain_network_g)
    - [4.8.1. Property `ReduxOptions > path > pretrain_network_g > anyOf > item 0`](#path_pretrain_network_g_anyOf_i0)
    - [4.8.2. Property `ReduxOptions > path > pretrain_network_g > anyOf > item 1`](#path_pretrain_network_g_anyOf_i1)
  - [4.9. Property `ReduxOptions > path > pretrain_network_g_path`](#path_pretrain_network_g_path)
    - [4.9.1. Property `ReduxOptions > path > pretrain_network_g_path > anyOf > item 0`](#path_pretrain_network_g_path_anyOf_i0)
    - [4.9.2. Property `ReduxOptions > path > pretrain_network_g_path > anyOf > item 1`](#path_pretrain_network_g_path_anyOf_i1)
  - [4.10. Property `ReduxOptions > path > param_key_g`](#path_param_key_g)
    - [4.10.1. Property `ReduxOptions > path > param_key_g > anyOf > item 0`](#path_param_key_g_anyOf_i0)
    - [4.10.2. Property `ReduxOptions > path > param_key_g > anyOf > item 1`](#path_param_key_g_anyOf_i1)
  - [4.11. Property `ReduxOptions > path > strict_load_g`](#path_strict_load_g)
  - [4.12. Property `ReduxOptions > path > resume_state`](#path_resume_state)
    - [4.12.1. Property `ReduxOptions > path > resume_state > anyOf > item 0`](#path_resume_state_anyOf_i0)
    - [4.12.2. Property `ReduxOptions > path > resume_state > anyOf > item 1`](#path_resume_state_anyOf_i1)
  - [4.13. Property `ReduxOptions > path > pretrain_network_g_ema`](#path_pretrain_network_g_ema)
    - [4.13.1. Property `ReduxOptions > path > pretrain_network_g_ema > anyOf > item 0`](#path_pretrain_network_g_ema_anyOf_i0)
    - [4.13.2. Property `ReduxOptions > path > pretrain_network_g_ema > anyOf > item 1`](#path_pretrain_network_g_ema_anyOf_i1)
  - [4.14. Property `ReduxOptions > path > pretrain_network_d`](#path_pretrain_network_d)
    - [4.14.1. Property `ReduxOptions > path > pretrain_network_d > anyOf > item 0`](#path_pretrain_network_d_anyOf_i0)
    - [4.14.2. Property `ReduxOptions > path > pretrain_network_d > anyOf > item 1`](#path_pretrain_network_d_anyOf_i1)
  - [4.15. Property `ReduxOptions > path > param_key_d`](#path_param_key_d)
    - [4.15.1. Property `ReduxOptions > path > param_key_d > anyOf > item 0`](#path_param_key_d_anyOf_i0)
    - [4.15.2. Property `ReduxOptions > path > param_key_d > anyOf > item 1`](#path_param_key_d_anyOf_i1)
  - [4.16. Property `ReduxOptions > path > strict_load_d`](#path_strict_load_d)
  - [4.17. Property `ReduxOptions > path > ignore_resume_networks`](#path_ignore_resume_networks)
    - [4.17.1. Property `ReduxOptions > path > ignore_resume_networks > anyOf > item 0`](#path_ignore_resume_networks_anyOf_i0)
      - [4.17.1.1. ReduxOptions > path > ignore_resume_networks > anyOf > item 0 > item 0 items](#path_ignore_resume_networks_anyOf_i0_items)
    - [4.17.2. Property `ReduxOptions > path > ignore_resume_networks > anyOf > item 1`](#path_ignore_resume_networks_anyOf_i1)
- [5. Property `ReduxOptions > network_g`](#network_g)
  - [5.1. Property `ReduxOptions > network_g > anyOf > artcnn`](#network_g_anyOf_i0)
    - [5.1.1. Property `ReduxOptions > network_g > anyOf > artcnn > type`](#network_g_anyOf_i0_type)
    - [5.1.2. Property `ReduxOptions > network_g > anyOf > artcnn > in_ch`](#network_g_anyOf_i0_in_ch)
    - [5.1.3. Property `ReduxOptions > network_g > anyOf > artcnn > scale`](#network_g_anyOf_i0_scale)
    - [5.1.4. Property `ReduxOptions > network_g > anyOf > artcnn > filters`](#network_g_anyOf_i0_filters)
    - [5.1.5. Property `ReduxOptions > network_g > anyOf > artcnn > n_block`](#network_g_anyOf_i0_n_block)
    - [5.1.6. Property `ReduxOptions > network_g > anyOf > artcnn > kernel_size`](#network_g_anyOf_i0_kernel_size)
  - [5.2. Property `ReduxOptions > network_g > anyOf > artcnn_r16f96`](#network_g_anyOf_i1)
    - [5.2.1. Property `ReduxOptions > network_g > anyOf > artcnn_r16f96 > type`](#network_g_anyOf_i1_type)
  - [5.3. Property `ReduxOptions > network_g > anyOf > artcnn_r8f64`](#network_g_anyOf_i2)
    - [5.3.1. Property `ReduxOptions > network_g > anyOf > artcnn_r8f64 > type`](#network_g_anyOf_i2_type)
  - [5.4. Property `ReduxOptions > network_g > anyOf > camixersr`](#network_g_anyOf_i3)
    - [5.4.1. Property `ReduxOptions > network_g > anyOf > camixersr > type`](#network_g_anyOf_i3_type)
  - [5.5. Property `ReduxOptions > network_g > anyOf > cfsr`](#network_g_anyOf_i4)
    - [5.5.1. Property `ReduxOptions > network_g > anyOf > cfsr > type`](#network_g_anyOf_i4_type)
  - [5.6. Property `ReduxOptions > network_g > anyOf > vggstylediscriminator`](#network_g_anyOf_i5)
    - [5.6.1. Property `ReduxOptions > network_g > anyOf > vggstylediscriminator > type`](#network_g_anyOf_i5_type)
    - [5.6.2. Property `ReduxOptions > network_g > anyOf > vggstylediscriminator > num_in_ch`](#network_g_anyOf_i5_num_in_ch)
    - [5.6.3. Property `ReduxOptions > network_g > anyOf > vggstylediscriminator > num_feat`](#network_g_anyOf_i5_num_feat)
    - [5.6.4. Property `ReduxOptions > network_g > anyOf > vggstylediscriminator > input_size`](#network_g_anyOf_i5_input_size)
  - [5.7. Property `ReduxOptions > network_g > anyOf > unetdiscriminatorsn_traiNNer`](#network_g_anyOf_i6)
    - [5.7.1. Property `ReduxOptions > network_g > anyOf > unetdiscriminatorsn_traiNNer > type`](#network_g_anyOf_i6_type)
    - [5.7.2. Property `ReduxOptions > network_g > anyOf > unetdiscriminatorsn_traiNNer > num_in_ch`](#network_g_anyOf_i6_num_in_ch)
    - [5.7.3. Property `ReduxOptions > network_g > anyOf > unetdiscriminatorsn_traiNNer > num_feat`](#network_g_anyOf_i6_num_feat)
    - [5.7.4. Property `ReduxOptions > network_g > anyOf > unetdiscriminatorsn_traiNNer > skip_connection`](#network_g_anyOf_i6_skip_connection)
  - [5.8. Property `ReduxOptions > network_g > anyOf > dunet`](#network_g_anyOf_i7)
    - [5.8.1. Property `ReduxOptions > network_g > anyOf > dunet > type`](#network_g_anyOf_i7_type)
    - [5.8.2. Property `ReduxOptions > network_g > anyOf > dunet > num_in_ch`](#network_g_anyOf_i7_num_in_ch)
    - [5.8.3. Property `ReduxOptions > network_g > anyOf > dunet > num_feat`](#network_g_anyOf_i7_num_feat)
  - [5.9. Property `ReduxOptions > network_g > anyOf > eimn`](#network_g_anyOf_i8)
    - [5.9.1. Property `ReduxOptions > network_g > anyOf > eimn > type`](#network_g_anyOf_i8_type)
    - [5.9.2. Property `ReduxOptions > network_g > anyOf > eimn > embed_dims`](#network_g_anyOf_i8_embed_dims)
    - [5.9.3. Property `ReduxOptions > network_g > anyOf > eimn > scale`](#network_g_anyOf_i8_scale)
    - [5.9.4. Property `ReduxOptions > network_g > anyOf > eimn > depths`](#network_g_anyOf_i8_depths)
    - [5.9.5. Property `ReduxOptions > network_g > anyOf > eimn > mlp_ratios`](#network_g_anyOf_i8_mlp_ratios)
    - [5.9.6. Property `ReduxOptions > network_g > anyOf > eimn > drop_rate`](#network_g_anyOf_i8_drop_rate)
    - [5.9.7. Property `ReduxOptions > network_g > anyOf > eimn > drop_path_rate`](#network_g_anyOf_i8_drop_path_rate)
    - [5.9.8. Property `ReduxOptions > network_g > anyOf > eimn > num_stages`](#network_g_anyOf_i8_num_stages)
    - [5.9.9. Property `ReduxOptions > network_g > anyOf > eimn > freeze_param`](#network_g_anyOf_i8_freeze_param)
  - [5.10. Property `ReduxOptions > network_g > anyOf > eimn_l`](#network_g_anyOf_i9)
    - [5.10.1. Property `ReduxOptions > network_g > anyOf > eimn_l > type`](#network_g_anyOf_i9_type)
  - [5.11. Property `ReduxOptions > network_g > anyOf > eimn_a`](#network_g_anyOf_i10)
    - [5.11.1. Property `ReduxOptions > network_g > anyOf > eimn_a > type`](#network_g_anyOf_i10_type)
  - [5.12. Property `ReduxOptions > network_g > anyOf > flexnet`](#network_g_anyOf_i11)
    - [5.12.1. Property `ReduxOptions > network_g > anyOf > flexnet > type`](#network_g_anyOf_i11_type)
    - [5.12.2. Property `ReduxOptions > network_g > anyOf > flexnet > inp_channels`](#network_g_anyOf_i11_inp_channels)
    - [5.12.3. Property `ReduxOptions > network_g > anyOf > flexnet > out_channels`](#network_g_anyOf_i11_out_channels)
    - [5.12.4. Property `ReduxOptions > network_g > anyOf > flexnet > scale`](#network_g_anyOf_i11_scale)
    - [5.12.5. Property `ReduxOptions > network_g > anyOf > flexnet > dim`](#network_g_anyOf_i11_dim)
    - [5.12.6. Property `ReduxOptions > network_g > anyOf > flexnet > num_blocks`](#network_g_anyOf_i11_num_blocks)
      - [5.12.6.1. ReduxOptions > network_g > anyOf > flexnet > num_blocks > num_blocks items](#network_g_anyOf_i11_num_blocks_items)
    - [5.12.7. Property `ReduxOptions > network_g > anyOf > flexnet > window_size`](#network_g_anyOf_i11_window_size)
    - [5.12.8. Property `ReduxOptions > network_g > anyOf > flexnet > hidden_rate`](#network_g_anyOf_i11_hidden_rate)
    - [5.12.9. Property `ReduxOptions > network_g > anyOf > flexnet > channel_norm`](#network_g_anyOf_i11_channel_norm)
    - [5.12.10. Property `ReduxOptions > network_g > anyOf > flexnet > attn_drop`](#network_g_anyOf_i11_attn_drop)
    - [5.12.11. Property `ReduxOptions > network_g > anyOf > flexnet > proj_drop`](#network_g_anyOf_i11_proj_drop)
    - [5.12.12. Property `ReduxOptions > network_g > anyOf > flexnet > pipeline_type`](#network_g_anyOf_i11_pipeline_type)
    - [5.12.13. Property `ReduxOptions > network_g > anyOf > flexnet > upsampler`](#network_g_anyOf_i11_upsampler)
  - [5.13. Property `ReduxOptions > network_g > anyOf > metaflexnet`](#network_g_anyOf_i12)
    - [5.13.1. Property `ReduxOptions > network_g > anyOf > metaflexnet > type`](#network_g_anyOf_i12_type)
  - [5.14. Property `ReduxOptions > network_g > anyOf > hit_sir`](#network_g_anyOf_i13)
    - [5.14.1. Property `ReduxOptions > network_g > anyOf > hit_sir > type`](#network_g_anyOf_i13_type)
  - [5.15. Property `ReduxOptions > network_g > anyOf > hit_sng`](#network_g_anyOf_i14)
    - [5.15.1. Property `ReduxOptions > network_g > anyOf > hit_sng > type`](#network_g_anyOf_i14_type)
  - [5.16. Property `ReduxOptions > network_g > anyOf > hit_srf`](#network_g_anyOf_i15)
    - [5.16.1. Property `ReduxOptions > network_g > anyOf > hit_srf > type`](#network_g_anyOf_i15_type)
  - [5.17. Property `ReduxOptions > network_g > anyOf > ipt`](#network_g_anyOf_i16)
    - [5.17.1. Property `ReduxOptions > network_g > anyOf > ipt > type`](#network_g_anyOf_i16_type)
  - [5.18. Property `ReduxOptions > network_g > anyOf > lmlt`](#network_g_anyOf_i17)
    - [5.18.1. Property `ReduxOptions > network_g > anyOf > lmlt > type`](#network_g_anyOf_i17_type)
    - [5.18.2. Property `ReduxOptions > network_g > anyOf > lmlt > dim`](#network_g_anyOf_i17_dim)
    - [5.18.3. Property `ReduxOptions > network_g > anyOf > lmlt > n_blocks`](#network_g_anyOf_i17_n_blocks)
    - [5.18.4. Property `ReduxOptions > network_g > anyOf > lmlt > ffn_scale`](#network_g_anyOf_i17_ffn_scale)
    - [5.18.5. Property `ReduxOptions > network_g > anyOf > lmlt > scale`](#network_g_anyOf_i17_scale)
    - [5.18.6. Property `ReduxOptions > network_g > anyOf > lmlt > drop_rate`](#network_g_anyOf_i17_drop_rate)
    - [5.18.7. Property `ReduxOptions > network_g > anyOf > lmlt > attn_drop_rate`](#network_g_anyOf_i17_attn_drop_rate)
    - [5.18.8. Property `ReduxOptions > network_g > anyOf > lmlt > drop_path_rate`](#network_g_anyOf_i17_drop_path_rate)
  - [5.19. Property `ReduxOptions > network_g > anyOf > lmlt_base`](#network_g_anyOf_i18)
    - [5.19.1. Property `ReduxOptions > network_g > anyOf > lmlt_base > type`](#network_g_anyOf_i18_type)
  - [5.20. Property `ReduxOptions > network_g > anyOf > lmlt_large`](#network_g_anyOf_i19)
    - [5.20.1. Property `ReduxOptions > network_g > anyOf > lmlt_large > type`](#network_g_anyOf_i19_type)
  - [5.21. Property `ReduxOptions > network_g > anyOf > lmlt_tiny`](#network_g_anyOf_i20)
    - [5.21.1. Property `ReduxOptions > network_g > anyOf > lmlt_tiny > type`](#network_g_anyOf_i20_type)
  - [5.22. Property `ReduxOptions > network_g > anyOf > man`](#network_g_anyOf_i21)
    - [5.22.1. Property `ReduxOptions > network_g > anyOf > man > type`](#network_g_anyOf_i21_type)
    - [5.22.2. Property `ReduxOptions > network_g > anyOf > man > n_resblocks`](#network_g_anyOf_i21_n_resblocks)
    - [5.22.3. Property `ReduxOptions > network_g > anyOf > man > n_resgroups`](#network_g_anyOf_i21_n_resgroups)
    - [5.22.4. Property `ReduxOptions > network_g > anyOf > man > n_colors`](#network_g_anyOf_i21_n_colors)
    - [5.22.5. Property `ReduxOptions > network_g > anyOf > man > n_feats`](#network_g_anyOf_i21_n_feats)
    - [5.22.6. Property `ReduxOptions > network_g > anyOf > man > scale`](#network_g_anyOf_i21_scale)
    - [5.22.7. Property `ReduxOptions > network_g > anyOf > man > res_scale`](#network_g_anyOf_i21_res_scale)
  - [5.23. Property `ReduxOptions > network_g > anyOf > man_tiny`](#network_g_anyOf_i22)
    - [5.23.1. Property `ReduxOptions > network_g > anyOf > man_tiny > type`](#network_g_anyOf_i22_type)
  - [5.24. Property `ReduxOptions > network_g > anyOf > man_light`](#network_g_anyOf_i23)
    - [5.24.1. Property `ReduxOptions > network_g > anyOf > man_light > type`](#network_g_anyOf_i23_type)
  - [5.25. Property `ReduxOptions > network_g > anyOf > moesr2`](#network_g_anyOf_i24)
    - [5.25.1. Property `ReduxOptions > network_g > anyOf > moesr2 > type`](#network_g_anyOf_i24_type)
    - [5.25.2. Property `ReduxOptions > network_g > anyOf > moesr2 > in_ch`](#network_g_anyOf_i24_in_ch)
    - [5.25.3. Property `ReduxOptions > network_g > anyOf > moesr2 > out_ch`](#network_g_anyOf_i24_out_ch)
    - [5.25.4. Property `ReduxOptions > network_g > anyOf > moesr2 > scale`](#network_g_anyOf_i24_scale)
    - [5.25.5. Property `ReduxOptions > network_g > anyOf > moesr2 > dim`](#network_g_anyOf_i24_dim)
    - [5.25.6. Property `ReduxOptions > network_g > anyOf > moesr2 > n_blocks`](#network_g_anyOf_i24_n_blocks)
    - [5.25.7. Property `ReduxOptions > network_g > anyOf > moesr2 > n_block`](#network_g_anyOf_i24_n_block)
    - [5.25.8. Property `ReduxOptions > network_g > anyOf > moesr2 > expansion_factor`](#network_g_anyOf_i24_expansion_factor)
    - [5.25.9. Property `ReduxOptions > network_g > anyOf > moesr2 > expansion_msg`](#network_g_anyOf_i24_expansion_msg)
    - [5.25.10. Property `ReduxOptions > network_g > anyOf > moesr2 > upsampler`](#network_g_anyOf_i24_upsampler)
    - [5.25.11. Property `ReduxOptions > network_g > anyOf > moesr2 > upsample_dim`](#network_g_anyOf_i24_upsample_dim)
  - [5.26. Property `ReduxOptions > network_g > anyOf > mosr`](#network_g_anyOf_i25)
    - [5.26.1. Property `ReduxOptions > network_g > anyOf > mosr > type`](#network_g_anyOf_i25_type)
  - [5.27. Property `ReduxOptions > network_g > anyOf > mosr_t`](#network_g_anyOf_i26)
    - [5.27.1. Property `ReduxOptions > network_g > anyOf > mosr_t > type`](#network_g_anyOf_i26_type)
  - [5.28. Property `ReduxOptions > network_g > anyOf > rcancab`](#network_g_anyOf_i27)
    - [5.28.1. Property `ReduxOptions > network_g > anyOf > rcancab > type`](#network_g_anyOf_i27_type)
    - [5.28.2. Property `ReduxOptions > network_g > anyOf > rcancab > scale`](#network_g_anyOf_i27_scale)
    - [5.28.3. Property `ReduxOptions > network_g > anyOf > rcancab > n_resgroups`](#network_g_anyOf_i27_n_resgroups)
    - [5.28.4. Property `ReduxOptions > network_g > anyOf > rcancab > n_resblocks`](#network_g_anyOf_i27_n_resblocks)
    - [5.28.5. Property `ReduxOptions > network_g > anyOf > rcancab > n_feats`](#network_g_anyOf_i27_n_feats)
    - [5.28.6. Property `ReduxOptions > network_g > anyOf > rcancab > n_colors`](#network_g_anyOf_i27_n_colors)
    - [5.28.7. Property `ReduxOptions > network_g > anyOf > rcancab > kernel_size`](#network_g_anyOf_i27_kernel_size)
    - [5.28.8. Property `ReduxOptions > network_g > anyOf > rcancab > reduction`](#network_g_anyOf_i27_reduction)
    - [5.28.9. Property `ReduxOptions > network_g > anyOf > rcancab > res_scale`](#network_g_anyOf_i27_res_scale)
  - [5.29. Property `ReduxOptions > network_g > anyOf > rcan`](#network_g_anyOf_i28)
    - [5.29.1. Property `ReduxOptions > network_g > anyOf > rcan > type`](#network_g_anyOf_i28_type)
    - [5.29.2. Property `ReduxOptions > network_g > anyOf > rcan > scale`](#network_g_anyOf_i28_scale)
    - [5.29.3. Property `ReduxOptions > network_g > anyOf > rcan > n_resgroups`](#network_g_anyOf_i28_n_resgroups)
    - [5.29.4. Property `ReduxOptions > network_g > anyOf > rcan > n_resblocks`](#network_g_anyOf_i28_n_resblocks)
    - [5.29.5. Property `ReduxOptions > network_g > anyOf > rcan > n_feats`](#network_g_anyOf_i28_n_feats)
    - [5.29.6. Property `ReduxOptions > network_g > anyOf > rcan > n_colors`](#network_g_anyOf_i28_n_colors)
    - [5.29.7. Property `ReduxOptions > network_g > anyOf > rcan > rgb_range`](#network_g_anyOf_i28_rgb_range)
    - [5.29.8. Property `ReduxOptions > network_g > anyOf > rcan > kernel_size`](#network_g_anyOf_i28_kernel_size)
    - [5.29.9. Property `ReduxOptions > network_g > anyOf > rcan > reduction`](#network_g_anyOf_i28_reduction)
    - [5.29.10. Property `ReduxOptions > network_g > anyOf > rcan > res_scale`](#network_g_anyOf_i28_res_scale)
  - [5.30. Property `ReduxOptions > network_g > anyOf > realplksr`](#network_g_anyOf_i29)
    - [5.30.1. Property `ReduxOptions > network_g > anyOf > realplksr > type`](#network_g_anyOf_i29_type)
  - [5.31. Property `ReduxOptions > network_g > anyOf > scunet_aaf6aa`](#network_g_anyOf_i30)
    - [5.31.1. Property `ReduxOptions > network_g > anyOf > scunet_aaf6aa > type`](#network_g_anyOf_i30_type)
  - [5.32. Property `ReduxOptions > network_g > anyOf > spanplus`](#network_g_anyOf_i31)
    - [5.32.1. Property `ReduxOptions > network_g > anyOf > spanplus > type`](#network_g_anyOf_i31_type)
  - [5.33. Property `ReduxOptions > network_g > anyOf > spanplus_sts`](#network_g_anyOf_i32)
    - [5.33.1. Property `ReduxOptions > network_g > anyOf > spanplus_sts > type`](#network_g_anyOf_i32_type)
  - [5.34. Property `ReduxOptions > network_g > anyOf > spanplus_s`](#network_g_anyOf_i33)
    - [5.34.1. Property `ReduxOptions > network_g > anyOf > spanplus_s > type`](#network_g_anyOf_i33_type)
  - [5.35. Property `ReduxOptions > network_g > anyOf > spanplus_st`](#network_g_anyOf_i34)
    - [5.35.1. Property `ReduxOptions > network_g > anyOf > spanplus_st > type`](#network_g_anyOf_i34_type)
  - [5.36. Property `ReduxOptions > network_g > anyOf > compact`](#network_g_anyOf_i35)
    - [5.36.1. Property `ReduxOptions > network_g > anyOf > compact > type`](#network_g_anyOf_i35_type)
  - [5.37. Property `ReduxOptions > network_g > anyOf > ultracompact`](#network_g_anyOf_i36)
    - [5.37.1. Property `ReduxOptions > network_g > anyOf > ultracompact > type`](#network_g_anyOf_i36_type)
  - [5.38. Property `ReduxOptions > network_g > anyOf > superultracompact`](#network_g_anyOf_i37)
    - [5.38.1. Property `ReduxOptions > network_g > anyOf > superultracompact > type`](#network_g_anyOf_i37_type)
  - [5.39. Property `ReduxOptions > network_g > anyOf > tscunet`](#network_g_anyOf_i38)
    - [5.39.1. Property `ReduxOptions > network_g > anyOf > tscunet > type`](#network_g_anyOf_i38_type)
  - [5.40. Property `ReduxOptions > network_g > anyOf > atd`](#network_g_anyOf_i39)
    - [5.40.1. Property `ReduxOptions > network_g > anyOf > atd > type`](#network_g_anyOf_i39_type)
  - [5.41. Property `ReduxOptions > network_g > anyOf > atd_light`](#network_g_anyOf_i40)
    - [5.41.1. Property `ReduxOptions > network_g > anyOf > atd_light > type`](#network_g_anyOf_i40_type)
  - [5.42. Property `ReduxOptions > network_g > anyOf > dat`](#network_g_anyOf_i41)
    - [5.42.1. Property `ReduxOptions > network_g > anyOf > dat > type`](#network_g_anyOf_i41_type)
  - [5.43. Property `ReduxOptions > network_g > anyOf > dat_s`](#network_g_anyOf_i42)
    - [5.43.1. Property `ReduxOptions > network_g > anyOf > dat_s > type`](#network_g_anyOf_i42_type)
  - [5.44. Property `ReduxOptions > network_g > anyOf > dat_2`](#network_g_anyOf_i43)
    - [5.44.1. Property `ReduxOptions > network_g > anyOf > dat_2 > type`](#network_g_anyOf_i43_type)
  - [5.45. Property `ReduxOptions > network_g > anyOf > dat_light`](#network_g_anyOf_i44)
    - [5.45.1. Property `ReduxOptions > network_g > anyOf > dat_light > type`](#network_g_anyOf_i44_type)
  - [5.46. Property `ReduxOptions > network_g > anyOf > drct`](#network_g_anyOf_i45)
    - [5.46.1. Property `ReduxOptions > network_g > anyOf > drct > type`](#network_g_anyOf_i45_type)
  - [5.47. Property `ReduxOptions > network_g > anyOf > drct_l`](#network_g_anyOf_i46)
    - [5.47.1. Property `ReduxOptions > network_g > anyOf > drct_l > type`](#network_g_anyOf_i46_type)
  - [5.48. Property `ReduxOptions > network_g > anyOf > drct_xl`](#network_g_anyOf_i47)
    - [5.48.1. Property `ReduxOptions > network_g > anyOf > drct_xl > type`](#network_g_anyOf_i47_type)
  - [5.49. Property `ReduxOptions > network_g > anyOf > hat`](#network_g_anyOf_i48)
    - [5.49.1. Property `ReduxOptions > network_g > anyOf > hat > type`](#network_g_anyOf_i48_type)
  - [5.50. Property `ReduxOptions > network_g > anyOf > hat_l`](#network_g_anyOf_i49)
    - [5.50.1. Property `ReduxOptions > network_g > anyOf > hat_l > type`](#network_g_anyOf_i49_type)
  - [5.51. Property `ReduxOptions > network_g > anyOf > hat_m`](#network_g_anyOf_i50)
    - [5.51.1. Property `ReduxOptions > network_g > anyOf > hat_m > type`](#network_g_anyOf_i50_type)
  - [5.52. Property `ReduxOptions > network_g > anyOf > hat_s`](#network_g_anyOf_i51)
    - [5.52.1. Property `ReduxOptions > network_g > anyOf > hat_s > type`](#network_g_anyOf_i51_type)
  - [5.53. Property `ReduxOptions > network_g > anyOf > omnisr`](#network_g_anyOf_i52)
    - [5.53.1. Property `ReduxOptions > network_g > anyOf > omnisr > type`](#network_g_anyOf_i52_type)
  - [5.54. Property `ReduxOptions > network_g > anyOf > plksr`](#network_g_anyOf_i53)
    - [5.54.1. Property `ReduxOptions > network_g > anyOf > plksr > type`](#network_g_anyOf_i53_type)
  - [5.55. Property `ReduxOptions > network_g > anyOf > plksr_tiny`](#network_g_anyOf_i54)
    - [5.55.1. Property `ReduxOptions > network_g > anyOf > plksr_tiny > type`](#network_g_anyOf_i54_type)
  - [5.56. Property `ReduxOptions > network_g > anyOf > realcugan`](#network_g_anyOf_i55)
    - [5.56.1. Property `ReduxOptions > network_g > anyOf > realcugan > type`](#network_g_anyOf_i55_type)
  - [5.57. Property `ReduxOptions > network_g > anyOf > rgt`](#network_g_anyOf_i56)
    - [5.57.1. Property `ReduxOptions > network_g > anyOf > rgt > type`](#network_g_anyOf_i56_type)
  - [5.58. Property `ReduxOptions > network_g > anyOf > rgt_s`](#network_g_anyOf_i57)
    - [5.58.1. Property `ReduxOptions > network_g > anyOf > rgt_s > type`](#network_g_anyOf_i57_type)
  - [5.59. Property `ReduxOptions > network_g > anyOf > esrgan`](#network_g_anyOf_i58)
    - [5.59.1. Property `ReduxOptions > network_g > anyOf > esrgan > type`](#network_g_anyOf_i58_type)
  - [5.60. Property `ReduxOptions > network_g > anyOf > esrgan_lite`](#network_g_anyOf_i59)
    - [5.60.1. Property `ReduxOptions > network_g > anyOf > esrgan_lite > type`](#network_g_anyOf_i59_type)
  - [5.61. Property `ReduxOptions > network_g > anyOf > span`](#network_g_anyOf_i60)
    - [5.61.1. Property `ReduxOptions > network_g > anyOf > span > type`](#network_g_anyOf_i60_type)
  - [5.62. Property `ReduxOptions > network_g > anyOf > srformer`](#network_g_anyOf_i61)
    - [5.62.1. Property `ReduxOptions > network_g > anyOf > srformer > type`](#network_g_anyOf_i61_type)
  - [5.63. Property `ReduxOptions > network_g > anyOf > srformer_light`](#network_g_anyOf_i62)
    - [5.63.1. Property `ReduxOptions > network_g > anyOf > srformer_light > type`](#network_g_anyOf_i62_type)
  - [5.64. Property `ReduxOptions > network_g > anyOf > swin2sr`](#network_g_anyOf_i63)
    - [5.64.1. Property `ReduxOptions > network_g > anyOf > swin2sr > type`](#network_g_anyOf_i63_type)
  - [5.65. Property `ReduxOptions > network_g > anyOf > swin2sr_l`](#network_g_anyOf_i64)
    - [5.65.1. Property `ReduxOptions > network_g > anyOf > swin2sr_l > type`](#network_g_anyOf_i64_type)
  - [5.66. Property `ReduxOptions > network_g > anyOf > swin2sr_m`](#network_g_anyOf_i65)
    - [5.66.1. Property `ReduxOptions > network_g > anyOf > swin2sr_m > type`](#network_g_anyOf_i65_type)
  - [5.67. Property `ReduxOptions > network_g > anyOf > swin2sr_s`](#network_g_anyOf_i66)
    - [5.67.1. Property `ReduxOptions > network_g > anyOf > swin2sr_s > type`](#network_g_anyOf_i66_type)
  - [5.68. Property `ReduxOptions > network_g > anyOf > swinir`](#network_g_anyOf_i67)
    - [5.68.1. Property `ReduxOptions > network_g > anyOf > swinir > type`](#network_g_anyOf_i67_type)
  - [5.69. Property `ReduxOptions > network_g > anyOf > swinir_l`](#network_g_anyOf_i68)
    - [5.69.1. Property `ReduxOptions > network_g > anyOf > swinir_l > type`](#network_g_anyOf_i68_type)
  - [5.70. Property `ReduxOptions > network_g > anyOf > swinir_m`](#network_g_anyOf_i69)
    - [5.70.1. Property `ReduxOptions > network_g > anyOf > swinir_m > type`](#network_g_anyOf_i69_type)
  - [5.71. Property `ReduxOptions > network_g > anyOf > swinir_s`](#network_g_anyOf_i70)
    - [5.71.1. Property `ReduxOptions > network_g > anyOf > swinir_s > type`](#network_g_anyOf_i70_type)
- [6. Property `ReduxOptions > network_d`](#network_d)
  - [6.1. Property `ReduxOptions > network_d > anyOf > item 0`](#network_d_anyOf_i0)
  - [6.2. Property `ReduxOptions > network_d > anyOf > item 1`](#network_d_anyOf_i1)
- [7. Property `ReduxOptions > manual_seed`](#manual_seed)
  - [7.1. Property `ReduxOptions > manual_seed > anyOf > item 0`](#manual_seed_anyOf_i0)
  - [7.2. Property `ReduxOptions > manual_seed > anyOf > item 1`](#manual_seed_anyOf_i1)
- [8. Property `ReduxOptions > deterministic`](#deterministic)
  - [8.1. Property `ReduxOptions > deterministic > anyOf > item 0`](#deterministic_anyOf_i0)
  - [8.2. Property `ReduxOptions > deterministic > anyOf > item 1`](#deterministic_anyOf_i1)
- [9. Property `ReduxOptions > dist`](#dist)
  - [9.1. Property `ReduxOptions > dist > anyOf > item 0`](#dist_anyOf_i0)
  - [9.2. Property `ReduxOptions > dist > anyOf > item 1`](#dist_anyOf_i1)
- [10. Property `ReduxOptions > launcher`](#launcher)
  - [10.1. Property `ReduxOptions > launcher > anyOf > item 0`](#launcher_anyOf_i0)
  - [10.2. Property `ReduxOptions > launcher > anyOf > item 1`](#launcher_anyOf_i1)
- [11. Property `ReduxOptions > rank`](#rank)
  - [11.1. Property `ReduxOptions > rank > anyOf > item 0`](#rank_anyOf_i0)
  - [11.2. Property `ReduxOptions > rank > anyOf > item 1`](#rank_anyOf_i1)
- [12. Property `ReduxOptions > world_size`](#world_size)
  - [12.1. Property `ReduxOptions > world_size > anyOf > item 0`](#world_size_anyOf_i0)
  - [12.2. Property `ReduxOptions > world_size > anyOf > item 1`](#world_size_anyOf_i1)
- [13. Property `ReduxOptions > auto_resume`](#auto_resume)
  - [13.1. Property `ReduxOptions > auto_resume > anyOf > item 0`](#auto_resume_anyOf_i0)
  - [13.2. Property `ReduxOptions > auto_resume > anyOf > item 1`](#auto_resume_anyOf_i1)
- [14. Property `ReduxOptions > resume`](#resume)
- [15. Property `ReduxOptions > is_train`](#is_train)
  - [15.1. Property `ReduxOptions > is_train > anyOf > item 0`](#is_train_anyOf_i0)
  - [15.2. Property `ReduxOptions > is_train > anyOf > item 1`](#is_train_anyOf_i1)
- [16. Property `ReduxOptions > root_path`](#root_path)
  - [16.1. Property `ReduxOptions > root_path > anyOf > item 0`](#root_path_anyOf_i0)
  - [16.2. Property `ReduxOptions > root_path > anyOf > item 1`](#root_path_anyOf_i1)
- [17. Property `ReduxOptions > use_amp`](#use_amp)
- [18. Property `ReduxOptions > amp_bf16`](#amp_bf16)
- [19. Property `ReduxOptions > use_channels_last`](#use_channels_last)
- [20. Property `ReduxOptions > fast_matmul`](#fast_matmul)
- [21. Property `ReduxOptions > use_compile`](#use_compile)
- [22. Property `ReduxOptions > detect_anomaly`](#detect_anomaly)
- [23. Property `ReduxOptions > high_order_degradation`](#high_order_degradation)
- [24. Property `ReduxOptions > high_order_degradations_debug`](#high_order_degradations_debug)
- [25. Property `ReduxOptions > high_order_degradations_debug_limit`](#high_order_degradations_debug_limit)
- [26. Property `ReduxOptions > dataroot_lq_prob`](#dataroot_lq_prob)
- [27. Property `ReduxOptions > lq_usm`](#lq_usm)
- [28. Property `ReduxOptions > lq_usm_radius_range`](#lq_usm_radius_range)
  - [28.1. ReduxOptions > lq_usm_radius_range > lq_usm_radius_range item 0](#autogenerated_heading_2)
  - [28.2. ReduxOptions > lq_usm_radius_range > lq_usm_radius_range item 1](#autogenerated_heading_3)
- [29. Property `ReduxOptions > blur_prob`](#blur_prob)
- [30. Property `ReduxOptions > resize_prob`](#resize_prob)
  - [30.1. ReduxOptions > resize_prob > resize_prob items](#resize_prob_items)
- [31. Property `ReduxOptions > resize_mode_list`](#resize_mode_list)
  - [31.1. ReduxOptions > resize_mode_list > resize_mode_list items](#resize_mode_list_items)
- [32. Property `ReduxOptions > resize_mode_prob`](#resize_mode_prob)
  - [32.1. ReduxOptions > resize_mode_prob > resize_mode_prob items](#resize_mode_prob_items)
- [33. Property `ReduxOptions > resize_range`](#resize_range)
  - [33.1. ReduxOptions > resize_range > resize_range item 0](#autogenerated_heading_4)
  - [33.2. ReduxOptions > resize_range > resize_range item 1](#autogenerated_heading_5)
- [34. Property `ReduxOptions > gaussian_noise_prob`](#gaussian_noise_prob)
- [35. Property `ReduxOptions > noise_range`](#noise_range)
  - [35.1. ReduxOptions > noise_range > noise_range item 0](#autogenerated_heading_6)
  - [35.2. ReduxOptions > noise_range > noise_range item 1](#autogenerated_heading_7)
- [36. Property `ReduxOptions > poisson_scale_range`](#poisson_scale_range)
  - [36.1. ReduxOptions > poisson_scale_range > poisson_scale_range item 0](#autogenerated_heading_8)
  - [36.2. ReduxOptions > poisson_scale_range > poisson_scale_range item 1](#autogenerated_heading_9)
- [37. Property `ReduxOptions > gray_noise_prob`](#gray_noise_prob)
- [38. Property `ReduxOptions > jpeg_prob`](#jpeg_prob)
- [39. Property `ReduxOptions > jpeg_range`](#jpeg_range)
  - [39.1. ReduxOptions > jpeg_range > jpeg_range item 0](#autogenerated_heading_10)
  - [39.2. ReduxOptions > jpeg_range > jpeg_range item 1](#autogenerated_heading_11)
- [40. Property `ReduxOptions > blur_prob2`](#blur_prob2)
- [41. Property `ReduxOptions > resize_prob2`](#resize_prob2)
  - [41.1. ReduxOptions > resize_prob2 > resize_prob2 items](#resize_prob2_items)
- [42. Property `ReduxOptions > resize_mode_list2`](#resize_mode_list2)
  - [42.1. ReduxOptions > resize_mode_list2 > resize_mode_list2 items](#resize_mode_list2_items)
- [43. Property `ReduxOptions > resize_mode_prob2`](#resize_mode_prob2)
  - [43.1. ReduxOptions > resize_mode_prob2 > resize_mode_prob2 items](#resize_mode_prob2_items)
- [44. Property `ReduxOptions > resize_range2`](#resize_range2)
  - [44.1. ReduxOptions > resize_range2 > resize_range2 item 0](#autogenerated_heading_12)
  - [44.2. ReduxOptions > resize_range2 > resize_range2 item 1](#autogenerated_heading_13)
- [45. Property `ReduxOptions > gaussian_noise_prob2`](#gaussian_noise_prob2)
- [46. Property `ReduxOptions > noise_range2`](#noise_range2)
  - [46.1. ReduxOptions > noise_range2 > noise_range2 item 0](#autogenerated_heading_14)
  - [46.2. ReduxOptions > noise_range2 > noise_range2 item 1](#autogenerated_heading_15)
- [47. Property `ReduxOptions > poisson_scale_range2`](#poisson_scale_range2)
  - [47.1. ReduxOptions > poisson_scale_range2 > poisson_scale_range2 item 0](#autogenerated_heading_16)
  - [47.2. ReduxOptions > poisson_scale_range2 > poisson_scale_range2 item 1](#autogenerated_heading_17)
- [48. Property `ReduxOptions > gray_noise_prob2`](#gray_noise_prob2)
- [49. Property `ReduxOptions > jpeg_prob2`](#jpeg_prob2)
- [50. Property `ReduxOptions > jpeg_range2`](#jpeg_range2)
  - [50.1. ReduxOptions > jpeg_range2 > jpeg_range2 items](#jpeg_range2_items)
- [51. Property `ReduxOptions > resize_mode_list3`](#resize_mode_list3)
  - [51.1. ReduxOptions > resize_mode_list3 > resize_mode_list3 items](#resize_mode_list3_items)
- [52. Property `ReduxOptions > resize_mode_prob3`](#resize_mode_prob3)
  - [52.1. ReduxOptions > resize_mode_prob3 > resize_mode_prob3 items](#resize_mode_prob3_items)
- [53. Property `ReduxOptions > queue_size`](#queue_size)
- [54. Property `ReduxOptions > datasets`](#datasets)
  - [54.1. Property `ReduxOptions > datasets > DatasetOptions`](#datasets_additionalProperties)
    - [54.1.1. Property `ReduxOptions > datasets > DatasetOptions > name`](#datasets_additionalProperties_name)
    - [54.1.2. Property `ReduxOptions > datasets > DatasetOptions > type`](#datasets_additionalProperties_type)
    - [54.1.3. Property `ReduxOptions > datasets > DatasetOptions > io_backend`](#datasets_additionalProperties_io_backend)
    - [54.1.4. Property `ReduxOptions > datasets > DatasetOptions > num_worker_per_gpu`](#datasets_additionalProperties_num_worker_per_gpu)
      - [54.1.4.1. Property `ReduxOptions > datasets > DatasetOptions > num_worker_per_gpu > anyOf > item 0`](#datasets_additionalProperties_num_worker_per_gpu_anyOf_i0)
      - [54.1.4.2. Property `ReduxOptions > datasets > DatasetOptions > num_worker_per_gpu > anyOf > item 1`](#datasets_additionalProperties_num_worker_per_gpu_anyOf_i1)
    - [54.1.5. Property `ReduxOptions > datasets > DatasetOptions > batch_size_per_gpu`](#datasets_additionalProperties_batch_size_per_gpu)
      - [54.1.5.1. Property `ReduxOptions > datasets > DatasetOptions > batch_size_per_gpu > anyOf > item 0`](#datasets_additionalProperties_batch_size_per_gpu_anyOf_i0)
      - [54.1.5.2. Property `ReduxOptions > datasets > DatasetOptions > batch_size_per_gpu > anyOf > item 1`](#datasets_additionalProperties_batch_size_per_gpu_anyOf_i1)
    - [54.1.6. Property `ReduxOptions > datasets > DatasetOptions > accum_iter`](#datasets_additionalProperties_accum_iter)
    - [54.1.7. Property `ReduxOptions > datasets > DatasetOptions > use_hflip`](#datasets_additionalProperties_use_hflip)
    - [54.1.8. Property `ReduxOptions > datasets > DatasetOptions > use_rot`](#datasets_additionalProperties_use_rot)
    - [54.1.9. Property `ReduxOptions > datasets > DatasetOptions > mean`](#datasets_additionalProperties_mean)
      - [54.1.9.1. Property `ReduxOptions > datasets > DatasetOptions > mean > anyOf > item 0`](#datasets_additionalProperties_mean_anyOf_i0)
        - [54.1.9.1.1. ReduxOptions > datasets > DatasetOptions > mean > anyOf > item 0 > item 0 items](#datasets_additionalProperties_mean_anyOf_i0_items)
      - [54.1.9.2. Property `ReduxOptions > datasets > DatasetOptions > mean > anyOf > item 1`](#datasets_additionalProperties_mean_anyOf_i1)
    - [54.1.10. Property `ReduxOptions > datasets > DatasetOptions > std`](#datasets_additionalProperties_std)
      - [54.1.10.1. Property `ReduxOptions > datasets > DatasetOptions > std > anyOf > item 0`](#datasets_additionalProperties_std_anyOf_i0)
        - [54.1.10.1.1. ReduxOptions > datasets > DatasetOptions > std > anyOf > item 0 > item 0 items](#datasets_additionalProperties_std_anyOf_i0_items)
      - [54.1.10.2. Property `ReduxOptions > datasets > DatasetOptions > std > anyOf > item 1`](#datasets_additionalProperties_std_anyOf_i1)
    - [54.1.11. Property `ReduxOptions > datasets > DatasetOptions > gt_size`](#datasets_additionalProperties_gt_size)
      - [54.1.11.1. Property `ReduxOptions > datasets > DatasetOptions > gt_size > anyOf > item 0`](#datasets_additionalProperties_gt_size_anyOf_i0)
      - [54.1.11.2. Property `ReduxOptions > datasets > DatasetOptions > gt_size > anyOf > item 1`](#datasets_additionalProperties_gt_size_anyOf_i1)
    - [54.1.12. Property `ReduxOptions > datasets > DatasetOptions > lq_size`](#datasets_additionalProperties_lq_size)
      - [54.1.12.1. Property `ReduxOptions > datasets > DatasetOptions > lq_size > anyOf > item 0`](#datasets_additionalProperties_lq_size_anyOf_i0)
      - [54.1.12.2. Property `ReduxOptions > datasets > DatasetOptions > lq_size > anyOf > item 1`](#datasets_additionalProperties_lq_size_anyOf_i1)
    - [54.1.13. Property `ReduxOptions > datasets > DatasetOptions > color`](#datasets_additionalProperties_color)
      - [54.1.13.1. Property `ReduxOptions > datasets > DatasetOptions > color > anyOf > item 0`](#datasets_additionalProperties_color_anyOf_i0)
      - [54.1.13.2. Property `ReduxOptions > datasets > DatasetOptions > color > anyOf > item 1`](#datasets_additionalProperties_color_anyOf_i1)
    - [54.1.14. Property `ReduxOptions > datasets > DatasetOptions > phase`](#datasets_additionalProperties_phase)
      - [54.1.14.1. Property `ReduxOptions > datasets > DatasetOptions > phase > anyOf > item 0`](#datasets_additionalProperties_phase_anyOf_i0)
      - [54.1.14.2. Property `ReduxOptions > datasets > DatasetOptions > phase > anyOf > item 1`](#datasets_additionalProperties_phase_anyOf_i1)
    - [54.1.15. Property `ReduxOptions > datasets > DatasetOptions > scale`](#datasets_additionalProperties_scale)
      - [54.1.15.1. Property `ReduxOptions > datasets > DatasetOptions > scale > anyOf > item 0`](#datasets_additionalProperties_scale_anyOf_i0)
      - [54.1.15.2. Property `ReduxOptions > datasets > DatasetOptions > scale > anyOf > item 1`](#datasets_additionalProperties_scale_anyOf_i1)
    - [54.1.16. Property `ReduxOptions > datasets > DatasetOptions > dataset_enlarge_ratio`](#datasets_additionalProperties_dataset_enlarge_ratio)
      - [54.1.16.1. Property `ReduxOptions > datasets > DatasetOptions > dataset_enlarge_ratio > anyOf > item 0`](#datasets_additionalProperties_dataset_enlarge_ratio_anyOf_i0)
      - [54.1.16.2. Property `ReduxOptions > datasets > DatasetOptions > dataset_enlarge_ratio > anyOf > item 1`](#datasets_additionalProperties_dataset_enlarge_ratio_anyOf_i1)
    - [54.1.17. Property `ReduxOptions > datasets > DatasetOptions > prefetch_mode`](#datasets_additionalProperties_prefetch_mode)
      - [54.1.17.1. Property `ReduxOptions > datasets > DatasetOptions > prefetch_mode > anyOf > item 0`](#datasets_additionalProperties_prefetch_mode_anyOf_i0)
      - [54.1.17.2. Property `ReduxOptions > datasets > DatasetOptions > prefetch_mode > anyOf > item 1`](#datasets_additionalProperties_prefetch_mode_anyOf_i1)
    - [54.1.18. Property `ReduxOptions > datasets > DatasetOptions > pin_memory`](#datasets_additionalProperties_pin_memory)
    - [54.1.19. Property `ReduxOptions > datasets > DatasetOptions > persistent_workers`](#datasets_additionalProperties_persistent_workers)
    - [54.1.20. Property `ReduxOptions > datasets > DatasetOptions > num_prefetch_queue`](#datasets_additionalProperties_num_prefetch_queue)
    - [54.1.21. Property `ReduxOptions > datasets > DatasetOptions > prefetch_factor`](#datasets_additionalProperties_prefetch_factor)
    - [54.1.22. Property `ReduxOptions > datasets > DatasetOptions > clip_size`](#datasets_additionalProperties_clip_size)
      - [54.1.22.1. Property `ReduxOptions > datasets > DatasetOptions > clip_size > anyOf > item 0`](#datasets_additionalProperties_clip_size_anyOf_i0)
      - [54.1.22.2. Property `ReduxOptions > datasets > DatasetOptions > clip_size > anyOf > item 1`](#datasets_additionalProperties_clip_size_anyOf_i1)
    - [54.1.23. Property `ReduxOptions > datasets > DatasetOptions > dataroot_gt`](#datasets_additionalProperties_dataroot_gt)
      - [54.1.23.1. Property `ReduxOptions > datasets > DatasetOptions > dataroot_gt > anyOf > item 0`](#datasets_additionalProperties_dataroot_gt_anyOf_i0)
      - [54.1.23.2. Property `ReduxOptions > datasets > DatasetOptions > dataroot_gt > anyOf > item 1`](#datasets_additionalProperties_dataroot_gt_anyOf_i1)
        - [54.1.23.2.1. ReduxOptions > datasets > DatasetOptions > dataroot_gt > anyOf > item 1 > item 1 items](#datasets_additionalProperties_dataroot_gt_anyOf_i1_items)
      - [54.1.23.3. Property `ReduxOptions > datasets > DatasetOptions > dataroot_gt > anyOf > item 2`](#datasets_additionalProperties_dataroot_gt_anyOf_i2)
    - [54.1.24. Property `ReduxOptions > datasets > DatasetOptions > dataroot_lq`](#datasets_additionalProperties_dataroot_lq)
      - [54.1.24.1. Property `ReduxOptions > datasets > DatasetOptions > dataroot_lq > anyOf > item 0`](#datasets_additionalProperties_dataroot_lq_anyOf_i0)
      - [54.1.24.2. Property `ReduxOptions > datasets > DatasetOptions > dataroot_lq > anyOf > item 1`](#datasets_additionalProperties_dataroot_lq_anyOf_i1)
        - [54.1.24.2.1. ReduxOptions > datasets > DatasetOptions > dataroot_lq > anyOf > item 1 > item 1 items](#datasets_additionalProperties_dataroot_lq_anyOf_i1_items)
      - [54.1.24.3. Property `ReduxOptions > datasets > DatasetOptions > dataroot_lq > anyOf > item 2`](#datasets_additionalProperties_dataroot_lq_anyOf_i2)
    - [54.1.25. Property `ReduxOptions > datasets > DatasetOptions > meta_info`](#datasets_additionalProperties_meta_info)
      - [54.1.25.1. Property `ReduxOptions > datasets > DatasetOptions > meta_info > anyOf > item 0`](#datasets_additionalProperties_meta_info_anyOf_i0)
      - [54.1.25.2. Property `ReduxOptions > datasets > DatasetOptions > meta_info > anyOf > item 1`](#datasets_additionalProperties_meta_info_anyOf_i1)
    - [54.1.26. Property `ReduxOptions > datasets > DatasetOptions > filename_tmpl`](#datasets_additionalProperties_filename_tmpl)
    - [54.1.27. Property `ReduxOptions > datasets > DatasetOptions > blur_kernel_size`](#datasets_additionalProperties_blur_kernel_size)
    - [54.1.28. Property `ReduxOptions > datasets > DatasetOptions > kernel_list`](#datasets_additionalProperties_kernel_list)
      - [54.1.28.1. ReduxOptions > datasets > DatasetOptions > kernel_list > kernel_list items](#datasets_additionalProperties_kernel_list_items)
    - [54.1.29. Property `ReduxOptions > datasets > DatasetOptions > kernel_prob`](#datasets_additionalProperties_kernel_prob)
      - [54.1.29.1. ReduxOptions > datasets > DatasetOptions > kernel_prob > kernel_prob items](#datasets_additionalProperties_kernel_prob_items)
    - [54.1.30. Property `ReduxOptions > datasets > DatasetOptions > kernel_range`](#datasets_additionalProperties_kernel_range)
      - [54.1.30.1. ReduxOptions > datasets > DatasetOptions > kernel_range > kernel_range item 0](#autogenerated_heading_18)
      - [54.1.30.2. ReduxOptions > datasets > DatasetOptions > kernel_range > kernel_range item 1](#autogenerated_heading_19)
    - [54.1.31. Property `ReduxOptions > datasets > DatasetOptions > sinc_prob`](#datasets_additionalProperties_sinc_prob)
    - [54.1.32. Property `ReduxOptions > datasets > DatasetOptions > blur_sigma`](#datasets_additionalProperties_blur_sigma)
      - [54.1.32.1. ReduxOptions > datasets > DatasetOptions > blur_sigma > blur_sigma item 0](#autogenerated_heading_20)
      - [54.1.32.2. ReduxOptions > datasets > DatasetOptions > blur_sigma > blur_sigma item 1](#autogenerated_heading_21)
    - [54.1.33. Property `ReduxOptions > datasets > DatasetOptions > betag_range`](#datasets_additionalProperties_betag_range)
      - [54.1.33.1. ReduxOptions > datasets > DatasetOptions > betag_range > betag_range item 0](#autogenerated_heading_22)
      - [54.1.33.2. ReduxOptions > datasets > DatasetOptions > betag_range > betag_range item 1](#autogenerated_heading_23)
    - [54.1.34. Property `ReduxOptions > datasets > DatasetOptions > betap_range`](#datasets_additionalProperties_betap_range)
      - [54.1.34.1. ReduxOptions > datasets > DatasetOptions > betap_range > betap_range item 0](#autogenerated_heading_24)
      - [54.1.34.2. ReduxOptions > datasets > DatasetOptions > betap_range > betap_range item 1](#autogenerated_heading_25)
    - [54.1.35. Property `ReduxOptions > datasets > DatasetOptions > blur_kernel_size2`](#datasets_additionalProperties_blur_kernel_size2)
    - [54.1.36. Property `ReduxOptions > datasets > DatasetOptions > kernel_list2`](#datasets_additionalProperties_kernel_list2)
      - [54.1.36.1. ReduxOptions > datasets > DatasetOptions > kernel_list2 > kernel_list2 items](#datasets_additionalProperties_kernel_list2_items)
    - [54.1.37. Property `ReduxOptions > datasets > DatasetOptions > kernel_prob2`](#datasets_additionalProperties_kernel_prob2)
      - [54.1.37.1. ReduxOptions > datasets > DatasetOptions > kernel_prob2 > kernel_prob2 items](#datasets_additionalProperties_kernel_prob2_items)
    - [54.1.38. Property `ReduxOptions > datasets > DatasetOptions > kernel_range2`](#datasets_additionalProperties_kernel_range2)
      - [54.1.38.1. ReduxOptions > datasets > DatasetOptions > kernel_range2 > kernel_range2 item 0](#autogenerated_heading_26)
      - [54.1.38.2. ReduxOptions > datasets > DatasetOptions > kernel_range2 > kernel_range2 item 1](#autogenerated_heading_27)
    - [54.1.39. Property `ReduxOptions > datasets > DatasetOptions > sinc_prob2`](#datasets_additionalProperties_sinc_prob2)
    - [54.1.40. Property `ReduxOptions > datasets > DatasetOptions > blur_sigma2`](#datasets_additionalProperties_blur_sigma2)
      - [54.1.40.1. ReduxOptions > datasets > DatasetOptions > blur_sigma2 > blur_sigma2 item 0](#autogenerated_heading_28)
      - [54.1.40.2. ReduxOptions > datasets > DatasetOptions > blur_sigma2 > blur_sigma2 item 1](#autogenerated_heading_29)
    - [54.1.41. Property `ReduxOptions > datasets > DatasetOptions > betag_range2`](#datasets_additionalProperties_betag_range2)
      - [54.1.41.1. ReduxOptions > datasets > DatasetOptions > betag_range2 > betag_range2 item 0](#autogenerated_heading_30)
      - [54.1.41.2. ReduxOptions > datasets > DatasetOptions > betag_range2 > betag_range2 item 1](#autogenerated_heading_31)
    - [54.1.42. Property `ReduxOptions > datasets > DatasetOptions > betap_range2`](#datasets_additionalProperties_betap_range2)
      - [54.1.42.1. ReduxOptions > datasets > DatasetOptions > betap_range2 > betap_range2 item 0](#autogenerated_heading_32)
      - [54.1.42.2. ReduxOptions > datasets > DatasetOptions > betap_range2 > betap_range2 item 1](#autogenerated_heading_33)
    - [54.1.43. Property `ReduxOptions > datasets > DatasetOptions > final_sinc_prob`](#datasets_additionalProperties_final_sinc_prob)
    - [54.1.44. Property `ReduxOptions > datasets > DatasetOptions > final_kernel_range`](#datasets_additionalProperties_final_kernel_range)
      - [54.1.44.1. ReduxOptions > datasets > DatasetOptions > final_kernel_range > final_kernel_range item 0](#autogenerated_heading_34)
      - [54.1.44.2. ReduxOptions > datasets > DatasetOptions > final_kernel_range > final_kernel_range item 1](#autogenerated_heading_35)
- [55. Property `ReduxOptions > train`](#train)
  - [55.1. Property `ReduxOptions > train > total_iter`](#train_total_iter)
  - [55.2. Property `ReduxOptions > train > optim_g`](#train_optim_g)
  - [55.3. Property `ReduxOptions > train > ema_decay`](#train_ema_decay)
  - [55.4. Property `ReduxOptions > train > grad_clip`](#train_grad_clip)
  - [55.5. Property `ReduxOptions > train > warmup_iter`](#train_warmup_iter)
  - [55.6. Property `ReduxOptions > train > scheduler`](#train_scheduler)
    - [55.6.1. Property `ReduxOptions > train > scheduler > anyOf > item 0`](#train_scheduler_anyOf_i0)
    - [55.6.2. Property `ReduxOptions > train > scheduler > anyOf > SchedulerOptions`](#train_scheduler_anyOf_i1)
      - [55.6.2.1. Property `ReduxOptions > train > scheduler > anyOf > SchedulerOptions > type`](#train_scheduler_anyOf_i1_type)
      - [55.6.2.2. Property `ReduxOptions > train > scheduler > anyOf > SchedulerOptions > milestones`](#train_scheduler_anyOf_i1_milestones)
        - [55.6.2.2.1. ReduxOptions > train > scheduler > anyOf > SchedulerOptions > milestones > milestones items](#train_scheduler_anyOf_i1_milestones_items)
      - [55.6.2.3. Property `ReduxOptions > train > scheduler > anyOf > SchedulerOptions > gamma`](#train_scheduler_anyOf_i1_gamma)
  - [55.7. Property `ReduxOptions > train > optim_d`](#train_optim_d)
    - [55.7.1. Property `ReduxOptions > train > optim_d > anyOf > item 0`](#train_optim_d_anyOf_i0)
    - [55.7.2. Property `ReduxOptions > train > optim_d > anyOf > item 1`](#train_optim_d_anyOf_i1)
  - [55.8. Property `ReduxOptions > train > losses`](#train_losses)
    - [55.8.1. ReduxOptions > train > losses > losses items](#train_losses_items)
      - [55.8.1.1. Property `ReduxOptions > train > losses > losses items > anyOf > ganloss`](#train_losses_items_anyOf_i0)
        - [55.8.1.1.1. Property `ReduxOptions > train > losses > losses items > anyOf > ganloss > type`](#train_losses_items_anyOf_i0_type)
        - [55.8.1.1.2. Property `ReduxOptions > train > losses > losses items > anyOf > ganloss > loss_weight`](#train_losses_items_anyOf_i0_loss_weight)
        - [55.8.1.1.3. Property `ReduxOptions > train > losses > losses items > anyOf > ganloss > gan_type`](#train_losses_items_anyOf_i0_gan_type)
        - [55.8.1.1.4. Property `ReduxOptions > train > losses > losses items > anyOf > ganloss > real_label_val`](#train_losses_items_anyOf_i0_real_label_val)
        - [55.8.1.1.5. Property `ReduxOptions > train > losses > losses items > anyOf > ganloss > fake_label_val`](#train_losses_items_anyOf_i0_fake_label_val)
      - [55.8.1.2. Property `ReduxOptions > train > losses > losses items > anyOf > multiscaleganloss`](#train_losses_items_anyOf_i1)
        - [55.8.1.2.1. Property `ReduxOptions > train > losses > losses items > anyOf > multiscaleganloss > type`](#train_losses_items_anyOf_i1_type)
        - [55.8.1.2.2. Property `ReduxOptions > train > losses > losses items > anyOf > multiscaleganloss > loss_weight`](#train_losses_items_anyOf_i1_loss_weight)
        - [55.8.1.2.3. Property `ReduxOptions > train > losses > losses items > anyOf > multiscaleganloss > gan_type`](#train_losses_items_anyOf_i1_gan_type)
        - [55.8.1.2.4. Property `ReduxOptions > train > losses > losses items > anyOf > multiscaleganloss > real_label_val`](#train_losses_items_anyOf_i1_real_label_val)
        - [55.8.1.2.5. Property `ReduxOptions > train > losses > losses items > anyOf > multiscaleganloss > fake_label_val`](#train_losses_items_anyOf_i1_fake_label_val)
      - [55.8.1.3. Property `ReduxOptions > train > losses > losses items > anyOf > adistsloss`](#train_losses_items_anyOf_i2)
        - [55.8.1.3.1. Property `ReduxOptions > train > losses > losses items > anyOf > adistsloss > type`](#train_losses_items_anyOf_i2_type)
        - [55.8.1.3.2. Property `ReduxOptions > train > losses > losses items > anyOf > adistsloss > loss_weight`](#train_losses_items_anyOf_i2_loss_weight)
        - [55.8.1.3.3. Property `ReduxOptions > train > losses > losses items > anyOf > adistsloss > window_size`](#train_losses_items_anyOf_i2_window_size)
        - [55.8.1.3.4. Property `ReduxOptions > train > losses > losses items > anyOf > adistsloss > resize_input`](#train_losses_items_anyOf_i2_resize_input)
      - [55.8.1.4. Property `ReduxOptions > train > losses > losses items > anyOf > l1loss`](#train_losses_items_anyOf_i3)
        - [55.8.1.4.1. Property `ReduxOptions > train > losses > losses items > anyOf > l1loss > type`](#train_losses_items_anyOf_i3_type)
        - [55.8.1.4.2. Property `ReduxOptions > train > losses > losses items > anyOf > l1loss > loss_weight`](#train_losses_items_anyOf_i3_loss_weight)
        - [55.8.1.4.3. Property `ReduxOptions > train > losses > losses items > anyOf > l1loss > reduction`](#train_losses_items_anyOf_i3_reduction)
      - [55.8.1.5. Property `ReduxOptions > train > losses > losses items > anyOf > mseloss`](#train_losses_items_anyOf_i4)
        - [55.8.1.5.1. Property `ReduxOptions > train > losses > losses items > anyOf > mseloss > type`](#train_losses_items_anyOf_i4_type)
        - [55.8.1.5.2. Property `ReduxOptions > train > losses > losses items > anyOf > mseloss > loss_weight`](#train_losses_items_anyOf_i4_loss_weight)
        - [55.8.1.5.3. Property `ReduxOptions > train > losses > losses items > anyOf > mseloss > reduction`](#train_losses_items_anyOf_i4_reduction)
      - [55.8.1.6. Property `ReduxOptions > train > losses > losses items > anyOf > charbonnierloss`](#train_losses_items_anyOf_i5)
        - [55.8.1.6.1. Property `ReduxOptions > train > losses > losses items > anyOf > charbonnierloss > type`](#train_losses_items_anyOf_i5_type)
        - [55.8.1.6.2. Property `ReduxOptions > train > losses > losses items > anyOf > charbonnierloss > loss_weight`](#train_losses_items_anyOf_i5_loss_weight)
        - [55.8.1.6.3. Property `ReduxOptions > train > losses > losses items > anyOf > charbonnierloss > reduction`](#train_losses_items_anyOf_i5_reduction)
        - [55.8.1.6.4. Property `ReduxOptions > train > losses > losses items > anyOf > charbonnierloss > eps`](#train_losses_items_anyOf_i5_eps)
      - [55.8.1.7. Property `ReduxOptions > train > losses > losses items > anyOf > colorloss`](#train_losses_items_anyOf_i6)
        - [55.8.1.7.1. Property `ReduxOptions > train > losses > losses items > anyOf > colorloss > type`](#train_losses_items_anyOf_i6_type)
        - [55.8.1.7.2. Property `ReduxOptions > train > losses > losses items > anyOf > colorloss > loss_weight`](#train_losses_items_anyOf_i6_loss_weight)
        - [55.8.1.7.3. Property `ReduxOptions > train > losses > losses items > anyOf > colorloss > criterion`](#train_losses_items_anyOf_i6_criterion)
        - [55.8.1.7.4. Property `ReduxOptions > train > losses > losses items > anyOf > colorloss > scale`](#train_losses_items_anyOf_i6_scale)
      - [55.8.1.8. Property `ReduxOptions > train > losses > losses items > anyOf > averageloss`](#train_losses_items_anyOf_i7)
        - [55.8.1.8.1. Property `ReduxOptions > train > losses > losses items > anyOf > averageloss > type`](#train_losses_items_anyOf_i7_type)
        - [55.8.1.8.2. Property `ReduxOptions > train > losses > losses items > anyOf > averageloss > loss_weight`](#train_losses_items_anyOf_i7_loss_weight)
        - [55.8.1.8.3. Property `ReduxOptions > train > losses > losses items > anyOf > averageloss > criterion`](#train_losses_items_anyOf_i7_criterion)
        - [55.8.1.8.4. Property `ReduxOptions > train > losses > losses items > anyOf > averageloss > scale`](#train_losses_items_anyOf_i7_scale)
      - [55.8.1.9. Property `ReduxOptions > train > losses > losses items > anyOf > bicubicloss`](#train_losses_items_anyOf_i8)
        - [55.8.1.9.1. Property `ReduxOptions > train > losses > losses items > anyOf > bicubicloss > type`](#train_losses_items_anyOf_i8_type)
        - [55.8.1.9.2. Property `ReduxOptions > train > losses > losses items > anyOf > bicubicloss > loss_weight`](#train_losses_items_anyOf_i8_loss_weight)
        - [55.8.1.9.3. Property `ReduxOptions > train > losses > losses items > anyOf > bicubicloss > criterion`](#train_losses_items_anyOf_i8_criterion)
        - [55.8.1.9.4. Property `ReduxOptions > train > losses > losses items > anyOf > bicubicloss > scale`](#train_losses_items_anyOf_i8_scale)
      - [55.8.1.10. Property `ReduxOptions > train > losses > losses items > anyOf > lumaloss`](#train_losses_items_anyOf_i9)
        - [55.8.1.10.1. Property `ReduxOptions > train > losses > losses items > anyOf > lumaloss > type`](#train_losses_items_anyOf_i9_type)
        - [55.8.1.10.2. Property `ReduxOptions > train > losses > losses items > anyOf > lumaloss > loss_weight`](#train_losses_items_anyOf_i9_loss_weight)
        - [55.8.1.10.3. Property `ReduxOptions > train > losses > losses items > anyOf > lumaloss > criterion`](#train_losses_items_anyOf_i9_criterion)
      - [55.8.1.11. Property `ReduxOptions > train > losses > losses items > anyOf > hsluvloss`](#train_losses_items_anyOf_i10)
        - [55.8.1.11.1. Property `ReduxOptions > train > losses > losses items > anyOf > hsluvloss > type`](#train_losses_items_anyOf_i10_type)
        - [55.8.1.11.2. Property `ReduxOptions > train > losses > losses items > anyOf > hsluvloss > loss_weight`](#train_losses_items_anyOf_i10_loss_weight)
        - [55.8.1.11.3. Property `ReduxOptions > train > losses > losses items > anyOf > hsluvloss > criterion`](#train_losses_items_anyOf_i10_criterion)
      - [55.8.1.12. Property `ReduxOptions > train > losses > losses items > anyOf > contextualloss`](#train_losses_items_anyOf_i11)
        - [55.8.1.12.1. Property `ReduxOptions > train > losses > losses items > anyOf > contextualloss > type`](#train_losses_items_anyOf_i11_type)
        - [55.8.1.12.2. Property `ReduxOptions > train > losses > losses items > anyOf > contextualloss > loss_weight`](#train_losses_items_anyOf_i11_loss_weight)
        - [55.8.1.12.3. Property `ReduxOptions > train > losses > losses items > anyOf > contextualloss > layer_weights`](#train_losses_items_anyOf_i11_layer_weights)
          - [55.8.1.12.3.1. Property `ReduxOptions > train > losses > losses items > anyOf > contextualloss > layer_weights > anyOf > item 0`](#train_losses_items_anyOf_i11_layer_weights_anyOf_i0)
            - [55.8.1.12.3.1.1. Property `ReduxOptions > train > losses > losses items > anyOf > contextualloss > layer_weights > anyOf > item 0 > additionalProperties`](#train_losses_items_anyOf_i11_layer_weights_anyOf_i0_additionalProperties)
          - [55.8.1.12.3.2. Property `ReduxOptions > train > losses > losses items > anyOf > contextualloss > layer_weights > anyOf > item 1`](#train_losses_items_anyOf_i11_layer_weights_anyOf_i1)
        - [55.8.1.12.4. Property `ReduxOptions > train > losses > losses items > anyOf > contextualloss > crop_quarter`](#train_losses_items_anyOf_i11_crop_quarter)
        - [55.8.1.12.5. Property `ReduxOptions > train > losses > losses items > anyOf > contextualloss > max_1d_size`](#train_losses_items_anyOf_i11_max_1d_size)
        - [55.8.1.12.6. Property `ReduxOptions > train > losses > losses items > anyOf > contextualloss > distance_type`](#train_losses_items_anyOf_i11_distance_type)
        - [55.8.1.12.7. Property `ReduxOptions > train > losses > losses items > anyOf > contextualloss > b`](#train_losses_items_anyOf_i11_b)
        - [55.8.1.12.8. Property `ReduxOptions > train > losses > losses items > anyOf > contextualloss > band_width`](#train_losses_items_anyOf_i11_band_width)
        - [55.8.1.12.9. Property `ReduxOptions > train > losses > losses items > anyOf > contextualloss > use_vgg`](#train_losses_items_anyOf_i11_use_vgg)
        - [55.8.1.12.10. Property `ReduxOptions > train > losses > losses items > anyOf > contextualloss > net`](#train_losses_items_anyOf_i11_net)
        - [55.8.1.12.11. Property `ReduxOptions > train > losses > losses items > anyOf > contextualloss > calc_type`](#train_losses_items_anyOf_i11_calc_type)
        - [55.8.1.12.12. Property `ReduxOptions > train > losses > losses items > anyOf > contextualloss > z_norm`](#train_losses_items_anyOf_i11_z_norm)
      - [55.8.1.13. Property `ReduxOptions > train > losses > losses items > anyOf > distsloss`](#train_losses_items_anyOf_i12)
        - [55.8.1.13.1. Property `ReduxOptions > train > losses > losses items > anyOf > distsloss > type`](#train_losses_items_anyOf_i12_type)
        - [55.8.1.13.2. Property `ReduxOptions > train > losses > losses items > anyOf > distsloss > loss_weight`](#train_losses_items_anyOf_i12_loss_weight)
        - [55.8.1.13.3. Property `ReduxOptions > train > losses > losses items > anyOf > distsloss > as_loss`](#train_losses_items_anyOf_i12_as_loss)
        - [55.8.1.13.4. Property `ReduxOptions > train > losses > losses items > anyOf > distsloss > load_weights`](#train_losses_items_anyOf_i12_load_weights)
        - [55.8.1.13.5. Property `ReduxOptions > train > losses > losses items > anyOf > distsloss > use_input_norm`](#train_losses_items_anyOf_i12_use_input_norm)
        - [55.8.1.13.6. Property `ReduxOptions > train > losses > losses items > anyOf > distsloss > clip_min`](#train_losses_items_anyOf_i12_clip_min)
      - [55.8.1.14. Property `ReduxOptions > train > losses > losses items > anyOf > ffloss`](#train_losses_items_anyOf_i13)
        - [55.8.1.14.1. Property `ReduxOptions > train > losses > losses items > anyOf > ffloss > type`](#train_losses_items_anyOf_i13_type)
        - [55.8.1.14.2. Property `ReduxOptions > train > losses > losses items > anyOf > ffloss > loss_weight`](#train_losses_items_anyOf_i13_loss_weight)
        - [55.8.1.14.3. Property `ReduxOptions > train > losses > losses items > anyOf > ffloss > alpha`](#train_losses_items_anyOf_i13_alpha)
        - [55.8.1.14.4. Property `ReduxOptions > train > losses > losses items > anyOf > ffloss > patch_factor`](#train_losses_items_anyOf_i13_patch_factor)
        - [55.8.1.14.5. Property `ReduxOptions > train > losses > losses items > anyOf > ffloss > ave_spectrum`](#train_losses_items_anyOf_i13_ave_spectrum)
        - [55.8.1.14.6. Property `ReduxOptions > train > losses > losses items > anyOf > ffloss > log_matrix`](#train_losses_items_anyOf_i13_log_matrix)
        - [55.8.1.14.7. Property `ReduxOptions > train > losses > losses items > anyOf > ffloss > batch_matrix`](#train_losses_items_anyOf_i13_batch_matrix)
      - [55.8.1.15. Property `ReduxOptions > train > losses > losses items > anyOf > ldlloss`](#train_losses_items_anyOf_i14)
        - [55.8.1.15.1. Property `ReduxOptions > train > losses > losses items > anyOf > ldlloss > type`](#train_losses_items_anyOf_i14_type)
        - [55.8.1.15.2. Property `ReduxOptions > train > losses > losses items > anyOf > ldlloss > loss_weight`](#train_losses_items_anyOf_i14_loss_weight)
        - [55.8.1.15.3. Property `ReduxOptions > train > losses > losses items > anyOf > ldlloss > criterion`](#train_losses_items_anyOf_i14_criterion)
      - [55.8.1.16. Property `ReduxOptions > train > losses > losses items > anyOf > mssimloss`](#train_losses_items_anyOf_i15)
        - [55.8.1.16.1. Property `ReduxOptions > train > losses > losses items > anyOf > mssimloss > type`](#train_losses_items_anyOf_i15_type)
        - [55.8.1.16.2. Property `ReduxOptions > train > losses > losses items > anyOf > mssimloss > loss_weight`](#train_losses_items_anyOf_i15_loss_weight)
        - [55.8.1.16.3. Property `ReduxOptions > train > losses > losses items > anyOf > mssimloss > window_size`](#train_losses_items_anyOf_i15_window_size)
        - [55.8.1.16.4. Property `ReduxOptions > train > losses > losses items > anyOf > mssimloss > in_channels`](#train_losses_items_anyOf_i15_in_channels)
        - [55.8.1.16.5. Property `ReduxOptions > train > losses > losses items > anyOf > mssimloss > sigma`](#train_losses_items_anyOf_i15_sigma)
        - [55.8.1.16.6. Property `ReduxOptions > train > losses > losses items > anyOf > mssimloss > k1`](#train_losses_items_anyOf_i15_k1)
        - [55.8.1.16.7. Property `ReduxOptions > train > losses > losses items > anyOf > mssimloss > k2`](#train_losses_items_anyOf_i15_k2)
        - [55.8.1.16.8. Property `ReduxOptions > train > losses > losses items > anyOf > mssimloss > l`](#train_losses_items_anyOf_i15_l)
        - [55.8.1.16.9. Property `ReduxOptions > train > losses > losses items > anyOf > mssimloss > padding`](#train_losses_items_anyOf_i15_padding)
          - [55.8.1.16.9.1. Property `ReduxOptions > train > losses > losses items > anyOf > mssimloss > padding > anyOf > item 0`](#train_losses_items_anyOf_i15_padding_anyOf_i0)
          - [55.8.1.16.9.2. Property `ReduxOptions > train > losses > losses items > anyOf > mssimloss > padding > anyOf > item 1`](#train_losses_items_anyOf_i15_padding_anyOf_i1)
        - [55.8.1.16.10. Property `ReduxOptions > train > losses > losses items > anyOf > mssimloss > cosim`](#train_losses_items_anyOf_i15_cosim)
        - [55.8.1.16.11. Property `ReduxOptions > train > losses > losses items > anyOf > mssimloss > cosim_lambda`](#train_losses_items_anyOf_i15_cosim_lambda)
      - [55.8.1.17. Property `ReduxOptions > train > losses > losses items > anyOf > msssiml1loss`](#train_losses_items_anyOf_i16)
        - [55.8.1.17.1. Property `ReduxOptions > train > losses > losses items > anyOf > msssiml1loss > type`](#train_losses_items_anyOf_i16_type)
        - [55.8.1.17.2. Property `ReduxOptions > train > losses > losses items > anyOf > msssiml1loss > loss_weight`](#train_losses_items_anyOf_i16_loss_weight)
        - [55.8.1.17.3. Property `ReduxOptions > train > losses > losses items > anyOf > msssiml1loss > gaussian_sigmas`](#train_losses_items_anyOf_i16_gaussian_sigmas)
          - [55.8.1.17.3.1. Property `ReduxOptions > train > losses > losses items > anyOf > msssiml1loss > gaussian_sigmas > anyOf > item 0`](#train_losses_items_anyOf_i16_gaussian_sigmas_anyOf_i0)
            - [55.8.1.17.3.1.1. ReduxOptions > train > losses > losses items > anyOf > msssiml1loss > gaussian_sigmas > anyOf > item 0 > item 0 items](#train_losses_items_anyOf_i16_gaussian_sigmas_anyOf_i0_items)
          - [55.8.1.17.3.2. Property `ReduxOptions > train > losses > losses items > anyOf > msssiml1loss > gaussian_sigmas > anyOf > item 1`](#train_losses_items_anyOf_i16_gaussian_sigmas_anyOf_i1)
        - [55.8.1.17.4. Property `ReduxOptions > train > losses > losses items > anyOf > msssiml1loss > data_range`](#train_losses_items_anyOf_i16_data_range)
        - [55.8.1.17.5. Property `ReduxOptions > train > losses > losses items > anyOf > msssiml1loss > k`](#train_losses_items_anyOf_i16_k)
          - [55.8.1.17.5.1. ReduxOptions > train > losses > losses items > anyOf > msssiml1loss > k > k item 0](#autogenerated_heading_36)
          - [55.8.1.17.5.2. ReduxOptions > train > losses > losses items > anyOf > msssiml1loss > k > k item 1](#autogenerated_heading_37)
        - [55.8.1.17.6. Property `ReduxOptions > train > losses > losses items > anyOf > msssiml1loss > alpha`](#train_losses_items_anyOf_i16_alpha)
        - [55.8.1.17.7. Property `ReduxOptions > train > losses > losses items > anyOf > msssiml1loss > cuda_dev`](#train_losses_items_anyOf_i16_cuda_dev)
      - [55.8.1.18. Property `ReduxOptions > train > losses > losses items > anyOf > nccloss`](#train_losses_items_anyOf_i17)
        - [55.8.1.18.1. Property `ReduxOptions > train > losses > losses items > anyOf > nccloss > type`](#train_losses_items_anyOf_i17_type)
        - [55.8.1.18.2. Property `ReduxOptions > train > losses > losses items > anyOf > nccloss > loss_weight`](#train_losses_items_anyOf_i17_loss_weight)
      - [55.8.1.19. Property `ReduxOptions > train > losses > losses items > anyOf > perceptualfp16loss`](#train_losses_items_anyOf_i18)
        - [55.8.1.19.1. Property `ReduxOptions > train > losses > losses items > anyOf > perceptualfp16loss > type`](#train_losses_items_anyOf_i18_type)
        - [55.8.1.19.2. Property `ReduxOptions > train > losses > losses items > anyOf > perceptualfp16loss > loss_weight`](#train_losses_items_anyOf_i18_loss_weight)
        - [55.8.1.19.3. Property `ReduxOptions > train > losses > losses items > anyOf > perceptualfp16loss > layer_weights`](#train_losses_items_anyOf_i18_layer_weights)
          - [55.8.1.19.3.1. Property `ReduxOptions > train > losses > losses items > anyOf > perceptualfp16loss > layer_weights > anyOf > item 0`](#train_losses_items_anyOf_i18_layer_weights_anyOf_i0)
            - [55.8.1.19.3.1.1. Property `ReduxOptions > train > losses > losses items > anyOf > perceptualfp16loss > layer_weights > anyOf > item 0 > additionalProperties`](#train_losses_items_anyOf_i18_layer_weights_anyOf_i0_additionalProperties)
          - [55.8.1.19.3.2. Property `ReduxOptions > train > losses > losses items > anyOf > perceptualfp16loss > layer_weights > anyOf > item 1`](#train_losses_items_anyOf_i18_layer_weights_anyOf_i1)
        - [55.8.1.19.4. Property `ReduxOptions > train > losses > losses items > anyOf > perceptualfp16loss > w_lambda`](#train_losses_items_anyOf_i18_w_lambda)
        - [55.8.1.19.5. Property `ReduxOptions > train > losses > losses items > anyOf > perceptualfp16loss > alpha`](#train_losses_items_anyOf_i18_alpha)
          - [55.8.1.19.5.1. Property `ReduxOptions > train > losses > losses items > anyOf > perceptualfp16loss > alpha > anyOf > item 0`](#train_losses_items_anyOf_i18_alpha_anyOf_i0)
            - [55.8.1.19.5.1.1. ReduxOptions > train > losses > losses items > anyOf > perceptualfp16loss > alpha > anyOf > item 0 > item 0 items](#train_losses_items_anyOf_i18_alpha_anyOf_i0_items)
          - [55.8.1.19.5.2. Property `ReduxOptions > train > losses > losses items > anyOf > perceptualfp16loss > alpha > anyOf > item 1`](#train_losses_items_anyOf_i18_alpha_anyOf_i1)
        - [55.8.1.19.6. Property `ReduxOptions > train > losses > losses items > anyOf > perceptualfp16loss > criterion`](#train_losses_items_anyOf_i18_criterion)
        - [55.8.1.19.7. Property `ReduxOptions > train > losses > losses items > anyOf > perceptualfp16loss > num_proj_fd`](#train_losses_items_anyOf_i18_num_proj_fd)
        - [55.8.1.19.8. Property `ReduxOptions > train > losses > losses items > anyOf > perceptualfp16loss > phase_weight_fd`](#train_losses_items_anyOf_i18_phase_weight_fd)
        - [55.8.1.19.9. Property `ReduxOptions > train > losses > losses items > anyOf > perceptualfp16loss > stride_fd`](#train_losses_items_anyOf_i18_stride_fd)
      - [55.8.1.20. Property `ReduxOptions > train > losses > losses items > anyOf > perceptualloss`](#train_losses_items_anyOf_i19)
        - [55.8.1.20.1. Property `ReduxOptions > train > losses > losses items > anyOf > perceptualloss > type`](#train_losses_items_anyOf_i19_type)
        - [55.8.1.20.2. Property `ReduxOptions > train > losses > losses items > anyOf > perceptualloss > loss_weight`](#train_losses_items_anyOf_i19_loss_weight)
        - [55.8.1.20.3. Property `ReduxOptions > train > losses > losses items > anyOf > perceptualloss > layer_weights`](#train_losses_items_anyOf_i19_layer_weights)
          - [55.8.1.20.3.1. Property `ReduxOptions > train > losses > losses items > anyOf > perceptualloss > layer_weights > anyOf > item 0`](#train_losses_items_anyOf_i19_layer_weights_anyOf_i0)
            - [55.8.1.20.3.1.1. Property `ReduxOptions > train > losses > losses items > anyOf > perceptualloss > layer_weights > anyOf > item 0 > additionalProperties`](#train_losses_items_anyOf_i19_layer_weights_anyOf_i0_additionalProperties)
          - [55.8.1.20.3.2. Property `ReduxOptions > train > losses > losses items > anyOf > perceptualloss > layer_weights > anyOf > item 1`](#train_losses_items_anyOf_i19_layer_weights_anyOf_i1)
        - [55.8.1.20.4. Property `ReduxOptions > train > losses > losses items > anyOf > perceptualloss > w_lambda`](#train_losses_items_anyOf_i19_w_lambda)
        - [55.8.1.20.5. Property `ReduxOptions > train > losses > losses items > anyOf > perceptualloss > alpha`](#train_losses_items_anyOf_i19_alpha)
          - [55.8.1.20.5.1. Property `ReduxOptions > train > losses > losses items > anyOf > perceptualloss > alpha > anyOf > item 0`](#train_losses_items_anyOf_i19_alpha_anyOf_i0)
            - [55.8.1.20.5.1.1. ReduxOptions > train > losses > losses items > anyOf > perceptualloss > alpha > anyOf > item 0 > item 0 items](#train_losses_items_anyOf_i19_alpha_anyOf_i0_items)
          - [55.8.1.20.5.2. Property `ReduxOptions > train > losses > losses items > anyOf > perceptualloss > alpha > anyOf > item 1`](#train_losses_items_anyOf_i19_alpha_anyOf_i1)
        - [55.8.1.20.6. Property `ReduxOptions > train > losses > losses items > anyOf > perceptualloss > criterion`](#train_losses_items_anyOf_i19_criterion)
        - [55.8.1.20.7. Property `ReduxOptions > train > losses > losses items > anyOf > perceptualloss > num_proj_fd`](#train_losses_items_anyOf_i19_num_proj_fd)
        - [55.8.1.20.8. Property `ReduxOptions > train > losses > losses items > anyOf > perceptualloss > phase_weight_fd`](#train_losses_items_anyOf_i19_phase_weight_fd)
        - [55.8.1.20.9. Property `ReduxOptions > train > losses > losses items > anyOf > perceptualloss > stride_fd`](#train_losses_items_anyOf_i19_stride_fd)
      - [55.8.1.21. Property `ReduxOptions > train > losses > losses items > anyOf > topiqloss`](#train_losses_items_anyOf_i20)
        - [55.8.1.21.1. Property `ReduxOptions > train > losses > losses items > anyOf > topiqloss > type`](#train_losses_items_anyOf_i20_type)
        - [55.8.1.21.2. Property `ReduxOptions > train > losses > losses items > anyOf > topiqloss > loss_weight`](#train_losses_items_anyOf_i20_loss_weight)
        - [55.8.1.21.3. Property `ReduxOptions > train > losses > losses items > anyOf > topiqloss > resize_input`](#train_losses_items_anyOf_i20_resize_input)
  - [55.9. Property `ReduxOptions > train > pixel_opt`](#train_pixel_opt)
    - [55.9.1. Property `ReduxOptions > train > pixel_opt > anyOf > item 0`](#train_pixel_opt_anyOf_i0)
    - [55.9.2. Property `ReduxOptions > train > pixel_opt > anyOf > item 1`](#train_pixel_opt_anyOf_i1)
  - [55.10. Property `ReduxOptions > train > mssim_opt`](#train_mssim_opt)
    - [55.10.1. Property `ReduxOptions > train > mssim_opt > anyOf > item 0`](#train_mssim_opt_anyOf_i0)
    - [55.10.2. Property `ReduxOptions > train > mssim_opt > anyOf > item 1`](#train_mssim_opt_anyOf_i1)
  - [55.11. Property `ReduxOptions > train > ms_ssim_l1_opt`](#train_ms_ssim_l1_opt)
    - [55.11.1. Property `ReduxOptions > train > ms_ssim_l1_opt > anyOf > item 0`](#train_ms_ssim_l1_opt_anyOf_i0)
    - [55.11.2. Property `ReduxOptions > train > ms_ssim_l1_opt > anyOf > item 1`](#train_ms_ssim_l1_opt_anyOf_i1)
  - [55.12. Property `ReduxOptions > train > perceptual_opt`](#train_perceptual_opt)
    - [55.12.1. Property `ReduxOptions > train > perceptual_opt > anyOf > item 0`](#train_perceptual_opt_anyOf_i0)
    - [55.12.2. Property `ReduxOptions > train > perceptual_opt > anyOf > item 1`](#train_perceptual_opt_anyOf_i1)
  - [55.13. Property `ReduxOptions > train > contextual_opt`](#train_contextual_opt)
    - [55.13.1. Property `ReduxOptions > train > contextual_opt > anyOf > item 0`](#train_contextual_opt_anyOf_i0)
    - [55.13.2. Property `ReduxOptions > train > contextual_opt > anyOf > item 1`](#train_contextual_opt_anyOf_i1)
  - [55.14. Property `ReduxOptions > train > dists_opt`](#train_dists_opt)
    - [55.14.1. Property `ReduxOptions > train > dists_opt > anyOf > item 0`](#train_dists_opt_anyOf_i0)
    - [55.14.2. Property `ReduxOptions > train > dists_opt > anyOf > item 1`](#train_dists_opt_anyOf_i1)
  - [55.15. Property `ReduxOptions > train > hr_inversion_opt`](#train_hr_inversion_opt)
    - [55.15.1. Property `ReduxOptions > train > hr_inversion_opt > anyOf > item 0`](#train_hr_inversion_opt_anyOf_i0)
    - [55.15.2. Property `ReduxOptions > train > hr_inversion_opt > anyOf > item 1`](#train_hr_inversion_opt_anyOf_i1)
  - [55.16. Property `ReduxOptions > train > dinov2_opt`](#train_dinov2_opt)
    - [55.16.1. Property `ReduxOptions > train > dinov2_opt > anyOf > item 0`](#train_dinov2_opt_anyOf_i0)
    - [55.16.2. Property `ReduxOptions > train > dinov2_opt > anyOf > item 1`](#train_dinov2_opt_anyOf_i1)
  - [55.17. Property `ReduxOptions > train > topiq_opt`](#train_topiq_opt)
    - [55.17.1. Property `ReduxOptions > train > topiq_opt > anyOf > item 0`](#train_topiq_opt_anyOf_i0)
    - [55.17.2. Property `ReduxOptions > train > topiq_opt > anyOf > item 1`](#train_topiq_opt_anyOf_i1)
  - [55.18. Property `ReduxOptions > train > pd_opt`](#train_pd_opt)
    - [55.18.1. Property `ReduxOptions > train > pd_opt > anyOf > item 0`](#train_pd_opt_anyOf_i0)
    - [55.18.2. Property `ReduxOptions > train > pd_opt > anyOf > item 1`](#train_pd_opt_anyOf_i1)
  - [55.19. Property `ReduxOptions > train > fd_opt`](#train_fd_opt)
    - [55.19.1. Property `ReduxOptions > train > fd_opt > anyOf > item 0`](#train_fd_opt_anyOf_i0)
    - [55.19.2. Property `ReduxOptions > train > fd_opt > anyOf > item 1`](#train_fd_opt_anyOf_i1)
  - [55.20. Property `ReduxOptions > train > ldl_opt`](#train_ldl_opt)
    - [55.20.1. Property `ReduxOptions > train > ldl_opt > anyOf > item 0`](#train_ldl_opt_anyOf_i0)
    - [55.20.2. Property `ReduxOptions > train > ldl_opt > anyOf > item 1`](#train_ldl_opt_anyOf_i1)
  - [55.21. Property `ReduxOptions > train > hsluv_opt`](#train_hsluv_opt)
    - [55.21.1. Property `ReduxOptions > train > hsluv_opt > anyOf > item 0`](#train_hsluv_opt_anyOf_i0)
    - [55.21.2. Property `ReduxOptions > train > hsluv_opt > anyOf > item 1`](#train_hsluv_opt_anyOf_i1)
  - [55.22. Property `ReduxOptions > train > gan_opt`](#train_gan_opt)
    - [55.22.1. Property `ReduxOptions > train > gan_opt > anyOf > item 0`](#train_gan_opt_anyOf_i0)
    - [55.22.2. Property `ReduxOptions > train > gan_opt > anyOf > item 1`](#train_gan_opt_anyOf_i1)
  - [55.23. Property `ReduxOptions > train > color_opt`](#train_color_opt)
    - [55.23.1. Property `ReduxOptions > train > color_opt > anyOf > item 0`](#train_color_opt_anyOf_i0)
    - [55.23.2. Property `ReduxOptions > train > color_opt > anyOf > item 1`](#train_color_opt_anyOf_i1)
  - [55.24. Property `ReduxOptions > train > luma_opt`](#train_luma_opt)
    - [55.24.1. Property `ReduxOptions > train > luma_opt > anyOf > item 0`](#train_luma_opt_anyOf_i0)
    - [55.24.2. Property `ReduxOptions > train > luma_opt > anyOf > item 1`](#train_luma_opt_anyOf_i1)
  - [55.25. Property `ReduxOptions > train > avg_opt`](#train_avg_opt)
    - [55.25.1. Property `ReduxOptions > train > avg_opt > anyOf > item 0`](#train_avg_opt_anyOf_i0)
    - [55.25.2. Property `ReduxOptions > train > avg_opt > anyOf > item 1`](#train_avg_opt_anyOf_i1)
  - [55.26. Property `ReduxOptions > train > bicubic_opt`](#train_bicubic_opt)
    - [55.26.1. Property `ReduxOptions > train > bicubic_opt > anyOf > item 0`](#train_bicubic_opt_anyOf_i0)
    - [55.26.2. Property `ReduxOptions > train > bicubic_opt > anyOf > item 1`](#train_bicubic_opt_anyOf_i1)
  - [55.27. Property `ReduxOptions > train > use_moa`](#train_use_moa)
  - [55.28. Property `ReduxOptions > train > moa_augs`](#train_moa_augs)
    - [55.28.1. ReduxOptions > train > moa_augs > moa_augs items](#train_moa_augs_items)
  - [55.29. Property `ReduxOptions > train > moa_probs`](#train_moa_probs)
    - [55.29.1. ReduxOptions > train > moa_probs > moa_probs items](#train_moa_probs_items)
  - [55.30. Property `ReduxOptions > train > moa_debug`](#train_moa_debug)
  - [55.31. Property `ReduxOptions > train > moa_debug_limit`](#train_moa_debug_limit)
- [56. Property `ReduxOptions > val`](#val)
  - [56.1. Property `ReduxOptions > val > anyOf > item 0`](#val_anyOf_i0)
  - [56.2. Property `ReduxOptions > val > anyOf > ValOptions`](#val_anyOf_i1)
    - [56.2.1. Property `ReduxOptions > val > anyOf > ValOptions > val_enabled`](#val_anyOf_i1_val_enabled)
    - [56.2.2. Property `ReduxOptions > val > anyOf > ValOptions > save_img`](#val_anyOf_i1_save_img)
    - [56.2.3. Property `ReduxOptions > val > anyOf > ValOptions > val_freq`](#val_anyOf_i1_val_freq)
      - [56.2.3.1. Property `ReduxOptions > val > anyOf > ValOptions > val_freq > anyOf > item 0`](#val_anyOf_i1_val_freq_anyOf_i0)
      - [56.2.3.2. Property `ReduxOptions > val > anyOf > ValOptions > val_freq > anyOf > item 1`](#val_anyOf_i1_val_freq_anyOf_i1)
    - [56.2.4. Property `ReduxOptions > val > anyOf > ValOptions > suffix`](#val_anyOf_i1_suffix)
      - [56.2.4.1. Property `ReduxOptions > val > anyOf > ValOptions > suffix > anyOf > item 0`](#val_anyOf_i1_suffix_anyOf_i0)
      - [56.2.4.2. Property `ReduxOptions > val > anyOf > ValOptions > suffix > anyOf > item 1`](#val_anyOf_i1_suffix_anyOf_i1)
    - [56.2.5. Property `ReduxOptions > val > anyOf > ValOptions > metrics_enabled`](#val_anyOf_i1_metrics_enabled)
    - [56.2.6. Property `ReduxOptions > val > anyOf > ValOptions > metrics`](#val_anyOf_i1_metrics)
      - [56.2.6.1. Property `ReduxOptions > val > anyOf > ValOptions > metrics > anyOf > item 0`](#val_anyOf_i1_metrics_anyOf_i0)
      - [56.2.6.2. Property `ReduxOptions > val > anyOf > ValOptions > metrics > anyOf > item 1`](#val_anyOf_i1_metrics_anyOf_i1)
    - [56.2.7. Property `ReduxOptions > val > anyOf > ValOptions > pbar`](#val_anyOf_i1_pbar)
- [57. Property `ReduxOptions > logger`](#logger)
  - [57.1. Property `ReduxOptions > logger > anyOf > item 0`](#logger_anyOf_i0)
  - [57.2. Property `ReduxOptions > logger > anyOf > LogOptions`](#logger_anyOf_i1)
    - [57.2.1. Property `ReduxOptions > logger > anyOf > LogOptions > print_freq`](#logger_anyOf_i1_print_freq)
    - [57.2.2. Property `ReduxOptions > logger > anyOf > LogOptions > save_checkpoint_freq`](#logger_anyOf_i1_save_checkpoint_freq)
    - [57.2.3. Property `ReduxOptions > logger > anyOf > LogOptions > use_tb_logger`](#logger_anyOf_i1_use_tb_logger)
    - [57.2.4. Property `ReduxOptions > logger > anyOf > LogOptions > save_checkpoint_format`](#logger_anyOf_i1_save_checkpoint_format)
    - [57.2.5. Property `ReduxOptions > logger > anyOf > LogOptions > wandb`](#logger_anyOf_i1_wandb)
      - [57.2.5.1. Property `ReduxOptions > logger > anyOf > LogOptions > wandb > anyOf > item 0`](#logger_anyOf_i1_wandb_anyOf_i0)
      - [57.2.5.2. Property `ReduxOptions > logger > anyOf > LogOptions > wandb > anyOf > WandbOptions`](#logger_anyOf_i1_wandb_anyOf_i1)
        - [57.2.5.2.1. Property `ReduxOptions > logger > anyOf > LogOptions > wandb > anyOf > WandbOptions > resume_id`](#logger_anyOf_i1_wandb_anyOf_i1_resume_id)
          - [57.2.5.2.1.1. Property `ReduxOptions > logger > anyOf > LogOptions > wandb > anyOf > WandbOptions > resume_id > anyOf > item 0`](#logger_anyOf_i1_wandb_anyOf_i1_resume_id_anyOf_i0)
          - [57.2.5.2.1.2. Property `ReduxOptions > logger > anyOf > LogOptions > wandb > anyOf > WandbOptions > resume_id > anyOf > item 1`](#logger_anyOf_i1_wandb_anyOf_i1_resume_id_anyOf_i1)
        - [57.2.5.2.2. Property `ReduxOptions > logger > anyOf > LogOptions > wandb > anyOf > WandbOptions > project`](#logger_anyOf_i1_wandb_anyOf_i1_project)
          - [57.2.5.2.2.1. Property `ReduxOptions > logger > anyOf > LogOptions > wandb > anyOf > WandbOptions > project > anyOf > item 0`](#logger_anyOf_i1_wandb_anyOf_i1_project_anyOf_i0)
          - [57.2.5.2.2.2. Property `ReduxOptions > logger > anyOf > LogOptions > wandb > anyOf > WandbOptions > project > anyOf > item 1`](#logger_anyOf_i1_wandb_anyOf_i1_project_anyOf_i1)
- [58. Property `ReduxOptions > dist_params`](#dist_params)
  - [58.1. Property `ReduxOptions > dist_params > anyOf > item 0`](#dist_params_anyOf_i0)
  - [58.2. Property `ReduxOptions > dist_params > anyOf > item 1`](#dist_params_anyOf_i1)
- [59. Property `ReduxOptions > onnx`](#onnx)
  - [59.1. Property `ReduxOptions > onnx > anyOf > item 0`](#onnx_anyOf_i0)
  - [59.2. Property `ReduxOptions > onnx > anyOf > OnnxOptions`](#onnx_anyOf_i1)
    - [59.2.1. Property `ReduxOptions > onnx > anyOf > OnnxOptions > dynamo`](#onnx_anyOf_i1_dynamo)
    - [59.2.2. Property `ReduxOptions > onnx > anyOf > OnnxOptions > opset`](#onnx_anyOf_i1_opset)
    - [59.2.3. Property `ReduxOptions > onnx > anyOf > OnnxOptions > use_static_shapes`](#onnx_anyOf_i1_use_static_shapes)
    - [59.2.4. Property `ReduxOptions > onnx > anyOf > OnnxOptions > shape`](#onnx_anyOf_i1_shape)
    - [59.2.5. Property `ReduxOptions > onnx > anyOf > OnnxOptions > verify`](#onnx_anyOf_i1_verify)
    - [59.2.6. Property `ReduxOptions > onnx > anyOf > OnnxOptions > fp16`](#onnx_anyOf_i1_fp16)
    - [59.2.7. Property `ReduxOptions > onnx > anyOf > OnnxOptions > optimize`](#onnx_anyOf_i1_optimize)
- [60. Property `ReduxOptions > find_unused_parameters`](#find_unused_parameters)

**Title:** ReduxOptions

|                           |                      |
| ------------------------- | -------------------- |
| **Type**                  | `object`             |
| **Required**              | No                   |
| **Additional properties** | Any type allowed     |
| **Defined in**            | #/$defs/ReduxOptions |

| Property                                                                       | Pattern | Type            | Deprecated | Definition              | Title/Description                                                                                                                                                                                                                                                                                         |
| ------------------------------------------------------------------------------ | ------- | --------------- | ---------- | ----------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| + [name](#name )                                                               | No      | string          | No         | -                       | Name of the experiment. It should be a unique name. If you enable auto resume, any experiment with this name will be resumed instead of starting a new training run.                                                                                                                                      |
| + [scale](#scale )                                                             | No      | integer         | No         | -                       | Scale of the model. Most architectures support a scale of 1, 2, 3, 4, or 8. A scale of 1 can be used for restoration models that don't change the resolution of the input image. A scale of 2 means the width and height of the input image are doubled, so a 640x480 input will be upscaled to 1280x960. |
| + [num_gpu](#num_gpu )                                                         | No      | Combination     | No         | -                       | The number of GPUs to use for training, if using multiple GPUs.                                                                                                                                                                                                                                           |
| + [path](#path )                                                               | No      | object          | No         | In #/$defs/PathOptions  | PathOptions                                                                                                                                                                                                                                                                                               |
| + [network_g](#network_g )                                                     | No      | Combination     | No         | -                       | -                                                                                                                                                                                                                                                                                                         |
| - [network_d](#network_d )                                                     | No      | Combination     | No         | -                       | The options for the discriminator model.                                                                                                                                                                                                                                                                  |
| - [manual_seed](#manual_seed )                                                 | No      | Combination     | No         | -                       | Deterministic mode, slows down training. Only use for reproducible experiments.                                                                                                                                                                                                                           |
| - [deterministic](#deterministic )                                             | No      | Combination     | No         | -                       | -                                                                                                                                                                                                                                                                                                         |
| - [dist](#dist )                                                               | No      | Combination     | No         | -                       | -                                                                                                                                                                                                                                                                                                         |
| - [launcher](#launcher )                                                       | No      | Combination     | No         | -                       | -                                                                                                                                                                                                                                                                                                         |
| - [rank](#rank )                                                               | No      | Combination     | No         | -                       | -                                                                                                                                                                                                                                                                                                         |
| - [world_size](#world_size )                                                   | No      | Combination     | No         | -                       | -                                                                                                                                                                                                                                                                                                         |
| - [auto_resume](#auto_resume )                                                 | No      | Combination     | No         | -                       | -                                                                                                                                                                                                                                                                                                         |
| - [resume](#resume )                                                           | No      | integer         | No         | -                       | -                                                                                                                                                                                                                                                                                                         |
| - [is_train](#is_train )                                                       | No      | Combination     | No         | -                       | -                                                                                                                                                                                                                                                                                                         |
| - [root_path](#root_path )                                                     | No      | Combination     | No         | -                       | -                                                                                                                                                                                                                                                                                                         |
| - [use_amp](#use_amp )                                                         | No      | boolean         | No         | -                       | Speed up training and reduce VRAM usage. NVIDIA only.                                                                                                                                                                                                                                                     |
| - [amp_bf16](#amp_bf16 )                                                       | No      | boolean         | No         | -                       | Use bf16 instead of fp16 for AMP, RTX 3000 series or newer only. Only recommended if fp16 doesn't work.                                                                                                                                                                                                   |
| - [use_channels_last](#use_channels_last )                                     | No      | boolean         | No         | -                       | Enable channels last memory format while using AMP. Reduces VRAM and speeds up training for most architectures, but some architectures are slower with channels last.                                                                                                                                     |
| - [fast_matmul](#fast_matmul )                                                 | No      | boolean         | No         | -                       | Trade precision for performance.                                                                                                                                                                                                                                                                          |
| - [use_compile](#use_compile )                                                 | No      | boolean         | No         | -                       | Enable torch.compile for the generator model, which takes time on startup to compile the model, but can speed up training after the model is compiled.                                                                                                                                                    |
| - [detect_anomaly](#detect_anomaly )                                           | No      | boolean         | No         | -                       | Whether or not to enable anomaly detection, which can be useful for debugging NaNs that occur during training. Has a significant performance hit and should be disabled when not debugging.                                                                                                               |
| - [high_order_degradation](#high_order_degradation )                           | No      | boolean         | No         | -                       | Whether or not to enable OTF (on the fly) degradations, which generates LRs on the fly.                                                                                                                                                                                                                   |
| - [high_order_degradations_debug](#high_order_degradations_debug )             | No      | boolean         | No         | -                       | Whether or not to enable debugging for OTF, which saves the OTF generated LR images so they can be inspected to view the effect of different OTF settings.                                                                                                                                                |
| - [high_order_degradations_debug_limit](#high_order_degradations_debug_limit ) | No      | integer         | No         | -                       | The maximum number of OTF images to save when debugging is enabled.                                                                                                                                                                                                                                       |
| - [dataroot_lq_prob](#dataroot_lq_prob )                                       | No      | number          | No         | -                       | Probability of using paired LR data instead of OTF LR data.                                                                                                                                                                                                                                               |
| - [lq_usm](#lq_usm )                                                           | No      | boolean         | No         | -                       | -                                                                                                                                                                                                                                                                                                         |
| - [lq_usm_radius_range](#lq_usm_radius_range )                                 | No      | array           | No         | -                       | For the unsharp mask of the LQ image, use a radius randomly selected from this range.                                                                                                                                                                                                                     |
| - [blur_prob](#blur_prob )                                                     | No      | number          | No         | -                       | -                                                                                                                                                                                                                                                                                                         |
| - [resize_prob](#resize_prob )                                                 | No      | array of number | No         | -                       | -                                                                                                                                                                                                                                                                                                         |
| - [resize_mode_list](#resize_mode_list )                                       | No      | array of string | No         | -                       | -                                                                                                                                                                                                                                                                                                         |
| - [resize_mode_prob](#resize_mode_prob )                                       | No      | array of number | No         | -                       | -                                                                                                                                                                                                                                                                                                         |
| - [resize_range](#resize_range )                                               | No      | array           | No         | -                       | -                                                                                                                                                                                                                                                                                                         |
| - [gaussian_noise_prob](#gaussian_noise_prob )                                 | No      | number          | No         | -                       | -                                                                                                                                                                                                                                                                                                         |
| - [noise_range](#noise_range )                                                 | No      | array           | No         | -                       | -                                                                                                                                                                                                                                                                                                         |
| - [poisson_scale_range](#poisson_scale_range )                                 | No      | array           | No         | -                       | -                                                                                                                                                                                                                                                                                                         |
| - [gray_noise_prob](#gray_noise_prob )                                         | No      | number          | No         | -                       | -                                                                                                                                                                                                                                                                                                         |
| - [jpeg_prob](#jpeg_prob )                                                     | No      | number          | No         | -                       | -                                                                                                                                                                                                                                                                                                         |
| - [jpeg_range](#jpeg_range )                                                   | No      | array           | No         | -                       | -                                                                                                                                                                                                                                                                                                         |
| - [blur_prob2](#blur_prob2 )                                                   | No      | number          | No         | -                       | -                                                                                                                                                                                                                                                                                                         |
| - [resize_prob2](#resize_prob2 )                                               | No      | array of number | No         | -                       | -                                                                                                                                                                                                                                                                                                         |
| - [resize_mode_list2](#resize_mode_list2 )                                     | No      | array of string | No         | -                       | -                                                                                                                                                                                                                                                                                                         |
| - [resize_mode_prob2](#resize_mode_prob2 )                                     | No      | array of number | No         | -                       | -                                                                                                                                                                                                                                                                                                         |
| - [resize_range2](#resize_range2 )                                             | No      | array           | No         | -                       | -                                                                                                                                                                                                                                                                                                         |
| - [gaussian_noise_prob2](#gaussian_noise_prob2 )                               | No      | number          | No         | -                       | -                                                                                                                                                                                                                                                                                                         |
| - [noise_range2](#noise_range2 )                                               | No      | array           | No         | -                       | -                                                                                                                                                                                                                                                                                                         |
| - [poisson_scale_range2](#poisson_scale_range2 )                               | No      | array           | No         | -                       | -                                                                                                                                                                                                                                                                                                         |
| - [gray_noise_prob2](#gray_noise_prob2 )                                       | No      | number          | No         | -                       | -                                                                                                                                                                                                                                                                                                         |
| - [jpeg_prob2](#jpeg_prob2 )                                                   | No      | number          | No         | -                       | -                                                                                                                                                                                                                                                                                                         |
| - [jpeg_range2](#jpeg_range2 )                                                 | No      | array of number | No         | -                       | -                                                                                                                                                                                                                                                                                                         |
| - [resize_mode_list3](#resize_mode_list3 )                                     | No      | array of string | No         | -                       | -                                                                                                                                                                                                                                                                                                         |
| - [resize_mode_prob3](#resize_mode_prob3 )                                     | No      | array of number | No         | -                       | -                                                                                                                                                                                                                                                                                                         |
| - [queue_size](#queue_size )                                                   | No      | integer         | No         | -                       | -                                                                                                                                                                                                                                                                                                         |
| - [datasets](#datasets )                                                       | No      | object          | No         | -                       | -                                                                                                                                                                                                                                                                                                         |
| - [train](#train )                                                             | No      | object          | No         | In #/$defs/TrainOptions | TrainOptions                                                                                                                                                                                                                                                                                              |
| - [val](#val )                                                                 | No      | Combination     | No         | -                       | -                                                                                                                                                                                                                                                                                                         |
| - [logger](#logger )                                                           | No      | Combination     | No         | -                       | -                                                                                                                                                                                                                                                                                                         |
| - [dist_params](#dist_params )                                                 | No      | Combination     | No         | -                       | -                                                                                                                                                                                                                                                                                                         |
| - [onnx](#onnx )                                                               | No      | Combination     | No         | -                       | -                                                                                                                                                                                                                                                                                                         |
| - [find_unused_parameters](#find_unused_parameters )                           | No      | boolean         | No         | -                       | -                                                                                                                                                                                                                                                                                                         |

## <a name="name"></a>1. Property `ReduxOptions > name`

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | Yes      |

**Description:** Name of the experiment. It should be a unique name. If you enable auto resume, any experiment with this name will be resumed instead of starting a new training run.

## <a name="scale"></a>2. Property `ReduxOptions > scale`

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | Yes       |

**Description:** Scale of the model. Most architectures support a scale of 1, 2, 3, 4, or 8. A scale of 1 can be used for restoration models that don't change the resolution of the input image. A scale of 2 means the width and height of the input image are doubled, so a 640x480 input will be upscaled to 1280x960.

## <a name="num_gpu"></a>3. Property `ReduxOptions > num_gpu`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `combining`      |
| **Required**              | Yes              |
| **Additional properties** | Any type allowed |

**Description:** The number of GPUs to use for training, if using multiple GPUs.

| Any of(Option)              |
| --------------------------- |
| [item 0](#num_gpu_anyOf_i0) |
| [item 1](#num_gpu_anyOf_i1) |

### <a name="num_gpu_anyOf_i0"></a>3.1. Property `ReduxOptions > num_gpu > anyOf > item 0`

|              |                    |
| ------------ | ------------------ |
| **Type**     | `enum (of string)` |
| **Required** | No                 |

Must be one of:
* "auto"

### <a name="num_gpu_anyOf_i1"></a>3.2. Property `ReduxOptions > num_gpu > anyOf > item 1`

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | No        |

## <a name="path"></a>4. Property `ReduxOptions > path`

**Title:** PathOptions

|                           |                     |
| ------------------------- | ------------------- |
| **Type**                  | `object`            |
| **Required**              | Yes                 |
| **Additional properties** | Not allowed         |
| **Defined in**            | #/$defs/PathOptions |

| Property                                                    | Pattern | Type        | Deprecated | Definition | Title/Description |
| ----------------------------------------------------------- | ------- | ----------- | ---------- | ---------- | ----------------- |
| - [experiments_root](#path_experiments_root )               | No      | Combination | No         | -          | -                 |
| - [models](#path_models )                                   | No      | Combination | No         | -          | -                 |
| - [resume_models](#path_resume_models )                     | No      | Combination | No         | -          | -                 |
| - [training_states](#path_training_states )                 | No      | Combination | No         | -          | -                 |
| - [log](#path_log )                                         | No      | Combination | No         | -          | -                 |
| - [visualization](#path_visualization )                     | No      | Combination | No         | -          | -                 |
| - [results_root](#path_results_root )                       | No      | Combination | No         | -          | -                 |
| - [pretrain_network_g](#path_pretrain_network_g )           | No      | Combination | No         | -          | -                 |
| - [pretrain_network_g_path](#path_pretrain_network_g_path ) | No      | Combination | No         | -          | -                 |
| - [param_key_g](#path_param_key_g )                         | No      | Combination | No         | -          | -                 |
| - [strict_load_g](#path_strict_load_g )                     | No      | boolean     | No         | -          | -                 |
| - [resume_state](#path_resume_state )                       | No      | Combination | No         | -          | -                 |
| - [pretrain_network_g_ema](#path_pretrain_network_g_ema )   | No      | Combination | No         | -          | -                 |
| - [pretrain_network_d](#path_pretrain_network_d )           | No      | Combination | No         | -          | -                 |
| - [param_key_d](#path_param_key_d )                         | No      | Combination | No         | -          | -                 |
| - [strict_load_d](#path_strict_load_d )                     | No      | boolean     | No         | -          | -                 |
| - [ignore_resume_networks](#path_ignore_resume_networks )   | No      | Combination | No         | -          | -                 |

### <a name="path_experiments_root"></a>4.1. Property `ReduxOptions > path > experiments_root`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `combining`      |
| **Required**              | No               |
| **Additional properties** | Any type allowed |
| **Default**               | `null`           |

| Any of(Option)                            |
| ----------------------------------------- |
| [item 0](#path_experiments_root_anyOf_i0) |
| [item 1](#path_experiments_root_anyOf_i1) |

#### <a name="path_experiments_root_anyOf_i0"></a>4.1.1. Property `ReduxOptions > path > experiments_root > anyOf > item 0`

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | No       |

#### <a name="path_experiments_root_anyOf_i1"></a>4.1.2. Property `ReduxOptions > path > experiments_root > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

### <a name="path_models"></a>4.2. Property `ReduxOptions > path > models`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `combining`      |
| **Required**              | No               |
| **Additional properties** | Any type allowed |
| **Default**               | `null`           |

| Any of(Option)                  |
| ------------------------------- |
| [item 0](#path_models_anyOf_i0) |
| [item 1](#path_models_anyOf_i1) |

#### <a name="path_models_anyOf_i0"></a>4.2.1. Property `ReduxOptions > path > models > anyOf > item 0`

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | No       |

#### <a name="path_models_anyOf_i1"></a>4.2.2. Property `ReduxOptions > path > models > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

### <a name="path_resume_models"></a>4.3. Property `ReduxOptions > path > resume_models`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `combining`      |
| **Required**              | No               |
| **Additional properties** | Any type allowed |
| **Default**               | `null`           |

| Any of(Option)                         |
| -------------------------------------- |
| [item 0](#path_resume_models_anyOf_i0) |
| [item 1](#path_resume_models_anyOf_i1) |

#### <a name="path_resume_models_anyOf_i0"></a>4.3.1. Property `ReduxOptions > path > resume_models > anyOf > item 0`

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | No       |

#### <a name="path_resume_models_anyOf_i1"></a>4.3.2. Property `ReduxOptions > path > resume_models > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

### <a name="path_training_states"></a>4.4. Property `ReduxOptions > path > training_states`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `combining`      |
| **Required**              | No               |
| **Additional properties** | Any type allowed |
| **Default**               | `null`           |

| Any of(Option)                           |
| ---------------------------------------- |
| [item 0](#path_training_states_anyOf_i0) |
| [item 1](#path_training_states_anyOf_i1) |

#### <a name="path_training_states_anyOf_i0"></a>4.4.1. Property `ReduxOptions > path > training_states > anyOf > item 0`

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | No       |

#### <a name="path_training_states_anyOf_i1"></a>4.4.2. Property `ReduxOptions > path > training_states > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

### <a name="path_log"></a>4.5. Property `ReduxOptions > path > log`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `combining`      |
| **Required**              | No               |
| **Additional properties** | Any type allowed |
| **Default**               | `null`           |

| Any of(Option)               |
| ---------------------------- |
| [item 0](#path_log_anyOf_i0) |
| [item 1](#path_log_anyOf_i1) |

#### <a name="path_log_anyOf_i0"></a>4.5.1. Property `ReduxOptions > path > log > anyOf > item 0`

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | No       |

#### <a name="path_log_anyOf_i1"></a>4.5.2. Property `ReduxOptions > path > log > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

### <a name="path_visualization"></a>4.6. Property `ReduxOptions > path > visualization`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `combining`      |
| **Required**              | No               |
| **Additional properties** | Any type allowed |
| **Default**               | `null`           |

| Any of(Option)                         |
| -------------------------------------- |
| [item 0](#path_visualization_anyOf_i0) |
| [item 1](#path_visualization_anyOf_i1) |

#### <a name="path_visualization_anyOf_i0"></a>4.6.1. Property `ReduxOptions > path > visualization > anyOf > item 0`

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | No       |

#### <a name="path_visualization_anyOf_i1"></a>4.6.2. Property `ReduxOptions > path > visualization > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

### <a name="path_results_root"></a>4.7. Property `ReduxOptions > path > results_root`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `combining`      |
| **Required**              | No               |
| **Additional properties** | Any type allowed |
| **Default**               | `null`           |

| Any of(Option)                        |
| ------------------------------------- |
| [item 0](#path_results_root_anyOf_i0) |
| [item 1](#path_results_root_anyOf_i1) |

#### <a name="path_results_root_anyOf_i0"></a>4.7.1. Property `ReduxOptions > path > results_root > anyOf > item 0`

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | No       |

#### <a name="path_results_root_anyOf_i1"></a>4.7.2. Property `ReduxOptions > path > results_root > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

### <a name="path_pretrain_network_g"></a>4.8. Property `ReduxOptions > path > pretrain_network_g`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `combining`      |
| **Required**              | No               |
| **Additional properties** | Any type allowed |
| **Default**               | `null`           |

| Any of(Option)                              |
| ------------------------------------------- |
| [item 0](#path_pretrain_network_g_anyOf_i0) |
| [item 1](#path_pretrain_network_g_anyOf_i1) |

#### <a name="path_pretrain_network_g_anyOf_i0"></a>4.8.1. Property `ReduxOptions > path > pretrain_network_g > anyOf > item 0`

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | No       |

#### <a name="path_pretrain_network_g_anyOf_i1"></a>4.8.2. Property `ReduxOptions > path > pretrain_network_g > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

### <a name="path_pretrain_network_g_path"></a>4.9. Property `ReduxOptions > path > pretrain_network_g_path`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `combining`      |
| **Required**              | No               |
| **Additional properties** | Any type allowed |
| **Default**               | `null`           |

| Any of(Option)                                   |
| ------------------------------------------------ |
| [item 0](#path_pretrain_network_g_path_anyOf_i0) |
| [item 1](#path_pretrain_network_g_path_anyOf_i1) |

#### <a name="path_pretrain_network_g_path_anyOf_i0"></a>4.9.1. Property `ReduxOptions > path > pretrain_network_g_path > anyOf > item 0`

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | No       |

#### <a name="path_pretrain_network_g_path_anyOf_i1"></a>4.9.2. Property `ReduxOptions > path > pretrain_network_g_path > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

### <a name="path_param_key_g"></a>4.10. Property `ReduxOptions > path > param_key_g`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `combining`      |
| **Required**              | No               |
| **Additional properties** | Any type allowed |
| **Default**               | `null`           |

| Any of(Option)                       |
| ------------------------------------ |
| [item 0](#path_param_key_g_anyOf_i0) |
| [item 1](#path_param_key_g_anyOf_i1) |

#### <a name="path_param_key_g_anyOf_i0"></a>4.10.1. Property `ReduxOptions > path > param_key_g > anyOf > item 0`

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | No       |

#### <a name="path_param_key_g_anyOf_i1"></a>4.10.2. Property `ReduxOptions > path > param_key_g > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

### <a name="path_strict_load_g"></a>4.11. Property `ReduxOptions > path > strict_load_g`

|              |           |
| ------------ | --------- |
| **Type**     | `boolean` |
| **Required** | No        |
| **Default**  | `true`    |

### <a name="path_resume_state"></a>4.12. Property `ReduxOptions > path > resume_state`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `combining`      |
| **Required**              | No               |
| **Additional properties** | Any type allowed |
| **Default**               | `null`           |

| Any of(Option)                        |
| ------------------------------------- |
| [item 0](#path_resume_state_anyOf_i0) |
| [item 1](#path_resume_state_anyOf_i1) |

#### <a name="path_resume_state_anyOf_i0"></a>4.12.1. Property `ReduxOptions > path > resume_state > anyOf > item 0`

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | No       |

#### <a name="path_resume_state_anyOf_i1"></a>4.12.2. Property `ReduxOptions > path > resume_state > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

### <a name="path_pretrain_network_g_ema"></a>4.13. Property `ReduxOptions > path > pretrain_network_g_ema`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `combining`      |
| **Required**              | No               |
| **Additional properties** | Any type allowed |
| **Default**               | `null`           |

| Any of(Option)                                  |
| ----------------------------------------------- |
| [item 0](#path_pretrain_network_g_ema_anyOf_i0) |
| [item 1](#path_pretrain_network_g_ema_anyOf_i1) |

#### <a name="path_pretrain_network_g_ema_anyOf_i0"></a>4.13.1. Property `ReduxOptions > path > pretrain_network_g_ema > anyOf > item 0`

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | No       |

#### <a name="path_pretrain_network_g_ema_anyOf_i1"></a>4.13.2. Property `ReduxOptions > path > pretrain_network_g_ema > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

### <a name="path_pretrain_network_d"></a>4.14. Property `ReduxOptions > path > pretrain_network_d`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `combining`      |
| **Required**              | No               |
| **Additional properties** | Any type allowed |
| **Default**               | `null`           |

| Any of(Option)                              |
| ------------------------------------------- |
| [item 0](#path_pretrain_network_d_anyOf_i0) |
| [item 1](#path_pretrain_network_d_anyOf_i1) |

#### <a name="path_pretrain_network_d_anyOf_i0"></a>4.14.1. Property `ReduxOptions > path > pretrain_network_d > anyOf > item 0`

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | No       |

#### <a name="path_pretrain_network_d_anyOf_i1"></a>4.14.2. Property `ReduxOptions > path > pretrain_network_d > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

### <a name="path_param_key_d"></a>4.15. Property `ReduxOptions > path > param_key_d`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `combining`      |
| **Required**              | No               |
| **Additional properties** | Any type allowed |
| **Default**               | `null`           |

| Any of(Option)                       |
| ------------------------------------ |
| [item 0](#path_param_key_d_anyOf_i0) |
| [item 1](#path_param_key_d_anyOf_i1) |

#### <a name="path_param_key_d_anyOf_i0"></a>4.15.1. Property `ReduxOptions > path > param_key_d > anyOf > item 0`

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | No       |

#### <a name="path_param_key_d_anyOf_i1"></a>4.15.2. Property `ReduxOptions > path > param_key_d > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

### <a name="path_strict_load_d"></a>4.16. Property `ReduxOptions > path > strict_load_d`

|              |           |
| ------------ | --------- |
| **Type**     | `boolean` |
| **Required** | No        |
| **Default**  | `true`    |

### <a name="path_ignore_resume_networks"></a>4.17. Property `ReduxOptions > path > ignore_resume_networks`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `combining`      |
| **Required**              | No               |
| **Additional properties** | Any type allowed |
| **Default**               | `null`           |

| Any of(Option)                                  |
| ----------------------------------------------- |
| [item 0](#path_ignore_resume_networks_anyOf_i0) |
| [item 1](#path_ignore_resume_networks_anyOf_i1) |

#### <a name="path_ignore_resume_networks_anyOf_i0"></a>4.17.1. Property `ReduxOptions > path > ignore_resume_networks > anyOf > item 0`

|              |                   |
| ------------ | ----------------- |
| **Type**     | `array of string` |
| **Required** | No                |

|                      | Array restrictions |
| -------------------- | ------------------ |
| **Min items**        | N/A                |
| **Max items**        | N/A                |
| **Items unicity**    | False              |
| **Additional items** | False              |
| **Tuple validation** | See below          |

| Each item of this array must be                             | Description |
| ----------------------------------------------------------- | ----------- |
| [item 0 items](#path_ignore_resume_networks_anyOf_i0_items) | -           |

##### <a name="path_ignore_resume_networks_anyOf_i0_items"></a>4.17.1.1. ReduxOptions > path > ignore_resume_networks > anyOf > item 0 > item 0 items

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | No       |

#### <a name="path_ignore_resume_networks_anyOf_i1"></a>4.17.2. Property `ReduxOptions > path > ignore_resume_networks > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

## <a name="network_g"></a>5. Property `ReduxOptions > network_g`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `combining`      |
| **Required**              | Yes              |
| **Additional properties** | Any type allowed |

| Any of(Option)                                      |
| --------------------------------------------------- |
| [artcnn](#network_g_anyOf_i0)                       |
| [artcnn_r16f96](#network_g_anyOf_i1)                |
| [artcnn_r8f64](#network_g_anyOf_i2)                 |
| [camixersr](#network_g_anyOf_i3)                    |
| [cfsr](#network_g_anyOf_i4)                         |
| [vggstylediscriminator](#network_g_anyOf_i5)        |
| [unetdiscriminatorsn_traiNNer](#network_g_anyOf_i6) |
| [dunet](#network_g_anyOf_i7)                        |
| [eimn](#network_g_anyOf_i8)                         |
| [eimn_l](#network_g_anyOf_i9)                       |
| [eimn_a](#network_g_anyOf_i10)                      |
| [flexnet](#network_g_anyOf_i11)                     |
| [metaflexnet](#network_g_anyOf_i12)                 |
| [hit_sir](#network_g_anyOf_i13)                     |
| [hit_sng](#network_g_anyOf_i14)                     |
| [hit_srf](#network_g_anyOf_i15)                     |
| [ipt](#network_g_anyOf_i16)                         |
| [lmlt](#network_g_anyOf_i17)                        |
| [lmlt_base](#network_g_anyOf_i18)                   |
| [lmlt_large](#network_g_anyOf_i19)                  |
| [lmlt_tiny](#network_g_anyOf_i20)                   |
| [man](#network_g_anyOf_i21)                         |
| [man_tiny](#network_g_anyOf_i22)                    |
| [man_light](#network_g_anyOf_i23)                   |
| [moesr2](#network_g_anyOf_i24)                      |
| [mosr](#network_g_anyOf_i25)                        |
| [mosr_t](#network_g_anyOf_i26)                      |
| [rcancab](#network_g_anyOf_i27)                     |
| [rcan](#network_g_anyOf_i28)                        |
| [realplksr](#network_g_anyOf_i29)                   |
| [scunet_aaf6aa](#network_g_anyOf_i30)               |
| [spanplus](#network_g_anyOf_i31)                    |
| [spanplus_sts](#network_g_anyOf_i32)                |
| [spanplus_s](#network_g_anyOf_i33)                  |
| [spanplus_st](#network_g_anyOf_i34)                 |
| [compact](#network_g_anyOf_i35)                     |
| [ultracompact](#network_g_anyOf_i36)                |
| [superultracompact](#network_g_anyOf_i37)           |
| [tscunet](#network_g_anyOf_i38)                     |
| [atd](#network_g_anyOf_i39)                         |
| [atd_light](#network_g_anyOf_i40)                   |
| [dat](#network_g_anyOf_i41)                         |
| [dat_s](#network_g_anyOf_i42)                       |
| [dat_2](#network_g_anyOf_i43)                       |
| [dat_light](#network_g_anyOf_i44)                   |
| [drct](#network_g_anyOf_i45)                        |
| [drct_l](#network_g_anyOf_i46)                      |
| [drct_xl](#network_g_anyOf_i47)                     |
| [hat](#network_g_anyOf_i48)                         |
| [hat_l](#network_g_anyOf_i49)                       |
| [hat_m](#network_g_anyOf_i50)                       |
| [hat_s](#network_g_anyOf_i51)                       |
| [omnisr](#network_g_anyOf_i52)                      |
| [plksr](#network_g_anyOf_i53)                       |
| [plksr_tiny](#network_g_anyOf_i54)                  |
| [realcugan](#network_g_anyOf_i55)                   |
| [rgt](#network_g_anyOf_i56)                         |
| [rgt_s](#network_g_anyOf_i57)                       |
| [esrgan](#network_g_anyOf_i58)                      |
| [esrgan_lite](#network_g_anyOf_i59)                 |
| [span](#network_g_anyOf_i60)                        |
| [srformer](#network_g_anyOf_i61)                    |
| [srformer_light](#network_g_anyOf_i62)              |
| [swin2sr](#network_g_anyOf_i63)                     |
| [swin2sr_l](#network_g_anyOf_i64)                   |
| [swin2sr_m](#network_g_anyOf_i65)                   |
| [swin2sr_s](#network_g_anyOf_i66)                   |
| [swinir](#network_g_anyOf_i67)                      |
| [swinir_l](#network_g_anyOf_i68)                    |
| [swinir_m](#network_g_anyOf_i69)                    |
| [swinir_s](#network_g_anyOf_i70)                    |

### <a name="network_g_anyOf_i0"></a>5.1. Property `ReduxOptions > network_g > anyOf > artcnn`

**Title:** artcnn

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `object`         |
| **Required**              | No               |
| **Additional properties** | Any type allowed |
| **Defined in**            | #/$defs/artcnn   |

| Property                                          | Pattern | Type             | Deprecated | Definition | Title/Description |
| ------------------------------------------------- | ------- | ---------------- | ---------- | ---------- | ----------------- |
| + [type](#network_g_anyOf_i0_type )               | No      | enum (of string) | No         | -          | -                 |
| - [in_ch](#network_g_anyOf_i0_in_ch )             | No      | integer          | No         | -          | -                 |
| - [scale](#network_g_anyOf_i0_scale )             | No      | integer          | No         | -          | -                 |
| - [filters](#network_g_anyOf_i0_filters )         | No      | integer          | No         | -          | -                 |
| - [n_block](#network_g_anyOf_i0_n_block )         | No      | integer          | No         | -          | -                 |
| - [kernel_size](#network_g_anyOf_i0_kernel_size ) | No      | integer          | No         | -          | -                 |

#### <a name="network_g_anyOf_i0_type"></a>5.1.1. Property `ReduxOptions > network_g > anyOf > artcnn > type`

|              |                    |
| ------------ | ------------------ |
| **Type**     | `enum (of string)` |
| **Required** | Yes                |

Must be one of:
* "artcnn"

#### <a name="network_g_anyOf_i0_in_ch"></a>5.1.2. Property `ReduxOptions > network_g > anyOf > artcnn > in_ch`

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | No        |
| **Default**  | `3`       |

#### <a name="network_g_anyOf_i0_scale"></a>5.1.3. Property `ReduxOptions > network_g > anyOf > artcnn > scale`

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | No        |
| **Default**  | `4`       |

#### <a name="network_g_anyOf_i0_filters"></a>5.1.4. Property `ReduxOptions > network_g > anyOf > artcnn > filters`

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | No        |
| **Default**  | `96`      |

#### <a name="network_g_anyOf_i0_n_block"></a>5.1.5. Property `ReduxOptions > network_g > anyOf > artcnn > n_block`

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | No        |
| **Default**  | `16`      |

#### <a name="network_g_anyOf_i0_kernel_size"></a>5.1.6. Property `ReduxOptions > network_g > anyOf > artcnn > kernel_size`

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | No        |
| **Default**  | `3`       |

### <a name="network_g_anyOf_i1"></a>5.2. Property `ReduxOptions > network_g > anyOf > artcnn_r16f96`

**Title:** artcnn_r16f96

|                           |                       |
| ------------------------- | --------------------- |
| **Type**                  | `object`              |
| **Required**              | No                    |
| **Additional properties** | Any type allowed      |
| **Defined in**            | #/$defs/artcnn_r16f96 |

| Property                            | Pattern | Type             | Deprecated | Definition | Title/Description |
| ----------------------------------- | ------- | ---------------- | ---------- | ---------- | ----------------- |
| + [type](#network_g_anyOf_i1_type ) | No      | enum (of string) | No         | -          | -                 |

#### <a name="network_g_anyOf_i1_type"></a>5.2.1. Property `ReduxOptions > network_g > anyOf > artcnn_r16f96 > type`

|              |                    |
| ------------ | ------------------ |
| **Type**     | `enum (of string)` |
| **Required** | Yes                |

Must be one of:
* "artcnn_r16f96"

### <a name="network_g_anyOf_i2"></a>5.3. Property `ReduxOptions > network_g > anyOf > artcnn_r8f64`

**Title:** artcnn_r8f64

|                           |                      |
| ------------------------- | -------------------- |
| **Type**                  | `object`             |
| **Required**              | No                   |
| **Additional properties** | Any type allowed     |
| **Defined in**            | #/$defs/artcnn_r8f64 |

| Property                            | Pattern | Type             | Deprecated | Definition | Title/Description |
| ----------------------------------- | ------- | ---------------- | ---------- | ---------- | ----------------- |
| + [type](#network_g_anyOf_i2_type ) | No      | enum (of string) | No         | -          | -                 |

#### <a name="network_g_anyOf_i2_type"></a>5.3.1. Property `ReduxOptions > network_g > anyOf > artcnn_r8f64 > type`

|              |                    |
| ------------ | ------------------ |
| **Type**     | `enum (of string)` |
| **Required** | Yes                |

Must be one of:
* "artcnn_r8f64"

### <a name="network_g_anyOf_i3"></a>5.4. Property `ReduxOptions > network_g > anyOf > camixersr`

**Title:** camixersr

|                           |                   |
| ------------------------- | ----------------- |
| **Type**                  | `object`          |
| **Required**              | No                |
| **Additional properties** | Any type allowed  |
| **Defined in**            | #/$defs/camixersr |

| Property                            | Pattern | Type             | Deprecated | Definition | Title/Description |
| ----------------------------------- | ------- | ---------------- | ---------- | ---------- | ----------------- |
| + [type](#network_g_anyOf_i3_type ) | No      | enum (of string) | No         | -          | -                 |

#### <a name="network_g_anyOf_i3_type"></a>5.4.1. Property `ReduxOptions > network_g > anyOf > camixersr > type`

|              |                    |
| ------------ | ------------------ |
| **Type**     | `enum (of string)` |
| **Required** | Yes                |

Must be one of:
* "camixersr"

### <a name="network_g_anyOf_i4"></a>5.5. Property `ReduxOptions > network_g > anyOf > cfsr`

**Title:** cfsr

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `object`         |
| **Required**              | No               |
| **Additional properties** | Any type allowed |
| **Defined in**            | #/$defs/cfsr     |

| Property                            | Pattern | Type             | Deprecated | Definition | Title/Description |
| ----------------------------------- | ------- | ---------------- | ---------- | ---------- | ----------------- |
| + [type](#network_g_anyOf_i4_type ) | No      | enum (of string) | No         | -          | -                 |

#### <a name="network_g_anyOf_i4_type"></a>5.5.1. Property `ReduxOptions > network_g > anyOf > cfsr > type`

|              |                    |
| ------------ | ------------------ |
| **Type**     | `enum (of string)` |
| **Required** | Yes                |

Must be one of:
* "cfsr"

### <a name="network_g_anyOf_i5"></a>5.6. Property `ReduxOptions > network_g > anyOf > vggstylediscriminator`

**Title:** vggstylediscriminator

|                           |                               |
| ------------------------- | ----------------------------- |
| **Type**                  | `object`                      |
| **Required**              | No                            |
| **Additional properties** | Any type allowed              |
| **Defined in**            | #/$defs/vggstylediscriminator |

| Property                                        | Pattern | Type             | Deprecated | Definition | Title/Description |
| ----------------------------------------------- | ------- | ---------------- | ---------- | ---------- | ----------------- |
| + [type](#network_g_anyOf_i5_type )             | No      | enum (of string) | No         | -          | -                 |
| + [num_in_ch](#network_g_anyOf_i5_num_in_ch )   | No      | integer          | No         | -          | -                 |
| + [num_feat](#network_g_anyOf_i5_num_feat )     | No      | integer          | No         | -          | -                 |
| - [input_size](#network_g_anyOf_i5_input_size ) | No      | integer          | No         | -          | -                 |

#### <a name="network_g_anyOf_i5_type"></a>5.6.1. Property `ReduxOptions > network_g > anyOf > vggstylediscriminator > type`

|              |                    |
| ------------ | ------------------ |
| **Type**     | `enum (of string)` |
| **Required** | Yes                |

Must be one of:
* "vggstylediscriminator"

#### <a name="network_g_anyOf_i5_num_in_ch"></a>5.6.2. Property `ReduxOptions > network_g > anyOf > vggstylediscriminator > num_in_ch`

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | Yes       |

#### <a name="network_g_anyOf_i5_num_feat"></a>5.6.3. Property `ReduxOptions > network_g > anyOf > vggstylediscriminator > num_feat`

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | Yes       |

#### <a name="network_g_anyOf_i5_input_size"></a>5.6.4. Property `ReduxOptions > network_g > anyOf > vggstylediscriminator > input_size`

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | No        |
| **Default**  | `128`     |

### <a name="network_g_anyOf_i6"></a>5.7. Property `ReduxOptions > network_g > anyOf > unetdiscriminatorsn_traiNNer`

**Title:** unetdiscriminatorsn_traiNNer

|                           |                                      |
| ------------------------- | ------------------------------------ |
| **Type**                  | `object`                             |
| **Required**              | No                                   |
| **Additional properties** | Any type allowed                     |
| **Defined in**            | #/$defs/unetdiscriminatorsn_traiNNer |

| Property                                                  | Pattern | Type             | Deprecated | Definition | Title/Description |
| --------------------------------------------------------- | ------- | ---------------- | ---------- | ---------- | ----------------- |
| + [type](#network_g_anyOf_i6_type )                       | No      | enum (of string) | No         | -          | -                 |
| + [num_in_ch](#network_g_anyOf_i6_num_in_ch )             | No      | integer          | No         | -          | -                 |
| - [num_feat](#network_g_anyOf_i6_num_feat )               | No      | integer          | No         | -          | -                 |
| - [skip_connection](#network_g_anyOf_i6_skip_connection ) | No      | boolean          | No         | -          | -                 |

#### <a name="network_g_anyOf_i6_type"></a>5.7.1. Property `ReduxOptions > network_g > anyOf > unetdiscriminatorsn_traiNNer > type`

|              |                    |
| ------------ | ------------------ |
| **Type**     | `enum (of string)` |
| **Required** | Yes                |

Must be one of:
* "unetdiscriminatorsn_traiNNer"

#### <a name="network_g_anyOf_i6_num_in_ch"></a>5.7.2. Property `ReduxOptions > network_g > anyOf > unetdiscriminatorsn_traiNNer > num_in_ch`

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | Yes       |

#### <a name="network_g_anyOf_i6_num_feat"></a>5.7.3. Property `ReduxOptions > network_g > anyOf > unetdiscriminatorsn_traiNNer > num_feat`

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | No        |
| **Default**  | `64`      |

#### <a name="network_g_anyOf_i6_skip_connection"></a>5.7.4. Property `ReduxOptions > network_g > anyOf > unetdiscriminatorsn_traiNNer > skip_connection`

|              |           |
| ------------ | --------- |
| **Type**     | `boolean` |
| **Required** | No        |
| **Default**  | `true`    |

### <a name="network_g_anyOf_i7"></a>5.8. Property `ReduxOptions > network_g > anyOf > dunet`

**Title:** dunet

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `object`         |
| **Required**              | No               |
| **Additional properties** | Any type allowed |
| **Defined in**            | #/$defs/dunet    |

| Property                                      | Pattern | Type             | Deprecated | Definition | Title/Description |
| --------------------------------------------- | ------- | ---------------- | ---------- | ---------- | ----------------- |
| + [type](#network_g_anyOf_i7_type )           | No      | enum (of string) | No         | -          | -                 |
| - [num_in_ch](#network_g_anyOf_i7_num_in_ch ) | No      | integer          | No         | -          | -                 |
| - [num_feat](#network_g_anyOf_i7_num_feat )   | No      | integer          | No         | -          | -                 |

#### <a name="network_g_anyOf_i7_type"></a>5.8.1. Property `ReduxOptions > network_g > anyOf > dunet > type`

|              |                    |
| ------------ | ------------------ |
| **Type**     | `enum (of string)` |
| **Required** | Yes                |

Must be one of:
* "dunet"

#### <a name="network_g_anyOf_i7_num_in_ch"></a>5.8.2. Property `ReduxOptions > network_g > anyOf > dunet > num_in_ch`

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | No        |
| **Default**  | `3`       |

#### <a name="network_g_anyOf_i7_num_feat"></a>5.8.3. Property `ReduxOptions > network_g > anyOf > dunet > num_feat`

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | No        |
| **Default**  | `64`      |

### <a name="network_g_anyOf_i8"></a>5.9. Property `ReduxOptions > network_g > anyOf > eimn`

**Title:** eimn

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `object`         |
| **Required**              | No               |
| **Additional properties** | Any type allowed |
| **Defined in**            | #/$defs/eimn     |

| Property                                                | Pattern | Type             | Deprecated | Definition | Title/Description |
| ------------------------------------------------------- | ------- | ---------------- | ---------- | ---------- | ----------------- |
| + [type](#network_g_anyOf_i8_type )                     | No      | enum (of string) | No         | -          | -                 |
| - [embed_dims](#network_g_anyOf_i8_embed_dims )         | No      | integer          | No         | -          | -                 |
| - [scale](#network_g_anyOf_i8_scale )                   | No      | integer          | No         | -          | -                 |
| - [depths](#network_g_anyOf_i8_depths )                 | No      | integer          | No         | -          | -                 |
| - [mlp_ratios](#network_g_anyOf_i8_mlp_ratios )         | No      | number           | No         | -          | -                 |
| - [drop_rate](#network_g_anyOf_i8_drop_rate )           | No      | number           | No         | -          | -                 |
| - [drop_path_rate](#network_g_anyOf_i8_drop_path_rate ) | No      | number           | No         | -          | -                 |
| - [num_stages](#network_g_anyOf_i8_num_stages )         | No      | integer          | No         | -          | -                 |
| - [freeze_param](#network_g_anyOf_i8_freeze_param )     | No      | boolean          | No         | -          | -                 |

#### <a name="network_g_anyOf_i8_type"></a>5.9.1. Property `ReduxOptions > network_g > anyOf > eimn > type`

|              |                    |
| ------------ | ------------------ |
| **Type**     | `enum (of string)` |
| **Required** | Yes                |

Must be one of:
* "eimn"

#### <a name="network_g_anyOf_i8_embed_dims"></a>5.9.2. Property `ReduxOptions > network_g > anyOf > eimn > embed_dims`

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | No        |
| **Default**  | `64`      |

#### <a name="network_g_anyOf_i8_scale"></a>5.9.3. Property `ReduxOptions > network_g > anyOf > eimn > scale`

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | No        |
| **Default**  | `2`       |

#### <a name="network_g_anyOf_i8_depths"></a>5.9.4. Property `ReduxOptions > network_g > anyOf > eimn > depths`

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | No        |
| **Default**  | `1`       |

#### <a name="network_g_anyOf_i8_mlp_ratios"></a>5.9.5. Property `ReduxOptions > network_g > anyOf > eimn > mlp_ratios`

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |
| **Default**  | `2.66`   |

#### <a name="network_g_anyOf_i8_drop_rate"></a>5.9.6. Property `ReduxOptions > network_g > anyOf > eimn > drop_rate`

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |
| **Default**  | `0.0`    |

#### <a name="network_g_anyOf_i8_drop_path_rate"></a>5.9.7. Property `ReduxOptions > network_g > anyOf > eimn > drop_path_rate`

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |
| **Default**  | `0.0`    |

#### <a name="network_g_anyOf_i8_num_stages"></a>5.9.8. Property `ReduxOptions > network_g > anyOf > eimn > num_stages`

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | No        |
| **Default**  | `16`      |

#### <a name="network_g_anyOf_i8_freeze_param"></a>5.9.9. Property `ReduxOptions > network_g > anyOf > eimn > freeze_param`

|              |           |
| ------------ | --------- |
| **Type**     | `boolean` |
| **Required** | No        |
| **Default**  | `false`   |

### <a name="network_g_anyOf_i9"></a>5.10. Property `ReduxOptions > network_g > anyOf > eimn_l`

**Title:** eimn_l

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `object`         |
| **Required**              | No               |
| **Additional properties** | Any type allowed |
| **Defined in**            | #/$defs/eimn_l   |

| Property                            | Pattern | Type             | Deprecated | Definition | Title/Description |
| ----------------------------------- | ------- | ---------------- | ---------- | ---------- | ----------------- |
| + [type](#network_g_anyOf_i9_type ) | No      | enum (of string) | No         | -          | -                 |

#### <a name="network_g_anyOf_i9_type"></a>5.10.1. Property `ReduxOptions > network_g > anyOf > eimn_l > type`

|              |                    |
| ------------ | ------------------ |
| **Type**     | `enum (of string)` |
| **Required** | Yes                |

Must be one of:
* "eimn_l"

### <a name="network_g_anyOf_i10"></a>5.11. Property `ReduxOptions > network_g > anyOf > eimn_a`

**Title:** eimn_a

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `object`         |
| **Required**              | No               |
| **Additional properties** | Any type allowed |
| **Defined in**            | #/$defs/eimn_a   |

| Property                             | Pattern | Type             | Deprecated | Definition | Title/Description |
| ------------------------------------ | ------- | ---------------- | ---------- | ---------- | ----------------- |
| + [type](#network_g_anyOf_i10_type ) | No      | enum (of string) | No         | -          | -                 |

#### <a name="network_g_anyOf_i10_type"></a>5.11.1. Property `ReduxOptions > network_g > anyOf > eimn_a > type`

|              |                    |
| ------------ | ------------------ |
| **Type**     | `enum (of string)` |
| **Required** | Yes                |

Must be one of:
* "eimn_a"

### <a name="network_g_anyOf_i11"></a>5.12. Property `ReduxOptions > network_g > anyOf > flexnet`

**Title:** flexnet

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `object`         |
| **Required**              | No               |
| **Additional properties** | Any type allowed |
| **Defined in**            | #/$defs/flexnet  |

| Property                                               | Pattern | Type             | Deprecated | Definition | Title/Description |
| ------------------------------------------------------ | ------- | ---------------- | ---------- | ---------- | ----------------- |
| + [type](#network_g_anyOf_i11_type )                   | No      | enum (of string) | No         | -          | -                 |
| - [inp_channels](#network_g_anyOf_i11_inp_channels )   | No      | integer          | No         | -          | -                 |
| - [out_channels](#network_g_anyOf_i11_out_channels )   | No      | integer          | No         | -          | -                 |
| - [scale](#network_g_anyOf_i11_scale )                 | No      | integer          | No         | -          | -                 |
| - [dim](#network_g_anyOf_i11_dim )                     | No      | integer          | No         | -          | -                 |
| - [num_blocks](#network_g_anyOf_i11_num_blocks )       | No      | array of integer | No         | -          | -                 |
| - [window_size](#network_g_anyOf_i11_window_size )     | No      | integer          | No         | -          | -                 |
| - [hidden_rate](#network_g_anyOf_i11_hidden_rate )     | No      | integer          | No         | -          | -                 |
| - [channel_norm](#network_g_anyOf_i11_channel_norm )   | No      | boolean          | No         | -          | -                 |
| - [attn_drop](#network_g_anyOf_i11_attn_drop )         | No      | number           | No         | -          | -                 |
| - [proj_drop](#network_g_anyOf_i11_proj_drop )         | No      | number           | No         | -          | -                 |
| - [pipeline_type](#network_g_anyOf_i11_pipeline_type ) | No      | enum (of string) | No         | -          | -                 |
| - [upsampler](#network_g_anyOf_i11_upsampler )         | No      | enum (of string) | No         | -          | -                 |

#### <a name="network_g_anyOf_i11_type"></a>5.12.1. Property `ReduxOptions > network_g > anyOf > flexnet > type`

|              |                    |
| ------------ | ------------------ |
| **Type**     | `enum (of string)` |
| **Required** | Yes                |

Must be one of:
* "flexnet"

#### <a name="network_g_anyOf_i11_inp_channels"></a>5.12.2. Property `ReduxOptions > network_g > anyOf > flexnet > inp_channels`

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | No        |
| **Default**  | `3`       |

#### <a name="network_g_anyOf_i11_out_channels"></a>5.12.3. Property `ReduxOptions > network_g > anyOf > flexnet > out_channels`

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | No        |
| **Default**  | `3`       |

#### <a name="network_g_anyOf_i11_scale"></a>5.12.4. Property `ReduxOptions > network_g > anyOf > flexnet > scale`

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | No        |
| **Default**  | `4`       |

#### <a name="network_g_anyOf_i11_dim"></a>5.12.5. Property `ReduxOptions > network_g > anyOf > flexnet > dim`

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | No        |
| **Default**  | `64`      |

#### <a name="network_g_anyOf_i11_num_blocks"></a>5.12.6. Property `ReduxOptions > network_g > anyOf > flexnet > num_blocks`

|              |                      |
| ------------ | -------------------- |
| **Type**     | `array of integer`   |
| **Required** | No                   |
| **Default**  | `[6, 6, 6, 6, 6, 6]` |

|                      | Array restrictions |
| -------------------- | ------------------ |
| **Min items**        | N/A                |
| **Max items**        | N/A                |
| **Items unicity**    | False              |
| **Additional items** | False              |
| **Tuple validation** | See below          |

| Each item of this array must be                           | Description |
| --------------------------------------------------------- | ----------- |
| [num_blocks items](#network_g_anyOf_i11_num_blocks_items) | -           |

##### <a name="network_g_anyOf_i11_num_blocks_items"></a>5.12.6.1. ReduxOptions > network_g > anyOf > flexnet > num_blocks > num_blocks items

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | No        |

#### <a name="network_g_anyOf_i11_window_size"></a>5.12.7. Property `ReduxOptions > network_g > anyOf > flexnet > window_size`

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | No        |
| **Default**  | `8`       |

#### <a name="network_g_anyOf_i11_hidden_rate"></a>5.12.8. Property `ReduxOptions > network_g > anyOf > flexnet > hidden_rate`

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | No        |
| **Default**  | `4`       |

#### <a name="network_g_anyOf_i11_channel_norm"></a>5.12.9. Property `ReduxOptions > network_g > anyOf > flexnet > channel_norm`

|              |           |
| ------------ | --------- |
| **Type**     | `boolean` |
| **Required** | No        |
| **Default**  | `false`   |

#### <a name="network_g_anyOf_i11_attn_drop"></a>5.12.10. Property `ReduxOptions > network_g > anyOf > flexnet > attn_drop`

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |
| **Default**  | `0.0`    |

#### <a name="network_g_anyOf_i11_proj_drop"></a>5.12.11. Property `ReduxOptions > network_g > anyOf > flexnet > proj_drop`

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |
| **Default**  | `0.0`    |

#### <a name="network_g_anyOf_i11_pipeline_type"></a>5.12.12. Property `ReduxOptions > network_g > anyOf > flexnet > pipeline_type`

|              |                    |
| ------------ | ------------------ |
| **Type**     | `enum (of string)` |
| **Required** | No                 |
| **Default**  | `"linear"`         |

Must be one of:
* "linear"
* "meta"

#### <a name="network_g_anyOf_i11_upsampler"></a>5.12.13. Property `ReduxOptions > network_g > anyOf > flexnet > upsampler`

|              |                    |
| ------------ | ------------------ |
| **Type**     | `enum (of string)` |
| **Required** | No                 |
| **Default**  | `"pixelshuffle"`   |

Must be one of:
* "dysample"
* "nearest+conv"
* "pixelshuffle"

### <a name="network_g_anyOf_i12"></a>5.13. Property `ReduxOptions > network_g > anyOf > metaflexnet`

**Title:** metaflexnet

|                           |                     |
| ------------------------- | ------------------- |
| **Type**                  | `object`            |
| **Required**              | No                  |
| **Additional properties** | Any type allowed    |
| **Defined in**            | #/$defs/metaflexnet |

| Property                             | Pattern | Type             | Deprecated | Definition | Title/Description |
| ------------------------------------ | ------- | ---------------- | ---------- | ---------- | ----------------- |
| + [type](#network_g_anyOf_i12_type ) | No      | enum (of string) | No         | -          | -                 |

#### <a name="network_g_anyOf_i12_type"></a>5.13.1. Property `ReduxOptions > network_g > anyOf > metaflexnet > type`

|              |                    |
| ------------ | ------------------ |
| **Type**     | `enum (of string)` |
| **Required** | Yes                |

Must be one of:
* "metaflexnet"

### <a name="network_g_anyOf_i13"></a>5.14. Property `ReduxOptions > network_g > anyOf > hit_sir`

**Title:** hit_sir

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `object`         |
| **Required**              | No               |
| **Additional properties** | Any type allowed |
| **Defined in**            | #/$defs/hit_sir  |

| Property                             | Pattern | Type             | Deprecated | Definition | Title/Description |
| ------------------------------------ | ------- | ---------------- | ---------- | ---------- | ----------------- |
| + [type](#network_g_anyOf_i13_type ) | No      | enum (of string) | No         | -          | -                 |

#### <a name="network_g_anyOf_i13_type"></a>5.14.1. Property `ReduxOptions > network_g > anyOf > hit_sir > type`

|              |                    |
| ------------ | ------------------ |
| **Type**     | `enum (of string)` |
| **Required** | Yes                |

Must be one of:
* "hit_sir"

### <a name="network_g_anyOf_i14"></a>5.15. Property `ReduxOptions > network_g > anyOf > hit_sng`

**Title:** hit_sng

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `object`         |
| **Required**              | No               |
| **Additional properties** | Any type allowed |
| **Defined in**            | #/$defs/hit_sng  |

| Property                             | Pattern | Type             | Deprecated | Definition | Title/Description |
| ------------------------------------ | ------- | ---------------- | ---------- | ---------- | ----------------- |
| + [type](#network_g_anyOf_i14_type ) | No      | enum (of string) | No         | -          | -                 |

#### <a name="network_g_anyOf_i14_type"></a>5.15.1. Property `ReduxOptions > network_g > anyOf > hit_sng > type`

|              |                    |
| ------------ | ------------------ |
| **Type**     | `enum (of string)` |
| **Required** | Yes                |

Must be one of:
* "hit_sng"

### <a name="network_g_anyOf_i15"></a>5.16. Property `ReduxOptions > network_g > anyOf > hit_srf`

**Title:** hit_srf

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `object`         |
| **Required**              | No               |
| **Additional properties** | Any type allowed |
| **Defined in**            | #/$defs/hit_srf  |

| Property                             | Pattern | Type             | Deprecated | Definition | Title/Description |
| ------------------------------------ | ------- | ---------------- | ---------- | ---------- | ----------------- |
| + [type](#network_g_anyOf_i15_type ) | No      | enum (of string) | No         | -          | -                 |

#### <a name="network_g_anyOf_i15_type"></a>5.16.1. Property `ReduxOptions > network_g > anyOf > hit_srf > type`

|              |                    |
| ------------ | ------------------ |
| **Type**     | `enum (of string)` |
| **Required** | Yes                |

Must be one of:
* "hit_srf"

### <a name="network_g_anyOf_i16"></a>5.17. Property `ReduxOptions > network_g > anyOf > ipt`

**Title:** ipt

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `object`         |
| **Required**              | No               |
| **Additional properties** | Any type allowed |
| **Defined in**            | #/$defs/ipt      |

| Property                             | Pattern | Type             | Deprecated | Definition | Title/Description |
| ------------------------------------ | ------- | ---------------- | ---------- | ---------- | ----------------- |
| + [type](#network_g_anyOf_i16_type ) | No      | enum (of string) | No         | -          | -                 |

#### <a name="network_g_anyOf_i16_type"></a>5.17.1. Property `ReduxOptions > network_g > anyOf > ipt > type`

|              |                    |
| ------------ | ------------------ |
| **Type**     | `enum (of string)` |
| **Required** | Yes                |

Must be one of:
* "ipt"

### <a name="network_g_anyOf_i17"></a>5.18. Property `ReduxOptions > network_g > anyOf > lmlt`

**Title:** lmlt

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `object`         |
| **Required**              | No               |
| **Additional properties** | Any type allowed |
| **Defined in**            | #/$defs/lmlt     |

| Property                                                 | Pattern | Type             | Deprecated | Definition | Title/Description |
| -------------------------------------------------------- | ------- | ---------------- | ---------- | ---------- | ----------------- |
| + [type](#network_g_anyOf_i17_type )                     | No      | enum (of string) | No         | -          | -                 |
| + [dim](#network_g_anyOf_i17_dim )                       | No      | integer          | No         | -          | -                 |
| - [n_blocks](#network_g_anyOf_i17_n_blocks )             | No      | integer          | No         | -          | -                 |
| - [ffn_scale](#network_g_anyOf_i17_ffn_scale )           | No      | number           | No         | -          | -                 |
| - [scale](#network_g_anyOf_i17_scale )                   | No      | integer          | No         | -          | -                 |
| - [drop_rate](#network_g_anyOf_i17_drop_rate )           | No      | number           | No         | -          | -                 |
| - [attn_drop_rate](#network_g_anyOf_i17_attn_drop_rate ) | No      | number           | No         | -          | -                 |
| - [drop_path_rate](#network_g_anyOf_i17_drop_path_rate ) | No      | number           | No         | -          | -                 |

#### <a name="network_g_anyOf_i17_type"></a>5.18.1. Property `ReduxOptions > network_g > anyOf > lmlt > type`

|              |                    |
| ------------ | ------------------ |
| **Type**     | `enum (of string)` |
| **Required** | Yes                |

Must be one of:
* "lmlt"

#### <a name="network_g_anyOf_i17_dim"></a>5.18.2. Property `ReduxOptions > network_g > anyOf > lmlt > dim`

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | Yes       |

#### <a name="network_g_anyOf_i17_n_blocks"></a>5.18.3. Property `ReduxOptions > network_g > anyOf > lmlt > n_blocks`

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | No        |
| **Default**  | `8`       |

#### <a name="network_g_anyOf_i17_ffn_scale"></a>5.18.4. Property `ReduxOptions > network_g > anyOf > lmlt > ffn_scale`

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |
| **Default**  | `2.0`    |

#### <a name="network_g_anyOf_i17_scale"></a>5.18.5. Property `ReduxOptions > network_g > anyOf > lmlt > scale`

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | No        |
| **Default**  | `4`       |

#### <a name="network_g_anyOf_i17_drop_rate"></a>5.18.6. Property `ReduxOptions > network_g > anyOf > lmlt > drop_rate`

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |
| **Default**  | `0.0`    |

#### <a name="network_g_anyOf_i17_attn_drop_rate"></a>5.18.7. Property `ReduxOptions > network_g > anyOf > lmlt > attn_drop_rate`

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |
| **Default**  | `0.0`    |

#### <a name="network_g_anyOf_i17_drop_path_rate"></a>5.18.8. Property `ReduxOptions > network_g > anyOf > lmlt > drop_path_rate`

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |
| **Default**  | `0.0`    |

### <a name="network_g_anyOf_i18"></a>5.19. Property `ReduxOptions > network_g > anyOf > lmlt_base`

**Title:** lmlt_base

|                           |                   |
| ------------------------- | ----------------- |
| **Type**                  | `object`          |
| **Required**              | No                |
| **Additional properties** | Any type allowed  |
| **Defined in**            | #/$defs/lmlt_base |

| Property                             | Pattern | Type             | Deprecated | Definition | Title/Description |
| ------------------------------------ | ------- | ---------------- | ---------- | ---------- | ----------------- |
| + [type](#network_g_anyOf_i18_type ) | No      | enum (of string) | No         | -          | -                 |

#### <a name="network_g_anyOf_i18_type"></a>5.19.1. Property `ReduxOptions > network_g > anyOf > lmlt_base > type`

|              |                    |
| ------------ | ------------------ |
| **Type**     | `enum (of string)` |
| **Required** | Yes                |

Must be one of:
* "lmlt_base"

### <a name="network_g_anyOf_i19"></a>5.20. Property `ReduxOptions > network_g > anyOf > lmlt_large`

**Title:** lmlt_large

|                           |                    |
| ------------------------- | ------------------ |
| **Type**                  | `object`           |
| **Required**              | No                 |
| **Additional properties** | Any type allowed   |
| **Defined in**            | #/$defs/lmlt_large |

| Property                             | Pattern | Type             | Deprecated | Definition | Title/Description |
| ------------------------------------ | ------- | ---------------- | ---------- | ---------- | ----------------- |
| + [type](#network_g_anyOf_i19_type ) | No      | enum (of string) | No         | -          | -                 |

#### <a name="network_g_anyOf_i19_type"></a>5.20.1. Property `ReduxOptions > network_g > anyOf > lmlt_large > type`

|              |                    |
| ------------ | ------------------ |
| **Type**     | `enum (of string)` |
| **Required** | Yes                |

Must be one of:
* "lmlt_large"

### <a name="network_g_anyOf_i20"></a>5.21. Property `ReduxOptions > network_g > anyOf > lmlt_tiny`

**Title:** lmlt_tiny

|                           |                   |
| ------------------------- | ----------------- |
| **Type**                  | `object`          |
| **Required**              | No                |
| **Additional properties** | Any type allowed  |
| **Defined in**            | #/$defs/lmlt_tiny |

| Property                             | Pattern | Type             | Deprecated | Definition | Title/Description |
| ------------------------------------ | ------- | ---------------- | ---------- | ---------- | ----------------- |
| + [type](#network_g_anyOf_i20_type ) | No      | enum (of string) | No         | -          | -                 |

#### <a name="network_g_anyOf_i20_type"></a>5.21.1. Property `ReduxOptions > network_g > anyOf > lmlt_tiny > type`

|              |                    |
| ------------ | ------------------ |
| **Type**     | `enum (of string)` |
| **Required** | Yes                |

Must be one of:
* "lmlt_tiny"

### <a name="network_g_anyOf_i21"></a>5.22. Property `ReduxOptions > network_g > anyOf > man`

**Title:** man

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `object`         |
| **Required**              | No               |
| **Additional properties** | Any type allowed |
| **Defined in**            | #/$defs/man      |

| Property                                           | Pattern | Type             | Deprecated | Definition | Title/Description |
| -------------------------------------------------- | ------- | ---------------- | ---------- | ---------- | ----------------- |
| + [type](#network_g_anyOf_i21_type )               | No      | enum (of string) | No         | -          | -                 |
| - [n_resblocks](#network_g_anyOf_i21_n_resblocks ) | No      | integer          | No         | -          | -                 |
| - [n_resgroups](#network_g_anyOf_i21_n_resgroups ) | No      | integer          | No         | -          | -                 |
| - [n_colors](#network_g_anyOf_i21_n_colors )       | No      | integer          | No         | -          | -                 |
| - [n_feats](#network_g_anyOf_i21_n_feats )         | No      | integer          | No         | -          | -                 |
| - [scale](#network_g_anyOf_i21_scale )             | No      | integer          | No         | -          | -                 |
| - [res_scale](#network_g_anyOf_i21_res_scale )     | No      | number           | No         | -          | -                 |

#### <a name="network_g_anyOf_i21_type"></a>5.22.1. Property `ReduxOptions > network_g > anyOf > man > type`

|              |                    |
| ------------ | ------------------ |
| **Type**     | `enum (of string)` |
| **Required** | Yes                |

Must be one of:
* "man"

#### <a name="network_g_anyOf_i21_n_resblocks"></a>5.22.2. Property `ReduxOptions > network_g > anyOf > man > n_resblocks`

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | No        |
| **Default**  | `36`      |

#### <a name="network_g_anyOf_i21_n_resgroups"></a>5.22.3. Property `ReduxOptions > network_g > anyOf > man > n_resgroups`

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | No        |
| **Default**  | `1`       |

#### <a name="network_g_anyOf_i21_n_colors"></a>5.22.4. Property `ReduxOptions > network_g > anyOf > man > n_colors`

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | No        |
| **Default**  | `3`       |

#### <a name="network_g_anyOf_i21_n_feats"></a>5.22.5. Property `ReduxOptions > network_g > anyOf > man > n_feats`

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | No        |
| **Default**  | `180`     |

#### <a name="network_g_anyOf_i21_scale"></a>5.22.6. Property `ReduxOptions > network_g > anyOf > man > scale`

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | No        |
| **Default**  | `2`       |

#### <a name="network_g_anyOf_i21_res_scale"></a>5.22.7. Property `ReduxOptions > network_g > anyOf > man > res_scale`

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |
| **Default**  | `1.0`    |

### <a name="network_g_anyOf_i22"></a>5.23. Property `ReduxOptions > network_g > anyOf > man_tiny`

**Title:** man_tiny

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `object`         |
| **Required**              | No               |
| **Additional properties** | Any type allowed |
| **Defined in**            | #/$defs/man_tiny |

| Property                             | Pattern | Type             | Deprecated | Definition | Title/Description |
| ------------------------------------ | ------- | ---------------- | ---------- | ---------- | ----------------- |
| + [type](#network_g_anyOf_i22_type ) | No      | enum (of string) | No         | -          | -                 |

#### <a name="network_g_anyOf_i22_type"></a>5.23.1. Property `ReduxOptions > network_g > anyOf > man_tiny > type`

|              |                    |
| ------------ | ------------------ |
| **Type**     | `enum (of string)` |
| **Required** | Yes                |

Must be one of:
* "man_tiny"

### <a name="network_g_anyOf_i23"></a>5.24. Property `ReduxOptions > network_g > anyOf > man_light`

**Title:** man_light

|                           |                   |
| ------------------------- | ----------------- |
| **Type**                  | `object`          |
| **Required**              | No                |
| **Additional properties** | Any type allowed  |
| **Defined in**            | #/$defs/man_light |

| Property                             | Pattern | Type             | Deprecated | Definition | Title/Description |
| ------------------------------------ | ------- | ---------------- | ---------- | ---------- | ----------------- |
| + [type](#network_g_anyOf_i23_type ) | No      | enum (of string) | No         | -          | -                 |

#### <a name="network_g_anyOf_i23_type"></a>5.24.1. Property `ReduxOptions > network_g > anyOf > man_light > type`

|              |                    |
| ------------ | ------------------ |
| **Type**     | `enum (of string)` |
| **Required** | Yes                |

Must be one of:
* "man_light"

### <a name="network_g_anyOf_i24"></a>5.25. Property `ReduxOptions > network_g > anyOf > moesr2`

**Title:** moesr2

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `object`         |
| **Required**              | No               |
| **Additional properties** | Any type allowed |
| **Defined in**            | #/$defs/moesr2   |

| Property                                                     | Pattern | Type             | Deprecated | Definition | Title/Description |
| ------------------------------------------------------------ | ------- | ---------------- | ---------- | ---------- | ----------------- |
| + [type](#network_g_anyOf_i24_type )                         | No      | enum (of string) | No         | -          | -                 |
| - [in_ch](#network_g_anyOf_i24_in_ch )                       | No      | integer          | No         | -          | -                 |
| - [out_ch](#network_g_anyOf_i24_out_ch )                     | No      | integer          | No         | -          | -                 |
| - [scale](#network_g_anyOf_i24_scale )                       | No      | integer          | No         | -          | -                 |
| - [dim](#network_g_anyOf_i24_dim )                           | No      | integer          | No         | -          | -                 |
| - [n_blocks](#network_g_anyOf_i24_n_blocks )                 | No      | integer          | No         | -          | -                 |
| - [n_block](#network_g_anyOf_i24_n_block )                   | No      | integer          | No         | -          | -                 |
| - [expansion_factor](#network_g_anyOf_i24_expansion_factor ) | No      | integer          | No         | -          | -                 |
| - [expansion_msg](#network_g_anyOf_i24_expansion_msg )       | No      | integer          | No         | -          | -                 |
| - [upsampler](#network_g_anyOf_i24_upsampler )               | No      | enum (of string) | No         | -          | -                 |
| - [upsample_dim](#network_g_anyOf_i24_upsample_dim )         | No      | integer          | No         | -          | -                 |

#### <a name="network_g_anyOf_i24_type"></a>5.25.1. Property `ReduxOptions > network_g > anyOf > moesr2 > type`

|              |                    |
| ------------ | ------------------ |
| **Type**     | `enum (of string)` |
| **Required** | Yes                |

Must be one of:
* "moesr2"

#### <a name="network_g_anyOf_i24_in_ch"></a>5.25.2. Property `ReduxOptions > network_g > anyOf > moesr2 > in_ch`

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | No        |
| **Default**  | `3`       |

#### <a name="network_g_anyOf_i24_out_ch"></a>5.25.3. Property `ReduxOptions > network_g > anyOf > moesr2 > out_ch`

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | No        |
| **Default**  | `3`       |

#### <a name="network_g_anyOf_i24_scale"></a>5.25.4. Property `ReduxOptions > network_g > anyOf > moesr2 > scale`

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | No        |
| **Default**  | `4`       |

#### <a name="network_g_anyOf_i24_dim"></a>5.25.5. Property `ReduxOptions > network_g > anyOf > moesr2 > dim`

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | No        |
| **Default**  | `64`      |

#### <a name="network_g_anyOf_i24_n_blocks"></a>5.25.6. Property `ReduxOptions > network_g > anyOf > moesr2 > n_blocks`

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | No        |
| **Default**  | `9`       |

#### <a name="network_g_anyOf_i24_n_block"></a>5.25.7. Property `ReduxOptions > network_g > anyOf > moesr2 > n_block`

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | No        |
| **Default**  | `4`       |

#### <a name="network_g_anyOf_i24_expansion_factor"></a>5.25.8. Property `ReduxOptions > network_g > anyOf > moesr2 > expansion_factor`

|              |                      |
| ------------ | -------------------- |
| **Type**     | `integer`            |
| **Required** | No                   |
| **Default**  | `2.6666666666666665` |

#### <a name="network_g_anyOf_i24_expansion_msg"></a>5.25.9. Property `ReduxOptions > network_g > anyOf > moesr2 > expansion_msg`

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | No        |
| **Default**  | `1.5`     |

#### <a name="network_g_anyOf_i24_upsampler"></a>5.25.10. Property `ReduxOptions > network_g > anyOf > moesr2 > upsampler`

|              |                        |
| ------------ | ---------------------- |
| **Type**     | `enum (of string)`     |
| **Required** | No                     |
| **Default**  | `"pixelshuffledirect"` |

Must be one of:
* "conv"
* "dysample"
* "nearest+conv"
* "pixelshuffle"
* "pixelshuffledirect"

#### <a name="network_g_anyOf_i24_upsample_dim"></a>5.25.11. Property `ReduxOptions > network_g > anyOf > moesr2 > upsample_dim`

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | No        |
| **Default**  | `64`      |

### <a name="network_g_anyOf_i25"></a>5.26. Property `ReduxOptions > network_g > anyOf > mosr`

**Title:** mosr

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `object`         |
| **Required**              | No               |
| **Additional properties** | Any type allowed |
| **Defined in**            | #/$defs/mosr     |

| Property                             | Pattern | Type             | Deprecated | Definition | Title/Description |
| ------------------------------------ | ------- | ---------------- | ---------- | ---------- | ----------------- |
| + [type](#network_g_anyOf_i25_type ) | No      | enum (of string) | No         | -          | -                 |

#### <a name="network_g_anyOf_i25_type"></a>5.26.1. Property `ReduxOptions > network_g > anyOf > mosr > type`

|              |                    |
| ------------ | ------------------ |
| **Type**     | `enum (of string)` |
| **Required** | Yes                |

Must be one of:
* "mosr"

### <a name="network_g_anyOf_i26"></a>5.27. Property `ReduxOptions > network_g > anyOf > mosr_t`

**Title:** mosr_t

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `object`         |
| **Required**              | No               |
| **Additional properties** | Any type allowed |
| **Defined in**            | #/$defs/mosr_t   |

| Property                             | Pattern | Type             | Deprecated | Definition | Title/Description |
| ------------------------------------ | ------- | ---------------- | ---------- | ---------- | ----------------- |
| + [type](#network_g_anyOf_i26_type ) | No      | enum (of string) | No         | -          | -                 |

#### <a name="network_g_anyOf_i26_type"></a>5.27.1. Property `ReduxOptions > network_g > anyOf > mosr_t > type`

|              |                    |
| ------------ | ------------------ |
| **Type**     | `enum (of string)` |
| **Required** | Yes                |

Must be one of:
* "mosr_t"

### <a name="network_g_anyOf_i27"></a>5.28. Property `ReduxOptions > network_g > anyOf > rcancab`

**Title:** rcancab

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `object`         |
| **Required**              | No               |
| **Additional properties** | Any type allowed |
| **Defined in**            | #/$defs/rcancab  |

| Property                                           | Pattern | Type             | Deprecated | Definition | Title/Description |
| -------------------------------------------------- | ------- | ---------------- | ---------- | ---------- | ----------------- |
| + [type](#network_g_anyOf_i27_type )               | No      | enum (of string) | No         | -          | -                 |
| - [scale](#network_g_anyOf_i27_scale )             | No      | integer          | No         | -          | -                 |
| - [n_resgroups](#network_g_anyOf_i27_n_resgroups ) | No      | integer          | No         | -          | -                 |
| - [n_resblocks](#network_g_anyOf_i27_n_resblocks ) | No      | integer          | No         | -          | -                 |
| - [n_feats](#network_g_anyOf_i27_n_feats )         | No      | integer          | No         | -          | -                 |
| - [n_colors](#network_g_anyOf_i27_n_colors )       | No      | integer          | No         | -          | -                 |
| - [kernel_size](#network_g_anyOf_i27_kernel_size ) | No      | integer          | No         | -          | -                 |
| - [reduction](#network_g_anyOf_i27_reduction )     | No      | integer          | No         | -          | -                 |
| - [res_scale](#network_g_anyOf_i27_res_scale )     | No      | number           | No         | -          | -                 |

#### <a name="network_g_anyOf_i27_type"></a>5.28.1. Property `ReduxOptions > network_g > anyOf > rcancab > type`

|              |                    |
| ------------ | ------------------ |
| **Type**     | `enum (of string)` |
| **Required** | Yes                |

Must be one of:
* "rcancab"

#### <a name="network_g_anyOf_i27_scale"></a>5.28.2. Property `ReduxOptions > network_g > anyOf > rcancab > scale`

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | No        |
| **Default**  | `4`       |

#### <a name="network_g_anyOf_i27_n_resgroups"></a>5.28.3. Property `ReduxOptions > network_g > anyOf > rcancab > n_resgroups`

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | No        |
| **Default**  | `10`      |

#### <a name="network_g_anyOf_i27_n_resblocks"></a>5.28.4. Property `ReduxOptions > network_g > anyOf > rcancab > n_resblocks`

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | No        |
| **Default**  | `20`      |

#### <a name="network_g_anyOf_i27_n_feats"></a>5.28.5. Property `ReduxOptions > network_g > anyOf > rcancab > n_feats`

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | No        |
| **Default**  | `64`      |

#### <a name="network_g_anyOf_i27_n_colors"></a>5.28.6. Property `ReduxOptions > network_g > anyOf > rcancab > n_colors`

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | No        |
| **Default**  | `3`       |

#### <a name="network_g_anyOf_i27_kernel_size"></a>5.28.7. Property `ReduxOptions > network_g > anyOf > rcancab > kernel_size`

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | No        |
| **Default**  | `3`       |

#### <a name="network_g_anyOf_i27_reduction"></a>5.28.8. Property `ReduxOptions > network_g > anyOf > rcancab > reduction`

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | No        |
| **Default**  | `16`      |

#### <a name="network_g_anyOf_i27_res_scale"></a>5.28.9. Property `ReduxOptions > network_g > anyOf > rcancab > res_scale`

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |
| **Default**  | `1`      |

### <a name="network_g_anyOf_i28"></a>5.29. Property `ReduxOptions > network_g > anyOf > rcan`

**Title:** rcan

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `object`         |
| **Required**              | No               |
| **Additional properties** | Any type allowed |
| **Defined in**            | #/$defs/rcan     |

| Property                                           | Pattern | Type             | Deprecated | Definition | Title/Description |
| -------------------------------------------------- | ------- | ---------------- | ---------- | ---------- | ----------------- |
| + [type](#network_g_anyOf_i28_type )               | No      | enum (of string) | No         | -          | -                 |
| - [scale](#network_g_anyOf_i28_scale )             | No      | integer          | No         | -          | -                 |
| - [n_resgroups](#network_g_anyOf_i28_n_resgroups ) | No      | integer          | No         | -          | -                 |
| - [n_resblocks](#network_g_anyOf_i28_n_resblocks ) | No      | integer          | No         | -          | -                 |
| - [n_feats](#network_g_anyOf_i28_n_feats )         | No      | integer          | No         | -          | -                 |
| - [n_colors](#network_g_anyOf_i28_n_colors )       | No      | integer          | No         | -          | -                 |
| - [rgb_range](#network_g_anyOf_i28_rgb_range )     | No      | integer          | No         | -          | -                 |
| - [kernel_size](#network_g_anyOf_i28_kernel_size ) | No      | integer          | No         | -          | -                 |
| - [reduction](#network_g_anyOf_i28_reduction )     | No      | integer          | No         | -          | -                 |
| - [res_scale](#network_g_anyOf_i28_res_scale )     | No      | number           | No         | -          | -                 |

#### <a name="network_g_anyOf_i28_type"></a>5.29.1. Property `ReduxOptions > network_g > anyOf > rcan > type`

|              |                    |
| ------------ | ------------------ |
| **Type**     | `enum (of string)` |
| **Required** | Yes                |

Must be one of:
* "rcan"

#### <a name="network_g_anyOf_i28_scale"></a>5.29.2. Property `ReduxOptions > network_g > anyOf > rcan > scale`

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | No        |
| **Default**  | `4`       |

#### <a name="network_g_anyOf_i28_n_resgroups"></a>5.29.3. Property `ReduxOptions > network_g > anyOf > rcan > n_resgroups`

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | No        |
| **Default**  | `10`      |

#### <a name="network_g_anyOf_i28_n_resblocks"></a>5.29.4. Property `ReduxOptions > network_g > anyOf > rcan > n_resblocks`

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | No        |
| **Default**  | `20`      |

#### <a name="network_g_anyOf_i28_n_feats"></a>5.29.5. Property `ReduxOptions > network_g > anyOf > rcan > n_feats`

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | No        |
| **Default**  | `64`      |

#### <a name="network_g_anyOf_i28_n_colors"></a>5.29.6. Property `ReduxOptions > network_g > anyOf > rcan > n_colors`

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | No        |
| **Default**  | `3`       |

#### <a name="network_g_anyOf_i28_rgb_range"></a>5.29.7. Property `ReduxOptions > network_g > anyOf > rcan > rgb_range`

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | No        |
| **Default**  | `255`     |

#### <a name="network_g_anyOf_i28_kernel_size"></a>5.29.8. Property `ReduxOptions > network_g > anyOf > rcan > kernel_size`

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | No        |
| **Default**  | `3`       |

#### <a name="network_g_anyOf_i28_reduction"></a>5.29.9. Property `ReduxOptions > network_g > anyOf > rcan > reduction`

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | No        |
| **Default**  | `16`      |

#### <a name="network_g_anyOf_i28_res_scale"></a>5.29.10. Property `ReduxOptions > network_g > anyOf > rcan > res_scale`

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |
| **Default**  | `1`      |

### <a name="network_g_anyOf_i29"></a>5.30. Property `ReduxOptions > network_g > anyOf > realplksr`

**Title:** realplksr

|                           |                   |
| ------------------------- | ----------------- |
| **Type**                  | `object`          |
| **Required**              | No                |
| **Additional properties** | Any type allowed  |
| **Defined in**            | #/$defs/realplksr |

| Property                             | Pattern | Type             | Deprecated | Definition | Title/Description |
| ------------------------------------ | ------- | ---------------- | ---------- | ---------- | ----------------- |
| + [type](#network_g_anyOf_i29_type ) | No      | enum (of string) | No         | -          | -                 |

#### <a name="network_g_anyOf_i29_type"></a>5.30.1. Property `ReduxOptions > network_g > anyOf > realplksr > type`

|              |                    |
| ------------ | ------------------ |
| **Type**     | `enum (of string)` |
| **Required** | Yes                |

Must be one of:
* "realplksr"

### <a name="network_g_anyOf_i30"></a>5.31. Property `ReduxOptions > network_g > anyOf > scunet_aaf6aa`

**Title:** scunet_aaf6aa

|                           |                       |
| ------------------------- | --------------------- |
| **Type**                  | `object`              |
| **Required**              | No                    |
| **Additional properties** | Any type allowed      |
| **Defined in**            | #/$defs/scunet_aaf6aa |

| Property                             | Pattern | Type             | Deprecated | Definition | Title/Description |
| ------------------------------------ | ------- | ---------------- | ---------- | ---------- | ----------------- |
| + [type](#network_g_anyOf_i30_type ) | No      | enum (of string) | No         | -          | -                 |

#### <a name="network_g_anyOf_i30_type"></a>5.31.1. Property `ReduxOptions > network_g > anyOf > scunet_aaf6aa > type`

|              |                    |
| ------------ | ------------------ |
| **Type**     | `enum (of string)` |
| **Required** | Yes                |

Must be one of:
* "scunet_aaf6aa"

### <a name="network_g_anyOf_i31"></a>5.32. Property `ReduxOptions > network_g > anyOf > spanplus`

**Title:** spanplus

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `object`         |
| **Required**              | No               |
| **Additional properties** | Any type allowed |
| **Defined in**            | #/$defs/spanplus |

| Property                             | Pattern | Type             | Deprecated | Definition | Title/Description |
| ------------------------------------ | ------- | ---------------- | ---------- | ---------- | ----------------- |
| + [type](#network_g_anyOf_i31_type ) | No      | enum (of string) | No         | -          | -                 |

#### <a name="network_g_anyOf_i31_type"></a>5.32.1. Property `ReduxOptions > network_g > anyOf > spanplus > type`

|              |                    |
| ------------ | ------------------ |
| **Type**     | `enum (of string)` |
| **Required** | Yes                |

Must be one of:
* "spanplus"

### <a name="network_g_anyOf_i32"></a>5.33. Property `ReduxOptions > network_g > anyOf > spanplus_sts`

**Title:** spanplus_sts

|                           |                      |
| ------------------------- | -------------------- |
| **Type**                  | `object`             |
| **Required**              | No                   |
| **Additional properties** | Any type allowed     |
| **Defined in**            | #/$defs/spanplus_sts |

| Property                             | Pattern | Type             | Deprecated | Definition | Title/Description |
| ------------------------------------ | ------- | ---------------- | ---------- | ---------- | ----------------- |
| + [type](#network_g_anyOf_i32_type ) | No      | enum (of string) | No         | -          | -                 |

#### <a name="network_g_anyOf_i32_type"></a>5.33.1. Property `ReduxOptions > network_g > anyOf > spanplus_sts > type`

|              |                    |
| ------------ | ------------------ |
| **Type**     | `enum (of string)` |
| **Required** | Yes                |

Must be one of:
* "spanplus_sts"

### <a name="network_g_anyOf_i33"></a>5.34. Property `ReduxOptions > network_g > anyOf > spanplus_s`

**Title:** spanplus_s

|                           |                    |
| ------------------------- | ------------------ |
| **Type**                  | `object`           |
| **Required**              | No                 |
| **Additional properties** | Any type allowed   |
| **Defined in**            | #/$defs/spanplus_s |

| Property                             | Pattern | Type             | Deprecated | Definition | Title/Description |
| ------------------------------------ | ------- | ---------------- | ---------- | ---------- | ----------------- |
| + [type](#network_g_anyOf_i33_type ) | No      | enum (of string) | No         | -          | -                 |

#### <a name="network_g_anyOf_i33_type"></a>5.34.1. Property `ReduxOptions > network_g > anyOf > spanplus_s > type`

|              |                    |
| ------------ | ------------------ |
| **Type**     | `enum (of string)` |
| **Required** | Yes                |

Must be one of:
* "spanplus_s"

### <a name="network_g_anyOf_i34"></a>5.35. Property `ReduxOptions > network_g > anyOf > spanplus_st`

**Title:** spanplus_st

|                           |                     |
| ------------------------- | ------------------- |
| **Type**                  | `object`            |
| **Required**              | No                  |
| **Additional properties** | Any type allowed    |
| **Defined in**            | #/$defs/spanplus_st |

| Property                             | Pattern | Type             | Deprecated | Definition | Title/Description |
| ------------------------------------ | ------- | ---------------- | ---------- | ---------- | ----------------- |
| + [type](#network_g_anyOf_i34_type ) | No      | enum (of string) | No         | -          | -                 |

#### <a name="network_g_anyOf_i34_type"></a>5.35.1. Property `ReduxOptions > network_g > anyOf > spanplus_st > type`

|              |                    |
| ------------ | ------------------ |
| **Type**     | `enum (of string)` |
| **Required** | Yes                |

Must be one of:
* "spanplus_st"

### <a name="network_g_anyOf_i35"></a>5.36. Property `ReduxOptions > network_g > anyOf > compact`

**Title:** compact

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `object`         |
| **Required**              | No               |
| **Additional properties** | Any type allowed |
| **Defined in**            | #/$defs/compact  |

| Property                             | Pattern | Type             | Deprecated | Definition | Title/Description |
| ------------------------------------ | ------- | ---------------- | ---------- | ---------- | ----------------- |
| + [type](#network_g_anyOf_i35_type ) | No      | enum (of string) | No         | -          | -                 |

#### <a name="network_g_anyOf_i35_type"></a>5.36.1. Property `ReduxOptions > network_g > anyOf > compact > type`

|              |                    |
| ------------ | ------------------ |
| **Type**     | `enum (of string)` |
| **Required** | Yes                |

Must be one of:
* "compact"

### <a name="network_g_anyOf_i36"></a>5.37. Property `ReduxOptions > network_g > anyOf > ultracompact`

**Title:** ultracompact

|                           |                      |
| ------------------------- | -------------------- |
| **Type**                  | `object`             |
| **Required**              | No                   |
| **Additional properties** | Any type allowed     |
| **Defined in**            | #/$defs/ultracompact |

| Property                             | Pattern | Type             | Deprecated | Definition | Title/Description |
| ------------------------------------ | ------- | ---------------- | ---------- | ---------- | ----------------- |
| + [type](#network_g_anyOf_i36_type ) | No      | enum (of string) | No         | -          | -                 |

#### <a name="network_g_anyOf_i36_type"></a>5.37.1. Property `ReduxOptions > network_g > anyOf > ultracompact > type`

|              |                    |
| ------------ | ------------------ |
| **Type**     | `enum (of string)` |
| **Required** | Yes                |

Must be one of:
* "ultracompact"

### <a name="network_g_anyOf_i37"></a>5.38. Property `ReduxOptions > network_g > anyOf > superultracompact`

**Title:** superultracompact

|                           |                           |
| ------------------------- | ------------------------- |
| **Type**                  | `object`                  |
| **Required**              | No                        |
| **Additional properties** | Any type allowed          |
| **Defined in**            | #/$defs/superultracompact |

| Property                             | Pattern | Type             | Deprecated | Definition | Title/Description |
| ------------------------------------ | ------- | ---------------- | ---------- | ---------- | ----------------- |
| + [type](#network_g_anyOf_i37_type ) | No      | enum (of string) | No         | -          | -                 |

#### <a name="network_g_anyOf_i37_type"></a>5.38.1. Property `ReduxOptions > network_g > anyOf > superultracompact > type`

|              |                    |
| ------------ | ------------------ |
| **Type**     | `enum (of string)` |
| **Required** | Yes                |

Must be one of:
* "superultracompact"

### <a name="network_g_anyOf_i38"></a>5.39. Property `ReduxOptions > network_g > anyOf > tscunet`

**Title:** tscunet

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `object`         |
| **Required**              | No               |
| **Additional properties** | Any type allowed |
| **Defined in**            | #/$defs/tscunet  |

| Property                             | Pattern | Type             | Deprecated | Definition | Title/Description |
| ------------------------------------ | ------- | ---------------- | ---------- | ---------- | ----------------- |
| + [type](#network_g_anyOf_i38_type ) | No      | enum (of string) | No         | -          | -                 |

#### <a name="network_g_anyOf_i38_type"></a>5.39.1. Property `ReduxOptions > network_g > anyOf > tscunet > type`

|              |                    |
| ------------ | ------------------ |
| **Type**     | `enum (of string)` |
| **Required** | Yes                |

Must be one of:
* "tscunet"

### <a name="network_g_anyOf_i39"></a>5.40. Property `ReduxOptions > network_g > anyOf > atd`

**Title:** atd

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `object`         |
| **Required**              | No               |
| **Additional properties** | Any type allowed |
| **Defined in**            | #/$defs/atd      |

| Property                             | Pattern | Type             | Deprecated | Definition | Title/Description |
| ------------------------------------ | ------- | ---------------- | ---------- | ---------- | ----------------- |
| + [type](#network_g_anyOf_i39_type ) | No      | enum (of string) | No         | -          | -                 |

#### <a name="network_g_anyOf_i39_type"></a>5.40.1. Property `ReduxOptions > network_g > anyOf > atd > type`

|              |                    |
| ------------ | ------------------ |
| **Type**     | `enum (of string)` |
| **Required** | Yes                |

Must be one of:
* "atd"

### <a name="network_g_anyOf_i40"></a>5.41. Property `ReduxOptions > network_g > anyOf > atd_light`

**Title:** atd_light

|                           |                   |
| ------------------------- | ----------------- |
| **Type**                  | `object`          |
| **Required**              | No                |
| **Additional properties** | Any type allowed  |
| **Defined in**            | #/$defs/atd_light |

| Property                             | Pattern | Type             | Deprecated | Definition | Title/Description |
| ------------------------------------ | ------- | ---------------- | ---------- | ---------- | ----------------- |
| + [type](#network_g_anyOf_i40_type ) | No      | enum (of string) | No         | -          | -                 |

#### <a name="network_g_anyOf_i40_type"></a>5.41.1. Property `ReduxOptions > network_g > anyOf > atd_light > type`

|              |                    |
| ------------ | ------------------ |
| **Type**     | `enum (of string)` |
| **Required** | Yes                |

Must be one of:
* "atd_light"

### <a name="network_g_anyOf_i41"></a>5.42. Property `ReduxOptions > network_g > anyOf > dat`

**Title:** dat

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `object`         |
| **Required**              | No               |
| **Additional properties** | Any type allowed |
| **Defined in**            | #/$defs/dat      |

| Property                             | Pattern | Type             | Deprecated | Definition | Title/Description |
| ------------------------------------ | ------- | ---------------- | ---------- | ---------- | ----------------- |
| + [type](#network_g_anyOf_i41_type ) | No      | enum (of string) | No         | -          | -                 |

#### <a name="network_g_anyOf_i41_type"></a>5.42.1. Property `ReduxOptions > network_g > anyOf > dat > type`

|              |                    |
| ------------ | ------------------ |
| **Type**     | `enum (of string)` |
| **Required** | Yes                |

Must be one of:
* "dat"

### <a name="network_g_anyOf_i42"></a>5.43. Property `ReduxOptions > network_g > anyOf > dat_s`

**Title:** dat_s

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `object`         |
| **Required**              | No               |
| **Additional properties** | Any type allowed |
| **Defined in**            | #/$defs/dat_s    |

| Property                             | Pattern | Type             | Deprecated | Definition | Title/Description |
| ------------------------------------ | ------- | ---------------- | ---------- | ---------- | ----------------- |
| + [type](#network_g_anyOf_i42_type ) | No      | enum (of string) | No         | -          | -                 |

#### <a name="network_g_anyOf_i42_type"></a>5.43.1. Property `ReduxOptions > network_g > anyOf > dat_s > type`

|              |                    |
| ------------ | ------------------ |
| **Type**     | `enum (of string)` |
| **Required** | Yes                |

Must be one of:
* "dat_s"

### <a name="network_g_anyOf_i43"></a>5.44. Property `ReduxOptions > network_g > anyOf > dat_2`

**Title:** dat_2

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `object`         |
| **Required**              | No               |
| **Additional properties** | Any type allowed |
| **Defined in**            | #/$defs/dat_2    |

| Property                             | Pattern | Type             | Deprecated | Definition | Title/Description |
| ------------------------------------ | ------- | ---------------- | ---------- | ---------- | ----------------- |
| + [type](#network_g_anyOf_i43_type ) | No      | enum (of string) | No         | -          | -                 |

#### <a name="network_g_anyOf_i43_type"></a>5.44.1. Property `ReduxOptions > network_g > anyOf > dat_2 > type`

|              |                    |
| ------------ | ------------------ |
| **Type**     | `enum (of string)` |
| **Required** | Yes                |

Must be one of:
* "dat_2"

### <a name="network_g_anyOf_i44"></a>5.45. Property `ReduxOptions > network_g > anyOf > dat_light`

**Title:** dat_light

|                           |                   |
| ------------------------- | ----------------- |
| **Type**                  | `object`          |
| **Required**              | No                |
| **Additional properties** | Any type allowed  |
| **Defined in**            | #/$defs/dat_light |

| Property                             | Pattern | Type             | Deprecated | Definition | Title/Description |
| ------------------------------------ | ------- | ---------------- | ---------- | ---------- | ----------------- |
| + [type](#network_g_anyOf_i44_type ) | No      | enum (of string) | No         | -          | -                 |

#### <a name="network_g_anyOf_i44_type"></a>5.45.1. Property `ReduxOptions > network_g > anyOf > dat_light > type`

|              |                    |
| ------------ | ------------------ |
| **Type**     | `enum (of string)` |
| **Required** | Yes                |

Must be one of:
* "dat_light"

### <a name="network_g_anyOf_i45"></a>5.46. Property `ReduxOptions > network_g > anyOf > drct`

**Title:** drct

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `object`         |
| **Required**              | No               |
| **Additional properties** | Any type allowed |
| **Defined in**            | #/$defs/drct     |

| Property                             | Pattern | Type             | Deprecated | Definition | Title/Description |
| ------------------------------------ | ------- | ---------------- | ---------- | ---------- | ----------------- |
| + [type](#network_g_anyOf_i45_type ) | No      | enum (of string) | No         | -          | -                 |

#### <a name="network_g_anyOf_i45_type"></a>5.46.1. Property `ReduxOptions > network_g > anyOf > drct > type`

|              |                    |
| ------------ | ------------------ |
| **Type**     | `enum (of string)` |
| **Required** | Yes                |

Must be one of:
* "drct"

### <a name="network_g_anyOf_i46"></a>5.47. Property `ReduxOptions > network_g > anyOf > drct_l`

**Title:** drct_l

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `object`         |
| **Required**              | No               |
| **Additional properties** | Any type allowed |
| **Defined in**            | #/$defs/drct_l   |

| Property                             | Pattern | Type             | Deprecated | Definition | Title/Description |
| ------------------------------------ | ------- | ---------------- | ---------- | ---------- | ----------------- |
| + [type](#network_g_anyOf_i46_type ) | No      | enum (of string) | No         | -          | -                 |

#### <a name="network_g_anyOf_i46_type"></a>5.47.1. Property `ReduxOptions > network_g > anyOf > drct_l > type`

|              |                    |
| ------------ | ------------------ |
| **Type**     | `enum (of string)` |
| **Required** | Yes                |

Must be one of:
* "drct_l"

### <a name="network_g_anyOf_i47"></a>5.48. Property `ReduxOptions > network_g > anyOf > drct_xl`

**Title:** drct_xl

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `object`         |
| **Required**              | No               |
| **Additional properties** | Any type allowed |
| **Defined in**            | #/$defs/drct_xl  |

| Property                             | Pattern | Type             | Deprecated | Definition | Title/Description |
| ------------------------------------ | ------- | ---------------- | ---------- | ---------- | ----------------- |
| + [type](#network_g_anyOf_i47_type ) | No      | enum (of string) | No         | -          | -                 |

#### <a name="network_g_anyOf_i47_type"></a>5.48.1. Property `ReduxOptions > network_g > anyOf > drct_xl > type`

|              |                    |
| ------------ | ------------------ |
| **Type**     | `enum (of string)` |
| **Required** | Yes                |

Must be one of:
* "drct_xl"

### <a name="network_g_anyOf_i48"></a>5.49. Property `ReduxOptions > network_g > anyOf > hat`

**Title:** hat

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `object`         |
| **Required**              | No               |
| **Additional properties** | Any type allowed |
| **Defined in**            | #/$defs/hat      |

| Property                             | Pattern | Type             | Deprecated | Definition | Title/Description |
| ------------------------------------ | ------- | ---------------- | ---------- | ---------- | ----------------- |
| + [type](#network_g_anyOf_i48_type ) | No      | enum (of string) | No         | -          | -                 |

#### <a name="network_g_anyOf_i48_type"></a>5.49.1. Property `ReduxOptions > network_g > anyOf > hat > type`

|              |                    |
| ------------ | ------------------ |
| **Type**     | `enum (of string)` |
| **Required** | Yes                |

Must be one of:
* "hat"

### <a name="network_g_anyOf_i49"></a>5.50. Property `ReduxOptions > network_g > anyOf > hat_l`

**Title:** hat_l

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `object`         |
| **Required**              | No               |
| **Additional properties** | Any type allowed |
| **Defined in**            | #/$defs/hat_l    |

| Property                             | Pattern | Type             | Deprecated | Definition | Title/Description |
| ------------------------------------ | ------- | ---------------- | ---------- | ---------- | ----------------- |
| + [type](#network_g_anyOf_i49_type ) | No      | enum (of string) | No         | -          | -                 |

#### <a name="network_g_anyOf_i49_type"></a>5.50.1. Property `ReduxOptions > network_g > anyOf > hat_l > type`

|              |                    |
| ------------ | ------------------ |
| **Type**     | `enum (of string)` |
| **Required** | Yes                |

Must be one of:
* "hat_l"

### <a name="network_g_anyOf_i50"></a>5.51. Property `ReduxOptions > network_g > anyOf > hat_m`

**Title:** hat_m

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `object`         |
| **Required**              | No               |
| **Additional properties** | Any type allowed |
| **Defined in**            | #/$defs/hat_m    |

| Property                             | Pattern | Type             | Deprecated | Definition | Title/Description |
| ------------------------------------ | ------- | ---------------- | ---------- | ---------- | ----------------- |
| + [type](#network_g_anyOf_i50_type ) | No      | enum (of string) | No         | -          | -                 |

#### <a name="network_g_anyOf_i50_type"></a>5.51.1. Property `ReduxOptions > network_g > anyOf > hat_m > type`

|              |                    |
| ------------ | ------------------ |
| **Type**     | `enum (of string)` |
| **Required** | Yes                |

Must be one of:
* "hat_m"

### <a name="network_g_anyOf_i51"></a>5.52. Property `ReduxOptions > network_g > anyOf > hat_s`

**Title:** hat_s

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `object`         |
| **Required**              | No               |
| **Additional properties** | Any type allowed |
| **Defined in**            | #/$defs/hat_s    |

| Property                             | Pattern | Type             | Deprecated | Definition | Title/Description |
| ------------------------------------ | ------- | ---------------- | ---------- | ---------- | ----------------- |
| + [type](#network_g_anyOf_i51_type ) | No      | enum (of string) | No         | -          | -                 |

#### <a name="network_g_anyOf_i51_type"></a>5.52.1. Property `ReduxOptions > network_g > anyOf > hat_s > type`

|              |                    |
| ------------ | ------------------ |
| **Type**     | `enum (of string)` |
| **Required** | Yes                |

Must be one of:
* "hat_s"

### <a name="network_g_anyOf_i52"></a>5.53. Property `ReduxOptions > network_g > anyOf > omnisr`

**Title:** omnisr

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `object`         |
| **Required**              | No               |
| **Additional properties** | Any type allowed |
| **Defined in**            | #/$defs/omnisr   |

| Property                             | Pattern | Type             | Deprecated | Definition | Title/Description |
| ------------------------------------ | ------- | ---------------- | ---------- | ---------- | ----------------- |
| + [type](#network_g_anyOf_i52_type ) | No      | enum (of string) | No         | -          | -                 |

#### <a name="network_g_anyOf_i52_type"></a>5.53.1. Property `ReduxOptions > network_g > anyOf > omnisr > type`

|              |                    |
| ------------ | ------------------ |
| **Type**     | `enum (of string)` |
| **Required** | Yes                |

Must be one of:
* "omnisr"

### <a name="network_g_anyOf_i53"></a>5.54. Property `ReduxOptions > network_g > anyOf > plksr`

**Title:** plksr

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `object`         |
| **Required**              | No               |
| **Additional properties** | Any type allowed |
| **Defined in**            | #/$defs/plksr    |

| Property                             | Pattern | Type             | Deprecated | Definition | Title/Description |
| ------------------------------------ | ------- | ---------------- | ---------- | ---------- | ----------------- |
| + [type](#network_g_anyOf_i53_type ) | No      | enum (of string) | No         | -          | -                 |

#### <a name="network_g_anyOf_i53_type"></a>5.54.1. Property `ReduxOptions > network_g > anyOf > plksr > type`

|              |                    |
| ------------ | ------------------ |
| **Type**     | `enum (of string)` |
| **Required** | Yes                |

Must be one of:
* "plksr"

### <a name="network_g_anyOf_i54"></a>5.55. Property `ReduxOptions > network_g > anyOf > plksr_tiny`

**Title:** plksr_tiny

|                           |                    |
| ------------------------- | ------------------ |
| **Type**                  | `object`           |
| **Required**              | No                 |
| **Additional properties** | Any type allowed   |
| **Defined in**            | #/$defs/plksr_tiny |

| Property                             | Pattern | Type             | Deprecated | Definition | Title/Description |
| ------------------------------------ | ------- | ---------------- | ---------- | ---------- | ----------------- |
| + [type](#network_g_anyOf_i54_type ) | No      | enum (of string) | No         | -          | -                 |

#### <a name="network_g_anyOf_i54_type"></a>5.55.1. Property `ReduxOptions > network_g > anyOf > plksr_tiny > type`

|              |                    |
| ------------ | ------------------ |
| **Type**     | `enum (of string)` |
| **Required** | Yes                |

Must be one of:
* "plksr_tiny"

### <a name="network_g_anyOf_i55"></a>5.56. Property `ReduxOptions > network_g > anyOf > realcugan`

**Title:** realcugan

|                           |                   |
| ------------------------- | ----------------- |
| **Type**                  | `object`          |
| **Required**              | No                |
| **Additional properties** | Any type allowed  |
| **Defined in**            | #/$defs/realcugan |

| Property                             | Pattern | Type             | Deprecated | Definition | Title/Description |
| ------------------------------------ | ------- | ---------------- | ---------- | ---------- | ----------------- |
| + [type](#network_g_anyOf_i55_type ) | No      | enum (of string) | No         | -          | -                 |

#### <a name="network_g_anyOf_i55_type"></a>5.56.1. Property `ReduxOptions > network_g > anyOf > realcugan > type`

|              |                    |
| ------------ | ------------------ |
| **Type**     | `enum (of string)` |
| **Required** | Yes                |

Must be one of:
* "realcugan"

### <a name="network_g_anyOf_i56"></a>5.57. Property `ReduxOptions > network_g > anyOf > rgt`

**Title:** rgt

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `object`         |
| **Required**              | No               |
| **Additional properties** | Any type allowed |
| **Defined in**            | #/$defs/rgt      |

| Property                             | Pattern | Type             | Deprecated | Definition | Title/Description |
| ------------------------------------ | ------- | ---------------- | ---------- | ---------- | ----------------- |
| + [type](#network_g_anyOf_i56_type ) | No      | enum (of string) | No         | -          | -                 |

#### <a name="network_g_anyOf_i56_type"></a>5.57.1. Property `ReduxOptions > network_g > anyOf > rgt > type`

|              |                    |
| ------------ | ------------------ |
| **Type**     | `enum (of string)` |
| **Required** | Yes                |

Must be one of:
* "rgt"

### <a name="network_g_anyOf_i57"></a>5.58. Property `ReduxOptions > network_g > anyOf > rgt_s`

**Title:** rgt_s

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `object`         |
| **Required**              | No               |
| **Additional properties** | Any type allowed |
| **Defined in**            | #/$defs/rgt_s    |

| Property                             | Pattern | Type             | Deprecated | Definition | Title/Description |
| ------------------------------------ | ------- | ---------------- | ---------- | ---------- | ----------------- |
| + [type](#network_g_anyOf_i57_type ) | No      | enum (of string) | No         | -          | -                 |

#### <a name="network_g_anyOf_i57_type"></a>5.58.1. Property `ReduxOptions > network_g > anyOf > rgt_s > type`

|              |                    |
| ------------ | ------------------ |
| **Type**     | `enum (of string)` |
| **Required** | Yes                |

Must be one of:
* "rgt_s"

### <a name="network_g_anyOf_i58"></a>5.59. Property `ReduxOptions > network_g > anyOf > esrgan`

**Title:** esrgan

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `object`         |
| **Required**              | No               |
| **Additional properties** | Any type allowed |
| **Defined in**            | #/$defs/esrgan   |

| Property                             | Pattern | Type             | Deprecated | Definition | Title/Description |
| ------------------------------------ | ------- | ---------------- | ---------- | ---------- | ----------------- |
| + [type](#network_g_anyOf_i58_type ) | No      | enum (of string) | No         | -          | -                 |

#### <a name="network_g_anyOf_i58_type"></a>5.59.1. Property `ReduxOptions > network_g > anyOf > esrgan > type`

|              |                    |
| ------------ | ------------------ |
| **Type**     | `enum (of string)` |
| **Required** | Yes                |

Must be one of:
* "esrgan"

### <a name="network_g_anyOf_i59"></a>5.60. Property `ReduxOptions > network_g > anyOf > esrgan_lite`

**Title:** esrgan_lite

|                           |                     |
| ------------------------- | ------------------- |
| **Type**                  | `object`            |
| **Required**              | No                  |
| **Additional properties** | Any type allowed    |
| **Defined in**            | #/$defs/esrgan_lite |

| Property                             | Pattern | Type             | Deprecated | Definition | Title/Description |
| ------------------------------------ | ------- | ---------------- | ---------- | ---------- | ----------------- |
| + [type](#network_g_anyOf_i59_type ) | No      | enum (of string) | No         | -          | -                 |

#### <a name="network_g_anyOf_i59_type"></a>5.60.1. Property `ReduxOptions > network_g > anyOf > esrgan_lite > type`

|              |                    |
| ------------ | ------------------ |
| **Type**     | `enum (of string)` |
| **Required** | Yes                |

Must be one of:
* "esrgan_lite"

### <a name="network_g_anyOf_i60"></a>5.61. Property `ReduxOptions > network_g > anyOf > span`

**Title:** span

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `object`         |
| **Required**              | No               |
| **Additional properties** | Any type allowed |
| **Defined in**            | #/$defs/span     |

| Property                             | Pattern | Type             | Deprecated | Definition | Title/Description |
| ------------------------------------ | ------- | ---------------- | ---------- | ---------- | ----------------- |
| + [type](#network_g_anyOf_i60_type ) | No      | enum (of string) | No         | -          | -                 |

#### <a name="network_g_anyOf_i60_type"></a>5.61.1. Property `ReduxOptions > network_g > anyOf > span > type`

|              |                    |
| ------------ | ------------------ |
| **Type**     | `enum (of string)` |
| **Required** | Yes                |

Must be one of:
* "span"

### <a name="network_g_anyOf_i61"></a>5.62. Property `ReduxOptions > network_g > anyOf > srformer`

**Title:** srformer

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `object`         |
| **Required**              | No               |
| **Additional properties** | Any type allowed |
| **Defined in**            | #/$defs/srformer |

| Property                             | Pattern | Type             | Deprecated | Definition | Title/Description |
| ------------------------------------ | ------- | ---------------- | ---------- | ---------- | ----------------- |
| + [type](#network_g_anyOf_i61_type ) | No      | enum (of string) | No         | -          | -                 |

#### <a name="network_g_anyOf_i61_type"></a>5.62.1. Property `ReduxOptions > network_g > anyOf > srformer > type`

|              |                    |
| ------------ | ------------------ |
| **Type**     | `enum (of string)` |
| **Required** | Yes                |

Must be one of:
* "srformer"

### <a name="network_g_anyOf_i62"></a>5.63. Property `ReduxOptions > network_g > anyOf > srformer_light`

**Title:** srformer_light

|                           |                        |
| ------------------------- | ---------------------- |
| **Type**                  | `object`               |
| **Required**              | No                     |
| **Additional properties** | Any type allowed       |
| **Defined in**            | #/$defs/srformer_light |

| Property                             | Pattern | Type             | Deprecated | Definition | Title/Description |
| ------------------------------------ | ------- | ---------------- | ---------- | ---------- | ----------------- |
| + [type](#network_g_anyOf_i62_type ) | No      | enum (of string) | No         | -          | -                 |

#### <a name="network_g_anyOf_i62_type"></a>5.63.1. Property `ReduxOptions > network_g > anyOf > srformer_light > type`

|              |                    |
| ------------ | ------------------ |
| **Type**     | `enum (of string)` |
| **Required** | Yes                |

Must be one of:
* "srformer_light"

### <a name="network_g_anyOf_i63"></a>5.64. Property `ReduxOptions > network_g > anyOf > swin2sr`

**Title:** swin2sr

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `object`         |
| **Required**              | No               |
| **Additional properties** | Any type allowed |
| **Defined in**            | #/$defs/swin2sr  |

| Property                             | Pattern | Type             | Deprecated | Definition | Title/Description |
| ------------------------------------ | ------- | ---------------- | ---------- | ---------- | ----------------- |
| + [type](#network_g_anyOf_i63_type ) | No      | enum (of string) | No         | -          | -                 |

#### <a name="network_g_anyOf_i63_type"></a>5.64.1. Property `ReduxOptions > network_g > anyOf > swin2sr > type`

|              |                    |
| ------------ | ------------------ |
| **Type**     | `enum (of string)` |
| **Required** | Yes                |

Must be one of:
* "swin2sr"

### <a name="network_g_anyOf_i64"></a>5.65. Property `ReduxOptions > network_g > anyOf > swin2sr_l`

**Title:** swin2sr_l

|                           |                   |
| ------------------------- | ----------------- |
| **Type**                  | `object`          |
| **Required**              | No                |
| **Additional properties** | Any type allowed  |
| **Defined in**            | #/$defs/swin2sr_l |

| Property                             | Pattern | Type             | Deprecated | Definition | Title/Description |
| ------------------------------------ | ------- | ---------------- | ---------- | ---------- | ----------------- |
| + [type](#network_g_anyOf_i64_type ) | No      | enum (of string) | No         | -          | -                 |

#### <a name="network_g_anyOf_i64_type"></a>5.65.1. Property `ReduxOptions > network_g > anyOf > swin2sr_l > type`

|              |                    |
| ------------ | ------------------ |
| **Type**     | `enum (of string)` |
| **Required** | Yes                |

Must be one of:
* "swin2sr_l"

### <a name="network_g_anyOf_i65"></a>5.66. Property `ReduxOptions > network_g > anyOf > swin2sr_m`

**Title:** swin2sr_m

|                           |                   |
| ------------------------- | ----------------- |
| **Type**                  | `object`          |
| **Required**              | No                |
| **Additional properties** | Any type allowed  |
| **Defined in**            | #/$defs/swin2sr_m |

| Property                             | Pattern | Type             | Deprecated | Definition | Title/Description |
| ------------------------------------ | ------- | ---------------- | ---------- | ---------- | ----------------- |
| + [type](#network_g_anyOf_i65_type ) | No      | enum (of string) | No         | -          | -                 |

#### <a name="network_g_anyOf_i65_type"></a>5.66.1. Property `ReduxOptions > network_g > anyOf > swin2sr_m > type`

|              |                    |
| ------------ | ------------------ |
| **Type**     | `enum (of string)` |
| **Required** | Yes                |

Must be one of:
* "swin2sr_m"

### <a name="network_g_anyOf_i66"></a>5.67. Property `ReduxOptions > network_g > anyOf > swin2sr_s`

**Title:** swin2sr_s

|                           |                   |
| ------------------------- | ----------------- |
| **Type**                  | `object`          |
| **Required**              | No                |
| **Additional properties** | Any type allowed  |
| **Defined in**            | #/$defs/swin2sr_s |

| Property                             | Pattern | Type             | Deprecated | Definition | Title/Description |
| ------------------------------------ | ------- | ---------------- | ---------- | ---------- | ----------------- |
| + [type](#network_g_anyOf_i66_type ) | No      | enum (of string) | No         | -          | -                 |

#### <a name="network_g_anyOf_i66_type"></a>5.67.1. Property `ReduxOptions > network_g > anyOf > swin2sr_s > type`

|              |                    |
| ------------ | ------------------ |
| **Type**     | `enum (of string)` |
| **Required** | Yes                |

Must be one of:
* "swin2sr_s"

### <a name="network_g_anyOf_i67"></a>5.68. Property `ReduxOptions > network_g > anyOf > swinir`

**Title:** swinir

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `object`         |
| **Required**              | No               |
| **Additional properties** | Any type allowed |
| **Defined in**            | #/$defs/swinir   |

| Property                             | Pattern | Type             | Deprecated | Definition | Title/Description |
| ------------------------------------ | ------- | ---------------- | ---------- | ---------- | ----------------- |
| + [type](#network_g_anyOf_i67_type ) | No      | enum (of string) | No         | -          | -                 |

#### <a name="network_g_anyOf_i67_type"></a>5.68.1. Property `ReduxOptions > network_g > anyOf > swinir > type`

|              |                    |
| ------------ | ------------------ |
| **Type**     | `enum (of string)` |
| **Required** | Yes                |

Must be one of:
* "swinir"

### <a name="network_g_anyOf_i68"></a>5.69. Property `ReduxOptions > network_g > anyOf > swinir_l`

**Title:** swinir_l

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `object`         |
| **Required**              | No               |
| **Additional properties** | Any type allowed |
| **Defined in**            | #/$defs/swinir_l |

| Property                             | Pattern | Type             | Deprecated | Definition | Title/Description |
| ------------------------------------ | ------- | ---------------- | ---------- | ---------- | ----------------- |
| + [type](#network_g_anyOf_i68_type ) | No      | enum (of string) | No         | -          | -                 |

#### <a name="network_g_anyOf_i68_type"></a>5.69.1. Property `ReduxOptions > network_g > anyOf > swinir_l > type`

|              |                    |
| ------------ | ------------------ |
| **Type**     | `enum (of string)` |
| **Required** | Yes                |

Must be one of:
* "swinir_l"

### <a name="network_g_anyOf_i69"></a>5.70. Property `ReduxOptions > network_g > anyOf > swinir_m`

**Title:** swinir_m

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `object`         |
| **Required**              | No               |
| **Additional properties** | Any type allowed |
| **Defined in**            | #/$defs/swinir_m |

| Property                             | Pattern | Type             | Deprecated | Definition | Title/Description |
| ------------------------------------ | ------- | ---------------- | ---------- | ---------- | ----------------- |
| + [type](#network_g_anyOf_i69_type ) | No      | enum (of string) | No         | -          | -                 |

#### <a name="network_g_anyOf_i69_type"></a>5.70.1. Property `ReduxOptions > network_g > anyOf > swinir_m > type`

|              |                    |
| ------------ | ------------------ |
| **Type**     | `enum (of string)` |
| **Required** | Yes                |

Must be one of:
* "swinir_m"

### <a name="network_g_anyOf_i70"></a>5.71. Property `ReduxOptions > network_g > anyOf > swinir_s`

**Title:** swinir_s

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `object`         |
| **Required**              | No               |
| **Additional properties** | Any type allowed |
| **Defined in**            | #/$defs/swinir_s |

| Property                             | Pattern | Type             | Deprecated | Definition | Title/Description |
| ------------------------------------ | ------- | ---------------- | ---------- | ---------- | ----------------- |
| + [type](#network_g_anyOf_i70_type ) | No      | enum (of string) | No         | -          | -                 |

#### <a name="network_g_anyOf_i70_type"></a>5.71.1. Property `ReduxOptions > network_g > anyOf > swinir_s > type`

|              |                    |
| ------------ | ------------------ |
| **Type**     | `enum (of string)` |
| **Required** | Yes                |

Must be one of:
* "swinir_s"

## <a name="network_d"></a>6. Property `ReduxOptions > network_d`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `combining`      |
| **Required**              | No               |
| **Additional properties** | Any type allowed |
| **Default**               | `null`           |

**Description:** The options for the discriminator model.

| Any of(Option)                |
| ----------------------------- |
| [item 0](#network_d_anyOf_i0) |
| [item 1](#network_d_anyOf_i1) |

### <a name="network_d_anyOf_i0"></a>6.1. Property `ReduxOptions > network_d > anyOf > item 0`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `object`         |
| **Required**              | No               |
| **Additional properties** | Any type allowed |

### <a name="network_d_anyOf_i1"></a>6.2. Property `ReduxOptions > network_d > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

## <a name="manual_seed"></a>7. Property `ReduxOptions > manual_seed`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `combining`      |
| **Required**              | No               |
| **Additional properties** | Any type allowed |
| **Default**               | `null`           |

**Description:** Deterministic mode, slows down training. Only use for reproducible experiments.

| Any of(Option)                  |
| ------------------------------- |
| [item 0](#manual_seed_anyOf_i0) |
| [item 1](#manual_seed_anyOf_i1) |

### <a name="manual_seed_anyOf_i0"></a>7.1. Property `ReduxOptions > manual_seed > anyOf > item 0`

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | No        |

### <a name="manual_seed_anyOf_i1"></a>7.2. Property `ReduxOptions > manual_seed > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

## <a name="deterministic"></a>8. Property `ReduxOptions > deterministic`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `combining`      |
| **Required**              | No               |
| **Additional properties** | Any type allowed |
| **Default**               | `null`           |

| Any of(Option)                    |
| --------------------------------- |
| [item 0](#deterministic_anyOf_i0) |
| [item 1](#deterministic_anyOf_i1) |

### <a name="deterministic_anyOf_i0"></a>8.1. Property `ReduxOptions > deterministic > anyOf > item 0`

|              |           |
| ------------ | --------- |
| **Type**     | `boolean` |
| **Required** | No        |

### <a name="deterministic_anyOf_i1"></a>8.2. Property `ReduxOptions > deterministic > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

## <a name="dist"></a>9. Property `ReduxOptions > dist`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `combining`      |
| **Required**              | No               |
| **Additional properties** | Any type allowed |
| **Default**               | `null`           |

| Any of(Option)           |
| ------------------------ |
| [item 0](#dist_anyOf_i0) |
| [item 1](#dist_anyOf_i1) |

### <a name="dist_anyOf_i0"></a>9.1. Property `ReduxOptions > dist > anyOf > item 0`

|              |           |
| ------------ | --------- |
| **Type**     | `boolean` |
| **Required** | No        |

### <a name="dist_anyOf_i1"></a>9.2. Property `ReduxOptions > dist > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

## <a name="launcher"></a>10. Property `ReduxOptions > launcher`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `combining`      |
| **Required**              | No               |
| **Additional properties** | Any type allowed |
| **Default**               | `null`           |

| Any of(Option)               |
| ---------------------------- |
| [item 0](#launcher_anyOf_i0) |
| [item 1](#launcher_anyOf_i1) |

### <a name="launcher_anyOf_i0"></a>10.1. Property `ReduxOptions > launcher > anyOf > item 0`

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | No       |

### <a name="launcher_anyOf_i1"></a>10.2. Property `ReduxOptions > launcher > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

## <a name="rank"></a>11. Property `ReduxOptions > rank`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `combining`      |
| **Required**              | No               |
| **Additional properties** | Any type allowed |
| **Default**               | `null`           |

| Any of(Option)           |
| ------------------------ |
| [item 0](#rank_anyOf_i0) |
| [item 1](#rank_anyOf_i1) |

### <a name="rank_anyOf_i0"></a>11.1. Property `ReduxOptions > rank > anyOf > item 0`

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | No        |

### <a name="rank_anyOf_i1"></a>11.2. Property `ReduxOptions > rank > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

## <a name="world_size"></a>12. Property `ReduxOptions > world_size`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `combining`      |
| **Required**              | No               |
| **Additional properties** | Any type allowed |
| **Default**               | `null`           |

| Any of(Option)                 |
| ------------------------------ |
| [item 0](#world_size_anyOf_i0) |
| [item 1](#world_size_anyOf_i1) |

### <a name="world_size_anyOf_i0"></a>12.1. Property `ReduxOptions > world_size > anyOf > item 0`

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | No        |

### <a name="world_size_anyOf_i1"></a>12.2. Property `ReduxOptions > world_size > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

## <a name="auto_resume"></a>13. Property `ReduxOptions > auto_resume`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `combining`      |
| **Required**              | No               |
| **Additional properties** | Any type allowed |
| **Default**               | `null`           |

| Any of(Option)                  |
| ------------------------------- |
| [item 0](#auto_resume_anyOf_i0) |
| [item 1](#auto_resume_anyOf_i1) |

### <a name="auto_resume_anyOf_i0"></a>13.1. Property `ReduxOptions > auto_resume > anyOf > item 0`

|              |           |
| ------------ | --------- |
| **Type**     | `boolean` |
| **Required** | No        |

### <a name="auto_resume_anyOf_i1"></a>13.2. Property `ReduxOptions > auto_resume > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

## <a name="resume"></a>14. Property `ReduxOptions > resume`

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | No        |
| **Default**  | `0`       |

## <a name="is_train"></a>15. Property `ReduxOptions > is_train`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `combining`      |
| **Required**              | No               |
| **Additional properties** | Any type allowed |
| **Default**               | `null`           |

| Any of(Option)               |
| ---------------------------- |
| [item 0](#is_train_anyOf_i0) |
| [item 1](#is_train_anyOf_i1) |

### <a name="is_train_anyOf_i0"></a>15.1. Property `ReduxOptions > is_train > anyOf > item 0`

|              |           |
| ------------ | --------- |
| **Type**     | `boolean` |
| **Required** | No        |

### <a name="is_train_anyOf_i1"></a>15.2. Property `ReduxOptions > is_train > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

## <a name="root_path"></a>16. Property `ReduxOptions > root_path`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `combining`      |
| **Required**              | No               |
| **Additional properties** | Any type allowed |
| **Default**               | `null`           |

| Any of(Option)                |
| ----------------------------- |
| [item 0](#root_path_anyOf_i0) |
| [item 1](#root_path_anyOf_i1) |

### <a name="root_path_anyOf_i0"></a>16.1. Property `ReduxOptions > root_path > anyOf > item 0`

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | No       |

### <a name="root_path_anyOf_i1"></a>16.2. Property `ReduxOptions > root_path > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

## <a name="use_amp"></a>17. Property `ReduxOptions > use_amp`

|              |           |
| ------------ | --------- |
| **Type**     | `boolean` |
| **Required** | No        |
| **Default**  | `false`   |

**Description:** Speed up training and reduce VRAM usage. NVIDIA only.

## <a name="amp_bf16"></a>18. Property `ReduxOptions > amp_bf16`

|              |           |
| ------------ | --------- |
| **Type**     | `boolean` |
| **Required** | No        |
| **Default**  | `false`   |

**Description:** Use bf16 instead of fp16 for AMP, RTX 3000 series or newer only. Only recommended if fp16 doesn't work.

## <a name="use_channels_last"></a>19. Property `ReduxOptions > use_channels_last`

|              |           |
| ------------ | --------- |
| **Type**     | `boolean` |
| **Required** | No        |
| **Default**  | `true`    |

**Description:** Enable channels last memory format while using AMP. Reduces VRAM and speeds up training for most architectures, but some architectures are slower with channels last.

## <a name="fast_matmul"></a>20. Property `ReduxOptions > fast_matmul`

|              |           |
| ------------ | --------- |
| **Type**     | `boolean` |
| **Required** | No        |
| **Default**  | `false`   |

**Description:** Trade precision for performance.

## <a name="use_compile"></a>21. Property `ReduxOptions > use_compile`

|              |           |
| ------------ | --------- |
| **Type**     | `boolean` |
| **Required** | No        |
| **Default**  | `false`   |

**Description:** Enable torch.compile for the generator model, which takes time on startup to compile the model, but can speed up training after the model is compiled.

## <a name="detect_anomaly"></a>22. Property `ReduxOptions > detect_anomaly`

|              |           |
| ------------ | --------- |
| **Type**     | `boolean` |
| **Required** | No        |
| **Default**  | `false`   |

**Description:** Whether or not to enable anomaly detection, which can be useful for debugging NaNs that occur during training. Has a significant performance hit and should be disabled when not debugging.

## <a name="high_order_degradation"></a>23. Property `ReduxOptions > high_order_degradation`

|              |           |
| ------------ | --------- |
| **Type**     | `boolean` |
| **Required** | No        |
| **Default**  | `false`   |

**Description:** Whether or not to enable OTF (on the fly) degradations, which generates LRs on the fly.

## <a name="high_order_degradations_debug"></a>24. Property `ReduxOptions > high_order_degradations_debug`

|              |           |
| ------------ | --------- |
| **Type**     | `boolean` |
| **Required** | No        |
| **Default**  | `false`   |

**Description:** Whether or not to enable debugging for OTF, which saves the OTF generated LR images so they can be inspected to view the effect of different OTF settings.

## <a name="high_order_degradations_debug_limit"></a>25. Property `ReduxOptions > high_order_degradations_debug_limit`

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | No        |
| **Default**  | `100`     |

**Description:** The maximum number of OTF images to save when debugging is enabled.

## <a name="dataroot_lq_prob"></a>26. Property `ReduxOptions > dataroot_lq_prob`

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |
| **Default**  | `0`      |

**Description:** Probability of using paired LR data instead of OTF LR data.

## <a name="lq_usm"></a>27. Property `ReduxOptions > lq_usm`

|              |           |
| ------------ | --------- |
| **Type**     | `boolean` |
| **Required** | No        |
| **Default**  | `false`   |

## <a name="lq_usm_radius_range"></a>28. Property `ReduxOptions > lq_usm_radius_range`

|              |           |
| ------------ | --------- |
| **Type**     | `array`   |
| **Required** | No        |
| **Default**  | `[1, 25]` |

**Description:** For the unsharp mask of the LQ image, use a radius randomly selected from this range.

|                      | Array restrictions |
| -------------------- | ------------------ |
| **Min items**        | 2                  |
| **Max items**        | 2                  |
| **Items unicity**    | False              |
| **Additional items** | False              |
| **Tuple validation** | See below          |

| Each item of this array must be                             | Description |
| ----------------------------------------------------------- | ----------- |
| [lq_usm_radius_range item 0](#lq_usm_radius_range_items_i0) | -           |
| [lq_usm_radius_range item 1](#lq_usm_radius_range_items_i1) | -           |

### <a name="autogenerated_heading_2"></a>28.1. ReduxOptions > lq_usm_radius_range > lq_usm_radius_range item 0

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | No        |

### <a name="autogenerated_heading_3"></a>28.2. ReduxOptions > lq_usm_radius_range > lq_usm_radius_range item 1

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | No        |

## <a name="blur_prob"></a>29. Property `ReduxOptions > blur_prob`

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |
| **Default**  | `0`      |

## <a name="resize_prob"></a>30. Property `ReduxOptions > resize_prob`

|              |                   |
| ------------ | ----------------- |
| **Type**     | `array of number` |
| **Required** | No                |

|                      | Array restrictions |
| -------------------- | ------------------ |
| **Min items**        | N/A                |
| **Max items**        | N/A                |
| **Items unicity**    | False              |
| **Additional items** | False              |
| **Tuple validation** | See below          |

| Each item of this array must be         | Description |
| --------------------------------------- | ----------- |
| [resize_prob items](#resize_prob_items) | -           |

### <a name="resize_prob_items"></a>30.1. ReduxOptions > resize_prob > resize_prob items

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |

## <a name="resize_mode_list"></a>31. Property `ReduxOptions > resize_mode_list`

|              |                   |
| ------------ | ----------------- |
| **Type**     | `array of string` |
| **Required** | No                |

|                      | Array restrictions |
| -------------------- | ------------------ |
| **Min items**        | N/A                |
| **Max items**        | N/A                |
| **Items unicity**    | False              |
| **Additional items** | False              |
| **Tuple validation** | See below          |

| Each item of this array must be                   | Description |
| ------------------------------------------------- | ----------- |
| [resize_mode_list items](#resize_mode_list_items) | -           |

### <a name="resize_mode_list_items"></a>31.1. ReduxOptions > resize_mode_list > resize_mode_list items

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | No       |

## <a name="resize_mode_prob"></a>32. Property `ReduxOptions > resize_mode_prob`

|              |                   |
| ------------ | ----------------- |
| **Type**     | `array of number` |
| **Required** | No                |

|                      | Array restrictions |
| -------------------- | ------------------ |
| **Min items**        | N/A                |
| **Max items**        | N/A                |
| **Items unicity**    | False              |
| **Additional items** | False              |
| **Tuple validation** | See below          |

| Each item of this array must be                   | Description |
| ------------------------------------------------- | ----------- |
| [resize_mode_prob items](#resize_mode_prob_items) | -           |

### <a name="resize_mode_prob_items"></a>32.1. ReduxOptions > resize_mode_prob > resize_mode_prob items

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |

## <a name="resize_range"></a>33. Property `ReduxOptions > resize_range`

|              |              |
| ------------ | ------------ |
| **Type**     | `array`      |
| **Required** | No           |
| **Default**  | `[0.4, 1.5]` |

|                      | Array restrictions |
| -------------------- | ------------------ |
| **Min items**        | 2                  |
| **Max items**        | 2                  |
| **Items unicity**    | False              |
| **Additional items** | False              |
| **Tuple validation** | See below          |

| Each item of this array must be               | Description |
| --------------------------------------------- | ----------- |
| [resize_range item 0](#resize_range_items_i0) | -           |
| [resize_range item 1](#resize_range_items_i1) | -           |

### <a name="autogenerated_heading_4"></a>33.1. ReduxOptions > resize_range > resize_range item 0

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |

### <a name="autogenerated_heading_5"></a>33.2. ReduxOptions > resize_range > resize_range item 1

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |

## <a name="gaussian_noise_prob"></a>34. Property `ReduxOptions > gaussian_noise_prob`

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |
| **Default**  | `0`      |

## <a name="noise_range"></a>35. Property `ReduxOptions > noise_range`

|              |          |
| ------------ | -------- |
| **Type**     | `array`  |
| **Required** | No       |
| **Default**  | `[0, 0]` |

|                      | Array restrictions |
| -------------------- | ------------------ |
| **Min items**        | 2                  |
| **Max items**        | 2                  |
| **Items unicity**    | False              |
| **Additional items** | False              |
| **Tuple validation** | See below          |

| Each item of this array must be             | Description |
| ------------------------------------------- | ----------- |
| [noise_range item 0](#noise_range_items_i0) | -           |
| [noise_range item 1](#noise_range_items_i1) | -           |

### <a name="autogenerated_heading_6"></a>35.1. ReduxOptions > noise_range > noise_range item 0

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |

### <a name="autogenerated_heading_7"></a>35.2. ReduxOptions > noise_range > noise_range item 1

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |

## <a name="poisson_scale_range"></a>36. Property `ReduxOptions > poisson_scale_range`

|              |          |
| ------------ | -------- |
| **Type**     | `array`  |
| **Required** | No       |
| **Default**  | `[0, 0]` |

|                      | Array restrictions |
| -------------------- | ------------------ |
| **Min items**        | 2                  |
| **Max items**        | 2                  |
| **Items unicity**    | False              |
| **Additional items** | False              |
| **Tuple validation** | See below          |

| Each item of this array must be                             | Description |
| ----------------------------------------------------------- | ----------- |
| [poisson_scale_range item 0](#poisson_scale_range_items_i0) | -           |
| [poisson_scale_range item 1](#poisson_scale_range_items_i1) | -           |

### <a name="autogenerated_heading_8"></a>36.1. ReduxOptions > poisson_scale_range > poisson_scale_range item 0

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |

### <a name="autogenerated_heading_9"></a>36.2. ReduxOptions > poisson_scale_range > poisson_scale_range item 1

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |

## <a name="gray_noise_prob"></a>37. Property `ReduxOptions > gray_noise_prob`

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |
| **Default**  | `0`      |

## <a name="jpeg_prob"></a>38. Property `ReduxOptions > jpeg_prob`

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |
| **Default**  | `1`      |

## <a name="jpeg_range"></a>39. Property `ReduxOptions > jpeg_range`

|              |            |
| ------------ | ---------- |
| **Type**     | `array`    |
| **Required** | No         |
| **Default**  | `[75, 95]` |

|                      | Array restrictions |
| -------------------- | ------------------ |
| **Min items**        | 2                  |
| **Max items**        | 2                  |
| **Items unicity**    | False              |
| **Additional items** | False              |
| **Tuple validation** | See below          |

| Each item of this array must be           | Description |
| ----------------------------------------- | ----------- |
| [jpeg_range item 0](#jpeg_range_items_i0) | -           |
| [jpeg_range item 1](#jpeg_range_items_i1) | -           |

### <a name="autogenerated_heading_10"></a>39.1. ReduxOptions > jpeg_range > jpeg_range item 0

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |

### <a name="autogenerated_heading_11"></a>39.2. ReduxOptions > jpeg_range > jpeg_range item 1

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |

## <a name="blur_prob2"></a>40. Property `ReduxOptions > blur_prob2`

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |
| **Default**  | `0`      |

## <a name="resize_prob2"></a>41. Property `ReduxOptions > resize_prob2`

|              |                   |
| ------------ | ----------------- |
| **Type**     | `array of number` |
| **Required** | No                |

|                      | Array restrictions |
| -------------------- | ------------------ |
| **Min items**        | N/A                |
| **Max items**        | N/A                |
| **Items unicity**    | False              |
| **Additional items** | False              |
| **Tuple validation** | See below          |

| Each item of this array must be           | Description |
| ----------------------------------------- | ----------- |
| [resize_prob2 items](#resize_prob2_items) | -           |

### <a name="resize_prob2_items"></a>41.1. ReduxOptions > resize_prob2 > resize_prob2 items

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |

## <a name="resize_mode_list2"></a>42. Property `ReduxOptions > resize_mode_list2`

|              |                   |
| ------------ | ----------------- |
| **Type**     | `array of string` |
| **Required** | No                |

|                      | Array restrictions |
| -------------------- | ------------------ |
| **Min items**        | N/A                |
| **Max items**        | N/A                |
| **Items unicity**    | False              |
| **Additional items** | False              |
| **Tuple validation** | See below          |

| Each item of this array must be                     | Description |
| --------------------------------------------------- | ----------- |
| [resize_mode_list2 items](#resize_mode_list2_items) | -           |

### <a name="resize_mode_list2_items"></a>42.1. ReduxOptions > resize_mode_list2 > resize_mode_list2 items

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | No       |

## <a name="resize_mode_prob2"></a>43. Property `ReduxOptions > resize_mode_prob2`

|              |                   |
| ------------ | ----------------- |
| **Type**     | `array of number` |
| **Required** | No                |

|                      | Array restrictions |
| -------------------- | ------------------ |
| **Min items**        | N/A                |
| **Max items**        | N/A                |
| **Items unicity**    | False              |
| **Additional items** | False              |
| **Tuple validation** | See below          |

| Each item of this array must be                     | Description |
| --------------------------------------------------- | ----------- |
| [resize_mode_prob2 items](#resize_mode_prob2_items) | -           |

### <a name="resize_mode_prob2_items"></a>43.1. ReduxOptions > resize_mode_prob2 > resize_mode_prob2 items

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |

## <a name="resize_range2"></a>44. Property `ReduxOptions > resize_range2`

|              |              |
| ------------ | ------------ |
| **Type**     | `array`      |
| **Required** | No           |
| **Default**  | `[0.6, 1.2]` |

|                      | Array restrictions |
| -------------------- | ------------------ |
| **Min items**        | 2                  |
| **Max items**        | 2                  |
| **Items unicity**    | False              |
| **Additional items** | False              |
| **Tuple validation** | See below          |

| Each item of this array must be                 | Description |
| ----------------------------------------------- | ----------- |
| [resize_range2 item 0](#resize_range2_items_i0) | -           |
| [resize_range2 item 1](#resize_range2_items_i1) | -           |

### <a name="autogenerated_heading_12"></a>44.1. ReduxOptions > resize_range2 > resize_range2 item 0

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |

### <a name="autogenerated_heading_13"></a>44.2. ReduxOptions > resize_range2 > resize_range2 item 1

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |

## <a name="gaussian_noise_prob2"></a>45. Property `ReduxOptions > gaussian_noise_prob2`

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |
| **Default**  | `0`      |

## <a name="noise_range2"></a>46. Property `ReduxOptions > noise_range2`

|              |          |
| ------------ | -------- |
| **Type**     | `array`  |
| **Required** | No       |
| **Default**  | `[0, 0]` |

|                      | Array restrictions |
| -------------------- | ------------------ |
| **Min items**        | 2                  |
| **Max items**        | 2                  |
| **Items unicity**    | False              |
| **Additional items** | False              |
| **Tuple validation** | See below          |

| Each item of this array must be               | Description |
| --------------------------------------------- | ----------- |
| [noise_range2 item 0](#noise_range2_items_i0) | -           |
| [noise_range2 item 1](#noise_range2_items_i1) | -           |

### <a name="autogenerated_heading_14"></a>46.1. ReduxOptions > noise_range2 > noise_range2 item 0

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |

### <a name="autogenerated_heading_15"></a>46.2. ReduxOptions > noise_range2 > noise_range2 item 1

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |

## <a name="poisson_scale_range2"></a>47. Property `ReduxOptions > poisson_scale_range2`

|              |          |
| ------------ | -------- |
| **Type**     | `array`  |
| **Required** | No       |
| **Default**  | `[0, 0]` |

|                      | Array restrictions |
| -------------------- | ------------------ |
| **Min items**        | 2                  |
| **Max items**        | 2                  |
| **Items unicity**    | False              |
| **Additional items** | False              |
| **Tuple validation** | See below          |

| Each item of this array must be                               | Description |
| ------------------------------------------------------------- | ----------- |
| [poisson_scale_range2 item 0](#poisson_scale_range2_items_i0) | -           |
| [poisson_scale_range2 item 1](#poisson_scale_range2_items_i1) | -           |

### <a name="autogenerated_heading_16"></a>47.1. ReduxOptions > poisson_scale_range2 > poisson_scale_range2 item 0

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |

### <a name="autogenerated_heading_17"></a>47.2. ReduxOptions > poisson_scale_range2 > poisson_scale_range2 item 1

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |

## <a name="gray_noise_prob2"></a>48. Property `ReduxOptions > gray_noise_prob2`

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |
| **Default**  | `0`      |

## <a name="jpeg_prob2"></a>49. Property `ReduxOptions > jpeg_prob2`

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |
| **Default**  | `1`      |

## <a name="jpeg_range2"></a>50. Property `ReduxOptions > jpeg_range2`

|              |                   |
| ------------ | ----------------- |
| **Type**     | `array of number` |
| **Required** | No                |

|                      | Array restrictions |
| -------------------- | ------------------ |
| **Min items**        | N/A                |
| **Max items**        | N/A                |
| **Items unicity**    | False              |
| **Additional items** | False              |
| **Tuple validation** | See below          |

| Each item of this array must be         | Description |
| --------------------------------------- | ----------- |
| [jpeg_range2 items](#jpeg_range2_items) | -           |

### <a name="jpeg_range2_items"></a>50.1. ReduxOptions > jpeg_range2 > jpeg_range2 items

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |

## <a name="resize_mode_list3"></a>51. Property `ReduxOptions > resize_mode_list3`

|              |                   |
| ------------ | ----------------- |
| **Type**     | `array of string` |
| **Required** | No                |

|                      | Array restrictions |
| -------------------- | ------------------ |
| **Min items**        | N/A                |
| **Max items**        | N/A                |
| **Items unicity**    | False              |
| **Additional items** | False              |
| **Tuple validation** | See below          |

| Each item of this array must be                     | Description |
| --------------------------------------------------- | ----------- |
| [resize_mode_list3 items](#resize_mode_list3_items) | -           |

### <a name="resize_mode_list3_items"></a>51.1. ReduxOptions > resize_mode_list3 > resize_mode_list3 items

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | No       |

## <a name="resize_mode_prob3"></a>52. Property `ReduxOptions > resize_mode_prob3`

|              |                   |
| ------------ | ----------------- |
| **Type**     | `array of number` |
| **Required** | No                |

|                      | Array restrictions |
| -------------------- | ------------------ |
| **Min items**        | N/A                |
| **Max items**        | N/A                |
| **Items unicity**    | False              |
| **Additional items** | False              |
| **Tuple validation** | See below          |

| Each item of this array must be                     | Description |
| --------------------------------------------------- | ----------- |
| [resize_mode_prob3 items](#resize_mode_prob3_items) | -           |

### <a name="resize_mode_prob3_items"></a>52.1. ReduxOptions > resize_mode_prob3 > resize_mode_prob3 items

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |

## <a name="queue_size"></a>53. Property `ReduxOptions > queue_size`

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | No        |
| **Default**  | `180`     |

## <a name="datasets"></a>54. Property `ReduxOptions > datasets`

|                           |                                                                                       |
| ------------------------- | ------------------------------------------------------------------------------------- |
| **Type**                  | `object`                                                                              |
| **Required**              | No                                                                                    |
| **Additional properties** | [Each additional property must conform to the schema](#datasets_additionalProperties) |
| **Default**               | `{}`                                                                                  |

| Property                              | Pattern | Type   | Deprecated | Definition                | Title/Description |
| ------------------------------------- | ------- | ------ | ---------- | ------------------------- | ----------------- |
| - [](#datasets_additionalProperties ) | No      | object | No         | In #/$defs/DatasetOptions | DatasetOptions    |

### <a name="datasets_additionalProperties"></a>54.1. Property `ReduxOptions > datasets > DatasetOptions`

**Title:** DatasetOptions

|                           |                        |
| ------------------------- | ---------------------- |
| **Type**                  | `object`               |
| **Required**              | No                     |
| **Additional properties** | Not allowed            |
| **Defined in**            | #/$defs/DatasetOptions |

| Property                                                                         | Pattern | Type            | Deprecated | Definition | Title/Description                                                                                                                                                                                                                                                                           |
| -------------------------------------------------------------------------------- | ------- | --------------- | ---------- | ---------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| + [name](#datasets_additionalProperties_name )                                   | No      | string          | No         | -          | Name of the dataset. It should be unique compared to other datasets in this config, but the exact name isn't very important.                                                                                                                                                                |
| + [type](#datasets_additionalProperties_type )                                   | No      | string          | No         | -          | -                                                                                                                                                                                                                                                                                           |
| + [io_backend](#datasets_additionalProperties_io_backend )                       | No      | object          | No         | -          | -                                                                                                                                                                                                                                                                                           |
| - [num_worker_per_gpu](#datasets_additionalProperties_num_worker_per_gpu )       | No      | Combination     | No         | -          | Number of subprocesses to use for data loading with PyTorch dataloader.                                                                                                                                                                                                                     |
| - [batch_size_per_gpu](#datasets_additionalProperties_batch_size_per_gpu )       | No      | Combination     | No         | -          | Increasing stabilizes training but going too high can cause issues. Use multiple of 8 for best performance with AMP. A higher batch size, like 32 or 64 is more important when training from scratch, while smaller batches like 8 can be used when training with a quality pretrain model. |
| - [accum_iter](#datasets_additionalProperties_accum_iter )                       | No      | integer         | No         | -          | Using values larger than 1 simulates higher batch size by trading performance for reduced VRAM usage. If accum_iter = 4 and batch_size_per_gpu = 6 then effective batch size = 4 * 6 = 24 but performance may be as much as 4 times as slow.                                                |
| - [use_hflip](#datasets_additionalProperties_use_hflip )                         | No      | boolean         | No         | -          | Randomly flip the images horizontally.                                                                                                                                                                                                                                                      |
| - [use_rot](#datasets_additionalProperties_use_rot )                             | No      | boolean         | No         | -          | Randomly rotate the images.                                                                                                                                                                                                                                                                 |
| - [mean](#datasets_additionalProperties_mean )                                   | No      | Combination     | No         | -          | -                                                                                                                                                                                                                                                                                           |
| - [std](#datasets_additionalProperties_std )                                     | No      | Combination     | No         | -          | -                                                                                                                                                                                                                                                                                           |
| - [gt_size](#datasets_additionalProperties_gt_size )                             | No      | Combination     | No         | -          | -                                                                                                                                                                                                                                                                                           |
| - [lq_size](#datasets_additionalProperties_lq_size )                             | No      | Combination     | No         | -          | During training, a square of this size is cropped from LR images. Larger is usually better but uses more VRAM. Previously gt_size, use lq_size = gt_size / scale to convert. Use multiple of 8 for best performance with AMP.                                                               |
| - [color](#datasets_additionalProperties_color )                                 | No      | Combination     | No         | -          | -                                                                                                                                                                                                                                                                                           |
| - [phase](#datasets_additionalProperties_phase )                                 | No      | Combination     | No         | -          | -                                                                                                                                                                                                                                                                                           |
| - [scale](#datasets_additionalProperties_scale )                                 | No      | Combination     | No         | -          | -                                                                                                                                                                                                                                                                                           |
| - [dataset_enlarge_ratio](#datasets_additionalProperties_dataset_enlarge_ratio ) | No      | Combination     | No         | -          | Increase if the dataset is less than 1000 images to avoid slowdowns. Auto will automatically enlarge small datasets only.                                                                                                                                                                   |
| - [prefetch_mode](#datasets_additionalProperties_prefetch_mode )                 | No      | Combination     | No         | -          | -                                                                                                                                                                                                                                                                                           |
| - [pin_memory](#datasets_additionalProperties_pin_memory )                       | No      | boolean         | No         | -          | -                                                                                                                                                                                                                                                                                           |
| - [persistent_workers](#datasets_additionalProperties_persistent_workers )       | No      | boolean         | No         | -          | -                                                                                                                                                                                                                                                                                           |
| - [num_prefetch_queue](#datasets_additionalProperties_num_prefetch_queue )       | No      | integer         | No         | -          | -                                                                                                                                                                                                                                                                                           |
| - [prefetch_factor](#datasets_additionalProperties_prefetch_factor )             | No      | integer         | No         | -          | -                                                                                                                                                                                                                                                                                           |
| - [clip_size](#datasets_additionalProperties_clip_size )                         | No      | Combination     | No         | -          | -                                                                                                                                                                                                                                                                                           |
| - [dataroot_gt](#datasets_additionalProperties_dataroot_gt )                     | No      | Combination     | No         | -          | Path to the HR (high res) images in your training dataset. Specify one or multiple folders, separated by commas.                                                                                                                                                                            |
| - [dataroot_lq](#datasets_additionalProperties_dataroot_lq )                     | No      | Combination     | No         | -          | Path to the LR (low res) images in your training dataset. Specify one or multiple folders, separated by commas.                                                                                                                                                                             |
| - [meta_info](#datasets_additionalProperties_meta_info )                         | No      | Combination     | No         | -          | -                                                                                                                                                                                                                                                                                           |
| - [filename_tmpl](#datasets_additionalProperties_filename_tmpl )                 | No      | string          | No         | -          | -                                                                                                                                                                                                                                                                                           |
| - [blur_kernel_size](#datasets_additionalProperties_blur_kernel_size )           | No      | integer         | No         | -          | -                                                                                                                                                                                                                                                                                           |
| - [kernel_list](#datasets_additionalProperties_kernel_list )                     | No      | array of string | No         | -          | -                                                                                                                                                                                                                                                                                           |
| - [kernel_prob](#datasets_additionalProperties_kernel_prob )                     | No      | array of number | No         | -          | -                                                                                                                                                                                                                                                                                           |
| - [kernel_range](#datasets_additionalProperties_kernel_range )                   | No      | array           | No         | -          | -                                                                                                                                                                                                                                                                                           |
| - [sinc_prob](#datasets_additionalProperties_sinc_prob )                         | No      | number          | No         | -          | -                                                                                                                                                                                                                                                                                           |
| - [blur_sigma](#datasets_additionalProperties_blur_sigma )                       | No      | array           | No         | -          | -                                                                                                                                                                                                                                                                                           |
| - [betag_range](#datasets_additionalProperties_betag_range )                     | No      | array           | No         | -          | -                                                                                                                                                                                                                                                                                           |
| - [betap_range](#datasets_additionalProperties_betap_range )                     | No      | array           | No         | -          | -                                                                                                                                                                                                                                                                                           |
| - [blur_kernel_size2](#datasets_additionalProperties_blur_kernel_size2 )         | No      | integer         | No         | -          | -                                                                                                                                                                                                                                                                                           |
| - [kernel_list2](#datasets_additionalProperties_kernel_list2 )                   | No      | array of string | No         | -          | -                                                                                                                                                                                                                                                                                           |
| - [kernel_prob2](#datasets_additionalProperties_kernel_prob2 )                   | No      | array of number | No         | -          | -                                                                                                                                                                                                                                                                                           |
| - [kernel_range2](#datasets_additionalProperties_kernel_range2 )                 | No      | array           | No         | -          | -                                                                                                                                                                                                                                                                                           |
| - [sinc_prob2](#datasets_additionalProperties_sinc_prob2 )                       | No      | number          | No         | -          | -                                                                                                                                                                                                                                                                                           |
| - [blur_sigma2](#datasets_additionalProperties_blur_sigma2 )                     | No      | array           | No         | -          | -                                                                                                                                                                                                                                                                                           |
| - [betag_range2](#datasets_additionalProperties_betag_range2 )                   | No      | array           | No         | -          | -                                                                                                                                                                                                                                                                                           |
| - [betap_range2](#datasets_additionalProperties_betap_range2 )                   | No      | array           | No         | -          | -                                                                                                                                                                                                                                                                                           |
| - [final_sinc_prob](#datasets_additionalProperties_final_sinc_prob )             | No      | number          | No         | -          | -                                                                                                                                                                                                                                                                                           |
| - [final_kernel_range](#datasets_additionalProperties_final_kernel_range )       | No      | array           | No         | -          | -                                                                                                                                                                                                                                                                                           |

#### <a name="datasets_additionalProperties_name"></a>54.1.1. Property `ReduxOptions > datasets > DatasetOptions > name`

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | Yes      |

**Description:** Name of the dataset. It should be unique compared to other datasets in this config, but the exact name isn't very important.

#### <a name="datasets_additionalProperties_type"></a>54.1.2. Property `ReduxOptions > datasets > DatasetOptions > type`

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | Yes      |

#### <a name="datasets_additionalProperties_io_backend"></a>54.1.3. Property `ReduxOptions > datasets > DatasetOptions > io_backend`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `object`         |
| **Required**              | Yes              |
| **Additional properties** | Any type allowed |

#### <a name="datasets_additionalProperties_num_worker_per_gpu"></a>54.1.4. Property `ReduxOptions > datasets > DatasetOptions > num_worker_per_gpu`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `combining`      |
| **Required**              | No               |
| **Additional properties** | Any type allowed |
| **Default**               | `null`           |

**Description:** Number of subprocesses to use for data loading with PyTorch dataloader.

| Any of(Option)                                                       |
| -------------------------------------------------------------------- |
| [item 0](#datasets_additionalProperties_num_worker_per_gpu_anyOf_i0) |
| [item 1](#datasets_additionalProperties_num_worker_per_gpu_anyOf_i1) |

##### <a name="datasets_additionalProperties_num_worker_per_gpu_anyOf_i0"></a>54.1.4.1. Property `ReduxOptions > datasets > DatasetOptions > num_worker_per_gpu > anyOf > item 0`

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | No        |

##### <a name="datasets_additionalProperties_num_worker_per_gpu_anyOf_i1"></a>54.1.4.2. Property `ReduxOptions > datasets > DatasetOptions > num_worker_per_gpu > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

#### <a name="datasets_additionalProperties_batch_size_per_gpu"></a>54.1.5. Property `ReduxOptions > datasets > DatasetOptions > batch_size_per_gpu`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `combining`      |
| **Required**              | No               |
| **Additional properties** | Any type allowed |
| **Default**               | `null`           |

**Description:** Increasing stabilizes training but going too high can cause issues. Use multiple of 8 for best performance with AMP. A higher batch size, like 32 or 64 is more important when training from scratch, while smaller batches like 8 can be used when training with a quality pretrain model.

| Any of(Option)                                                       |
| -------------------------------------------------------------------- |
| [item 0](#datasets_additionalProperties_batch_size_per_gpu_anyOf_i0) |
| [item 1](#datasets_additionalProperties_batch_size_per_gpu_anyOf_i1) |

##### <a name="datasets_additionalProperties_batch_size_per_gpu_anyOf_i0"></a>54.1.5.1. Property `ReduxOptions > datasets > DatasetOptions > batch_size_per_gpu > anyOf > item 0`

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | No        |

##### <a name="datasets_additionalProperties_batch_size_per_gpu_anyOf_i1"></a>54.1.5.2. Property `ReduxOptions > datasets > DatasetOptions > batch_size_per_gpu > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

#### <a name="datasets_additionalProperties_accum_iter"></a>54.1.6. Property `ReduxOptions > datasets > DatasetOptions > accum_iter`

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | No        |
| **Default**  | `1`       |

**Description:** Using values larger than 1 simulates higher batch size by trading performance for reduced VRAM usage. If accum_iter = 4 and batch_size_per_gpu = 6 then effective batch size = 4 * 6 = 24 but performance may be as much as 4 times as slow.

#### <a name="datasets_additionalProperties_use_hflip"></a>54.1.7. Property `ReduxOptions > datasets > DatasetOptions > use_hflip`

|              |           |
| ------------ | --------- |
| **Type**     | `boolean` |
| **Required** | No        |
| **Default**  | `true`    |

**Description:** Randomly flip the images horizontally.

#### <a name="datasets_additionalProperties_use_rot"></a>54.1.8. Property `ReduxOptions > datasets > DatasetOptions > use_rot`

|              |           |
| ------------ | --------- |
| **Type**     | `boolean` |
| **Required** | No        |
| **Default**  | `true`    |

**Description:** Randomly rotate the images.

#### <a name="datasets_additionalProperties_mean"></a>54.1.9. Property `ReduxOptions > datasets > DatasetOptions > mean`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `combining`      |
| **Required**              | No               |
| **Additional properties** | Any type allowed |
| **Default**               | `null`           |

| Any of(Option)                                         |
| ------------------------------------------------------ |
| [item 0](#datasets_additionalProperties_mean_anyOf_i0) |
| [item 1](#datasets_additionalProperties_mean_anyOf_i1) |

##### <a name="datasets_additionalProperties_mean_anyOf_i0"></a>54.1.9.1. Property `ReduxOptions > datasets > DatasetOptions > mean > anyOf > item 0`

|              |                   |
| ------------ | ----------------- |
| **Type**     | `array of number` |
| **Required** | No                |

|                      | Array restrictions |
| -------------------- | ------------------ |
| **Min items**        | N/A                |
| **Max items**        | N/A                |
| **Items unicity**    | False              |
| **Additional items** | False              |
| **Tuple validation** | See below          |

| Each item of this array must be                                    | Description |
| ------------------------------------------------------------------ | ----------- |
| [item 0 items](#datasets_additionalProperties_mean_anyOf_i0_items) | -           |

###### <a name="datasets_additionalProperties_mean_anyOf_i0_items"></a>54.1.9.1.1. ReduxOptions > datasets > DatasetOptions > mean > anyOf > item 0 > item 0 items

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |

##### <a name="datasets_additionalProperties_mean_anyOf_i1"></a>54.1.9.2. Property `ReduxOptions > datasets > DatasetOptions > mean > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

#### <a name="datasets_additionalProperties_std"></a>54.1.10. Property `ReduxOptions > datasets > DatasetOptions > std`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `combining`      |
| **Required**              | No               |
| **Additional properties** | Any type allowed |
| **Default**               | `null`           |

| Any of(Option)                                        |
| ----------------------------------------------------- |
| [item 0](#datasets_additionalProperties_std_anyOf_i0) |
| [item 1](#datasets_additionalProperties_std_anyOf_i1) |

##### <a name="datasets_additionalProperties_std_anyOf_i0"></a>54.1.10.1. Property `ReduxOptions > datasets > DatasetOptions > std > anyOf > item 0`

|              |                   |
| ------------ | ----------------- |
| **Type**     | `array of number` |
| **Required** | No                |

|                      | Array restrictions |
| -------------------- | ------------------ |
| **Min items**        | N/A                |
| **Max items**        | N/A                |
| **Items unicity**    | False              |
| **Additional items** | False              |
| **Tuple validation** | See below          |

| Each item of this array must be                                   | Description |
| ----------------------------------------------------------------- | ----------- |
| [item 0 items](#datasets_additionalProperties_std_anyOf_i0_items) | -           |

###### <a name="datasets_additionalProperties_std_anyOf_i0_items"></a>54.1.10.1.1. ReduxOptions > datasets > DatasetOptions > std > anyOf > item 0 > item 0 items

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |

##### <a name="datasets_additionalProperties_std_anyOf_i1"></a>54.1.10.2. Property `ReduxOptions > datasets > DatasetOptions > std > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

#### <a name="datasets_additionalProperties_gt_size"></a>54.1.11. Property `ReduxOptions > datasets > DatasetOptions > gt_size`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `combining`      |
| **Required**              | No               |
| **Additional properties** | Any type allowed |
| **Default**               | `null`           |

| Any of(Option)                                            |
| --------------------------------------------------------- |
| [item 0](#datasets_additionalProperties_gt_size_anyOf_i0) |
| [item 1](#datasets_additionalProperties_gt_size_anyOf_i1) |

##### <a name="datasets_additionalProperties_gt_size_anyOf_i0"></a>54.1.11.1. Property `ReduxOptions > datasets > DatasetOptions > gt_size > anyOf > item 0`

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | No        |

##### <a name="datasets_additionalProperties_gt_size_anyOf_i1"></a>54.1.11.2. Property `ReduxOptions > datasets > DatasetOptions > gt_size > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

#### <a name="datasets_additionalProperties_lq_size"></a>54.1.12. Property `ReduxOptions > datasets > DatasetOptions > lq_size`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `combining`      |
| **Required**              | No               |
| **Additional properties** | Any type allowed |
| **Default**               | `null`           |

**Description:** During training, a square of this size is cropped from LR images. Larger is usually better but uses more VRAM. Previously gt_size, use lq_size = gt_size / scale to convert. Use multiple of 8 for best performance with AMP.

| Any of(Option)                                            |
| --------------------------------------------------------- |
| [item 0](#datasets_additionalProperties_lq_size_anyOf_i0) |
| [item 1](#datasets_additionalProperties_lq_size_anyOf_i1) |

##### <a name="datasets_additionalProperties_lq_size_anyOf_i0"></a>54.1.12.1. Property `ReduxOptions > datasets > DatasetOptions > lq_size > anyOf > item 0`

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | No        |

##### <a name="datasets_additionalProperties_lq_size_anyOf_i1"></a>54.1.12.2. Property `ReduxOptions > datasets > DatasetOptions > lq_size > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

#### <a name="datasets_additionalProperties_color"></a>54.1.13. Property `ReduxOptions > datasets > DatasetOptions > color`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `combining`      |
| **Required**              | No               |
| **Additional properties** | Any type allowed |
| **Default**               | `null`           |

| Any of(Option)                                          |
| ------------------------------------------------------- |
| [item 0](#datasets_additionalProperties_color_anyOf_i0) |
| [item 1](#datasets_additionalProperties_color_anyOf_i1) |

##### <a name="datasets_additionalProperties_color_anyOf_i0"></a>54.1.13.1. Property `ReduxOptions > datasets > DatasetOptions > color > anyOf > item 0`

|              |                    |
| ------------ | ------------------ |
| **Type**     | `enum (of string)` |
| **Required** | No                 |

Must be one of:
* "y"

##### <a name="datasets_additionalProperties_color_anyOf_i1"></a>54.1.13.2. Property `ReduxOptions > datasets > DatasetOptions > color > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

#### <a name="datasets_additionalProperties_phase"></a>54.1.14. Property `ReduxOptions > datasets > DatasetOptions > phase`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `combining`      |
| **Required**              | No               |
| **Additional properties** | Any type allowed |
| **Default**               | `null`           |

| Any of(Option)                                          |
| ------------------------------------------------------- |
| [item 0](#datasets_additionalProperties_phase_anyOf_i0) |
| [item 1](#datasets_additionalProperties_phase_anyOf_i1) |

##### <a name="datasets_additionalProperties_phase_anyOf_i0"></a>54.1.14.1. Property `ReduxOptions > datasets > DatasetOptions > phase > anyOf > item 0`

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | No       |

##### <a name="datasets_additionalProperties_phase_anyOf_i1"></a>54.1.14.2. Property `ReduxOptions > datasets > DatasetOptions > phase > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

#### <a name="datasets_additionalProperties_scale"></a>54.1.15. Property `ReduxOptions > datasets > DatasetOptions > scale`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `combining`      |
| **Required**              | No               |
| **Additional properties** | Any type allowed |
| **Default**               | `null`           |

| Any of(Option)                                          |
| ------------------------------------------------------- |
| [item 0](#datasets_additionalProperties_scale_anyOf_i0) |
| [item 1](#datasets_additionalProperties_scale_anyOf_i1) |

##### <a name="datasets_additionalProperties_scale_anyOf_i0"></a>54.1.15.1. Property `ReduxOptions > datasets > DatasetOptions > scale > anyOf > item 0`

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | No        |

##### <a name="datasets_additionalProperties_scale_anyOf_i1"></a>54.1.15.2. Property `ReduxOptions > datasets > DatasetOptions > scale > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

#### <a name="datasets_additionalProperties_dataset_enlarge_ratio"></a>54.1.16. Property `ReduxOptions > datasets > DatasetOptions > dataset_enlarge_ratio`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `combining`      |
| **Required**              | No               |
| **Additional properties** | Any type allowed |
| **Default**               | `"auto"`         |

**Description:** Increase if the dataset is less than 1000 images to avoid slowdowns. Auto will automatically enlarge small datasets only.

| Any of(Option)                                                          |
| ----------------------------------------------------------------------- |
| [item 0](#datasets_additionalProperties_dataset_enlarge_ratio_anyOf_i0) |
| [item 1](#datasets_additionalProperties_dataset_enlarge_ratio_anyOf_i1) |

##### <a name="datasets_additionalProperties_dataset_enlarge_ratio_anyOf_i0"></a>54.1.16.1. Property `ReduxOptions > datasets > DatasetOptions > dataset_enlarge_ratio > anyOf > item 0`

|              |                    |
| ------------ | ------------------ |
| **Type**     | `enum (of string)` |
| **Required** | No                 |

Must be one of:
* "auto"

##### <a name="datasets_additionalProperties_dataset_enlarge_ratio_anyOf_i1"></a>54.1.16.2. Property `ReduxOptions > datasets > DatasetOptions > dataset_enlarge_ratio > anyOf > item 1`

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | No        |

#### <a name="datasets_additionalProperties_prefetch_mode"></a>54.1.17. Property `ReduxOptions > datasets > DatasetOptions > prefetch_mode`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `combining`      |
| **Required**              | No               |
| **Additional properties** | Any type allowed |
| **Default**               | `null`           |

| Any of(Option)                                                  |
| --------------------------------------------------------------- |
| [item 0](#datasets_additionalProperties_prefetch_mode_anyOf_i0) |
| [item 1](#datasets_additionalProperties_prefetch_mode_anyOf_i1) |

##### <a name="datasets_additionalProperties_prefetch_mode_anyOf_i0"></a>54.1.17.1. Property `ReduxOptions > datasets > DatasetOptions > prefetch_mode > anyOf > item 0`

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | No       |

##### <a name="datasets_additionalProperties_prefetch_mode_anyOf_i1"></a>54.1.17.2. Property `ReduxOptions > datasets > DatasetOptions > prefetch_mode > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

#### <a name="datasets_additionalProperties_pin_memory"></a>54.1.18. Property `ReduxOptions > datasets > DatasetOptions > pin_memory`

|              |           |
| ------------ | --------- |
| **Type**     | `boolean` |
| **Required** | No        |
| **Default**  | `true`    |

#### <a name="datasets_additionalProperties_persistent_workers"></a>54.1.19. Property `ReduxOptions > datasets > DatasetOptions > persistent_workers`

|              |           |
| ------------ | --------- |
| **Type**     | `boolean` |
| **Required** | No        |
| **Default**  | `true`    |

#### <a name="datasets_additionalProperties_num_prefetch_queue"></a>54.1.20. Property `ReduxOptions > datasets > DatasetOptions > num_prefetch_queue`

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | No        |
| **Default**  | `1`       |

#### <a name="datasets_additionalProperties_prefetch_factor"></a>54.1.21. Property `ReduxOptions > datasets > DatasetOptions > prefetch_factor`

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | No        |
| **Default**  | `2`       |

#### <a name="datasets_additionalProperties_clip_size"></a>54.1.22. Property `ReduxOptions > datasets > DatasetOptions > clip_size`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `combining`      |
| **Required**              | No               |
| **Additional properties** | Any type allowed |
| **Default**               | `null`           |

| Any of(Option)                                              |
| ----------------------------------------------------------- |
| [item 0](#datasets_additionalProperties_clip_size_anyOf_i0) |
| [item 1](#datasets_additionalProperties_clip_size_anyOf_i1) |

##### <a name="datasets_additionalProperties_clip_size_anyOf_i0"></a>54.1.22.1. Property `ReduxOptions > datasets > DatasetOptions > clip_size > anyOf > item 0`

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | No        |

##### <a name="datasets_additionalProperties_clip_size_anyOf_i1"></a>54.1.22.2. Property `ReduxOptions > datasets > DatasetOptions > clip_size > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

#### <a name="datasets_additionalProperties_dataroot_gt"></a>54.1.23. Property `ReduxOptions > datasets > DatasetOptions > dataroot_gt`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `combining`      |
| **Required**              | No               |
| **Additional properties** | Any type allowed |
| **Default**               | `null`           |

**Description:** Path to the HR (high res) images in your training dataset. Specify one or multiple folders, separated by commas.

| Any of(Option)                                                |
| ------------------------------------------------------------- |
| [item 0](#datasets_additionalProperties_dataroot_gt_anyOf_i0) |
| [item 1](#datasets_additionalProperties_dataroot_gt_anyOf_i1) |
| [item 2](#datasets_additionalProperties_dataroot_gt_anyOf_i2) |

##### <a name="datasets_additionalProperties_dataroot_gt_anyOf_i0"></a>54.1.23.1. Property `ReduxOptions > datasets > DatasetOptions > dataroot_gt > anyOf > item 0`

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | No       |

##### <a name="datasets_additionalProperties_dataroot_gt_anyOf_i1"></a>54.1.23.2. Property `ReduxOptions > datasets > DatasetOptions > dataroot_gt > anyOf > item 1`

|              |                   |
| ------------ | ----------------- |
| **Type**     | `array of string` |
| **Required** | No                |

|                      | Array restrictions |
| -------------------- | ------------------ |
| **Min items**        | N/A                |
| **Max items**        | N/A                |
| **Items unicity**    | False              |
| **Additional items** | False              |
| **Tuple validation** | See below          |

| Each item of this array must be                                           | Description |
| ------------------------------------------------------------------------- | ----------- |
| [item 1 items](#datasets_additionalProperties_dataroot_gt_anyOf_i1_items) | -           |

###### <a name="datasets_additionalProperties_dataroot_gt_anyOf_i1_items"></a>54.1.23.2.1. ReduxOptions > datasets > DatasetOptions > dataroot_gt > anyOf > item 1 > item 1 items

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | No       |

##### <a name="datasets_additionalProperties_dataroot_gt_anyOf_i2"></a>54.1.23.3. Property `ReduxOptions > datasets > DatasetOptions > dataroot_gt > anyOf > item 2`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

#### <a name="datasets_additionalProperties_dataroot_lq"></a>54.1.24. Property `ReduxOptions > datasets > DatasetOptions > dataroot_lq`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `combining`      |
| **Required**              | No               |
| **Additional properties** | Any type allowed |
| **Default**               | `null`           |

**Description:** Path to the LR (low res) images in your training dataset. Specify one or multiple folders, separated by commas.

| Any of(Option)                                                |
| ------------------------------------------------------------- |
| [item 0](#datasets_additionalProperties_dataroot_lq_anyOf_i0) |
| [item 1](#datasets_additionalProperties_dataroot_lq_anyOf_i1) |
| [item 2](#datasets_additionalProperties_dataroot_lq_anyOf_i2) |

##### <a name="datasets_additionalProperties_dataroot_lq_anyOf_i0"></a>54.1.24.1. Property `ReduxOptions > datasets > DatasetOptions > dataroot_lq > anyOf > item 0`

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | No       |

##### <a name="datasets_additionalProperties_dataroot_lq_anyOf_i1"></a>54.1.24.2. Property `ReduxOptions > datasets > DatasetOptions > dataroot_lq > anyOf > item 1`

|              |                   |
| ------------ | ----------------- |
| **Type**     | `array of string` |
| **Required** | No                |

|                      | Array restrictions |
| -------------------- | ------------------ |
| **Min items**        | N/A                |
| **Max items**        | N/A                |
| **Items unicity**    | False              |
| **Additional items** | False              |
| **Tuple validation** | See below          |

| Each item of this array must be                                           | Description |
| ------------------------------------------------------------------------- | ----------- |
| [item 1 items](#datasets_additionalProperties_dataroot_lq_anyOf_i1_items) | -           |

###### <a name="datasets_additionalProperties_dataroot_lq_anyOf_i1_items"></a>54.1.24.2.1. ReduxOptions > datasets > DatasetOptions > dataroot_lq > anyOf > item 1 > item 1 items

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | No       |

##### <a name="datasets_additionalProperties_dataroot_lq_anyOf_i2"></a>54.1.24.3. Property `ReduxOptions > datasets > DatasetOptions > dataroot_lq > anyOf > item 2`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

#### <a name="datasets_additionalProperties_meta_info"></a>54.1.25. Property `ReduxOptions > datasets > DatasetOptions > meta_info`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `combining`      |
| **Required**              | No               |
| **Additional properties** | Any type allowed |
| **Default**               | `null`           |

| Any of(Option)                                              |
| ----------------------------------------------------------- |
| [item 0](#datasets_additionalProperties_meta_info_anyOf_i0) |
| [item 1](#datasets_additionalProperties_meta_info_anyOf_i1) |

##### <a name="datasets_additionalProperties_meta_info_anyOf_i0"></a>54.1.25.1. Property `ReduxOptions > datasets > DatasetOptions > meta_info > anyOf > item 0`

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | No       |

##### <a name="datasets_additionalProperties_meta_info_anyOf_i1"></a>54.1.25.2. Property `ReduxOptions > datasets > DatasetOptions > meta_info > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

#### <a name="datasets_additionalProperties_filename_tmpl"></a>54.1.26. Property `ReduxOptions > datasets > DatasetOptions > filename_tmpl`

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | No       |
| **Default**  | `"{}"`   |

#### <a name="datasets_additionalProperties_blur_kernel_size"></a>54.1.27. Property `ReduxOptions > datasets > DatasetOptions > blur_kernel_size`

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | No        |
| **Default**  | `12`      |

#### <a name="datasets_additionalProperties_kernel_list"></a>54.1.28. Property `ReduxOptions > datasets > DatasetOptions > kernel_list`

|              |                   |
| ------------ | ----------------- |
| **Type**     | `array of string` |
| **Required** | No                |

|                      | Array restrictions |
| -------------------- | ------------------ |
| **Min items**        | N/A                |
| **Max items**        | N/A                |
| **Items unicity**    | False              |
| **Additional items** | False              |
| **Tuple validation** | See below          |

| Each item of this array must be                                       | Description |
| --------------------------------------------------------------------- | ----------- |
| [kernel_list items](#datasets_additionalProperties_kernel_list_items) | -           |

##### <a name="datasets_additionalProperties_kernel_list_items"></a>54.1.28.1. ReduxOptions > datasets > DatasetOptions > kernel_list > kernel_list items

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | No       |

#### <a name="datasets_additionalProperties_kernel_prob"></a>54.1.29. Property `ReduxOptions > datasets > DatasetOptions > kernel_prob`

|              |                   |
| ------------ | ----------------- |
| **Type**     | `array of number` |
| **Required** | No                |

|                      | Array restrictions |
| -------------------- | ------------------ |
| **Min items**        | N/A                |
| **Max items**        | N/A                |
| **Items unicity**    | False              |
| **Additional items** | False              |
| **Tuple validation** | See below          |

| Each item of this array must be                                       | Description |
| --------------------------------------------------------------------- | ----------- |
| [kernel_prob items](#datasets_additionalProperties_kernel_prob_items) | -           |

##### <a name="datasets_additionalProperties_kernel_prob_items"></a>54.1.29.1. ReduxOptions > datasets > DatasetOptions > kernel_prob > kernel_prob items

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |

#### <a name="datasets_additionalProperties_kernel_range"></a>54.1.30. Property `ReduxOptions > datasets > DatasetOptions > kernel_range`

|              |           |
| ------------ | --------- |
| **Type**     | `array`   |
| **Required** | No        |
| **Default**  | `[5, 17]` |

|                      | Array restrictions |
| -------------------- | ------------------ |
| **Min items**        | 2                  |
| **Max items**        | 2                  |
| **Items unicity**    | False              |
| **Additional items** | False              |
| **Tuple validation** | See below          |

| Each item of this array must be                                             | Description |
| --------------------------------------------------------------------------- | ----------- |
| [kernel_range item 0](#datasets_additionalProperties_kernel_range_items_i0) | -           |
| [kernel_range item 1](#datasets_additionalProperties_kernel_range_items_i1) | -           |

##### <a name="autogenerated_heading_18"></a>54.1.30.1. ReduxOptions > datasets > DatasetOptions > kernel_range > kernel_range item 0

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | No        |

##### <a name="autogenerated_heading_19"></a>54.1.30.2. ReduxOptions > datasets > DatasetOptions > kernel_range > kernel_range item 1

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | No        |

#### <a name="datasets_additionalProperties_sinc_prob"></a>54.1.31. Property `ReduxOptions > datasets > DatasetOptions > sinc_prob`

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |
| **Default**  | `0`      |

#### <a name="datasets_additionalProperties_blur_sigma"></a>54.1.32. Property `ReduxOptions > datasets > DatasetOptions > blur_sigma`

|              |            |
| ------------ | ---------- |
| **Type**     | `array`    |
| **Required** | No         |
| **Default**  | `[0.2, 2]` |

|                      | Array restrictions |
| -------------------- | ------------------ |
| **Min items**        | 2                  |
| **Max items**        | 2                  |
| **Items unicity**    | False              |
| **Additional items** | False              |
| **Tuple validation** | See below          |

| Each item of this array must be                                         | Description |
| ----------------------------------------------------------------------- | ----------- |
| [blur_sigma item 0](#datasets_additionalProperties_blur_sigma_items_i0) | -           |
| [blur_sigma item 1](#datasets_additionalProperties_blur_sigma_items_i1) | -           |

##### <a name="autogenerated_heading_20"></a>54.1.32.1. ReduxOptions > datasets > DatasetOptions > blur_sigma > blur_sigma item 0

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |

##### <a name="autogenerated_heading_21"></a>54.1.32.2. ReduxOptions > datasets > DatasetOptions > blur_sigma > blur_sigma item 1

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |

#### <a name="datasets_additionalProperties_betag_range"></a>54.1.33. Property `ReduxOptions > datasets > DatasetOptions > betag_range`

|              |            |
| ------------ | ---------- |
| **Type**     | `array`    |
| **Required** | No         |
| **Default**  | `[0.5, 4]` |

|                      | Array restrictions |
| -------------------- | ------------------ |
| **Min items**        | 2                  |
| **Max items**        | 2                  |
| **Items unicity**    | False              |
| **Additional items** | False              |
| **Tuple validation** | See below          |

| Each item of this array must be                                           | Description |
| ------------------------------------------------------------------------- | ----------- |
| [betag_range item 0](#datasets_additionalProperties_betag_range_items_i0) | -           |
| [betag_range item 1](#datasets_additionalProperties_betag_range_items_i1) | -           |

##### <a name="autogenerated_heading_22"></a>54.1.33.1. ReduxOptions > datasets > DatasetOptions > betag_range > betag_range item 0

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |

##### <a name="autogenerated_heading_23"></a>54.1.33.2. ReduxOptions > datasets > DatasetOptions > betag_range > betag_range item 1

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |

#### <a name="datasets_additionalProperties_betap_range"></a>54.1.34. Property `ReduxOptions > datasets > DatasetOptions > betap_range`

|              |          |
| ------------ | -------- |
| **Type**     | `array`  |
| **Required** | No       |
| **Default**  | `[1, 2]` |

|                      | Array restrictions |
| -------------------- | ------------------ |
| **Min items**        | 2                  |
| **Max items**        | 2                  |
| **Items unicity**    | False              |
| **Additional items** | False              |
| **Tuple validation** | See below          |

| Each item of this array must be                                           | Description |
| ------------------------------------------------------------------------- | ----------- |
| [betap_range item 0](#datasets_additionalProperties_betap_range_items_i0) | -           |
| [betap_range item 1](#datasets_additionalProperties_betap_range_items_i1) | -           |

##### <a name="autogenerated_heading_24"></a>54.1.34.1. ReduxOptions > datasets > DatasetOptions > betap_range > betap_range item 0

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |

##### <a name="autogenerated_heading_25"></a>54.1.34.2. ReduxOptions > datasets > DatasetOptions > betap_range > betap_range item 1

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |

#### <a name="datasets_additionalProperties_blur_kernel_size2"></a>54.1.35. Property `ReduxOptions > datasets > DatasetOptions > blur_kernel_size2`

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | No        |
| **Default**  | `12`      |

#### <a name="datasets_additionalProperties_kernel_list2"></a>54.1.36. Property `ReduxOptions > datasets > DatasetOptions > kernel_list2`

|              |                   |
| ------------ | ----------------- |
| **Type**     | `array of string` |
| **Required** | No                |

|                      | Array restrictions |
| -------------------- | ------------------ |
| **Min items**        | N/A                |
| **Max items**        | N/A                |
| **Items unicity**    | False              |
| **Additional items** | False              |
| **Tuple validation** | See below          |

| Each item of this array must be                                         | Description |
| ----------------------------------------------------------------------- | ----------- |
| [kernel_list2 items](#datasets_additionalProperties_kernel_list2_items) | -           |

##### <a name="datasets_additionalProperties_kernel_list2_items"></a>54.1.36.1. ReduxOptions > datasets > DatasetOptions > kernel_list2 > kernel_list2 items

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | No       |

#### <a name="datasets_additionalProperties_kernel_prob2"></a>54.1.37. Property `ReduxOptions > datasets > DatasetOptions > kernel_prob2`

|              |                   |
| ------------ | ----------------- |
| **Type**     | `array of number` |
| **Required** | No                |

|                      | Array restrictions |
| -------------------- | ------------------ |
| **Min items**        | N/A                |
| **Max items**        | N/A                |
| **Items unicity**    | False              |
| **Additional items** | False              |
| **Tuple validation** | See below          |

| Each item of this array must be                                         | Description |
| ----------------------------------------------------------------------- | ----------- |
| [kernel_prob2 items](#datasets_additionalProperties_kernel_prob2_items) | -           |

##### <a name="datasets_additionalProperties_kernel_prob2_items"></a>54.1.37.1. ReduxOptions > datasets > DatasetOptions > kernel_prob2 > kernel_prob2 items

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |

#### <a name="datasets_additionalProperties_kernel_range2"></a>54.1.38. Property `ReduxOptions > datasets > DatasetOptions > kernel_range2`

|              |           |
| ------------ | --------- |
| **Type**     | `array`   |
| **Required** | No        |
| **Default**  | `[5, 17]` |

|                      | Array restrictions |
| -------------------- | ------------------ |
| **Min items**        | 2                  |
| **Max items**        | 2                  |
| **Items unicity**    | False              |
| **Additional items** | False              |
| **Tuple validation** | See below          |

| Each item of this array must be                                               | Description |
| ----------------------------------------------------------------------------- | ----------- |
| [kernel_range2 item 0](#datasets_additionalProperties_kernel_range2_items_i0) | -           |
| [kernel_range2 item 1](#datasets_additionalProperties_kernel_range2_items_i1) | -           |

##### <a name="autogenerated_heading_26"></a>54.1.38.1. ReduxOptions > datasets > DatasetOptions > kernel_range2 > kernel_range2 item 0

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | No        |

##### <a name="autogenerated_heading_27"></a>54.1.38.2. ReduxOptions > datasets > DatasetOptions > kernel_range2 > kernel_range2 item 1

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | No        |

#### <a name="datasets_additionalProperties_sinc_prob2"></a>54.1.39. Property `ReduxOptions > datasets > DatasetOptions > sinc_prob2`

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |
| **Default**  | `0`      |

#### <a name="datasets_additionalProperties_blur_sigma2"></a>54.1.40. Property `ReduxOptions > datasets > DatasetOptions > blur_sigma2`

|              |            |
| ------------ | ---------- |
| **Type**     | `array`    |
| **Required** | No         |
| **Default**  | `[0.2, 1]` |

|                      | Array restrictions |
| -------------------- | ------------------ |
| **Min items**        | 2                  |
| **Max items**        | 2                  |
| **Items unicity**    | False              |
| **Additional items** | False              |
| **Tuple validation** | See below          |

| Each item of this array must be                                           | Description |
| ------------------------------------------------------------------------- | ----------- |
| [blur_sigma2 item 0](#datasets_additionalProperties_blur_sigma2_items_i0) | -           |
| [blur_sigma2 item 1](#datasets_additionalProperties_blur_sigma2_items_i1) | -           |

##### <a name="autogenerated_heading_28"></a>54.1.40.1. ReduxOptions > datasets > DatasetOptions > blur_sigma2 > blur_sigma2 item 0

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |

##### <a name="autogenerated_heading_29"></a>54.1.40.2. ReduxOptions > datasets > DatasetOptions > blur_sigma2 > blur_sigma2 item 1

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |

#### <a name="datasets_additionalProperties_betag_range2"></a>54.1.41. Property `ReduxOptions > datasets > DatasetOptions > betag_range2`

|              |            |
| ------------ | ---------- |
| **Type**     | `array`    |
| **Required** | No         |
| **Default**  | `[0.5, 4]` |

|                      | Array restrictions |
| -------------------- | ------------------ |
| **Min items**        | 2                  |
| **Max items**        | 2                  |
| **Items unicity**    | False              |
| **Additional items** | False              |
| **Tuple validation** | See below          |

| Each item of this array must be                                             | Description |
| --------------------------------------------------------------------------- | ----------- |
| [betag_range2 item 0](#datasets_additionalProperties_betag_range2_items_i0) | -           |
| [betag_range2 item 1](#datasets_additionalProperties_betag_range2_items_i1) | -           |

##### <a name="autogenerated_heading_30"></a>54.1.41.1. ReduxOptions > datasets > DatasetOptions > betag_range2 > betag_range2 item 0

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |

##### <a name="autogenerated_heading_31"></a>54.1.41.2. ReduxOptions > datasets > DatasetOptions > betag_range2 > betag_range2 item 1

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |

#### <a name="datasets_additionalProperties_betap_range2"></a>54.1.42. Property `ReduxOptions > datasets > DatasetOptions > betap_range2`

|              |          |
| ------------ | -------- |
| **Type**     | `array`  |
| **Required** | No       |
| **Default**  | `[1, 2]` |

|                      | Array restrictions |
| -------------------- | ------------------ |
| **Min items**        | 2                  |
| **Max items**        | 2                  |
| **Items unicity**    | False              |
| **Additional items** | False              |
| **Tuple validation** | See below          |

| Each item of this array must be                                             | Description |
| --------------------------------------------------------------------------- | ----------- |
| [betap_range2 item 0](#datasets_additionalProperties_betap_range2_items_i0) | -           |
| [betap_range2 item 1](#datasets_additionalProperties_betap_range2_items_i1) | -           |

##### <a name="autogenerated_heading_32"></a>54.1.42.1. ReduxOptions > datasets > DatasetOptions > betap_range2 > betap_range2 item 0

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |

##### <a name="autogenerated_heading_33"></a>54.1.42.2. ReduxOptions > datasets > DatasetOptions > betap_range2 > betap_range2 item 1

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |

#### <a name="datasets_additionalProperties_final_sinc_prob"></a>54.1.43. Property `ReduxOptions > datasets > DatasetOptions > final_sinc_prob`

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |
| **Default**  | `0`      |

#### <a name="datasets_additionalProperties_final_kernel_range"></a>54.1.44. Property `ReduxOptions > datasets > DatasetOptions > final_kernel_range`

|              |           |
| ------------ | --------- |
| **Type**     | `array`   |
| **Required** | No        |
| **Default**  | `[5, 17]` |

|                      | Array restrictions |
| -------------------- | ------------------ |
| **Min items**        | 2                  |
| **Max items**        | 2                  |
| **Items unicity**    | False              |
| **Additional items** | False              |
| **Tuple validation** | See below          |

| Each item of this array must be                                                         | Description |
| --------------------------------------------------------------------------------------- | ----------- |
| [final_kernel_range item 0](#datasets_additionalProperties_final_kernel_range_items_i0) | -           |
| [final_kernel_range item 1](#datasets_additionalProperties_final_kernel_range_items_i1) | -           |

##### <a name="autogenerated_heading_34"></a>54.1.44.1. ReduxOptions > datasets > DatasetOptions > final_kernel_range > final_kernel_range item 0

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | No        |

##### <a name="autogenerated_heading_35"></a>54.1.44.2. ReduxOptions > datasets > DatasetOptions > final_kernel_range > final_kernel_range item 1

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | No        |

## <a name="train"></a>55. Property `ReduxOptions > train`

**Title:** TrainOptions

|                           |                      |
| ------------------------- | -------------------- |
| **Type**                  | `object`             |
| **Required**              | No                   |
| **Additional properties** | Any type allowed     |
| **Default**               | `null`               |
| **Defined in**            | #/$defs/TrainOptions |

| Property                                       | Pattern | Type            | Deprecated | Definition | Title/Description                                                                                                                                       |
| ---------------------------------------------- | ------- | --------------- | ---------- | ---------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| + [total_iter](#train_total_iter )             | No      | integer         | No         | -          | The total number of iterations to train.                                                                                                                |
| + [optim_g](#train_optim_g )                   | No      | object          | No         | -          | The optimizer to use for the generator model.                                                                                                           |
| - [ema_decay](#train_ema_decay )               | No      | number          | No         | -          | The decay factor to use for EMA (exponential moving average). Set to 0 to disable EMA.                                                                  |
| - [grad_clip](#train_grad_clip )               | No      | boolean         | No         | -          | Whether or not to enable gradient clipping, which can improve stability when using higher learning rates, but can also cause issues in some situations. |
| - [warmup_iter](#train_warmup_iter )           | No      | integer         | No         | -          | Gradually ramp up learning rates until this iteration, to stabilize early training. Use -1 to disable.                                                  |
| - [scheduler](#train_scheduler )               | No      | Combination     | No         | -          | Options for the optimizer scheduler. If there are multiple optimizers, both will use the same scheduler options.                                        |
| - [optim_d](#train_optim_d )                   | No      | Combination     | No         | -          | The optimizer to use for the discriminator model.                                                                                                       |
| - [losses](#train_losses )                     | No      | array           | No         | -          | The list of loss functions to optimize.                                                                                                                 |
| - [pixel_opt](#train_pixel_opt )               | No      | Combination     | No         | -          | -                                                                                                                                                       |
| - [mssim_opt](#train_mssim_opt )               | No      | Combination     | No         | -          | -                                                                                                                                                       |
| - [ms_ssim_l1_opt](#train_ms_ssim_l1_opt )     | No      | Combination     | No         | -          | -                                                                                                                                                       |
| - [perceptual_opt](#train_perceptual_opt )     | No      | Combination     | No         | -          | -                                                                                                                                                       |
| - [contextual_opt](#train_contextual_opt )     | No      | Combination     | No         | -          | -                                                                                                                                                       |
| - [dists_opt](#train_dists_opt )               | No      | Combination     | No         | -          | -                                                                                                                                                       |
| - [hr_inversion_opt](#train_hr_inversion_opt ) | No      | Combination     | No         | -          | -                                                                                                                                                       |
| - [dinov2_opt](#train_dinov2_opt )             | No      | Combination     | No         | -          | -                                                                                                                                                       |
| - [topiq_opt](#train_topiq_opt )               | No      | Combination     | No         | -          | -                                                                                                                                                       |
| - [pd_opt](#train_pd_opt )                     | No      | Combination     | No         | -          | -                                                                                                                                                       |
| - [fd_opt](#train_fd_opt )                     | No      | Combination     | No         | -          | -                                                                                                                                                       |
| - [ldl_opt](#train_ldl_opt )                   | No      | Combination     | No         | -          | -                                                                                                                                                       |
| - [hsluv_opt](#train_hsluv_opt )               | No      | Combination     | No         | -          | -                                                                                                                                                       |
| - [gan_opt](#train_gan_opt )                   | No      | Combination     | No         | -          | -                                                                                                                                                       |
| - [color_opt](#train_color_opt )               | No      | Combination     | No         | -          | -                                                                                                                                                       |
| - [luma_opt](#train_luma_opt )                 | No      | Combination     | No         | -          | -                                                                                                                                                       |
| - [avg_opt](#train_avg_opt )                   | No      | Combination     | No         | -          | -                                                                                                                                                       |
| - [bicubic_opt](#train_bicubic_opt )           | No      | Combination     | No         | -          | -                                                                                                                                                       |
| - [use_moa](#train_use_moa )                   | No      | boolean         | No         | -          | Whether to enable mixture of augmentations, which augments the dataset on the fly to create more variety and help the model generalize.                 |
| - [moa_augs](#train_moa_augs )                 | No      | array of string | No         | -          | The list of augmentations to choose from, only one is selected per iteration.                                                                           |
| - [moa_probs](#train_moa_probs )               | No      | array of number | No         | -          | The probability each augmentation in moa_augs will be applied. Total should add up to 1.                                                                |
| - [moa_debug](#train_moa_debug )               | No      | boolean         | No         | -          | Save images before and after augment to debug/moa folder inside of the root training directory.                                                         |
| - [moa_debug_limit](#train_moa_debug_limit )   | No      | integer         | No         | -          | The max number of iterations to save augmentation images for.                                                                                           |

### <a name="train_total_iter"></a>55.1. Property `ReduxOptions > train > total_iter`

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | Yes       |

**Description:** The total number of iterations to train.

### <a name="train_optim_g"></a>55.2. Property `ReduxOptions > train > optim_g`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `object`         |
| **Required**              | Yes              |
| **Additional properties** | Any type allowed |

**Description:** The optimizer to use for the generator model.

### <a name="train_ema_decay"></a>55.3. Property `ReduxOptions > train > ema_decay`

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |
| **Default**  | `0`      |

**Description:** The decay factor to use for EMA (exponential moving average). Set to 0 to disable EMA.

### <a name="train_grad_clip"></a>55.4. Property `ReduxOptions > train > grad_clip`

|              |           |
| ------------ | --------- |
| **Type**     | `boolean` |
| **Required** | No        |
| **Default**  | `false`   |

**Description:** Whether or not to enable gradient clipping, which can improve stability when using higher learning rates, but can also cause issues in some situations.

### <a name="train_warmup_iter"></a>55.5. Property `ReduxOptions > train > warmup_iter`

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | No        |
| **Default**  | `-1`      |

**Description:** Gradually ramp up learning rates until this iteration, to stabilize early training. Use -1 to disable.

### <a name="train_scheduler"></a>55.6. Property `ReduxOptions > train > scheduler`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `combining`      |
| **Required**              | No               |
| **Additional properties** | Any type allowed |
| **Default**               | `null`           |

**Description:** Options for the optimizer scheduler. If there are multiple optimizers, both will use the same scheduler options.

| Any of(Option)                                |
| --------------------------------------------- |
| [item 0](#train_scheduler_anyOf_i0)           |
| [SchedulerOptions](#train_scheduler_anyOf_i1) |

#### <a name="train_scheduler_anyOf_i0"></a>55.6.1. Property `ReduxOptions > train > scheduler > anyOf > item 0`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

#### <a name="train_scheduler_anyOf_i1"></a>55.6.2. Property `ReduxOptions > train > scheduler > anyOf > SchedulerOptions`

**Title:** SchedulerOptions

|                           |                          |
| ------------------------- | ------------------------ |
| **Type**                  | `object`                 |
| **Required**              | No                       |
| **Additional properties** | Not allowed              |
| **Defined in**            | #/$defs/SchedulerOptions |

| Property                                              | Pattern | Type             | Deprecated | Definition | Title/Description                                                                                                                      |
| ----------------------------------------------------- | ------- | ---------------- | ---------- | ---------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| + [type](#train_scheduler_anyOf_i1_type )             | No      | string           | No         | -          | -                                                                                                                                      |
| + [milestones](#train_scheduler_anyOf_i1_milestones ) | No      | array of integer | No         | -          | List of milestones, iterations where the learning rate is reduced.                                                                     |
| + [gamma](#train_scheduler_anyOf_i1_gamma )           | No      | number           | No         | -          | At each milestone, the learning rate is multiplied by this number, so a gamma of 0.5 cuts the learning rate in half at each milestone. |

##### <a name="train_scheduler_anyOf_i1_type"></a>55.6.2.1. Property `ReduxOptions > train > scheduler > anyOf > SchedulerOptions > type`

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | Yes      |

##### <a name="train_scheduler_anyOf_i1_milestones"></a>55.6.2.2. Property `ReduxOptions > train > scheduler > anyOf > SchedulerOptions > milestones`

|              |                    |
| ------------ | ------------------ |
| **Type**     | `array of integer` |
| **Required** | Yes                |

**Description:** List of milestones, iterations where the learning rate is reduced.

|                      | Array restrictions |
| -------------------- | ------------------ |
| **Min items**        | N/A                |
| **Max items**        | N/A                |
| **Items unicity**    | False              |
| **Additional items** | False              |
| **Tuple validation** | See below          |

| Each item of this array must be                                | Description |
| -------------------------------------------------------------- | ----------- |
| [milestones items](#train_scheduler_anyOf_i1_milestones_items) | -           |

###### <a name="train_scheduler_anyOf_i1_milestones_items"></a>55.6.2.2.1. ReduxOptions > train > scheduler > anyOf > SchedulerOptions > milestones > milestones items

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | No        |

##### <a name="train_scheduler_anyOf_i1_gamma"></a>55.6.2.3. Property `ReduxOptions > train > scheduler > anyOf > SchedulerOptions > gamma`

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | Yes      |

**Description:** At each milestone, the learning rate is multiplied by this number, so a gamma of 0.5 cuts the learning rate in half at each milestone.

### <a name="train_optim_d"></a>55.7. Property `ReduxOptions > train > optim_d`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `combining`      |
| **Required**              | No               |
| **Additional properties** | Any type allowed |
| **Default**               | `null`           |

**Description:** The optimizer to use for the discriminator model.

| Any of(Option)                    |
| --------------------------------- |
| [item 0](#train_optim_d_anyOf_i0) |
| [item 1](#train_optim_d_anyOf_i1) |

#### <a name="train_optim_d_anyOf_i0"></a>55.7.1. Property `ReduxOptions > train > optim_d > anyOf > item 0`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `object`         |
| **Required**              | No               |
| **Additional properties** | Any type allowed |

#### <a name="train_optim_d_anyOf_i1"></a>55.7.2. Property `ReduxOptions > train > optim_d > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

### <a name="train_losses"></a>55.8. Property `ReduxOptions > train > losses`

|              |         |
| ------------ | ------- |
| **Type**     | `array` |
| **Required** | No      |
| **Default**  | `null`  |

**Description:** The list of loss functions to optimize.

|                      | Array restrictions |
| -------------------- | ------------------ |
| **Min items**        | N/A                |
| **Max items**        | N/A                |
| **Items unicity**    | False              |
| **Additional items** | False              |
| **Tuple validation** | See below          |

| Each item of this array must be     | Description |
| ----------------------------------- | ----------- |
| [losses items](#train_losses_items) | -           |

#### <a name="train_losses_items"></a>55.8.1. ReduxOptions > train > losses > losses items

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `combining`      |
| **Required**              | No               |
| **Additional properties** | Any type allowed |

| Any of(Option)                                      |
| --------------------------------------------------- |
| [ganloss](#train_losses_items_anyOf_i0)             |
| [multiscaleganloss](#train_losses_items_anyOf_i1)   |
| [adistsloss](#train_losses_items_anyOf_i2)          |
| [l1loss](#train_losses_items_anyOf_i3)              |
| [mseloss](#train_losses_items_anyOf_i4)             |
| [charbonnierloss](#train_losses_items_anyOf_i5)     |
| [colorloss](#train_losses_items_anyOf_i6)           |
| [averageloss](#train_losses_items_anyOf_i7)         |
| [bicubicloss](#train_losses_items_anyOf_i8)         |
| [lumaloss](#train_losses_items_anyOf_i9)            |
| [hsluvloss](#train_losses_items_anyOf_i10)          |
| [contextualloss](#train_losses_items_anyOf_i11)     |
| [distsloss](#train_losses_items_anyOf_i12)          |
| [ffloss](#train_losses_items_anyOf_i13)             |
| [ldlloss](#train_losses_items_anyOf_i14)            |
| [mssimloss](#train_losses_items_anyOf_i15)          |
| [msssiml1loss](#train_losses_items_anyOf_i16)       |
| [nccloss](#train_losses_items_anyOf_i17)            |
| [perceptualfp16loss](#train_losses_items_anyOf_i18) |
| [perceptualloss](#train_losses_items_anyOf_i19)     |
| [topiqloss](#train_losses_items_anyOf_i20)          |

##### <a name="train_losses_items_anyOf_i0"></a>55.8.1.1. Property `ReduxOptions > train > losses > losses items > anyOf > ganloss`

**Title:** ganloss

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `object`         |
| **Required**              | No               |
| **Additional properties** | Any type allowed |
| **Defined in**            | #/$defs/ganloss  |

| Property                                                         | Pattern | Type             | Deprecated | Definition | Title/Description |
| ---------------------------------------------------------------- | ------- | ---------------- | ---------- | ---------- | ----------------- |
| + [type](#train_losses_items_anyOf_i0_type )                     | No      | enum (of string) | No         | -          | -                 |
| + [loss_weight](#train_losses_items_anyOf_i0_loss_weight )       | No      | number           | No         | -          | -                 |
| - [gan_type](#train_losses_items_anyOf_i0_gan_type )             | No      | string           | No         | -          | -                 |
| - [real_label_val](#train_losses_items_anyOf_i0_real_label_val ) | No      | number           | No         | -          | -                 |
| - [fake_label_val](#train_losses_items_anyOf_i0_fake_label_val ) | No      | number           | No         | -          | -                 |

###### <a name="train_losses_items_anyOf_i0_type"></a>55.8.1.1.1. Property `ReduxOptions > train > losses > losses items > anyOf > ganloss > type`

|              |                    |
| ------------ | ------------------ |
| **Type**     | `enum (of string)` |
| **Required** | Yes                |

Must be one of:
* "ganloss"

###### <a name="train_losses_items_anyOf_i0_loss_weight"></a>55.8.1.1.2. Property `ReduxOptions > train > losses > losses items > anyOf > ganloss > loss_weight`

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | Yes      |

###### <a name="train_losses_items_anyOf_i0_gan_type"></a>55.8.1.1.3. Property `ReduxOptions > train > losses > losses items > anyOf > ganloss > gan_type`

|              |             |
| ------------ | ----------- |
| **Type**     | `string`    |
| **Required** | No          |
| **Default**  | `"vanilla"` |

###### <a name="train_losses_items_anyOf_i0_real_label_val"></a>55.8.1.1.4. Property `ReduxOptions > train > losses > losses items > anyOf > ganloss > real_label_val`

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |
| **Default**  | `1.0`    |

###### <a name="train_losses_items_anyOf_i0_fake_label_val"></a>55.8.1.1.5. Property `ReduxOptions > train > losses > losses items > anyOf > ganloss > fake_label_val`

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |
| **Default**  | `0.0`    |

##### <a name="train_losses_items_anyOf_i1"></a>55.8.1.2. Property `ReduxOptions > train > losses > losses items > anyOf > multiscaleganloss`

**Title:** multiscaleganloss

|                           |                           |
| ------------------------- | ------------------------- |
| **Type**                  | `object`                  |
| **Required**              | No                        |
| **Additional properties** | Any type allowed          |
| **Defined in**            | #/$defs/multiscaleganloss |

| Property                                                         | Pattern | Type             | Deprecated | Definition | Title/Description |
| ---------------------------------------------------------------- | ------- | ---------------- | ---------- | ---------- | ----------------- |
| + [type](#train_losses_items_anyOf_i1_type )                     | No      | enum (of string) | No         | -          | -                 |
| + [loss_weight](#train_losses_items_anyOf_i1_loss_weight )       | No      | number           | No         | -          | -                 |
| + [gan_type](#train_losses_items_anyOf_i1_gan_type )             | No      | string           | No         | -          | -                 |
| - [real_label_val](#train_losses_items_anyOf_i1_real_label_val ) | No      | number           | No         | -          | -                 |
| - [fake_label_val](#train_losses_items_anyOf_i1_fake_label_val ) | No      | number           | No         | -          | -                 |

###### <a name="train_losses_items_anyOf_i1_type"></a>55.8.1.2.1. Property `ReduxOptions > train > losses > losses items > anyOf > multiscaleganloss > type`

|              |                    |
| ------------ | ------------------ |
| **Type**     | `enum (of string)` |
| **Required** | Yes                |

Must be one of:
* "multiscaleganloss"

###### <a name="train_losses_items_anyOf_i1_loss_weight"></a>55.8.1.2.2. Property `ReduxOptions > train > losses > losses items > anyOf > multiscaleganloss > loss_weight`

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | Yes      |

###### <a name="train_losses_items_anyOf_i1_gan_type"></a>55.8.1.2.3. Property `ReduxOptions > train > losses > losses items > anyOf > multiscaleganloss > gan_type`

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | Yes      |

###### <a name="train_losses_items_anyOf_i1_real_label_val"></a>55.8.1.2.4. Property `ReduxOptions > train > losses > losses items > anyOf > multiscaleganloss > real_label_val`

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |
| **Default**  | `1.0`    |

###### <a name="train_losses_items_anyOf_i1_fake_label_val"></a>55.8.1.2.5. Property `ReduxOptions > train > losses > losses items > anyOf > multiscaleganloss > fake_label_val`

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |
| **Default**  | `0.0`    |

##### <a name="train_losses_items_anyOf_i2"></a>55.8.1.3. Property `ReduxOptions > train > losses > losses items > anyOf > adistsloss`

**Title:** adistsloss

|                           |                    |
| ------------------------- | ------------------ |
| **Type**                  | `object`           |
| **Required**              | No                 |
| **Additional properties** | Any type allowed   |
| **Defined in**            | #/$defs/adistsloss |

| Property                                                     | Pattern | Type             | Deprecated | Definition | Title/Description |
| ------------------------------------------------------------ | ------- | ---------------- | ---------- | ---------- | ----------------- |
| + [type](#train_losses_items_anyOf_i2_type )                 | No      | enum (of string) | No         | -          | -                 |
| + [loss_weight](#train_losses_items_anyOf_i2_loss_weight )   | No      | number           | No         | -          | -                 |
| - [window_size](#train_losses_items_anyOf_i2_window_size )   | No      | integer          | No         | -          | -                 |
| - [resize_input](#train_losses_items_anyOf_i2_resize_input ) | No      | boolean          | No         | -          | -                 |

###### <a name="train_losses_items_anyOf_i2_type"></a>55.8.1.3.1. Property `ReduxOptions > train > losses > losses items > anyOf > adistsloss > type`

|              |                    |
| ------------ | ------------------ |
| **Type**     | `enum (of string)` |
| **Required** | Yes                |

Must be one of:
* "adistsloss"

###### <a name="train_losses_items_anyOf_i2_loss_weight"></a>55.8.1.3.2. Property `ReduxOptions > train > losses > losses items > anyOf > adistsloss > loss_weight`

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | Yes      |

###### <a name="train_losses_items_anyOf_i2_window_size"></a>55.8.1.3.3. Property `ReduxOptions > train > losses > losses items > anyOf > adistsloss > window_size`

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | No        |
| **Default**  | `21`      |

###### <a name="train_losses_items_anyOf_i2_resize_input"></a>55.8.1.3.4. Property `ReduxOptions > train > losses > losses items > anyOf > adistsloss > resize_input`

|              |           |
| ------------ | --------- |
| **Type**     | `boolean` |
| **Required** | No        |
| **Default**  | `false`   |

##### <a name="train_losses_items_anyOf_i3"></a>55.8.1.4. Property `ReduxOptions > train > losses > losses items > anyOf > l1loss`

**Title:** l1loss

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `object`         |
| **Required**              | No               |
| **Additional properties** | Any type allowed |
| **Defined in**            | #/$defs/l1loss   |

| Property                                                   | Pattern | Type             | Deprecated | Definition | Title/Description |
| ---------------------------------------------------------- | ------- | ---------------- | ---------- | ---------- | ----------------- |
| + [type](#train_losses_items_anyOf_i3_type )               | No      | enum (of string) | No         | -          | -                 |
| + [loss_weight](#train_losses_items_anyOf_i3_loss_weight ) | No      | number           | No         | -          | -                 |
| - [reduction](#train_losses_items_anyOf_i3_reduction )     | No      | string           | No         | -          | -                 |

###### <a name="train_losses_items_anyOf_i3_type"></a>55.8.1.4.1. Property `ReduxOptions > train > losses > losses items > anyOf > l1loss > type`

|              |                    |
| ------------ | ------------------ |
| **Type**     | `enum (of string)` |
| **Required** | Yes                |

Must be one of:
* "l1loss"

###### <a name="train_losses_items_anyOf_i3_loss_weight"></a>55.8.1.4.2. Property `ReduxOptions > train > losses > losses items > anyOf > l1loss > loss_weight`

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | Yes      |

###### <a name="train_losses_items_anyOf_i3_reduction"></a>55.8.1.4.3. Property `ReduxOptions > train > losses > losses items > anyOf > l1loss > reduction`

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | No       |
| **Default**  | `"mean"` |

##### <a name="train_losses_items_anyOf_i4"></a>55.8.1.5. Property `ReduxOptions > train > losses > losses items > anyOf > mseloss`

**Title:** mseloss

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `object`         |
| **Required**              | No               |
| **Additional properties** | Any type allowed |
| **Defined in**            | #/$defs/mseloss  |

| Property                                                   | Pattern | Type             | Deprecated | Definition | Title/Description |
| ---------------------------------------------------------- | ------- | ---------------- | ---------- | ---------- | ----------------- |
| + [type](#train_losses_items_anyOf_i4_type )               | No      | enum (of string) | No         | -          | -                 |
| + [loss_weight](#train_losses_items_anyOf_i4_loss_weight ) | No      | number           | No         | -          | -                 |
| - [reduction](#train_losses_items_anyOf_i4_reduction )     | No      | string           | No         | -          | -                 |

###### <a name="train_losses_items_anyOf_i4_type"></a>55.8.1.5.1. Property `ReduxOptions > train > losses > losses items > anyOf > mseloss > type`

|              |                    |
| ------------ | ------------------ |
| **Type**     | `enum (of string)` |
| **Required** | Yes                |

Must be one of:
* "mseloss"

###### <a name="train_losses_items_anyOf_i4_loss_weight"></a>55.8.1.5.2. Property `ReduxOptions > train > losses > losses items > anyOf > mseloss > loss_weight`

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | Yes      |

###### <a name="train_losses_items_anyOf_i4_reduction"></a>55.8.1.5.3. Property `ReduxOptions > train > losses > losses items > anyOf > mseloss > reduction`

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | No       |
| **Default**  | `"mean"` |

##### <a name="train_losses_items_anyOf_i5"></a>55.8.1.6. Property `ReduxOptions > train > losses > losses items > anyOf > charbonnierloss`

**Title:** charbonnierloss

|                           |                         |
| ------------------------- | ----------------------- |
| **Type**                  | `object`                |
| **Required**              | No                      |
| **Additional properties** | Any type allowed        |
| **Defined in**            | #/$defs/charbonnierloss |

| Property                                                   | Pattern | Type             | Deprecated | Definition | Title/Description |
| ---------------------------------------------------------- | ------- | ---------------- | ---------- | ---------- | ----------------- |
| + [type](#train_losses_items_anyOf_i5_type )               | No      | enum (of string) | No         | -          | -                 |
| + [loss_weight](#train_losses_items_anyOf_i5_loss_weight ) | No      | number           | No         | -          | -                 |
| - [reduction](#train_losses_items_anyOf_i5_reduction )     | No      | string           | No         | -          | -                 |
| - [eps](#train_losses_items_anyOf_i5_eps )                 | No      | number           | No         | -          | -                 |

###### <a name="train_losses_items_anyOf_i5_type"></a>55.8.1.6.1. Property `ReduxOptions > train > losses > losses items > anyOf > charbonnierloss > type`

|              |                    |
| ------------ | ------------------ |
| **Type**     | `enum (of string)` |
| **Required** | Yes                |

Must be one of:
* "charbonnierloss"

###### <a name="train_losses_items_anyOf_i5_loss_weight"></a>55.8.1.6.2. Property `ReduxOptions > train > losses > losses items > anyOf > charbonnierloss > loss_weight`

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | Yes      |

###### <a name="train_losses_items_anyOf_i5_reduction"></a>55.8.1.6.3. Property `ReduxOptions > train > losses > losses items > anyOf > charbonnierloss > reduction`

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | No       |
| **Default**  | `"mean"` |

###### <a name="train_losses_items_anyOf_i5_eps"></a>55.8.1.6.4. Property `ReduxOptions > train > losses > losses items > anyOf > charbonnierloss > eps`

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |
| **Default**  | `1e-12`  |

##### <a name="train_losses_items_anyOf_i6"></a>55.8.1.7. Property `ReduxOptions > train > losses > losses items > anyOf > colorloss`

**Title:** colorloss

|                           |                   |
| ------------------------- | ----------------- |
| **Type**                  | `object`          |
| **Required**              | No                |
| **Additional properties** | Any type allowed  |
| **Defined in**            | #/$defs/colorloss |

| Property                                                   | Pattern | Type             | Deprecated | Definition | Title/Description |
| ---------------------------------------------------------- | ------- | ---------------- | ---------- | ---------- | ----------------- |
| + [type](#train_losses_items_anyOf_i6_type )               | No      | enum (of string) | No         | -          | -                 |
| + [loss_weight](#train_losses_items_anyOf_i6_loss_weight ) | No      | number           | No         | -          | -                 |
| - [criterion](#train_losses_items_anyOf_i6_criterion )     | No      | string           | No         | -          | -                 |
| - [scale](#train_losses_items_anyOf_i6_scale )             | No      | integer          | No         | -          | -                 |

###### <a name="train_losses_items_anyOf_i6_type"></a>55.8.1.7.1. Property `ReduxOptions > train > losses > losses items > anyOf > colorloss > type`

|              |                    |
| ------------ | ------------------ |
| **Type**     | `enum (of string)` |
| **Required** | Yes                |

Must be one of:
* "colorloss"

###### <a name="train_losses_items_anyOf_i6_loss_weight"></a>55.8.1.7.2. Property `ReduxOptions > train > losses > losses items > anyOf > colorloss > loss_weight`

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | Yes      |

###### <a name="train_losses_items_anyOf_i6_criterion"></a>55.8.1.7.3. Property `ReduxOptions > train > losses > losses items > anyOf > colorloss > criterion`

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | No       |
| **Default**  | `"l1"`   |

###### <a name="train_losses_items_anyOf_i6_scale"></a>55.8.1.7.4. Property `ReduxOptions > train > losses > losses items > anyOf > colorloss > scale`

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | No        |
| **Default**  | `4`       |

##### <a name="train_losses_items_anyOf_i7"></a>55.8.1.8. Property `ReduxOptions > train > losses > losses items > anyOf > averageloss`

**Title:** averageloss

|                           |                     |
| ------------------------- | ------------------- |
| **Type**                  | `object`            |
| **Required**              | No                  |
| **Additional properties** | Any type allowed    |
| **Defined in**            | #/$defs/averageloss |

| Property                                                   | Pattern | Type             | Deprecated | Definition | Title/Description |
| ---------------------------------------------------------- | ------- | ---------------- | ---------- | ---------- | ----------------- |
| + [type](#train_losses_items_anyOf_i7_type )               | No      | enum (of string) | No         | -          | -                 |
| + [loss_weight](#train_losses_items_anyOf_i7_loss_weight ) | No      | number           | No         | -          | -                 |
| - [criterion](#train_losses_items_anyOf_i7_criterion )     | No      | string           | No         | -          | -                 |
| - [scale](#train_losses_items_anyOf_i7_scale )             | No      | integer          | No         | -          | -                 |

###### <a name="train_losses_items_anyOf_i7_type"></a>55.8.1.8.1. Property `ReduxOptions > train > losses > losses items > anyOf > averageloss > type`

|              |                    |
| ------------ | ------------------ |
| **Type**     | `enum (of string)` |
| **Required** | Yes                |

Must be one of:
* "averageloss"

###### <a name="train_losses_items_anyOf_i7_loss_weight"></a>55.8.1.8.2. Property `ReduxOptions > train > losses > losses items > anyOf > averageloss > loss_weight`

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | Yes      |

###### <a name="train_losses_items_anyOf_i7_criterion"></a>55.8.1.8.3. Property `ReduxOptions > train > losses > losses items > anyOf > averageloss > criterion`

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | No       |
| **Default**  | `"l1"`   |

###### <a name="train_losses_items_anyOf_i7_scale"></a>55.8.1.8.4. Property `ReduxOptions > train > losses > losses items > anyOf > averageloss > scale`

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | No        |
| **Default**  | `4`       |

##### <a name="train_losses_items_anyOf_i8"></a>55.8.1.9. Property `ReduxOptions > train > losses > losses items > anyOf > bicubicloss`

**Title:** bicubicloss

|                           |                     |
| ------------------------- | ------------------- |
| **Type**                  | `object`            |
| **Required**              | No                  |
| **Additional properties** | Any type allowed    |
| **Defined in**            | #/$defs/bicubicloss |

| Property                                                   | Pattern | Type             | Deprecated | Definition | Title/Description |
| ---------------------------------------------------------- | ------- | ---------------- | ---------- | ---------- | ----------------- |
| + [type](#train_losses_items_anyOf_i8_type )               | No      | enum (of string) | No         | -          | -                 |
| + [loss_weight](#train_losses_items_anyOf_i8_loss_weight ) | No      | number           | No         | -          | -                 |
| - [criterion](#train_losses_items_anyOf_i8_criterion )     | No      | string           | No         | -          | -                 |
| - [scale](#train_losses_items_anyOf_i8_scale )             | No      | integer          | No         | -          | -                 |

###### <a name="train_losses_items_anyOf_i8_type"></a>55.8.1.9.1. Property `ReduxOptions > train > losses > losses items > anyOf > bicubicloss > type`

|              |                    |
| ------------ | ------------------ |
| **Type**     | `enum (of string)` |
| **Required** | Yes                |

Must be one of:
* "bicubicloss"

###### <a name="train_losses_items_anyOf_i8_loss_weight"></a>55.8.1.9.2. Property `ReduxOptions > train > losses > losses items > anyOf > bicubicloss > loss_weight`

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | Yes      |

###### <a name="train_losses_items_anyOf_i8_criterion"></a>55.8.1.9.3. Property `ReduxOptions > train > losses > losses items > anyOf > bicubicloss > criterion`

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | No       |
| **Default**  | `"l1"`   |

###### <a name="train_losses_items_anyOf_i8_scale"></a>55.8.1.9.4. Property `ReduxOptions > train > losses > losses items > anyOf > bicubicloss > scale`

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | No        |
| **Default**  | `4`       |

##### <a name="train_losses_items_anyOf_i9"></a>55.8.1.10. Property `ReduxOptions > train > losses > losses items > anyOf > lumaloss`

**Title:** lumaloss

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `object`         |
| **Required**              | No               |
| **Additional properties** | Any type allowed |
| **Defined in**            | #/$defs/lumaloss |

| Property                                                   | Pattern | Type             | Deprecated | Definition | Title/Description |
| ---------------------------------------------------------- | ------- | ---------------- | ---------- | ---------- | ----------------- |
| + [type](#train_losses_items_anyOf_i9_type )               | No      | enum (of string) | No         | -          | -                 |
| + [loss_weight](#train_losses_items_anyOf_i9_loss_weight ) | No      | number           | No         | -          | -                 |
| - [criterion](#train_losses_items_anyOf_i9_criterion )     | No      | string           | No         | -          | -                 |

###### <a name="train_losses_items_anyOf_i9_type"></a>55.8.1.10.1. Property `ReduxOptions > train > losses > losses items > anyOf > lumaloss > type`

|              |                    |
| ------------ | ------------------ |
| **Type**     | `enum (of string)` |
| **Required** | Yes                |

Must be one of:
* "lumaloss"

###### <a name="train_losses_items_anyOf_i9_loss_weight"></a>55.8.1.10.2. Property `ReduxOptions > train > losses > losses items > anyOf > lumaloss > loss_weight`

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | Yes      |

###### <a name="train_losses_items_anyOf_i9_criterion"></a>55.8.1.10.3. Property `ReduxOptions > train > losses > losses items > anyOf > lumaloss > criterion`

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | No       |
| **Default**  | `"l1"`   |

##### <a name="train_losses_items_anyOf_i10"></a>55.8.1.11. Property `ReduxOptions > train > losses > losses items > anyOf > hsluvloss`

**Title:** hsluvloss

|                           |                   |
| ------------------------- | ----------------- |
| **Type**                  | `object`          |
| **Required**              | No                |
| **Additional properties** | Any type allowed  |
| **Defined in**            | #/$defs/hsluvloss |

| Property                                                    | Pattern | Type             | Deprecated | Definition | Title/Description |
| ----------------------------------------------------------- | ------- | ---------------- | ---------- | ---------- | ----------------- |
| + [type](#train_losses_items_anyOf_i10_type )               | No      | enum (of string) | No         | -          | -                 |
| + [loss_weight](#train_losses_items_anyOf_i10_loss_weight ) | No      | number           | No         | -          | -                 |
| - [criterion](#train_losses_items_anyOf_i10_criterion )     | No      | string           | No         | -          | -                 |

###### <a name="train_losses_items_anyOf_i10_type"></a>55.8.1.11.1. Property `ReduxOptions > train > losses > losses items > anyOf > hsluvloss > type`

|              |                    |
| ------------ | ------------------ |
| **Type**     | `enum (of string)` |
| **Required** | Yes                |

Must be one of:
* "hsluvloss"

###### <a name="train_losses_items_anyOf_i10_loss_weight"></a>55.8.1.11.2. Property `ReduxOptions > train > losses > losses items > anyOf > hsluvloss > loss_weight`

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | Yes      |

###### <a name="train_losses_items_anyOf_i10_criterion"></a>55.8.1.11.3. Property `ReduxOptions > train > losses > losses items > anyOf > hsluvloss > criterion`

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | No       |
| **Default**  | `"l1"`   |

##### <a name="train_losses_items_anyOf_i11"></a>55.8.1.12. Property `ReduxOptions > train > losses > losses items > anyOf > contextualloss`

**Title:** contextualloss

|                           |                        |
| ------------------------- | ---------------------- |
| **Type**                  | `object`               |
| **Required**              | No                     |
| **Additional properties** | Any type allowed       |
| **Defined in**            | #/$defs/contextualloss |

| Property                                                        | Pattern | Type             | Deprecated | Definition | Title/Description |
| --------------------------------------------------------------- | ------- | ---------------- | ---------- | ---------- | ----------------- |
| + [type](#train_losses_items_anyOf_i11_type )                   | No      | enum (of string) | No         | -          | -                 |
| + [loss_weight](#train_losses_items_anyOf_i11_loss_weight )     | No      | number           | No         | -          | -                 |
| - [layer_weights](#train_losses_items_anyOf_i11_layer_weights ) | No      | Combination      | No         | -          | -                 |
| - [crop_quarter](#train_losses_items_anyOf_i11_crop_quarter )   | No      | boolean          | No         | -          | -                 |
| - [max_1d_size](#train_losses_items_anyOf_i11_max_1d_size )     | No      | integer          | No         | -          | -                 |
| - [distance_type](#train_losses_items_anyOf_i11_distance_type ) | No      | string           | No         | -          | -                 |
| - [b](#train_losses_items_anyOf_i11_b )                         | No      | number           | No         | -          | -                 |
| - [band_width](#train_losses_items_anyOf_i11_band_width )       | No      | number           | No         | -          | -                 |
| - [use_vgg](#train_losses_items_anyOf_i11_use_vgg )             | No      | boolean          | No         | -          | -                 |
| - [net](#train_losses_items_anyOf_i11_net )                     | No      | string           | No         | -          | -                 |
| - [calc_type](#train_losses_items_anyOf_i11_calc_type )         | No      | string           | No         | -          | -                 |
| - [z_norm](#train_losses_items_anyOf_i11_z_norm )               | No      | boolean          | No         | -          | -                 |

###### <a name="train_losses_items_anyOf_i11_type"></a>55.8.1.12.1. Property `ReduxOptions > train > losses > losses items > anyOf > contextualloss > type`

|              |                    |
| ------------ | ------------------ |
| **Type**     | `enum (of string)` |
| **Required** | Yes                |

Must be one of:
* "contextualloss"

###### <a name="train_losses_items_anyOf_i11_loss_weight"></a>55.8.1.12.2. Property `ReduxOptions > train > losses > losses items > anyOf > contextualloss > loss_weight`

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | Yes      |

###### <a name="train_losses_items_anyOf_i11_layer_weights"></a>55.8.1.12.3. Property `ReduxOptions > train > losses > losses items > anyOf > contextualloss > layer_weights`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `combining`      |
| **Required**              | No               |
| **Additional properties** | Any type allowed |
| **Default**               | `null`           |

| Any of(Option)                                                 |
| -------------------------------------------------------------- |
| [item 0](#train_losses_items_anyOf_i11_layer_weights_anyOf_i0) |
| [item 1](#train_losses_items_anyOf_i11_layer_weights_anyOf_i1) |

###### <a name="train_losses_items_anyOf_i11_layer_weights_anyOf_i0"></a>55.8.1.12.3.1. Property `ReduxOptions > train > losses > losses items > anyOf > contextualloss > layer_weights > anyOf > item 0`

|                           |                                                                                                                                  |
| ------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| **Type**                  | `object`                                                                                                                         |
| **Required**              | No                                                                                                                               |
| **Additional properties** | [Each additional property must conform to the schema](#train_losses_items_anyOf_i11_layer_weights_anyOf_i0_additionalProperties) |

| Property                                                                         | Pattern | Type   | Deprecated | Definition | Title/Description |
| -------------------------------------------------------------------------------- | ------- | ------ | ---------- | ---------- | ----------------- |
| - [](#train_losses_items_anyOf_i11_layer_weights_anyOf_i0_additionalProperties ) | No      | number | No         | -          | -                 |

###### <a name="train_losses_items_anyOf_i11_layer_weights_anyOf_i0_additionalProperties"></a>55.8.1.12.3.1.1. Property `ReduxOptions > train > losses > losses items > anyOf > contextualloss > layer_weights > anyOf > item 0 > additionalProperties`

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |

###### <a name="train_losses_items_anyOf_i11_layer_weights_anyOf_i1"></a>55.8.1.12.3.2. Property `ReduxOptions > train > losses > losses items > anyOf > contextualloss > layer_weights > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

###### <a name="train_losses_items_anyOf_i11_crop_quarter"></a>55.8.1.12.4. Property `ReduxOptions > train > losses > losses items > anyOf > contextualloss > crop_quarter`

|              |           |
| ------------ | --------- |
| **Type**     | `boolean` |
| **Required** | No        |
| **Default**  | `false`   |

###### <a name="train_losses_items_anyOf_i11_max_1d_size"></a>55.8.1.12.5. Property `ReduxOptions > train > losses > losses items > anyOf > contextualloss > max_1d_size`

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | No        |
| **Default**  | `100`     |

###### <a name="train_losses_items_anyOf_i11_distance_type"></a>55.8.1.12.6. Property `ReduxOptions > train > losses > losses items > anyOf > contextualloss > distance_type`

|              |            |
| ------------ | ---------- |
| **Type**     | `string`   |
| **Required** | No         |
| **Default**  | `"cosine"` |

###### <a name="train_losses_items_anyOf_i11_b"></a>55.8.1.12.7. Property `ReduxOptions > train > losses > losses items > anyOf > contextualloss > b`

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |
| **Default**  | `1.0`    |

###### <a name="train_losses_items_anyOf_i11_band_width"></a>55.8.1.12.8. Property `ReduxOptions > train > losses > losses items > anyOf > contextualloss > band_width`

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |
| **Default**  | `0.5`    |

###### <a name="train_losses_items_anyOf_i11_use_vgg"></a>55.8.1.12.9. Property `ReduxOptions > train > losses > losses items > anyOf > contextualloss > use_vgg`

|              |           |
| ------------ | --------- |
| **Type**     | `boolean` |
| **Required** | No        |
| **Default**  | `true`    |

###### <a name="train_losses_items_anyOf_i11_net"></a>55.8.1.12.10. Property `ReduxOptions > train > losses > losses items > anyOf > contextualloss > net`

|              |           |
| ------------ | --------- |
| **Type**     | `string`  |
| **Required** | No        |
| **Default**  | `"vgg19"` |

###### <a name="train_losses_items_anyOf_i11_calc_type"></a>55.8.1.12.11. Property `ReduxOptions > train > losses > losses items > anyOf > contextualloss > calc_type`

|              |             |
| ------------ | ----------- |
| **Type**     | `string`    |
| **Required** | No          |
| **Default**  | `"regular"` |

###### <a name="train_losses_items_anyOf_i11_z_norm"></a>55.8.1.12.12. Property `ReduxOptions > train > losses > losses items > anyOf > contextualloss > z_norm`

|              |           |
| ------------ | --------- |
| **Type**     | `boolean` |
| **Required** | No        |
| **Default**  | `false`   |

##### <a name="train_losses_items_anyOf_i12"></a>55.8.1.13. Property `ReduxOptions > train > losses > losses items > anyOf > distsloss`

**Title:** distsloss

|                           |                   |
| ------------------------- | ----------------- |
| **Type**                  | `object`          |
| **Required**              | No                |
| **Additional properties** | Any type allowed  |
| **Defined in**            | #/$defs/distsloss |

| Property                                                          | Pattern | Type             | Deprecated | Definition | Title/Description |
| ----------------------------------------------------------------- | ------- | ---------------- | ---------- | ---------- | ----------------- |
| + [type](#train_losses_items_anyOf_i12_type )                     | No      | enum (of string) | No         | -          | -                 |
| + [loss_weight](#train_losses_items_anyOf_i12_loss_weight )       | No      | number           | No         | -          | -                 |
| - [as_loss](#train_losses_items_anyOf_i12_as_loss )               | No      | boolean          | No         | -          | -                 |
| - [load_weights](#train_losses_items_anyOf_i12_load_weights )     | No      | boolean          | No         | -          | -                 |
| - [use_input_norm](#train_losses_items_anyOf_i12_use_input_norm ) | No      | boolean          | No         | -          | -                 |
| - [clip_min](#train_losses_items_anyOf_i12_clip_min )             | No      | integer          | No         | -          | -                 |

###### <a name="train_losses_items_anyOf_i12_type"></a>55.8.1.13.1. Property `ReduxOptions > train > losses > losses items > anyOf > distsloss > type`

|              |                    |
| ------------ | ------------------ |
| **Type**     | `enum (of string)` |
| **Required** | Yes                |

Must be one of:
* "distsloss"

###### <a name="train_losses_items_anyOf_i12_loss_weight"></a>55.8.1.13.2. Property `ReduxOptions > train > losses > losses items > anyOf > distsloss > loss_weight`

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | Yes      |

###### <a name="train_losses_items_anyOf_i12_as_loss"></a>55.8.1.13.3. Property `ReduxOptions > train > losses > losses items > anyOf > distsloss > as_loss`

|              |           |
| ------------ | --------- |
| **Type**     | `boolean` |
| **Required** | No        |
| **Default**  | `true`    |

###### <a name="train_losses_items_anyOf_i12_load_weights"></a>55.8.1.13.4. Property `ReduxOptions > train > losses > losses items > anyOf > distsloss > load_weights`

|              |           |
| ------------ | --------- |
| **Type**     | `boolean` |
| **Required** | No        |
| **Default**  | `true`    |

###### <a name="train_losses_items_anyOf_i12_use_input_norm"></a>55.8.1.13.5. Property `ReduxOptions > train > losses > losses items > anyOf > distsloss > use_input_norm`

|              |           |
| ------------ | --------- |
| **Type**     | `boolean` |
| **Required** | No        |
| **Default**  | `true`    |

###### <a name="train_losses_items_anyOf_i12_clip_min"></a>55.8.1.13.6. Property `ReduxOptions > train > losses > losses items > anyOf > distsloss > clip_min`

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | No        |
| **Default**  | `0`       |

##### <a name="train_losses_items_anyOf_i13"></a>55.8.1.14. Property `ReduxOptions > train > losses > losses items > anyOf > ffloss`

**Title:** ffloss

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `object`         |
| **Required**              | No               |
| **Additional properties** | Any type allowed |
| **Defined in**            | #/$defs/ffloss   |

| Property                                                      | Pattern | Type             | Deprecated | Definition | Title/Description |
| ------------------------------------------------------------- | ------- | ---------------- | ---------- | ---------- | ----------------- |
| + [type](#train_losses_items_anyOf_i13_type )                 | No      | enum (of string) | No         | -          | -                 |
| + [loss_weight](#train_losses_items_anyOf_i13_loss_weight )   | No      | number           | No         | -          | -                 |
| - [alpha](#train_losses_items_anyOf_i13_alpha )               | No      | number           | No         | -          | -                 |
| - [patch_factor](#train_losses_items_anyOf_i13_patch_factor ) | No      | integer          | No         | -          | -                 |
| - [ave_spectrum](#train_losses_items_anyOf_i13_ave_spectrum ) | No      | boolean          | No         | -          | -                 |
| - [log_matrix](#train_losses_items_anyOf_i13_log_matrix )     | No      | boolean          | No         | -          | -                 |
| - [batch_matrix](#train_losses_items_anyOf_i13_batch_matrix ) | No      | boolean          | No         | -          | -                 |

###### <a name="train_losses_items_anyOf_i13_type"></a>55.8.1.14.1. Property `ReduxOptions > train > losses > losses items > anyOf > ffloss > type`

|              |                    |
| ------------ | ------------------ |
| **Type**     | `enum (of string)` |
| **Required** | Yes                |

Must be one of:
* "ffloss"

###### <a name="train_losses_items_anyOf_i13_loss_weight"></a>55.8.1.14.2. Property `ReduxOptions > train > losses > losses items > anyOf > ffloss > loss_weight`

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | Yes      |

###### <a name="train_losses_items_anyOf_i13_alpha"></a>55.8.1.14.3. Property `ReduxOptions > train > losses > losses items > anyOf > ffloss > alpha`

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |
| **Default**  | `1.0`    |

###### <a name="train_losses_items_anyOf_i13_patch_factor"></a>55.8.1.14.4. Property `ReduxOptions > train > losses > losses items > anyOf > ffloss > patch_factor`

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | No        |
| **Default**  | `1`       |

###### <a name="train_losses_items_anyOf_i13_ave_spectrum"></a>55.8.1.14.5. Property `ReduxOptions > train > losses > losses items > anyOf > ffloss > ave_spectrum`

|              |           |
| ------------ | --------- |
| **Type**     | `boolean` |
| **Required** | No        |
| **Default**  | `true`    |

###### <a name="train_losses_items_anyOf_i13_log_matrix"></a>55.8.1.14.6. Property `ReduxOptions > train > losses > losses items > anyOf > ffloss > log_matrix`

|              |           |
| ------------ | --------- |
| **Type**     | `boolean` |
| **Required** | No        |
| **Default**  | `false`   |

###### <a name="train_losses_items_anyOf_i13_batch_matrix"></a>55.8.1.14.7. Property `ReduxOptions > train > losses > losses items > anyOf > ffloss > batch_matrix`

|              |           |
| ------------ | --------- |
| **Type**     | `boolean` |
| **Required** | No        |
| **Default**  | `false`   |

##### <a name="train_losses_items_anyOf_i14"></a>55.8.1.15. Property `ReduxOptions > train > losses > losses items > anyOf > ldlloss`

**Title:** ldlloss

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `object`         |
| **Required**              | No               |
| **Additional properties** | Any type allowed |
| **Defined in**            | #/$defs/ldlloss  |

| Property                                                    | Pattern | Type             | Deprecated | Definition | Title/Description |
| ----------------------------------------------------------- | ------- | ---------------- | ---------- | ---------- | ----------------- |
| + [type](#train_losses_items_anyOf_i14_type )               | No      | enum (of string) | No         | -          | -                 |
| + [loss_weight](#train_losses_items_anyOf_i14_loss_weight ) | No      | number           | No         | -          | -                 |
| - [criterion](#train_losses_items_anyOf_i14_criterion )     | No      | string           | No         | -          | -                 |

###### <a name="train_losses_items_anyOf_i14_type"></a>55.8.1.15.1. Property `ReduxOptions > train > losses > losses items > anyOf > ldlloss > type`

|              |                    |
| ------------ | ------------------ |
| **Type**     | `enum (of string)` |
| **Required** | Yes                |

Must be one of:
* "ldlloss"

###### <a name="train_losses_items_anyOf_i14_loss_weight"></a>55.8.1.15.2. Property `ReduxOptions > train > losses > losses items > anyOf > ldlloss > loss_weight`

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | Yes      |

###### <a name="train_losses_items_anyOf_i14_criterion"></a>55.8.1.15.3. Property `ReduxOptions > train > losses > losses items > anyOf > ldlloss > criterion`

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | No       |
| **Default**  | `"l1"`   |

##### <a name="train_losses_items_anyOf_i15"></a>55.8.1.16. Property `ReduxOptions > train > losses > losses items > anyOf > mssimloss`

**Title:** mssimloss

|                           |                   |
| ------------------------- | ----------------- |
| **Type**                  | `object`          |
| **Required**              | No                |
| **Additional properties** | Any type allowed  |
| **Defined in**            | #/$defs/mssimloss |

| Property                                                      | Pattern | Type             | Deprecated | Definition | Title/Description |
| ------------------------------------------------------------- | ------- | ---------------- | ---------- | ---------- | ----------------- |
| + [type](#train_losses_items_anyOf_i15_type )                 | No      | enum (of string) | No         | -          | -                 |
| + [loss_weight](#train_losses_items_anyOf_i15_loss_weight )   | No      | number           | No         | -          | -                 |
| - [window_size](#train_losses_items_anyOf_i15_window_size )   | No      | integer          | No         | -          | -                 |
| - [in_channels](#train_losses_items_anyOf_i15_in_channels )   | No      | integer          | No         | -          | -                 |
| - [sigma](#train_losses_items_anyOf_i15_sigma )               | No      | number           | No         | -          | -                 |
| - [k1](#train_losses_items_anyOf_i15_k1 )                     | No      | number           | No         | -          | -                 |
| - [k2](#train_losses_items_anyOf_i15_k2 )                     | No      | number           | No         | -          | -                 |
| - [l](#train_losses_items_anyOf_i15_l )                       | No      | integer          | No         | -          | -                 |
| - [padding](#train_losses_items_anyOf_i15_padding )           | No      | Combination      | No         | -          | -                 |
| - [cosim](#train_losses_items_anyOf_i15_cosim )               | No      | boolean          | No         | -          | -                 |
| - [cosim_lambda](#train_losses_items_anyOf_i15_cosim_lambda ) | No      | integer          | No         | -          | -                 |

###### <a name="train_losses_items_anyOf_i15_type"></a>55.8.1.16.1. Property `ReduxOptions > train > losses > losses items > anyOf > mssimloss > type`

|              |                    |
| ------------ | ------------------ |
| **Type**     | `enum (of string)` |
| **Required** | Yes                |

Must be one of:
* "mssimloss"

###### <a name="train_losses_items_anyOf_i15_loss_weight"></a>55.8.1.16.2. Property `ReduxOptions > train > losses > losses items > anyOf > mssimloss > loss_weight`

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | Yes      |

###### <a name="train_losses_items_anyOf_i15_window_size"></a>55.8.1.16.3. Property `ReduxOptions > train > losses > losses items > anyOf > mssimloss > window_size`

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | No        |
| **Default**  | `11`      |

###### <a name="train_losses_items_anyOf_i15_in_channels"></a>55.8.1.16.4. Property `ReduxOptions > train > losses > losses items > anyOf > mssimloss > in_channels`

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | No        |
| **Default**  | `3`       |

###### <a name="train_losses_items_anyOf_i15_sigma"></a>55.8.1.16.5. Property `ReduxOptions > train > losses > losses items > anyOf > mssimloss > sigma`

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |
| **Default**  | `1.5`    |

###### <a name="train_losses_items_anyOf_i15_k1"></a>55.8.1.16.6. Property `ReduxOptions > train > losses > losses items > anyOf > mssimloss > k1`

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |
| **Default**  | `0.01`   |

###### <a name="train_losses_items_anyOf_i15_k2"></a>55.8.1.16.7. Property `ReduxOptions > train > losses > losses items > anyOf > mssimloss > k2`

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |
| **Default**  | `0.03`   |

###### <a name="train_losses_items_anyOf_i15_l"></a>55.8.1.16.8. Property `ReduxOptions > train > losses > losses items > anyOf > mssimloss > l`

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | No        |
| **Default**  | `1`       |

###### <a name="train_losses_items_anyOf_i15_padding"></a>55.8.1.16.9. Property `ReduxOptions > train > losses > losses items > anyOf > mssimloss > padding`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `combining`      |
| **Required**              | No               |
| **Additional properties** | Any type allowed |
| **Default**               | `null`           |

| Any of(Option)                                           |
| -------------------------------------------------------- |
| [item 0](#train_losses_items_anyOf_i15_padding_anyOf_i0) |
| [item 1](#train_losses_items_anyOf_i15_padding_anyOf_i1) |

###### <a name="train_losses_items_anyOf_i15_padding_anyOf_i0"></a>55.8.1.16.9.1. Property `ReduxOptions > train > losses > losses items > anyOf > mssimloss > padding > anyOf > item 0`

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | No        |

###### <a name="train_losses_items_anyOf_i15_padding_anyOf_i1"></a>55.8.1.16.9.2. Property `ReduxOptions > train > losses > losses items > anyOf > mssimloss > padding > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

###### <a name="train_losses_items_anyOf_i15_cosim"></a>55.8.1.16.10. Property `ReduxOptions > train > losses > losses items > anyOf > mssimloss > cosim`

|              |           |
| ------------ | --------- |
| **Type**     | `boolean` |
| **Required** | No        |
| **Default**  | `true`    |

###### <a name="train_losses_items_anyOf_i15_cosim_lambda"></a>55.8.1.16.11. Property `ReduxOptions > train > losses > losses items > anyOf > mssimloss > cosim_lambda`

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | No        |
| **Default**  | `5`       |

##### <a name="train_losses_items_anyOf_i16"></a>55.8.1.17. Property `ReduxOptions > train > losses > losses items > anyOf > msssiml1loss`

**Title:** msssiml1loss

|                           |                      |
| ------------------------- | -------------------- |
| **Type**                  | `object`             |
| **Required**              | No                   |
| **Additional properties** | Any type allowed     |
| **Defined in**            | #/$defs/msssiml1loss |

| Property                                                            | Pattern | Type             | Deprecated | Definition | Title/Description |
| ------------------------------------------------------------------- | ------- | ---------------- | ---------- | ---------- | ----------------- |
| + [type](#train_losses_items_anyOf_i16_type )                       | No      | enum (of string) | No         | -          | -                 |
| + [loss_weight](#train_losses_items_anyOf_i16_loss_weight )         | No      | number           | No         | -          | -                 |
| - [gaussian_sigmas](#train_losses_items_anyOf_i16_gaussian_sigmas ) | No      | Combination      | No         | -          | -                 |
| - [data_range](#train_losses_items_anyOf_i16_data_range )           | No      | number           | No         | -          | -                 |
| - [k](#train_losses_items_anyOf_i16_k )                             | No      | array            | No         | -          | -                 |
| - [alpha](#train_losses_items_anyOf_i16_alpha )                     | No      | number           | No         | -          | -                 |
| - [cuda_dev](#train_losses_items_anyOf_i16_cuda_dev )               | No      | integer          | No         | -          | -                 |

###### <a name="train_losses_items_anyOf_i16_type"></a>55.8.1.17.1. Property `ReduxOptions > train > losses > losses items > anyOf > msssiml1loss > type`

|              |                    |
| ------------ | ------------------ |
| **Type**     | `enum (of string)` |
| **Required** | Yes                |

Must be one of:
* "msssiml1loss"

###### <a name="train_losses_items_anyOf_i16_loss_weight"></a>55.8.1.17.2. Property `ReduxOptions > train > losses > losses items > anyOf > msssiml1loss > loss_weight`

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | Yes      |

###### <a name="train_losses_items_anyOf_i16_gaussian_sigmas"></a>55.8.1.17.3. Property `ReduxOptions > train > losses > losses items > anyOf > msssiml1loss > gaussian_sigmas`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `combining`      |
| **Required**              | No               |
| **Additional properties** | Any type allowed |
| **Default**               | `null`           |

| Any of(Option)                                                   |
| ---------------------------------------------------------------- |
| [item 0](#train_losses_items_anyOf_i16_gaussian_sigmas_anyOf_i0) |
| [item 1](#train_losses_items_anyOf_i16_gaussian_sigmas_anyOf_i1) |

###### <a name="train_losses_items_anyOf_i16_gaussian_sigmas_anyOf_i0"></a>55.8.1.17.3.1. Property `ReduxOptions > train > losses > losses items > anyOf > msssiml1loss > gaussian_sigmas > anyOf > item 0`

|              |                   |
| ------------ | ----------------- |
| **Type**     | `array of number` |
| **Required** | No                |

|                      | Array restrictions |
| -------------------- | ------------------ |
| **Min items**        | N/A                |
| **Max items**        | N/A                |
| **Items unicity**    | False              |
| **Additional items** | False              |
| **Tuple validation** | See below          |

| Each item of this array must be                                              | Description |
| ---------------------------------------------------------------------------- | ----------- |
| [item 0 items](#train_losses_items_anyOf_i16_gaussian_sigmas_anyOf_i0_items) | -           |

###### <a name="train_losses_items_anyOf_i16_gaussian_sigmas_anyOf_i0_items"></a>55.8.1.17.3.1.1. ReduxOptions > train > losses > losses items > anyOf > msssiml1loss > gaussian_sigmas > anyOf > item 0 > item 0 items

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |

###### <a name="train_losses_items_anyOf_i16_gaussian_sigmas_anyOf_i1"></a>55.8.1.17.3.2. Property `ReduxOptions > train > losses > losses items > anyOf > msssiml1loss > gaussian_sigmas > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

###### <a name="train_losses_items_anyOf_i16_data_range"></a>55.8.1.17.4. Property `ReduxOptions > train > losses > losses items > anyOf > msssiml1loss > data_range`

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |
| **Default**  | `1.0`    |

###### <a name="train_losses_items_anyOf_i16_k"></a>55.8.1.17.5. Property `ReduxOptions > train > losses > losses items > anyOf > msssiml1loss > k`

|              |                |
| ------------ | -------------- |
| **Type**     | `array`        |
| **Required** | No             |
| **Default**  | `[0.01, 0.03]` |

|                      | Array restrictions |
| -------------------- | ------------------ |
| **Min items**        | 2                  |
| **Max items**        | 2                  |
| **Items unicity**    | False              |
| **Additional items** | False              |
| **Tuple validation** | See below          |

| Each item of this array must be                      | Description |
| ---------------------------------------------------- | ----------- |
| [k item 0](#train_losses_items_anyOf_i16_k_items_i0) | -           |
| [k item 1](#train_losses_items_anyOf_i16_k_items_i1) | -           |

###### <a name="autogenerated_heading_36"></a>55.8.1.17.5.1. ReduxOptions > train > losses > losses items > anyOf > msssiml1loss > k > k item 0

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |

###### <a name="autogenerated_heading_37"></a>55.8.1.17.5.2. ReduxOptions > train > losses > losses items > anyOf > msssiml1loss > k > k item 1

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |

###### <a name="train_losses_items_anyOf_i16_alpha"></a>55.8.1.17.6. Property `ReduxOptions > train > losses > losses items > anyOf > msssiml1loss > alpha`

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |
| **Default**  | `0.1`    |

###### <a name="train_losses_items_anyOf_i16_cuda_dev"></a>55.8.1.17.7. Property `ReduxOptions > train > losses > losses items > anyOf > msssiml1loss > cuda_dev`

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | No        |
| **Default**  | `0`       |

##### <a name="train_losses_items_anyOf_i17"></a>55.8.1.18. Property `ReduxOptions > train > losses > losses items > anyOf > nccloss`

**Title:** nccloss

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `object`         |
| **Required**              | No               |
| **Additional properties** | Any type allowed |
| **Defined in**            | #/$defs/nccloss  |

| Property                                                    | Pattern | Type             | Deprecated | Definition | Title/Description |
| ----------------------------------------------------------- | ------- | ---------------- | ---------- | ---------- | ----------------- |
| + [type](#train_losses_items_anyOf_i17_type )               | No      | enum (of string) | No         | -          | -                 |
| + [loss_weight](#train_losses_items_anyOf_i17_loss_weight ) | No      | number           | No         | -          | -                 |

###### <a name="train_losses_items_anyOf_i17_type"></a>55.8.1.18.1. Property `ReduxOptions > train > losses > losses items > anyOf > nccloss > type`

|              |                    |
| ------------ | ------------------ |
| **Type**     | `enum (of string)` |
| **Required** | Yes                |

Must be one of:
* "nccloss"

###### <a name="train_losses_items_anyOf_i17_loss_weight"></a>55.8.1.18.2. Property `ReduxOptions > train > losses > losses items > anyOf > nccloss > loss_weight`

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | Yes      |

##### <a name="train_losses_items_anyOf_i18"></a>55.8.1.19. Property `ReduxOptions > train > losses > losses items > anyOf > perceptualfp16loss`

**Title:** perceptualfp16loss

|                           |                            |
| ------------------------- | -------------------------- |
| **Type**                  | `object`                   |
| **Required**              | No                         |
| **Additional properties** | Any type allowed           |
| **Defined in**            | #/$defs/perceptualfp16loss |

| Property                                                            | Pattern | Type             | Deprecated | Definition | Title/Description |
| ------------------------------------------------------------------- | ------- | ---------------- | ---------- | ---------- | ----------------- |
| + [type](#train_losses_items_anyOf_i18_type )                       | No      | enum (of string) | No         | -          | -                 |
| + [loss_weight](#train_losses_items_anyOf_i18_loss_weight )         | No      | number           | No         | -          | -                 |
| - [layer_weights](#train_losses_items_anyOf_i18_layer_weights )     | No      | Combination      | No         | -          | -                 |
| - [w_lambda](#train_losses_items_anyOf_i18_w_lambda )               | No      | number           | No         | -          | -                 |
| - [alpha](#train_losses_items_anyOf_i18_alpha )                     | No      | Combination      | No         | -          | -                 |
| - [criterion](#train_losses_items_anyOf_i18_criterion )             | No      | enum (of string) | No         | -          | -                 |
| - [num_proj_fd](#train_losses_items_anyOf_i18_num_proj_fd )         | No      | integer          | No         | -          | -                 |
| - [phase_weight_fd](#train_losses_items_anyOf_i18_phase_weight_fd ) | No      | number           | No         | -          | -                 |
| - [stride_fd](#train_losses_items_anyOf_i18_stride_fd )             | No      | integer          | No         | -          | -                 |

###### <a name="train_losses_items_anyOf_i18_type"></a>55.8.1.19.1. Property `ReduxOptions > train > losses > losses items > anyOf > perceptualfp16loss > type`

|              |                    |
| ------------ | ------------------ |
| **Type**     | `enum (of string)` |
| **Required** | Yes                |

Must be one of:
* "perceptualfp16loss"

###### <a name="train_losses_items_anyOf_i18_loss_weight"></a>55.8.1.19.2. Property `ReduxOptions > train > losses > losses items > anyOf > perceptualfp16loss > loss_weight`

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | Yes      |

###### <a name="train_losses_items_anyOf_i18_layer_weights"></a>55.8.1.19.3. Property `ReduxOptions > train > losses > losses items > anyOf > perceptualfp16loss > layer_weights`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `combining`      |
| **Required**              | No               |
| **Additional properties** | Any type allowed |
| **Default**               | `null`           |

| Any of(Option)                                                 |
| -------------------------------------------------------------- |
| [item 0](#train_losses_items_anyOf_i18_layer_weights_anyOf_i0) |
| [item 1](#train_losses_items_anyOf_i18_layer_weights_anyOf_i1) |

###### <a name="train_losses_items_anyOf_i18_layer_weights_anyOf_i0"></a>55.8.1.19.3.1. Property `ReduxOptions > train > losses > losses items > anyOf > perceptualfp16loss > layer_weights > anyOf > item 0`

|                           |                                                                                                                                  |
| ------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| **Type**                  | `object`                                                                                                                         |
| **Required**              | No                                                                                                                               |
| **Additional properties** | [Each additional property must conform to the schema](#train_losses_items_anyOf_i18_layer_weights_anyOf_i0_additionalProperties) |

| Property                                                                         | Pattern | Type   | Deprecated | Definition | Title/Description |
| -------------------------------------------------------------------------------- | ------- | ------ | ---------- | ---------- | ----------------- |
| - [](#train_losses_items_anyOf_i18_layer_weights_anyOf_i0_additionalProperties ) | No      | number | No         | -          | -                 |

###### <a name="train_losses_items_anyOf_i18_layer_weights_anyOf_i0_additionalProperties"></a>55.8.1.19.3.1.1. Property `ReduxOptions > train > losses > losses items > anyOf > perceptualfp16loss > layer_weights > anyOf > item 0 > additionalProperties`

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |

###### <a name="train_losses_items_anyOf_i18_layer_weights_anyOf_i1"></a>55.8.1.19.3.2. Property `ReduxOptions > train > losses > losses items > anyOf > perceptualfp16loss > layer_weights > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

###### <a name="train_losses_items_anyOf_i18_w_lambda"></a>55.8.1.19.4. Property `ReduxOptions > train > losses > losses items > anyOf > perceptualfp16loss > w_lambda`

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |
| **Default**  | `0.01`   |

###### <a name="train_losses_items_anyOf_i18_alpha"></a>55.8.1.19.5. Property `ReduxOptions > train > losses > losses items > anyOf > perceptualfp16loss > alpha`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `combining`      |
| **Required**              | No               |
| **Additional properties** | Any type allowed |
| **Default**               | `null`           |

| Any of(Option)                                         |
| ------------------------------------------------------ |
| [item 0](#train_losses_items_anyOf_i18_alpha_anyOf_i0) |
| [item 1](#train_losses_items_anyOf_i18_alpha_anyOf_i1) |

###### <a name="train_losses_items_anyOf_i18_alpha_anyOf_i0"></a>55.8.1.19.5.1. Property `ReduxOptions > train > losses > losses items > anyOf > perceptualfp16loss > alpha > anyOf > item 0`

|              |                   |
| ------------ | ----------------- |
| **Type**     | `array of number` |
| **Required** | No                |

|                      | Array restrictions |
| -------------------- | ------------------ |
| **Min items**        | N/A                |
| **Max items**        | N/A                |
| **Items unicity**    | False              |
| **Additional items** | False              |
| **Tuple validation** | See below          |

| Each item of this array must be                                    | Description |
| ------------------------------------------------------------------ | ----------- |
| [item 0 items](#train_losses_items_anyOf_i18_alpha_anyOf_i0_items) | -           |

###### <a name="train_losses_items_anyOf_i18_alpha_anyOf_i0_items"></a>55.8.1.19.5.1.1. ReduxOptions > train > losses > losses items > anyOf > perceptualfp16loss > alpha > anyOf > item 0 > item 0 items

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |

###### <a name="train_losses_items_anyOf_i18_alpha_anyOf_i1"></a>55.8.1.19.5.2. Property `ReduxOptions > train > losses > losses items > anyOf > perceptualfp16loss > alpha > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

###### <a name="train_losses_items_anyOf_i18_criterion"></a>55.8.1.19.6. Property `ReduxOptions > train > losses > losses items > anyOf > perceptualfp16loss > criterion`

|              |                    |
| ------------ | ------------------ |
| **Type**     | `enum (of string)` |
| **Required** | No                 |
| **Default**  | `"pd+l1"`          |

Must be one of:
* "charbonnier"
* "fd"
* "fd+l1pd"
* "l1"
* "pd+l1"

###### <a name="train_losses_items_anyOf_i18_num_proj_fd"></a>55.8.1.19.7. Property `ReduxOptions > train > losses > losses items > anyOf > perceptualfp16loss > num_proj_fd`

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | No        |
| **Default**  | `256`     |

###### <a name="train_losses_items_anyOf_i18_phase_weight_fd"></a>55.8.1.19.8. Property `ReduxOptions > train > losses > losses items > anyOf > perceptualfp16loss > phase_weight_fd`

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |
| **Default**  | `1.0`    |

###### <a name="train_losses_items_anyOf_i18_stride_fd"></a>55.8.1.19.9. Property `ReduxOptions > train > losses > losses items > anyOf > perceptualfp16loss > stride_fd`

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | No        |
| **Default**  | `1`       |

##### <a name="train_losses_items_anyOf_i19"></a>55.8.1.20. Property `ReduxOptions > train > losses > losses items > anyOf > perceptualloss`

**Title:** perceptualloss

|                           |                        |
| ------------------------- | ---------------------- |
| **Type**                  | `object`               |
| **Required**              | No                     |
| **Additional properties** | Any type allowed       |
| **Defined in**            | #/$defs/perceptualloss |

| Property                                                            | Pattern | Type             | Deprecated | Definition | Title/Description |
| ------------------------------------------------------------------- | ------- | ---------------- | ---------- | ---------- | ----------------- |
| + [type](#train_losses_items_anyOf_i19_type )                       | No      | enum (of string) | No         | -          | -                 |
| + [loss_weight](#train_losses_items_anyOf_i19_loss_weight )         | No      | number           | No         | -          | -                 |
| - [layer_weights](#train_losses_items_anyOf_i19_layer_weights )     | No      | Combination      | No         | -          | -                 |
| - [w_lambda](#train_losses_items_anyOf_i19_w_lambda )               | No      | number           | No         | -          | -                 |
| - [alpha](#train_losses_items_anyOf_i19_alpha )                     | No      | Combination      | No         | -          | -                 |
| - [criterion](#train_losses_items_anyOf_i19_criterion )             | No      | enum (of string) | No         | -          | -                 |
| - [num_proj_fd](#train_losses_items_anyOf_i19_num_proj_fd )         | No      | integer          | No         | -          | -                 |
| - [phase_weight_fd](#train_losses_items_anyOf_i19_phase_weight_fd ) | No      | number           | No         | -          | -                 |
| - [stride_fd](#train_losses_items_anyOf_i19_stride_fd )             | No      | integer          | No         | -          | -                 |

###### <a name="train_losses_items_anyOf_i19_type"></a>55.8.1.20.1. Property `ReduxOptions > train > losses > losses items > anyOf > perceptualloss > type`

|              |                    |
| ------------ | ------------------ |
| **Type**     | `enum (of string)` |
| **Required** | Yes                |

Must be one of:
* "perceptualloss"

###### <a name="train_losses_items_anyOf_i19_loss_weight"></a>55.8.1.20.2. Property `ReduxOptions > train > losses > losses items > anyOf > perceptualloss > loss_weight`

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | Yes      |

###### <a name="train_losses_items_anyOf_i19_layer_weights"></a>55.8.1.20.3. Property `ReduxOptions > train > losses > losses items > anyOf > perceptualloss > layer_weights`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `combining`      |
| **Required**              | No               |
| **Additional properties** | Any type allowed |
| **Default**               | `null`           |

| Any of(Option)                                                 |
| -------------------------------------------------------------- |
| [item 0](#train_losses_items_anyOf_i19_layer_weights_anyOf_i0) |
| [item 1](#train_losses_items_anyOf_i19_layer_weights_anyOf_i1) |

###### <a name="train_losses_items_anyOf_i19_layer_weights_anyOf_i0"></a>55.8.1.20.3.1. Property `ReduxOptions > train > losses > losses items > anyOf > perceptualloss > layer_weights > anyOf > item 0`

|                           |                                                                                                                                  |
| ------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| **Type**                  | `object`                                                                                                                         |
| **Required**              | No                                                                                                                               |
| **Additional properties** | [Each additional property must conform to the schema](#train_losses_items_anyOf_i19_layer_weights_anyOf_i0_additionalProperties) |

| Property                                                                         | Pattern | Type   | Deprecated | Definition | Title/Description |
| -------------------------------------------------------------------------------- | ------- | ------ | ---------- | ---------- | ----------------- |
| - [](#train_losses_items_anyOf_i19_layer_weights_anyOf_i0_additionalProperties ) | No      | number | No         | -          | -                 |

###### <a name="train_losses_items_anyOf_i19_layer_weights_anyOf_i0_additionalProperties"></a>55.8.1.20.3.1.1. Property `ReduxOptions > train > losses > losses items > anyOf > perceptualloss > layer_weights > anyOf > item 0 > additionalProperties`

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |

###### <a name="train_losses_items_anyOf_i19_layer_weights_anyOf_i1"></a>55.8.1.20.3.2. Property `ReduxOptions > train > losses > losses items > anyOf > perceptualloss > layer_weights > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

###### <a name="train_losses_items_anyOf_i19_w_lambda"></a>55.8.1.20.4. Property `ReduxOptions > train > losses > losses items > anyOf > perceptualloss > w_lambda`

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |
| **Default**  | `0.01`   |

###### <a name="train_losses_items_anyOf_i19_alpha"></a>55.8.1.20.5. Property `ReduxOptions > train > losses > losses items > anyOf > perceptualloss > alpha`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `combining`      |
| **Required**              | No               |
| **Additional properties** | Any type allowed |
| **Default**               | `null`           |

| Any of(Option)                                         |
| ------------------------------------------------------ |
| [item 0](#train_losses_items_anyOf_i19_alpha_anyOf_i0) |
| [item 1](#train_losses_items_anyOf_i19_alpha_anyOf_i1) |

###### <a name="train_losses_items_anyOf_i19_alpha_anyOf_i0"></a>55.8.1.20.5.1. Property `ReduxOptions > train > losses > losses items > anyOf > perceptualloss > alpha > anyOf > item 0`

|              |                   |
| ------------ | ----------------- |
| **Type**     | `array of number` |
| **Required** | No                |

|                      | Array restrictions |
| -------------------- | ------------------ |
| **Min items**        | N/A                |
| **Max items**        | N/A                |
| **Items unicity**    | False              |
| **Additional items** | False              |
| **Tuple validation** | See below          |

| Each item of this array must be                                    | Description |
| ------------------------------------------------------------------ | ----------- |
| [item 0 items](#train_losses_items_anyOf_i19_alpha_anyOf_i0_items) | -           |

###### <a name="train_losses_items_anyOf_i19_alpha_anyOf_i0_items"></a>55.8.1.20.5.1.1. ReduxOptions > train > losses > losses items > anyOf > perceptualloss > alpha > anyOf > item 0 > item 0 items

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |

###### <a name="train_losses_items_anyOf_i19_alpha_anyOf_i1"></a>55.8.1.20.5.2. Property `ReduxOptions > train > losses > losses items > anyOf > perceptualloss > alpha > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

###### <a name="train_losses_items_anyOf_i19_criterion"></a>55.8.1.20.6. Property `ReduxOptions > train > losses > losses items > anyOf > perceptualloss > criterion`

|              |                    |
| ------------ | ------------------ |
| **Type**     | `enum (of string)` |
| **Required** | No                 |
| **Default**  | `"pd+l1"`          |

Must be one of:
* "charbonnier"
* "fd"
* "fd+l1pd"
* "l1"
* "pd+l1"

###### <a name="train_losses_items_anyOf_i19_num_proj_fd"></a>55.8.1.20.7. Property `ReduxOptions > train > losses > losses items > anyOf > perceptualloss > num_proj_fd`

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | No        |
| **Default**  | `256`     |

###### <a name="train_losses_items_anyOf_i19_phase_weight_fd"></a>55.8.1.20.8. Property `ReduxOptions > train > losses > losses items > anyOf > perceptualloss > phase_weight_fd`

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |
| **Default**  | `1.0`    |

###### <a name="train_losses_items_anyOf_i19_stride_fd"></a>55.8.1.20.9. Property `ReduxOptions > train > losses > losses items > anyOf > perceptualloss > stride_fd`

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | No        |
| **Default**  | `1`       |

##### <a name="train_losses_items_anyOf_i20"></a>55.8.1.21. Property `ReduxOptions > train > losses > losses items > anyOf > topiqloss`

**Title:** topiqloss

|                           |                   |
| ------------------------- | ----------------- |
| **Type**                  | `object`          |
| **Required**              | No                |
| **Additional properties** | Any type allowed  |
| **Defined in**            | #/$defs/topiqloss |

| Property                                                      | Pattern | Type             | Deprecated | Definition | Title/Description |
| ------------------------------------------------------------- | ------- | ---------------- | ---------- | ---------- | ----------------- |
| + [type](#train_losses_items_anyOf_i20_type )                 | No      | enum (of string) | No         | -          | -                 |
| + [loss_weight](#train_losses_items_anyOf_i20_loss_weight )   | No      | number           | No         | -          | -                 |
| - [resize_input](#train_losses_items_anyOf_i20_resize_input ) | No      | boolean          | No         | -          | -                 |

###### <a name="train_losses_items_anyOf_i20_type"></a>55.8.1.21.1. Property `ReduxOptions > train > losses > losses items > anyOf > topiqloss > type`

|              |                    |
| ------------ | ------------------ |
| **Type**     | `enum (of string)` |
| **Required** | Yes                |

Must be one of:
* "topiqloss"

###### <a name="train_losses_items_anyOf_i20_loss_weight"></a>55.8.1.21.2. Property `ReduxOptions > train > losses > losses items > anyOf > topiqloss > loss_weight`

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | Yes      |

###### <a name="train_losses_items_anyOf_i20_resize_input"></a>55.8.1.21.3. Property `ReduxOptions > train > losses > losses items > anyOf > topiqloss > resize_input`

|              |           |
| ------------ | --------- |
| **Type**     | `boolean` |
| **Required** | No        |
| **Default**  | `true`    |

### <a name="train_pixel_opt"></a>55.9. Property `ReduxOptions > train > pixel_opt`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `combining`      |
| **Required**              | No               |
| **Additional properties** | Any type allowed |
| **Default**               | `null`           |

| Any of(Option)                      |
| ----------------------------------- |
| [item 0](#train_pixel_opt_anyOf_i0) |
| [item 1](#train_pixel_opt_anyOf_i1) |

#### <a name="train_pixel_opt_anyOf_i0"></a>55.9.1. Property `ReduxOptions > train > pixel_opt > anyOf > item 0`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `object`         |
| **Required**              | No               |
| **Additional properties** | Any type allowed |

#### <a name="train_pixel_opt_anyOf_i1"></a>55.9.2. Property `ReduxOptions > train > pixel_opt > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

### <a name="train_mssim_opt"></a>55.10. Property `ReduxOptions > train > mssim_opt`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `combining`      |
| **Required**              | No               |
| **Additional properties** | Any type allowed |
| **Default**               | `null`           |

| Any of(Option)                      |
| ----------------------------------- |
| [item 0](#train_mssim_opt_anyOf_i0) |
| [item 1](#train_mssim_opt_anyOf_i1) |

#### <a name="train_mssim_opt_anyOf_i0"></a>55.10.1. Property `ReduxOptions > train > mssim_opt > anyOf > item 0`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `object`         |
| **Required**              | No               |
| **Additional properties** | Any type allowed |

#### <a name="train_mssim_opt_anyOf_i1"></a>55.10.2. Property `ReduxOptions > train > mssim_opt > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

### <a name="train_ms_ssim_l1_opt"></a>55.11. Property `ReduxOptions > train > ms_ssim_l1_opt`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `combining`      |
| **Required**              | No               |
| **Additional properties** | Any type allowed |
| **Default**               | `null`           |

| Any of(Option)                           |
| ---------------------------------------- |
| [item 0](#train_ms_ssim_l1_opt_anyOf_i0) |
| [item 1](#train_ms_ssim_l1_opt_anyOf_i1) |

#### <a name="train_ms_ssim_l1_opt_anyOf_i0"></a>55.11.1. Property `ReduxOptions > train > ms_ssim_l1_opt > anyOf > item 0`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `object`         |
| **Required**              | No               |
| **Additional properties** | Any type allowed |

#### <a name="train_ms_ssim_l1_opt_anyOf_i1"></a>55.11.2. Property `ReduxOptions > train > ms_ssim_l1_opt > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

### <a name="train_perceptual_opt"></a>55.12. Property `ReduxOptions > train > perceptual_opt`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `combining`      |
| **Required**              | No               |
| **Additional properties** | Any type allowed |
| **Default**               | `null`           |

| Any of(Option)                           |
| ---------------------------------------- |
| [item 0](#train_perceptual_opt_anyOf_i0) |
| [item 1](#train_perceptual_opt_anyOf_i1) |

#### <a name="train_perceptual_opt_anyOf_i0"></a>55.12.1. Property `ReduxOptions > train > perceptual_opt > anyOf > item 0`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `object`         |
| **Required**              | No               |
| **Additional properties** | Any type allowed |

#### <a name="train_perceptual_opt_anyOf_i1"></a>55.12.2. Property `ReduxOptions > train > perceptual_opt > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

### <a name="train_contextual_opt"></a>55.13. Property `ReduxOptions > train > contextual_opt`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `combining`      |
| **Required**              | No               |
| **Additional properties** | Any type allowed |
| **Default**               | `null`           |

| Any of(Option)                           |
| ---------------------------------------- |
| [item 0](#train_contextual_opt_anyOf_i0) |
| [item 1](#train_contextual_opt_anyOf_i1) |

#### <a name="train_contextual_opt_anyOf_i0"></a>55.13.1. Property `ReduxOptions > train > contextual_opt > anyOf > item 0`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `object`         |
| **Required**              | No               |
| **Additional properties** | Any type allowed |

#### <a name="train_contextual_opt_anyOf_i1"></a>55.13.2. Property `ReduxOptions > train > contextual_opt > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

### <a name="train_dists_opt"></a>55.14. Property `ReduxOptions > train > dists_opt`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `combining`      |
| **Required**              | No               |
| **Additional properties** | Any type allowed |
| **Default**               | `null`           |

| Any of(Option)                      |
| ----------------------------------- |
| [item 0](#train_dists_opt_anyOf_i0) |
| [item 1](#train_dists_opt_anyOf_i1) |

#### <a name="train_dists_opt_anyOf_i0"></a>55.14.1. Property `ReduxOptions > train > dists_opt > anyOf > item 0`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `object`         |
| **Required**              | No               |
| **Additional properties** | Any type allowed |

#### <a name="train_dists_opt_anyOf_i1"></a>55.14.2. Property `ReduxOptions > train > dists_opt > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

### <a name="train_hr_inversion_opt"></a>55.15. Property `ReduxOptions > train > hr_inversion_opt`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `combining`      |
| **Required**              | No               |
| **Additional properties** | Any type allowed |
| **Default**               | `null`           |

| Any of(Option)                             |
| ------------------------------------------ |
| [item 0](#train_hr_inversion_opt_anyOf_i0) |
| [item 1](#train_hr_inversion_opt_anyOf_i1) |

#### <a name="train_hr_inversion_opt_anyOf_i0"></a>55.15.1. Property `ReduxOptions > train > hr_inversion_opt > anyOf > item 0`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `object`         |
| **Required**              | No               |
| **Additional properties** | Any type allowed |

#### <a name="train_hr_inversion_opt_anyOf_i1"></a>55.15.2. Property `ReduxOptions > train > hr_inversion_opt > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

### <a name="train_dinov2_opt"></a>55.16. Property `ReduxOptions > train > dinov2_opt`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `combining`      |
| **Required**              | No               |
| **Additional properties** | Any type allowed |
| **Default**               | `null`           |

| Any of(Option)                       |
| ------------------------------------ |
| [item 0](#train_dinov2_opt_anyOf_i0) |
| [item 1](#train_dinov2_opt_anyOf_i1) |

#### <a name="train_dinov2_opt_anyOf_i0"></a>55.16.1. Property `ReduxOptions > train > dinov2_opt > anyOf > item 0`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `object`         |
| **Required**              | No               |
| **Additional properties** | Any type allowed |

#### <a name="train_dinov2_opt_anyOf_i1"></a>55.16.2. Property `ReduxOptions > train > dinov2_opt > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

### <a name="train_topiq_opt"></a>55.17. Property `ReduxOptions > train > topiq_opt`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `combining`      |
| **Required**              | No               |
| **Additional properties** | Any type allowed |
| **Default**               | `null`           |

| Any of(Option)                      |
| ----------------------------------- |
| [item 0](#train_topiq_opt_anyOf_i0) |
| [item 1](#train_topiq_opt_anyOf_i1) |

#### <a name="train_topiq_opt_anyOf_i0"></a>55.17.1. Property `ReduxOptions > train > topiq_opt > anyOf > item 0`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `object`         |
| **Required**              | No               |
| **Additional properties** | Any type allowed |

#### <a name="train_topiq_opt_anyOf_i1"></a>55.17.2. Property `ReduxOptions > train > topiq_opt > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

### <a name="train_pd_opt"></a>55.18. Property `ReduxOptions > train > pd_opt`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `combining`      |
| **Required**              | No               |
| **Additional properties** | Any type allowed |
| **Default**               | `null`           |

| Any of(Option)                   |
| -------------------------------- |
| [item 0](#train_pd_opt_anyOf_i0) |
| [item 1](#train_pd_opt_anyOf_i1) |

#### <a name="train_pd_opt_anyOf_i0"></a>55.18.1. Property `ReduxOptions > train > pd_opt > anyOf > item 0`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `object`         |
| **Required**              | No               |
| **Additional properties** | Any type allowed |

#### <a name="train_pd_opt_anyOf_i1"></a>55.18.2. Property `ReduxOptions > train > pd_opt > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

### <a name="train_fd_opt"></a>55.19. Property `ReduxOptions > train > fd_opt`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `combining`      |
| **Required**              | No               |
| **Additional properties** | Any type allowed |
| **Default**               | `null`           |

| Any of(Option)                   |
| -------------------------------- |
| [item 0](#train_fd_opt_anyOf_i0) |
| [item 1](#train_fd_opt_anyOf_i1) |

#### <a name="train_fd_opt_anyOf_i0"></a>55.19.1. Property `ReduxOptions > train > fd_opt > anyOf > item 0`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `object`         |
| **Required**              | No               |
| **Additional properties** | Any type allowed |

#### <a name="train_fd_opt_anyOf_i1"></a>55.19.2. Property `ReduxOptions > train > fd_opt > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

### <a name="train_ldl_opt"></a>55.20. Property `ReduxOptions > train > ldl_opt`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `combining`      |
| **Required**              | No               |
| **Additional properties** | Any type allowed |
| **Default**               | `null`           |

| Any of(Option)                    |
| --------------------------------- |
| [item 0](#train_ldl_opt_anyOf_i0) |
| [item 1](#train_ldl_opt_anyOf_i1) |

#### <a name="train_ldl_opt_anyOf_i0"></a>55.20.1. Property `ReduxOptions > train > ldl_opt > anyOf > item 0`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `object`         |
| **Required**              | No               |
| **Additional properties** | Any type allowed |

#### <a name="train_ldl_opt_anyOf_i1"></a>55.20.2. Property `ReduxOptions > train > ldl_opt > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

### <a name="train_hsluv_opt"></a>55.21. Property `ReduxOptions > train > hsluv_opt`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `combining`      |
| **Required**              | No               |
| **Additional properties** | Any type allowed |
| **Default**               | `null`           |

| Any of(Option)                      |
| ----------------------------------- |
| [item 0](#train_hsluv_opt_anyOf_i0) |
| [item 1](#train_hsluv_opt_anyOf_i1) |

#### <a name="train_hsluv_opt_anyOf_i0"></a>55.21.1. Property `ReduxOptions > train > hsluv_opt > anyOf > item 0`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `object`         |
| **Required**              | No               |
| **Additional properties** | Any type allowed |

#### <a name="train_hsluv_opt_anyOf_i1"></a>55.21.2. Property `ReduxOptions > train > hsluv_opt > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

### <a name="train_gan_opt"></a>55.22. Property `ReduxOptions > train > gan_opt`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `combining`      |
| **Required**              | No               |
| **Additional properties** | Any type allowed |
| **Default**               | `null`           |

| Any of(Option)                    |
| --------------------------------- |
| [item 0](#train_gan_opt_anyOf_i0) |
| [item 1](#train_gan_opt_anyOf_i1) |

#### <a name="train_gan_opt_anyOf_i0"></a>55.22.1. Property `ReduxOptions > train > gan_opt > anyOf > item 0`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `object`         |
| **Required**              | No               |
| **Additional properties** | Any type allowed |

#### <a name="train_gan_opt_anyOf_i1"></a>55.22.2. Property `ReduxOptions > train > gan_opt > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

### <a name="train_color_opt"></a>55.23. Property `ReduxOptions > train > color_opt`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `combining`      |
| **Required**              | No               |
| **Additional properties** | Any type allowed |
| **Default**               | `null`           |

| Any of(Option)                      |
| ----------------------------------- |
| [item 0](#train_color_opt_anyOf_i0) |
| [item 1](#train_color_opt_anyOf_i1) |

#### <a name="train_color_opt_anyOf_i0"></a>55.23.1. Property `ReduxOptions > train > color_opt > anyOf > item 0`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `object`         |
| **Required**              | No               |
| **Additional properties** | Any type allowed |

#### <a name="train_color_opt_anyOf_i1"></a>55.23.2. Property `ReduxOptions > train > color_opt > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

### <a name="train_luma_opt"></a>55.24. Property `ReduxOptions > train > luma_opt`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `combining`      |
| **Required**              | No               |
| **Additional properties** | Any type allowed |
| **Default**               | `null`           |

| Any of(Option)                     |
| ---------------------------------- |
| [item 0](#train_luma_opt_anyOf_i0) |
| [item 1](#train_luma_opt_anyOf_i1) |

#### <a name="train_luma_opt_anyOf_i0"></a>55.24.1. Property `ReduxOptions > train > luma_opt > anyOf > item 0`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `object`         |
| **Required**              | No               |
| **Additional properties** | Any type allowed |

#### <a name="train_luma_opt_anyOf_i1"></a>55.24.2. Property `ReduxOptions > train > luma_opt > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

### <a name="train_avg_opt"></a>55.25. Property `ReduxOptions > train > avg_opt`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `combining`      |
| **Required**              | No               |
| **Additional properties** | Any type allowed |
| **Default**               | `null`           |

| Any of(Option)                    |
| --------------------------------- |
| [item 0](#train_avg_opt_anyOf_i0) |
| [item 1](#train_avg_opt_anyOf_i1) |

#### <a name="train_avg_opt_anyOf_i0"></a>55.25.1. Property `ReduxOptions > train > avg_opt > anyOf > item 0`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `object`         |
| **Required**              | No               |
| **Additional properties** | Any type allowed |

#### <a name="train_avg_opt_anyOf_i1"></a>55.25.2. Property `ReduxOptions > train > avg_opt > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

### <a name="train_bicubic_opt"></a>55.26. Property `ReduxOptions > train > bicubic_opt`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `combining`      |
| **Required**              | No               |
| **Additional properties** | Any type allowed |
| **Default**               | `null`           |

| Any of(Option)                        |
| ------------------------------------- |
| [item 0](#train_bicubic_opt_anyOf_i0) |
| [item 1](#train_bicubic_opt_anyOf_i1) |

#### <a name="train_bicubic_opt_anyOf_i0"></a>55.26.1. Property `ReduxOptions > train > bicubic_opt > anyOf > item 0`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `object`         |
| **Required**              | No               |
| **Additional properties** | Any type allowed |

#### <a name="train_bicubic_opt_anyOf_i1"></a>55.26.2. Property `ReduxOptions > train > bicubic_opt > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

### <a name="train_use_moa"></a>55.27. Property `ReduxOptions > train > use_moa`

|              |           |
| ------------ | --------- |
| **Type**     | `boolean` |
| **Required** | No        |
| **Default**  | `false`   |

**Description:** Whether to enable mixture of augmentations, which augments the dataset on the fly to create more variety and help the model generalize.

### <a name="train_moa_augs"></a>55.28. Property `ReduxOptions > train > moa_augs`

|              |                   |
| ------------ | ----------------- |
| **Type**     | `array of string` |
| **Required** | No                |

**Description:** The list of augmentations to choose from, only one is selected per iteration.

|                      | Array restrictions |
| -------------------- | ------------------ |
| **Min items**        | N/A                |
| **Max items**        | N/A                |
| **Items unicity**    | False              |
| **Additional items** | False              |
| **Tuple validation** | See below          |

| Each item of this array must be         | Description |
| --------------------------------------- | ----------- |
| [moa_augs items](#train_moa_augs_items) | -           |

#### <a name="train_moa_augs_items"></a>55.28.1. ReduxOptions > train > moa_augs > moa_augs items

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | No       |

### <a name="train_moa_probs"></a>55.29. Property `ReduxOptions > train > moa_probs`

|              |                   |
| ------------ | ----------------- |
| **Type**     | `array of number` |
| **Required** | No                |

**Description:** The probability each augmentation in moa_augs will be applied. Total should add up to 1.

|                      | Array restrictions |
| -------------------- | ------------------ |
| **Min items**        | N/A                |
| **Max items**        | N/A                |
| **Items unicity**    | False              |
| **Additional items** | False              |
| **Tuple validation** | See below          |

| Each item of this array must be           | Description |
| ----------------------------------------- | ----------- |
| [moa_probs items](#train_moa_probs_items) | -           |

#### <a name="train_moa_probs_items"></a>55.29.1. ReduxOptions > train > moa_probs > moa_probs items

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |

### <a name="train_moa_debug"></a>55.30. Property `ReduxOptions > train > moa_debug`

|              |           |
| ------------ | --------- |
| **Type**     | `boolean` |
| **Required** | No        |
| **Default**  | `false`   |

**Description:** Save images before and after augment to debug/moa folder inside of the root training directory.

### <a name="train_moa_debug_limit"></a>55.31. Property `ReduxOptions > train > moa_debug_limit`

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | No        |
| **Default**  | `100`     |

**Description:** The max number of iterations to save augmentation images for.

## <a name="val"></a>56. Property `ReduxOptions > val`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `combining`      |
| **Required**              | No               |
| **Additional properties** | Any type allowed |
| **Default**               | `null`           |

| Any of(Option)              |
| --------------------------- |
| [item 0](#val_anyOf_i0)     |
| [ValOptions](#val_anyOf_i1) |

### <a name="val_anyOf_i0"></a>56.1. Property `ReduxOptions > val > anyOf > item 0`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

### <a name="val_anyOf_i1"></a>56.2. Property `ReduxOptions > val > anyOf > ValOptions`

**Title:** ValOptions

|                           |                    |
| ------------------------- | ------------------ |
| **Type**                  | `object`           |
| **Required**              | No                 |
| **Additional properties** | Not allowed        |
| **Defined in**            | #/$defs/ValOptions |

| Property                                            | Pattern | Type        | Deprecated | Definition | Title/Description                                                                                        |
| --------------------------------------------------- | ------- | ----------- | ---------- | ---------- | -------------------------------------------------------------------------------------------------------- |
| + [val_enabled](#val_anyOf_i1_val_enabled )         | No      | boolean     | No         | -          | Whether to enable validations. If disabled, all validation settings below are ignored.                   |
| + [save_img](#val_anyOf_i1_save_img )               | No      | boolean     | No         | -          | Whether to save the validation images during validation, in the experiments/<name>/visualization folder. |
| - [val_freq](#val_anyOf_i1_val_freq )               | No      | Combination | No         | -          | How often to run validations, in iterations.                                                             |
| - [suffix](#val_anyOf_i1_suffix )                   | No      | Combination | No         | -          | Optional suffix to append to saved filenames.                                                            |
| - [metrics_enabled](#val_anyOf_i1_metrics_enabled ) | No      | boolean     | No         | -          | Whether to run metrics calculations during validation.                                                   |
| - [metrics](#val_anyOf_i1_metrics )                 | No      | Combination | No         | -          | -                                                                                                        |
| - [pbar](#val_anyOf_i1_pbar )                       | No      | boolean     | No         | -          | -                                                                                                        |

#### <a name="val_anyOf_i1_val_enabled"></a>56.2.1. Property `ReduxOptions > val > anyOf > ValOptions > val_enabled`

|              |           |
| ------------ | --------- |
| **Type**     | `boolean` |
| **Required** | Yes       |

**Description:** Whether to enable validations. If disabled, all validation settings below are ignored.

#### <a name="val_anyOf_i1_save_img"></a>56.2.2. Property `ReduxOptions > val > anyOf > ValOptions > save_img`

|              |           |
| ------------ | --------- |
| **Type**     | `boolean` |
| **Required** | Yes       |

**Description:** Whether to save the validation images during validation, in the experiments/<name>/visualization folder.

#### <a name="val_anyOf_i1_val_freq"></a>56.2.3. Property `ReduxOptions > val > anyOf > ValOptions > val_freq`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `combining`      |
| **Required**              | No               |
| **Additional properties** | Any type allowed |
| **Default**               | `null`           |

**Description:** How often to run validations, in iterations.

| Any of(Option)                            |
| ----------------------------------------- |
| [item 0](#val_anyOf_i1_val_freq_anyOf_i0) |
| [item 1](#val_anyOf_i1_val_freq_anyOf_i1) |

##### <a name="val_anyOf_i1_val_freq_anyOf_i0"></a>56.2.3.1. Property `ReduxOptions > val > anyOf > ValOptions > val_freq > anyOf > item 0`

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | No        |

##### <a name="val_anyOf_i1_val_freq_anyOf_i1"></a>56.2.3.2. Property `ReduxOptions > val > anyOf > ValOptions > val_freq > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

#### <a name="val_anyOf_i1_suffix"></a>56.2.4. Property `ReduxOptions > val > anyOf > ValOptions > suffix`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `combining`      |
| **Required**              | No               |
| **Additional properties** | Any type allowed |
| **Default**               | `null`           |

**Description:** Optional suffix to append to saved filenames.

| Any of(Option)                          |
| --------------------------------------- |
| [item 0](#val_anyOf_i1_suffix_anyOf_i0) |
| [item 1](#val_anyOf_i1_suffix_anyOf_i1) |

##### <a name="val_anyOf_i1_suffix_anyOf_i0"></a>56.2.4.1. Property `ReduxOptions > val > anyOf > ValOptions > suffix > anyOf > item 0`

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | No       |

##### <a name="val_anyOf_i1_suffix_anyOf_i1"></a>56.2.4.2. Property `ReduxOptions > val > anyOf > ValOptions > suffix > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

#### <a name="val_anyOf_i1_metrics_enabled"></a>56.2.5. Property `ReduxOptions > val > anyOf > ValOptions > metrics_enabled`

|              |           |
| ------------ | --------- |
| **Type**     | `boolean` |
| **Required** | No        |
| **Default**  | `false`   |

**Description:** Whether to run metrics calculations during validation.

#### <a name="val_anyOf_i1_metrics"></a>56.2.6. Property `ReduxOptions > val > anyOf > ValOptions > metrics`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `combining`      |
| **Required**              | No               |
| **Additional properties** | Any type allowed |
| **Default**               | `null`           |

| Any of(Option)                           |
| ---------------------------------------- |
| [item 0](#val_anyOf_i1_metrics_anyOf_i0) |
| [item 1](#val_anyOf_i1_metrics_anyOf_i1) |

##### <a name="val_anyOf_i1_metrics_anyOf_i0"></a>56.2.6.1. Property `ReduxOptions > val > anyOf > ValOptions > metrics > anyOf > item 0`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `object`         |
| **Required**              | No               |
| **Additional properties** | Any type allowed |

##### <a name="val_anyOf_i1_metrics_anyOf_i1"></a>56.2.6.2. Property `ReduxOptions > val > anyOf > ValOptions > metrics > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

#### <a name="val_anyOf_i1_pbar"></a>56.2.7. Property `ReduxOptions > val > anyOf > ValOptions > pbar`

|              |           |
| ------------ | --------- |
| **Type**     | `boolean` |
| **Required** | No        |
| **Default**  | `true`    |

## <a name="logger"></a>57. Property `ReduxOptions > logger`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `combining`      |
| **Required**              | No               |
| **Additional properties** | Any type allowed |
| **Default**               | `null`           |

| Any of(Option)                 |
| ------------------------------ |
| [item 0](#logger_anyOf_i0)     |
| [LogOptions](#logger_anyOf_i1) |

### <a name="logger_anyOf_i0"></a>57.1. Property `ReduxOptions > logger > anyOf > item 0`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

### <a name="logger_anyOf_i1"></a>57.2. Property `ReduxOptions > logger > anyOf > LogOptions`

**Title:** LogOptions

|                           |                    |
| ------------------------- | ------------------ |
| **Type**                  | `object`           |
| **Required**              | No                 |
| **Additional properties** | Not allowed        |
| **Defined in**            | #/$defs/LogOptions |

| Property                                                             | Pattern | Type             | Deprecated | Definition | Title/Description                                                       |
| -------------------------------------------------------------------- | ------- | ---------------- | ---------- | ---------- | ----------------------------------------------------------------------- |
| + [print_freq](#logger_anyOf_i1_print_freq )                         | No      | integer          | No         | -          | How often to print logs to the console, in iterations.                  |
| + [save_checkpoint_freq](#logger_anyOf_i1_save_checkpoint_freq )     | No      | integer          | No         | -          | How often to save model checkpoints and training states, in iterations. |
| + [use_tb_logger](#logger_anyOf_i1_use_tb_logger )                   | No      | boolean          | No         | -          | Whether or not to enable TensorBoard logging.                           |
| - [save_checkpoint_format](#logger_anyOf_i1_save_checkpoint_format ) | No      | enum (of string) | No         | -          | Format to save model checkpoints.                                       |
| - [wandb](#logger_anyOf_i1_wandb )                                   | No      | Combination      | No         | -          | -                                                                       |

#### <a name="logger_anyOf_i1_print_freq"></a>57.2.1. Property `ReduxOptions > logger > anyOf > LogOptions > print_freq`

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | Yes       |

**Description:** How often to print logs to the console, in iterations.

#### <a name="logger_anyOf_i1_save_checkpoint_freq"></a>57.2.2. Property `ReduxOptions > logger > anyOf > LogOptions > save_checkpoint_freq`

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | Yes       |

**Description:** How often to save model checkpoints and training states, in iterations.

#### <a name="logger_anyOf_i1_use_tb_logger"></a>57.2.3. Property `ReduxOptions > logger > anyOf > LogOptions > use_tb_logger`

|              |           |
| ------------ | --------- |
| **Type**     | `boolean` |
| **Required** | Yes       |

**Description:** Whether or not to enable TensorBoard logging.

#### <a name="logger_anyOf_i1_save_checkpoint_format"></a>57.2.4. Property `ReduxOptions > logger > anyOf > LogOptions > save_checkpoint_format`

|              |                    |
| ------------ | ------------------ |
| **Type**     | `enum (of string)` |
| **Required** | No                 |
| **Default**  | `"safetensors"`    |

**Description:** Format to save model checkpoints.

Must be one of:
* "pth"
* "safetensors"

#### <a name="logger_anyOf_i1_wandb"></a>57.2.5. Property `ReduxOptions > logger > anyOf > LogOptions > wandb`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `combining`      |
| **Required**              | No               |
| **Additional properties** | Any type allowed |
| **Default**               | `null`           |

| Any of(Option)                                  |
| ----------------------------------------------- |
| [item 0](#logger_anyOf_i1_wandb_anyOf_i0)       |
| [WandbOptions](#logger_anyOf_i1_wandb_anyOf_i1) |

##### <a name="logger_anyOf_i1_wandb_anyOf_i0"></a>57.2.5.1. Property `ReduxOptions > logger > anyOf > LogOptions > wandb > anyOf > item 0`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

##### <a name="logger_anyOf_i1_wandb_anyOf_i1"></a>57.2.5.2. Property `ReduxOptions > logger > anyOf > LogOptions > wandb > anyOf > WandbOptions`

**Title:** WandbOptions

|                           |                      |
| ------------------------- | -------------------- |
| **Type**                  | `object`             |
| **Required**              | No                   |
| **Additional properties** | Not allowed          |
| **Defined in**            | #/$defs/WandbOptions |

| Property                                                  | Pattern | Type        | Deprecated | Definition | Title/Description |
| --------------------------------------------------------- | ------- | ----------- | ---------- | ---------- | ----------------- |
| - [resume_id](#logger_anyOf_i1_wandb_anyOf_i1_resume_id ) | No      | Combination | No         | -          | -                 |
| - [project](#logger_anyOf_i1_wandb_anyOf_i1_project )     | No      | Combination | No         | -          | -                 |

###### <a name="logger_anyOf_i1_wandb_anyOf_i1_resume_id"></a>57.2.5.2.1. Property `ReduxOptions > logger > anyOf > LogOptions > wandb > anyOf > WandbOptions > resume_id`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `combining`      |
| **Required**              | No               |
| **Additional properties** | Any type allowed |
| **Default**               | `null`           |

| Any of(Option)                                               |
| ------------------------------------------------------------ |
| [item 0](#logger_anyOf_i1_wandb_anyOf_i1_resume_id_anyOf_i0) |
| [item 1](#logger_anyOf_i1_wandb_anyOf_i1_resume_id_anyOf_i1) |

###### <a name="logger_anyOf_i1_wandb_anyOf_i1_resume_id_anyOf_i0"></a>57.2.5.2.1.1. Property `ReduxOptions > logger > anyOf > LogOptions > wandb > anyOf > WandbOptions > resume_id > anyOf > item 0`

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | No       |

###### <a name="logger_anyOf_i1_wandb_anyOf_i1_resume_id_anyOf_i1"></a>57.2.5.2.1.2. Property `ReduxOptions > logger > anyOf > LogOptions > wandb > anyOf > WandbOptions > resume_id > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

###### <a name="logger_anyOf_i1_wandb_anyOf_i1_project"></a>57.2.5.2.2. Property `ReduxOptions > logger > anyOf > LogOptions > wandb > anyOf > WandbOptions > project`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `combining`      |
| **Required**              | No               |
| **Additional properties** | Any type allowed |
| **Default**               | `null`           |

| Any of(Option)                                             |
| ---------------------------------------------------------- |
| [item 0](#logger_anyOf_i1_wandb_anyOf_i1_project_anyOf_i0) |
| [item 1](#logger_anyOf_i1_wandb_anyOf_i1_project_anyOf_i1) |

###### <a name="logger_anyOf_i1_wandb_anyOf_i1_project_anyOf_i0"></a>57.2.5.2.2.1. Property `ReduxOptions > logger > anyOf > LogOptions > wandb > anyOf > WandbOptions > project > anyOf > item 0`

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | No       |

###### <a name="logger_anyOf_i1_wandb_anyOf_i1_project_anyOf_i1"></a>57.2.5.2.2.2. Property `ReduxOptions > logger > anyOf > LogOptions > wandb > anyOf > WandbOptions > project > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

## <a name="dist_params"></a>58. Property `ReduxOptions > dist_params`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `combining`      |
| **Required**              | No               |
| **Additional properties** | Any type allowed |
| **Default**               | `null`           |

| Any of(Option)                  |
| ------------------------------- |
| [item 0](#dist_params_anyOf_i0) |
| [item 1](#dist_params_anyOf_i1) |

### <a name="dist_params_anyOf_i0"></a>58.1. Property `ReduxOptions > dist_params > anyOf > item 0`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `object`         |
| **Required**              | No               |
| **Additional properties** | Any type allowed |

### <a name="dist_params_anyOf_i1"></a>58.2. Property `ReduxOptions > dist_params > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

## <a name="onnx"></a>59. Property `ReduxOptions > onnx`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `combining`      |
| **Required**              | No               |
| **Additional properties** | Any type allowed |
| **Default**               | `null`           |

| Any of(Option)                |
| ----------------------------- |
| [item 0](#onnx_anyOf_i0)      |
| [OnnxOptions](#onnx_anyOf_i1) |

### <a name="onnx_anyOf_i0"></a>59.1. Property `ReduxOptions > onnx > anyOf > item 0`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

### <a name="onnx_anyOf_i1"></a>59.2. Property `ReduxOptions > onnx > anyOf > OnnxOptions`

**Title:** OnnxOptions

|                           |                     |
| ------------------------- | ------------------- |
| **Type**                  | `object`            |
| **Required**              | No                  |
| **Additional properties** | Not allowed         |
| **Defined in**            | #/$defs/OnnxOptions |

| Property                                                 | Pattern | Type    | Deprecated | Definition | Title/Description |
| -------------------------------------------------------- | ------- | ------- | ---------- | ---------- | ----------------- |
| - [dynamo](#onnx_anyOf_i1_dynamo )                       | No      | boolean | No         | -          | -                 |
| - [opset](#onnx_anyOf_i1_opset )                         | No      | integer | No         | -          | -                 |
| - [use_static_shapes](#onnx_anyOf_i1_use_static_shapes ) | No      | boolean | No         | -          | -                 |
| - [shape](#onnx_anyOf_i1_shape )                         | No      | string  | No         | -          | -                 |
| - [verify](#onnx_anyOf_i1_verify )                       | No      | boolean | No         | -          | -                 |
| - [fp16](#onnx_anyOf_i1_fp16 )                           | No      | boolean | No         | -          | -                 |
| - [optimize](#onnx_anyOf_i1_optimize )                   | No      | boolean | No         | -          | -                 |

#### <a name="onnx_anyOf_i1_dynamo"></a>59.2.1. Property `ReduxOptions > onnx > anyOf > OnnxOptions > dynamo`

|              |           |
| ------------ | --------- |
| **Type**     | `boolean` |
| **Required** | No        |
| **Default**  | `false`   |

#### <a name="onnx_anyOf_i1_opset"></a>59.2.2. Property `ReduxOptions > onnx > anyOf > OnnxOptions > opset`

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | No        |
| **Default**  | `20`      |

#### <a name="onnx_anyOf_i1_use_static_shapes"></a>59.2.3. Property `ReduxOptions > onnx > anyOf > OnnxOptions > use_static_shapes`

|              |           |
| ------------ | --------- |
| **Type**     | `boolean` |
| **Required** | No        |
| **Default**  | `false`   |

#### <a name="onnx_anyOf_i1_shape"></a>59.2.4. Property `ReduxOptions > onnx > anyOf > OnnxOptions > shape`

|              |               |
| ------------ | ------------- |
| **Type**     | `string`      |
| **Required** | No            |
| **Default**  | `"3x256x256"` |

#### <a name="onnx_anyOf_i1_verify"></a>59.2.5. Property `ReduxOptions > onnx > anyOf > OnnxOptions > verify`

|              |           |
| ------------ | --------- |
| **Type**     | `boolean` |
| **Required** | No        |
| **Default**  | `true`    |

#### <a name="onnx_anyOf_i1_fp16"></a>59.2.6. Property `ReduxOptions > onnx > anyOf > OnnxOptions > fp16`

|              |           |
| ------------ | --------- |
| **Type**     | `boolean` |
| **Required** | No        |
| **Default**  | `false`   |

#### <a name="onnx_anyOf_i1_optimize"></a>59.2.7. Property `ReduxOptions > onnx > anyOf > OnnxOptions > optimize`

|              |           |
| ------------ | --------- |
| **Type**     | `boolean` |
| **Required** | No        |
| **Default**  | `true`    |

## <a name="find_unused_parameters"></a>60. Property `ReduxOptions > find_unused_parameters`

|              |           |
| ------------ | --------- |
| **Type**     | `boolean` |
| **Required** | No        |
| **Default**  | `false`   |

----------------------------------------------------------------------------------------------------------------------------
Generated using [json-schema-for-humans](https://github.com/coveooss/json-schema-for-humans) on 2024-12-19 at 08:05:25 -0500
