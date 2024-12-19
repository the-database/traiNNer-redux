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
- [27. Property `ReduxOptions > force_high_order_degradation_filename_masks`](#force_high_order_degradation_filename_masks)
  - [27.1. ReduxOptions > force_high_order_degradation_filename_masks > force_high_order_degradation_filename_masks items](#force_high_order_degradation_filename_masks_items)
- [28. Property `ReduxOptions > force_dataroot_lq_filename_masks`](#force_dataroot_lq_filename_masks)
  - [28.1. ReduxOptions > force_dataroot_lq_filename_masks > force_dataroot_lq_filename_masks items](#force_dataroot_lq_filename_masks_items)
- [29. Property `ReduxOptions > lq_usm`](#lq_usm)
- [30. Property `ReduxOptions > lq_usm_radius_range`](#lq_usm_radius_range)
  - [30.1. ReduxOptions > lq_usm_radius_range > lq_usm_radius_range item 0](#autogenerated_heading_2)
  - [30.2. ReduxOptions > lq_usm_radius_range > lq_usm_radius_range item 1](#autogenerated_heading_3)
- [31. Property `ReduxOptions > blur_prob`](#blur_prob)
- [32. Property `ReduxOptions > resize_prob`](#resize_prob)
  - [32.1. ReduxOptions > resize_prob > resize_prob items](#resize_prob_items)
- [33. Property `ReduxOptions > resize_mode_list`](#resize_mode_list)
  - [33.1. ReduxOptions > resize_mode_list > resize_mode_list items](#resize_mode_list_items)
- [34. Property `ReduxOptions > resize_mode_prob`](#resize_mode_prob)
  - [34.1. ReduxOptions > resize_mode_prob > resize_mode_prob items](#resize_mode_prob_items)
- [35. Property `ReduxOptions > resize_range`](#resize_range)
  - [35.1. ReduxOptions > resize_range > resize_range item 0](#autogenerated_heading_4)
  - [35.2. ReduxOptions > resize_range > resize_range item 1](#autogenerated_heading_5)
- [36. Property `ReduxOptions > gaussian_noise_prob`](#gaussian_noise_prob)
- [37. Property `ReduxOptions > noise_range`](#noise_range)
  - [37.1. ReduxOptions > noise_range > noise_range item 0](#autogenerated_heading_6)
  - [37.2. ReduxOptions > noise_range > noise_range item 1](#autogenerated_heading_7)
- [38. Property `ReduxOptions > poisson_scale_range`](#poisson_scale_range)
  - [38.1. ReduxOptions > poisson_scale_range > poisson_scale_range item 0](#autogenerated_heading_8)
  - [38.2. ReduxOptions > poisson_scale_range > poisson_scale_range item 1](#autogenerated_heading_9)
- [39. Property `ReduxOptions > gray_noise_prob`](#gray_noise_prob)
- [40. Property `ReduxOptions > jpeg_prob`](#jpeg_prob)
- [41. Property `ReduxOptions > jpeg_range`](#jpeg_range)
  - [41.1. ReduxOptions > jpeg_range > jpeg_range item 0](#autogenerated_heading_10)
  - [41.2. ReduxOptions > jpeg_range > jpeg_range item 1](#autogenerated_heading_11)
- [42. Property `ReduxOptions > blur_prob2`](#blur_prob2)
- [43. Property `ReduxOptions > resize_prob2`](#resize_prob2)
  - [43.1. ReduxOptions > resize_prob2 > resize_prob2 items](#resize_prob2_items)
- [44. Property `ReduxOptions > resize_mode_list2`](#resize_mode_list2)
  - [44.1. ReduxOptions > resize_mode_list2 > resize_mode_list2 items](#resize_mode_list2_items)
- [45. Property `ReduxOptions > resize_mode_prob2`](#resize_mode_prob2)
  - [45.1. ReduxOptions > resize_mode_prob2 > resize_mode_prob2 items](#resize_mode_prob2_items)
- [46. Property `ReduxOptions > resize_range2`](#resize_range2)
  - [46.1. ReduxOptions > resize_range2 > resize_range2 item 0](#autogenerated_heading_12)
  - [46.2. ReduxOptions > resize_range2 > resize_range2 item 1](#autogenerated_heading_13)
- [47. Property `ReduxOptions > gaussian_noise_prob2`](#gaussian_noise_prob2)
- [48. Property `ReduxOptions > noise_range2`](#noise_range2)
  - [48.1. ReduxOptions > noise_range2 > noise_range2 item 0](#autogenerated_heading_14)
  - [48.2. ReduxOptions > noise_range2 > noise_range2 item 1](#autogenerated_heading_15)
- [49. Property `ReduxOptions > poisson_scale_range2`](#poisson_scale_range2)
  - [49.1. ReduxOptions > poisson_scale_range2 > poisson_scale_range2 item 0](#autogenerated_heading_16)
  - [49.2. ReduxOptions > poisson_scale_range2 > poisson_scale_range2 item 1](#autogenerated_heading_17)
- [50. Property `ReduxOptions > gray_noise_prob2`](#gray_noise_prob2)
- [51. Property `ReduxOptions > jpeg_prob2`](#jpeg_prob2)
- [52. Property `ReduxOptions > jpeg_range2`](#jpeg_range2)
  - [52.1. ReduxOptions > jpeg_range2 > jpeg_range2 items](#jpeg_range2_items)
- [53. Property `ReduxOptions > resize_mode_list3`](#resize_mode_list3)
  - [53.1. ReduxOptions > resize_mode_list3 > resize_mode_list3 items](#resize_mode_list3_items)
- [54. Property `ReduxOptions > resize_mode_prob3`](#resize_mode_prob3)
  - [54.1. ReduxOptions > resize_mode_prob3 > resize_mode_prob3 items](#resize_mode_prob3_items)
- [55. Property `ReduxOptions > queue_size`](#queue_size)
- [56. Property `ReduxOptions > datasets`](#datasets)
  - [56.1. Property `ReduxOptions > datasets > DatasetOptions`](#datasets_additionalProperties)
    - [56.1.1. Property `ReduxOptions > datasets > DatasetOptions > name`](#datasets_additionalProperties_name)
    - [56.1.2. Property `ReduxOptions > datasets > DatasetOptions > type`](#datasets_additionalProperties_type)
    - [56.1.3. Property `ReduxOptions > datasets > DatasetOptions > io_backend`](#datasets_additionalProperties_io_backend)
    - [56.1.4. Property `ReduxOptions > datasets > DatasetOptions > num_worker_per_gpu`](#datasets_additionalProperties_num_worker_per_gpu)
      - [56.1.4.1. Property `ReduxOptions > datasets > DatasetOptions > num_worker_per_gpu > anyOf > item 0`](#datasets_additionalProperties_num_worker_per_gpu_anyOf_i0)
      - [56.1.4.2. Property `ReduxOptions > datasets > DatasetOptions > num_worker_per_gpu > anyOf > item 1`](#datasets_additionalProperties_num_worker_per_gpu_anyOf_i1)
    - [56.1.5. Property `ReduxOptions > datasets > DatasetOptions > batch_size_per_gpu`](#datasets_additionalProperties_batch_size_per_gpu)
      - [56.1.5.1. Property `ReduxOptions > datasets > DatasetOptions > batch_size_per_gpu > anyOf > item 0`](#datasets_additionalProperties_batch_size_per_gpu_anyOf_i0)
      - [56.1.5.2. Property `ReduxOptions > datasets > DatasetOptions > batch_size_per_gpu > anyOf > item 1`](#datasets_additionalProperties_batch_size_per_gpu_anyOf_i1)
    - [56.1.6. Property `ReduxOptions > datasets > DatasetOptions > accum_iter`](#datasets_additionalProperties_accum_iter)
    - [56.1.7. Property `ReduxOptions > datasets > DatasetOptions > use_hflip`](#datasets_additionalProperties_use_hflip)
    - [56.1.8. Property `ReduxOptions > datasets > DatasetOptions > use_rot`](#datasets_additionalProperties_use_rot)
    - [56.1.9. Property `ReduxOptions > datasets > DatasetOptions > mean`](#datasets_additionalProperties_mean)
      - [56.1.9.1. Property `ReduxOptions > datasets > DatasetOptions > mean > anyOf > item 0`](#datasets_additionalProperties_mean_anyOf_i0)
        - [56.1.9.1.1. ReduxOptions > datasets > DatasetOptions > mean > anyOf > item 0 > item 0 items](#datasets_additionalProperties_mean_anyOf_i0_items)
      - [56.1.9.2. Property `ReduxOptions > datasets > DatasetOptions > mean > anyOf > item 1`](#datasets_additionalProperties_mean_anyOf_i1)
    - [56.1.10. Property `ReduxOptions > datasets > DatasetOptions > std`](#datasets_additionalProperties_std)
      - [56.1.10.1. Property `ReduxOptions > datasets > DatasetOptions > std > anyOf > item 0`](#datasets_additionalProperties_std_anyOf_i0)
        - [56.1.10.1.1. ReduxOptions > datasets > DatasetOptions > std > anyOf > item 0 > item 0 items](#datasets_additionalProperties_std_anyOf_i0_items)
      - [56.1.10.2. Property `ReduxOptions > datasets > DatasetOptions > std > anyOf > item 1`](#datasets_additionalProperties_std_anyOf_i1)
    - [56.1.11. Property `ReduxOptions > datasets > DatasetOptions > gt_size`](#datasets_additionalProperties_gt_size)
      - [56.1.11.1. Property `ReduxOptions > datasets > DatasetOptions > gt_size > anyOf > item 0`](#datasets_additionalProperties_gt_size_anyOf_i0)
      - [56.1.11.2. Property `ReduxOptions > datasets > DatasetOptions > gt_size > anyOf > item 1`](#datasets_additionalProperties_gt_size_anyOf_i1)
    - [56.1.12. Property `ReduxOptions > datasets > DatasetOptions > lq_size`](#datasets_additionalProperties_lq_size)
      - [56.1.12.1. Property `ReduxOptions > datasets > DatasetOptions > lq_size > anyOf > item 0`](#datasets_additionalProperties_lq_size_anyOf_i0)
      - [56.1.12.2. Property `ReduxOptions > datasets > DatasetOptions > lq_size > anyOf > item 1`](#datasets_additionalProperties_lq_size_anyOf_i1)
    - [56.1.13. Property `ReduxOptions > datasets > DatasetOptions > color`](#datasets_additionalProperties_color)
      - [56.1.13.1. Property `ReduxOptions > datasets > DatasetOptions > color > anyOf > item 0`](#datasets_additionalProperties_color_anyOf_i0)
      - [56.1.13.2. Property `ReduxOptions > datasets > DatasetOptions > color > anyOf > item 1`](#datasets_additionalProperties_color_anyOf_i1)
    - [56.1.14. Property `ReduxOptions > datasets > DatasetOptions > phase`](#datasets_additionalProperties_phase)
      - [56.1.14.1. Property `ReduxOptions > datasets > DatasetOptions > phase > anyOf > item 0`](#datasets_additionalProperties_phase_anyOf_i0)
      - [56.1.14.2. Property `ReduxOptions > datasets > DatasetOptions > phase > anyOf > item 1`](#datasets_additionalProperties_phase_anyOf_i1)
    - [56.1.15. Property `ReduxOptions > datasets > DatasetOptions > scale`](#datasets_additionalProperties_scale)
      - [56.1.15.1. Property `ReduxOptions > datasets > DatasetOptions > scale > anyOf > item 0`](#datasets_additionalProperties_scale_anyOf_i0)
      - [56.1.15.2. Property `ReduxOptions > datasets > DatasetOptions > scale > anyOf > item 1`](#datasets_additionalProperties_scale_anyOf_i1)
    - [56.1.16. Property `ReduxOptions > datasets > DatasetOptions > dataset_enlarge_ratio`](#datasets_additionalProperties_dataset_enlarge_ratio)
      - [56.1.16.1. Property `ReduxOptions > datasets > DatasetOptions > dataset_enlarge_ratio > anyOf > item 0`](#datasets_additionalProperties_dataset_enlarge_ratio_anyOf_i0)
      - [56.1.16.2. Property `ReduxOptions > datasets > DatasetOptions > dataset_enlarge_ratio > anyOf > item 1`](#datasets_additionalProperties_dataset_enlarge_ratio_anyOf_i1)
    - [56.1.17. Property `ReduxOptions > datasets > DatasetOptions > prefetch_mode`](#datasets_additionalProperties_prefetch_mode)
      - [56.1.17.1. Property `ReduxOptions > datasets > DatasetOptions > prefetch_mode > anyOf > item 0`](#datasets_additionalProperties_prefetch_mode_anyOf_i0)
      - [56.1.17.2. Property `ReduxOptions > datasets > DatasetOptions > prefetch_mode > anyOf > item 1`](#datasets_additionalProperties_prefetch_mode_anyOf_i1)
    - [56.1.18. Property `ReduxOptions > datasets > DatasetOptions > pin_memory`](#datasets_additionalProperties_pin_memory)
    - [56.1.19. Property `ReduxOptions > datasets > DatasetOptions > persistent_workers`](#datasets_additionalProperties_persistent_workers)
    - [56.1.20. Property `ReduxOptions > datasets > DatasetOptions > num_prefetch_queue`](#datasets_additionalProperties_num_prefetch_queue)
    - [56.1.21. Property `ReduxOptions > datasets > DatasetOptions > prefetch_factor`](#datasets_additionalProperties_prefetch_factor)
    - [56.1.22. Property `ReduxOptions > datasets > DatasetOptions > clip_size`](#datasets_additionalProperties_clip_size)
      - [56.1.22.1. Property `ReduxOptions > datasets > DatasetOptions > clip_size > anyOf > item 0`](#datasets_additionalProperties_clip_size_anyOf_i0)
      - [56.1.22.2. Property `ReduxOptions > datasets > DatasetOptions > clip_size > anyOf > item 1`](#datasets_additionalProperties_clip_size_anyOf_i1)
    - [56.1.23. Property `ReduxOptions > datasets > DatasetOptions > dataroot_gt`](#datasets_additionalProperties_dataroot_gt)
      - [56.1.23.1. Property `ReduxOptions > datasets > DatasetOptions > dataroot_gt > anyOf > item 0`](#datasets_additionalProperties_dataroot_gt_anyOf_i0)
      - [56.1.23.2. Property `ReduxOptions > datasets > DatasetOptions > dataroot_gt > anyOf > item 1`](#datasets_additionalProperties_dataroot_gt_anyOf_i1)
        - [56.1.23.2.1. ReduxOptions > datasets > DatasetOptions > dataroot_gt > anyOf > item 1 > item 1 items](#datasets_additionalProperties_dataroot_gt_anyOf_i1_items)
      - [56.1.23.3. Property `ReduxOptions > datasets > DatasetOptions > dataroot_gt > anyOf > item 2`](#datasets_additionalProperties_dataroot_gt_anyOf_i2)
    - [56.1.24. Property `ReduxOptions > datasets > DatasetOptions > dataroot_lq`](#datasets_additionalProperties_dataroot_lq)
      - [56.1.24.1. Property `ReduxOptions > datasets > DatasetOptions > dataroot_lq > anyOf > item 0`](#datasets_additionalProperties_dataroot_lq_anyOf_i0)
      - [56.1.24.2. Property `ReduxOptions > datasets > DatasetOptions > dataroot_lq > anyOf > item 1`](#datasets_additionalProperties_dataroot_lq_anyOf_i1)
        - [56.1.24.2.1. ReduxOptions > datasets > DatasetOptions > dataroot_lq > anyOf > item 1 > item 1 items](#datasets_additionalProperties_dataroot_lq_anyOf_i1_items)
      - [56.1.24.3. Property `ReduxOptions > datasets > DatasetOptions > dataroot_lq > anyOf > item 2`](#datasets_additionalProperties_dataroot_lq_anyOf_i2)
    - [56.1.25. Property `ReduxOptions > datasets > DatasetOptions > meta_info`](#datasets_additionalProperties_meta_info)
      - [56.1.25.1. Property `ReduxOptions > datasets > DatasetOptions > meta_info > anyOf > item 0`](#datasets_additionalProperties_meta_info_anyOf_i0)
      - [56.1.25.2. Property `ReduxOptions > datasets > DatasetOptions > meta_info > anyOf > item 1`](#datasets_additionalProperties_meta_info_anyOf_i1)
    - [56.1.26. Property `ReduxOptions > datasets > DatasetOptions > filename_tmpl`](#datasets_additionalProperties_filename_tmpl)
    - [56.1.27. Property `ReduxOptions > datasets > DatasetOptions > blur_kernel_size`](#datasets_additionalProperties_blur_kernel_size)
    - [56.1.28. Property `ReduxOptions > datasets > DatasetOptions > kernel_list`](#datasets_additionalProperties_kernel_list)
      - [56.1.28.1. ReduxOptions > datasets > DatasetOptions > kernel_list > kernel_list items](#datasets_additionalProperties_kernel_list_items)
    - [56.1.29. Property `ReduxOptions > datasets > DatasetOptions > kernel_prob`](#datasets_additionalProperties_kernel_prob)
      - [56.1.29.1. ReduxOptions > datasets > DatasetOptions > kernel_prob > kernel_prob items](#datasets_additionalProperties_kernel_prob_items)
    - [56.1.30. Property `ReduxOptions > datasets > DatasetOptions > kernel_range`](#datasets_additionalProperties_kernel_range)
      - [56.1.30.1. ReduxOptions > datasets > DatasetOptions > kernel_range > kernel_range item 0](#autogenerated_heading_18)
      - [56.1.30.2. ReduxOptions > datasets > DatasetOptions > kernel_range > kernel_range item 1](#autogenerated_heading_19)
    - [56.1.31. Property `ReduxOptions > datasets > DatasetOptions > sinc_prob`](#datasets_additionalProperties_sinc_prob)
    - [56.1.32. Property `ReduxOptions > datasets > DatasetOptions > blur_sigma`](#datasets_additionalProperties_blur_sigma)
      - [56.1.32.1. ReduxOptions > datasets > DatasetOptions > blur_sigma > blur_sigma item 0](#autogenerated_heading_20)
      - [56.1.32.2. ReduxOptions > datasets > DatasetOptions > blur_sigma > blur_sigma item 1](#autogenerated_heading_21)
    - [56.1.33. Property `ReduxOptions > datasets > DatasetOptions > betag_range`](#datasets_additionalProperties_betag_range)
      - [56.1.33.1. ReduxOptions > datasets > DatasetOptions > betag_range > betag_range item 0](#autogenerated_heading_22)
      - [56.1.33.2. ReduxOptions > datasets > DatasetOptions > betag_range > betag_range item 1](#autogenerated_heading_23)
    - [56.1.34. Property `ReduxOptions > datasets > DatasetOptions > betap_range`](#datasets_additionalProperties_betap_range)
      - [56.1.34.1. ReduxOptions > datasets > DatasetOptions > betap_range > betap_range item 0](#autogenerated_heading_24)
      - [56.1.34.2. ReduxOptions > datasets > DatasetOptions > betap_range > betap_range item 1](#autogenerated_heading_25)
    - [56.1.35. Property `ReduxOptions > datasets > DatasetOptions > blur_kernel_size2`](#datasets_additionalProperties_blur_kernel_size2)
    - [56.1.36. Property `ReduxOptions > datasets > DatasetOptions > kernel_list2`](#datasets_additionalProperties_kernel_list2)
      - [56.1.36.1. ReduxOptions > datasets > DatasetOptions > kernel_list2 > kernel_list2 items](#datasets_additionalProperties_kernel_list2_items)
    - [56.1.37. Property `ReduxOptions > datasets > DatasetOptions > kernel_prob2`](#datasets_additionalProperties_kernel_prob2)
      - [56.1.37.1. ReduxOptions > datasets > DatasetOptions > kernel_prob2 > kernel_prob2 items](#datasets_additionalProperties_kernel_prob2_items)
    - [56.1.38. Property `ReduxOptions > datasets > DatasetOptions > kernel_range2`](#datasets_additionalProperties_kernel_range2)
      - [56.1.38.1. ReduxOptions > datasets > DatasetOptions > kernel_range2 > kernel_range2 item 0](#autogenerated_heading_26)
      - [56.1.38.2. ReduxOptions > datasets > DatasetOptions > kernel_range2 > kernel_range2 item 1](#autogenerated_heading_27)
    - [56.1.39. Property `ReduxOptions > datasets > DatasetOptions > sinc_prob2`](#datasets_additionalProperties_sinc_prob2)
    - [56.1.40. Property `ReduxOptions > datasets > DatasetOptions > blur_sigma2`](#datasets_additionalProperties_blur_sigma2)
      - [56.1.40.1. ReduxOptions > datasets > DatasetOptions > blur_sigma2 > blur_sigma2 item 0](#autogenerated_heading_28)
      - [56.1.40.2. ReduxOptions > datasets > DatasetOptions > blur_sigma2 > blur_sigma2 item 1](#autogenerated_heading_29)
    - [56.1.41. Property `ReduxOptions > datasets > DatasetOptions > betag_range2`](#datasets_additionalProperties_betag_range2)
      - [56.1.41.1. ReduxOptions > datasets > DatasetOptions > betag_range2 > betag_range2 item 0](#autogenerated_heading_30)
      - [56.1.41.2. ReduxOptions > datasets > DatasetOptions > betag_range2 > betag_range2 item 1](#autogenerated_heading_31)
    - [56.1.42. Property `ReduxOptions > datasets > DatasetOptions > betap_range2`](#datasets_additionalProperties_betap_range2)
      - [56.1.42.1. ReduxOptions > datasets > DatasetOptions > betap_range2 > betap_range2 item 0](#autogenerated_heading_32)
      - [56.1.42.2. ReduxOptions > datasets > DatasetOptions > betap_range2 > betap_range2 item 1](#autogenerated_heading_33)
    - [56.1.43. Property `ReduxOptions > datasets > DatasetOptions > final_sinc_prob`](#datasets_additionalProperties_final_sinc_prob)
    - [56.1.44. Property `ReduxOptions > datasets > DatasetOptions > final_kernel_range`](#datasets_additionalProperties_final_kernel_range)
      - [56.1.44.1. ReduxOptions > datasets > DatasetOptions > final_kernel_range > final_kernel_range item 0](#autogenerated_heading_34)
      - [56.1.44.2. ReduxOptions > datasets > DatasetOptions > final_kernel_range > final_kernel_range item 1](#autogenerated_heading_35)
- [57. Property `ReduxOptions > train`](#train)
  - [57.1. Property `ReduxOptions > train > anyOf > item 0`](#train_anyOf_i0)
  - [57.2. Property `ReduxOptions > train > anyOf > TrainOptions`](#train_anyOf_i1)
    - [57.2.1. Property `ReduxOptions > train > anyOf > TrainOptions > total_iter`](#train_anyOf_i1_total_iter)
    - [57.2.2. Property `ReduxOptions > train > anyOf > TrainOptions > optim_g`](#train_anyOf_i1_optim_g)
    - [57.2.3. Property `ReduxOptions > train > anyOf > TrainOptions > ema_decay`](#train_anyOf_i1_ema_decay)
    - [57.2.4. Property `ReduxOptions > train > anyOf > TrainOptions > grad_clip`](#train_anyOf_i1_grad_clip)
    - [57.2.5. Property `ReduxOptions > train > anyOf > TrainOptions > warmup_iter`](#train_anyOf_i1_warmup_iter)
    - [57.2.6. Property `ReduxOptions > train > anyOf > TrainOptions > scheduler`](#train_anyOf_i1_scheduler)
      - [57.2.6.1. Property `ReduxOptions > train > anyOf > TrainOptions > scheduler > anyOf > item 0`](#train_anyOf_i1_scheduler_anyOf_i0)
      - [57.2.6.2. Property `ReduxOptions > train > anyOf > TrainOptions > scheduler > anyOf > SchedulerOptions`](#train_anyOf_i1_scheduler_anyOf_i1)
        - [57.2.6.2.1. Property `ReduxOptions > train > anyOf > TrainOptions > scheduler > anyOf > SchedulerOptions > type`](#train_anyOf_i1_scheduler_anyOf_i1_type)
        - [57.2.6.2.2. Property `ReduxOptions > train > anyOf > TrainOptions > scheduler > anyOf > SchedulerOptions > milestones`](#train_anyOf_i1_scheduler_anyOf_i1_milestones)
          - [57.2.6.2.2.1. ReduxOptions > train > anyOf > TrainOptions > scheduler > anyOf > SchedulerOptions > milestones > milestones items](#train_anyOf_i1_scheduler_anyOf_i1_milestones_items)
        - [57.2.6.2.3. Property `ReduxOptions > train > anyOf > TrainOptions > scheduler > anyOf > SchedulerOptions > gamma`](#train_anyOf_i1_scheduler_anyOf_i1_gamma)
    - [57.2.7. Property `ReduxOptions > train > anyOf > TrainOptions > optim_d`](#train_anyOf_i1_optim_d)
      - [57.2.7.1. Property `ReduxOptions > train > anyOf > TrainOptions > optim_d > anyOf > item 0`](#train_anyOf_i1_optim_d_anyOf_i0)
      - [57.2.7.2. Property `ReduxOptions > train > anyOf > TrainOptions > optim_d > anyOf > item 1`](#train_anyOf_i1_optim_d_anyOf_i1)
    - [57.2.8. Property `ReduxOptions > train > anyOf > TrainOptions > losses`](#train_anyOf_i1_losses)
      - [57.2.8.1. Property `ReduxOptions > train > anyOf > TrainOptions > losses > anyOf > item 0`](#train_anyOf_i1_losses_anyOf_i0)
        - [57.2.8.1.1. ReduxOptions > train > anyOf > TrainOptions > losses > anyOf > item 0 > item 0 items](#train_anyOf_i1_losses_anyOf_i0_items)
      - [57.2.8.2. Property `ReduxOptions > train > anyOf > TrainOptions > losses > anyOf > item 1`](#train_anyOf_i1_losses_anyOf_i1)
    - [57.2.9. Property `ReduxOptions > train > anyOf > TrainOptions > pixel_opt`](#train_anyOf_i1_pixel_opt)
      - [57.2.9.1. Property `ReduxOptions > train > anyOf > TrainOptions > pixel_opt > anyOf > item 0`](#train_anyOf_i1_pixel_opt_anyOf_i0)
      - [57.2.9.2. Property `ReduxOptions > train > anyOf > TrainOptions > pixel_opt > anyOf > item 1`](#train_anyOf_i1_pixel_opt_anyOf_i1)
    - [57.2.10. Property `ReduxOptions > train > anyOf > TrainOptions > mssim_opt`](#train_anyOf_i1_mssim_opt)
      - [57.2.10.1. Property `ReduxOptions > train > anyOf > TrainOptions > mssim_opt > anyOf > item 0`](#train_anyOf_i1_mssim_opt_anyOf_i0)
      - [57.2.10.2. Property `ReduxOptions > train > anyOf > TrainOptions > mssim_opt > anyOf > item 1`](#train_anyOf_i1_mssim_opt_anyOf_i1)
    - [57.2.11. Property `ReduxOptions > train > anyOf > TrainOptions > ms_ssim_l1_opt`](#train_anyOf_i1_ms_ssim_l1_opt)
      - [57.2.11.1. Property `ReduxOptions > train > anyOf > TrainOptions > ms_ssim_l1_opt > anyOf > item 0`](#train_anyOf_i1_ms_ssim_l1_opt_anyOf_i0)
      - [57.2.11.2. Property `ReduxOptions > train > anyOf > TrainOptions > ms_ssim_l1_opt > anyOf > item 1`](#train_anyOf_i1_ms_ssim_l1_opt_anyOf_i1)
    - [57.2.12. Property `ReduxOptions > train > anyOf > TrainOptions > perceptual_opt`](#train_anyOf_i1_perceptual_opt)
      - [57.2.12.1. Property `ReduxOptions > train > anyOf > TrainOptions > perceptual_opt > anyOf > item 0`](#train_anyOf_i1_perceptual_opt_anyOf_i0)
      - [57.2.12.2. Property `ReduxOptions > train > anyOf > TrainOptions > perceptual_opt > anyOf > item 1`](#train_anyOf_i1_perceptual_opt_anyOf_i1)
    - [57.2.13. Property `ReduxOptions > train > anyOf > TrainOptions > contextual_opt`](#train_anyOf_i1_contextual_opt)
      - [57.2.13.1. Property `ReduxOptions > train > anyOf > TrainOptions > contextual_opt > anyOf > item 0`](#train_anyOf_i1_contextual_opt_anyOf_i0)
      - [57.2.13.2. Property `ReduxOptions > train > anyOf > TrainOptions > contextual_opt > anyOf > item 1`](#train_anyOf_i1_contextual_opt_anyOf_i1)
    - [57.2.14. Property `ReduxOptions > train > anyOf > TrainOptions > dists_opt`](#train_anyOf_i1_dists_opt)
      - [57.2.14.1. Property `ReduxOptions > train > anyOf > TrainOptions > dists_opt > anyOf > item 0`](#train_anyOf_i1_dists_opt_anyOf_i0)
      - [57.2.14.2. Property `ReduxOptions > train > anyOf > TrainOptions > dists_opt > anyOf > item 1`](#train_anyOf_i1_dists_opt_anyOf_i1)
    - [57.2.15. Property `ReduxOptions > train > anyOf > TrainOptions > hr_inversion_opt`](#train_anyOf_i1_hr_inversion_opt)
      - [57.2.15.1. Property `ReduxOptions > train > anyOf > TrainOptions > hr_inversion_opt > anyOf > item 0`](#train_anyOf_i1_hr_inversion_opt_anyOf_i0)
      - [57.2.15.2. Property `ReduxOptions > train > anyOf > TrainOptions > hr_inversion_opt > anyOf > item 1`](#train_anyOf_i1_hr_inversion_opt_anyOf_i1)
    - [57.2.16. Property `ReduxOptions > train > anyOf > TrainOptions > dinov2_opt`](#train_anyOf_i1_dinov2_opt)
      - [57.2.16.1. Property `ReduxOptions > train > anyOf > TrainOptions > dinov2_opt > anyOf > item 0`](#train_anyOf_i1_dinov2_opt_anyOf_i0)
      - [57.2.16.2. Property `ReduxOptions > train > anyOf > TrainOptions > dinov2_opt > anyOf > item 1`](#train_anyOf_i1_dinov2_opt_anyOf_i1)
    - [57.2.17. Property `ReduxOptions > train > anyOf > TrainOptions > topiq_opt`](#train_anyOf_i1_topiq_opt)
      - [57.2.17.1. Property `ReduxOptions > train > anyOf > TrainOptions > topiq_opt > anyOf > item 0`](#train_anyOf_i1_topiq_opt_anyOf_i0)
      - [57.2.17.2. Property `ReduxOptions > train > anyOf > TrainOptions > topiq_opt > anyOf > item 1`](#train_anyOf_i1_topiq_opt_anyOf_i1)
    - [57.2.18. Property `ReduxOptions > train > anyOf > TrainOptions > pd_opt`](#train_anyOf_i1_pd_opt)
      - [57.2.18.1. Property `ReduxOptions > train > anyOf > TrainOptions > pd_opt > anyOf > item 0`](#train_anyOf_i1_pd_opt_anyOf_i0)
      - [57.2.18.2. Property `ReduxOptions > train > anyOf > TrainOptions > pd_opt > anyOf > item 1`](#train_anyOf_i1_pd_opt_anyOf_i1)
    - [57.2.19. Property `ReduxOptions > train > anyOf > TrainOptions > fd_opt`](#train_anyOf_i1_fd_opt)
      - [57.2.19.1. Property `ReduxOptions > train > anyOf > TrainOptions > fd_opt > anyOf > item 0`](#train_anyOf_i1_fd_opt_anyOf_i0)
      - [57.2.19.2. Property `ReduxOptions > train > anyOf > TrainOptions > fd_opt > anyOf > item 1`](#train_anyOf_i1_fd_opt_anyOf_i1)
    - [57.2.20. Property `ReduxOptions > train > anyOf > TrainOptions > ldl_opt`](#train_anyOf_i1_ldl_opt)
      - [57.2.20.1. Property `ReduxOptions > train > anyOf > TrainOptions > ldl_opt > anyOf > item 0`](#train_anyOf_i1_ldl_opt_anyOf_i0)
      - [57.2.20.2. Property `ReduxOptions > train > anyOf > TrainOptions > ldl_opt > anyOf > item 1`](#train_anyOf_i1_ldl_opt_anyOf_i1)
    - [57.2.21. Property `ReduxOptions > train > anyOf > TrainOptions > hsluv_opt`](#train_anyOf_i1_hsluv_opt)
      - [57.2.21.1. Property `ReduxOptions > train > anyOf > TrainOptions > hsluv_opt > anyOf > item 0`](#train_anyOf_i1_hsluv_opt_anyOf_i0)
      - [57.2.21.2. Property `ReduxOptions > train > anyOf > TrainOptions > hsluv_opt > anyOf > item 1`](#train_anyOf_i1_hsluv_opt_anyOf_i1)
    - [57.2.22. Property `ReduxOptions > train > anyOf > TrainOptions > gan_opt`](#train_anyOf_i1_gan_opt)
      - [57.2.22.1. Property `ReduxOptions > train > anyOf > TrainOptions > gan_opt > anyOf > item 0`](#train_anyOf_i1_gan_opt_anyOf_i0)
      - [57.2.22.2. Property `ReduxOptions > train > anyOf > TrainOptions > gan_opt > anyOf > item 1`](#train_anyOf_i1_gan_opt_anyOf_i1)
    - [57.2.23. Property `ReduxOptions > train > anyOf > TrainOptions > color_opt`](#train_anyOf_i1_color_opt)
      - [57.2.23.1. Property `ReduxOptions > train > anyOf > TrainOptions > color_opt > anyOf > item 0`](#train_anyOf_i1_color_opt_anyOf_i0)
      - [57.2.23.2. Property `ReduxOptions > train > anyOf > TrainOptions > color_opt > anyOf > item 1`](#train_anyOf_i1_color_opt_anyOf_i1)
    - [57.2.24. Property `ReduxOptions > train > anyOf > TrainOptions > luma_opt`](#train_anyOf_i1_luma_opt)
      - [57.2.24.1. Property `ReduxOptions > train > anyOf > TrainOptions > luma_opt > anyOf > item 0`](#train_anyOf_i1_luma_opt_anyOf_i0)
      - [57.2.24.2. Property `ReduxOptions > train > anyOf > TrainOptions > luma_opt > anyOf > item 1`](#train_anyOf_i1_luma_opt_anyOf_i1)
    - [57.2.25. Property `ReduxOptions > train > anyOf > TrainOptions > avg_opt`](#train_anyOf_i1_avg_opt)
      - [57.2.25.1. Property `ReduxOptions > train > anyOf > TrainOptions > avg_opt > anyOf > item 0`](#train_anyOf_i1_avg_opt_anyOf_i0)
      - [57.2.25.2. Property `ReduxOptions > train > anyOf > TrainOptions > avg_opt > anyOf > item 1`](#train_anyOf_i1_avg_opt_anyOf_i1)
    - [57.2.26. Property `ReduxOptions > train > anyOf > TrainOptions > bicubic_opt`](#train_anyOf_i1_bicubic_opt)
      - [57.2.26.1. Property `ReduxOptions > train > anyOf > TrainOptions > bicubic_opt > anyOf > item 0`](#train_anyOf_i1_bicubic_opt_anyOf_i0)
      - [57.2.26.2. Property `ReduxOptions > train > anyOf > TrainOptions > bicubic_opt > anyOf > item 1`](#train_anyOf_i1_bicubic_opt_anyOf_i1)
    - [57.2.27. Property `ReduxOptions > train > anyOf > TrainOptions > use_moa`](#train_anyOf_i1_use_moa)
    - [57.2.28. Property `ReduxOptions > train > anyOf > TrainOptions > moa_augs`](#train_anyOf_i1_moa_augs)
      - [57.2.28.1. ReduxOptions > train > anyOf > TrainOptions > moa_augs > moa_augs items](#train_anyOf_i1_moa_augs_items)
    - [57.2.29. Property `ReduxOptions > train > anyOf > TrainOptions > moa_probs`](#train_anyOf_i1_moa_probs)
      - [57.2.29.1. ReduxOptions > train > anyOf > TrainOptions > moa_probs > moa_probs items](#train_anyOf_i1_moa_probs_items)
    - [57.2.30. Property `ReduxOptions > train > anyOf > TrainOptions > moa_debug`](#train_anyOf_i1_moa_debug)
    - [57.2.31. Property `ReduxOptions > train > anyOf > TrainOptions > moa_debug_limit`](#train_anyOf_i1_moa_debug_limit)
- [58. Property `ReduxOptions > val`](#val)
  - [58.1. Property `ReduxOptions > val > anyOf > item 0`](#val_anyOf_i0)
  - [58.2. Property `ReduxOptions > val > anyOf > ValOptions`](#val_anyOf_i1)
    - [58.2.1. Property `ReduxOptions > val > anyOf > ValOptions > val_enabled`](#val_anyOf_i1_val_enabled)
    - [58.2.2. Property `ReduxOptions > val > anyOf > ValOptions > save_img`](#val_anyOf_i1_save_img)
    - [58.2.3. Property `ReduxOptions > val > anyOf > ValOptions > val_freq`](#val_anyOf_i1_val_freq)
      - [58.2.3.1. Property `ReduxOptions > val > anyOf > ValOptions > val_freq > anyOf > item 0`](#val_anyOf_i1_val_freq_anyOf_i0)
      - [58.2.3.2. Property `ReduxOptions > val > anyOf > ValOptions > val_freq > anyOf > item 1`](#val_anyOf_i1_val_freq_anyOf_i1)
    - [58.2.4. Property `ReduxOptions > val > anyOf > ValOptions > suffix`](#val_anyOf_i1_suffix)
      - [58.2.4.1. Property `ReduxOptions > val > anyOf > ValOptions > suffix > anyOf > item 0`](#val_anyOf_i1_suffix_anyOf_i0)
      - [58.2.4.2. Property `ReduxOptions > val > anyOf > ValOptions > suffix > anyOf > item 1`](#val_anyOf_i1_suffix_anyOf_i1)
    - [58.2.5. Property `ReduxOptions > val > anyOf > ValOptions > metrics_enabled`](#val_anyOf_i1_metrics_enabled)
    - [58.2.6. Property `ReduxOptions > val > anyOf > ValOptions > metrics`](#val_anyOf_i1_metrics)
      - [58.2.6.1. Property `ReduxOptions > val > anyOf > ValOptions > metrics > anyOf > item 0`](#val_anyOf_i1_metrics_anyOf_i0)
      - [58.2.6.2. Property `ReduxOptions > val > anyOf > ValOptions > metrics > anyOf > item 1`](#val_anyOf_i1_metrics_anyOf_i1)
    - [58.2.7. Property `ReduxOptions > val > anyOf > ValOptions > pbar`](#val_anyOf_i1_pbar)
- [59. Property `ReduxOptions > logger`](#logger)
  - [59.1. Property `ReduxOptions > logger > anyOf > item 0`](#logger_anyOf_i0)
  - [59.2. Property `ReduxOptions > logger > anyOf > LogOptions`](#logger_anyOf_i1)
    - [59.2.1. Property `ReduxOptions > logger > anyOf > LogOptions > print_freq`](#logger_anyOf_i1_print_freq)
    - [59.2.2. Property `ReduxOptions > logger > anyOf > LogOptions > save_checkpoint_freq`](#logger_anyOf_i1_save_checkpoint_freq)
    - [59.2.3. Property `ReduxOptions > logger > anyOf > LogOptions > use_tb_logger`](#logger_anyOf_i1_use_tb_logger)
    - [59.2.4. Property `ReduxOptions > logger > anyOf > LogOptions > save_checkpoint_format`](#logger_anyOf_i1_save_checkpoint_format)
    - [59.2.5. Property `ReduxOptions > logger > anyOf > LogOptions > wandb`](#logger_anyOf_i1_wandb)
      - [59.2.5.1. Property `ReduxOptions > logger > anyOf > LogOptions > wandb > anyOf > item 0`](#logger_anyOf_i1_wandb_anyOf_i0)
      - [59.2.5.2. Property `ReduxOptions > logger > anyOf > LogOptions > wandb > anyOf > WandbOptions`](#logger_anyOf_i1_wandb_anyOf_i1)
        - [59.2.5.2.1. Property `ReduxOptions > logger > anyOf > LogOptions > wandb > anyOf > WandbOptions > resume_id`](#logger_anyOf_i1_wandb_anyOf_i1_resume_id)
          - [59.2.5.2.1.1. Property `ReduxOptions > logger > anyOf > LogOptions > wandb > anyOf > WandbOptions > resume_id > anyOf > item 0`](#logger_anyOf_i1_wandb_anyOf_i1_resume_id_anyOf_i0)
          - [59.2.5.2.1.2. Property `ReduxOptions > logger > anyOf > LogOptions > wandb > anyOf > WandbOptions > resume_id > anyOf > item 1`](#logger_anyOf_i1_wandb_anyOf_i1_resume_id_anyOf_i1)
        - [59.2.5.2.2. Property `ReduxOptions > logger > anyOf > LogOptions > wandb > anyOf > WandbOptions > project`](#logger_anyOf_i1_wandb_anyOf_i1_project)
          - [59.2.5.2.2.1. Property `ReduxOptions > logger > anyOf > LogOptions > wandb > anyOf > WandbOptions > project > anyOf > item 0`](#logger_anyOf_i1_wandb_anyOf_i1_project_anyOf_i0)
          - [59.2.5.2.2.2. Property `ReduxOptions > logger > anyOf > LogOptions > wandb > anyOf > WandbOptions > project > anyOf > item 1`](#logger_anyOf_i1_wandb_anyOf_i1_project_anyOf_i1)
- [60. Property `ReduxOptions > dist_params`](#dist_params)
  - [60.1. Property `ReduxOptions > dist_params > anyOf > item 0`](#dist_params_anyOf_i0)
  - [60.2. Property `ReduxOptions > dist_params > anyOf > item 1`](#dist_params_anyOf_i1)
- [61. Property `ReduxOptions > onnx`](#onnx)
  - [61.1. Property `ReduxOptions > onnx > anyOf > item 0`](#onnx_anyOf_i0)
  - [61.2. Property `ReduxOptions > onnx > anyOf > OnnxOptions`](#onnx_anyOf_i1)
    - [61.2.1. Property `ReduxOptions > onnx > anyOf > OnnxOptions > dynamo`](#onnx_anyOf_i1_dynamo)
    - [61.2.2. Property `ReduxOptions > onnx > anyOf > OnnxOptions > opset`](#onnx_anyOf_i1_opset)
    - [61.2.3. Property `ReduxOptions > onnx > anyOf > OnnxOptions > use_static_shapes`](#onnx_anyOf_i1_use_static_shapes)
    - [61.2.4. Property `ReduxOptions > onnx > anyOf > OnnxOptions > shape`](#onnx_anyOf_i1_shape)
    - [61.2.5. Property `ReduxOptions > onnx > anyOf > OnnxOptions > verify`](#onnx_anyOf_i1_verify)
    - [61.2.6. Property `ReduxOptions > onnx > anyOf > OnnxOptions > fp16`](#onnx_anyOf_i1_fp16)
    - [61.2.7. Property `ReduxOptions > onnx > anyOf > OnnxOptions > optimize`](#onnx_anyOf_i1_optimize)
- [62. Property `ReduxOptions > find_unused_parameters`](#find_unused_parameters)

**Title:** ReduxOptions

|                           |                      |
| ------------------------- | -------------------- |
| **Type**                  | `object`             |
| **Required**              | No                   |
| **Additional properties** | Not allowed          |
| **Defined in**            | #/$defs/ReduxOptions |

| Property                                                                                       | Pattern | Type            | Deprecated | Definition             | Title/Description                                                                                                                                                                                                                                                                                         |
| ---------------------------------------------------------------------------------------------- | ------- | --------------- | ---------- | ---------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| + [name](#name )                                                                               | No      | string          | No         | -                      | Name of the experiment. It should be a unique name. If you enable auto resume, any experiment with this name will be resumed instead of starting a new training run.                                                                                                                                      |
| + [scale](#scale )                                                                             | No      | integer         | No         | -                      | Scale of the model. Most architectures support a scale of 1, 2, 3, 4, or 8. A scale of 1 can be used for restoration models that don't change the resolution of the input image. A scale of 2 means the width and height of the input image are doubled, so a 640x480 input will be upscaled to 1280x960. |
| + [num_gpu](#num_gpu )                                                                         | No      | Combination     | No         | -                      | The number of GPUs to use for training, if using multiple GPUs.                                                                                                                                                                                                                                           |
| + [path](#path )                                                                               | No      | object          | No         | In #/$defs/PathOptions | PathOptions                                                                                                                                                                                                                                                                                               |
| + [network_g](#network_g )                                                                     | No      | object          | No         | -                      | The options for the generator model.                                                                                                                                                                                                                                                                      |
| - [network_d](#network_d )                                                                     | No      | Combination     | No         | -                      | The options for the discriminator model.                                                                                                                                                                                                                                                                  |
| - [manual_seed](#manual_seed )                                                                 | No      | Combination     | No         | -                      | Deterministic mode, slows down training. Only use for reproducible experiments.                                                                                                                                                                                                                           |
| - [deterministic](#deterministic )                                                             | No      | Combination     | No         | -                      | -                                                                                                                                                                                                                                                                                                         |
| - [dist](#dist )                                                                               | No      | Combination     | No         | -                      | -                                                                                                                                                                                                                                                                                                         |
| - [launcher](#launcher )                                                                       | No      | Combination     | No         | -                      | -                                                                                                                                                                                                                                                                                                         |
| - [rank](#rank )                                                                               | No      | Combination     | No         | -                      | -                                                                                                                                                                                                                                                                                                         |
| - [world_size](#world_size )                                                                   | No      | Combination     | No         | -                      | -                                                                                                                                                                                                                                                                                                         |
| - [auto_resume](#auto_resume )                                                                 | No      | Combination     | No         | -                      | -                                                                                                                                                                                                                                                                                                         |
| - [resume](#resume )                                                                           | No      | integer         | No         | -                      | -                                                                                                                                                                                                                                                                                                         |
| - [is_train](#is_train )                                                                       | No      | Combination     | No         | -                      | -                                                                                                                                                                                                                                                                                                         |
| - [root_path](#root_path )                                                                     | No      | Combination     | No         | -                      | -                                                                                                                                                                                                                                                                                                         |
| - [use_amp](#use_amp )                                                                         | No      | boolean         | No         | -                      | Speed up training and reduce VRAM usage. NVIDIA only.                                                                                                                                                                                                                                                     |
| - [amp_bf16](#amp_bf16 )                                                                       | No      | boolean         | No         | -                      | Use bf16 instead of fp16 for AMP, RTX 3000 series or newer only. Only recommended if fp16 doesn't work.                                                                                                                                                                                                   |
| - [use_channels_last](#use_channels_last )                                                     | No      | boolean         | No         | -                      | Enable channels last memory format while using AMP. Reduces VRAM and speeds up training for most architectures, but some architectures are slower with channels last.                                                                                                                                     |
| - [fast_matmul](#fast_matmul )                                                                 | No      | boolean         | No         | -                      | Trade precision for performance.                                                                                                                                                                                                                                                                          |
| - [use_compile](#use_compile )                                                                 | No      | boolean         | No         | -                      | Enable torch.compile for the generator model, which takes time on startup to compile the model, but can speed up training after the model is compiled.                                                                                                                                                    |
| - [detect_anomaly](#detect_anomaly )                                                           | No      | boolean         | No         | -                      | Whether or not to enable anomaly detection, which can be useful for debugging NaNs that occur during training. Has a significant performance hit and should be disabled when not debugging.                                                                                                               |
| - [high_order_degradation](#high_order_degradation )                                           | No      | boolean         | No         | -                      | Whether or not to enable OTF (on the fly) degradations, which generates LRs on the fly.                                                                                                                                                                                                                   |
| - [high_order_degradations_debug](#high_order_degradations_debug )                             | No      | boolean         | No         | -                      | Whether or not to enable debugging for OTF, which saves the OTF generated LR images so they can be inspected to view the effect of different OTF settings.                                                                                                                                                |
| - [high_order_degradations_debug_limit](#high_order_degradations_debug_limit )                 | No      | integer         | No         | -                      | The maximum number of OTF images to save when debugging is enabled.                                                                                                                                                                                                                                       |
| - [dataroot_lq_prob](#dataroot_lq_prob )                                                       | No      | number          | No         | -                      | Probability of using paired LR data instead of OTF LR data.                                                                                                                                                                                                                                               |
| - [force_high_order_degradation_filename_masks](#force_high_order_degradation_filename_masks ) | No      | array of string | No         | -                      | -                                                                                                                                                                                                                                                                                                         |
| - [force_dataroot_lq_filename_masks](#force_dataroot_lq_filename_masks )                       | No      | array of string | No         | -                      | -                                                                                                                                                                                                                                                                                                         |
| - [lq_usm](#lq_usm )                                                                           | No      | boolean         | No         | -                      | -                                                                                                                                                                                                                                                                                                         |
| - [lq_usm_radius_range](#lq_usm_radius_range )                                                 | No      | array           | No         | -                      | -                                                                                                                                                                                                                                                                                                         |
| - [blur_prob](#blur_prob )                                                                     | No      | number          | No         | -                      | -                                                                                                                                                                                                                                                                                                         |
| - [resize_prob](#resize_prob )                                                                 | No      | array of number | No         | -                      | -                                                                                                                                                                                                                                                                                                         |
| - [resize_mode_list](#resize_mode_list )                                                       | No      | array of string | No         | -                      | -                                                                                                                                                                                                                                                                                                         |
| - [resize_mode_prob](#resize_mode_prob )                                                       | No      | array of number | No         | -                      | -                                                                                                                                                                                                                                                                                                         |
| - [resize_range](#resize_range )                                                               | No      | array           | No         | -                      | -                                                                                                                                                                                                                                                                                                         |
| - [gaussian_noise_prob](#gaussian_noise_prob )                                                 | No      | number          | No         | -                      | -                                                                                                                                                                                                                                                                                                         |
| - [noise_range](#noise_range )                                                                 | No      | array           | No         | -                      | -                                                                                                                                                                                                                                                                                                         |
| - [poisson_scale_range](#poisson_scale_range )                                                 | No      | array           | No         | -                      | -                                                                                                                                                                                                                                                                                                         |
| - [gray_noise_prob](#gray_noise_prob )                                                         | No      | number          | No         | -                      | -                                                                                                                                                                                                                                                                                                         |
| - [jpeg_prob](#jpeg_prob )                                                                     | No      | number          | No         | -                      | -                                                                                                                                                                                                                                                                                                         |
| - [jpeg_range](#jpeg_range )                                                                   | No      | array           | No         | -                      | -                                                                                                                                                                                                                                                                                                         |
| - [blur_prob2](#blur_prob2 )                                                                   | No      | number          | No         | -                      | -                                                                                                                                                                                                                                                                                                         |
| - [resize_prob2](#resize_prob2 )                                                               | No      | array of number | No         | -                      | -                                                                                                                                                                                                                                                                                                         |
| - [resize_mode_list2](#resize_mode_list2 )                                                     | No      | array of string | No         | -                      | -                                                                                                                                                                                                                                                                                                         |
| - [resize_mode_prob2](#resize_mode_prob2 )                                                     | No      | array of number | No         | -                      | -                                                                                                                                                                                                                                                                                                         |
| - [resize_range2](#resize_range2 )                                                             | No      | array           | No         | -                      | -                                                                                                                                                                                                                                                                                                         |
| - [gaussian_noise_prob2](#gaussian_noise_prob2 )                                               | No      | number          | No         | -                      | -                                                                                                                                                                                                                                                                                                         |
| - [noise_range2](#noise_range2 )                                                               | No      | array           | No         | -                      | -                                                                                                                                                                                                                                                                                                         |
| - [poisson_scale_range2](#poisson_scale_range2 )                                               | No      | array           | No         | -                      | -                                                                                                                                                                                                                                                                                                         |
| - [gray_noise_prob2](#gray_noise_prob2 )                                                       | No      | number          | No         | -                      | -                                                                                                                                                                                                                                                                                                         |
| - [jpeg_prob2](#jpeg_prob2 )                                                                   | No      | number          | No         | -                      | -                                                                                                                                                                                                                                                                                                         |
| - [jpeg_range2](#jpeg_range2 )                                                                 | No      | array of number | No         | -                      | -                                                                                                                                                                                                                                                                                                         |
| - [resize_mode_list3](#resize_mode_list3 )                                                     | No      | array of string | No         | -                      | -                                                                                                                                                                                                                                                                                                         |
| - [resize_mode_prob3](#resize_mode_prob3 )                                                     | No      | array of number | No         | -                      | -                                                                                                                                                                                                                                                                                                         |
| - [queue_size](#queue_size )                                                                   | No      | integer         | No         | -                      | -                                                                                                                                                                                                                                                                                                         |
| - [datasets](#datasets )                                                                       | No      | object          | No         | -                      | -                                                                                                                                                                                                                                                                                                         |
| - [train](#train )                                                                             | No      | Combination     | No         | -                      | -                                                                                                                                                                                                                                                                                                         |
| - [val](#val )                                                                                 | No      | Combination     | No         | -                      | -                                                                                                                                                                                                                                                                                                         |
| - [logger](#logger )                                                                           | No      | Combination     | No         | -                      | -                                                                                                                                                                                                                                                                                                         |
| - [dist_params](#dist_params )                                                                 | No      | Combination     | No         | -                      | -                                                                                                                                                                                                                                                                                                         |
| - [onnx](#onnx )                                                                               | No      | Combination     | No         | -                      | -                                                                                                                                                                                                                                                                                                         |
| - [find_unused_parameters](#find_unused_parameters )                                           | No      | boolean         | No         | -                      | -                                                                                                                                                                                                                                                                                                         |

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
| **Type**                  | `object`         |
| **Required**              | Yes              |
| **Additional properties** | Any type allowed |

**Description:** The options for the generator model.

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

## <a name="force_high_order_degradation_filename_masks"></a>27. Property `ReduxOptions > force_high_order_degradation_filename_masks`

|              |                   |
| ------------ | ----------------- |
| **Type**     | `array of string` |
| **Required** | No                |
| **Default**  | `[]`              |

|                      | Array restrictions |
| -------------------- | ------------------ |
| **Min items**        | N/A                |
| **Max items**        | N/A                |
| **Items unicity**    | False              |
| **Additional items** | False              |
| **Tuple validation** | See below          |

| Each item of this array must be                                                                         | Description |
| ------------------------------------------------------------------------------------------------------- | ----------- |
| [force_high_order_degradation_filename_masks items](#force_high_order_degradation_filename_masks_items) | -           |

### <a name="force_high_order_degradation_filename_masks_items"></a>27.1. ReduxOptions > force_high_order_degradation_filename_masks > force_high_order_degradation_filename_masks items

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | No       |

## <a name="force_dataroot_lq_filename_masks"></a>28. Property `ReduxOptions > force_dataroot_lq_filename_masks`

|              |                   |
| ------------ | ----------------- |
| **Type**     | `array of string` |
| **Required** | No                |
| **Default**  | `[]`              |

|                      | Array restrictions |
| -------------------- | ------------------ |
| **Min items**        | N/A                |
| **Max items**        | N/A                |
| **Items unicity**    | False              |
| **Additional items** | False              |
| **Tuple validation** | See below          |

| Each item of this array must be                                                   | Description |
| --------------------------------------------------------------------------------- | ----------- |
| [force_dataroot_lq_filename_masks items](#force_dataroot_lq_filename_masks_items) | -           |

### <a name="force_dataroot_lq_filename_masks_items"></a>28.1. ReduxOptions > force_dataroot_lq_filename_masks > force_dataroot_lq_filename_masks items

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | No       |

## <a name="lq_usm"></a>29. Property `ReduxOptions > lq_usm`

|              |           |
| ------------ | --------- |
| **Type**     | `boolean` |
| **Required** | No        |
| **Default**  | `false`   |

## <a name="lq_usm_radius_range"></a>30. Property `ReduxOptions > lq_usm_radius_range`

|              |           |
| ------------ | --------- |
| **Type**     | `array`   |
| **Required** | No        |
| **Default**  | `[1, 25]` |

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

### <a name="autogenerated_heading_2"></a>30.1. ReduxOptions > lq_usm_radius_range > lq_usm_radius_range item 0

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | No        |

### <a name="autogenerated_heading_3"></a>30.2. ReduxOptions > lq_usm_radius_range > lq_usm_radius_range item 1

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | No        |

## <a name="blur_prob"></a>31. Property `ReduxOptions > blur_prob`

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |
| **Default**  | `0`      |

## <a name="resize_prob"></a>32. Property `ReduxOptions > resize_prob`

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

### <a name="resize_prob_items"></a>32.1. ReduxOptions > resize_prob > resize_prob items

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |

## <a name="resize_mode_list"></a>33. Property `ReduxOptions > resize_mode_list`

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

### <a name="resize_mode_list_items"></a>33.1. ReduxOptions > resize_mode_list > resize_mode_list items

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | No       |

## <a name="resize_mode_prob"></a>34. Property `ReduxOptions > resize_mode_prob`

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

### <a name="resize_mode_prob_items"></a>34.1. ReduxOptions > resize_mode_prob > resize_mode_prob items

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |

## <a name="resize_range"></a>35. Property `ReduxOptions > resize_range`

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

### <a name="autogenerated_heading_4"></a>35.1. ReduxOptions > resize_range > resize_range item 0

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |

### <a name="autogenerated_heading_5"></a>35.2. ReduxOptions > resize_range > resize_range item 1

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |

## <a name="gaussian_noise_prob"></a>36. Property `ReduxOptions > gaussian_noise_prob`

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |
| **Default**  | `0`      |

## <a name="noise_range"></a>37. Property `ReduxOptions > noise_range`

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

### <a name="autogenerated_heading_6"></a>37.1. ReduxOptions > noise_range > noise_range item 0

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |

### <a name="autogenerated_heading_7"></a>37.2. ReduxOptions > noise_range > noise_range item 1

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |

## <a name="poisson_scale_range"></a>38. Property `ReduxOptions > poisson_scale_range`

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

### <a name="autogenerated_heading_8"></a>38.1. ReduxOptions > poisson_scale_range > poisson_scale_range item 0

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |

### <a name="autogenerated_heading_9"></a>38.2. ReduxOptions > poisson_scale_range > poisson_scale_range item 1

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |

## <a name="gray_noise_prob"></a>39. Property `ReduxOptions > gray_noise_prob`

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |
| **Default**  | `0`      |

## <a name="jpeg_prob"></a>40. Property `ReduxOptions > jpeg_prob`

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |
| **Default**  | `1`      |

## <a name="jpeg_range"></a>41. Property `ReduxOptions > jpeg_range`

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

### <a name="autogenerated_heading_10"></a>41.1. ReduxOptions > jpeg_range > jpeg_range item 0

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |

### <a name="autogenerated_heading_11"></a>41.2. ReduxOptions > jpeg_range > jpeg_range item 1

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |

## <a name="blur_prob2"></a>42. Property `ReduxOptions > blur_prob2`

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |
| **Default**  | `0`      |

## <a name="resize_prob2"></a>43. Property `ReduxOptions > resize_prob2`

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

### <a name="resize_prob2_items"></a>43.1. ReduxOptions > resize_prob2 > resize_prob2 items

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |

## <a name="resize_mode_list2"></a>44. Property `ReduxOptions > resize_mode_list2`

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

### <a name="resize_mode_list2_items"></a>44.1. ReduxOptions > resize_mode_list2 > resize_mode_list2 items

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | No       |

## <a name="resize_mode_prob2"></a>45. Property `ReduxOptions > resize_mode_prob2`

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

### <a name="resize_mode_prob2_items"></a>45.1. ReduxOptions > resize_mode_prob2 > resize_mode_prob2 items

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |

## <a name="resize_range2"></a>46. Property `ReduxOptions > resize_range2`

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

### <a name="autogenerated_heading_12"></a>46.1. ReduxOptions > resize_range2 > resize_range2 item 0

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |

### <a name="autogenerated_heading_13"></a>46.2. ReduxOptions > resize_range2 > resize_range2 item 1

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |

## <a name="gaussian_noise_prob2"></a>47. Property `ReduxOptions > gaussian_noise_prob2`

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |
| **Default**  | `0`      |

## <a name="noise_range2"></a>48. Property `ReduxOptions > noise_range2`

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

### <a name="autogenerated_heading_14"></a>48.1. ReduxOptions > noise_range2 > noise_range2 item 0

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |

### <a name="autogenerated_heading_15"></a>48.2. ReduxOptions > noise_range2 > noise_range2 item 1

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |

## <a name="poisson_scale_range2"></a>49. Property `ReduxOptions > poisson_scale_range2`

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

### <a name="autogenerated_heading_16"></a>49.1. ReduxOptions > poisson_scale_range2 > poisson_scale_range2 item 0

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |

### <a name="autogenerated_heading_17"></a>49.2. ReduxOptions > poisson_scale_range2 > poisson_scale_range2 item 1

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |

## <a name="gray_noise_prob2"></a>50. Property `ReduxOptions > gray_noise_prob2`

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |
| **Default**  | `0`      |

## <a name="jpeg_prob2"></a>51. Property `ReduxOptions > jpeg_prob2`

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |
| **Default**  | `1`      |

## <a name="jpeg_range2"></a>52. Property `ReduxOptions > jpeg_range2`

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

### <a name="jpeg_range2_items"></a>52.1. ReduxOptions > jpeg_range2 > jpeg_range2 items

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |

## <a name="resize_mode_list3"></a>53. Property `ReduxOptions > resize_mode_list3`

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

### <a name="resize_mode_list3_items"></a>53.1. ReduxOptions > resize_mode_list3 > resize_mode_list3 items

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | No       |

## <a name="resize_mode_prob3"></a>54. Property `ReduxOptions > resize_mode_prob3`

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

### <a name="resize_mode_prob3_items"></a>54.1. ReduxOptions > resize_mode_prob3 > resize_mode_prob3 items

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |

## <a name="queue_size"></a>55. Property `ReduxOptions > queue_size`

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | No        |
| **Default**  | `180`     |

## <a name="datasets"></a>56. Property `ReduxOptions > datasets`

|                           |                                                                                       |
| ------------------------- | ------------------------------------------------------------------------------------- |
| **Type**                  | `object`                                                                              |
| **Required**              | No                                                                                    |
| **Additional properties** | [Each additional property must conform to the schema](#datasets_additionalProperties) |
| **Default**               | `{}`                                                                                  |

| Property                              | Pattern | Type   | Deprecated | Definition                | Title/Description |
| ------------------------------------- | ------- | ------ | ---------- | ------------------------- | ----------------- |
| - [](#datasets_additionalProperties ) | No      | object | No         | In #/$defs/DatasetOptions | DatasetOptions    |

### <a name="datasets_additionalProperties"></a>56.1. Property `ReduxOptions > datasets > DatasetOptions`

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

#### <a name="datasets_additionalProperties_name"></a>56.1.1. Property `ReduxOptions > datasets > DatasetOptions > name`

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | Yes      |

**Description:** Name of the dataset. It should be unique compared to other datasets in this config, but the exact name isn't very important.

#### <a name="datasets_additionalProperties_type"></a>56.1.2. Property `ReduxOptions > datasets > DatasetOptions > type`

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | Yes      |

#### <a name="datasets_additionalProperties_io_backend"></a>56.1.3. Property `ReduxOptions > datasets > DatasetOptions > io_backend`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `object`         |
| **Required**              | Yes              |
| **Additional properties** | Any type allowed |

#### <a name="datasets_additionalProperties_num_worker_per_gpu"></a>56.1.4. Property `ReduxOptions > datasets > DatasetOptions > num_worker_per_gpu`

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

##### <a name="datasets_additionalProperties_num_worker_per_gpu_anyOf_i0"></a>56.1.4.1. Property `ReduxOptions > datasets > DatasetOptions > num_worker_per_gpu > anyOf > item 0`

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | No        |

##### <a name="datasets_additionalProperties_num_worker_per_gpu_anyOf_i1"></a>56.1.4.2. Property `ReduxOptions > datasets > DatasetOptions > num_worker_per_gpu > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

#### <a name="datasets_additionalProperties_batch_size_per_gpu"></a>56.1.5. Property `ReduxOptions > datasets > DatasetOptions > batch_size_per_gpu`

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

##### <a name="datasets_additionalProperties_batch_size_per_gpu_anyOf_i0"></a>56.1.5.1. Property `ReduxOptions > datasets > DatasetOptions > batch_size_per_gpu > anyOf > item 0`

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | No        |

##### <a name="datasets_additionalProperties_batch_size_per_gpu_anyOf_i1"></a>56.1.5.2. Property `ReduxOptions > datasets > DatasetOptions > batch_size_per_gpu > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

#### <a name="datasets_additionalProperties_accum_iter"></a>56.1.6. Property `ReduxOptions > datasets > DatasetOptions > accum_iter`

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | No        |
| **Default**  | `1`       |

**Description:** Using values larger than 1 simulates higher batch size by trading performance for reduced VRAM usage. If accum_iter = 4 and batch_size_per_gpu = 6 then effective batch size = 4 * 6 = 24 but performance may be as much as 4 times as slow.

#### <a name="datasets_additionalProperties_use_hflip"></a>56.1.7. Property `ReduxOptions > datasets > DatasetOptions > use_hflip`

|              |           |
| ------------ | --------- |
| **Type**     | `boolean` |
| **Required** | No        |
| **Default**  | `true`    |

**Description:** Randomly flip the images horizontally.

#### <a name="datasets_additionalProperties_use_rot"></a>56.1.8. Property `ReduxOptions > datasets > DatasetOptions > use_rot`

|              |           |
| ------------ | --------- |
| **Type**     | `boolean` |
| **Required** | No        |
| **Default**  | `true`    |

**Description:** Randomly rotate the images.

#### <a name="datasets_additionalProperties_mean"></a>56.1.9. Property `ReduxOptions > datasets > DatasetOptions > mean`

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

##### <a name="datasets_additionalProperties_mean_anyOf_i0"></a>56.1.9.1. Property `ReduxOptions > datasets > DatasetOptions > mean > anyOf > item 0`

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

###### <a name="datasets_additionalProperties_mean_anyOf_i0_items"></a>56.1.9.1.1. ReduxOptions > datasets > DatasetOptions > mean > anyOf > item 0 > item 0 items

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |

##### <a name="datasets_additionalProperties_mean_anyOf_i1"></a>56.1.9.2. Property `ReduxOptions > datasets > DatasetOptions > mean > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

#### <a name="datasets_additionalProperties_std"></a>56.1.10. Property `ReduxOptions > datasets > DatasetOptions > std`

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

##### <a name="datasets_additionalProperties_std_anyOf_i0"></a>56.1.10.1. Property `ReduxOptions > datasets > DatasetOptions > std > anyOf > item 0`

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

###### <a name="datasets_additionalProperties_std_anyOf_i0_items"></a>56.1.10.1.1. ReduxOptions > datasets > DatasetOptions > std > anyOf > item 0 > item 0 items

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |

##### <a name="datasets_additionalProperties_std_anyOf_i1"></a>56.1.10.2. Property `ReduxOptions > datasets > DatasetOptions > std > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

#### <a name="datasets_additionalProperties_gt_size"></a>56.1.11. Property `ReduxOptions > datasets > DatasetOptions > gt_size`

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

##### <a name="datasets_additionalProperties_gt_size_anyOf_i0"></a>56.1.11.1. Property `ReduxOptions > datasets > DatasetOptions > gt_size > anyOf > item 0`

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | No        |

##### <a name="datasets_additionalProperties_gt_size_anyOf_i1"></a>56.1.11.2. Property `ReduxOptions > datasets > DatasetOptions > gt_size > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

#### <a name="datasets_additionalProperties_lq_size"></a>56.1.12. Property `ReduxOptions > datasets > DatasetOptions > lq_size`

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

##### <a name="datasets_additionalProperties_lq_size_anyOf_i0"></a>56.1.12.1. Property `ReduxOptions > datasets > DatasetOptions > lq_size > anyOf > item 0`

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | No        |

##### <a name="datasets_additionalProperties_lq_size_anyOf_i1"></a>56.1.12.2. Property `ReduxOptions > datasets > DatasetOptions > lq_size > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

#### <a name="datasets_additionalProperties_color"></a>56.1.13. Property `ReduxOptions > datasets > DatasetOptions > color`

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

##### <a name="datasets_additionalProperties_color_anyOf_i0"></a>56.1.13.1. Property `ReduxOptions > datasets > DatasetOptions > color > anyOf > item 0`

|              |                    |
| ------------ | ------------------ |
| **Type**     | `enum (of string)` |
| **Required** | No                 |

Must be one of:
* "y"

##### <a name="datasets_additionalProperties_color_anyOf_i1"></a>56.1.13.2. Property `ReduxOptions > datasets > DatasetOptions > color > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

#### <a name="datasets_additionalProperties_phase"></a>56.1.14. Property `ReduxOptions > datasets > DatasetOptions > phase`

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

##### <a name="datasets_additionalProperties_phase_anyOf_i0"></a>56.1.14.1. Property `ReduxOptions > datasets > DatasetOptions > phase > anyOf > item 0`

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | No       |

##### <a name="datasets_additionalProperties_phase_anyOf_i1"></a>56.1.14.2. Property `ReduxOptions > datasets > DatasetOptions > phase > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

#### <a name="datasets_additionalProperties_scale"></a>56.1.15. Property `ReduxOptions > datasets > DatasetOptions > scale`

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

##### <a name="datasets_additionalProperties_scale_anyOf_i0"></a>56.1.15.1. Property `ReduxOptions > datasets > DatasetOptions > scale > anyOf > item 0`

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | No        |

##### <a name="datasets_additionalProperties_scale_anyOf_i1"></a>56.1.15.2. Property `ReduxOptions > datasets > DatasetOptions > scale > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

#### <a name="datasets_additionalProperties_dataset_enlarge_ratio"></a>56.1.16. Property `ReduxOptions > datasets > DatasetOptions > dataset_enlarge_ratio`

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

##### <a name="datasets_additionalProperties_dataset_enlarge_ratio_anyOf_i0"></a>56.1.16.1. Property `ReduxOptions > datasets > DatasetOptions > dataset_enlarge_ratio > anyOf > item 0`

|              |                    |
| ------------ | ------------------ |
| **Type**     | `enum (of string)` |
| **Required** | No                 |

Must be one of:
* "auto"

##### <a name="datasets_additionalProperties_dataset_enlarge_ratio_anyOf_i1"></a>56.1.16.2. Property `ReduxOptions > datasets > DatasetOptions > dataset_enlarge_ratio > anyOf > item 1`

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | No        |

#### <a name="datasets_additionalProperties_prefetch_mode"></a>56.1.17. Property `ReduxOptions > datasets > DatasetOptions > prefetch_mode`

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

##### <a name="datasets_additionalProperties_prefetch_mode_anyOf_i0"></a>56.1.17.1. Property `ReduxOptions > datasets > DatasetOptions > prefetch_mode > anyOf > item 0`

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | No       |

##### <a name="datasets_additionalProperties_prefetch_mode_anyOf_i1"></a>56.1.17.2. Property `ReduxOptions > datasets > DatasetOptions > prefetch_mode > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

#### <a name="datasets_additionalProperties_pin_memory"></a>56.1.18. Property `ReduxOptions > datasets > DatasetOptions > pin_memory`

|              |           |
| ------------ | --------- |
| **Type**     | `boolean` |
| **Required** | No        |
| **Default**  | `true`    |

#### <a name="datasets_additionalProperties_persistent_workers"></a>56.1.19. Property `ReduxOptions > datasets > DatasetOptions > persistent_workers`

|              |           |
| ------------ | --------- |
| **Type**     | `boolean` |
| **Required** | No        |
| **Default**  | `true`    |

#### <a name="datasets_additionalProperties_num_prefetch_queue"></a>56.1.20. Property `ReduxOptions > datasets > DatasetOptions > num_prefetch_queue`

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | No        |
| **Default**  | `1`       |

#### <a name="datasets_additionalProperties_prefetch_factor"></a>56.1.21. Property `ReduxOptions > datasets > DatasetOptions > prefetch_factor`

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | No        |
| **Default**  | `2`       |

#### <a name="datasets_additionalProperties_clip_size"></a>56.1.22. Property `ReduxOptions > datasets > DatasetOptions > clip_size`

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

##### <a name="datasets_additionalProperties_clip_size_anyOf_i0"></a>56.1.22.1. Property `ReduxOptions > datasets > DatasetOptions > clip_size > anyOf > item 0`

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | No        |

##### <a name="datasets_additionalProperties_clip_size_anyOf_i1"></a>56.1.22.2. Property `ReduxOptions > datasets > DatasetOptions > clip_size > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

#### <a name="datasets_additionalProperties_dataroot_gt"></a>56.1.23. Property `ReduxOptions > datasets > DatasetOptions > dataroot_gt`

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

##### <a name="datasets_additionalProperties_dataroot_gt_anyOf_i0"></a>56.1.23.1. Property `ReduxOptions > datasets > DatasetOptions > dataroot_gt > anyOf > item 0`

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | No       |

##### <a name="datasets_additionalProperties_dataroot_gt_anyOf_i1"></a>56.1.23.2. Property `ReduxOptions > datasets > DatasetOptions > dataroot_gt > anyOf > item 1`

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

###### <a name="datasets_additionalProperties_dataroot_gt_anyOf_i1_items"></a>56.1.23.2.1. ReduxOptions > datasets > DatasetOptions > dataroot_gt > anyOf > item 1 > item 1 items

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | No       |

##### <a name="datasets_additionalProperties_dataroot_gt_anyOf_i2"></a>56.1.23.3. Property `ReduxOptions > datasets > DatasetOptions > dataroot_gt > anyOf > item 2`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

#### <a name="datasets_additionalProperties_dataroot_lq"></a>56.1.24. Property `ReduxOptions > datasets > DatasetOptions > dataroot_lq`

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

##### <a name="datasets_additionalProperties_dataroot_lq_anyOf_i0"></a>56.1.24.1. Property `ReduxOptions > datasets > DatasetOptions > dataroot_lq > anyOf > item 0`

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | No       |

##### <a name="datasets_additionalProperties_dataroot_lq_anyOf_i1"></a>56.1.24.2. Property `ReduxOptions > datasets > DatasetOptions > dataroot_lq > anyOf > item 1`

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

###### <a name="datasets_additionalProperties_dataroot_lq_anyOf_i1_items"></a>56.1.24.2.1. ReduxOptions > datasets > DatasetOptions > dataroot_lq > anyOf > item 1 > item 1 items

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | No       |

##### <a name="datasets_additionalProperties_dataroot_lq_anyOf_i2"></a>56.1.24.3. Property `ReduxOptions > datasets > DatasetOptions > dataroot_lq > anyOf > item 2`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

#### <a name="datasets_additionalProperties_meta_info"></a>56.1.25. Property `ReduxOptions > datasets > DatasetOptions > meta_info`

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

##### <a name="datasets_additionalProperties_meta_info_anyOf_i0"></a>56.1.25.1. Property `ReduxOptions > datasets > DatasetOptions > meta_info > anyOf > item 0`

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | No       |

##### <a name="datasets_additionalProperties_meta_info_anyOf_i1"></a>56.1.25.2. Property `ReduxOptions > datasets > DatasetOptions > meta_info > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

#### <a name="datasets_additionalProperties_filename_tmpl"></a>56.1.26. Property `ReduxOptions > datasets > DatasetOptions > filename_tmpl`

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | No       |
| **Default**  | `"{}"`   |

#### <a name="datasets_additionalProperties_blur_kernel_size"></a>56.1.27. Property `ReduxOptions > datasets > DatasetOptions > blur_kernel_size`

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | No        |
| **Default**  | `12`      |

#### <a name="datasets_additionalProperties_kernel_list"></a>56.1.28. Property `ReduxOptions > datasets > DatasetOptions > kernel_list`

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

##### <a name="datasets_additionalProperties_kernel_list_items"></a>56.1.28.1. ReduxOptions > datasets > DatasetOptions > kernel_list > kernel_list items

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | No       |

#### <a name="datasets_additionalProperties_kernel_prob"></a>56.1.29. Property `ReduxOptions > datasets > DatasetOptions > kernel_prob`

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

##### <a name="datasets_additionalProperties_kernel_prob_items"></a>56.1.29.1. ReduxOptions > datasets > DatasetOptions > kernel_prob > kernel_prob items

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |

#### <a name="datasets_additionalProperties_kernel_range"></a>56.1.30. Property `ReduxOptions > datasets > DatasetOptions > kernel_range`

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

##### <a name="autogenerated_heading_18"></a>56.1.30.1. ReduxOptions > datasets > DatasetOptions > kernel_range > kernel_range item 0

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | No        |

##### <a name="autogenerated_heading_19"></a>56.1.30.2. ReduxOptions > datasets > DatasetOptions > kernel_range > kernel_range item 1

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | No        |

#### <a name="datasets_additionalProperties_sinc_prob"></a>56.1.31. Property `ReduxOptions > datasets > DatasetOptions > sinc_prob`

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |
| **Default**  | `0`      |

#### <a name="datasets_additionalProperties_blur_sigma"></a>56.1.32. Property `ReduxOptions > datasets > DatasetOptions > blur_sigma`

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

##### <a name="autogenerated_heading_20"></a>56.1.32.1. ReduxOptions > datasets > DatasetOptions > blur_sigma > blur_sigma item 0

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |

##### <a name="autogenerated_heading_21"></a>56.1.32.2. ReduxOptions > datasets > DatasetOptions > blur_sigma > blur_sigma item 1

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |

#### <a name="datasets_additionalProperties_betag_range"></a>56.1.33. Property `ReduxOptions > datasets > DatasetOptions > betag_range`

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

##### <a name="autogenerated_heading_22"></a>56.1.33.1. ReduxOptions > datasets > DatasetOptions > betag_range > betag_range item 0

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |

##### <a name="autogenerated_heading_23"></a>56.1.33.2. ReduxOptions > datasets > DatasetOptions > betag_range > betag_range item 1

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |

#### <a name="datasets_additionalProperties_betap_range"></a>56.1.34. Property `ReduxOptions > datasets > DatasetOptions > betap_range`

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

##### <a name="autogenerated_heading_24"></a>56.1.34.1. ReduxOptions > datasets > DatasetOptions > betap_range > betap_range item 0

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |

##### <a name="autogenerated_heading_25"></a>56.1.34.2. ReduxOptions > datasets > DatasetOptions > betap_range > betap_range item 1

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |

#### <a name="datasets_additionalProperties_blur_kernel_size2"></a>56.1.35. Property `ReduxOptions > datasets > DatasetOptions > blur_kernel_size2`

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | No        |
| **Default**  | `12`      |

#### <a name="datasets_additionalProperties_kernel_list2"></a>56.1.36. Property `ReduxOptions > datasets > DatasetOptions > kernel_list2`

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

##### <a name="datasets_additionalProperties_kernel_list2_items"></a>56.1.36.1. ReduxOptions > datasets > DatasetOptions > kernel_list2 > kernel_list2 items

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | No       |

#### <a name="datasets_additionalProperties_kernel_prob2"></a>56.1.37. Property `ReduxOptions > datasets > DatasetOptions > kernel_prob2`

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

##### <a name="datasets_additionalProperties_kernel_prob2_items"></a>56.1.37.1. ReduxOptions > datasets > DatasetOptions > kernel_prob2 > kernel_prob2 items

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |

#### <a name="datasets_additionalProperties_kernel_range2"></a>56.1.38. Property `ReduxOptions > datasets > DatasetOptions > kernel_range2`

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

##### <a name="autogenerated_heading_26"></a>56.1.38.1. ReduxOptions > datasets > DatasetOptions > kernel_range2 > kernel_range2 item 0

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | No        |

##### <a name="autogenerated_heading_27"></a>56.1.38.2. ReduxOptions > datasets > DatasetOptions > kernel_range2 > kernel_range2 item 1

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | No        |

#### <a name="datasets_additionalProperties_sinc_prob2"></a>56.1.39. Property `ReduxOptions > datasets > DatasetOptions > sinc_prob2`

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |
| **Default**  | `0`      |

#### <a name="datasets_additionalProperties_blur_sigma2"></a>56.1.40. Property `ReduxOptions > datasets > DatasetOptions > blur_sigma2`

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

##### <a name="autogenerated_heading_28"></a>56.1.40.1. ReduxOptions > datasets > DatasetOptions > blur_sigma2 > blur_sigma2 item 0

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |

##### <a name="autogenerated_heading_29"></a>56.1.40.2. ReduxOptions > datasets > DatasetOptions > blur_sigma2 > blur_sigma2 item 1

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |

#### <a name="datasets_additionalProperties_betag_range2"></a>56.1.41. Property `ReduxOptions > datasets > DatasetOptions > betag_range2`

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

##### <a name="autogenerated_heading_30"></a>56.1.41.1. ReduxOptions > datasets > DatasetOptions > betag_range2 > betag_range2 item 0

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |

##### <a name="autogenerated_heading_31"></a>56.1.41.2. ReduxOptions > datasets > DatasetOptions > betag_range2 > betag_range2 item 1

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |

#### <a name="datasets_additionalProperties_betap_range2"></a>56.1.42. Property `ReduxOptions > datasets > DatasetOptions > betap_range2`

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

##### <a name="autogenerated_heading_32"></a>56.1.42.1. ReduxOptions > datasets > DatasetOptions > betap_range2 > betap_range2 item 0

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |

##### <a name="autogenerated_heading_33"></a>56.1.42.2. ReduxOptions > datasets > DatasetOptions > betap_range2 > betap_range2 item 1

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |

#### <a name="datasets_additionalProperties_final_sinc_prob"></a>56.1.43. Property `ReduxOptions > datasets > DatasetOptions > final_sinc_prob`

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |
| **Default**  | `0`      |

#### <a name="datasets_additionalProperties_final_kernel_range"></a>56.1.44. Property `ReduxOptions > datasets > DatasetOptions > final_kernel_range`

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

##### <a name="autogenerated_heading_34"></a>56.1.44.1. ReduxOptions > datasets > DatasetOptions > final_kernel_range > final_kernel_range item 0

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | No        |

##### <a name="autogenerated_heading_35"></a>56.1.44.2. ReduxOptions > datasets > DatasetOptions > final_kernel_range > final_kernel_range item 1

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | No        |

## <a name="train"></a>57. Property `ReduxOptions > train`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `combining`      |
| **Required**              | No               |
| **Additional properties** | Any type allowed |
| **Default**               | `null`           |

| Any of(Option)                  |
| ------------------------------- |
| [item 0](#train_anyOf_i0)       |
| [TrainOptions](#train_anyOf_i1) |

### <a name="train_anyOf_i0"></a>57.1. Property `ReduxOptions > train > anyOf > item 0`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

### <a name="train_anyOf_i1"></a>57.2. Property `ReduxOptions > train > anyOf > TrainOptions`

**Title:** TrainOptions

|                           |                      |
| ------------------------- | -------------------- |
| **Type**                  | `object`             |
| **Required**              | No                   |
| **Additional properties** | Not allowed          |
| **Defined in**            | #/$defs/TrainOptions |

| Property                                                | Pattern | Type            | Deprecated | Definition | Title/Description                                                                                                                                       |
| ------------------------------------------------------- | ------- | --------------- | ---------- | ---------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| + [total_iter](#train_anyOf_i1_total_iter )             | No      | integer         | No         | -          | The total number of iterations to train.                                                                                                                |
| + [optim_g](#train_anyOf_i1_optim_g )                   | No      | object          | No         | -          | The optimizer to use for the generator model.                                                                                                           |
| - [ema_decay](#train_anyOf_i1_ema_decay )               | No      | number          | No         | -          | The decay factor to use for EMA (exponential moving average). Set to 0 to disable EMA.                                                                  |
| - [grad_clip](#train_anyOf_i1_grad_clip )               | No      | boolean         | No         | -          | Whether or not to enable gradient clipping, which can improve stability when using higher learning rates, but can also cause issues in some situations. |
| - [warmup_iter](#train_anyOf_i1_warmup_iter )           | No      | integer         | No         | -          | Gradually ramp up learning rates until this iteration, to stabilize early training. Use -1 to disable.                                                  |
| - [scheduler](#train_anyOf_i1_scheduler )               | No      | Combination     | No         | -          | Options for the optimizer scheduler. If there are multiple optimizers, both will use the same scheduler options.                                        |
| - [optim_d](#train_anyOf_i1_optim_d )                   | No      | Combination     | No         | -          | The optimizer to use for the discriminator model.                                                                                                       |
| - [losses](#train_anyOf_i1_losses )                     | No      | Combination     | No         | -          | The list of loss functions to optimize.                                                                                                                 |
| - [pixel_opt](#train_anyOf_i1_pixel_opt )               | No      | Combination     | No         | -          | -                                                                                                                                                       |
| - [mssim_opt](#train_anyOf_i1_mssim_opt )               | No      | Combination     | No         | -          | -                                                                                                                                                       |
| - [ms_ssim_l1_opt](#train_anyOf_i1_ms_ssim_l1_opt )     | No      | Combination     | No         | -          | -                                                                                                                                                       |
| - [perceptual_opt](#train_anyOf_i1_perceptual_opt )     | No      | Combination     | No         | -          | -                                                                                                                                                       |
| - [contextual_opt](#train_anyOf_i1_contextual_opt )     | No      | Combination     | No         | -          | -                                                                                                                                                       |
| - [dists_opt](#train_anyOf_i1_dists_opt )               | No      | Combination     | No         | -          | -                                                                                                                                                       |
| - [hr_inversion_opt](#train_anyOf_i1_hr_inversion_opt ) | No      | Combination     | No         | -          | -                                                                                                                                                       |
| - [dinov2_opt](#train_anyOf_i1_dinov2_opt )             | No      | Combination     | No         | -          | -                                                                                                                                                       |
| - [topiq_opt](#train_anyOf_i1_topiq_opt )               | No      | Combination     | No         | -          | -                                                                                                                                                       |
| - [pd_opt](#train_anyOf_i1_pd_opt )                     | No      | Combination     | No         | -          | -                                                                                                                                                       |
| - [fd_opt](#train_anyOf_i1_fd_opt )                     | No      | Combination     | No         | -          | -                                                                                                                                                       |
| - [ldl_opt](#train_anyOf_i1_ldl_opt )                   | No      | Combination     | No         | -          | -                                                                                                                                                       |
| - [hsluv_opt](#train_anyOf_i1_hsluv_opt )               | No      | Combination     | No         | -          | -                                                                                                                                                       |
| - [gan_opt](#train_anyOf_i1_gan_opt )                   | No      | Combination     | No         | -          | -                                                                                                                                                       |
| - [color_opt](#train_anyOf_i1_color_opt )               | No      | Combination     | No         | -          | -                                                                                                                                                       |
| - [luma_opt](#train_anyOf_i1_luma_opt )                 | No      | Combination     | No         | -          | -                                                                                                                                                       |
| - [avg_opt](#train_anyOf_i1_avg_opt )                   | No      | Combination     | No         | -          | -                                                                                                                                                       |
| - [bicubic_opt](#train_anyOf_i1_bicubic_opt )           | No      | Combination     | No         | -          | -                                                                                                                                                       |
| - [use_moa](#train_anyOf_i1_use_moa )                   | No      | boolean         | No         | -          | Whether to enable mixture of augmentations, which augments the dataset on the fly to create more variety and help the model generalize.                 |
| - [moa_augs](#train_anyOf_i1_moa_augs )                 | No      | array of string | No         | -          | The list of augmentations to choose from, only one is selected per iteration.                                                                           |
| - [moa_probs](#train_anyOf_i1_moa_probs )               | No      | array of number | No         | -          | The probability each augmentation in moa_augs will be applied. Total should add up to 1.                                                                |
| - [moa_debug](#train_anyOf_i1_moa_debug )               | No      | boolean         | No         | -          | Save images before and after augment to debug/moa folder inside of the root training directory.                                                         |
| - [moa_debug_limit](#train_anyOf_i1_moa_debug_limit )   | No      | integer         | No         | -          | The max number of iterations to save augmentation images for.                                                                                           |

#### <a name="train_anyOf_i1_total_iter"></a>57.2.1. Property `ReduxOptions > train > anyOf > TrainOptions > total_iter`

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | Yes       |

**Description:** The total number of iterations to train.

#### <a name="train_anyOf_i1_optim_g"></a>57.2.2. Property `ReduxOptions > train > anyOf > TrainOptions > optim_g`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `object`         |
| **Required**              | Yes              |
| **Additional properties** | Any type allowed |

**Description:** The optimizer to use for the generator model.

#### <a name="train_anyOf_i1_ema_decay"></a>57.2.3. Property `ReduxOptions > train > anyOf > TrainOptions > ema_decay`

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |
| **Default**  | `0`      |

**Description:** The decay factor to use for EMA (exponential moving average). Set to 0 to disable EMA.

#### <a name="train_anyOf_i1_grad_clip"></a>57.2.4. Property `ReduxOptions > train > anyOf > TrainOptions > grad_clip`

|              |           |
| ------------ | --------- |
| **Type**     | `boolean` |
| **Required** | No        |
| **Default**  | `false`   |

**Description:** Whether or not to enable gradient clipping, which can improve stability when using higher learning rates, but can also cause issues in some situations.

#### <a name="train_anyOf_i1_warmup_iter"></a>57.2.5. Property `ReduxOptions > train > anyOf > TrainOptions > warmup_iter`

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | No        |
| **Default**  | `-1`      |

**Description:** Gradually ramp up learning rates until this iteration, to stabilize early training. Use -1 to disable.

#### <a name="train_anyOf_i1_scheduler"></a>57.2.6. Property `ReduxOptions > train > anyOf > TrainOptions > scheduler`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `combining`      |
| **Required**              | No               |
| **Additional properties** | Any type allowed |
| **Default**               | `null`           |

**Description:** Options for the optimizer scheduler. If there are multiple optimizers, both will use the same scheduler options.

| Any of(Option)                                         |
| ------------------------------------------------------ |
| [item 0](#train_anyOf_i1_scheduler_anyOf_i0)           |
| [SchedulerOptions](#train_anyOf_i1_scheduler_anyOf_i1) |

##### <a name="train_anyOf_i1_scheduler_anyOf_i0"></a>57.2.6.1. Property `ReduxOptions > train > anyOf > TrainOptions > scheduler > anyOf > item 0`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

##### <a name="train_anyOf_i1_scheduler_anyOf_i1"></a>57.2.6.2. Property `ReduxOptions > train > anyOf > TrainOptions > scheduler > anyOf > SchedulerOptions`

**Title:** SchedulerOptions

|                           |                          |
| ------------------------- | ------------------------ |
| **Type**                  | `object`                 |
| **Required**              | No                       |
| **Additional properties** | Not allowed              |
| **Defined in**            | #/$defs/SchedulerOptions |

| Property                                                       | Pattern | Type             | Deprecated | Definition | Title/Description                                                                                                                      |
| -------------------------------------------------------------- | ------- | ---------------- | ---------- | ---------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| + [type](#train_anyOf_i1_scheduler_anyOf_i1_type )             | No      | string           | No         | -          | -                                                                                                                                      |
| + [milestones](#train_anyOf_i1_scheduler_anyOf_i1_milestones ) | No      | array of integer | No         | -          | List of milestones, iterations where the learning rate is reduced.                                                                     |
| + [gamma](#train_anyOf_i1_scheduler_anyOf_i1_gamma )           | No      | number           | No         | -          | At each milestone, the learning rate is multiplied by this number, so a gamma of 0.5 cuts the learning rate in half at each milestone. |

###### <a name="train_anyOf_i1_scheduler_anyOf_i1_type"></a>57.2.6.2.1. Property `ReduxOptions > train > anyOf > TrainOptions > scheduler > anyOf > SchedulerOptions > type`

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | Yes      |

###### <a name="train_anyOf_i1_scheduler_anyOf_i1_milestones"></a>57.2.6.2.2. Property `ReduxOptions > train > anyOf > TrainOptions > scheduler > anyOf > SchedulerOptions > milestones`

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

| Each item of this array must be                                         | Description |
| ----------------------------------------------------------------------- | ----------- |
| [milestones items](#train_anyOf_i1_scheduler_anyOf_i1_milestones_items) | -           |

###### <a name="train_anyOf_i1_scheduler_anyOf_i1_milestones_items"></a>57.2.6.2.2.1. ReduxOptions > train > anyOf > TrainOptions > scheduler > anyOf > SchedulerOptions > milestones > milestones items

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | No        |

###### <a name="train_anyOf_i1_scheduler_anyOf_i1_gamma"></a>57.2.6.2.3. Property `ReduxOptions > train > anyOf > TrainOptions > scheduler > anyOf > SchedulerOptions > gamma`

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | Yes      |

**Description:** At each milestone, the learning rate is multiplied by this number, so a gamma of 0.5 cuts the learning rate in half at each milestone.

#### <a name="train_anyOf_i1_optim_d"></a>57.2.7. Property `ReduxOptions > train > anyOf > TrainOptions > optim_d`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `combining`      |
| **Required**              | No               |
| **Additional properties** | Any type allowed |
| **Default**               | `null`           |

**Description:** The optimizer to use for the discriminator model.

| Any of(Option)                             |
| ------------------------------------------ |
| [item 0](#train_anyOf_i1_optim_d_anyOf_i0) |
| [item 1](#train_anyOf_i1_optim_d_anyOf_i1) |

##### <a name="train_anyOf_i1_optim_d_anyOf_i0"></a>57.2.7.1. Property `ReduxOptions > train > anyOf > TrainOptions > optim_d > anyOf > item 0`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `object`         |
| **Required**              | No               |
| **Additional properties** | Any type allowed |

##### <a name="train_anyOf_i1_optim_d_anyOf_i1"></a>57.2.7.2. Property `ReduxOptions > train > anyOf > TrainOptions > optim_d > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

#### <a name="train_anyOf_i1_losses"></a>57.2.8. Property `ReduxOptions > train > anyOf > TrainOptions > losses`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `combining`      |
| **Required**              | No               |
| **Additional properties** | Any type allowed |
| **Default**               | `null`           |

**Description:** The list of loss functions to optimize.

| Any of(Option)                            |
| ----------------------------------------- |
| [item 0](#train_anyOf_i1_losses_anyOf_i0) |
| [item 1](#train_anyOf_i1_losses_anyOf_i1) |

##### <a name="train_anyOf_i1_losses_anyOf_i0"></a>57.2.8.1. Property `ReduxOptions > train > anyOf > TrainOptions > losses > anyOf > item 0`

|              |                   |
| ------------ | ----------------- |
| **Type**     | `array of object` |
| **Required** | No                |

|                      | Array restrictions |
| -------------------- | ------------------ |
| **Min items**        | N/A                |
| **Max items**        | N/A                |
| **Items unicity**    | False              |
| **Additional items** | False              |
| **Tuple validation** | See below          |

| Each item of this array must be                       | Description |
| ----------------------------------------------------- | ----------- |
| [item 0 items](#train_anyOf_i1_losses_anyOf_i0_items) | -           |

###### <a name="train_anyOf_i1_losses_anyOf_i0_items"></a>57.2.8.1.1. ReduxOptions > train > anyOf > TrainOptions > losses > anyOf > item 0 > item 0 items

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `object`         |
| **Required**              | No               |
| **Additional properties** | Any type allowed |

##### <a name="train_anyOf_i1_losses_anyOf_i1"></a>57.2.8.2. Property `ReduxOptions > train > anyOf > TrainOptions > losses > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

#### <a name="train_anyOf_i1_pixel_opt"></a>57.2.9. Property `ReduxOptions > train > anyOf > TrainOptions > pixel_opt`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `combining`      |
| **Required**              | No               |
| **Additional properties** | Any type allowed |
| **Default**               | `null`           |

| Any of(Option)                               |
| -------------------------------------------- |
| [item 0](#train_anyOf_i1_pixel_opt_anyOf_i0) |
| [item 1](#train_anyOf_i1_pixel_opt_anyOf_i1) |

##### <a name="train_anyOf_i1_pixel_opt_anyOf_i0"></a>57.2.9.1. Property `ReduxOptions > train > anyOf > TrainOptions > pixel_opt > anyOf > item 0`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `object`         |
| **Required**              | No               |
| **Additional properties** | Any type allowed |

##### <a name="train_anyOf_i1_pixel_opt_anyOf_i1"></a>57.2.9.2. Property `ReduxOptions > train > anyOf > TrainOptions > pixel_opt > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

#### <a name="train_anyOf_i1_mssim_opt"></a>57.2.10. Property `ReduxOptions > train > anyOf > TrainOptions > mssim_opt`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `combining`      |
| **Required**              | No               |
| **Additional properties** | Any type allowed |
| **Default**               | `null`           |

| Any of(Option)                               |
| -------------------------------------------- |
| [item 0](#train_anyOf_i1_mssim_opt_anyOf_i0) |
| [item 1](#train_anyOf_i1_mssim_opt_anyOf_i1) |

##### <a name="train_anyOf_i1_mssim_opt_anyOf_i0"></a>57.2.10.1. Property `ReduxOptions > train > anyOf > TrainOptions > mssim_opt > anyOf > item 0`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `object`         |
| **Required**              | No               |
| **Additional properties** | Any type allowed |

##### <a name="train_anyOf_i1_mssim_opt_anyOf_i1"></a>57.2.10.2. Property `ReduxOptions > train > anyOf > TrainOptions > mssim_opt > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

#### <a name="train_anyOf_i1_ms_ssim_l1_opt"></a>57.2.11. Property `ReduxOptions > train > anyOf > TrainOptions > ms_ssim_l1_opt`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `combining`      |
| **Required**              | No               |
| **Additional properties** | Any type allowed |
| **Default**               | `null`           |

| Any of(Option)                                    |
| ------------------------------------------------- |
| [item 0](#train_anyOf_i1_ms_ssim_l1_opt_anyOf_i0) |
| [item 1](#train_anyOf_i1_ms_ssim_l1_opt_anyOf_i1) |

##### <a name="train_anyOf_i1_ms_ssim_l1_opt_anyOf_i0"></a>57.2.11.1. Property `ReduxOptions > train > anyOf > TrainOptions > ms_ssim_l1_opt > anyOf > item 0`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `object`         |
| **Required**              | No               |
| **Additional properties** | Any type allowed |

##### <a name="train_anyOf_i1_ms_ssim_l1_opt_anyOf_i1"></a>57.2.11.2. Property `ReduxOptions > train > anyOf > TrainOptions > ms_ssim_l1_opt > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

#### <a name="train_anyOf_i1_perceptual_opt"></a>57.2.12. Property `ReduxOptions > train > anyOf > TrainOptions > perceptual_opt`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `combining`      |
| **Required**              | No               |
| **Additional properties** | Any type allowed |
| **Default**               | `null`           |

| Any of(Option)                                    |
| ------------------------------------------------- |
| [item 0](#train_anyOf_i1_perceptual_opt_anyOf_i0) |
| [item 1](#train_anyOf_i1_perceptual_opt_anyOf_i1) |

##### <a name="train_anyOf_i1_perceptual_opt_anyOf_i0"></a>57.2.12.1. Property `ReduxOptions > train > anyOf > TrainOptions > perceptual_opt > anyOf > item 0`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `object`         |
| **Required**              | No               |
| **Additional properties** | Any type allowed |

##### <a name="train_anyOf_i1_perceptual_opt_anyOf_i1"></a>57.2.12.2. Property `ReduxOptions > train > anyOf > TrainOptions > perceptual_opt > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

#### <a name="train_anyOf_i1_contextual_opt"></a>57.2.13. Property `ReduxOptions > train > anyOf > TrainOptions > contextual_opt`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `combining`      |
| **Required**              | No               |
| **Additional properties** | Any type allowed |
| **Default**               | `null`           |

| Any of(Option)                                    |
| ------------------------------------------------- |
| [item 0](#train_anyOf_i1_contextual_opt_anyOf_i0) |
| [item 1](#train_anyOf_i1_contextual_opt_anyOf_i1) |

##### <a name="train_anyOf_i1_contextual_opt_anyOf_i0"></a>57.2.13.1. Property `ReduxOptions > train > anyOf > TrainOptions > contextual_opt > anyOf > item 0`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `object`         |
| **Required**              | No               |
| **Additional properties** | Any type allowed |

##### <a name="train_anyOf_i1_contextual_opt_anyOf_i1"></a>57.2.13.2. Property `ReduxOptions > train > anyOf > TrainOptions > contextual_opt > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

#### <a name="train_anyOf_i1_dists_opt"></a>57.2.14. Property `ReduxOptions > train > anyOf > TrainOptions > dists_opt`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `combining`      |
| **Required**              | No               |
| **Additional properties** | Any type allowed |
| **Default**               | `null`           |

| Any of(Option)                               |
| -------------------------------------------- |
| [item 0](#train_anyOf_i1_dists_opt_anyOf_i0) |
| [item 1](#train_anyOf_i1_dists_opt_anyOf_i1) |

##### <a name="train_anyOf_i1_dists_opt_anyOf_i0"></a>57.2.14.1. Property `ReduxOptions > train > anyOf > TrainOptions > dists_opt > anyOf > item 0`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `object`         |
| **Required**              | No               |
| **Additional properties** | Any type allowed |

##### <a name="train_anyOf_i1_dists_opt_anyOf_i1"></a>57.2.14.2. Property `ReduxOptions > train > anyOf > TrainOptions > dists_opt > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

#### <a name="train_anyOf_i1_hr_inversion_opt"></a>57.2.15. Property `ReduxOptions > train > anyOf > TrainOptions > hr_inversion_opt`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `combining`      |
| **Required**              | No               |
| **Additional properties** | Any type allowed |
| **Default**               | `null`           |

| Any of(Option)                                      |
| --------------------------------------------------- |
| [item 0](#train_anyOf_i1_hr_inversion_opt_anyOf_i0) |
| [item 1](#train_anyOf_i1_hr_inversion_opt_anyOf_i1) |

##### <a name="train_anyOf_i1_hr_inversion_opt_anyOf_i0"></a>57.2.15.1. Property `ReduxOptions > train > anyOf > TrainOptions > hr_inversion_opt > anyOf > item 0`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `object`         |
| **Required**              | No               |
| **Additional properties** | Any type allowed |

##### <a name="train_anyOf_i1_hr_inversion_opt_anyOf_i1"></a>57.2.15.2. Property `ReduxOptions > train > anyOf > TrainOptions > hr_inversion_opt > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

#### <a name="train_anyOf_i1_dinov2_opt"></a>57.2.16. Property `ReduxOptions > train > anyOf > TrainOptions > dinov2_opt`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `combining`      |
| **Required**              | No               |
| **Additional properties** | Any type allowed |
| **Default**               | `null`           |

| Any of(Option)                                |
| --------------------------------------------- |
| [item 0](#train_anyOf_i1_dinov2_opt_anyOf_i0) |
| [item 1](#train_anyOf_i1_dinov2_opt_anyOf_i1) |

##### <a name="train_anyOf_i1_dinov2_opt_anyOf_i0"></a>57.2.16.1. Property `ReduxOptions > train > anyOf > TrainOptions > dinov2_opt > anyOf > item 0`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `object`         |
| **Required**              | No               |
| **Additional properties** | Any type allowed |

##### <a name="train_anyOf_i1_dinov2_opt_anyOf_i1"></a>57.2.16.2. Property `ReduxOptions > train > anyOf > TrainOptions > dinov2_opt > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

#### <a name="train_anyOf_i1_topiq_opt"></a>57.2.17. Property `ReduxOptions > train > anyOf > TrainOptions > topiq_opt`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `combining`      |
| **Required**              | No               |
| **Additional properties** | Any type allowed |
| **Default**               | `null`           |

| Any of(Option)                               |
| -------------------------------------------- |
| [item 0](#train_anyOf_i1_topiq_opt_anyOf_i0) |
| [item 1](#train_anyOf_i1_topiq_opt_anyOf_i1) |

##### <a name="train_anyOf_i1_topiq_opt_anyOf_i0"></a>57.2.17.1. Property `ReduxOptions > train > anyOf > TrainOptions > topiq_opt > anyOf > item 0`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `object`         |
| **Required**              | No               |
| **Additional properties** | Any type allowed |

##### <a name="train_anyOf_i1_topiq_opt_anyOf_i1"></a>57.2.17.2. Property `ReduxOptions > train > anyOf > TrainOptions > topiq_opt > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

#### <a name="train_anyOf_i1_pd_opt"></a>57.2.18. Property `ReduxOptions > train > anyOf > TrainOptions > pd_opt`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `combining`      |
| **Required**              | No               |
| **Additional properties** | Any type allowed |
| **Default**               | `null`           |

| Any of(Option)                            |
| ----------------------------------------- |
| [item 0](#train_anyOf_i1_pd_opt_anyOf_i0) |
| [item 1](#train_anyOf_i1_pd_opt_anyOf_i1) |

##### <a name="train_anyOf_i1_pd_opt_anyOf_i0"></a>57.2.18.1. Property `ReduxOptions > train > anyOf > TrainOptions > pd_opt > anyOf > item 0`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `object`         |
| **Required**              | No               |
| **Additional properties** | Any type allowed |

##### <a name="train_anyOf_i1_pd_opt_anyOf_i1"></a>57.2.18.2. Property `ReduxOptions > train > anyOf > TrainOptions > pd_opt > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

#### <a name="train_anyOf_i1_fd_opt"></a>57.2.19. Property `ReduxOptions > train > anyOf > TrainOptions > fd_opt`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `combining`      |
| **Required**              | No               |
| **Additional properties** | Any type allowed |
| **Default**               | `null`           |

| Any of(Option)                            |
| ----------------------------------------- |
| [item 0](#train_anyOf_i1_fd_opt_anyOf_i0) |
| [item 1](#train_anyOf_i1_fd_opt_anyOf_i1) |

##### <a name="train_anyOf_i1_fd_opt_anyOf_i0"></a>57.2.19.1. Property `ReduxOptions > train > anyOf > TrainOptions > fd_opt > anyOf > item 0`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `object`         |
| **Required**              | No               |
| **Additional properties** | Any type allowed |

##### <a name="train_anyOf_i1_fd_opt_anyOf_i1"></a>57.2.19.2. Property `ReduxOptions > train > anyOf > TrainOptions > fd_opt > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

#### <a name="train_anyOf_i1_ldl_opt"></a>57.2.20. Property `ReduxOptions > train > anyOf > TrainOptions > ldl_opt`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `combining`      |
| **Required**              | No               |
| **Additional properties** | Any type allowed |
| **Default**               | `null`           |

| Any of(Option)                             |
| ------------------------------------------ |
| [item 0](#train_anyOf_i1_ldl_opt_anyOf_i0) |
| [item 1](#train_anyOf_i1_ldl_opt_anyOf_i1) |

##### <a name="train_anyOf_i1_ldl_opt_anyOf_i0"></a>57.2.20.1. Property `ReduxOptions > train > anyOf > TrainOptions > ldl_opt > anyOf > item 0`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `object`         |
| **Required**              | No               |
| **Additional properties** | Any type allowed |

##### <a name="train_anyOf_i1_ldl_opt_anyOf_i1"></a>57.2.20.2. Property `ReduxOptions > train > anyOf > TrainOptions > ldl_opt > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

#### <a name="train_anyOf_i1_hsluv_opt"></a>57.2.21. Property `ReduxOptions > train > anyOf > TrainOptions > hsluv_opt`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `combining`      |
| **Required**              | No               |
| **Additional properties** | Any type allowed |
| **Default**               | `null`           |

| Any of(Option)                               |
| -------------------------------------------- |
| [item 0](#train_anyOf_i1_hsluv_opt_anyOf_i0) |
| [item 1](#train_anyOf_i1_hsluv_opt_anyOf_i1) |

##### <a name="train_anyOf_i1_hsluv_opt_anyOf_i0"></a>57.2.21.1. Property `ReduxOptions > train > anyOf > TrainOptions > hsluv_opt > anyOf > item 0`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `object`         |
| **Required**              | No               |
| **Additional properties** | Any type allowed |

##### <a name="train_anyOf_i1_hsluv_opt_anyOf_i1"></a>57.2.21.2. Property `ReduxOptions > train > anyOf > TrainOptions > hsluv_opt > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

#### <a name="train_anyOf_i1_gan_opt"></a>57.2.22. Property `ReduxOptions > train > anyOf > TrainOptions > gan_opt`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `combining`      |
| **Required**              | No               |
| **Additional properties** | Any type allowed |
| **Default**               | `null`           |

| Any of(Option)                             |
| ------------------------------------------ |
| [item 0](#train_anyOf_i1_gan_opt_anyOf_i0) |
| [item 1](#train_anyOf_i1_gan_opt_anyOf_i1) |

##### <a name="train_anyOf_i1_gan_opt_anyOf_i0"></a>57.2.22.1. Property `ReduxOptions > train > anyOf > TrainOptions > gan_opt > anyOf > item 0`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `object`         |
| **Required**              | No               |
| **Additional properties** | Any type allowed |

##### <a name="train_anyOf_i1_gan_opt_anyOf_i1"></a>57.2.22.2. Property `ReduxOptions > train > anyOf > TrainOptions > gan_opt > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

#### <a name="train_anyOf_i1_color_opt"></a>57.2.23. Property `ReduxOptions > train > anyOf > TrainOptions > color_opt`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `combining`      |
| **Required**              | No               |
| **Additional properties** | Any type allowed |
| **Default**               | `null`           |

| Any of(Option)                               |
| -------------------------------------------- |
| [item 0](#train_anyOf_i1_color_opt_anyOf_i0) |
| [item 1](#train_anyOf_i1_color_opt_anyOf_i1) |

##### <a name="train_anyOf_i1_color_opt_anyOf_i0"></a>57.2.23.1. Property `ReduxOptions > train > anyOf > TrainOptions > color_opt > anyOf > item 0`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `object`         |
| **Required**              | No               |
| **Additional properties** | Any type allowed |

##### <a name="train_anyOf_i1_color_opt_anyOf_i1"></a>57.2.23.2. Property `ReduxOptions > train > anyOf > TrainOptions > color_opt > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

#### <a name="train_anyOf_i1_luma_opt"></a>57.2.24. Property `ReduxOptions > train > anyOf > TrainOptions > luma_opt`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `combining`      |
| **Required**              | No               |
| **Additional properties** | Any type allowed |
| **Default**               | `null`           |

| Any of(Option)                              |
| ------------------------------------------- |
| [item 0](#train_anyOf_i1_luma_opt_anyOf_i0) |
| [item 1](#train_anyOf_i1_luma_opt_anyOf_i1) |

##### <a name="train_anyOf_i1_luma_opt_anyOf_i0"></a>57.2.24.1. Property `ReduxOptions > train > anyOf > TrainOptions > luma_opt > anyOf > item 0`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `object`         |
| **Required**              | No               |
| **Additional properties** | Any type allowed |

##### <a name="train_anyOf_i1_luma_opt_anyOf_i1"></a>57.2.24.2. Property `ReduxOptions > train > anyOf > TrainOptions > luma_opt > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

#### <a name="train_anyOf_i1_avg_opt"></a>57.2.25. Property `ReduxOptions > train > anyOf > TrainOptions > avg_opt`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `combining`      |
| **Required**              | No               |
| **Additional properties** | Any type allowed |
| **Default**               | `null`           |

| Any of(Option)                             |
| ------------------------------------------ |
| [item 0](#train_anyOf_i1_avg_opt_anyOf_i0) |
| [item 1](#train_anyOf_i1_avg_opt_anyOf_i1) |

##### <a name="train_anyOf_i1_avg_opt_anyOf_i0"></a>57.2.25.1. Property `ReduxOptions > train > anyOf > TrainOptions > avg_opt > anyOf > item 0`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `object`         |
| **Required**              | No               |
| **Additional properties** | Any type allowed |

##### <a name="train_anyOf_i1_avg_opt_anyOf_i1"></a>57.2.25.2. Property `ReduxOptions > train > anyOf > TrainOptions > avg_opt > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

#### <a name="train_anyOf_i1_bicubic_opt"></a>57.2.26. Property `ReduxOptions > train > anyOf > TrainOptions > bicubic_opt`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `combining`      |
| **Required**              | No               |
| **Additional properties** | Any type allowed |
| **Default**               | `null`           |

| Any of(Option)                                 |
| ---------------------------------------------- |
| [item 0](#train_anyOf_i1_bicubic_opt_anyOf_i0) |
| [item 1](#train_anyOf_i1_bicubic_opt_anyOf_i1) |

##### <a name="train_anyOf_i1_bicubic_opt_anyOf_i0"></a>57.2.26.1. Property `ReduxOptions > train > anyOf > TrainOptions > bicubic_opt > anyOf > item 0`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `object`         |
| **Required**              | No               |
| **Additional properties** | Any type allowed |

##### <a name="train_anyOf_i1_bicubic_opt_anyOf_i1"></a>57.2.26.2. Property `ReduxOptions > train > anyOf > TrainOptions > bicubic_opt > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

#### <a name="train_anyOf_i1_use_moa"></a>57.2.27. Property `ReduxOptions > train > anyOf > TrainOptions > use_moa`

|              |           |
| ------------ | --------- |
| **Type**     | `boolean` |
| **Required** | No        |
| **Default**  | `false`   |

**Description:** Whether to enable mixture of augmentations, which augments the dataset on the fly to create more variety and help the model generalize.

#### <a name="train_anyOf_i1_moa_augs"></a>57.2.28. Property `ReduxOptions > train > anyOf > TrainOptions > moa_augs`

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

| Each item of this array must be                  | Description |
| ------------------------------------------------ | ----------- |
| [moa_augs items](#train_anyOf_i1_moa_augs_items) | -           |

##### <a name="train_anyOf_i1_moa_augs_items"></a>57.2.28.1. ReduxOptions > train > anyOf > TrainOptions > moa_augs > moa_augs items

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | No       |

#### <a name="train_anyOf_i1_moa_probs"></a>57.2.29. Property `ReduxOptions > train > anyOf > TrainOptions > moa_probs`

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

| Each item of this array must be                    | Description |
| -------------------------------------------------- | ----------- |
| [moa_probs items](#train_anyOf_i1_moa_probs_items) | -           |

##### <a name="train_anyOf_i1_moa_probs_items"></a>57.2.29.1. ReduxOptions > train > anyOf > TrainOptions > moa_probs > moa_probs items

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |

#### <a name="train_anyOf_i1_moa_debug"></a>57.2.30. Property `ReduxOptions > train > anyOf > TrainOptions > moa_debug`

|              |           |
| ------------ | --------- |
| **Type**     | `boolean` |
| **Required** | No        |
| **Default**  | `false`   |

**Description:** Save images before and after augment to debug/moa folder inside of the root training directory.

#### <a name="train_anyOf_i1_moa_debug_limit"></a>57.2.31. Property `ReduxOptions > train > anyOf > TrainOptions > moa_debug_limit`

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | No        |
| **Default**  | `100`     |

**Description:** The max number of iterations to save augmentation images for.

## <a name="val"></a>58. Property `ReduxOptions > val`

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

### <a name="val_anyOf_i0"></a>58.1. Property `ReduxOptions > val > anyOf > item 0`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

### <a name="val_anyOf_i1"></a>58.2. Property `ReduxOptions > val > anyOf > ValOptions`

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

#### <a name="val_anyOf_i1_val_enabled"></a>58.2.1. Property `ReduxOptions > val > anyOf > ValOptions > val_enabled`

|              |           |
| ------------ | --------- |
| **Type**     | `boolean` |
| **Required** | Yes       |

**Description:** Whether to enable validations. If disabled, all validation settings below are ignored.

#### <a name="val_anyOf_i1_save_img"></a>58.2.2. Property `ReduxOptions > val > anyOf > ValOptions > save_img`

|              |           |
| ------------ | --------- |
| **Type**     | `boolean` |
| **Required** | Yes       |

**Description:** Whether to save the validation images during validation, in the experiments/<name>/visualization folder.

#### <a name="val_anyOf_i1_val_freq"></a>58.2.3. Property `ReduxOptions > val > anyOf > ValOptions > val_freq`

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

##### <a name="val_anyOf_i1_val_freq_anyOf_i0"></a>58.2.3.1. Property `ReduxOptions > val > anyOf > ValOptions > val_freq > anyOf > item 0`

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | No        |

##### <a name="val_anyOf_i1_val_freq_anyOf_i1"></a>58.2.3.2. Property `ReduxOptions > val > anyOf > ValOptions > val_freq > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

#### <a name="val_anyOf_i1_suffix"></a>58.2.4. Property `ReduxOptions > val > anyOf > ValOptions > suffix`

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

##### <a name="val_anyOf_i1_suffix_anyOf_i0"></a>58.2.4.1. Property `ReduxOptions > val > anyOf > ValOptions > suffix > anyOf > item 0`

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | No       |

##### <a name="val_anyOf_i1_suffix_anyOf_i1"></a>58.2.4.2. Property `ReduxOptions > val > anyOf > ValOptions > suffix > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

#### <a name="val_anyOf_i1_metrics_enabled"></a>58.2.5. Property `ReduxOptions > val > anyOf > ValOptions > metrics_enabled`

|              |           |
| ------------ | --------- |
| **Type**     | `boolean` |
| **Required** | No        |
| **Default**  | `false`   |

**Description:** Whether to run metrics calculations during validation.

#### <a name="val_anyOf_i1_metrics"></a>58.2.6. Property `ReduxOptions > val > anyOf > ValOptions > metrics`

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

##### <a name="val_anyOf_i1_metrics_anyOf_i0"></a>58.2.6.1. Property `ReduxOptions > val > anyOf > ValOptions > metrics > anyOf > item 0`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `object`         |
| **Required**              | No               |
| **Additional properties** | Any type allowed |

##### <a name="val_anyOf_i1_metrics_anyOf_i1"></a>58.2.6.2. Property `ReduxOptions > val > anyOf > ValOptions > metrics > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

#### <a name="val_anyOf_i1_pbar"></a>58.2.7. Property `ReduxOptions > val > anyOf > ValOptions > pbar`

|              |           |
| ------------ | --------- |
| **Type**     | `boolean` |
| **Required** | No        |
| **Default**  | `true`    |

## <a name="logger"></a>59. Property `ReduxOptions > logger`

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

### <a name="logger_anyOf_i0"></a>59.1. Property `ReduxOptions > logger > anyOf > item 0`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

### <a name="logger_anyOf_i1"></a>59.2. Property `ReduxOptions > logger > anyOf > LogOptions`

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

#### <a name="logger_anyOf_i1_print_freq"></a>59.2.1. Property `ReduxOptions > logger > anyOf > LogOptions > print_freq`

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | Yes       |

**Description:** How often to print logs to the console, in iterations.

#### <a name="logger_anyOf_i1_save_checkpoint_freq"></a>59.2.2. Property `ReduxOptions > logger > anyOf > LogOptions > save_checkpoint_freq`

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | Yes       |

**Description:** How often to save model checkpoints and training states, in iterations.

#### <a name="logger_anyOf_i1_use_tb_logger"></a>59.2.3. Property `ReduxOptions > logger > anyOf > LogOptions > use_tb_logger`

|              |           |
| ------------ | --------- |
| **Type**     | `boolean` |
| **Required** | Yes       |

**Description:** Whether or not to enable TensorBoard logging.

#### <a name="logger_anyOf_i1_save_checkpoint_format"></a>59.2.4. Property `ReduxOptions > logger > anyOf > LogOptions > save_checkpoint_format`

|              |                    |
| ------------ | ------------------ |
| **Type**     | `enum (of string)` |
| **Required** | No                 |
| **Default**  | `"safetensors"`    |

**Description:** Format to save model checkpoints.

Must be one of:
* "pth"
* "safetensors"

#### <a name="logger_anyOf_i1_wandb"></a>59.2.5. Property `ReduxOptions > logger > anyOf > LogOptions > wandb`

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

##### <a name="logger_anyOf_i1_wandb_anyOf_i0"></a>59.2.5.1. Property `ReduxOptions > logger > anyOf > LogOptions > wandb > anyOf > item 0`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

##### <a name="logger_anyOf_i1_wandb_anyOf_i1"></a>59.2.5.2. Property `ReduxOptions > logger > anyOf > LogOptions > wandb > anyOf > WandbOptions`

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

###### <a name="logger_anyOf_i1_wandb_anyOf_i1_resume_id"></a>59.2.5.2.1. Property `ReduxOptions > logger > anyOf > LogOptions > wandb > anyOf > WandbOptions > resume_id`

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

###### <a name="logger_anyOf_i1_wandb_anyOf_i1_resume_id_anyOf_i0"></a>59.2.5.2.1.1. Property `ReduxOptions > logger > anyOf > LogOptions > wandb > anyOf > WandbOptions > resume_id > anyOf > item 0`

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | No       |

###### <a name="logger_anyOf_i1_wandb_anyOf_i1_resume_id_anyOf_i1"></a>59.2.5.2.1.2. Property `ReduxOptions > logger > anyOf > LogOptions > wandb > anyOf > WandbOptions > resume_id > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

###### <a name="logger_anyOf_i1_wandb_anyOf_i1_project"></a>59.2.5.2.2. Property `ReduxOptions > logger > anyOf > LogOptions > wandb > anyOf > WandbOptions > project`

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

###### <a name="logger_anyOf_i1_wandb_anyOf_i1_project_anyOf_i0"></a>59.2.5.2.2.1. Property `ReduxOptions > logger > anyOf > LogOptions > wandb > anyOf > WandbOptions > project > anyOf > item 0`

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | No       |

###### <a name="logger_anyOf_i1_wandb_anyOf_i1_project_anyOf_i1"></a>59.2.5.2.2.2. Property `ReduxOptions > logger > anyOf > LogOptions > wandb > anyOf > WandbOptions > project > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

## <a name="dist_params"></a>60. Property `ReduxOptions > dist_params`

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

### <a name="dist_params_anyOf_i0"></a>60.1. Property `ReduxOptions > dist_params > anyOf > item 0`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `object`         |
| **Required**              | No               |
| **Additional properties** | Any type allowed |

### <a name="dist_params_anyOf_i1"></a>60.2. Property `ReduxOptions > dist_params > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

## <a name="onnx"></a>61. Property `ReduxOptions > onnx`

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

### <a name="onnx_anyOf_i0"></a>61.1. Property `ReduxOptions > onnx > anyOf > item 0`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

### <a name="onnx_anyOf_i1"></a>61.2. Property `ReduxOptions > onnx > anyOf > OnnxOptions`

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

#### <a name="onnx_anyOf_i1_dynamo"></a>61.2.1. Property `ReduxOptions > onnx > anyOf > OnnxOptions > dynamo`

|              |           |
| ------------ | --------- |
| **Type**     | `boolean` |
| **Required** | No        |
| **Default**  | `false`   |

#### <a name="onnx_anyOf_i1_opset"></a>61.2.2. Property `ReduxOptions > onnx > anyOf > OnnxOptions > opset`

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | No        |
| **Default**  | `20`      |

#### <a name="onnx_anyOf_i1_use_static_shapes"></a>61.2.3. Property `ReduxOptions > onnx > anyOf > OnnxOptions > use_static_shapes`

|              |           |
| ------------ | --------- |
| **Type**     | `boolean` |
| **Required** | No        |
| **Default**  | `false`   |

#### <a name="onnx_anyOf_i1_shape"></a>61.2.4. Property `ReduxOptions > onnx > anyOf > OnnxOptions > shape`

|              |               |
| ------------ | ------------- |
| **Type**     | `string`      |
| **Required** | No            |
| **Default**  | `"3x256x256"` |

#### <a name="onnx_anyOf_i1_verify"></a>61.2.5. Property `ReduxOptions > onnx > anyOf > OnnxOptions > verify`

|              |           |
| ------------ | --------- |
| **Type**     | `boolean` |
| **Required** | No        |
| **Default**  | `true`    |

#### <a name="onnx_anyOf_i1_fp16"></a>61.2.6. Property `ReduxOptions > onnx > anyOf > OnnxOptions > fp16`

|              |           |
| ------------ | --------- |
| **Type**     | `boolean` |
| **Required** | No        |
| **Default**  | `false`   |

#### <a name="onnx_anyOf_i1_optimize"></a>61.2.7. Property `ReduxOptions > onnx > anyOf > OnnxOptions > optimize`

|              |           |
| ------------ | --------- |
| **Type**     | `boolean` |
| **Required** | No        |
| **Default**  | `true`    |

## <a name="find_unused_parameters"></a>62. Property `ReduxOptions > find_unused_parameters`

|              |           |
| ------------ | --------- |
| **Type**     | `boolean` |
| **Required** | No        |
| **Default**  | `false`   |

----------------------------------------------------------------------------------------------------------------------------
Generated using [json-schema-for-humans](https://github.com/coveooss/json-schema-for-humans) on 2024-12-18 at 19:17:14 -0500
