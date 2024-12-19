# Schema Docs

- [1. Property `ReduxOptions > type`](#type)
- [2. Property `ReduxOptions > name`](#name)
- [3. Property `ReduxOptions > scale`](#scale)
- [4. Property `ReduxOptions > num_gpu`](#num_gpu)
  - [4.1. Property `ReduxOptions > num_gpu > anyOf > item 0`](#num_gpu_anyOf_i0)
  - [4.2. Property `ReduxOptions > num_gpu > anyOf > item 1`](#num_gpu_anyOf_i1)
- [5. Property `ReduxOptions > path`](#path)
  - [5.1. Property `ReduxOptions > path > experiments_root`](#path_experiments_root)
    - [5.1.1. Property `ReduxOptions > path > experiments_root > anyOf > item 0`](#path_experiments_root_anyOf_i0)
    - [5.1.2. Property `ReduxOptions > path > experiments_root > anyOf > item 1`](#path_experiments_root_anyOf_i1)
  - [5.2. Property `ReduxOptions > path > models`](#path_models)
    - [5.2.1. Property `ReduxOptions > path > models > anyOf > item 0`](#path_models_anyOf_i0)
    - [5.2.2. Property `ReduxOptions > path > models > anyOf > item 1`](#path_models_anyOf_i1)
  - [5.3. Property `ReduxOptions > path > resume_models`](#path_resume_models)
    - [5.3.1. Property `ReduxOptions > path > resume_models > anyOf > item 0`](#path_resume_models_anyOf_i0)
    - [5.3.2. Property `ReduxOptions > path > resume_models > anyOf > item 1`](#path_resume_models_anyOf_i1)
  - [5.4. Property `ReduxOptions > path > training_states`](#path_training_states)
    - [5.4.1. Property `ReduxOptions > path > training_states > anyOf > item 0`](#path_training_states_anyOf_i0)
    - [5.4.2. Property `ReduxOptions > path > training_states > anyOf > item 1`](#path_training_states_anyOf_i1)
  - [5.5. Property `ReduxOptions > path > log`](#path_log)
    - [5.5.1. Property `ReduxOptions > path > log > anyOf > item 0`](#path_log_anyOf_i0)
    - [5.5.2. Property `ReduxOptions > path > log > anyOf > item 1`](#path_log_anyOf_i1)
  - [5.6. Property `ReduxOptions > path > visualization`](#path_visualization)
    - [5.6.1. Property `ReduxOptions > path > visualization > anyOf > item 0`](#path_visualization_anyOf_i0)
    - [5.6.2. Property `ReduxOptions > path > visualization > anyOf > item 1`](#path_visualization_anyOf_i1)
  - [5.7. Property `ReduxOptions > path > results_root`](#path_results_root)
    - [5.7.1. Property `ReduxOptions > path > results_root > anyOf > item 0`](#path_results_root_anyOf_i0)
    - [5.7.2. Property `ReduxOptions > path > results_root > anyOf > item 1`](#path_results_root_anyOf_i1)
  - [5.8. Property `ReduxOptions > path > pretrain_network_g`](#path_pretrain_network_g)
    - [5.8.1. Property `ReduxOptions > path > pretrain_network_g > anyOf > item 0`](#path_pretrain_network_g_anyOf_i0)
    - [5.8.2. Property `ReduxOptions > path > pretrain_network_g > anyOf > item 1`](#path_pretrain_network_g_anyOf_i1)
  - [5.9. Property `ReduxOptions > path > pretrain_network_g_path`](#path_pretrain_network_g_path)
    - [5.9.1. Property `ReduxOptions > path > pretrain_network_g_path > anyOf > item 0`](#path_pretrain_network_g_path_anyOf_i0)
    - [5.9.2. Property `ReduxOptions > path > pretrain_network_g_path > anyOf > item 1`](#path_pretrain_network_g_path_anyOf_i1)
  - [5.10. Property `ReduxOptions > path > param_key_g`](#path_param_key_g)
    - [5.10.1. Property `ReduxOptions > path > param_key_g > anyOf > item 0`](#path_param_key_g_anyOf_i0)
    - [5.10.2. Property `ReduxOptions > path > param_key_g > anyOf > item 1`](#path_param_key_g_anyOf_i1)
  - [5.11. Property `ReduxOptions > path > strict_load_g`](#path_strict_load_g)
  - [5.12. Property `ReduxOptions > path > resume_state`](#path_resume_state)
    - [5.12.1. Property `ReduxOptions > path > resume_state > anyOf > item 0`](#path_resume_state_anyOf_i0)
    - [5.12.2. Property `ReduxOptions > path > resume_state > anyOf > item 1`](#path_resume_state_anyOf_i1)
  - [5.13. Property `ReduxOptions > path > pretrain_network_g_ema`](#path_pretrain_network_g_ema)
    - [5.13.1. Property `ReduxOptions > path > pretrain_network_g_ema > anyOf > item 0`](#path_pretrain_network_g_ema_anyOf_i0)
    - [5.13.2. Property `ReduxOptions > path > pretrain_network_g_ema > anyOf > item 1`](#path_pretrain_network_g_ema_anyOf_i1)
  - [5.14. Property `ReduxOptions > path > pretrain_network_d`](#path_pretrain_network_d)
    - [5.14.1. Property `ReduxOptions > path > pretrain_network_d > anyOf > item 0`](#path_pretrain_network_d_anyOf_i0)
    - [5.14.2. Property `ReduxOptions > path > pretrain_network_d > anyOf > item 1`](#path_pretrain_network_d_anyOf_i1)
  - [5.15. Property `ReduxOptions > path > param_key_d`](#path_param_key_d)
    - [5.15.1. Property `ReduxOptions > path > param_key_d > anyOf > item 0`](#path_param_key_d_anyOf_i0)
    - [5.15.2. Property `ReduxOptions > path > param_key_d > anyOf > item 1`](#path_param_key_d_anyOf_i1)
  - [5.16. Property `ReduxOptions > path > strict_load_d`](#path_strict_load_d)
  - [5.17. Property `ReduxOptions > path > ignore_resume_networks`](#path_ignore_resume_networks)
    - [5.17.1. Property `ReduxOptions > path > ignore_resume_networks > anyOf > item 0`](#path_ignore_resume_networks_anyOf_i0)
      - [5.17.1.1. ReduxOptions > path > ignore_resume_networks > anyOf > item 0 > item 0 items](#path_ignore_resume_networks_anyOf_i0_items)
    - [5.17.2. Property `ReduxOptions > path > ignore_resume_networks > anyOf > item 1`](#path_ignore_resume_networks_anyOf_i1)
- [6. Property `ReduxOptions > network_g`](#network_g)
- [7. Property `ReduxOptions > network_d`](#network_d)
  - [7.1. Property `ReduxOptions > network_d > anyOf > item 0`](#network_d_anyOf_i0)
  - [7.2. Property `ReduxOptions > network_d > anyOf > item 1`](#network_d_anyOf_i1)
- [8. Property `ReduxOptions > manual_seed`](#manual_seed)
  - [8.1. Property `ReduxOptions > manual_seed > anyOf > item 0`](#manual_seed_anyOf_i0)
  - [8.2. Property `ReduxOptions > manual_seed > anyOf > item 1`](#manual_seed_anyOf_i1)
- [9. Property `ReduxOptions > deterministic`](#deterministic)
  - [9.1. Property `ReduxOptions > deterministic > anyOf > item 0`](#deterministic_anyOf_i0)
  - [9.2. Property `ReduxOptions > deterministic > anyOf > item 1`](#deterministic_anyOf_i1)
- [10. Property `ReduxOptions > dist`](#dist)
  - [10.1. Property `ReduxOptions > dist > anyOf > item 0`](#dist_anyOf_i0)
  - [10.2. Property `ReduxOptions > dist > anyOf > item 1`](#dist_anyOf_i1)
- [11. Property `ReduxOptions > launcher`](#launcher)
  - [11.1. Property `ReduxOptions > launcher > anyOf > item 0`](#launcher_anyOf_i0)
  - [11.2. Property `ReduxOptions > launcher > anyOf > item 1`](#launcher_anyOf_i1)
- [12. Property `ReduxOptions > rank`](#rank)
  - [12.1. Property `ReduxOptions > rank > anyOf > item 0`](#rank_anyOf_i0)
  - [12.2. Property `ReduxOptions > rank > anyOf > item 1`](#rank_anyOf_i1)
- [13. Property `ReduxOptions > world_size`](#world_size)
  - [13.1. Property `ReduxOptions > world_size > anyOf > item 0`](#world_size_anyOf_i0)
  - [13.2. Property `ReduxOptions > world_size > anyOf > item 1`](#world_size_anyOf_i1)
- [14. Property `ReduxOptions > auto_resume`](#auto_resume)
  - [14.1. Property `ReduxOptions > auto_resume > anyOf > item 0`](#auto_resume_anyOf_i0)
  - [14.2. Property `ReduxOptions > auto_resume > anyOf > item 1`](#auto_resume_anyOf_i1)
- [15. Property `ReduxOptions > resume`](#resume)
- [16. Property `ReduxOptions > is_train`](#is_train)
  - [16.1. Property `ReduxOptions > is_train > anyOf > item 0`](#is_train_anyOf_i0)
  - [16.2. Property `ReduxOptions > is_train > anyOf > item 1`](#is_train_anyOf_i1)
- [17. Property `ReduxOptions > root_path`](#root_path)
  - [17.1. Property `ReduxOptions > root_path > anyOf > item 0`](#root_path_anyOf_i0)
  - [17.2. Property `ReduxOptions > root_path > anyOf > item 1`](#root_path_anyOf_i1)
- [18. Property `ReduxOptions > use_amp`](#use_amp)
- [19. Property `ReduxOptions > amp_bf16`](#amp_bf16)
- [20. Property `ReduxOptions > use_channels_last`](#use_channels_last)
- [21. Property `ReduxOptions > fast_matmul`](#fast_matmul)
- [22. Property `ReduxOptions > use_compile`](#use_compile)
- [23. Property `ReduxOptions > detect_anomaly`](#detect_anomaly)
- [24. Property `ReduxOptions > high_order_degradation`](#high_order_degradation)
- [25. Property `ReduxOptions > high_order_degradations_debug`](#high_order_degradations_debug)
- [26. Property `ReduxOptions > high_order_degradations_debug_limit`](#high_order_degradations_debug_limit)
- [27. Property `ReduxOptions > dataroot_lq_prob`](#dataroot_lq_prob)
- [28. Property `ReduxOptions > force_high_order_degradation_filename_masks`](#force_high_order_degradation_filename_masks)
  - [28.1. ReduxOptions > force_high_order_degradation_filename_masks > force_high_order_degradation_filename_masks items](#force_high_order_degradation_filename_masks_items)
- [29. Property `ReduxOptions > force_dataroot_lq_filename_masks`](#force_dataroot_lq_filename_masks)
  - [29.1. ReduxOptions > force_dataroot_lq_filename_masks > force_dataroot_lq_filename_masks items](#force_dataroot_lq_filename_masks_items)
- [30. Property `ReduxOptions > lq_usm`](#lq_usm)
- [31. Property `ReduxOptions > lq_usm_radius_range`](#lq_usm_radius_range)
  - [31.1. ReduxOptions > lq_usm_radius_range > lq_usm_radius_range item 0](#autogenerated_heading_2)
  - [31.2. ReduxOptions > lq_usm_radius_range > lq_usm_radius_range item 1](#autogenerated_heading_3)
- [32. Property `ReduxOptions > blur_prob`](#blur_prob)
- [33. Property `ReduxOptions > resize_prob`](#resize_prob)
  - [33.1. ReduxOptions > resize_prob > resize_prob items](#resize_prob_items)
- [34. Property `ReduxOptions > resize_mode_list`](#resize_mode_list)
  - [34.1. ReduxOptions > resize_mode_list > resize_mode_list items](#resize_mode_list_items)
- [35. Property `ReduxOptions > resize_mode_prob`](#resize_mode_prob)
  - [35.1. ReduxOptions > resize_mode_prob > resize_mode_prob items](#resize_mode_prob_items)
- [36. Property `ReduxOptions > resize_range`](#resize_range)
  - [36.1. ReduxOptions > resize_range > resize_range item 0](#autogenerated_heading_4)
  - [36.2. ReduxOptions > resize_range > resize_range item 1](#autogenerated_heading_5)
- [37. Property `ReduxOptions > gaussian_noise_prob`](#gaussian_noise_prob)
- [38. Property `ReduxOptions > noise_range`](#noise_range)
  - [38.1. ReduxOptions > noise_range > noise_range item 0](#autogenerated_heading_6)
  - [38.2. ReduxOptions > noise_range > noise_range item 1](#autogenerated_heading_7)
- [39. Property `ReduxOptions > poisson_scale_range`](#poisson_scale_range)
  - [39.1. ReduxOptions > poisson_scale_range > poisson_scale_range item 0](#autogenerated_heading_8)
  - [39.2. ReduxOptions > poisson_scale_range > poisson_scale_range item 1](#autogenerated_heading_9)
- [40. Property `ReduxOptions > gray_noise_prob`](#gray_noise_prob)
- [41. Property `ReduxOptions > jpeg_prob`](#jpeg_prob)
- [42. Property `ReduxOptions > jpeg_range`](#jpeg_range)
  - [42.1. ReduxOptions > jpeg_range > jpeg_range item 0](#autogenerated_heading_10)
  - [42.2. ReduxOptions > jpeg_range > jpeg_range item 1](#autogenerated_heading_11)
- [43. Property `ReduxOptions > blur_prob2`](#blur_prob2)
- [44. Property `ReduxOptions > resize_prob2`](#resize_prob2)
  - [44.1. ReduxOptions > resize_prob2 > resize_prob2 items](#resize_prob2_items)
- [45. Property `ReduxOptions > resize_mode_list2`](#resize_mode_list2)
  - [45.1. ReduxOptions > resize_mode_list2 > resize_mode_list2 items](#resize_mode_list2_items)
- [46. Property `ReduxOptions > resize_mode_prob2`](#resize_mode_prob2)
  - [46.1. ReduxOptions > resize_mode_prob2 > resize_mode_prob2 items](#resize_mode_prob2_items)
- [47. Property `ReduxOptions > resize_range2`](#resize_range2)
  - [47.1. ReduxOptions > resize_range2 > resize_range2 item 0](#autogenerated_heading_12)
  - [47.2. ReduxOptions > resize_range2 > resize_range2 item 1](#autogenerated_heading_13)
- [48. Property `ReduxOptions > gaussian_noise_prob2`](#gaussian_noise_prob2)
- [49. Property `ReduxOptions > noise_range2`](#noise_range2)
  - [49.1. ReduxOptions > noise_range2 > noise_range2 item 0](#autogenerated_heading_14)
  - [49.2. ReduxOptions > noise_range2 > noise_range2 item 1](#autogenerated_heading_15)
- [50. Property `ReduxOptions > poisson_scale_range2`](#poisson_scale_range2)
  - [50.1. ReduxOptions > poisson_scale_range2 > poisson_scale_range2 item 0](#autogenerated_heading_16)
  - [50.2. ReduxOptions > poisson_scale_range2 > poisson_scale_range2 item 1](#autogenerated_heading_17)
- [51. Property `ReduxOptions > gray_noise_prob2`](#gray_noise_prob2)
- [52. Property `ReduxOptions > jpeg_prob2`](#jpeg_prob2)
- [53. Property `ReduxOptions > jpeg_range2`](#jpeg_range2)
  - [53.1. ReduxOptions > jpeg_range2 > jpeg_range2 items](#jpeg_range2_items)
- [54. Property `ReduxOptions > resize_mode_list3`](#resize_mode_list3)
  - [54.1. ReduxOptions > resize_mode_list3 > resize_mode_list3 items](#resize_mode_list3_items)
- [55. Property `ReduxOptions > resize_mode_prob3`](#resize_mode_prob3)
  - [55.1. ReduxOptions > resize_mode_prob3 > resize_mode_prob3 items](#resize_mode_prob3_items)
- [56. Property `ReduxOptions > queue_size`](#queue_size)
- [57. Property `ReduxOptions > datasets`](#datasets)
  - [57.1. Property `ReduxOptions > datasets > DatasetOptions`](#datasets_additionalProperties)
    - [57.1.1. Property `ReduxOptions > datasets > DatasetOptions > name`](#datasets_additionalProperties_name)
    - [57.1.2. Property `ReduxOptions > datasets > DatasetOptions > type`](#datasets_additionalProperties_type)
    - [57.1.3. Property `ReduxOptions > datasets > DatasetOptions > io_backend`](#datasets_additionalProperties_io_backend)
    - [57.1.4. Property `ReduxOptions > datasets > DatasetOptions > num_worker_per_gpu`](#datasets_additionalProperties_num_worker_per_gpu)
      - [57.1.4.1. Property `ReduxOptions > datasets > DatasetOptions > num_worker_per_gpu > anyOf > item 0`](#datasets_additionalProperties_num_worker_per_gpu_anyOf_i0)
      - [57.1.4.2. Property `ReduxOptions > datasets > DatasetOptions > num_worker_per_gpu > anyOf > item 1`](#datasets_additionalProperties_num_worker_per_gpu_anyOf_i1)
    - [57.1.5. Property `ReduxOptions > datasets > DatasetOptions > batch_size_per_gpu`](#datasets_additionalProperties_batch_size_per_gpu)
      - [57.1.5.1. Property `ReduxOptions > datasets > DatasetOptions > batch_size_per_gpu > anyOf > item 0`](#datasets_additionalProperties_batch_size_per_gpu_anyOf_i0)
      - [57.1.5.2. Property `ReduxOptions > datasets > DatasetOptions > batch_size_per_gpu > anyOf > item 1`](#datasets_additionalProperties_batch_size_per_gpu_anyOf_i1)
    - [57.1.6. Property `ReduxOptions > datasets > DatasetOptions > accum_iter`](#datasets_additionalProperties_accum_iter)
    - [57.1.7. Property `ReduxOptions > datasets > DatasetOptions > use_hflip`](#datasets_additionalProperties_use_hflip)
    - [57.1.8. Property `ReduxOptions > datasets > DatasetOptions > use_rot`](#datasets_additionalProperties_use_rot)
    - [57.1.9. Property `ReduxOptions > datasets > DatasetOptions > mean`](#datasets_additionalProperties_mean)
      - [57.1.9.1. Property `ReduxOptions > datasets > DatasetOptions > mean > anyOf > item 0`](#datasets_additionalProperties_mean_anyOf_i0)
        - [57.1.9.1.1. ReduxOptions > datasets > DatasetOptions > mean > anyOf > item 0 > item 0 items](#datasets_additionalProperties_mean_anyOf_i0_items)
      - [57.1.9.2. Property `ReduxOptions > datasets > DatasetOptions > mean > anyOf > item 1`](#datasets_additionalProperties_mean_anyOf_i1)
    - [57.1.10. Property `ReduxOptions > datasets > DatasetOptions > std`](#datasets_additionalProperties_std)
      - [57.1.10.1. Property `ReduxOptions > datasets > DatasetOptions > std > anyOf > item 0`](#datasets_additionalProperties_std_anyOf_i0)
        - [57.1.10.1.1. ReduxOptions > datasets > DatasetOptions > std > anyOf > item 0 > item 0 items](#datasets_additionalProperties_std_anyOf_i0_items)
      - [57.1.10.2. Property `ReduxOptions > datasets > DatasetOptions > std > anyOf > item 1`](#datasets_additionalProperties_std_anyOf_i1)
    - [57.1.11. Property `ReduxOptions > datasets > DatasetOptions > gt_size`](#datasets_additionalProperties_gt_size)
      - [57.1.11.1. Property `ReduxOptions > datasets > DatasetOptions > gt_size > anyOf > item 0`](#datasets_additionalProperties_gt_size_anyOf_i0)
      - [57.1.11.2. Property `ReduxOptions > datasets > DatasetOptions > gt_size > anyOf > item 1`](#datasets_additionalProperties_gt_size_anyOf_i1)
    - [57.1.12. Property `ReduxOptions > datasets > DatasetOptions > lq_size`](#datasets_additionalProperties_lq_size)
      - [57.1.12.1. Property `ReduxOptions > datasets > DatasetOptions > lq_size > anyOf > item 0`](#datasets_additionalProperties_lq_size_anyOf_i0)
      - [57.1.12.2. Property `ReduxOptions > datasets > DatasetOptions > lq_size > anyOf > item 1`](#datasets_additionalProperties_lq_size_anyOf_i1)
    - [57.1.13. Property `ReduxOptions > datasets > DatasetOptions > color`](#datasets_additionalProperties_color)
      - [57.1.13.1. Property `ReduxOptions > datasets > DatasetOptions > color > anyOf > item 0`](#datasets_additionalProperties_color_anyOf_i0)
      - [57.1.13.2. Property `ReduxOptions > datasets > DatasetOptions > color > anyOf > item 1`](#datasets_additionalProperties_color_anyOf_i1)
    - [57.1.14. Property `ReduxOptions > datasets > DatasetOptions > phase`](#datasets_additionalProperties_phase)
      - [57.1.14.1. Property `ReduxOptions > datasets > DatasetOptions > phase > anyOf > item 0`](#datasets_additionalProperties_phase_anyOf_i0)
      - [57.1.14.2. Property `ReduxOptions > datasets > DatasetOptions > phase > anyOf > item 1`](#datasets_additionalProperties_phase_anyOf_i1)
    - [57.1.15. Property `ReduxOptions > datasets > DatasetOptions > scale`](#datasets_additionalProperties_scale)
      - [57.1.15.1. Property `ReduxOptions > datasets > DatasetOptions > scale > anyOf > item 0`](#datasets_additionalProperties_scale_anyOf_i0)
      - [57.1.15.2. Property `ReduxOptions > datasets > DatasetOptions > scale > anyOf > item 1`](#datasets_additionalProperties_scale_anyOf_i1)
    - [57.1.16. Property `ReduxOptions > datasets > DatasetOptions > dataset_enlarge_ratio`](#datasets_additionalProperties_dataset_enlarge_ratio)
      - [57.1.16.1. Property `ReduxOptions > datasets > DatasetOptions > dataset_enlarge_ratio > anyOf > item 0`](#datasets_additionalProperties_dataset_enlarge_ratio_anyOf_i0)
      - [57.1.16.2. Property `ReduxOptions > datasets > DatasetOptions > dataset_enlarge_ratio > anyOf > item 1`](#datasets_additionalProperties_dataset_enlarge_ratio_anyOf_i1)
    - [57.1.17. Property `ReduxOptions > datasets > DatasetOptions > prefetch_mode`](#datasets_additionalProperties_prefetch_mode)
      - [57.1.17.1. Property `ReduxOptions > datasets > DatasetOptions > prefetch_mode > anyOf > item 0`](#datasets_additionalProperties_prefetch_mode_anyOf_i0)
      - [57.1.17.2. Property `ReduxOptions > datasets > DatasetOptions > prefetch_mode > anyOf > item 1`](#datasets_additionalProperties_prefetch_mode_anyOf_i1)
    - [57.1.18. Property `ReduxOptions > datasets > DatasetOptions > pin_memory`](#datasets_additionalProperties_pin_memory)
    - [57.1.19. Property `ReduxOptions > datasets > DatasetOptions > persistent_workers`](#datasets_additionalProperties_persistent_workers)
    - [57.1.20. Property `ReduxOptions > datasets > DatasetOptions > num_prefetch_queue`](#datasets_additionalProperties_num_prefetch_queue)
    - [57.1.21. Property `ReduxOptions > datasets > DatasetOptions > prefetch_factor`](#datasets_additionalProperties_prefetch_factor)
    - [57.1.22. Property `ReduxOptions > datasets > DatasetOptions > clip_size`](#datasets_additionalProperties_clip_size)
      - [57.1.22.1. Property `ReduxOptions > datasets > DatasetOptions > clip_size > anyOf > item 0`](#datasets_additionalProperties_clip_size_anyOf_i0)
      - [57.1.22.2. Property `ReduxOptions > datasets > DatasetOptions > clip_size > anyOf > item 1`](#datasets_additionalProperties_clip_size_anyOf_i1)
    - [57.1.23. Property `ReduxOptions > datasets > DatasetOptions > dataroot_gt`](#datasets_additionalProperties_dataroot_gt)
      - [57.1.23.1. Property `ReduxOptions > datasets > DatasetOptions > dataroot_gt > anyOf > item 0`](#datasets_additionalProperties_dataroot_gt_anyOf_i0)
      - [57.1.23.2. Property `ReduxOptions > datasets > DatasetOptions > dataroot_gt > anyOf > item 1`](#datasets_additionalProperties_dataroot_gt_anyOf_i1)
        - [57.1.23.2.1. ReduxOptions > datasets > DatasetOptions > dataroot_gt > anyOf > item 1 > item 1 items](#datasets_additionalProperties_dataroot_gt_anyOf_i1_items)
      - [57.1.23.3. Property `ReduxOptions > datasets > DatasetOptions > dataroot_gt > anyOf > item 2`](#datasets_additionalProperties_dataroot_gt_anyOf_i2)
    - [57.1.24. Property `ReduxOptions > datasets > DatasetOptions > dataroot_lq`](#datasets_additionalProperties_dataroot_lq)
      - [57.1.24.1. Property `ReduxOptions > datasets > DatasetOptions > dataroot_lq > anyOf > item 0`](#datasets_additionalProperties_dataroot_lq_anyOf_i0)
      - [57.1.24.2. Property `ReduxOptions > datasets > DatasetOptions > dataroot_lq > anyOf > item 1`](#datasets_additionalProperties_dataroot_lq_anyOf_i1)
        - [57.1.24.2.1. ReduxOptions > datasets > DatasetOptions > dataroot_lq > anyOf > item 1 > item 1 items](#datasets_additionalProperties_dataroot_lq_anyOf_i1_items)
      - [57.1.24.3. Property `ReduxOptions > datasets > DatasetOptions > dataroot_lq > anyOf > item 2`](#datasets_additionalProperties_dataroot_lq_anyOf_i2)
    - [57.1.25. Property `ReduxOptions > datasets > DatasetOptions > meta_info`](#datasets_additionalProperties_meta_info)
      - [57.1.25.1. Property `ReduxOptions > datasets > DatasetOptions > meta_info > anyOf > item 0`](#datasets_additionalProperties_meta_info_anyOf_i0)
      - [57.1.25.2. Property `ReduxOptions > datasets > DatasetOptions > meta_info > anyOf > item 1`](#datasets_additionalProperties_meta_info_anyOf_i1)
    - [57.1.26. Property `ReduxOptions > datasets > DatasetOptions > filename_tmpl`](#datasets_additionalProperties_filename_tmpl)
    - [57.1.27. Property `ReduxOptions > datasets > DatasetOptions > blur_kernel_size`](#datasets_additionalProperties_blur_kernel_size)
    - [57.1.28. Property `ReduxOptions > datasets > DatasetOptions > kernel_list`](#datasets_additionalProperties_kernel_list)
      - [57.1.28.1. ReduxOptions > datasets > DatasetOptions > kernel_list > kernel_list items](#datasets_additionalProperties_kernel_list_items)
    - [57.1.29. Property `ReduxOptions > datasets > DatasetOptions > kernel_prob`](#datasets_additionalProperties_kernel_prob)
      - [57.1.29.1. ReduxOptions > datasets > DatasetOptions > kernel_prob > kernel_prob items](#datasets_additionalProperties_kernel_prob_items)
    - [57.1.30. Property `ReduxOptions > datasets > DatasetOptions > kernel_range`](#datasets_additionalProperties_kernel_range)
      - [57.1.30.1. ReduxOptions > datasets > DatasetOptions > kernel_range > kernel_range item 0](#autogenerated_heading_18)
      - [57.1.30.2. ReduxOptions > datasets > DatasetOptions > kernel_range > kernel_range item 1](#autogenerated_heading_19)
    - [57.1.31. Property `ReduxOptions > datasets > DatasetOptions > sinc_prob`](#datasets_additionalProperties_sinc_prob)
    - [57.1.32. Property `ReduxOptions > datasets > DatasetOptions > blur_sigma`](#datasets_additionalProperties_blur_sigma)
      - [57.1.32.1. ReduxOptions > datasets > DatasetOptions > blur_sigma > blur_sigma item 0](#autogenerated_heading_20)
      - [57.1.32.2. ReduxOptions > datasets > DatasetOptions > blur_sigma > blur_sigma item 1](#autogenerated_heading_21)
    - [57.1.33. Property `ReduxOptions > datasets > DatasetOptions > betag_range`](#datasets_additionalProperties_betag_range)
      - [57.1.33.1. ReduxOptions > datasets > DatasetOptions > betag_range > betag_range item 0](#autogenerated_heading_22)
      - [57.1.33.2. ReduxOptions > datasets > DatasetOptions > betag_range > betag_range item 1](#autogenerated_heading_23)
    - [57.1.34. Property `ReduxOptions > datasets > DatasetOptions > betap_range`](#datasets_additionalProperties_betap_range)
      - [57.1.34.1. ReduxOptions > datasets > DatasetOptions > betap_range > betap_range item 0](#autogenerated_heading_24)
      - [57.1.34.2. ReduxOptions > datasets > DatasetOptions > betap_range > betap_range item 1](#autogenerated_heading_25)
    - [57.1.35. Property `ReduxOptions > datasets > DatasetOptions > blur_kernel_size2`](#datasets_additionalProperties_blur_kernel_size2)
    - [57.1.36. Property `ReduxOptions > datasets > DatasetOptions > kernel_list2`](#datasets_additionalProperties_kernel_list2)
      - [57.1.36.1. ReduxOptions > datasets > DatasetOptions > kernel_list2 > kernel_list2 items](#datasets_additionalProperties_kernel_list2_items)
    - [57.1.37. Property `ReduxOptions > datasets > DatasetOptions > kernel_prob2`](#datasets_additionalProperties_kernel_prob2)
      - [57.1.37.1. ReduxOptions > datasets > DatasetOptions > kernel_prob2 > kernel_prob2 items](#datasets_additionalProperties_kernel_prob2_items)
    - [57.1.38. Property `ReduxOptions > datasets > DatasetOptions > kernel_range2`](#datasets_additionalProperties_kernel_range2)
      - [57.1.38.1. ReduxOptions > datasets > DatasetOptions > kernel_range2 > kernel_range2 item 0](#autogenerated_heading_26)
      - [57.1.38.2. ReduxOptions > datasets > DatasetOptions > kernel_range2 > kernel_range2 item 1](#autogenerated_heading_27)
    - [57.1.39. Property `ReduxOptions > datasets > DatasetOptions > sinc_prob2`](#datasets_additionalProperties_sinc_prob2)
    - [57.1.40. Property `ReduxOptions > datasets > DatasetOptions > blur_sigma2`](#datasets_additionalProperties_blur_sigma2)
      - [57.1.40.1. ReduxOptions > datasets > DatasetOptions > blur_sigma2 > blur_sigma2 item 0](#autogenerated_heading_28)
      - [57.1.40.2. ReduxOptions > datasets > DatasetOptions > blur_sigma2 > blur_sigma2 item 1](#autogenerated_heading_29)
    - [57.1.41. Property `ReduxOptions > datasets > DatasetOptions > betag_range2`](#datasets_additionalProperties_betag_range2)
      - [57.1.41.1. ReduxOptions > datasets > DatasetOptions > betag_range2 > betag_range2 item 0](#autogenerated_heading_30)
      - [57.1.41.2. ReduxOptions > datasets > DatasetOptions > betag_range2 > betag_range2 item 1](#autogenerated_heading_31)
    - [57.1.42. Property `ReduxOptions > datasets > DatasetOptions > betap_range2`](#datasets_additionalProperties_betap_range2)
      - [57.1.42.1. ReduxOptions > datasets > DatasetOptions > betap_range2 > betap_range2 item 0](#autogenerated_heading_32)
      - [57.1.42.2. ReduxOptions > datasets > DatasetOptions > betap_range2 > betap_range2 item 1](#autogenerated_heading_33)
    - [57.1.43. Property `ReduxOptions > datasets > DatasetOptions > final_sinc_prob`](#datasets_additionalProperties_final_sinc_prob)
    - [57.1.44. Property `ReduxOptions > datasets > DatasetOptions > final_kernel_range`](#datasets_additionalProperties_final_kernel_range)
      - [57.1.44.1. ReduxOptions > datasets > DatasetOptions > final_kernel_range > final_kernel_range item 0](#autogenerated_heading_34)
      - [57.1.44.2. ReduxOptions > datasets > DatasetOptions > final_kernel_range > final_kernel_range item 1](#autogenerated_heading_35)
- [58. Property `ReduxOptions > train`](#train)
  - [58.1. Property `ReduxOptions > train > type`](#train_type)
  - [58.2. Property `ReduxOptions > train > total_iter`](#train_total_iter)
  - [58.3. Property `ReduxOptions > train > optim_g`](#train_optim_g)
  - [58.4. Property `ReduxOptions > train > ema_decay`](#train_ema_decay)
  - [58.5. Property `ReduxOptions > train > grad_clip`](#train_grad_clip)
  - [58.6. Property `ReduxOptions > train > warmup_iter`](#train_warmup_iter)
  - [58.7. Property `ReduxOptions > train > scheduler`](#train_scheduler)
    - [58.7.1. Property `ReduxOptions > train > scheduler > anyOf > item 0`](#train_scheduler_anyOf_i0)
    - [58.7.2. Property `ReduxOptions > train > scheduler > anyOf > SchedulerOptions`](#train_scheduler_anyOf_i1)
      - [58.7.2.1. Property `ReduxOptions > train > scheduler > anyOf > SchedulerOptions > type`](#train_scheduler_anyOf_i1_type)
      - [58.7.2.2. Property `ReduxOptions > train > scheduler > anyOf > SchedulerOptions > milestones`](#train_scheduler_anyOf_i1_milestones)
        - [58.7.2.2.1. ReduxOptions > train > scheduler > anyOf > SchedulerOptions > milestones > milestones items](#train_scheduler_anyOf_i1_milestones_items)
      - [58.7.2.3. Property `ReduxOptions > train > scheduler > anyOf > SchedulerOptions > gamma`](#train_scheduler_anyOf_i1_gamma)
  - [58.8. Property `ReduxOptions > train > optim_d`](#train_optim_d)
    - [58.8.1. Property `ReduxOptions > train > optim_d > anyOf > item 0`](#train_optim_d_anyOf_i0)
    - [58.8.2. Property `ReduxOptions > train > optim_d > anyOf > item 1`](#train_optim_d_anyOf_i1)
  - [58.9. Property `ReduxOptions > train > losses`](#train_losses)
    - [58.9.1. ReduxOptions > train > losses > losses items](#train_losses_items)
      - [58.9.1.1. Property `ReduxOptions > train > losses > losses items > anyOf > ganloss`](#train_losses_items_anyOf_i0)
        - [58.9.1.1.1. Property `ReduxOptions > train > losses > losses items > anyOf > ganloss > type`](#train_losses_items_anyOf_i0_type)
        - [58.9.1.1.2. Property `ReduxOptions > train > losses > losses items > anyOf > ganloss > gan_type`](#train_losses_items_anyOf_i0_gan_type)
        - [58.9.1.1.3. Property `ReduxOptions > train > losses > losses items > anyOf > ganloss > real_label_val`](#train_losses_items_anyOf_i0_real_label_val)
        - [58.9.1.1.4. Property `ReduxOptions > train > losses > losses items > anyOf > ganloss > fake_label_val`](#train_losses_items_anyOf_i0_fake_label_val)
        - [58.9.1.1.5. Property `ReduxOptions > train > losses > losses items > anyOf > ganloss > loss_weight`](#train_losses_items_anyOf_i0_loss_weight)
      - [58.9.1.2. Property `ReduxOptions > train > losses > losses items > anyOf > multiscaleganloss`](#train_losses_items_anyOf_i1)
        - [58.9.1.2.1. Property `ReduxOptions > train > losses > losses items > anyOf > multiscaleganloss > type`](#train_losses_items_anyOf_i1_type)
        - [58.9.1.2.2. Property `ReduxOptions > train > losses > losses items > anyOf > multiscaleganloss > gan_type`](#train_losses_items_anyOf_i1_gan_type)
        - [58.9.1.2.3. Property `ReduxOptions > train > losses > losses items > anyOf > multiscaleganloss > real_label_val`](#train_losses_items_anyOf_i1_real_label_val)
        - [58.9.1.2.4. Property `ReduxOptions > train > losses > losses items > anyOf > multiscaleganloss > fake_label_val`](#train_losses_items_anyOf_i1_fake_label_val)
        - [58.9.1.2.5. Property `ReduxOptions > train > losses > losses items > anyOf > multiscaleganloss > loss_weight`](#train_losses_items_anyOf_i1_loss_weight)
      - [58.9.1.3. Property `ReduxOptions > train > losses > losses items > anyOf > adistsloss`](#train_losses_items_anyOf_i2)
        - [58.9.1.3.1. Property `ReduxOptions > train > losses > losses items > anyOf > adistsloss > type`](#train_losses_items_anyOf_i2_type)
        - [58.9.1.3.2. Property `ReduxOptions > train > losses > losses items > anyOf > adistsloss > window_size`](#train_losses_items_anyOf_i2_window_size)
        - [58.9.1.3.3. Property `ReduxOptions > train > losses > losses items > anyOf > adistsloss > resize_input`](#train_losses_items_anyOf_i2_resize_input)
        - [58.9.1.3.4. Property `ReduxOptions > train > losses > losses items > anyOf > adistsloss > loss_weight`](#train_losses_items_anyOf_i2_loss_weight)
      - [58.9.1.4. Property `ReduxOptions > train > losses > losses items > anyOf > l1loss`](#train_losses_items_anyOf_i3)
        - [58.9.1.4.1. Property `ReduxOptions > train > losses > losses items > anyOf > l1loss > type`](#train_losses_items_anyOf_i3_type)
        - [58.9.1.4.2. Property `ReduxOptions > train > losses > losses items > anyOf > l1loss > loss_weight`](#train_losses_items_anyOf_i3_loss_weight)
        - [58.9.1.4.3. Property `ReduxOptions > train > losses > losses items > anyOf > l1loss > reduction`](#train_losses_items_anyOf_i3_reduction)
      - [58.9.1.5. Property `ReduxOptions > train > losses > losses items > anyOf > mseloss`](#train_losses_items_anyOf_i4)
        - [58.9.1.5.1. Property `ReduxOptions > train > losses > losses items > anyOf > mseloss > type`](#train_losses_items_anyOf_i4_type)
        - [58.9.1.5.2. Property `ReduxOptions > train > losses > losses items > anyOf > mseloss > loss_weight`](#train_losses_items_anyOf_i4_loss_weight)
        - [58.9.1.5.3. Property `ReduxOptions > train > losses > losses items > anyOf > mseloss > reduction`](#train_losses_items_anyOf_i4_reduction)
      - [58.9.1.6. Property `ReduxOptions > train > losses > losses items > anyOf > charbonnierloss`](#train_losses_items_anyOf_i5)
        - [58.9.1.6.1. Property `ReduxOptions > train > losses > losses items > anyOf > charbonnierloss > type`](#train_losses_items_anyOf_i5_type)
        - [58.9.1.6.2. Property `ReduxOptions > train > losses > losses items > anyOf > charbonnierloss > loss_weight`](#train_losses_items_anyOf_i5_loss_weight)
        - [58.9.1.6.3. Property `ReduxOptions > train > losses > losses items > anyOf > charbonnierloss > reduction`](#train_losses_items_anyOf_i5_reduction)
        - [58.9.1.6.4. Property `ReduxOptions > train > losses > losses items > anyOf > charbonnierloss > eps`](#train_losses_items_anyOf_i5_eps)
      - [58.9.1.7. Property `ReduxOptions > train > losses > losses items > anyOf > colorloss`](#train_losses_items_anyOf_i6)
        - [58.9.1.7.1. Property `ReduxOptions > train > losses > losses items > anyOf > colorloss > type`](#train_losses_items_anyOf_i6_type)
        - [58.9.1.7.2. Property `ReduxOptions > train > losses > losses items > anyOf > colorloss > criterion`](#train_losses_items_anyOf_i6_criterion)
        - [58.9.1.7.3. Property `ReduxOptions > train > losses > losses items > anyOf > colorloss > loss_weight`](#train_losses_items_anyOf_i6_loss_weight)
        - [58.9.1.7.4. Property `ReduxOptions > train > losses > losses items > anyOf > colorloss > scale`](#train_losses_items_anyOf_i6_scale)
      - [58.9.1.8. Property `ReduxOptions > train > losses > losses items > anyOf > averageloss`](#train_losses_items_anyOf_i7)
        - [58.9.1.8.1. Property `ReduxOptions > train > losses > losses items > anyOf > averageloss > type`](#train_losses_items_anyOf_i7_type)
        - [58.9.1.8.2. Property `ReduxOptions > train > losses > losses items > anyOf > averageloss > criterion`](#train_losses_items_anyOf_i7_criterion)
        - [58.9.1.8.3. Property `ReduxOptions > train > losses > losses items > anyOf > averageloss > loss_weight`](#train_losses_items_anyOf_i7_loss_weight)
        - [58.9.1.8.4. Property `ReduxOptions > train > losses > losses items > anyOf > averageloss > scale`](#train_losses_items_anyOf_i7_scale)
      - [58.9.1.9. Property `ReduxOptions > train > losses > losses items > anyOf > bicubicloss`](#train_losses_items_anyOf_i8)
        - [58.9.1.9.1. Property `ReduxOptions > train > losses > losses items > anyOf > bicubicloss > type`](#train_losses_items_anyOf_i8_type)
        - [58.9.1.9.2. Property `ReduxOptions > train > losses > losses items > anyOf > bicubicloss > criterion`](#train_losses_items_anyOf_i8_criterion)
        - [58.9.1.9.3. Property `ReduxOptions > train > losses > losses items > anyOf > bicubicloss > loss_weight`](#train_losses_items_anyOf_i8_loss_weight)
        - [58.9.1.9.4. Property `ReduxOptions > train > losses > losses items > anyOf > bicubicloss > scale`](#train_losses_items_anyOf_i8_scale)
      - [58.9.1.10. Property `ReduxOptions > train > losses > losses items > anyOf > lumaloss`](#train_losses_items_anyOf_i9)
        - [58.9.1.10.1. Property `ReduxOptions > train > losses > losses items > anyOf > lumaloss > type`](#train_losses_items_anyOf_i9_type)
        - [58.9.1.10.2. Property `ReduxOptions > train > losses > losses items > anyOf > lumaloss > criterion`](#train_losses_items_anyOf_i9_criterion)
        - [58.9.1.10.3. Property `ReduxOptions > train > losses > losses items > anyOf > lumaloss > loss_weight`](#train_losses_items_anyOf_i9_loss_weight)
      - [58.9.1.11. Property `ReduxOptions > train > losses > losses items > anyOf > hsluvloss`](#train_losses_items_anyOf_i10)
        - [58.9.1.11.1. Property `ReduxOptions > train > losses > losses items > anyOf > hsluvloss > type`](#train_losses_items_anyOf_i10_type)
        - [58.9.1.11.2. Property `ReduxOptions > train > losses > losses items > anyOf > hsluvloss > criterion`](#train_losses_items_anyOf_i10_criterion)
        - [58.9.1.11.3. Property `ReduxOptions > train > losses > losses items > anyOf > hsluvloss > loss_weight`](#train_losses_items_anyOf_i10_loss_weight)
      - [58.9.1.12. Property `ReduxOptions > train > losses > losses items > anyOf > contextualloss`](#train_losses_items_anyOf_i11)
        - [58.9.1.12.1. Property `ReduxOptions > train > losses > losses items > anyOf > contextualloss > type`](#train_losses_items_anyOf_i11_type)
        - [58.9.1.12.2. Property `ReduxOptions > train > losses > losses items > anyOf > contextualloss > loss_weight`](#train_losses_items_anyOf_i11_loss_weight)
        - [58.9.1.12.3. Property `ReduxOptions > train > losses > losses items > anyOf > contextualloss > layer_weights`](#train_losses_items_anyOf_i11_layer_weights)
          - [58.9.1.12.3.1. Property `ReduxOptions > train > losses > losses items > anyOf > contextualloss > layer_weights > anyOf > item 0`](#train_losses_items_anyOf_i11_layer_weights_anyOf_i0)
            - [58.9.1.12.3.1.1. Property `ReduxOptions > train > losses > losses items > anyOf > contextualloss > layer_weights > anyOf > item 0 > additionalProperties`](#train_losses_items_anyOf_i11_layer_weights_anyOf_i0_additionalProperties)
          - [58.9.1.12.3.2. Property `ReduxOptions > train > losses > losses items > anyOf > contextualloss > layer_weights > anyOf > item 1`](#train_losses_items_anyOf_i11_layer_weights_anyOf_i1)
        - [58.9.1.12.4. Property `ReduxOptions > train > losses > losses items > anyOf > contextualloss > crop_quarter`](#train_losses_items_anyOf_i11_crop_quarter)
        - [58.9.1.12.5. Property `ReduxOptions > train > losses > losses items > anyOf > contextualloss > max_1d_size`](#train_losses_items_anyOf_i11_max_1d_size)
        - [58.9.1.12.6. Property `ReduxOptions > train > losses > losses items > anyOf > contextualloss > distance_type`](#train_losses_items_anyOf_i11_distance_type)
        - [58.9.1.12.7. Property `ReduxOptions > train > losses > losses items > anyOf > contextualloss > b`](#train_losses_items_anyOf_i11_b)
        - [58.9.1.12.8. Property `ReduxOptions > train > losses > losses items > anyOf > contextualloss > band_width`](#train_losses_items_anyOf_i11_band_width)
        - [58.9.1.12.9. Property `ReduxOptions > train > losses > losses items > anyOf > contextualloss > use_vgg`](#train_losses_items_anyOf_i11_use_vgg)
        - [58.9.1.12.10. Property `ReduxOptions > train > losses > losses items > anyOf > contextualloss > net`](#train_losses_items_anyOf_i11_net)
        - [58.9.1.12.11. Property `ReduxOptions > train > losses > losses items > anyOf > contextualloss > calc_type`](#train_losses_items_anyOf_i11_calc_type)
        - [58.9.1.12.12. Property `ReduxOptions > train > losses > losses items > anyOf > contextualloss > z_norm`](#train_losses_items_anyOf_i11_z_norm)
      - [58.9.1.13. Property `ReduxOptions > train > losses > losses items > anyOf > distsloss`](#train_losses_items_anyOf_i12)
        - [58.9.1.13.1. Property `ReduxOptions > train > losses > losses items > anyOf > distsloss > type`](#train_losses_items_anyOf_i12_type)
        - [58.9.1.13.2. Property `ReduxOptions > train > losses > losses items > anyOf > distsloss > as_loss`](#train_losses_items_anyOf_i12_as_loss)
        - [58.9.1.13.3. Property `ReduxOptions > train > losses > losses items > anyOf > distsloss > loss_weight`](#train_losses_items_anyOf_i12_loss_weight)
        - [58.9.1.13.4. Property `ReduxOptions > train > losses > losses items > anyOf > distsloss > load_weights`](#train_losses_items_anyOf_i12_load_weights)
        - [58.9.1.13.5. Property `ReduxOptions > train > losses > losses items > anyOf > distsloss > use_input_norm`](#train_losses_items_anyOf_i12_use_input_norm)
        - [58.9.1.13.6. Property `ReduxOptions > train > losses > losses items > anyOf > distsloss > clip_min`](#train_losses_items_anyOf_i12_clip_min)
      - [58.9.1.14. Property `ReduxOptions > train > losses > losses items > anyOf > dsdloss`](#train_losses_items_anyOf_i13)
        - [58.9.1.14.1. Property `ReduxOptions > train > losses > losses items > anyOf > dsdloss > type`](#train_losses_items_anyOf_i13_type)
        - [58.9.1.14.2. Property `ReduxOptions > train > losses > losses items > anyOf > dsdloss > features_to_compute`](#train_losses_items_anyOf_i13_features_to_compute)
          - [58.9.1.14.2.1. ReduxOptions > train > losses > losses items > anyOf > dsdloss > features_to_compute > features_to_compute items](#train_losses_items_anyOf_i13_features_to_compute_items)
        - [58.9.1.14.3. Property `ReduxOptions > train > losses > losses items > anyOf > dsdloss > scales`](#train_losses_items_anyOf_i13_scales)
          - [58.9.1.14.3.1. ReduxOptions > train > losses > losses items > anyOf > dsdloss > scales > scales items](#train_losses_items_anyOf_i13_scales_items)
        - [58.9.1.14.4. Property `ReduxOptions > train > losses > losses items > anyOf > dsdloss > criterion`](#train_losses_items_anyOf_i13_criterion)
        - [58.9.1.14.5. Property `ReduxOptions > train > losses > losses items > anyOf > dsdloss > shave_edge`](#train_losses_items_anyOf_i13_shave_edge)
          - [58.9.1.14.5.1. Property `ReduxOptions > train > losses > losses items > anyOf > dsdloss > shave_edge > anyOf > item 0`](#train_losses_items_anyOf_i13_shave_edge_anyOf_i0)
          - [58.9.1.14.5.2. Property `ReduxOptions > train > losses > losses items > anyOf > dsdloss > shave_edge > anyOf > item 1`](#train_losses_items_anyOf_i13_shave_edge_anyOf_i1)
        - [58.9.1.14.6. Property `ReduxOptions > train > losses > losses items > anyOf > dsdloss > loss_weight`](#train_losses_items_anyOf_i13_loss_weight)
      - [58.9.1.15. Property `ReduxOptions > train > losses > losses items > anyOf > ffloss`](#train_losses_items_anyOf_i14)
        - [58.9.1.15.1. Property `ReduxOptions > train > losses > losses items > anyOf > ffloss > type`](#train_losses_items_anyOf_i14_type)
        - [58.9.1.15.2. Property `ReduxOptions > train > losses > losses items > anyOf > ffloss > loss_weight`](#train_losses_items_anyOf_i14_loss_weight)
        - [58.9.1.15.3. Property `ReduxOptions > train > losses > losses items > anyOf > ffloss > alpha`](#train_losses_items_anyOf_i14_alpha)
        - [58.9.1.15.4. Property `ReduxOptions > train > losses > losses items > anyOf > ffloss > patch_factor`](#train_losses_items_anyOf_i14_patch_factor)
        - [58.9.1.15.5. Property `ReduxOptions > train > losses > losses items > anyOf > ffloss > ave_spectrum`](#train_losses_items_anyOf_i14_ave_spectrum)
        - [58.9.1.15.6. Property `ReduxOptions > train > losses > losses items > anyOf > ffloss > log_matrix`](#train_losses_items_anyOf_i14_log_matrix)
        - [58.9.1.15.7. Property `ReduxOptions > train > losses > losses items > anyOf > ffloss > batch_matrix`](#train_losses_items_anyOf_i14_batch_matrix)
      - [58.9.1.16. Property `ReduxOptions > train > losses > losses items > anyOf > ldlloss`](#train_losses_items_anyOf_i15)
        - [58.9.1.16.1. Property `ReduxOptions > train > losses > losses items > anyOf > ldlloss > type`](#train_losses_items_anyOf_i15_type)
        - [58.9.1.16.2. Property `ReduxOptions > train > losses > losses items > anyOf > ldlloss > criterion`](#train_losses_items_anyOf_i15_criterion)
        - [58.9.1.16.3. Property `ReduxOptions > train > losses > losses items > anyOf > ldlloss > loss_weight`](#train_losses_items_anyOf_i15_loss_weight)
      - [58.9.1.17. Property `ReduxOptions > train > losses > losses items > anyOf > mssimloss`](#train_losses_items_anyOf_i16)
        - [58.9.1.17.1. Property `ReduxOptions > train > losses > losses items > anyOf > mssimloss > type`](#train_losses_items_anyOf_i16_type)
        - [58.9.1.17.2. Property `ReduxOptions > train > losses > losses items > anyOf > mssimloss > window_size`](#train_losses_items_anyOf_i16_window_size)
        - [58.9.1.17.3. Property `ReduxOptions > train > losses > losses items > anyOf > mssimloss > in_channels`](#train_losses_items_anyOf_i16_in_channels)
        - [58.9.1.17.4. Property `ReduxOptions > train > losses > losses items > anyOf > mssimloss > sigma`](#train_losses_items_anyOf_i16_sigma)
        - [58.9.1.17.5. Property `ReduxOptions > train > losses > losses items > anyOf > mssimloss > k1`](#train_losses_items_anyOf_i16_k1)
        - [58.9.1.17.6. Property `ReduxOptions > train > losses > losses items > anyOf > mssimloss > k2`](#train_losses_items_anyOf_i16_k2)
        - [58.9.1.17.7. Property `ReduxOptions > train > losses > losses items > anyOf > mssimloss > l`](#train_losses_items_anyOf_i16_l)
        - [58.9.1.17.8. Property `ReduxOptions > train > losses > losses items > anyOf > mssimloss > padding`](#train_losses_items_anyOf_i16_padding)
          - [58.9.1.17.8.1. Property `ReduxOptions > train > losses > losses items > anyOf > mssimloss > padding > anyOf > item 0`](#train_losses_items_anyOf_i16_padding_anyOf_i0)
          - [58.9.1.17.8.2. Property `ReduxOptions > train > losses > losses items > anyOf > mssimloss > padding > anyOf > item 1`](#train_losses_items_anyOf_i16_padding_anyOf_i1)
        - [58.9.1.17.9. Property `ReduxOptions > train > losses > losses items > anyOf > mssimloss > cosim`](#train_losses_items_anyOf_i16_cosim)
        - [58.9.1.17.10. Property `ReduxOptions > train > losses > losses items > anyOf > mssimloss > cosim_lambda`](#train_losses_items_anyOf_i16_cosim_lambda)
        - [58.9.1.17.11. Property `ReduxOptions > train > losses > losses items > anyOf > mssimloss > loss_weight`](#train_losses_items_anyOf_i16_loss_weight)
      - [58.9.1.18. Property `ReduxOptions > train > losses > losses items > anyOf > msssiml1loss`](#train_losses_items_anyOf_i17)
        - [58.9.1.18.1. Property `ReduxOptions > train > losses > losses items > anyOf > msssiml1loss > type`](#train_losses_items_anyOf_i17_type)
        - [58.9.1.18.2. Property `ReduxOptions > train > losses > losses items > anyOf > msssiml1loss > gaussian_sigmas`](#train_losses_items_anyOf_i17_gaussian_sigmas)
          - [58.9.1.18.2.1. Property `ReduxOptions > train > losses > losses items > anyOf > msssiml1loss > gaussian_sigmas > anyOf > item 0`](#train_losses_items_anyOf_i17_gaussian_sigmas_anyOf_i0)
            - [58.9.1.18.2.1.1. ReduxOptions > train > losses > losses items > anyOf > msssiml1loss > gaussian_sigmas > anyOf > item 0 > item 0 items](#train_losses_items_anyOf_i17_gaussian_sigmas_anyOf_i0_items)
          - [58.9.1.18.2.2. Property `ReduxOptions > train > losses > losses items > anyOf > msssiml1loss > gaussian_sigmas > anyOf > item 1`](#train_losses_items_anyOf_i17_gaussian_sigmas_anyOf_i1)
        - [58.9.1.18.3. Property `ReduxOptions > train > losses > losses items > anyOf > msssiml1loss > data_range`](#train_losses_items_anyOf_i17_data_range)
        - [58.9.1.18.4. Property `ReduxOptions > train > losses > losses items > anyOf > msssiml1loss > k`](#train_losses_items_anyOf_i17_k)
          - [58.9.1.18.4.1. ReduxOptions > train > losses > losses items > anyOf > msssiml1loss > k > k item 0](#autogenerated_heading_36)
          - [58.9.1.18.4.2. ReduxOptions > train > losses > losses items > anyOf > msssiml1loss > k > k item 1](#autogenerated_heading_37)
        - [58.9.1.18.5. Property `ReduxOptions > train > losses > losses items > anyOf > msssiml1loss > alpha`](#train_losses_items_anyOf_i17_alpha)
        - [58.9.1.18.6. Property `ReduxOptions > train > losses > losses items > anyOf > msssiml1loss > cuda_dev`](#train_losses_items_anyOf_i17_cuda_dev)
        - [58.9.1.18.7. Property `ReduxOptions > train > losses > losses items > anyOf > msssiml1loss > loss_weight`](#train_losses_items_anyOf_i17_loss_weight)
      - [58.9.1.19. Property `ReduxOptions > train > losses > losses items > anyOf > nccloss`](#train_losses_items_anyOf_i18)
        - [58.9.1.19.1. Property `ReduxOptions > train > losses > losses items > anyOf > nccloss > type`](#train_losses_items_anyOf_i18_type)
        - [58.9.1.19.2. Property `ReduxOptions > train > losses > losses items > anyOf > nccloss > loss_weight`](#train_losses_items_anyOf_i18_loss_weight)
      - [58.9.1.20. Property `ReduxOptions > train > losses > losses items > anyOf > perceptualfp16loss`](#train_losses_items_anyOf_i19)
        - [58.9.1.20.1. Property `ReduxOptions > train > losses > losses items > anyOf > perceptualfp16loss > type`](#train_losses_items_anyOf_i19_type)
        - [58.9.1.20.2. Property `ReduxOptions > train > losses > losses items > anyOf > perceptualfp16loss > layer_weights`](#train_losses_items_anyOf_i19_layer_weights)
          - [58.9.1.20.2.1. Property `ReduxOptions > train > losses > losses items > anyOf > perceptualfp16loss > layer_weights > anyOf > item 0`](#train_losses_items_anyOf_i19_layer_weights_anyOf_i0)
            - [58.9.1.20.2.1.1. Property `ReduxOptions > train > losses > losses items > anyOf > perceptualfp16loss > layer_weights > anyOf > item 0 > additionalProperties`](#train_losses_items_anyOf_i19_layer_weights_anyOf_i0_additionalProperties)
          - [58.9.1.20.2.2. Property `ReduxOptions > train > losses > losses items > anyOf > perceptualfp16loss > layer_weights > anyOf > item 1`](#train_losses_items_anyOf_i19_layer_weights_anyOf_i1)
        - [58.9.1.20.3. Property `ReduxOptions > train > losses > losses items > anyOf > perceptualfp16loss > w_lambda`](#train_losses_items_anyOf_i19_w_lambda)
        - [58.9.1.20.4. Property `ReduxOptions > train > losses > losses items > anyOf > perceptualfp16loss > loss_weight`](#train_losses_items_anyOf_i19_loss_weight)
        - [58.9.1.20.5. Property `ReduxOptions > train > losses > losses items > anyOf > perceptualfp16loss > alpha`](#train_losses_items_anyOf_i19_alpha)
          - [58.9.1.20.5.1. Property `ReduxOptions > train > losses > losses items > anyOf > perceptualfp16loss > alpha > anyOf > item 0`](#train_losses_items_anyOf_i19_alpha_anyOf_i0)
            - [58.9.1.20.5.1.1. ReduxOptions > train > losses > losses items > anyOf > perceptualfp16loss > alpha > anyOf > item 0 > item 0 items](#train_losses_items_anyOf_i19_alpha_anyOf_i0_items)
          - [58.9.1.20.5.2. Property `ReduxOptions > train > losses > losses items > anyOf > perceptualfp16loss > alpha > anyOf > item 1`](#train_losses_items_anyOf_i19_alpha_anyOf_i1)
        - [58.9.1.20.6. Property `ReduxOptions > train > losses > losses items > anyOf > perceptualfp16loss > criterion`](#train_losses_items_anyOf_i19_criterion)
        - [58.9.1.20.7. Property `ReduxOptions > train > losses > losses items > anyOf > perceptualfp16loss > num_proj_fd`](#train_losses_items_anyOf_i19_num_proj_fd)
        - [58.9.1.20.8. Property `ReduxOptions > train > losses > losses items > anyOf > perceptualfp16loss > phase_weight_fd`](#train_losses_items_anyOf_i19_phase_weight_fd)
        - [58.9.1.20.9. Property `ReduxOptions > train > losses > losses items > anyOf > perceptualfp16loss > stride_fd`](#train_losses_items_anyOf_i19_stride_fd)
      - [58.9.1.21. Property `ReduxOptions > train > losses > losses items > anyOf > perceptualloss`](#train_losses_items_anyOf_i20)
        - [58.9.1.21.1. Property `ReduxOptions > train > losses > losses items > anyOf > perceptualloss > type`](#train_losses_items_anyOf_i20_type)
        - [58.9.1.21.2. Property `ReduxOptions > train > losses > losses items > anyOf > perceptualloss > layer_weights`](#train_losses_items_anyOf_i20_layer_weights)
          - [58.9.1.21.2.1. Property `ReduxOptions > train > losses > losses items > anyOf > perceptualloss > layer_weights > anyOf > item 0`](#train_losses_items_anyOf_i20_layer_weights_anyOf_i0)
            - [58.9.1.21.2.1.1. Property `ReduxOptions > train > losses > losses items > anyOf > perceptualloss > layer_weights > anyOf > item 0 > additionalProperties`](#train_losses_items_anyOf_i20_layer_weights_anyOf_i0_additionalProperties)
          - [58.9.1.21.2.2. Property `ReduxOptions > train > losses > losses items > anyOf > perceptualloss > layer_weights > anyOf > item 1`](#train_losses_items_anyOf_i20_layer_weights_anyOf_i1)
        - [58.9.1.21.3. Property `ReduxOptions > train > losses > losses items > anyOf > perceptualloss > w_lambda`](#train_losses_items_anyOf_i20_w_lambda)
        - [58.9.1.21.4. Property `ReduxOptions > train > losses > losses items > anyOf > perceptualloss > loss_weight`](#train_losses_items_anyOf_i20_loss_weight)
        - [58.9.1.21.5. Property `ReduxOptions > train > losses > losses items > anyOf > perceptualloss > alpha`](#train_losses_items_anyOf_i20_alpha)
          - [58.9.1.21.5.1. Property `ReduxOptions > train > losses > losses items > anyOf > perceptualloss > alpha > anyOf > item 0`](#train_losses_items_anyOf_i20_alpha_anyOf_i0)
            - [58.9.1.21.5.1.1. ReduxOptions > train > losses > losses items > anyOf > perceptualloss > alpha > anyOf > item 0 > item 0 items](#train_losses_items_anyOf_i20_alpha_anyOf_i0_items)
          - [58.9.1.21.5.2. Property `ReduxOptions > train > losses > losses items > anyOf > perceptualloss > alpha > anyOf > item 1`](#train_losses_items_anyOf_i20_alpha_anyOf_i1)
        - [58.9.1.21.6. Property `ReduxOptions > train > losses > losses items > anyOf > perceptualloss > criterion`](#train_losses_items_anyOf_i20_criterion)
        - [58.9.1.21.7. Property `ReduxOptions > train > losses > losses items > anyOf > perceptualloss > num_proj_fd`](#train_losses_items_anyOf_i20_num_proj_fd)
        - [58.9.1.21.8. Property `ReduxOptions > train > losses > losses items > anyOf > perceptualloss > phase_weight_fd`](#train_losses_items_anyOf_i20_phase_weight_fd)
        - [58.9.1.21.9. Property `ReduxOptions > train > losses > losses items > anyOf > perceptualloss > stride_fd`](#train_losses_items_anyOf_i20_stride_fd)
      - [58.9.1.22. Property `ReduxOptions > train > losses > losses items > anyOf > topiqloss`](#train_losses_items_anyOf_i21)
        - [58.9.1.22.1. Property `ReduxOptions > train > losses > losses items > anyOf > topiqloss > type`](#train_losses_items_anyOf_i21_type)
        - [58.9.1.22.2. Property `ReduxOptions > train > losses > losses items > anyOf > topiqloss > loss_weight`](#train_losses_items_anyOf_i21_loss_weight)
        - [58.9.1.22.3. Property `ReduxOptions > train > losses > losses items > anyOf > topiqloss > resize_input`](#train_losses_items_anyOf_i21_resize_input)
  - [58.10. Property `ReduxOptions > train > pixel_opt`](#train_pixel_opt)
    - [58.10.1. Property `ReduxOptions > train > pixel_opt > anyOf > item 0`](#train_pixel_opt_anyOf_i0)
    - [58.10.2. Property `ReduxOptions > train > pixel_opt > anyOf > item 1`](#train_pixel_opt_anyOf_i1)
  - [58.11. Property `ReduxOptions > train > mssim_opt`](#train_mssim_opt)
    - [58.11.1. Property `ReduxOptions > train > mssim_opt > anyOf > item 0`](#train_mssim_opt_anyOf_i0)
    - [58.11.2. Property `ReduxOptions > train > mssim_opt > anyOf > item 1`](#train_mssim_opt_anyOf_i1)
  - [58.12. Property `ReduxOptions > train > ms_ssim_l1_opt`](#train_ms_ssim_l1_opt)
    - [58.12.1. Property `ReduxOptions > train > ms_ssim_l1_opt > anyOf > item 0`](#train_ms_ssim_l1_opt_anyOf_i0)
    - [58.12.2. Property `ReduxOptions > train > ms_ssim_l1_opt > anyOf > item 1`](#train_ms_ssim_l1_opt_anyOf_i1)
  - [58.13. Property `ReduxOptions > train > perceptual_opt`](#train_perceptual_opt)
    - [58.13.1. Property `ReduxOptions > train > perceptual_opt > anyOf > item 0`](#train_perceptual_opt_anyOf_i0)
    - [58.13.2. Property `ReduxOptions > train > perceptual_opt > anyOf > item 1`](#train_perceptual_opt_anyOf_i1)
  - [58.14. Property `ReduxOptions > train > contextual_opt`](#train_contextual_opt)
    - [58.14.1. Property `ReduxOptions > train > contextual_opt > anyOf > item 0`](#train_contextual_opt_anyOf_i0)
    - [58.14.2. Property `ReduxOptions > train > contextual_opt > anyOf > item 1`](#train_contextual_opt_anyOf_i1)
  - [58.15. Property `ReduxOptions > train > dists_opt`](#train_dists_opt)
    - [58.15.1. Property `ReduxOptions > train > dists_opt > anyOf > item 0`](#train_dists_opt_anyOf_i0)
    - [58.15.2. Property `ReduxOptions > train > dists_opt > anyOf > item 1`](#train_dists_opt_anyOf_i1)
  - [58.16. Property `ReduxOptions > train > hr_inversion_opt`](#train_hr_inversion_opt)
    - [58.16.1. Property `ReduxOptions > train > hr_inversion_opt > anyOf > item 0`](#train_hr_inversion_opt_anyOf_i0)
    - [58.16.2. Property `ReduxOptions > train > hr_inversion_opt > anyOf > item 1`](#train_hr_inversion_opt_anyOf_i1)
  - [58.17. Property `ReduxOptions > train > dinov2_opt`](#train_dinov2_opt)
    - [58.17.1. Property `ReduxOptions > train > dinov2_opt > anyOf > item 0`](#train_dinov2_opt_anyOf_i0)
    - [58.17.2. Property `ReduxOptions > train > dinov2_opt > anyOf > item 1`](#train_dinov2_opt_anyOf_i1)
  - [58.18. Property `ReduxOptions > train > topiq_opt`](#train_topiq_opt)
    - [58.18.1. Property `ReduxOptions > train > topiq_opt > anyOf > item 0`](#train_topiq_opt_anyOf_i0)
    - [58.18.2. Property `ReduxOptions > train > topiq_opt > anyOf > item 1`](#train_topiq_opt_anyOf_i1)
  - [58.19. Property `ReduxOptions > train > pd_opt`](#train_pd_opt)
    - [58.19.1. Property `ReduxOptions > train > pd_opt > anyOf > item 0`](#train_pd_opt_anyOf_i0)
    - [58.19.2. Property `ReduxOptions > train > pd_opt > anyOf > item 1`](#train_pd_opt_anyOf_i1)
  - [58.20. Property `ReduxOptions > train > fd_opt`](#train_fd_opt)
    - [58.20.1. Property `ReduxOptions > train > fd_opt > anyOf > item 0`](#train_fd_opt_anyOf_i0)
    - [58.20.2. Property `ReduxOptions > train > fd_opt > anyOf > item 1`](#train_fd_opt_anyOf_i1)
  - [58.21. Property `ReduxOptions > train > ldl_opt`](#train_ldl_opt)
    - [58.21.1. Property `ReduxOptions > train > ldl_opt > anyOf > item 0`](#train_ldl_opt_anyOf_i0)
    - [58.21.2. Property `ReduxOptions > train > ldl_opt > anyOf > item 1`](#train_ldl_opt_anyOf_i1)
  - [58.22. Property `ReduxOptions > train > hsluv_opt`](#train_hsluv_opt)
    - [58.22.1. Property `ReduxOptions > train > hsluv_opt > anyOf > item 0`](#train_hsluv_opt_anyOf_i0)
    - [58.22.2. Property `ReduxOptions > train > hsluv_opt > anyOf > item 1`](#train_hsluv_opt_anyOf_i1)
  - [58.23. Property `ReduxOptions > train > gan_opt`](#train_gan_opt)
    - [58.23.1. Property `ReduxOptions > train > gan_opt > anyOf > item 0`](#train_gan_opt_anyOf_i0)
    - [58.23.2. Property `ReduxOptions > train > gan_opt > anyOf > item 1`](#train_gan_opt_anyOf_i1)
  - [58.24. Property `ReduxOptions > train > color_opt`](#train_color_opt)
    - [58.24.1. Property `ReduxOptions > train > color_opt > anyOf > item 0`](#train_color_opt_anyOf_i0)
    - [58.24.2. Property `ReduxOptions > train > color_opt > anyOf > item 1`](#train_color_opt_anyOf_i1)
  - [58.25. Property `ReduxOptions > train > luma_opt`](#train_luma_opt)
    - [58.25.1. Property `ReduxOptions > train > luma_opt > anyOf > item 0`](#train_luma_opt_anyOf_i0)
    - [58.25.2. Property `ReduxOptions > train > luma_opt > anyOf > item 1`](#train_luma_opt_anyOf_i1)
  - [58.26. Property `ReduxOptions > train > avg_opt`](#train_avg_opt)
    - [58.26.1. Property `ReduxOptions > train > avg_opt > anyOf > item 0`](#train_avg_opt_anyOf_i0)
    - [58.26.2. Property `ReduxOptions > train > avg_opt > anyOf > item 1`](#train_avg_opt_anyOf_i1)
  - [58.27. Property `ReduxOptions > train > bicubic_opt`](#train_bicubic_opt)
    - [58.27.1. Property `ReduxOptions > train > bicubic_opt > anyOf > item 0`](#train_bicubic_opt_anyOf_i0)
    - [58.27.2. Property `ReduxOptions > train > bicubic_opt > anyOf > item 1`](#train_bicubic_opt_anyOf_i1)
  - [58.28. Property `ReduxOptions > train > use_moa`](#train_use_moa)
  - [58.29. Property `ReduxOptions > train > moa_augs`](#train_moa_augs)
    - [58.29.1. ReduxOptions > train > moa_augs > moa_augs items](#train_moa_augs_items)
  - [58.30. Property `ReduxOptions > train > moa_probs`](#train_moa_probs)
    - [58.30.1. ReduxOptions > train > moa_probs > moa_probs items](#train_moa_probs_items)
  - [58.31. Property `ReduxOptions > train > moa_debug`](#train_moa_debug)
  - [58.32. Property `ReduxOptions > train > moa_debug_limit`](#train_moa_debug_limit)
- [59. Property `ReduxOptions > val`](#val)
  - [59.1. Property `ReduxOptions > val > anyOf > item 0`](#val_anyOf_i0)
  - [59.2. Property `ReduxOptions > val > anyOf > ValOptions`](#val_anyOf_i1)
    - [59.2.1. Property `ReduxOptions > val > anyOf > ValOptions > val_enabled`](#val_anyOf_i1_val_enabled)
    - [59.2.2. Property `ReduxOptions > val > anyOf > ValOptions > save_img`](#val_anyOf_i1_save_img)
    - [59.2.3. Property `ReduxOptions > val > anyOf > ValOptions > val_freq`](#val_anyOf_i1_val_freq)
      - [59.2.3.1. Property `ReduxOptions > val > anyOf > ValOptions > val_freq > anyOf > item 0`](#val_anyOf_i1_val_freq_anyOf_i0)
      - [59.2.3.2. Property `ReduxOptions > val > anyOf > ValOptions > val_freq > anyOf > item 1`](#val_anyOf_i1_val_freq_anyOf_i1)
    - [59.2.4. Property `ReduxOptions > val > anyOf > ValOptions > suffix`](#val_anyOf_i1_suffix)
      - [59.2.4.1. Property `ReduxOptions > val > anyOf > ValOptions > suffix > anyOf > item 0`](#val_anyOf_i1_suffix_anyOf_i0)
      - [59.2.4.2. Property `ReduxOptions > val > anyOf > ValOptions > suffix > anyOf > item 1`](#val_anyOf_i1_suffix_anyOf_i1)
    - [59.2.5. Property `ReduxOptions > val > anyOf > ValOptions > metrics_enabled`](#val_anyOf_i1_metrics_enabled)
    - [59.2.6. Property `ReduxOptions > val > anyOf > ValOptions > metrics`](#val_anyOf_i1_metrics)
      - [59.2.6.1. Property `ReduxOptions > val > anyOf > ValOptions > metrics > anyOf > item 0`](#val_anyOf_i1_metrics_anyOf_i0)
      - [59.2.6.2. Property `ReduxOptions > val > anyOf > ValOptions > metrics > anyOf > item 1`](#val_anyOf_i1_metrics_anyOf_i1)
    - [59.2.7. Property `ReduxOptions > val > anyOf > ValOptions > pbar`](#val_anyOf_i1_pbar)
- [60. Property `ReduxOptions > logger`](#logger)
  - [60.1. Property `ReduxOptions > logger > anyOf > item 0`](#logger_anyOf_i0)
  - [60.2. Property `ReduxOptions > logger > anyOf > LogOptions`](#logger_anyOf_i1)
    - [60.2.1. Property `ReduxOptions > logger > anyOf > LogOptions > print_freq`](#logger_anyOf_i1_print_freq)
    - [60.2.2. Property `ReduxOptions > logger > anyOf > LogOptions > save_checkpoint_freq`](#logger_anyOf_i1_save_checkpoint_freq)
    - [60.2.3. Property `ReduxOptions > logger > anyOf > LogOptions > use_tb_logger`](#logger_anyOf_i1_use_tb_logger)
    - [60.2.4. Property `ReduxOptions > logger > anyOf > LogOptions > save_checkpoint_format`](#logger_anyOf_i1_save_checkpoint_format)
    - [60.2.5. Property `ReduxOptions > logger > anyOf > LogOptions > wandb`](#logger_anyOf_i1_wandb)
      - [60.2.5.1. Property `ReduxOptions > logger > anyOf > LogOptions > wandb > anyOf > item 0`](#logger_anyOf_i1_wandb_anyOf_i0)
      - [60.2.5.2. Property `ReduxOptions > logger > anyOf > LogOptions > wandb > anyOf > WandbOptions`](#logger_anyOf_i1_wandb_anyOf_i1)
        - [60.2.5.2.1. Property `ReduxOptions > logger > anyOf > LogOptions > wandb > anyOf > WandbOptions > resume_id`](#logger_anyOf_i1_wandb_anyOf_i1_resume_id)
          - [60.2.5.2.1.1. Property `ReduxOptions > logger > anyOf > LogOptions > wandb > anyOf > WandbOptions > resume_id > anyOf > item 0`](#logger_anyOf_i1_wandb_anyOf_i1_resume_id_anyOf_i0)
          - [60.2.5.2.1.2. Property `ReduxOptions > logger > anyOf > LogOptions > wandb > anyOf > WandbOptions > resume_id > anyOf > item 1`](#logger_anyOf_i1_wandb_anyOf_i1_resume_id_anyOf_i1)
        - [60.2.5.2.2. Property `ReduxOptions > logger > anyOf > LogOptions > wandb > anyOf > WandbOptions > project`](#logger_anyOf_i1_wandb_anyOf_i1_project)
          - [60.2.5.2.2.1. Property `ReduxOptions > logger > anyOf > LogOptions > wandb > anyOf > WandbOptions > project > anyOf > item 0`](#logger_anyOf_i1_wandb_anyOf_i1_project_anyOf_i0)
          - [60.2.5.2.2.2. Property `ReduxOptions > logger > anyOf > LogOptions > wandb > anyOf > WandbOptions > project > anyOf > item 1`](#logger_anyOf_i1_wandb_anyOf_i1_project_anyOf_i1)
- [61. Property `ReduxOptions > dist_params`](#dist_params)
  - [61.1. Property `ReduxOptions > dist_params > anyOf > item 0`](#dist_params_anyOf_i0)
  - [61.2. Property `ReduxOptions > dist_params > anyOf > item 1`](#dist_params_anyOf_i1)
- [62. Property `ReduxOptions > onnx`](#onnx)
  - [62.1. Property `ReduxOptions > onnx > anyOf > item 0`](#onnx_anyOf_i0)
  - [62.2. Property `ReduxOptions > onnx > anyOf > OnnxOptions`](#onnx_anyOf_i1)
    - [62.2.1. Property `ReduxOptions > onnx > anyOf > OnnxOptions > dynamo`](#onnx_anyOf_i1_dynamo)
    - [62.2.2. Property `ReduxOptions > onnx > anyOf > OnnxOptions > opset`](#onnx_anyOf_i1_opset)
    - [62.2.3. Property `ReduxOptions > onnx > anyOf > OnnxOptions > use_static_shapes`](#onnx_anyOf_i1_use_static_shapes)
    - [62.2.4. Property `ReduxOptions > onnx > anyOf > OnnxOptions > shape`](#onnx_anyOf_i1_shape)
    - [62.2.5. Property `ReduxOptions > onnx > anyOf > OnnxOptions > verify`](#onnx_anyOf_i1_verify)
    - [62.2.6. Property `ReduxOptions > onnx > anyOf > OnnxOptions > fp16`](#onnx_anyOf_i1_fp16)
    - [62.2.7. Property `ReduxOptions > onnx > anyOf > OnnxOptions > optimize`](#onnx_anyOf_i1_optimize)
- [63. Property `ReduxOptions > find_unused_parameters`](#find_unused_parameters)

**Title:** ReduxOptions

|                           |                      |
| ------------------------- | -------------------- |
| **Type**                  | `object`             |
| **Required**              | No                   |
| **Additional properties** | Any type allowed     |
| **Defined in**            | #/$defs/ReduxOptions |

| Property                                                                                       | Pattern | Type             | Deprecated | Definition              | Title/Description                                                                                                                                                                                                                                                                                         |
| ---------------------------------------------------------------------------------------------- | ------- | ---------------- | ---------- | ----------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| + [type](#type )                                                                               | No      | enum (of string) | No         | -                       | -                                                                                                                                                                                                                                                                                                         |
| + [name](#name )                                                                               | No      | string           | No         | -                       | Name of the experiment. It should be a unique name. If you enable auto resume, any experiment with this name will be resumed instead of starting a new training run.                                                                                                                                      |
| + [scale](#scale )                                                                             | No      | integer          | No         | -                       | Scale of the model. Most architectures support a scale of 1, 2, 3, 4, or 8. A scale of 1 can be used for restoration models that don't change the resolution of the input image. A scale of 2 means the width and height of the input image are doubled, so a 640x480 input will be upscaled to 1280x960. |
| + [num_gpu](#num_gpu )                                                                         | No      | Combination      | No         | -                       | The number of GPUs to use for training, if using multiple GPUs.                                                                                                                                                                                                                                           |
| + [path](#path )                                                                               | No      | object           | No         | In #/$defs/PathOptions  | PathOptions                                                                                                                                                                                                                                                                                               |
| + [network_g](#network_g )                                                                     | No      | object           | No         | -                       | The options for the generator model.                                                                                                                                                                                                                                                                      |
| + [network_d](#network_d )                                                                     | No      | Combination      | No         | -                       | The options for the discriminator model.                                                                                                                                                                                                                                                                  |
| + [manual_seed](#manual_seed )                                                                 | No      | Combination      | No         | -                       | Deterministic mode, slows down training. Only use for reproducible experiments.                                                                                                                                                                                                                           |
| + [deterministic](#deterministic )                                                             | No      | Combination      | No         | -                       | -                                                                                                                                                                                                                                                                                                         |
| + [dist](#dist )                                                                               | No      | Combination      | No         | -                       | -                                                                                                                                                                                                                                                                                                         |
| + [launcher](#launcher )                                                                       | No      | Combination      | No         | -                       | -                                                                                                                                                                                                                                                                                                         |
| + [rank](#rank )                                                                               | No      | Combination      | No         | -                       | -                                                                                                                                                                                                                                                                                                         |
| + [world_size](#world_size )                                                                   | No      | Combination      | No         | -                       | -                                                                                                                                                                                                                                                                                                         |
| + [auto_resume](#auto_resume )                                                                 | No      | Combination      | No         | -                       | -                                                                                                                                                                                                                                                                                                         |
| + [resume](#resume )                                                                           | No      | integer          | No         | -                       | -                                                                                                                                                                                                                                                                                                         |
| + [is_train](#is_train )                                                                       | No      | Combination      | No         | -                       | -                                                                                                                                                                                                                                                                                                         |
| + [root_path](#root_path )                                                                     | No      | Combination      | No         | -                       | -                                                                                                                                                                                                                                                                                                         |
| + [use_amp](#use_amp )                                                                         | No      | boolean          | No         | -                       | Speed up training and reduce VRAM usage. NVIDIA only.                                                                                                                                                                                                                                                     |
| + [amp_bf16](#amp_bf16 )                                                                       | No      | boolean          | No         | -                       | Use bf16 instead of fp16 for AMP, RTX 3000 series or newer only. Only recommended if fp16 doesn't work.                                                                                                                                                                                                   |
| + [use_channels_last](#use_channels_last )                                                     | No      | boolean          | No         | -                       | Enable channels last memory format while using AMP. Reduces VRAM and speeds up training for most architectures, but some architectures are slower with channels last.                                                                                                                                     |
| + [fast_matmul](#fast_matmul )                                                                 | No      | boolean          | No         | -                       | Trade precision for performance.                                                                                                                                                                                                                                                                          |
| + [use_compile](#use_compile )                                                                 | No      | boolean          | No         | -                       | Enable torch.compile for the generator model, which takes time on startup to compile the model, but can speed up training after the model is compiled.                                                                                                                                                    |
| + [detect_anomaly](#detect_anomaly )                                                           | No      | boolean          | No         | -                       | Whether or not to enable anomaly detection, which can be useful for debugging NaNs that occur during training. Has a significant performance hit and should be disabled when not debugging.                                                                                                               |
| + [high_order_degradation](#high_order_degradation )                                           | No      | boolean          | No         | -                       | Whether or not to enable OTF (on the fly) degradations, which generates LRs on the fly.                                                                                                                                                                                                                   |
| + [high_order_degradations_debug](#high_order_degradations_debug )                             | No      | boolean          | No         | -                       | Whether or not to enable debugging for OTF, which saves the OTF generated LR images so they can be inspected to view the effect of different OTF settings.                                                                                                                                                |
| + [high_order_degradations_debug_limit](#high_order_degradations_debug_limit )                 | No      | integer          | No         | -                       | The maximum number of OTF images to save when debugging is enabled.                                                                                                                                                                                                                                       |
| + [dataroot_lq_prob](#dataroot_lq_prob )                                                       | No      | number           | No         | -                       | Probability of using paired LR data instead of OTF LR data.                                                                                                                                                                                                                                               |
| + [force_high_order_degradation_filename_masks](#force_high_order_degradation_filename_masks ) | No      | array of string  | No         | -                       | -                                                                                                                                                                                                                                                                                                         |
| + [force_dataroot_lq_filename_masks](#force_dataroot_lq_filename_masks )                       | No      | array of string  | No         | -                       | -                                                                                                                                                                                                                                                                                                         |
| + [lq_usm](#lq_usm )                                                                           | No      | boolean          | No         | -                       | -                                                                                                                                                                                                                                                                                                         |
| + [lq_usm_radius_range](#lq_usm_radius_range )                                                 | No      | array            | No         | -                       | For the unsharp mask of the LQ image, use a radius randomly selected from this range.                                                                                                                                                                                                                     |
| + [blur_prob](#blur_prob )                                                                     | No      | number           | No         | -                       | -                                                                                                                                                                                                                                                                                                         |
| + [resize_prob](#resize_prob )                                                                 | No      | array of number  | No         | -                       | -                                                                                                                                                                                                                                                                                                         |
| + [resize_mode_list](#resize_mode_list )                                                       | No      | array of string  | No         | -                       | -                                                                                                                                                                                                                                                                                                         |
| + [resize_mode_prob](#resize_mode_prob )                                                       | No      | array of number  | No         | -                       | -                                                                                                                                                                                                                                                                                                         |
| + [resize_range](#resize_range )                                                               | No      | array            | No         | -                       | -                                                                                                                                                                                                                                                                                                         |
| + [gaussian_noise_prob](#gaussian_noise_prob )                                                 | No      | number           | No         | -                       | -                                                                                                                                                                                                                                                                                                         |
| + [noise_range](#noise_range )                                                                 | No      | array            | No         | -                       | -                                                                                                                                                                                                                                                                                                         |
| + [poisson_scale_range](#poisson_scale_range )                                                 | No      | array            | No         | -                       | -                                                                                                                                                                                                                                                                                                         |
| + [gray_noise_prob](#gray_noise_prob )                                                         | No      | number           | No         | -                       | -                                                                                                                                                                                                                                                                                                         |
| + [jpeg_prob](#jpeg_prob )                                                                     | No      | number           | No         | -                       | -                                                                                                                                                                                                                                                                                                         |
| + [jpeg_range](#jpeg_range )                                                                   | No      | array            | No         | -                       | -                                                                                                                                                                                                                                                                                                         |
| + [blur_prob2](#blur_prob2 )                                                                   | No      | number           | No         | -                       | -                                                                                                                                                                                                                                                                                                         |
| + [resize_prob2](#resize_prob2 )                                                               | No      | array of number  | No         | -                       | -                                                                                                                                                                                                                                                                                                         |
| + [resize_mode_list2](#resize_mode_list2 )                                                     | No      | array of string  | No         | -                       | -                                                                                                                                                                                                                                                                                                         |
| + [resize_mode_prob2](#resize_mode_prob2 )                                                     | No      | array of number  | No         | -                       | -                                                                                                                                                                                                                                                                                                         |
| + [resize_range2](#resize_range2 )                                                             | No      | array            | No         | -                       | -                                                                                                                                                                                                                                                                                                         |
| + [gaussian_noise_prob2](#gaussian_noise_prob2 )                                               | No      | number           | No         | -                       | -                                                                                                                                                                                                                                                                                                         |
| + [noise_range2](#noise_range2 )                                                               | No      | array            | No         | -                       | -                                                                                                                                                                                                                                                                                                         |
| + [poisson_scale_range2](#poisson_scale_range2 )                                               | No      | array            | No         | -                       | -                                                                                                                                                                                                                                                                                                         |
| + [gray_noise_prob2](#gray_noise_prob2 )                                                       | No      | number           | No         | -                       | -                                                                                                                                                                                                                                                                                                         |
| + [jpeg_prob2](#jpeg_prob2 )                                                                   | No      | number           | No         | -                       | -                                                                                                                                                                                                                                                                                                         |
| + [jpeg_range2](#jpeg_range2 )                                                                 | No      | array of number  | No         | -                       | -                                                                                                                                                                                                                                                                                                         |
| + [resize_mode_list3](#resize_mode_list3 )                                                     | No      | array of string  | No         | -                       | -                                                                                                                                                                                                                                                                                                         |
| + [resize_mode_prob3](#resize_mode_prob3 )                                                     | No      | array of number  | No         | -                       | -                                                                                                                                                                                                                                                                                                         |
| + [queue_size](#queue_size )                                                                   | No      | integer          | No         | -                       | -                                                                                                                                                                                                                                                                                                         |
| + [datasets](#datasets )                                                                       | No      | object           | No         | -                       | -                                                                                                                                                                                                                                                                                                         |
| + [train](#train )                                                                             | No      | object           | No         | In #/$defs/TrainOptions | TrainOptions                                                                                                                                                                                                                                                                                              |
| + [val](#val )                                                                                 | No      | Combination      | No         | -                       | -                                                                                                                                                                                                                                                                                                         |
| + [logger](#logger )                                                                           | No      | Combination      | No         | -                       | -                                                                                                                                                                                                                                                                                                         |
| + [dist_params](#dist_params )                                                                 | No      | Combination      | No         | -                       | -                                                                                                                                                                                                                                                                                                         |
| + [onnx](#onnx )                                                                               | No      | Combination      | No         | -                       | -                                                                                                                                                                                                                                                                                                         |
| + [find_unused_parameters](#find_unused_parameters )                                           | No      | boolean          | No         | -                       | -                                                                                                                                                                                                                                                                                                         |

## <a name="type"></a>1. Property `ReduxOptions > type`

|              |                    |
| ------------ | ------------------ |
| **Type**     | `enum (of string)` |
| **Required** | Yes                |

Must be one of:
* "ReduxOptions"

## <a name="name"></a>2. Property `ReduxOptions > name`

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | Yes      |

**Description:** Name of the experiment. It should be a unique name. If you enable auto resume, any experiment with this name will be resumed instead of starting a new training run.

## <a name="scale"></a>3. Property `ReduxOptions > scale`

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | Yes       |

**Description:** Scale of the model. Most architectures support a scale of 1, 2, 3, 4, or 8. A scale of 1 can be used for restoration models that don't change the resolution of the input image. A scale of 2 means the width and height of the input image are doubled, so a 640x480 input will be upscaled to 1280x960.

## <a name="num_gpu"></a>4. Property `ReduxOptions > num_gpu`

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

### <a name="num_gpu_anyOf_i0"></a>4.1. Property `ReduxOptions > num_gpu > anyOf > item 0`

|              |                    |
| ------------ | ------------------ |
| **Type**     | `enum (of string)` |
| **Required** | No                 |

Must be one of:
* "auto"

### <a name="num_gpu_anyOf_i1"></a>4.2. Property `ReduxOptions > num_gpu > anyOf > item 1`

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | No        |

## <a name="path"></a>5. Property `ReduxOptions > path`

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

### <a name="path_experiments_root"></a>5.1. Property `ReduxOptions > path > experiments_root`

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

#### <a name="path_experiments_root_anyOf_i0"></a>5.1.1. Property `ReduxOptions > path > experiments_root > anyOf > item 0`

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | No       |

#### <a name="path_experiments_root_anyOf_i1"></a>5.1.2. Property `ReduxOptions > path > experiments_root > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

### <a name="path_models"></a>5.2. Property `ReduxOptions > path > models`

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

#### <a name="path_models_anyOf_i0"></a>5.2.1. Property `ReduxOptions > path > models > anyOf > item 0`

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | No       |

#### <a name="path_models_anyOf_i1"></a>5.2.2. Property `ReduxOptions > path > models > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

### <a name="path_resume_models"></a>5.3. Property `ReduxOptions > path > resume_models`

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

#### <a name="path_resume_models_anyOf_i0"></a>5.3.1. Property `ReduxOptions > path > resume_models > anyOf > item 0`

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | No       |

#### <a name="path_resume_models_anyOf_i1"></a>5.3.2. Property `ReduxOptions > path > resume_models > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

### <a name="path_training_states"></a>5.4. Property `ReduxOptions > path > training_states`

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

#### <a name="path_training_states_anyOf_i0"></a>5.4.1. Property `ReduxOptions > path > training_states > anyOf > item 0`

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | No       |

#### <a name="path_training_states_anyOf_i1"></a>5.4.2. Property `ReduxOptions > path > training_states > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

### <a name="path_log"></a>5.5. Property `ReduxOptions > path > log`

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

#### <a name="path_log_anyOf_i0"></a>5.5.1. Property `ReduxOptions > path > log > anyOf > item 0`

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | No       |

#### <a name="path_log_anyOf_i1"></a>5.5.2. Property `ReduxOptions > path > log > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

### <a name="path_visualization"></a>5.6. Property `ReduxOptions > path > visualization`

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

#### <a name="path_visualization_anyOf_i0"></a>5.6.1. Property `ReduxOptions > path > visualization > anyOf > item 0`

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | No       |

#### <a name="path_visualization_anyOf_i1"></a>5.6.2. Property `ReduxOptions > path > visualization > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

### <a name="path_results_root"></a>5.7. Property `ReduxOptions > path > results_root`

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

#### <a name="path_results_root_anyOf_i0"></a>5.7.1. Property `ReduxOptions > path > results_root > anyOf > item 0`

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | No       |

#### <a name="path_results_root_anyOf_i1"></a>5.7.2. Property `ReduxOptions > path > results_root > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

### <a name="path_pretrain_network_g"></a>5.8. Property `ReduxOptions > path > pretrain_network_g`

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

#### <a name="path_pretrain_network_g_anyOf_i0"></a>5.8.1. Property `ReduxOptions > path > pretrain_network_g > anyOf > item 0`

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | No       |

#### <a name="path_pretrain_network_g_anyOf_i1"></a>5.8.2. Property `ReduxOptions > path > pretrain_network_g > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

### <a name="path_pretrain_network_g_path"></a>5.9. Property `ReduxOptions > path > pretrain_network_g_path`

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

#### <a name="path_pretrain_network_g_path_anyOf_i0"></a>5.9.1. Property `ReduxOptions > path > pretrain_network_g_path > anyOf > item 0`

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | No       |

#### <a name="path_pretrain_network_g_path_anyOf_i1"></a>5.9.2. Property `ReduxOptions > path > pretrain_network_g_path > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

### <a name="path_param_key_g"></a>5.10. Property `ReduxOptions > path > param_key_g`

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

#### <a name="path_param_key_g_anyOf_i0"></a>5.10.1. Property `ReduxOptions > path > param_key_g > anyOf > item 0`

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | No       |

#### <a name="path_param_key_g_anyOf_i1"></a>5.10.2. Property `ReduxOptions > path > param_key_g > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

### <a name="path_strict_load_g"></a>5.11. Property `ReduxOptions > path > strict_load_g`

|              |           |
| ------------ | --------- |
| **Type**     | `boolean` |
| **Required** | No        |
| **Default**  | `true`    |

### <a name="path_resume_state"></a>5.12. Property `ReduxOptions > path > resume_state`

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

#### <a name="path_resume_state_anyOf_i0"></a>5.12.1. Property `ReduxOptions > path > resume_state > anyOf > item 0`

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | No       |

#### <a name="path_resume_state_anyOf_i1"></a>5.12.2. Property `ReduxOptions > path > resume_state > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

### <a name="path_pretrain_network_g_ema"></a>5.13. Property `ReduxOptions > path > pretrain_network_g_ema`

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

#### <a name="path_pretrain_network_g_ema_anyOf_i0"></a>5.13.1. Property `ReduxOptions > path > pretrain_network_g_ema > anyOf > item 0`

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | No       |

#### <a name="path_pretrain_network_g_ema_anyOf_i1"></a>5.13.2. Property `ReduxOptions > path > pretrain_network_g_ema > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

### <a name="path_pretrain_network_d"></a>5.14. Property `ReduxOptions > path > pretrain_network_d`

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

#### <a name="path_pretrain_network_d_anyOf_i0"></a>5.14.1. Property `ReduxOptions > path > pretrain_network_d > anyOf > item 0`

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | No       |

#### <a name="path_pretrain_network_d_anyOf_i1"></a>5.14.2. Property `ReduxOptions > path > pretrain_network_d > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

### <a name="path_param_key_d"></a>5.15. Property `ReduxOptions > path > param_key_d`

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

#### <a name="path_param_key_d_anyOf_i0"></a>5.15.1. Property `ReduxOptions > path > param_key_d > anyOf > item 0`

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | No       |

#### <a name="path_param_key_d_anyOf_i1"></a>5.15.2. Property `ReduxOptions > path > param_key_d > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

### <a name="path_strict_load_d"></a>5.16. Property `ReduxOptions > path > strict_load_d`

|              |           |
| ------------ | --------- |
| **Type**     | `boolean` |
| **Required** | No        |
| **Default**  | `true`    |

### <a name="path_ignore_resume_networks"></a>5.17. Property `ReduxOptions > path > ignore_resume_networks`

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

#### <a name="path_ignore_resume_networks_anyOf_i0"></a>5.17.1. Property `ReduxOptions > path > ignore_resume_networks > anyOf > item 0`

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

##### <a name="path_ignore_resume_networks_anyOf_i0_items"></a>5.17.1.1. ReduxOptions > path > ignore_resume_networks > anyOf > item 0 > item 0 items

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | No       |

#### <a name="path_ignore_resume_networks_anyOf_i1"></a>5.17.2. Property `ReduxOptions > path > ignore_resume_networks > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

## <a name="network_g"></a>6. Property `ReduxOptions > network_g`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `object`         |
| **Required**              | Yes              |
| **Additional properties** | Any type allowed |

**Description:** The options for the generator model.

## <a name="network_d"></a>7. Property `ReduxOptions > network_d`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `combining`      |
| **Required**              | Yes              |
| **Additional properties** | Any type allowed |

**Description:** The options for the discriminator model.

| Any of(Option)                |
| ----------------------------- |
| [item 0](#network_d_anyOf_i0) |
| [item 1](#network_d_anyOf_i1) |

### <a name="network_d_anyOf_i0"></a>7.1. Property `ReduxOptions > network_d > anyOf > item 0`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `object`         |
| **Required**              | No               |
| **Additional properties** | Any type allowed |

### <a name="network_d_anyOf_i1"></a>7.2. Property `ReduxOptions > network_d > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

## <a name="manual_seed"></a>8. Property `ReduxOptions > manual_seed`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `combining`      |
| **Required**              | Yes              |
| **Additional properties** | Any type allowed |

**Description:** Deterministic mode, slows down training. Only use for reproducible experiments.

| Any of(Option)                  |
| ------------------------------- |
| [item 0](#manual_seed_anyOf_i0) |
| [item 1](#manual_seed_anyOf_i1) |

### <a name="manual_seed_anyOf_i0"></a>8.1. Property `ReduxOptions > manual_seed > anyOf > item 0`

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | No        |

### <a name="manual_seed_anyOf_i1"></a>8.2. Property `ReduxOptions > manual_seed > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

## <a name="deterministic"></a>9. Property `ReduxOptions > deterministic`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `combining`      |
| **Required**              | Yes              |
| **Additional properties** | Any type allowed |

| Any of(Option)                    |
| --------------------------------- |
| [item 0](#deterministic_anyOf_i0) |
| [item 1](#deterministic_anyOf_i1) |

### <a name="deterministic_anyOf_i0"></a>9.1. Property `ReduxOptions > deterministic > anyOf > item 0`

|              |           |
| ------------ | --------- |
| **Type**     | `boolean` |
| **Required** | No        |

### <a name="deterministic_anyOf_i1"></a>9.2. Property `ReduxOptions > deterministic > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

## <a name="dist"></a>10. Property `ReduxOptions > dist`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `combining`      |
| **Required**              | Yes              |
| **Additional properties** | Any type allowed |

| Any of(Option)           |
| ------------------------ |
| [item 0](#dist_anyOf_i0) |
| [item 1](#dist_anyOf_i1) |

### <a name="dist_anyOf_i0"></a>10.1. Property `ReduxOptions > dist > anyOf > item 0`

|              |           |
| ------------ | --------- |
| **Type**     | `boolean` |
| **Required** | No        |

### <a name="dist_anyOf_i1"></a>10.2. Property `ReduxOptions > dist > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

## <a name="launcher"></a>11. Property `ReduxOptions > launcher`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `combining`      |
| **Required**              | Yes              |
| **Additional properties** | Any type allowed |

| Any of(Option)               |
| ---------------------------- |
| [item 0](#launcher_anyOf_i0) |
| [item 1](#launcher_anyOf_i1) |

### <a name="launcher_anyOf_i0"></a>11.1. Property `ReduxOptions > launcher > anyOf > item 0`

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | No       |

### <a name="launcher_anyOf_i1"></a>11.2. Property `ReduxOptions > launcher > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

## <a name="rank"></a>12. Property `ReduxOptions > rank`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `combining`      |
| **Required**              | Yes              |
| **Additional properties** | Any type allowed |

| Any of(Option)           |
| ------------------------ |
| [item 0](#rank_anyOf_i0) |
| [item 1](#rank_anyOf_i1) |

### <a name="rank_anyOf_i0"></a>12.1. Property `ReduxOptions > rank > anyOf > item 0`

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | No        |

### <a name="rank_anyOf_i1"></a>12.2. Property `ReduxOptions > rank > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

## <a name="world_size"></a>13. Property `ReduxOptions > world_size`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `combining`      |
| **Required**              | Yes              |
| **Additional properties** | Any type allowed |

| Any of(Option)                 |
| ------------------------------ |
| [item 0](#world_size_anyOf_i0) |
| [item 1](#world_size_anyOf_i1) |

### <a name="world_size_anyOf_i0"></a>13.1. Property `ReduxOptions > world_size > anyOf > item 0`

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | No        |

### <a name="world_size_anyOf_i1"></a>13.2. Property `ReduxOptions > world_size > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

## <a name="auto_resume"></a>14. Property `ReduxOptions > auto_resume`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `combining`      |
| **Required**              | Yes              |
| **Additional properties** | Any type allowed |

| Any of(Option)                  |
| ------------------------------- |
| [item 0](#auto_resume_anyOf_i0) |
| [item 1](#auto_resume_anyOf_i1) |

### <a name="auto_resume_anyOf_i0"></a>14.1. Property `ReduxOptions > auto_resume > anyOf > item 0`

|              |           |
| ------------ | --------- |
| **Type**     | `boolean` |
| **Required** | No        |

### <a name="auto_resume_anyOf_i1"></a>14.2. Property `ReduxOptions > auto_resume > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

## <a name="resume"></a>15. Property `ReduxOptions > resume`

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | Yes       |

## <a name="is_train"></a>16. Property `ReduxOptions > is_train`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `combining`      |
| **Required**              | Yes              |
| **Additional properties** | Any type allowed |

| Any of(Option)               |
| ---------------------------- |
| [item 0](#is_train_anyOf_i0) |
| [item 1](#is_train_anyOf_i1) |

### <a name="is_train_anyOf_i0"></a>16.1. Property `ReduxOptions > is_train > anyOf > item 0`

|              |           |
| ------------ | --------- |
| **Type**     | `boolean` |
| **Required** | No        |

### <a name="is_train_anyOf_i1"></a>16.2. Property `ReduxOptions > is_train > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

## <a name="root_path"></a>17. Property `ReduxOptions > root_path`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `combining`      |
| **Required**              | Yes              |
| **Additional properties** | Any type allowed |

| Any of(Option)                |
| ----------------------------- |
| [item 0](#root_path_anyOf_i0) |
| [item 1](#root_path_anyOf_i1) |

### <a name="root_path_anyOf_i0"></a>17.1. Property `ReduxOptions > root_path > anyOf > item 0`

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | No       |

### <a name="root_path_anyOf_i1"></a>17.2. Property `ReduxOptions > root_path > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

## <a name="use_amp"></a>18. Property `ReduxOptions > use_amp`

|              |           |
| ------------ | --------- |
| **Type**     | `boolean` |
| **Required** | Yes       |

**Description:** Speed up training and reduce VRAM usage. NVIDIA only.

## <a name="amp_bf16"></a>19. Property `ReduxOptions > amp_bf16`

|              |           |
| ------------ | --------- |
| **Type**     | `boolean` |
| **Required** | Yes       |

**Description:** Use bf16 instead of fp16 for AMP, RTX 3000 series or newer only. Only recommended if fp16 doesn't work.

## <a name="use_channels_last"></a>20. Property `ReduxOptions > use_channels_last`

|              |           |
| ------------ | --------- |
| **Type**     | `boolean` |
| **Required** | Yes       |

**Description:** Enable channels last memory format while using AMP. Reduces VRAM and speeds up training for most architectures, but some architectures are slower with channels last.

## <a name="fast_matmul"></a>21. Property `ReduxOptions > fast_matmul`

|              |           |
| ------------ | --------- |
| **Type**     | `boolean` |
| **Required** | Yes       |

**Description:** Trade precision for performance.

## <a name="use_compile"></a>22. Property `ReduxOptions > use_compile`

|              |           |
| ------------ | --------- |
| **Type**     | `boolean` |
| **Required** | Yes       |

**Description:** Enable torch.compile for the generator model, which takes time on startup to compile the model, but can speed up training after the model is compiled.

## <a name="detect_anomaly"></a>23. Property `ReduxOptions > detect_anomaly`

|              |           |
| ------------ | --------- |
| **Type**     | `boolean` |
| **Required** | Yes       |

**Description:** Whether or not to enable anomaly detection, which can be useful for debugging NaNs that occur during training. Has a significant performance hit and should be disabled when not debugging.

## <a name="high_order_degradation"></a>24. Property `ReduxOptions > high_order_degradation`

|              |           |
| ------------ | --------- |
| **Type**     | `boolean` |
| **Required** | Yes       |

**Description:** Whether or not to enable OTF (on the fly) degradations, which generates LRs on the fly.

## <a name="high_order_degradations_debug"></a>25. Property `ReduxOptions > high_order_degradations_debug`

|              |           |
| ------------ | --------- |
| **Type**     | `boolean` |
| **Required** | Yes       |

**Description:** Whether or not to enable debugging for OTF, which saves the OTF generated LR images so they can be inspected to view the effect of different OTF settings.

## <a name="high_order_degradations_debug_limit"></a>26. Property `ReduxOptions > high_order_degradations_debug_limit`

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | Yes       |

**Description:** The maximum number of OTF images to save when debugging is enabled.

## <a name="dataroot_lq_prob"></a>27. Property `ReduxOptions > dataroot_lq_prob`

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | Yes      |

**Description:** Probability of using paired LR data instead of OTF LR data.

## <a name="force_high_order_degradation_filename_masks"></a>28. Property `ReduxOptions > force_high_order_degradation_filename_masks`

|              |                   |
| ------------ | ----------------- |
| **Type**     | `array of string` |
| **Required** | Yes               |

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

### <a name="force_high_order_degradation_filename_masks_items"></a>28.1. ReduxOptions > force_high_order_degradation_filename_masks > force_high_order_degradation_filename_masks items

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | No       |

## <a name="force_dataroot_lq_filename_masks"></a>29. Property `ReduxOptions > force_dataroot_lq_filename_masks`

|              |                   |
| ------------ | ----------------- |
| **Type**     | `array of string` |
| **Required** | Yes               |

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

### <a name="force_dataroot_lq_filename_masks_items"></a>29.1. ReduxOptions > force_dataroot_lq_filename_masks > force_dataroot_lq_filename_masks items

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | No       |

## <a name="lq_usm"></a>30. Property `ReduxOptions > lq_usm`

|              |           |
| ------------ | --------- |
| **Type**     | `boolean` |
| **Required** | Yes       |

## <a name="lq_usm_radius_range"></a>31. Property `ReduxOptions > lq_usm_radius_range`

|              |         |
| ------------ | ------- |
| **Type**     | `array` |
| **Required** | Yes     |

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

### <a name="autogenerated_heading_2"></a>31.1. ReduxOptions > lq_usm_radius_range > lq_usm_radius_range item 0

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | No        |

### <a name="autogenerated_heading_3"></a>31.2. ReduxOptions > lq_usm_radius_range > lq_usm_radius_range item 1

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | No        |

## <a name="blur_prob"></a>32. Property `ReduxOptions > blur_prob`

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | Yes      |

## <a name="resize_prob"></a>33. Property `ReduxOptions > resize_prob`

|              |                   |
| ------------ | ----------------- |
| **Type**     | `array of number` |
| **Required** | Yes               |

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

### <a name="resize_prob_items"></a>33.1. ReduxOptions > resize_prob > resize_prob items

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |

## <a name="resize_mode_list"></a>34. Property `ReduxOptions > resize_mode_list`

|              |                   |
| ------------ | ----------------- |
| **Type**     | `array of string` |
| **Required** | Yes               |

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

### <a name="resize_mode_list_items"></a>34.1. ReduxOptions > resize_mode_list > resize_mode_list items

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | No       |

## <a name="resize_mode_prob"></a>35. Property `ReduxOptions > resize_mode_prob`

|              |                   |
| ------------ | ----------------- |
| **Type**     | `array of number` |
| **Required** | Yes               |

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

### <a name="resize_mode_prob_items"></a>35.1. ReduxOptions > resize_mode_prob > resize_mode_prob items

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |

## <a name="resize_range"></a>36. Property `ReduxOptions > resize_range`

|              |         |
| ------------ | ------- |
| **Type**     | `array` |
| **Required** | Yes     |

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

### <a name="autogenerated_heading_4"></a>36.1. ReduxOptions > resize_range > resize_range item 0

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |

### <a name="autogenerated_heading_5"></a>36.2. ReduxOptions > resize_range > resize_range item 1

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |

## <a name="gaussian_noise_prob"></a>37. Property `ReduxOptions > gaussian_noise_prob`

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | Yes      |

## <a name="noise_range"></a>38. Property `ReduxOptions > noise_range`

|              |         |
| ------------ | ------- |
| **Type**     | `array` |
| **Required** | Yes     |

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

### <a name="autogenerated_heading_6"></a>38.1. ReduxOptions > noise_range > noise_range item 0

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |

### <a name="autogenerated_heading_7"></a>38.2. ReduxOptions > noise_range > noise_range item 1

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |

## <a name="poisson_scale_range"></a>39. Property `ReduxOptions > poisson_scale_range`

|              |         |
| ------------ | ------- |
| **Type**     | `array` |
| **Required** | Yes     |

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

### <a name="autogenerated_heading_8"></a>39.1. ReduxOptions > poisson_scale_range > poisson_scale_range item 0

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |

### <a name="autogenerated_heading_9"></a>39.2. ReduxOptions > poisson_scale_range > poisson_scale_range item 1

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |

## <a name="gray_noise_prob"></a>40. Property `ReduxOptions > gray_noise_prob`

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | Yes      |

## <a name="jpeg_prob"></a>41. Property `ReduxOptions > jpeg_prob`

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | Yes      |

## <a name="jpeg_range"></a>42. Property `ReduxOptions > jpeg_range`

|              |         |
| ------------ | ------- |
| **Type**     | `array` |
| **Required** | Yes     |

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

### <a name="autogenerated_heading_10"></a>42.1. ReduxOptions > jpeg_range > jpeg_range item 0

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |

### <a name="autogenerated_heading_11"></a>42.2. ReduxOptions > jpeg_range > jpeg_range item 1

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |

## <a name="blur_prob2"></a>43. Property `ReduxOptions > blur_prob2`

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | Yes      |

## <a name="resize_prob2"></a>44. Property `ReduxOptions > resize_prob2`

|              |                   |
| ------------ | ----------------- |
| **Type**     | `array of number` |
| **Required** | Yes               |

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

### <a name="resize_prob2_items"></a>44.1. ReduxOptions > resize_prob2 > resize_prob2 items

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |

## <a name="resize_mode_list2"></a>45. Property `ReduxOptions > resize_mode_list2`

|              |                   |
| ------------ | ----------------- |
| **Type**     | `array of string` |
| **Required** | Yes               |

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

### <a name="resize_mode_list2_items"></a>45.1. ReduxOptions > resize_mode_list2 > resize_mode_list2 items

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | No       |

## <a name="resize_mode_prob2"></a>46. Property `ReduxOptions > resize_mode_prob2`

|              |                   |
| ------------ | ----------------- |
| **Type**     | `array of number` |
| **Required** | Yes               |

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

### <a name="resize_mode_prob2_items"></a>46.1. ReduxOptions > resize_mode_prob2 > resize_mode_prob2 items

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |

## <a name="resize_range2"></a>47. Property `ReduxOptions > resize_range2`

|              |         |
| ------------ | ------- |
| **Type**     | `array` |
| **Required** | Yes     |

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

### <a name="autogenerated_heading_12"></a>47.1. ReduxOptions > resize_range2 > resize_range2 item 0

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |

### <a name="autogenerated_heading_13"></a>47.2. ReduxOptions > resize_range2 > resize_range2 item 1

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |

## <a name="gaussian_noise_prob2"></a>48. Property `ReduxOptions > gaussian_noise_prob2`

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | Yes      |

## <a name="noise_range2"></a>49. Property `ReduxOptions > noise_range2`

|              |         |
| ------------ | ------- |
| **Type**     | `array` |
| **Required** | Yes     |

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

### <a name="autogenerated_heading_14"></a>49.1. ReduxOptions > noise_range2 > noise_range2 item 0

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |

### <a name="autogenerated_heading_15"></a>49.2. ReduxOptions > noise_range2 > noise_range2 item 1

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |

## <a name="poisson_scale_range2"></a>50. Property `ReduxOptions > poisson_scale_range2`

|              |         |
| ------------ | ------- |
| **Type**     | `array` |
| **Required** | Yes     |

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

### <a name="autogenerated_heading_16"></a>50.1. ReduxOptions > poisson_scale_range2 > poisson_scale_range2 item 0

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |

### <a name="autogenerated_heading_17"></a>50.2. ReduxOptions > poisson_scale_range2 > poisson_scale_range2 item 1

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |

## <a name="gray_noise_prob2"></a>51. Property `ReduxOptions > gray_noise_prob2`

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | Yes      |

## <a name="jpeg_prob2"></a>52. Property `ReduxOptions > jpeg_prob2`

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | Yes      |

## <a name="jpeg_range2"></a>53. Property `ReduxOptions > jpeg_range2`

|              |                   |
| ------------ | ----------------- |
| **Type**     | `array of number` |
| **Required** | Yes               |

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

### <a name="jpeg_range2_items"></a>53.1. ReduxOptions > jpeg_range2 > jpeg_range2 items

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |

## <a name="resize_mode_list3"></a>54. Property `ReduxOptions > resize_mode_list3`

|              |                   |
| ------------ | ----------------- |
| **Type**     | `array of string` |
| **Required** | Yes               |

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

### <a name="resize_mode_list3_items"></a>54.1. ReduxOptions > resize_mode_list3 > resize_mode_list3 items

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | No       |

## <a name="resize_mode_prob3"></a>55. Property `ReduxOptions > resize_mode_prob3`

|              |                   |
| ------------ | ----------------- |
| **Type**     | `array of number` |
| **Required** | Yes               |

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

### <a name="resize_mode_prob3_items"></a>55.1. ReduxOptions > resize_mode_prob3 > resize_mode_prob3 items

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |

## <a name="queue_size"></a>56. Property `ReduxOptions > queue_size`

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | Yes       |

## <a name="datasets"></a>57. Property `ReduxOptions > datasets`

|                           |                                                                                       |
| ------------------------- | ------------------------------------------------------------------------------------- |
| **Type**                  | `object`                                                                              |
| **Required**              | Yes                                                                                   |
| **Additional properties** | [Each additional property must conform to the schema](#datasets_additionalProperties) |

| Property                              | Pattern | Type   | Deprecated | Definition                | Title/Description |
| ------------------------------------- | ------- | ------ | ---------- | ------------------------- | ----------------- |
| - [](#datasets_additionalProperties ) | No      | object | No         | In #/$defs/DatasetOptions | DatasetOptions    |

### <a name="datasets_additionalProperties"></a>57.1. Property `ReduxOptions > datasets > DatasetOptions`

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

#### <a name="datasets_additionalProperties_name"></a>57.1.1. Property `ReduxOptions > datasets > DatasetOptions > name`

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | Yes      |

**Description:** Name of the dataset. It should be unique compared to other datasets in this config, but the exact name isn't very important.

#### <a name="datasets_additionalProperties_type"></a>57.1.2. Property `ReduxOptions > datasets > DatasetOptions > type`

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | Yes      |

#### <a name="datasets_additionalProperties_io_backend"></a>57.1.3. Property `ReduxOptions > datasets > DatasetOptions > io_backend`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `object`         |
| **Required**              | Yes              |
| **Additional properties** | Any type allowed |

#### <a name="datasets_additionalProperties_num_worker_per_gpu"></a>57.1.4. Property `ReduxOptions > datasets > DatasetOptions > num_worker_per_gpu`

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

##### <a name="datasets_additionalProperties_num_worker_per_gpu_anyOf_i0"></a>57.1.4.1. Property `ReduxOptions > datasets > DatasetOptions > num_worker_per_gpu > anyOf > item 0`

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | No        |

##### <a name="datasets_additionalProperties_num_worker_per_gpu_anyOf_i1"></a>57.1.4.2. Property `ReduxOptions > datasets > DatasetOptions > num_worker_per_gpu > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

#### <a name="datasets_additionalProperties_batch_size_per_gpu"></a>57.1.5. Property `ReduxOptions > datasets > DatasetOptions > batch_size_per_gpu`

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

##### <a name="datasets_additionalProperties_batch_size_per_gpu_anyOf_i0"></a>57.1.5.1. Property `ReduxOptions > datasets > DatasetOptions > batch_size_per_gpu > anyOf > item 0`

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | No        |

##### <a name="datasets_additionalProperties_batch_size_per_gpu_anyOf_i1"></a>57.1.5.2. Property `ReduxOptions > datasets > DatasetOptions > batch_size_per_gpu > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

#### <a name="datasets_additionalProperties_accum_iter"></a>57.1.6. Property `ReduxOptions > datasets > DatasetOptions > accum_iter`

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | No        |
| **Default**  | `1`       |

**Description:** Using values larger than 1 simulates higher batch size by trading performance for reduced VRAM usage. If accum_iter = 4 and batch_size_per_gpu = 6 then effective batch size = 4 * 6 = 24 but performance may be as much as 4 times as slow.

#### <a name="datasets_additionalProperties_use_hflip"></a>57.1.7. Property `ReduxOptions > datasets > DatasetOptions > use_hflip`

|              |           |
| ------------ | --------- |
| **Type**     | `boolean` |
| **Required** | No        |
| **Default**  | `true`    |

**Description:** Randomly flip the images horizontally.

#### <a name="datasets_additionalProperties_use_rot"></a>57.1.8. Property `ReduxOptions > datasets > DatasetOptions > use_rot`

|              |           |
| ------------ | --------- |
| **Type**     | `boolean` |
| **Required** | No        |
| **Default**  | `true`    |

**Description:** Randomly rotate the images.

#### <a name="datasets_additionalProperties_mean"></a>57.1.9. Property `ReduxOptions > datasets > DatasetOptions > mean`

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

##### <a name="datasets_additionalProperties_mean_anyOf_i0"></a>57.1.9.1. Property `ReduxOptions > datasets > DatasetOptions > mean > anyOf > item 0`

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

###### <a name="datasets_additionalProperties_mean_anyOf_i0_items"></a>57.1.9.1.1. ReduxOptions > datasets > DatasetOptions > mean > anyOf > item 0 > item 0 items

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |

##### <a name="datasets_additionalProperties_mean_anyOf_i1"></a>57.1.9.2. Property `ReduxOptions > datasets > DatasetOptions > mean > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

#### <a name="datasets_additionalProperties_std"></a>57.1.10. Property `ReduxOptions > datasets > DatasetOptions > std`

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

##### <a name="datasets_additionalProperties_std_anyOf_i0"></a>57.1.10.1. Property `ReduxOptions > datasets > DatasetOptions > std > anyOf > item 0`

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

###### <a name="datasets_additionalProperties_std_anyOf_i0_items"></a>57.1.10.1.1. ReduxOptions > datasets > DatasetOptions > std > anyOf > item 0 > item 0 items

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |

##### <a name="datasets_additionalProperties_std_anyOf_i1"></a>57.1.10.2. Property `ReduxOptions > datasets > DatasetOptions > std > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

#### <a name="datasets_additionalProperties_gt_size"></a>57.1.11. Property `ReduxOptions > datasets > DatasetOptions > gt_size`

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

##### <a name="datasets_additionalProperties_gt_size_anyOf_i0"></a>57.1.11.1. Property `ReduxOptions > datasets > DatasetOptions > gt_size > anyOf > item 0`

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | No        |

##### <a name="datasets_additionalProperties_gt_size_anyOf_i1"></a>57.1.11.2. Property `ReduxOptions > datasets > DatasetOptions > gt_size > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

#### <a name="datasets_additionalProperties_lq_size"></a>57.1.12. Property `ReduxOptions > datasets > DatasetOptions > lq_size`

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

##### <a name="datasets_additionalProperties_lq_size_anyOf_i0"></a>57.1.12.1. Property `ReduxOptions > datasets > DatasetOptions > lq_size > anyOf > item 0`

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | No        |

##### <a name="datasets_additionalProperties_lq_size_anyOf_i1"></a>57.1.12.2. Property `ReduxOptions > datasets > DatasetOptions > lq_size > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

#### <a name="datasets_additionalProperties_color"></a>57.1.13. Property `ReduxOptions > datasets > DatasetOptions > color`

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

##### <a name="datasets_additionalProperties_color_anyOf_i0"></a>57.1.13.1. Property `ReduxOptions > datasets > DatasetOptions > color > anyOf > item 0`

|              |                    |
| ------------ | ------------------ |
| **Type**     | `enum (of string)` |
| **Required** | No                 |

Must be one of:
* "y"

##### <a name="datasets_additionalProperties_color_anyOf_i1"></a>57.1.13.2. Property `ReduxOptions > datasets > DatasetOptions > color > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

#### <a name="datasets_additionalProperties_phase"></a>57.1.14. Property `ReduxOptions > datasets > DatasetOptions > phase`

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

##### <a name="datasets_additionalProperties_phase_anyOf_i0"></a>57.1.14.1. Property `ReduxOptions > datasets > DatasetOptions > phase > anyOf > item 0`

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | No       |

##### <a name="datasets_additionalProperties_phase_anyOf_i1"></a>57.1.14.2. Property `ReduxOptions > datasets > DatasetOptions > phase > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

#### <a name="datasets_additionalProperties_scale"></a>57.1.15. Property `ReduxOptions > datasets > DatasetOptions > scale`

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

##### <a name="datasets_additionalProperties_scale_anyOf_i0"></a>57.1.15.1. Property `ReduxOptions > datasets > DatasetOptions > scale > anyOf > item 0`

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | No        |

##### <a name="datasets_additionalProperties_scale_anyOf_i1"></a>57.1.15.2. Property `ReduxOptions > datasets > DatasetOptions > scale > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

#### <a name="datasets_additionalProperties_dataset_enlarge_ratio"></a>57.1.16. Property `ReduxOptions > datasets > DatasetOptions > dataset_enlarge_ratio`

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

##### <a name="datasets_additionalProperties_dataset_enlarge_ratio_anyOf_i0"></a>57.1.16.1. Property `ReduxOptions > datasets > DatasetOptions > dataset_enlarge_ratio > anyOf > item 0`

|              |                    |
| ------------ | ------------------ |
| **Type**     | `enum (of string)` |
| **Required** | No                 |

Must be one of:
* "auto"

##### <a name="datasets_additionalProperties_dataset_enlarge_ratio_anyOf_i1"></a>57.1.16.2. Property `ReduxOptions > datasets > DatasetOptions > dataset_enlarge_ratio > anyOf > item 1`

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | No        |

#### <a name="datasets_additionalProperties_prefetch_mode"></a>57.1.17. Property `ReduxOptions > datasets > DatasetOptions > prefetch_mode`

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

##### <a name="datasets_additionalProperties_prefetch_mode_anyOf_i0"></a>57.1.17.1. Property `ReduxOptions > datasets > DatasetOptions > prefetch_mode > anyOf > item 0`

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | No       |

##### <a name="datasets_additionalProperties_prefetch_mode_anyOf_i1"></a>57.1.17.2. Property `ReduxOptions > datasets > DatasetOptions > prefetch_mode > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

#### <a name="datasets_additionalProperties_pin_memory"></a>57.1.18. Property `ReduxOptions > datasets > DatasetOptions > pin_memory`

|              |           |
| ------------ | --------- |
| **Type**     | `boolean` |
| **Required** | No        |
| **Default**  | `true`    |

#### <a name="datasets_additionalProperties_persistent_workers"></a>57.1.19. Property `ReduxOptions > datasets > DatasetOptions > persistent_workers`

|              |           |
| ------------ | --------- |
| **Type**     | `boolean` |
| **Required** | No        |
| **Default**  | `true`    |

#### <a name="datasets_additionalProperties_num_prefetch_queue"></a>57.1.20. Property `ReduxOptions > datasets > DatasetOptions > num_prefetch_queue`

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | No        |
| **Default**  | `1`       |

#### <a name="datasets_additionalProperties_prefetch_factor"></a>57.1.21. Property `ReduxOptions > datasets > DatasetOptions > prefetch_factor`

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | No        |
| **Default**  | `2`       |

#### <a name="datasets_additionalProperties_clip_size"></a>57.1.22. Property `ReduxOptions > datasets > DatasetOptions > clip_size`

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

##### <a name="datasets_additionalProperties_clip_size_anyOf_i0"></a>57.1.22.1. Property `ReduxOptions > datasets > DatasetOptions > clip_size > anyOf > item 0`

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | No        |

##### <a name="datasets_additionalProperties_clip_size_anyOf_i1"></a>57.1.22.2. Property `ReduxOptions > datasets > DatasetOptions > clip_size > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

#### <a name="datasets_additionalProperties_dataroot_gt"></a>57.1.23. Property `ReduxOptions > datasets > DatasetOptions > dataroot_gt`

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

##### <a name="datasets_additionalProperties_dataroot_gt_anyOf_i0"></a>57.1.23.1. Property `ReduxOptions > datasets > DatasetOptions > dataroot_gt > anyOf > item 0`

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | No       |

##### <a name="datasets_additionalProperties_dataroot_gt_anyOf_i1"></a>57.1.23.2. Property `ReduxOptions > datasets > DatasetOptions > dataroot_gt > anyOf > item 1`

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

###### <a name="datasets_additionalProperties_dataroot_gt_anyOf_i1_items"></a>57.1.23.2.1. ReduxOptions > datasets > DatasetOptions > dataroot_gt > anyOf > item 1 > item 1 items

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | No       |

##### <a name="datasets_additionalProperties_dataroot_gt_anyOf_i2"></a>57.1.23.3. Property `ReduxOptions > datasets > DatasetOptions > dataroot_gt > anyOf > item 2`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

#### <a name="datasets_additionalProperties_dataroot_lq"></a>57.1.24. Property `ReduxOptions > datasets > DatasetOptions > dataroot_lq`

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

##### <a name="datasets_additionalProperties_dataroot_lq_anyOf_i0"></a>57.1.24.1. Property `ReduxOptions > datasets > DatasetOptions > dataroot_lq > anyOf > item 0`

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | No       |

##### <a name="datasets_additionalProperties_dataroot_lq_anyOf_i1"></a>57.1.24.2. Property `ReduxOptions > datasets > DatasetOptions > dataroot_lq > anyOf > item 1`

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

###### <a name="datasets_additionalProperties_dataroot_lq_anyOf_i1_items"></a>57.1.24.2.1. ReduxOptions > datasets > DatasetOptions > dataroot_lq > anyOf > item 1 > item 1 items

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | No       |

##### <a name="datasets_additionalProperties_dataroot_lq_anyOf_i2"></a>57.1.24.3. Property `ReduxOptions > datasets > DatasetOptions > dataroot_lq > anyOf > item 2`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

#### <a name="datasets_additionalProperties_meta_info"></a>57.1.25. Property `ReduxOptions > datasets > DatasetOptions > meta_info`

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

##### <a name="datasets_additionalProperties_meta_info_anyOf_i0"></a>57.1.25.1. Property `ReduxOptions > datasets > DatasetOptions > meta_info > anyOf > item 0`

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | No       |

##### <a name="datasets_additionalProperties_meta_info_anyOf_i1"></a>57.1.25.2. Property `ReduxOptions > datasets > DatasetOptions > meta_info > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

#### <a name="datasets_additionalProperties_filename_tmpl"></a>57.1.26. Property `ReduxOptions > datasets > DatasetOptions > filename_tmpl`

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | No       |
| **Default**  | `"{}"`   |

#### <a name="datasets_additionalProperties_blur_kernel_size"></a>57.1.27. Property `ReduxOptions > datasets > DatasetOptions > blur_kernel_size`

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | No        |
| **Default**  | `12`      |

#### <a name="datasets_additionalProperties_kernel_list"></a>57.1.28. Property `ReduxOptions > datasets > DatasetOptions > kernel_list`

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

##### <a name="datasets_additionalProperties_kernel_list_items"></a>57.1.28.1. ReduxOptions > datasets > DatasetOptions > kernel_list > kernel_list items

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | No       |

#### <a name="datasets_additionalProperties_kernel_prob"></a>57.1.29. Property `ReduxOptions > datasets > DatasetOptions > kernel_prob`

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

##### <a name="datasets_additionalProperties_kernel_prob_items"></a>57.1.29.1. ReduxOptions > datasets > DatasetOptions > kernel_prob > kernel_prob items

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |

#### <a name="datasets_additionalProperties_kernel_range"></a>57.1.30. Property `ReduxOptions > datasets > DatasetOptions > kernel_range`

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

##### <a name="autogenerated_heading_18"></a>57.1.30.1. ReduxOptions > datasets > DatasetOptions > kernel_range > kernel_range item 0

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | No        |

##### <a name="autogenerated_heading_19"></a>57.1.30.2. ReduxOptions > datasets > DatasetOptions > kernel_range > kernel_range item 1

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | No        |

#### <a name="datasets_additionalProperties_sinc_prob"></a>57.1.31. Property `ReduxOptions > datasets > DatasetOptions > sinc_prob`

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |
| **Default**  | `0`      |

#### <a name="datasets_additionalProperties_blur_sigma"></a>57.1.32. Property `ReduxOptions > datasets > DatasetOptions > blur_sigma`

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

##### <a name="autogenerated_heading_20"></a>57.1.32.1. ReduxOptions > datasets > DatasetOptions > blur_sigma > blur_sigma item 0

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |

##### <a name="autogenerated_heading_21"></a>57.1.32.2. ReduxOptions > datasets > DatasetOptions > blur_sigma > blur_sigma item 1

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |

#### <a name="datasets_additionalProperties_betag_range"></a>57.1.33. Property `ReduxOptions > datasets > DatasetOptions > betag_range`

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

##### <a name="autogenerated_heading_22"></a>57.1.33.1. ReduxOptions > datasets > DatasetOptions > betag_range > betag_range item 0

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |

##### <a name="autogenerated_heading_23"></a>57.1.33.2. ReduxOptions > datasets > DatasetOptions > betag_range > betag_range item 1

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |

#### <a name="datasets_additionalProperties_betap_range"></a>57.1.34. Property `ReduxOptions > datasets > DatasetOptions > betap_range`

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

##### <a name="autogenerated_heading_24"></a>57.1.34.1. ReduxOptions > datasets > DatasetOptions > betap_range > betap_range item 0

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |

##### <a name="autogenerated_heading_25"></a>57.1.34.2. ReduxOptions > datasets > DatasetOptions > betap_range > betap_range item 1

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |

#### <a name="datasets_additionalProperties_blur_kernel_size2"></a>57.1.35. Property `ReduxOptions > datasets > DatasetOptions > blur_kernel_size2`

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | No        |
| **Default**  | `12`      |

#### <a name="datasets_additionalProperties_kernel_list2"></a>57.1.36. Property `ReduxOptions > datasets > DatasetOptions > kernel_list2`

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

##### <a name="datasets_additionalProperties_kernel_list2_items"></a>57.1.36.1. ReduxOptions > datasets > DatasetOptions > kernel_list2 > kernel_list2 items

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | No       |

#### <a name="datasets_additionalProperties_kernel_prob2"></a>57.1.37. Property `ReduxOptions > datasets > DatasetOptions > kernel_prob2`

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

##### <a name="datasets_additionalProperties_kernel_prob2_items"></a>57.1.37.1. ReduxOptions > datasets > DatasetOptions > kernel_prob2 > kernel_prob2 items

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |

#### <a name="datasets_additionalProperties_kernel_range2"></a>57.1.38. Property `ReduxOptions > datasets > DatasetOptions > kernel_range2`

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

##### <a name="autogenerated_heading_26"></a>57.1.38.1. ReduxOptions > datasets > DatasetOptions > kernel_range2 > kernel_range2 item 0

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | No        |

##### <a name="autogenerated_heading_27"></a>57.1.38.2. ReduxOptions > datasets > DatasetOptions > kernel_range2 > kernel_range2 item 1

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | No        |

#### <a name="datasets_additionalProperties_sinc_prob2"></a>57.1.39. Property `ReduxOptions > datasets > DatasetOptions > sinc_prob2`

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |
| **Default**  | `0`      |

#### <a name="datasets_additionalProperties_blur_sigma2"></a>57.1.40. Property `ReduxOptions > datasets > DatasetOptions > blur_sigma2`

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

##### <a name="autogenerated_heading_28"></a>57.1.40.1. ReduxOptions > datasets > DatasetOptions > blur_sigma2 > blur_sigma2 item 0

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |

##### <a name="autogenerated_heading_29"></a>57.1.40.2. ReduxOptions > datasets > DatasetOptions > blur_sigma2 > blur_sigma2 item 1

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |

#### <a name="datasets_additionalProperties_betag_range2"></a>57.1.41. Property `ReduxOptions > datasets > DatasetOptions > betag_range2`

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

##### <a name="autogenerated_heading_30"></a>57.1.41.1. ReduxOptions > datasets > DatasetOptions > betag_range2 > betag_range2 item 0

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |

##### <a name="autogenerated_heading_31"></a>57.1.41.2. ReduxOptions > datasets > DatasetOptions > betag_range2 > betag_range2 item 1

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |

#### <a name="datasets_additionalProperties_betap_range2"></a>57.1.42. Property `ReduxOptions > datasets > DatasetOptions > betap_range2`

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

##### <a name="autogenerated_heading_32"></a>57.1.42.1. ReduxOptions > datasets > DatasetOptions > betap_range2 > betap_range2 item 0

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |

##### <a name="autogenerated_heading_33"></a>57.1.42.2. ReduxOptions > datasets > DatasetOptions > betap_range2 > betap_range2 item 1

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |

#### <a name="datasets_additionalProperties_final_sinc_prob"></a>57.1.43. Property `ReduxOptions > datasets > DatasetOptions > final_sinc_prob`

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |
| **Default**  | `0`      |

#### <a name="datasets_additionalProperties_final_kernel_range"></a>57.1.44. Property `ReduxOptions > datasets > DatasetOptions > final_kernel_range`

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

##### <a name="autogenerated_heading_34"></a>57.1.44.1. ReduxOptions > datasets > DatasetOptions > final_kernel_range > final_kernel_range item 0

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | No        |

##### <a name="autogenerated_heading_35"></a>57.1.44.2. ReduxOptions > datasets > DatasetOptions > final_kernel_range > final_kernel_range item 1

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | No        |

## <a name="train"></a>58. Property `ReduxOptions > train`

**Title:** TrainOptions

|                           |                      |
| ------------------------- | -------------------- |
| **Type**                  | `object`             |
| **Required**              | Yes                  |
| **Additional properties** | Any type allowed     |
| **Defined in**            | #/$defs/TrainOptions |

| Property                                       | Pattern | Type             | Deprecated | Definition | Title/Description                                                                                                                                       |
| ---------------------------------------------- | ------- | ---------------- | ---------- | ---------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| + [type](#train_type )                         | No      | enum (of string) | No         | -          | -                                                                                                                                                       |
| + [total_iter](#train_total_iter )             | No      | integer          | No         | -          | The total number of iterations to train.                                                                                                                |
| + [optim_g](#train_optim_g )                   | No      | object           | No         | -          | The optimizer to use for the generator model.                                                                                                           |
| + [ema_decay](#train_ema_decay )               | No      | number           | No         | -          | The decay factor to use for EMA (exponential moving average). Set to 0 to disable EMA.                                                                  |
| + [grad_clip](#train_grad_clip )               | No      | boolean          | No         | -          | Whether or not to enable gradient clipping, which can improve stability when using higher learning rates, but can also cause issues in some situations. |
| + [warmup_iter](#train_warmup_iter )           | No      | integer          | No         | -          | Gradually ramp up learning rates until this iteration, to stabilize early training. Use -1 to disable.                                                  |
| + [scheduler](#train_scheduler )               | No      | Combination      | No         | -          | Options for the optimizer scheduler. If there are multiple optimizers, both will use the same scheduler options.                                        |
| + [optim_d](#train_optim_d )                   | No      | Combination      | No         | -          | The optimizer to use for the discriminator model.                                                                                                       |
| + [losses](#train_losses )                     | No      | array            | No         | -          | The list of loss functions to optimize.                                                                                                                 |
| + [pixel_opt](#train_pixel_opt )               | No      | Combination      | No         | -          | -                                                                                                                                                       |
| + [mssim_opt](#train_mssim_opt )               | No      | Combination      | No         | -          | -                                                                                                                                                       |
| + [ms_ssim_l1_opt](#train_ms_ssim_l1_opt )     | No      | Combination      | No         | -          | -                                                                                                                                                       |
| + [perceptual_opt](#train_perceptual_opt )     | No      | Combination      | No         | -          | -                                                                                                                                                       |
| + [contextual_opt](#train_contextual_opt )     | No      | Combination      | No         | -          | -                                                                                                                                                       |
| + [dists_opt](#train_dists_opt )               | No      | Combination      | No         | -          | -                                                                                                                                                       |
| + [hr_inversion_opt](#train_hr_inversion_opt ) | No      | Combination      | No         | -          | -                                                                                                                                                       |
| + [dinov2_opt](#train_dinov2_opt )             | No      | Combination      | No         | -          | -                                                                                                                                                       |
| + [topiq_opt](#train_topiq_opt )               | No      | Combination      | No         | -          | -                                                                                                                                                       |
| + [pd_opt](#train_pd_opt )                     | No      | Combination      | No         | -          | -                                                                                                                                                       |
| + [fd_opt](#train_fd_opt )                     | No      | Combination      | No         | -          | -                                                                                                                                                       |
| + [ldl_opt](#train_ldl_opt )                   | No      | Combination      | No         | -          | -                                                                                                                                                       |
| + [hsluv_opt](#train_hsluv_opt )               | No      | Combination      | No         | -          | -                                                                                                                                                       |
| + [gan_opt](#train_gan_opt )                   | No      | Combination      | No         | -          | -                                                                                                                                                       |
| + [color_opt](#train_color_opt )               | No      | Combination      | No         | -          | -                                                                                                                                                       |
| + [luma_opt](#train_luma_opt )                 | No      | Combination      | No         | -          | -                                                                                                                                                       |
| + [avg_opt](#train_avg_opt )                   | No      | Combination      | No         | -          | -                                                                                                                                                       |
| + [bicubic_opt](#train_bicubic_opt )           | No      | Combination      | No         | -          | -                                                                                                                                                       |
| + [use_moa](#train_use_moa )                   | No      | boolean          | No         | -          | Whether to enable mixture of augmentations, which augments the dataset on the fly to create more variety and help the model generalize.                 |
| + [moa_augs](#train_moa_augs )                 | No      | array of string  | No         | -          | The list of augmentations to choose from, only one is selected per iteration.                                                                           |
| + [moa_probs](#train_moa_probs )               | No      | array of number  | No         | -          | The probability each augmentation in moa_augs will be applied. Total should add up to 1.                                                                |
| + [moa_debug](#train_moa_debug )               | No      | boolean          | No         | -          | Save images before and after augment to debug/moa folder inside of the root training directory.                                                         |
| + [moa_debug_limit](#train_moa_debug_limit )   | No      | integer          | No         | -          | The max number of iterations to save augmentation images for.                                                                                           |

### <a name="train_type"></a>58.1. Property `ReduxOptions > train > type`

|              |                    |
| ------------ | ------------------ |
| **Type**     | `enum (of string)` |
| **Required** | Yes                |

Must be one of:
* "TrainOptions"

### <a name="train_total_iter"></a>58.2. Property `ReduxOptions > train > total_iter`

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | Yes       |

**Description:** The total number of iterations to train.

### <a name="train_optim_g"></a>58.3. Property `ReduxOptions > train > optim_g`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `object`         |
| **Required**              | Yes              |
| **Additional properties** | Any type allowed |

**Description:** The optimizer to use for the generator model.

### <a name="train_ema_decay"></a>58.4. Property `ReduxOptions > train > ema_decay`

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | Yes      |

**Description:** The decay factor to use for EMA (exponential moving average). Set to 0 to disable EMA.

### <a name="train_grad_clip"></a>58.5. Property `ReduxOptions > train > grad_clip`

|              |           |
| ------------ | --------- |
| **Type**     | `boolean` |
| **Required** | Yes       |

**Description:** Whether or not to enable gradient clipping, which can improve stability when using higher learning rates, but can also cause issues in some situations.

### <a name="train_warmup_iter"></a>58.6. Property `ReduxOptions > train > warmup_iter`

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | Yes       |

**Description:** Gradually ramp up learning rates until this iteration, to stabilize early training. Use -1 to disable.

### <a name="train_scheduler"></a>58.7. Property `ReduxOptions > train > scheduler`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `combining`      |
| **Required**              | Yes              |
| **Additional properties** | Any type allowed |

**Description:** Options for the optimizer scheduler. If there are multiple optimizers, both will use the same scheduler options.

| Any of(Option)                                |
| --------------------------------------------- |
| [item 0](#train_scheduler_anyOf_i0)           |
| [SchedulerOptions](#train_scheduler_anyOf_i1) |

#### <a name="train_scheduler_anyOf_i0"></a>58.7.1. Property `ReduxOptions > train > scheduler > anyOf > item 0`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

#### <a name="train_scheduler_anyOf_i1"></a>58.7.2. Property `ReduxOptions > train > scheduler > anyOf > SchedulerOptions`

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

##### <a name="train_scheduler_anyOf_i1_type"></a>58.7.2.1. Property `ReduxOptions > train > scheduler > anyOf > SchedulerOptions > type`

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | Yes      |

##### <a name="train_scheduler_anyOf_i1_milestones"></a>58.7.2.2. Property `ReduxOptions > train > scheduler > anyOf > SchedulerOptions > milestones`

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

###### <a name="train_scheduler_anyOf_i1_milestones_items"></a>58.7.2.2.1. ReduxOptions > train > scheduler > anyOf > SchedulerOptions > milestones > milestones items

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | No        |

##### <a name="train_scheduler_anyOf_i1_gamma"></a>58.7.2.3. Property `ReduxOptions > train > scheduler > anyOf > SchedulerOptions > gamma`

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | Yes      |

**Description:** At each milestone, the learning rate is multiplied by this number, so a gamma of 0.5 cuts the learning rate in half at each milestone.

### <a name="train_optim_d"></a>58.8. Property `ReduxOptions > train > optim_d`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `combining`      |
| **Required**              | Yes              |
| **Additional properties** | Any type allowed |

**Description:** The optimizer to use for the discriminator model.

| Any of(Option)                    |
| --------------------------------- |
| [item 0](#train_optim_d_anyOf_i0) |
| [item 1](#train_optim_d_anyOf_i1) |

#### <a name="train_optim_d_anyOf_i0"></a>58.8.1. Property `ReduxOptions > train > optim_d > anyOf > item 0`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `object`         |
| **Required**              | No               |
| **Additional properties** | Any type allowed |

#### <a name="train_optim_d_anyOf_i1"></a>58.8.2. Property `ReduxOptions > train > optim_d > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

### <a name="train_losses"></a>58.9. Property `ReduxOptions > train > losses`

|              |         |
| ------------ | ------- |
| **Type**     | `array` |
| **Required** | Yes     |

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

#### <a name="train_losses_items"></a>58.9.1. ReduxOptions > train > losses > losses items

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
| [dsdloss](#train_losses_items_anyOf_i13)            |
| [ffloss](#train_losses_items_anyOf_i14)             |
| [ldlloss](#train_losses_items_anyOf_i15)            |
| [mssimloss](#train_losses_items_anyOf_i16)          |
| [msssiml1loss](#train_losses_items_anyOf_i17)       |
| [nccloss](#train_losses_items_anyOf_i18)            |
| [perceptualfp16loss](#train_losses_items_anyOf_i19) |
| [perceptualloss](#train_losses_items_anyOf_i20)     |
| [topiqloss](#train_losses_items_anyOf_i21)          |

##### <a name="train_losses_items_anyOf_i0"></a>58.9.1.1. Property `ReduxOptions > train > losses > losses items > anyOf > ganloss`

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
| + [gan_type](#train_losses_items_anyOf_i0_gan_type )             | No      | string           | No         | -          | -                 |
| + [real_label_val](#train_losses_items_anyOf_i0_real_label_val ) | No      | number           | No         | -          | -                 |
| + [fake_label_val](#train_losses_items_anyOf_i0_fake_label_val ) | No      | number           | No         | -          | -                 |
| + [loss_weight](#train_losses_items_anyOf_i0_loss_weight )       | No      | number           | No         | -          | -                 |

###### <a name="train_losses_items_anyOf_i0_type"></a>58.9.1.1.1. Property `ReduxOptions > train > losses > losses items > anyOf > ganloss > type`

|              |                    |
| ------------ | ------------------ |
| **Type**     | `enum (of string)` |
| **Required** | Yes                |

Must be one of:
* "ganloss"

###### <a name="train_losses_items_anyOf_i0_gan_type"></a>58.9.1.1.2. Property `ReduxOptions > train > losses > losses items > anyOf > ganloss > gan_type`

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | Yes      |

###### <a name="train_losses_items_anyOf_i0_real_label_val"></a>58.9.1.1.3. Property `ReduxOptions > train > losses > losses items > anyOf > ganloss > real_label_val`

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | Yes      |

###### <a name="train_losses_items_anyOf_i0_fake_label_val"></a>58.9.1.1.4. Property `ReduxOptions > train > losses > losses items > anyOf > ganloss > fake_label_val`

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | Yes      |

###### <a name="train_losses_items_anyOf_i0_loss_weight"></a>58.9.1.1.5. Property `ReduxOptions > train > losses > losses items > anyOf > ganloss > loss_weight`

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | Yes      |

##### <a name="train_losses_items_anyOf_i1"></a>58.9.1.2. Property `ReduxOptions > train > losses > losses items > anyOf > multiscaleganloss`

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
| + [gan_type](#train_losses_items_anyOf_i1_gan_type )             | No      | string           | No         | -          | -                 |
| + [real_label_val](#train_losses_items_anyOf_i1_real_label_val ) | No      | number           | No         | -          | -                 |
| + [fake_label_val](#train_losses_items_anyOf_i1_fake_label_val ) | No      | number           | No         | -          | -                 |
| + [loss_weight](#train_losses_items_anyOf_i1_loss_weight )       | No      | number           | No         | -          | -                 |

###### <a name="train_losses_items_anyOf_i1_type"></a>58.9.1.2.1. Property `ReduxOptions > train > losses > losses items > anyOf > multiscaleganloss > type`

|              |                    |
| ------------ | ------------------ |
| **Type**     | `enum (of string)` |
| **Required** | Yes                |

Must be one of:
* "multiscaleganloss"

###### <a name="train_losses_items_anyOf_i1_gan_type"></a>58.9.1.2.2. Property `ReduxOptions > train > losses > losses items > anyOf > multiscaleganloss > gan_type`

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | Yes      |

###### <a name="train_losses_items_anyOf_i1_real_label_val"></a>58.9.1.2.3. Property `ReduxOptions > train > losses > losses items > anyOf > multiscaleganloss > real_label_val`

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | Yes      |

###### <a name="train_losses_items_anyOf_i1_fake_label_val"></a>58.9.1.2.4. Property `ReduxOptions > train > losses > losses items > anyOf > multiscaleganloss > fake_label_val`

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | Yes      |

###### <a name="train_losses_items_anyOf_i1_loss_weight"></a>58.9.1.2.5. Property `ReduxOptions > train > losses > losses items > anyOf > multiscaleganloss > loss_weight`

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | Yes      |

##### <a name="train_losses_items_anyOf_i2"></a>58.9.1.3. Property `ReduxOptions > train > losses > losses items > anyOf > adistsloss`

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
| + [window_size](#train_losses_items_anyOf_i2_window_size )   | No      | integer          | No         | -          | -                 |
| + [resize_input](#train_losses_items_anyOf_i2_resize_input ) | No      | boolean          | No         | -          | -                 |
| + [loss_weight](#train_losses_items_anyOf_i2_loss_weight )   | No      | number           | No         | -          | -                 |

###### <a name="train_losses_items_anyOf_i2_type"></a>58.9.1.3.1. Property `ReduxOptions > train > losses > losses items > anyOf > adistsloss > type`

|              |                    |
| ------------ | ------------------ |
| **Type**     | `enum (of string)` |
| **Required** | Yes                |

Must be one of:
* "adistsloss"

###### <a name="train_losses_items_anyOf_i2_window_size"></a>58.9.1.3.2. Property `ReduxOptions > train > losses > losses items > anyOf > adistsloss > window_size`

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | Yes       |

###### <a name="train_losses_items_anyOf_i2_resize_input"></a>58.9.1.3.3. Property `ReduxOptions > train > losses > losses items > anyOf > adistsloss > resize_input`

|              |           |
| ------------ | --------- |
| **Type**     | `boolean` |
| **Required** | Yes       |

###### <a name="train_losses_items_anyOf_i2_loss_weight"></a>58.9.1.3.4. Property `ReduxOptions > train > losses > losses items > anyOf > adistsloss > loss_weight`

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | Yes      |

##### <a name="train_losses_items_anyOf_i3"></a>58.9.1.4. Property `ReduxOptions > train > losses > losses items > anyOf > l1loss`

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
| + [reduction](#train_losses_items_anyOf_i3_reduction )     | No      | string           | No         | -          | -                 |

###### <a name="train_losses_items_anyOf_i3_type"></a>58.9.1.4.1. Property `ReduxOptions > train > losses > losses items > anyOf > l1loss > type`

|              |                    |
| ------------ | ------------------ |
| **Type**     | `enum (of string)` |
| **Required** | Yes                |

Must be one of:
* "l1loss"

###### <a name="train_losses_items_anyOf_i3_loss_weight"></a>58.9.1.4.2. Property `ReduxOptions > train > losses > losses items > anyOf > l1loss > loss_weight`

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | Yes      |

###### <a name="train_losses_items_anyOf_i3_reduction"></a>58.9.1.4.3. Property `ReduxOptions > train > losses > losses items > anyOf > l1loss > reduction`

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | Yes      |

##### <a name="train_losses_items_anyOf_i4"></a>58.9.1.5. Property `ReduxOptions > train > losses > losses items > anyOf > mseloss`

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
| + [reduction](#train_losses_items_anyOf_i4_reduction )     | No      | string           | No         | -          | -                 |

###### <a name="train_losses_items_anyOf_i4_type"></a>58.9.1.5.1. Property `ReduxOptions > train > losses > losses items > anyOf > mseloss > type`

|              |                    |
| ------------ | ------------------ |
| **Type**     | `enum (of string)` |
| **Required** | Yes                |

Must be one of:
* "mseloss"

###### <a name="train_losses_items_anyOf_i4_loss_weight"></a>58.9.1.5.2. Property `ReduxOptions > train > losses > losses items > anyOf > mseloss > loss_weight`

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | Yes      |

###### <a name="train_losses_items_anyOf_i4_reduction"></a>58.9.1.5.3. Property `ReduxOptions > train > losses > losses items > anyOf > mseloss > reduction`

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | Yes      |

##### <a name="train_losses_items_anyOf_i5"></a>58.9.1.6. Property `ReduxOptions > train > losses > losses items > anyOf > charbonnierloss`

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
| + [reduction](#train_losses_items_anyOf_i5_reduction )     | No      | string           | No         | -          | -                 |
| + [eps](#train_losses_items_anyOf_i5_eps )                 | No      | number           | No         | -          | -                 |

###### <a name="train_losses_items_anyOf_i5_type"></a>58.9.1.6.1. Property `ReduxOptions > train > losses > losses items > anyOf > charbonnierloss > type`

|              |                    |
| ------------ | ------------------ |
| **Type**     | `enum (of string)` |
| **Required** | Yes                |

Must be one of:
* "charbonnierloss"

###### <a name="train_losses_items_anyOf_i5_loss_weight"></a>58.9.1.6.2. Property `ReduxOptions > train > losses > losses items > anyOf > charbonnierloss > loss_weight`

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | Yes      |

###### <a name="train_losses_items_anyOf_i5_reduction"></a>58.9.1.6.3. Property `ReduxOptions > train > losses > losses items > anyOf > charbonnierloss > reduction`

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | Yes      |

###### <a name="train_losses_items_anyOf_i5_eps"></a>58.9.1.6.4. Property `ReduxOptions > train > losses > losses items > anyOf > charbonnierloss > eps`

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | Yes      |

##### <a name="train_losses_items_anyOf_i6"></a>58.9.1.7. Property `ReduxOptions > train > losses > losses items > anyOf > colorloss`

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
| + [criterion](#train_losses_items_anyOf_i6_criterion )     | No      | string           | No         | -          | -                 |
| + [loss_weight](#train_losses_items_anyOf_i6_loss_weight ) | No      | number           | No         | -          | -                 |
| + [scale](#train_losses_items_anyOf_i6_scale )             | No      | integer          | No         | -          | -                 |

###### <a name="train_losses_items_anyOf_i6_type"></a>58.9.1.7.1. Property `ReduxOptions > train > losses > losses items > anyOf > colorloss > type`

|              |                    |
| ------------ | ------------------ |
| **Type**     | `enum (of string)` |
| **Required** | Yes                |

Must be one of:
* "colorloss"

###### <a name="train_losses_items_anyOf_i6_criterion"></a>58.9.1.7.2. Property `ReduxOptions > train > losses > losses items > anyOf > colorloss > criterion`

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | Yes      |

###### <a name="train_losses_items_anyOf_i6_loss_weight"></a>58.9.1.7.3. Property `ReduxOptions > train > losses > losses items > anyOf > colorloss > loss_weight`

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | Yes      |

###### <a name="train_losses_items_anyOf_i6_scale"></a>58.9.1.7.4. Property `ReduxOptions > train > losses > losses items > anyOf > colorloss > scale`

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | Yes       |

##### <a name="train_losses_items_anyOf_i7"></a>58.9.1.8. Property `ReduxOptions > train > losses > losses items > anyOf > averageloss`

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
| + [criterion](#train_losses_items_anyOf_i7_criterion )     | No      | string           | No         | -          | -                 |
| + [loss_weight](#train_losses_items_anyOf_i7_loss_weight ) | No      | number           | No         | -          | -                 |
| + [scale](#train_losses_items_anyOf_i7_scale )             | No      | integer          | No         | -          | -                 |

###### <a name="train_losses_items_anyOf_i7_type"></a>58.9.1.8.1. Property `ReduxOptions > train > losses > losses items > anyOf > averageloss > type`

|              |                    |
| ------------ | ------------------ |
| **Type**     | `enum (of string)` |
| **Required** | Yes                |

Must be one of:
* "averageloss"

###### <a name="train_losses_items_anyOf_i7_criterion"></a>58.9.1.8.2. Property `ReduxOptions > train > losses > losses items > anyOf > averageloss > criterion`

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | Yes      |

###### <a name="train_losses_items_anyOf_i7_loss_weight"></a>58.9.1.8.3. Property `ReduxOptions > train > losses > losses items > anyOf > averageloss > loss_weight`

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | Yes      |

###### <a name="train_losses_items_anyOf_i7_scale"></a>58.9.1.8.4. Property `ReduxOptions > train > losses > losses items > anyOf > averageloss > scale`

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | Yes       |

##### <a name="train_losses_items_anyOf_i8"></a>58.9.1.9. Property `ReduxOptions > train > losses > losses items > anyOf > bicubicloss`

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
| + [criterion](#train_losses_items_anyOf_i8_criterion )     | No      | string           | No         | -          | -                 |
| + [loss_weight](#train_losses_items_anyOf_i8_loss_weight ) | No      | number           | No         | -          | -                 |
| + [scale](#train_losses_items_anyOf_i8_scale )             | No      | integer          | No         | -          | -                 |

###### <a name="train_losses_items_anyOf_i8_type"></a>58.9.1.9.1. Property `ReduxOptions > train > losses > losses items > anyOf > bicubicloss > type`

|              |                    |
| ------------ | ------------------ |
| **Type**     | `enum (of string)` |
| **Required** | Yes                |

Must be one of:
* "bicubicloss"

###### <a name="train_losses_items_anyOf_i8_criterion"></a>58.9.1.9.2. Property `ReduxOptions > train > losses > losses items > anyOf > bicubicloss > criterion`

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | Yes      |

###### <a name="train_losses_items_anyOf_i8_loss_weight"></a>58.9.1.9.3. Property `ReduxOptions > train > losses > losses items > anyOf > bicubicloss > loss_weight`

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | Yes      |

###### <a name="train_losses_items_anyOf_i8_scale"></a>58.9.1.9.4. Property `ReduxOptions > train > losses > losses items > anyOf > bicubicloss > scale`

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | Yes       |

##### <a name="train_losses_items_anyOf_i9"></a>58.9.1.10. Property `ReduxOptions > train > losses > losses items > anyOf > lumaloss`

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
| + [criterion](#train_losses_items_anyOf_i9_criterion )     | No      | string           | No         | -          | -                 |
| + [loss_weight](#train_losses_items_anyOf_i9_loss_weight ) | No      | number           | No         | -          | -                 |

###### <a name="train_losses_items_anyOf_i9_type"></a>58.9.1.10.1. Property `ReduxOptions > train > losses > losses items > anyOf > lumaloss > type`

|              |                    |
| ------------ | ------------------ |
| **Type**     | `enum (of string)` |
| **Required** | Yes                |

Must be one of:
* "lumaloss"

###### <a name="train_losses_items_anyOf_i9_criterion"></a>58.9.1.10.2. Property `ReduxOptions > train > losses > losses items > anyOf > lumaloss > criterion`

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | Yes      |

###### <a name="train_losses_items_anyOf_i9_loss_weight"></a>58.9.1.10.3. Property `ReduxOptions > train > losses > losses items > anyOf > lumaloss > loss_weight`

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | Yes      |

##### <a name="train_losses_items_anyOf_i10"></a>58.9.1.11. Property `ReduxOptions > train > losses > losses items > anyOf > hsluvloss`

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
| + [criterion](#train_losses_items_anyOf_i10_criterion )     | No      | string           | No         | -          | -                 |
| + [loss_weight](#train_losses_items_anyOf_i10_loss_weight ) | No      | number           | No         | -          | -                 |

###### <a name="train_losses_items_anyOf_i10_type"></a>58.9.1.11.1. Property `ReduxOptions > train > losses > losses items > anyOf > hsluvloss > type`

|              |                    |
| ------------ | ------------------ |
| **Type**     | `enum (of string)` |
| **Required** | Yes                |

Must be one of:
* "hsluvloss"

###### <a name="train_losses_items_anyOf_i10_criterion"></a>58.9.1.11.2. Property `ReduxOptions > train > losses > losses items > anyOf > hsluvloss > criterion`

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | Yes      |

###### <a name="train_losses_items_anyOf_i10_loss_weight"></a>58.9.1.11.3. Property `ReduxOptions > train > losses > losses items > anyOf > hsluvloss > loss_weight`

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | Yes      |

##### <a name="train_losses_items_anyOf_i11"></a>58.9.1.12. Property `ReduxOptions > train > losses > losses items > anyOf > contextualloss`

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
| + [layer_weights](#train_losses_items_anyOf_i11_layer_weights ) | No      | Combination      | No         | -          | -                 |
| + [crop_quarter](#train_losses_items_anyOf_i11_crop_quarter )   | No      | boolean          | No         | -          | -                 |
| + [max_1d_size](#train_losses_items_anyOf_i11_max_1d_size )     | No      | integer          | No         | -          | -                 |
| + [distance_type](#train_losses_items_anyOf_i11_distance_type ) | No      | string           | No         | -          | -                 |
| + [b](#train_losses_items_anyOf_i11_b )                         | No      | number           | No         | -          | -                 |
| + [band_width](#train_losses_items_anyOf_i11_band_width )       | No      | number           | No         | -          | -                 |
| + [use_vgg](#train_losses_items_anyOf_i11_use_vgg )             | No      | boolean          | No         | -          | -                 |
| + [net](#train_losses_items_anyOf_i11_net )                     | No      | string           | No         | -          | -                 |
| + [calc_type](#train_losses_items_anyOf_i11_calc_type )         | No      | string           | No         | -          | -                 |
| + [z_norm](#train_losses_items_anyOf_i11_z_norm )               | No      | boolean          | No         | -          | -                 |

###### <a name="train_losses_items_anyOf_i11_type"></a>58.9.1.12.1. Property `ReduxOptions > train > losses > losses items > anyOf > contextualloss > type`

|              |                    |
| ------------ | ------------------ |
| **Type**     | `enum (of string)` |
| **Required** | Yes                |

Must be one of:
* "contextualloss"

###### <a name="train_losses_items_anyOf_i11_loss_weight"></a>58.9.1.12.2. Property `ReduxOptions > train > losses > losses items > anyOf > contextualloss > loss_weight`

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | Yes      |

###### <a name="train_losses_items_anyOf_i11_layer_weights"></a>58.9.1.12.3. Property `ReduxOptions > train > losses > losses items > anyOf > contextualloss > layer_weights`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `combining`      |
| **Required**              | Yes              |
| **Additional properties** | Any type allowed |

| Any of(Option)                                                 |
| -------------------------------------------------------------- |
| [item 0](#train_losses_items_anyOf_i11_layer_weights_anyOf_i0) |
| [item 1](#train_losses_items_anyOf_i11_layer_weights_anyOf_i1) |

###### <a name="train_losses_items_anyOf_i11_layer_weights_anyOf_i0"></a>58.9.1.12.3.1. Property `ReduxOptions > train > losses > losses items > anyOf > contextualloss > layer_weights > anyOf > item 0`

|                           |                                                                                                                                  |
| ------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| **Type**                  | `object`                                                                                                                         |
| **Required**              | No                                                                                                                               |
| **Additional properties** | [Each additional property must conform to the schema](#train_losses_items_anyOf_i11_layer_weights_anyOf_i0_additionalProperties) |

| Property                                                                         | Pattern | Type   | Deprecated | Definition | Title/Description |
| -------------------------------------------------------------------------------- | ------- | ------ | ---------- | ---------- | ----------------- |
| - [](#train_losses_items_anyOf_i11_layer_weights_anyOf_i0_additionalProperties ) | No      | number | No         | -          | -                 |

###### <a name="train_losses_items_anyOf_i11_layer_weights_anyOf_i0_additionalProperties"></a>58.9.1.12.3.1.1. Property `ReduxOptions > train > losses > losses items > anyOf > contextualloss > layer_weights > anyOf > item 0 > additionalProperties`

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |

###### <a name="train_losses_items_anyOf_i11_layer_weights_anyOf_i1"></a>58.9.1.12.3.2. Property `ReduxOptions > train > losses > losses items > anyOf > contextualloss > layer_weights > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

###### <a name="train_losses_items_anyOf_i11_crop_quarter"></a>58.9.1.12.4. Property `ReduxOptions > train > losses > losses items > anyOf > contextualloss > crop_quarter`

|              |           |
| ------------ | --------- |
| **Type**     | `boolean` |
| **Required** | Yes       |

###### <a name="train_losses_items_anyOf_i11_max_1d_size"></a>58.9.1.12.5. Property `ReduxOptions > train > losses > losses items > anyOf > contextualloss > max_1d_size`

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | Yes       |

###### <a name="train_losses_items_anyOf_i11_distance_type"></a>58.9.1.12.6. Property `ReduxOptions > train > losses > losses items > anyOf > contextualloss > distance_type`

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | Yes      |

###### <a name="train_losses_items_anyOf_i11_b"></a>58.9.1.12.7. Property `ReduxOptions > train > losses > losses items > anyOf > contextualloss > b`

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | Yes      |

###### <a name="train_losses_items_anyOf_i11_band_width"></a>58.9.1.12.8. Property `ReduxOptions > train > losses > losses items > anyOf > contextualloss > band_width`

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | Yes      |

###### <a name="train_losses_items_anyOf_i11_use_vgg"></a>58.9.1.12.9. Property `ReduxOptions > train > losses > losses items > anyOf > contextualloss > use_vgg`

|              |           |
| ------------ | --------- |
| **Type**     | `boolean` |
| **Required** | Yes       |

###### <a name="train_losses_items_anyOf_i11_net"></a>58.9.1.12.10. Property `ReduxOptions > train > losses > losses items > anyOf > contextualloss > net`

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | Yes      |

###### <a name="train_losses_items_anyOf_i11_calc_type"></a>58.9.1.12.11. Property `ReduxOptions > train > losses > losses items > anyOf > contextualloss > calc_type`

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | Yes      |

###### <a name="train_losses_items_anyOf_i11_z_norm"></a>58.9.1.12.12. Property `ReduxOptions > train > losses > losses items > anyOf > contextualloss > z_norm`

|              |           |
| ------------ | --------- |
| **Type**     | `boolean` |
| **Required** | Yes       |

##### <a name="train_losses_items_anyOf_i12"></a>58.9.1.13. Property `ReduxOptions > train > losses > losses items > anyOf > distsloss`

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
| + [as_loss](#train_losses_items_anyOf_i12_as_loss )               | No      | boolean          | No         | -          | -                 |
| + [loss_weight](#train_losses_items_anyOf_i12_loss_weight )       | No      | number           | No         | -          | -                 |
| + [load_weights](#train_losses_items_anyOf_i12_load_weights )     | No      | boolean          | No         | -          | -                 |
| + [use_input_norm](#train_losses_items_anyOf_i12_use_input_norm ) | No      | boolean          | No         | -          | -                 |
| + [clip_min](#train_losses_items_anyOf_i12_clip_min )             | No      | integer          | No         | -          | -                 |

###### <a name="train_losses_items_anyOf_i12_type"></a>58.9.1.13.1. Property `ReduxOptions > train > losses > losses items > anyOf > distsloss > type`

|              |                    |
| ------------ | ------------------ |
| **Type**     | `enum (of string)` |
| **Required** | Yes                |

Must be one of:
* "distsloss"

###### <a name="train_losses_items_anyOf_i12_as_loss"></a>58.9.1.13.2. Property `ReduxOptions > train > losses > losses items > anyOf > distsloss > as_loss`

|              |           |
| ------------ | --------- |
| **Type**     | `boolean` |
| **Required** | Yes       |

###### <a name="train_losses_items_anyOf_i12_loss_weight"></a>58.9.1.13.3. Property `ReduxOptions > train > losses > losses items > anyOf > distsloss > loss_weight`

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | Yes      |

###### <a name="train_losses_items_anyOf_i12_load_weights"></a>58.9.1.13.4. Property `ReduxOptions > train > losses > losses items > anyOf > distsloss > load_weights`

|              |           |
| ------------ | --------- |
| **Type**     | `boolean` |
| **Required** | Yes       |

###### <a name="train_losses_items_anyOf_i12_use_input_norm"></a>58.9.1.13.5. Property `ReduxOptions > train > losses > losses items > anyOf > distsloss > use_input_norm`

|              |           |
| ------------ | --------- |
| **Type**     | `boolean` |
| **Required** | Yes       |

###### <a name="train_losses_items_anyOf_i12_clip_min"></a>58.9.1.13.6. Property `ReduxOptions > train > losses > losses items > anyOf > distsloss > clip_min`

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | Yes       |

##### <a name="train_losses_items_anyOf_i13"></a>58.9.1.14. Property `ReduxOptions > train > losses > losses items > anyOf > dsdloss`

**Title:** dsdloss

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `object`         |
| **Required**              | No               |
| **Additional properties** | Any type allowed |
| **Defined in**            | #/$defs/dsdloss  |

| Property                                                                    | Pattern | Type             | Deprecated | Definition | Title/Description |
| --------------------------------------------------------------------------- | ------- | ---------------- | ---------- | ---------- | ----------------- |
| + [type](#train_losses_items_anyOf_i13_type )                               | No      | enum (of string) | No         | -          | -                 |
| + [features_to_compute](#train_losses_items_anyOf_i13_features_to_compute ) | No      | array of string  | No         | -          | -                 |
| + [scales](#train_losses_items_anyOf_i13_scales )                           | No      | array of number  | No         | -          | -                 |
| + [criterion](#train_losses_items_anyOf_i13_criterion )                     | No      | string           | No         | -          | -                 |
| + [shave_edge](#train_losses_items_anyOf_i13_shave_edge )                   | No      | Combination      | No         | -          | -                 |
| + [loss_weight](#train_losses_items_anyOf_i13_loss_weight )                 | No      | number           | No         | -          | -                 |

###### <a name="train_losses_items_anyOf_i13_type"></a>58.9.1.14.1. Property `ReduxOptions > train > losses > losses items > anyOf > dsdloss > type`

|              |                    |
| ------------ | ------------------ |
| **Type**     | `enum (of string)` |
| **Required** | Yes                |

Must be one of:
* "dsdloss"

###### <a name="train_losses_items_anyOf_i13_features_to_compute"></a>58.9.1.14.2. Property `ReduxOptions > train > losses > losses items > anyOf > dsdloss > features_to_compute`

|              |                   |
| ------------ | ----------------- |
| **Type**     | `array of string` |
| **Required** | Yes               |

|                      | Array restrictions |
| -------------------- | ------------------ |
| **Min items**        | N/A                |
| **Max items**        | N/A                |
| **Items unicity**    | False              |
| **Additional items** | False              |
| **Tuple validation** | See below          |

| Each item of this array must be                                                      | Description |
| ------------------------------------------------------------------------------------ | ----------- |
| [features_to_compute items](#train_losses_items_anyOf_i13_features_to_compute_items) | -           |

###### <a name="train_losses_items_anyOf_i13_features_to_compute_items"></a>58.9.1.14.2.1. ReduxOptions > train > losses > losses items > anyOf > dsdloss > features_to_compute > features_to_compute items

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | No       |

###### <a name="train_losses_items_anyOf_i13_scales"></a>58.9.1.14.3. Property `ReduxOptions > train > losses > losses items > anyOf > dsdloss > scales`

|              |                   |
| ------------ | ----------------- |
| **Type**     | `array of number` |
| **Required** | Yes               |

|                      | Array restrictions |
| -------------------- | ------------------ |
| **Min items**        | N/A                |
| **Max items**        | N/A                |
| **Items unicity**    | False              |
| **Additional items** | False              |
| **Tuple validation** | See below          |

| Each item of this array must be                            | Description |
| ---------------------------------------------------------- | ----------- |
| [scales items](#train_losses_items_anyOf_i13_scales_items) | -           |

###### <a name="train_losses_items_anyOf_i13_scales_items"></a>58.9.1.14.3.1. ReduxOptions > train > losses > losses items > anyOf > dsdloss > scales > scales items

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |

###### <a name="train_losses_items_anyOf_i13_criterion"></a>58.9.1.14.4. Property `ReduxOptions > train > losses > losses items > anyOf > dsdloss > criterion`

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | Yes      |

###### <a name="train_losses_items_anyOf_i13_shave_edge"></a>58.9.1.14.5. Property `ReduxOptions > train > losses > losses items > anyOf > dsdloss > shave_edge`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `combining`      |
| **Required**              | Yes              |
| **Additional properties** | Any type allowed |

| Any of(Option)                                              |
| ----------------------------------------------------------- |
| [item 0](#train_losses_items_anyOf_i13_shave_edge_anyOf_i0) |
| [item 1](#train_losses_items_anyOf_i13_shave_edge_anyOf_i1) |

###### <a name="train_losses_items_anyOf_i13_shave_edge_anyOf_i0"></a>58.9.1.14.5.1. Property `ReduxOptions > train > losses > losses items > anyOf > dsdloss > shave_edge > anyOf > item 0`

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | No        |

###### <a name="train_losses_items_anyOf_i13_shave_edge_anyOf_i1"></a>58.9.1.14.5.2. Property `ReduxOptions > train > losses > losses items > anyOf > dsdloss > shave_edge > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

###### <a name="train_losses_items_anyOf_i13_loss_weight"></a>58.9.1.14.6. Property `ReduxOptions > train > losses > losses items > anyOf > dsdloss > loss_weight`

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | Yes      |

##### <a name="train_losses_items_anyOf_i14"></a>58.9.1.15. Property `ReduxOptions > train > losses > losses items > anyOf > ffloss`

**Title:** ffloss

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `object`         |
| **Required**              | No               |
| **Additional properties** | Any type allowed |
| **Defined in**            | #/$defs/ffloss   |

| Property                                                      | Pattern | Type             | Deprecated | Definition | Title/Description |
| ------------------------------------------------------------- | ------- | ---------------- | ---------- | ---------- | ----------------- |
| + [type](#train_losses_items_anyOf_i14_type )                 | No      | enum (of string) | No         | -          | -                 |
| + [loss_weight](#train_losses_items_anyOf_i14_loss_weight )   | No      | number           | No         | -          | -                 |
| + [alpha](#train_losses_items_anyOf_i14_alpha )               | No      | number           | No         | -          | -                 |
| + [patch_factor](#train_losses_items_anyOf_i14_patch_factor ) | No      | integer          | No         | -          | -                 |
| + [ave_spectrum](#train_losses_items_anyOf_i14_ave_spectrum ) | No      | boolean          | No         | -          | -                 |
| + [log_matrix](#train_losses_items_anyOf_i14_log_matrix )     | No      | boolean          | No         | -          | -                 |
| + [batch_matrix](#train_losses_items_anyOf_i14_batch_matrix ) | No      | boolean          | No         | -          | -                 |

###### <a name="train_losses_items_anyOf_i14_type"></a>58.9.1.15.1. Property `ReduxOptions > train > losses > losses items > anyOf > ffloss > type`

|              |                    |
| ------------ | ------------------ |
| **Type**     | `enum (of string)` |
| **Required** | Yes                |

Must be one of:
* "ffloss"

###### <a name="train_losses_items_anyOf_i14_loss_weight"></a>58.9.1.15.2. Property `ReduxOptions > train > losses > losses items > anyOf > ffloss > loss_weight`

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | Yes      |

###### <a name="train_losses_items_anyOf_i14_alpha"></a>58.9.1.15.3. Property `ReduxOptions > train > losses > losses items > anyOf > ffloss > alpha`

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | Yes      |

###### <a name="train_losses_items_anyOf_i14_patch_factor"></a>58.9.1.15.4. Property `ReduxOptions > train > losses > losses items > anyOf > ffloss > patch_factor`

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | Yes       |

###### <a name="train_losses_items_anyOf_i14_ave_spectrum"></a>58.9.1.15.5. Property `ReduxOptions > train > losses > losses items > anyOf > ffloss > ave_spectrum`

|              |           |
| ------------ | --------- |
| **Type**     | `boolean` |
| **Required** | Yes       |

###### <a name="train_losses_items_anyOf_i14_log_matrix"></a>58.9.1.15.6. Property `ReduxOptions > train > losses > losses items > anyOf > ffloss > log_matrix`

|              |           |
| ------------ | --------- |
| **Type**     | `boolean` |
| **Required** | Yes       |

###### <a name="train_losses_items_anyOf_i14_batch_matrix"></a>58.9.1.15.7. Property `ReduxOptions > train > losses > losses items > anyOf > ffloss > batch_matrix`

|              |           |
| ------------ | --------- |
| **Type**     | `boolean` |
| **Required** | Yes       |

##### <a name="train_losses_items_anyOf_i15"></a>58.9.1.16. Property `ReduxOptions > train > losses > losses items > anyOf > ldlloss`

**Title:** ldlloss

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `object`         |
| **Required**              | No               |
| **Additional properties** | Any type allowed |
| **Defined in**            | #/$defs/ldlloss  |

| Property                                                    | Pattern | Type             | Deprecated | Definition | Title/Description |
| ----------------------------------------------------------- | ------- | ---------------- | ---------- | ---------- | ----------------- |
| + [type](#train_losses_items_anyOf_i15_type )               | No      | enum (of string) | No         | -          | -                 |
| + [criterion](#train_losses_items_anyOf_i15_criterion )     | No      | string           | No         | -          | -                 |
| + [loss_weight](#train_losses_items_anyOf_i15_loss_weight ) | No      | number           | No         | -          | -                 |

###### <a name="train_losses_items_anyOf_i15_type"></a>58.9.1.16.1. Property `ReduxOptions > train > losses > losses items > anyOf > ldlloss > type`

|              |                    |
| ------------ | ------------------ |
| **Type**     | `enum (of string)` |
| **Required** | Yes                |

Must be one of:
* "ldlloss"

###### <a name="train_losses_items_anyOf_i15_criterion"></a>58.9.1.16.2. Property `ReduxOptions > train > losses > losses items > anyOf > ldlloss > criterion`

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | Yes      |

###### <a name="train_losses_items_anyOf_i15_loss_weight"></a>58.9.1.16.3. Property `ReduxOptions > train > losses > losses items > anyOf > ldlloss > loss_weight`

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | Yes      |

##### <a name="train_losses_items_anyOf_i16"></a>58.9.1.17. Property `ReduxOptions > train > losses > losses items > anyOf > mssimloss`

**Title:** mssimloss

|                           |                   |
| ------------------------- | ----------------- |
| **Type**                  | `object`          |
| **Required**              | No                |
| **Additional properties** | Any type allowed  |
| **Defined in**            | #/$defs/mssimloss |

| Property                                                      | Pattern | Type             | Deprecated | Definition | Title/Description |
| ------------------------------------------------------------- | ------- | ---------------- | ---------- | ---------- | ----------------- |
| + [type](#train_losses_items_anyOf_i16_type )                 | No      | enum (of string) | No         | -          | -                 |
| + [window_size](#train_losses_items_anyOf_i16_window_size )   | No      | integer          | No         | -          | -                 |
| + [in_channels](#train_losses_items_anyOf_i16_in_channels )   | No      | integer          | No         | -          | -                 |
| + [sigma](#train_losses_items_anyOf_i16_sigma )               | No      | number           | No         | -          | -                 |
| + [k1](#train_losses_items_anyOf_i16_k1 )                     | No      | number           | No         | -          | -                 |
| + [k2](#train_losses_items_anyOf_i16_k2 )                     | No      | number           | No         | -          | -                 |
| + [l](#train_losses_items_anyOf_i16_l )                       | No      | integer          | No         | -          | -                 |
| + [padding](#train_losses_items_anyOf_i16_padding )           | No      | Combination      | No         | -          | -                 |
| + [cosim](#train_losses_items_anyOf_i16_cosim )               | No      | boolean          | No         | -          | -                 |
| + [cosim_lambda](#train_losses_items_anyOf_i16_cosim_lambda ) | No      | integer          | No         | -          | -                 |
| + [loss_weight](#train_losses_items_anyOf_i16_loss_weight )   | No      | number           | No         | -          | -                 |

###### <a name="train_losses_items_anyOf_i16_type"></a>58.9.1.17.1. Property `ReduxOptions > train > losses > losses items > anyOf > mssimloss > type`

|              |                    |
| ------------ | ------------------ |
| **Type**     | `enum (of string)` |
| **Required** | Yes                |

Must be one of:
* "mssimloss"

###### <a name="train_losses_items_anyOf_i16_window_size"></a>58.9.1.17.2. Property `ReduxOptions > train > losses > losses items > anyOf > mssimloss > window_size`

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | Yes       |

###### <a name="train_losses_items_anyOf_i16_in_channels"></a>58.9.1.17.3. Property `ReduxOptions > train > losses > losses items > anyOf > mssimloss > in_channels`

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | Yes       |

###### <a name="train_losses_items_anyOf_i16_sigma"></a>58.9.1.17.4. Property `ReduxOptions > train > losses > losses items > anyOf > mssimloss > sigma`

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | Yes      |

###### <a name="train_losses_items_anyOf_i16_k1"></a>58.9.1.17.5. Property `ReduxOptions > train > losses > losses items > anyOf > mssimloss > k1`

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | Yes      |

###### <a name="train_losses_items_anyOf_i16_k2"></a>58.9.1.17.6. Property `ReduxOptions > train > losses > losses items > anyOf > mssimloss > k2`

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | Yes      |

###### <a name="train_losses_items_anyOf_i16_l"></a>58.9.1.17.7. Property `ReduxOptions > train > losses > losses items > anyOf > mssimloss > l`

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | Yes       |

###### <a name="train_losses_items_anyOf_i16_padding"></a>58.9.1.17.8. Property `ReduxOptions > train > losses > losses items > anyOf > mssimloss > padding`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `combining`      |
| **Required**              | Yes              |
| **Additional properties** | Any type allowed |

| Any of(Option)                                           |
| -------------------------------------------------------- |
| [item 0](#train_losses_items_anyOf_i16_padding_anyOf_i0) |
| [item 1](#train_losses_items_anyOf_i16_padding_anyOf_i1) |

###### <a name="train_losses_items_anyOf_i16_padding_anyOf_i0"></a>58.9.1.17.8.1. Property `ReduxOptions > train > losses > losses items > anyOf > mssimloss > padding > anyOf > item 0`

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | No        |

###### <a name="train_losses_items_anyOf_i16_padding_anyOf_i1"></a>58.9.1.17.8.2. Property `ReduxOptions > train > losses > losses items > anyOf > mssimloss > padding > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

###### <a name="train_losses_items_anyOf_i16_cosim"></a>58.9.1.17.9. Property `ReduxOptions > train > losses > losses items > anyOf > mssimloss > cosim`

|              |           |
| ------------ | --------- |
| **Type**     | `boolean` |
| **Required** | Yes       |

###### <a name="train_losses_items_anyOf_i16_cosim_lambda"></a>58.9.1.17.10. Property `ReduxOptions > train > losses > losses items > anyOf > mssimloss > cosim_lambda`

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | Yes       |

###### <a name="train_losses_items_anyOf_i16_loss_weight"></a>58.9.1.17.11. Property `ReduxOptions > train > losses > losses items > anyOf > mssimloss > loss_weight`

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | Yes      |

##### <a name="train_losses_items_anyOf_i17"></a>58.9.1.18. Property `ReduxOptions > train > losses > losses items > anyOf > msssiml1loss`

**Title:** msssiml1loss

|                           |                      |
| ------------------------- | -------------------- |
| **Type**                  | `object`             |
| **Required**              | No                   |
| **Additional properties** | Any type allowed     |
| **Defined in**            | #/$defs/msssiml1loss |

| Property                                                            | Pattern | Type             | Deprecated | Definition | Title/Description |
| ------------------------------------------------------------------- | ------- | ---------------- | ---------- | ---------- | ----------------- |
| + [type](#train_losses_items_anyOf_i17_type )                       | No      | enum (of string) | No         | -          | -                 |
| + [gaussian_sigmas](#train_losses_items_anyOf_i17_gaussian_sigmas ) | No      | Combination      | No         | -          | -                 |
| + [data_range](#train_losses_items_anyOf_i17_data_range )           | No      | number           | No         | -          | -                 |
| + [k](#train_losses_items_anyOf_i17_k )                             | No      | array            | No         | -          | -                 |
| + [alpha](#train_losses_items_anyOf_i17_alpha )                     | No      | number           | No         | -          | -                 |
| + [cuda_dev](#train_losses_items_anyOf_i17_cuda_dev )               | No      | integer          | No         | -          | -                 |
| + [loss_weight](#train_losses_items_anyOf_i17_loss_weight )         | No      | number           | No         | -          | -                 |

###### <a name="train_losses_items_anyOf_i17_type"></a>58.9.1.18.1. Property `ReduxOptions > train > losses > losses items > anyOf > msssiml1loss > type`

|              |                    |
| ------------ | ------------------ |
| **Type**     | `enum (of string)` |
| **Required** | Yes                |

Must be one of:
* "msssiml1loss"

###### <a name="train_losses_items_anyOf_i17_gaussian_sigmas"></a>58.9.1.18.2. Property `ReduxOptions > train > losses > losses items > anyOf > msssiml1loss > gaussian_sigmas`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `combining`      |
| **Required**              | Yes              |
| **Additional properties** | Any type allowed |

| Any of(Option)                                                   |
| ---------------------------------------------------------------- |
| [item 0](#train_losses_items_anyOf_i17_gaussian_sigmas_anyOf_i0) |
| [item 1](#train_losses_items_anyOf_i17_gaussian_sigmas_anyOf_i1) |

###### <a name="train_losses_items_anyOf_i17_gaussian_sigmas_anyOf_i0"></a>58.9.1.18.2.1. Property `ReduxOptions > train > losses > losses items > anyOf > msssiml1loss > gaussian_sigmas > anyOf > item 0`

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
| [item 0 items](#train_losses_items_anyOf_i17_gaussian_sigmas_anyOf_i0_items) | -           |

###### <a name="train_losses_items_anyOf_i17_gaussian_sigmas_anyOf_i0_items"></a>58.9.1.18.2.1.1. ReduxOptions > train > losses > losses items > anyOf > msssiml1loss > gaussian_sigmas > anyOf > item 0 > item 0 items

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |

###### <a name="train_losses_items_anyOf_i17_gaussian_sigmas_anyOf_i1"></a>58.9.1.18.2.2. Property `ReduxOptions > train > losses > losses items > anyOf > msssiml1loss > gaussian_sigmas > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

###### <a name="train_losses_items_anyOf_i17_data_range"></a>58.9.1.18.3. Property `ReduxOptions > train > losses > losses items > anyOf > msssiml1loss > data_range`

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | Yes      |

###### <a name="train_losses_items_anyOf_i17_k"></a>58.9.1.18.4. Property `ReduxOptions > train > losses > losses items > anyOf > msssiml1loss > k`

|              |         |
| ------------ | ------- |
| **Type**     | `array` |
| **Required** | Yes     |

|                      | Array restrictions |
| -------------------- | ------------------ |
| **Min items**        | 2                  |
| **Max items**        | 2                  |
| **Items unicity**    | False              |
| **Additional items** | False              |
| **Tuple validation** | See below          |

| Each item of this array must be                      | Description |
| ---------------------------------------------------- | ----------- |
| [k item 0](#train_losses_items_anyOf_i17_k_items_i0) | -           |
| [k item 1](#train_losses_items_anyOf_i17_k_items_i1) | -           |

###### <a name="autogenerated_heading_36"></a>58.9.1.18.4.1. ReduxOptions > train > losses > losses items > anyOf > msssiml1loss > k > k item 0

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |

###### <a name="autogenerated_heading_37"></a>58.9.1.18.4.2. ReduxOptions > train > losses > losses items > anyOf > msssiml1loss > k > k item 1

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |

###### <a name="train_losses_items_anyOf_i17_alpha"></a>58.9.1.18.5. Property `ReduxOptions > train > losses > losses items > anyOf > msssiml1loss > alpha`

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | Yes      |

###### <a name="train_losses_items_anyOf_i17_cuda_dev"></a>58.9.1.18.6. Property `ReduxOptions > train > losses > losses items > anyOf > msssiml1loss > cuda_dev`

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | Yes       |

###### <a name="train_losses_items_anyOf_i17_loss_weight"></a>58.9.1.18.7. Property `ReduxOptions > train > losses > losses items > anyOf > msssiml1loss > loss_weight`

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | Yes      |

##### <a name="train_losses_items_anyOf_i18"></a>58.9.1.19. Property `ReduxOptions > train > losses > losses items > anyOf > nccloss`

**Title:** nccloss

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `object`         |
| **Required**              | No               |
| **Additional properties** | Any type allowed |
| **Defined in**            | #/$defs/nccloss  |

| Property                                                    | Pattern | Type             | Deprecated | Definition | Title/Description |
| ----------------------------------------------------------- | ------- | ---------------- | ---------- | ---------- | ----------------- |
| + [type](#train_losses_items_anyOf_i18_type )               | No      | enum (of string) | No         | -          | -                 |
| + [loss_weight](#train_losses_items_anyOf_i18_loss_weight ) | No      | number           | No         | -          | -                 |

###### <a name="train_losses_items_anyOf_i18_type"></a>58.9.1.19.1. Property `ReduxOptions > train > losses > losses items > anyOf > nccloss > type`

|              |                    |
| ------------ | ------------------ |
| **Type**     | `enum (of string)` |
| **Required** | Yes                |

Must be one of:
* "nccloss"

###### <a name="train_losses_items_anyOf_i18_loss_weight"></a>58.9.1.19.2. Property `ReduxOptions > train > losses > losses items > anyOf > nccloss > loss_weight`

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | Yes      |

##### <a name="train_losses_items_anyOf_i19"></a>58.9.1.20. Property `ReduxOptions > train > losses > losses items > anyOf > perceptualfp16loss`

**Title:** perceptualfp16loss

|                           |                            |
| ------------------------- | -------------------------- |
| **Type**                  | `object`                   |
| **Required**              | No                         |
| **Additional properties** | Any type allowed           |
| **Defined in**            | #/$defs/perceptualfp16loss |

| Property                                                            | Pattern | Type             | Deprecated | Definition | Title/Description |
| ------------------------------------------------------------------- | ------- | ---------------- | ---------- | ---------- | ----------------- |
| + [type](#train_losses_items_anyOf_i19_type )                       | No      | enum (of string) | No         | -          | -                 |
| + [layer_weights](#train_losses_items_anyOf_i19_layer_weights )     | No      | Combination      | No         | -          | -                 |
| + [w_lambda](#train_losses_items_anyOf_i19_w_lambda )               | No      | number           | No         | -          | -                 |
| + [loss_weight](#train_losses_items_anyOf_i19_loss_weight )         | No      | number           | No         | -          | -                 |
| + [alpha](#train_losses_items_anyOf_i19_alpha )                     | No      | Combination      | No         | -          | -                 |
| + [criterion](#train_losses_items_anyOf_i19_criterion )             | No      | enum (of string) | No         | -          | -                 |
| + [num_proj_fd](#train_losses_items_anyOf_i19_num_proj_fd )         | No      | integer          | No         | -          | -                 |
| + [phase_weight_fd](#train_losses_items_anyOf_i19_phase_weight_fd ) | No      | number           | No         | -          | -                 |
| + [stride_fd](#train_losses_items_anyOf_i19_stride_fd )             | No      | integer          | No         | -          | -                 |

###### <a name="train_losses_items_anyOf_i19_type"></a>58.9.1.20.1. Property `ReduxOptions > train > losses > losses items > anyOf > perceptualfp16loss > type`

|              |                    |
| ------------ | ------------------ |
| **Type**     | `enum (of string)` |
| **Required** | Yes                |

Must be one of:
* "perceptualfp16loss"

###### <a name="train_losses_items_anyOf_i19_layer_weights"></a>58.9.1.20.2. Property `ReduxOptions > train > losses > losses items > anyOf > perceptualfp16loss > layer_weights`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `combining`      |
| **Required**              | Yes              |
| **Additional properties** | Any type allowed |

| Any of(Option)                                                 |
| -------------------------------------------------------------- |
| [item 0](#train_losses_items_anyOf_i19_layer_weights_anyOf_i0) |
| [item 1](#train_losses_items_anyOf_i19_layer_weights_anyOf_i1) |

###### <a name="train_losses_items_anyOf_i19_layer_weights_anyOf_i0"></a>58.9.1.20.2.1. Property `ReduxOptions > train > losses > losses items > anyOf > perceptualfp16loss > layer_weights > anyOf > item 0`

|                           |                                                                                                                                  |
| ------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| **Type**                  | `object`                                                                                                                         |
| **Required**              | No                                                                                                                               |
| **Additional properties** | [Each additional property must conform to the schema](#train_losses_items_anyOf_i19_layer_weights_anyOf_i0_additionalProperties) |

| Property                                                                         | Pattern | Type   | Deprecated | Definition | Title/Description |
| -------------------------------------------------------------------------------- | ------- | ------ | ---------- | ---------- | ----------------- |
| - [](#train_losses_items_anyOf_i19_layer_weights_anyOf_i0_additionalProperties ) | No      | number | No         | -          | -                 |

###### <a name="train_losses_items_anyOf_i19_layer_weights_anyOf_i0_additionalProperties"></a>58.9.1.20.2.1.1. Property `ReduxOptions > train > losses > losses items > anyOf > perceptualfp16loss > layer_weights > anyOf > item 0 > additionalProperties`

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |

###### <a name="train_losses_items_anyOf_i19_layer_weights_anyOf_i1"></a>58.9.1.20.2.2. Property `ReduxOptions > train > losses > losses items > anyOf > perceptualfp16loss > layer_weights > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

###### <a name="train_losses_items_anyOf_i19_w_lambda"></a>58.9.1.20.3. Property `ReduxOptions > train > losses > losses items > anyOf > perceptualfp16loss > w_lambda`

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | Yes      |

###### <a name="train_losses_items_anyOf_i19_loss_weight"></a>58.9.1.20.4. Property `ReduxOptions > train > losses > losses items > anyOf > perceptualfp16loss > loss_weight`

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | Yes      |

###### <a name="train_losses_items_anyOf_i19_alpha"></a>58.9.1.20.5. Property `ReduxOptions > train > losses > losses items > anyOf > perceptualfp16loss > alpha`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `combining`      |
| **Required**              | Yes              |
| **Additional properties** | Any type allowed |

| Any of(Option)                                         |
| ------------------------------------------------------ |
| [item 0](#train_losses_items_anyOf_i19_alpha_anyOf_i0) |
| [item 1](#train_losses_items_anyOf_i19_alpha_anyOf_i1) |

###### <a name="train_losses_items_anyOf_i19_alpha_anyOf_i0"></a>58.9.1.20.5.1. Property `ReduxOptions > train > losses > losses items > anyOf > perceptualfp16loss > alpha > anyOf > item 0`

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

###### <a name="train_losses_items_anyOf_i19_alpha_anyOf_i0_items"></a>58.9.1.20.5.1.1. ReduxOptions > train > losses > losses items > anyOf > perceptualfp16loss > alpha > anyOf > item 0 > item 0 items

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |

###### <a name="train_losses_items_anyOf_i19_alpha_anyOf_i1"></a>58.9.1.20.5.2. Property `ReduxOptions > train > losses > losses items > anyOf > perceptualfp16loss > alpha > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

###### <a name="train_losses_items_anyOf_i19_criterion"></a>58.9.1.20.6. Property `ReduxOptions > train > losses > losses items > anyOf > perceptualfp16loss > criterion`

|              |                    |
| ------------ | ------------------ |
| **Type**     | `enum (of string)` |
| **Required** | Yes                |

Must be one of:
* "charbonnier"
* "fd"
* "fd+l1pd"
* "l1"
* "pd+l1"

###### <a name="train_losses_items_anyOf_i19_num_proj_fd"></a>58.9.1.20.7. Property `ReduxOptions > train > losses > losses items > anyOf > perceptualfp16loss > num_proj_fd`

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | Yes       |

###### <a name="train_losses_items_anyOf_i19_phase_weight_fd"></a>58.9.1.20.8. Property `ReduxOptions > train > losses > losses items > anyOf > perceptualfp16loss > phase_weight_fd`

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | Yes      |

###### <a name="train_losses_items_anyOf_i19_stride_fd"></a>58.9.1.20.9. Property `ReduxOptions > train > losses > losses items > anyOf > perceptualfp16loss > stride_fd`

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | Yes       |

##### <a name="train_losses_items_anyOf_i20"></a>58.9.1.21. Property `ReduxOptions > train > losses > losses items > anyOf > perceptualloss`

**Title:** perceptualloss

|                           |                        |
| ------------------------- | ---------------------- |
| **Type**                  | `object`               |
| **Required**              | No                     |
| **Additional properties** | Any type allowed       |
| **Defined in**            | #/$defs/perceptualloss |

| Property                                                            | Pattern | Type             | Deprecated | Definition | Title/Description |
| ------------------------------------------------------------------- | ------- | ---------------- | ---------- | ---------- | ----------------- |
| + [type](#train_losses_items_anyOf_i20_type )                       | No      | enum (of string) | No         | -          | -                 |
| + [layer_weights](#train_losses_items_anyOf_i20_layer_weights )     | No      | Combination      | No         | -          | -                 |
| + [w_lambda](#train_losses_items_anyOf_i20_w_lambda )               | No      | number           | No         | -          | -                 |
| + [loss_weight](#train_losses_items_anyOf_i20_loss_weight )         | No      | number           | No         | -          | -                 |
| + [alpha](#train_losses_items_anyOf_i20_alpha )                     | No      | Combination      | No         | -          | -                 |
| + [criterion](#train_losses_items_anyOf_i20_criterion )             | No      | enum (of string) | No         | -          | -                 |
| + [num_proj_fd](#train_losses_items_anyOf_i20_num_proj_fd )         | No      | integer          | No         | -          | -                 |
| + [phase_weight_fd](#train_losses_items_anyOf_i20_phase_weight_fd ) | No      | number           | No         | -          | -                 |
| + [stride_fd](#train_losses_items_anyOf_i20_stride_fd )             | No      | integer          | No         | -          | -                 |

###### <a name="train_losses_items_anyOf_i20_type"></a>58.9.1.21.1. Property `ReduxOptions > train > losses > losses items > anyOf > perceptualloss > type`

|              |                    |
| ------------ | ------------------ |
| **Type**     | `enum (of string)` |
| **Required** | Yes                |

Must be one of:
* "perceptualloss"

###### <a name="train_losses_items_anyOf_i20_layer_weights"></a>58.9.1.21.2. Property `ReduxOptions > train > losses > losses items > anyOf > perceptualloss > layer_weights`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `combining`      |
| **Required**              | Yes              |
| **Additional properties** | Any type allowed |

| Any of(Option)                                                 |
| -------------------------------------------------------------- |
| [item 0](#train_losses_items_anyOf_i20_layer_weights_anyOf_i0) |
| [item 1](#train_losses_items_anyOf_i20_layer_weights_anyOf_i1) |

###### <a name="train_losses_items_anyOf_i20_layer_weights_anyOf_i0"></a>58.9.1.21.2.1. Property `ReduxOptions > train > losses > losses items > anyOf > perceptualloss > layer_weights > anyOf > item 0`

|                           |                                                                                                                                  |
| ------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| **Type**                  | `object`                                                                                                                         |
| **Required**              | No                                                                                                                               |
| **Additional properties** | [Each additional property must conform to the schema](#train_losses_items_anyOf_i20_layer_weights_anyOf_i0_additionalProperties) |

| Property                                                                         | Pattern | Type   | Deprecated | Definition | Title/Description |
| -------------------------------------------------------------------------------- | ------- | ------ | ---------- | ---------- | ----------------- |
| - [](#train_losses_items_anyOf_i20_layer_weights_anyOf_i0_additionalProperties ) | No      | number | No         | -          | -                 |

###### <a name="train_losses_items_anyOf_i20_layer_weights_anyOf_i0_additionalProperties"></a>58.9.1.21.2.1.1. Property `ReduxOptions > train > losses > losses items > anyOf > perceptualloss > layer_weights > anyOf > item 0 > additionalProperties`

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |

###### <a name="train_losses_items_anyOf_i20_layer_weights_anyOf_i1"></a>58.9.1.21.2.2. Property `ReduxOptions > train > losses > losses items > anyOf > perceptualloss > layer_weights > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

###### <a name="train_losses_items_anyOf_i20_w_lambda"></a>58.9.1.21.3. Property `ReduxOptions > train > losses > losses items > anyOf > perceptualloss > w_lambda`

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | Yes      |

###### <a name="train_losses_items_anyOf_i20_loss_weight"></a>58.9.1.21.4. Property `ReduxOptions > train > losses > losses items > anyOf > perceptualloss > loss_weight`

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | Yes      |

###### <a name="train_losses_items_anyOf_i20_alpha"></a>58.9.1.21.5. Property `ReduxOptions > train > losses > losses items > anyOf > perceptualloss > alpha`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `combining`      |
| **Required**              | Yes              |
| **Additional properties** | Any type allowed |

| Any of(Option)                                         |
| ------------------------------------------------------ |
| [item 0](#train_losses_items_anyOf_i20_alpha_anyOf_i0) |
| [item 1](#train_losses_items_anyOf_i20_alpha_anyOf_i1) |

###### <a name="train_losses_items_anyOf_i20_alpha_anyOf_i0"></a>58.9.1.21.5.1. Property `ReduxOptions > train > losses > losses items > anyOf > perceptualloss > alpha > anyOf > item 0`

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
| [item 0 items](#train_losses_items_anyOf_i20_alpha_anyOf_i0_items) | -           |

###### <a name="train_losses_items_anyOf_i20_alpha_anyOf_i0_items"></a>58.9.1.21.5.1.1. ReduxOptions > train > losses > losses items > anyOf > perceptualloss > alpha > anyOf > item 0 > item 0 items

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |

###### <a name="train_losses_items_anyOf_i20_alpha_anyOf_i1"></a>58.9.1.21.5.2. Property `ReduxOptions > train > losses > losses items > anyOf > perceptualloss > alpha > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

###### <a name="train_losses_items_anyOf_i20_criterion"></a>58.9.1.21.6. Property `ReduxOptions > train > losses > losses items > anyOf > perceptualloss > criterion`

|              |                    |
| ------------ | ------------------ |
| **Type**     | `enum (of string)` |
| **Required** | Yes                |

Must be one of:
* "charbonnier"
* "fd"
* "fd+l1pd"
* "l1"
* "pd+l1"

###### <a name="train_losses_items_anyOf_i20_num_proj_fd"></a>58.9.1.21.7. Property `ReduxOptions > train > losses > losses items > anyOf > perceptualloss > num_proj_fd`

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | Yes       |

###### <a name="train_losses_items_anyOf_i20_phase_weight_fd"></a>58.9.1.21.8. Property `ReduxOptions > train > losses > losses items > anyOf > perceptualloss > phase_weight_fd`

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | Yes      |

###### <a name="train_losses_items_anyOf_i20_stride_fd"></a>58.9.1.21.9. Property `ReduxOptions > train > losses > losses items > anyOf > perceptualloss > stride_fd`

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | Yes       |

##### <a name="train_losses_items_anyOf_i21"></a>58.9.1.22. Property `ReduxOptions > train > losses > losses items > anyOf > topiqloss`

**Title:** topiqloss

|                           |                   |
| ------------------------- | ----------------- |
| **Type**                  | `object`          |
| **Required**              | No                |
| **Additional properties** | Any type allowed  |
| **Defined in**            | #/$defs/topiqloss |

| Property                                                      | Pattern | Type             | Deprecated | Definition | Title/Description |
| ------------------------------------------------------------- | ------- | ---------------- | ---------- | ---------- | ----------------- |
| + [type](#train_losses_items_anyOf_i21_type )                 | No      | enum (of string) | No         | -          | -                 |
| + [loss_weight](#train_losses_items_anyOf_i21_loss_weight )   | No      | number           | No         | -          | -                 |
| + [resize_input](#train_losses_items_anyOf_i21_resize_input ) | No      | boolean          | No         | -          | -                 |

###### <a name="train_losses_items_anyOf_i21_type"></a>58.9.1.22.1. Property `ReduxOptions > train > losses > losses items > anyOf > topiqloss > type`

|              |                    |
| ------------ | ------------------ |
| **Type**     | `enum (of string)` |
| **Required** | Yes                |

Must be one of:
* "topiqloss"

###### <a name="train_losses_items_anyOf_i21_loss_weight"></a>58.9.1.22.2. Property `ReduxOptions > train > losses > losses items > anyOf > topiqloss > loss_weight`

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | Yes      |

###### <a name="train_losses_items_anyOf_i21_resize_input"></a>58.9.1.22.3. Property `ReduxOptions > train > losses > losses items > anyOf > topiqloss > resize_input`

|              |           |
| ------------ | --------- |
| **Type**     | `boolean` |
| **Required** | Yes       |

### <a name="train_pixel_opt"></a>58.10. Property `ReduxOptions > train > pixel_opt`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `combining`      |
| **Required**              | Yes              |
| **Additional properties** | Any type allowed |

| Any of(Option)                      |
| ----------------------------------- |
| [item 0](#train_pixel_opt_anyOf_i0) |
| [item 1](#train_pixel_opt_anyOf_i1) |

#### <a name="train_pixel_opt_anyOf_i0"></a>58.10.1. Property `ReduxOptions > train > pixel_opt > anyOf > item 0`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `object`         |
| **Required**              | No               |
| **Additional properties** | Any type allowed |

#### <a name="train_pixel_opt_anyOf_i1"></a>58.10.2. Property `ReduxOptions > train > pixel_opt > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

### <a name="train_mssim_opt"></a>58.11. Property `ReduxOptions > train > mssim_opt`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `combining`      |
| **Required**              | Yes              |
| **Additional properties** | Any type allowed |

| Any of(Option)                      |
| ----------------------------------- |
| [item 0](#train_mssim_opt_anyOf_i0) |
| [item 1](#train_mssim_opt_anyOf_i1) |

#### <a name="train_mssim_opt_anyOf_i0"></a>58.11.1. Property `ReduxOptions > train > mssim_opt > anyOf > item 0`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `object`         |
| **Required**              | No               |
| **Additional properties** | Any type allowed |

#### <a name="train_mssim_opt_anyOf_i1"></a>58.11.2. Property `ReduxOptions > train > mssim_opt > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

### <a name="train_ms_ssim_l1_opt"></a>58.12. Property `ReduxOptions > train > ms_ssim_l1_opt`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `combining`      |
| **Required**              | Yes              |
| **Additional properties** | Any type allowed |

| Any of(Option)                           |
| ---------------------------------------- |
| [item 0](#train_ms_ssim_l1_opt_anyOf_i0) |
| [item 1](#train_ms_ssim_l1_opt_anyOf_i1) |

#### <a name="train_ms_ssim_l1_opt_anyOf_i0"></a>58.12.1. Property `ReduxOptions > train > ms_ssim_l1_opt > anyOf > item 0`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `object`         |
| **Required**              | No               |
| **Additional properties** | Any type allowed |

#### <a name="train_ms_ssim_l1_opt_anyOf_i1"></a>58.12.2. Property `ReduxOptions > train > ms_ssim_l1_opt > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

### <a name="train_perceptual_opt"></a>58.13. Property `ReduxOptions > train > perceptual_opt`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `combining`      |
| **Required**              | Yes              |
| **Additional properties** | Any type allowed |

| Any of(Option)                           |
| ---------------------------------------- |
| [item 0](#train_perceptual_opt_anyOf_i0) |
| [item 1](#train_perceptual_opt_anyOf_i1) |

#### <a name="train_perceptual_opt_anyOf_i0"></a>58.13.1. Property `ReduxOptions > train > perceptual_opt > anyOf > item 0`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `object`         |
| **Required**              | No               |
| **Additional properties** | Any type allowed |

#### <a name="train_perceptual_opt_anyOf_i1"></a>58.13.2. Property `ReduxOptions > train > perceptual_opt > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

### <a name="train_contextual_opt"></a>58.14. Property `ReduxOptions > train > contextual_opt`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `combining`      |
| **Required**              | Yes              |
| **Additional properties** | Any type allowed |

| Any of(Option)                           |
| ---------------------------------------- |
| [item 0](#train_contextual_opt_anyOf_i0) |
| [item 1](#train_contextual_opt_anyOf_i1) |

#### <a name="train_contextual_opt_anyOf_i0"></a>58.14.1. Property `ReduxOptions > train > contextual_opt > anyOf > item 0`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `object`         |
| **Required**              | No               |
| **Additional properties** | Any type allowed |

#### <a name="train_contextual_opt_anyOf_i1"></a>58.14.2. Property `ReduxOptions > train > contextual_opt > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

### <a name="train_dists_opt"></a>58.15. Property `ReduxOptions > train > dists_opt`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `combining`      |
| **Required**              | Yes              |
| **Additional properties** | Any type allowed |

| Any of(Option)                      |
| ----------------------------------- |
| [item 0](#train_dists_opt_anyOf_i0) |
| [item 1](#train_dists_opt_anyOf_i1) |

#### <a name="train_dists_opt_anyOf_i0"></a>58.15.1. Property `ReduxOptions > train > dists_opt > anyOf > item 0`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `object`         |
| **Required**              | No               |
| **Additional properties** | Any type allowed |

#### <a name="train_dists_opt_anyOf_i1"></a>58.15.2. Property `ReduxOptions > train > dists_opt > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

### <a name="train_hr_inversion_opt"></a>58.16. Property `ReduxOptions > train > hr_inversion_opt`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `combining`      |
| **Required**              | Yes              |
| **Additional properties** | Any type allowed |

| Any of(Option)                             |
| ------------------------------------------ |
| [item 0](#train_hr_inversion_opt_anyOf_i0) |
| [item 1](#train_hr_inversion_opt_anyOf_i1) |

#### <a name="train_hr_inversion_opt_anyOf_i0"></a>58.16.1. Property `ReduxOptions > train > hr_inversion_opt > anyOf > item 0`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `object`         |
| **Required**              | No               |
| **Additional properties** | Any type allowed |

#### <a name="train_hr_inversion_opt_anyOf_i1"></a>58.16.2. Property `ReduxOptions > train > hr_inversion_opt > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

### <a name="train_dinov2_opt"></a>58.17. Property `ReduxOptions > train > dinov2_opt`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `combining`      |
| **Required**              | Yes              |
| **Additional properties** | Any type allowed |

| Any of(Option)                       |
| ------------------------------------ |
| [item 0](#train_dinov2_opt_anyOf_i0) |
| [item 1](#train_dinov2_opt_anyOf_i1) |

#### <a name="train_dinov2_opt_anyOf_i0"></a>58.17.1. Property `ReduxOptions > train > dinov2_opt > anyOf > item 0`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `object`         |
| **Required**              | No               |
| **Additional properties** | Any type allowed |

#### <a name="train_dinov2_opt_anyOf_i1"></a>58.17.2. Property `ReduxOptions > train > dinov2_opt > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

### <a name="train_topiq_opt"></a>58.18. Property `ReduxOptions > train > topiq_opt`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `combining`      |
| **Required**              | Yes              |
| **Additional properties** | Any type allowed |

| Any of(Option)                      |
| ----------------------------------- |
| [item 0](#train_topiq_opt_anyOf_i0) |
| [item 1](#train_topiq_opt_anyOf_i1) |

#### <a name="train_topiq_opt_anyOf_i0"></a>58.18.1. Property `ReduxOptions > train > topiq_opt > anyOf > item 0`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `object`         |
| **Required**              | No               |
| **Additional properties** | Any type allowed |

#### <a name="train_topiq_opt_anyOf_i1"></a>58.18.2. Property `ReduxOptions > train > topiq_opt > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

### <a name="train_pd_opt"></a>58.19. Property `ReduxOptions > train > pd_opt`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `combining`      |
| **Required**              | Yes              |
| **Additional properties** | Any type allowed |

| Any of(Option)                   |
| -------------------------------- |
| [item 0](#train_pd_opt_anyOf_i0) |
| [item 1](#train_pd_opt_anyOf_i1) |

#### <a name="train_pd_opt_anyOf_i0"></a>58.19.1. Property `ReduxOptions > train > pd_opt > anyOf > item 0`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `object`         |
| **Required**              | No               |
| **Additional properties** | Any type allowed |

#### <a name="train_pd_opt_anyOf_i1"></a>58.19.2. Property `ReduxOptions > train > pd_opt > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

### <a name="train_fd_opt"></a>58.20. Property `ReduxOptions > train > fd_opt`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `combining`      |
| **Required**              | Yes              |
| **Additional properties** | Any type allowed |

| Any of(Option)                   |
| -------------------------------- |
| [item 0](#train_fd_opt_anyOf_i0) |
| [item 1](#train_fd_opt_anyOf_i1) |

#### <a name="train_fd_opt_anyOf_i0"></a>58.20.1. Property `ReduxOptions > train > fd_opt > anyOf > item 0`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `object`         |
| **Required**              | No               |
| **Additional properties** | Any type allowed |

#### <a name="train_fd_opt_anyOf_i1"></a>58.20.2. Property `ReduxOptions > train > fd_opt > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

### <a name="train_ldl_opt"></a>58.21. Property `ReduxOptions > train > ldl_opt`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `combining`      |
| **Required**              | Yes              |
| **Additional properties** | Any type allowed |

| Any of(Option)                    |
| --------------------------------- |
| [item 0](#train_ldl_opt_anyOf_i0) |
| [item 1](#train_ldl_opt_anyOf_i1) |

#### <a name="train_ldl_opt_anyOf_i0"></a>58.21.1. Property `ReduxOptions > train > ldl_opt > anyOf > item 0`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `object`         |
| **Required**              | No               |
| **Additional properties** | Any type allowed |

#### <a name="train_ldl_opt_anyOf_i1"></a>58.21.2. Property `ReduxOptions > train > ldl_opt > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

### <a name="train_hsluv_opt"></a>58.22. Property `ReduxOptions > train > hsluv_opt`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `combining`      |
| **Required**              | Yes              |
| **Additional properties** | Any type allowed |

| Any of(Option)                      |
| ----------------------------------- |
| [item 0](#train_hsluv_opt_anyOf_i0) |
| [item 1](#train_hsluv_opt_anyOf_i1) |

#### <a name="train_hsluv_opt_anyOf_i0"></a>58.22.1. Property `ReduxOptions > train > hsluv_opt > anyOf > item 0`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `object`         |
| **Required**              | No               |
| **Additional properties** | Any type allowed |

#### <a name="train_hsluv_opt_anyOf_i1"></a>58.22.2. Property `ReduxOptions > train > hsluv_opt > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

### <a name="train_gan_opt"></a>58.23. Property `ReduxOptions > train > gan_opt`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `combining`      |
| **Required**              | Yes              |
| **Additional properties** | Any type allowed |

| Any of(Option)                    |
| --------------------------------- |
| [item 0](#train_gan_opt_anyOf_i0) |
| [item 1](#train_gan_opt_anyOf_i1) |

#### <a name="train_gan_opt_anyOf_i0"></a>58.23.1. Property `ReduxOptions > train > gan_opt > anyOf > item 0`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `object`         |
| **Required**              | No               |
| **Additional properties** | Any type allowed |

#### <a name="train_gan_opt_anyOf_i1"></a>58.23.2. Property `ReduxOptions > train > gan_opt > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

### <a name="train_color_opt"></a>58.24. Property `ReduxOptions > train > color_opt`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `combining`      |
| **Required**              | Yes              |
| **Additional properties** | Any type allowed |

| Any of(Option)                      |
| ----------------------------------- |
| [item 0](#train_color_opt_anyOf_i0) |
| [item 1](#train_color_opt_anyOf_i1) |

#### <a name="train_color_opt_anyOf_i0"></a>58.24.1. Property `ReduxOptions > train > color_opt > anyOf > item 0`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `object`         |
| **Required**              | No               |
| **Additional properties** | Any type allowed |

#### <a name="train_color_opt_anyOf_i1"></a>58.24.2. Property `ReduxOptions > train > color_opt > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

### <a name="train_luma_opt"></a>58.25. Property `ReduxOptions > train > luma_opt`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `combining`      |
| **Required**              | Yes              |
| **Additional properties** | Any type allowed |

| Any of(Option)                     |
| ---------------------------------- |
| [item 0](#train_luma_opt_anyOf_i0) |
| [item 1](#train_luma_opt_anyOf_i1) |

#### <a name="train_luma_opt_anyOf_i0"></a>58.25.1. Property `ReduxOptions > train > luma_opt > anyOf > item 0`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `object`         |
| **Required**              | No               |
| **Additional properties** | Any type allowed |

#### <a name="train_luma_opt_anyOf_i1"></a>58.25.2. Property `ReduxOptions > train > luma_opt > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

### <a name="train_avg_opt"></a>58.26. Property `ReduxOptions > train > avg_opt`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `combining`      |
| **Required**              | Yes              |
| **Additional properties** | Any type allowed |

| Any of(Option)                    |
| --------------------------------- |
| [item 0](#train_avg_opt_anyOf_i0) |
| [item 1](#train_avg_opt_anyOf_i1) |

#### <a name="train_avg_opt_anyOf_i0"></a>58.26.1. Property `ReduxOptions > train > avg_opt > anyOf > item 0`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `object`         |
| **Required**              | No               |
| **Additional properties** | Any type allowed |

#### <a name="train_avg_opt_anyOf_i1"></a>58.26.2. Property `ReduxOptions > train > avg_opt > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

### <a name="train_bicubic_opt"></a>58.27. Property `ReduxOptions > train > bicubic_opt`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `combining`      |
| **Required**              | Yes              |
| **Additional properties** | Any type allowed |

| Any of(Option)                        |
| ------------------------------------- |
| [item 0](#train_bicubic_opt_anyOf_i0) |
| [item 1](#train_bicubic_opt_anyOf_i1) |

#### <a name="train_bicubic_opt_anyOf_i0"></a>58.27.1. Property `ReduxOptions > train > bicubic_opt > anyOf > item 0`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `object`         |
| **Required**              | No               |
| **Additional properties** | Any type allowed |

#### <a name="train_bicubic_opt_anyOf_i1"></a>58.27.2. Property `ReduxOptions > train > bicubic_opt > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

### <a name="train_use_moa"></a>58.28. Property `ReduxOptions > train > use_moa`

|              |           |
| ------------ | --------- |
| **Type**     | `boolean` |
| **Required** | Yes       |

**Description:** Whether to enable mixture of augmentations, which augments the dataset on the fly to create more variety and help the model generalize.

### <a name="train_moa_augs"></a>58.29. Property `ReduxOptions > train > moa_augs`

|              |                   |
| ------------ | ----------------- |
| **Type**     | `array of string` |
| **Required** | Yes               |

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

#### <a name="train_moa_augs_items"></a>58.29.1. ReduxOptions > train > moa_augs > moa_augs items

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | No       |

### <a name="train_moa_probs"></a>58.30. Property `ReduxOptions > train > moa_probs`

|              |                   |
| ------------ | ----------------- |
| **Type**     | `array of number` |
| **Required** | Yes               |

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

#### <a name="train_moa_probs_items"></a>58.30.1. ReduxOptions > train > moa_probs > moa_probs items

|              |          |
| ------------ | -------- |
| **Type**     | `number` |
| **Required** | No       |

### <a name="train_moa_debug"></a>58.31. Property `ReduxOptions > train > moa_debug`

|              |           |
| ------------ | --------- |
| **Type**     | `boolean` |
| **Required** | Yes       |

**Description:** Save images before and after augment to debug/moa folder inside of the root training directory.

### <a name="train_moa_debug_limit"></a>58.32. Property `ReduxOptions > train > moa_debug_limit`

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | Yes       |

**Description:** The max number of iterations to save augmentation images for.

## <a name="val"></a>59. Property `ReduxOptions > val`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `combining`      |
| **Required**              | Yes              |
| **Additional properties** | Any type allowed |

| Any of(Option)              |
| --------------------------- |
| [item 0](#val_anyOf_i0)     |
| [ValOptions](#val_anyOf_i1) |

### <a name="val_anyOf_i0"></a>59.1. Property `ReduxOptions > val > anyOf > item 0`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

### <a name="val_anyOf_i1"></a>59.2. Property `ReduxOptions > val > anyOf > ValOptions`

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

#### <a name="val_anyOf_i1_val_enabled"></a>59.2.1. Property `ReduxOptions > val > anyOf > ValOptions > val_enabled`

|              |           |
| ------------ | --------- |
| **Type**     | `boolean` |
| **Required** | Yes       |

**Description:** Whether to enable validations. If disabled, all validation settings below are ignored.

#### <a name="val_anyOf_i1_save_img"></a>59.2.2. Property `ReduxOptions > val > anyOf > ValOptions > save_img`

|              |           |
| ------------ | --------- |
| **Type**     | `boolean` |
| **Required** | Yes       |

**Description:** Whether to save the validation images during validation, in the experiments/<name>/visualization folder.

#### <a name="val_anyOf_i1_val_freq"></a>59.2.3. Property `ReduxOptions > val > anyOf > ValOptions > val_freq`

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

##### <a name="val_anyOf_i1_val_freq_anyOf_i0"></a>59.2.3.1. Property `ReduxOptions > val > anyOf > ValOptions > val_freq > anyOf > item 0`

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | No        |

##### <a name="val_anyOf_i1_val_freq_anyOf_i1"></a>59.2.3.2. Property `ReduxOptions > val > anyOf > ValOptions > val_freq > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

#### <a name="val_anyOf_i1_suffix"></a>59.2.4. Property `ReduxOptions > val > anyOf > ValOptions > suffix`

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

##### <a name="val_anyOf_i1_suffix_anyOf_i0"></a>59.2.4.1. Property `ReduxOptions > val > anyOf > ValOptions > suffix > anyOf > item 0`

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | No       |

##### <a name="val_anyOf_i1_suffix_anyOf_i1"></a>59.2.4.2. Property `ReduxOptions > val > anyOf > ValOptions > suffix > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

#### <a name="val_anyOf_i1_metrics_enabled"></a>59.2.5. Property `ReduxOptions > val > anyOf > ValOptions > metrics_enabled`

|              |           |
| ------------ | --------- |
| **Type**     | `boolean` |
| **Required** | No        |
| **Default**  | `false`   |

**Description:** Whether to run metrics calculations during validation.

#### <a name="val_anyOf_i1_metrics"></a>59.2.6. Property `ReduxOptions > val > anyOf > ValOptions > metrics`

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

##### <a name="val_anyOf_i1_metrics_anyOf_i0"></a>59.2.6.1. Property `ReduxOptions > val > anyOf > ValOptions > metrics > anyOf > item 0`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `object`         |
| **Required**              | No               |
| **Additional properties** | Any type allowed |

##### <a name="val_anyOf_i1_metrics_anyOf_i1"></a>59.2.6.2. Property `ReduxOptions > val > anyOf > ValOptions > metrics > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

#### <a name="val_anyOf_i1_pbar"></a>59.2.7. Property `ReduxOptions > val > anyOf > ValOptions > pbar`

|              |           |
| ------------ | --------- |
| **Type**     | `boolean` |
| **Required** | No        |
| **Default**  | `true`    |

## <a name="logger"></a>60. Property `ReduxOptions > logger`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `combining`      |
| **Required**              | Yes              |
| **Additional properties** | Any type allowed |

| Any of(Option)                 |
| ------------------------------ |
| [item 0](#logger_anyOf_i0)     |
| [LogOptions](#logger_anyOf_i1) |

### <a name="logger_anyOf_i0"></a>60.1. Property `ReduxOptions > logger > anyOf > item 0`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

### <a name="logger_anyOf_i1"></a>60.2. Property `ReduxOptions > logger > anyOf > LogOptions`

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

#### <a name="logger_anyOf_i1_print_freq"></a>60.2.1. Property `ReduxOptions > logger > anyOf > LogOptions > print_freq`

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | Yes       |

**Description:** How often to print logs to the console, in iterations.

#### <a name="logger_anyOf_i1_save_checkpoint_freq"></a>60.2.2. Property `ReduxOptions > logger > anyOf > LogOptions > save_checkpoint_freq`

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | Yes       |

**Description:** How often to save model checkpoints and training states, in iterations.

#### <a name="logger_anyOf_i1_use_tb_logger"></a>60.2.3. Property `ReduxOptions > logger > anyOf > LogOptions > use_tb_logger`

|              |           |
| ------------ | --------- |
| **Type**     | `boolean` |
| **Required** | Yes       |

**Description:** Whether or not to enable TensorBoard logging.

#### <a name="logger_anyOf_i1_save_checkpoint_format"></a>60.2.4. Property `ReduxOptions > logger > anyOf > LogOptions > save_checkpoint_format`

|              |                    |
| ------------ | ------------------ |
| **Type**     | `enum (of string)` |
| **Required** | No                 |
| **Default**  | `"safetensors"`    |

**Description:** Format to save model checkpoints.

Must be one of:
* "pth"
* "safetensors"

#### <a name="logger_anyOf_i1_wandb"></a>60.2.5. Property `ReduxOptions > logger > anyOf > LogOptions > wandb`

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

##### <a name="logger_anyOf_i1_wandb_anyOf_i0"></a>60.2.5.1. Property `ReduxOptions > logger > anyOf > LogOptions > wandb > anyOf > item 0`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

##### <a name="logger_anyOf_i1_wandb_anyOf_i1"></a>60.2.5.2. Property `ReduxOptions > logger > anyOf > LogOptions > wandb > anyOf > WandbOptions`

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

###### <a name="logger_anyOf_i1_wandb_anyOf_i1_resume_id"></a>60.2.5.2.1. Property `ReduxOptions > logger > anyOf > LogOptions > wandb > anyOf > WandbOptions > resume_id`

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

###### <a name="logger_anyOf_i1_wandb_anyOf_i1_resume_id_anyOf_i0"></a>60.2.5.2.1.1. Property `ReduxOptions > logger > anyOf > LogOptions > wandb > anyOf > WandbOptions > resume_id > anyOf > item 0`

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | No       |

###### <a name="logger_anyOf_i1_wandb_anyOf_i1_resume_id_anyOf_i1"></a>60.2.5.2.1.2. Property `ReduxOptions > logger > anyOf > LogOptions > wandb > anyOf > WandbOptions > resume_id > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

###### <a name="logger_anyOf_i1_wandb_anyOf_i1_project"></a>60.2.5.2.2. Property `ReduxOptions > logger > anyOf > LogOptions > wandb > anyOf > WandbOptions > project`

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

###### <a name="logger_anyOf_i1_wandb_anyOf_i1_project_anyOf_i0"></a>60.2.5.2.2.1. Property `ReduxOptions > logger > anyOf > LogOptions > wandb > anyOf > WandbOptions > project > anyOf > item 0`

|              |          |
| ------------ | -------- |
| **Type**     | `string` |
| **Required** | No       |

###### <a name="logger_anyOf_i1_wandb_anyOf_i1_project_anyOf_i1"></a>60.2.5.2.2.2. Property `ReduxOptions > logger > anyOf > LogOptions > wandb > anyOf > WandbOptions > project > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

## <a name="dist_params"></a>61. Property `ReduxOptions > dist_params`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `combining`      |
| **Required**              | Yes              |
| **Additional properties** | Any type allowed |

| Any of(Option)                  |
| ------------------------------- |
| [item 0](#dist_params_anyOf_i0) |
| [item 1](#dist_params_anyOf_i1) |

### <a name="dist_params_anyOf_i0"></a>61.1. Property `ReduxOptions > dist_params > anyOf > item 0`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `object`         |
| **Required**              | No               |
| **Additional properties** | Any type allowed |

### <a name="dist_params_anyOf_i1"></a>61.2. Property `ReduxOptions > dist_params > anyOf > item 1`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

## <a name="onnx"></a>62. Property `ReduxOptions > onnx`

|                           |                  |
| ------------------------- | ---------------- |
| **Type**                  | `combining`      |
| **Required**              | Yes              |
| **Additional properties** | Any type allowed |

| Any of(Option)                |
| ----------------------------- |
| [item 0](#onnx_anyOf_i0)      |
| [OnnxOptions](#onnx_anyOf_i1) |

### <a name="onnx_anyOf_i0"></a>62.1. Property `ReduxOptions > onnx > anyOf > item 0`

|              |        |
| ------------ | ------ |
| **Type**     | `null` |
| **Required** | No     |

### <a name="onnx_anyOf_i1"></a>62.2. Property `ReduxOptions > onnx > anyOf > OnnxOptions`

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

#### <a name="onnx_anyOf_i1_dynamo"></a>62.2.1. Property `ReduxOptions > onnx > anyOf > OnnxOptions > dynamo`

|              |           |
| ------------ | --------- |
| **Type**     | `boolean` |
| **Required** | No        |
| **Default**  | `false`   |

#### <a name="onnx_anyOf_i1_opset"></a>62.2.2. Property `ReduxOptions > onnx > anyOf > OnnxOptions > opset`

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | No        |
| **Default**  | `20`      |

#### <a name="onnx_anyOf_i1_use_static_shapes"></a>62.2.3. Property `ReduxOptions > onnx > anyOf > OnnxOptions > use_static_shapes`

|              |           |
| ------------ | --------- |
| **Type**     | `boolean` |
| **Required** | No        |
| **Default**  | `false`   |

#### <a name="onnx_anyOf_i1_shape"></a>62.2.4. Property `ReduxOptions > onnx > anyOf > OnnxOptions > shape`

|              |               |
| ------------ | ------------- |
| **Type**     | `string`      |
| **Required** | No            |
| **Default**  | `"3x256x256"` |

#### <a name="onnx_anyOf_i1_verify"></a>62.2.5. Property `ReduxOptions > onnx > anyOf > OnnxOptions > verify`

|              |           |
| ------------ | --------- |
| **Type**     | `boolean` |
| **Required** | No        |
| **Default**  | `true`    |

#### <a name="onnx_anyOf_i1_fp16"></a>62.2.6. Property `ReduxOptions > onnx > anyOf > OnnxOptions > fp16`

|              |           |
| ------------ | --------- |
| **Type**     | `boolean` |
| **Required** | No        |
| **Default**  | `false`   |

#### <a name="onnx_anyOf_i1_optimize"></a>62.2.7. Property `ReduxOptions > onnx > anyOf > OnnxOptions > optimize`

|              |           |
| ------------ | --------- |
| **Type**     | `boolean` |
| **Required** | No        |
| **Default**  | `true`    |

## <a name="find_unused_parameters"></a>63. Property `ReduxOptions > find_unused_parameters`

|              |           |
| ------------ | --------- |
| **Type**     | `boolean` |
| **Required** | Yes       |

----------------------------------------------------------------------------------------------------------------------------
Generated using [json-schema-for-humans](https://github.com/coveooss/json-schema-for-humans) on 2024-12-19 at 00:45:45 -0500
