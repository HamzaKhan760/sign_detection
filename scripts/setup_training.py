"""
STEP 6: Download pre-trained model and generate pipeline config.

Usage: python scripts/setup_training.py

Downloads SSD MobileNet V2 FPNLite 320x320 and creates pipeline.config
"""

import os
import tarfile
import urllib.request

# ─── CONFIG ──────────────────────────────────────────────────────────────────

MODEL_NAME = "ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8"
MODEL_URL = f"http://download.tensorflow.org/models/object_detection/tf2/20200711/{MODEL_NAME}.tar.gz"
MODEL_DIR = "pretrained_model"
TRAINING_DIR = "training"

# ─── DOWNLOAD ────────────────────────────────────────────────────────────────

os.makedirs(MODEL_DIR, exist_ok=True)
tar_path = os.path.join(MODEL_DIR, f"{MODEL_NAME}.tar.gz")

if not os.path.exists(os.path.join(MODEL_DIR, MODEL_NAME)):
    print(f"Downloading {MODEL_NAME}...")
    urllib.request.urlretrieve(MODEL_URL, tar_path)
    print("Extracting...")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(MODEL_DIR)
    os.remove(tar_path)
    print("Done!")
else:
    print("Model already downloaded.")

checkpoint_dir = os.path.join(MODEL_DIR, MODEL_NAME, "checkpoint")
print(f"Checkpoint at: {checkpoint_dir}")

# ─── GENERATE PIPELINE CONFIG ────────────────────────────────────────────────

# Get absolute paths (TF OD API prefers these)
base_dir = os.getcwd()

pipeline_config = f"""model {{
  ssd {{
    num_classes: 4
    image_resizer {{
      fixed_shape_resizer {{
        height: 320
        width: 320
      }}
    }}
    feature_extractor {{
      type: "ssd_mobilenet_v2_fpn_keras"
      depth_multiplier: 1.0
      min_depth: 16
      conv_hyperparams {{
        regularizer {{
          l2_regularizer {{
            weight: 3.9999998989515007e-05
          }}
        }}
        initializer {{
          random_normal_initializer {{
            mean: 0.0
            stddev: 0.01
          }}
        }}
        activation: RELU_6
        batch_norm {{
          decay: 0.997
          scale: true
          epsilon: 0.001
        }}
      }}
      use_depthwise: true
      override_base_feature_extractor_hyperparams: true
      fpn {{
        min_level: 3
        max_level: 7
        additional_layer_depth: 128
      }}
    }}
    box_coder {{
      faster_rcnn_box_coder {{
        y_scale: 10.0
        x_scale: 10.0
        height_scale: 5.0
        width_scale: 5.0
      }}
    }}
    matcher {{
      argmax_matcher {{
        matched_threshold: 0.5
        unmatched_threshold: 0.5
        ignore_thresholds: false
        negatives_lower_than_unmatched: true
        force_match_for_each_row: true
        use_matmul_gather: true
      }}
    }}
    similarity_calculator {{
      iou_similarity {{
      }}
    }}
    box_predictor {{
      weight_shared_convolutional_box_predictor {{
        conv_hyperparams {{
          regularizer {{
            l2_regularizer {{
              weight: 3.9999998989515007e-05
            }}
          }}
          initializer {{
            random_normal_initializer {{
              mean: 0.0
              stddev: 0.01
            }}
          }}
          activation: RELU_6
          batch_norm {{
            decay: 0.997
            scale: true
            epsilon: 0.001
          }}
        }}
        depth: 128
        num_layers_before_predictor: 4
        kernel_size: 3
        class_prediction_bias_init: -4.599999904632568
        share_prediction_tower: true
        use_depthwise: true
      }}
    }}
    anchor_generator {{
      multiscale_anchor_generator {{
        min_level: 3
        max_level: 7
        anchor_scale: 4.0
        aspect_ratios: 1.0
        aspect_ratios: 2.0
        aspect_ratios: 0.5
        scales_per_octave: 2
      }}
    }}
    post_processing {{
      batch_non_max_suppression {{
        score_threshold: 0.30000001192092896
        iou_threshold: 0.6000000238418579
        max_detections_per_class: 10
        max_total_detections: 10
        use_static_shapes: false
      }}
      score_converter: SIGMOID
    }}
    normalize_loss_by_num_matches: true
    loss {{
      localization_loss {{
        weighted_smooth_l1 {{
        }}
      }}
      classification_loss {{
        weighted_sigmoid_focal {{
          gamma: 2.0
          alpha: 0.25
        }}
      }}
      classification_weight: 1.0
      localization_weight: 1.0
    }}
    encode_background_as_zeros: true
    normalize_loc_loss_by_codesize: true
    inplace_batchnorm_update: true
    freeze_batchnorm: false
  }}
}}

train_config {{
  batch_size: 8
  data_augmentation_options {{
    random_horizontal_flip {{
    }}
  }}
  data_augmentation_options {{
    random_crop_image {{
      min_object_covered: 0.0
      min_aspect_ratio: 0.75
      max_aspect_ratio: 3.0
      min_area: 0.75
      max_area: 1.0
      overlap_thresh: 0.0
    }}
  }}
  data_augmentation_options {{
    random_adjust_brightness {{
    }}
  }}
  data_augmentation_options {{
    random_adjust_contrast {{
    }}
  }}
  sync_replicas: false
  optimizer {{
    momentum_optimizer {{
      learning_rate {{
        cosine_decay_learning_rate {{
          learning_rate_base: 0.04
          total_steps: 15000
          warmup_learning_rate: 0.005
          warmup_steps: 500
        }}
      }}
      momentum_optimizer_value: 0.9
    }}
    use_moving_average: false
  }}
  fine_tune_checkpoint: "{MODEL_DIR}/{MODEL_NAME}/checkpoint/ckpt-0"
  num_steps: 15000
  startup_delay_steps: 0.0
  replicas_to_aggregate: 8
  max_number_of_boxes: 10
  unpad_groundtruth_tensors: false
  fine_tune_checkpoint_type: "detection"
  fine_tune_checkpoint_version: V2
}}

train_input_reader {{
  label_map_path: "training/label_map.pbtxt"
  tf_record_input_reader {{
    input_path: "training/train/train.tfrecord"
  }}
}}

eval_config {{
  metrics_set: "coco_detection_metrics"
  use_moving_averages: false
  batch_size: 1
}}

eval_input_reader {{
  label_map_path: "training/label_map.pbtxt"
  shuffle: false
  num_epochs: 1
  tf_record_input_reader {{
    input_path: "training/val/val.tfrecord"
  }}
}}
"""

os.makedirs(TRAINING_DIR, exist_ok=True)
config_path = os.path.join(TRAINING_DIR, "pipeline.config")

with open(config_path, "w") as f:
    f.write(pipeline_config)

print(f"Pipeline config saved to: {config_path}")
print(f"\nReady to train! Next run: python scripts/train.py")
