# Processing Logs Summary

## Cartastrophe.mp4

| step                             | wall_time_sec | cpu_time_sec | ram_used_MB | io_read_MB | io_write_MB |
| :------------------------------- | ------------: | -----------: | ----------: | ---------: | ----------: |
| PySceneDetect*                   |         1.242 |        5.934 |       0.235 |     95.733 |           0 |
| AST sound descriptions*          |        10.253 |       80.754 |      -6.803 |     95.366 |           0 |
| ASR speech transcription*        |        40.745 |      314.884 |     177.584 |    311.819 |           0 |
| Masked clips saving              |         0.095 |        0.001 |           0 |      0.003 |           0 |
| Frame sampling                   |         0.142 |        1.414 |       1.372 |     18.501 |       0.041 |
| BLIP caption                     |         3.924 |        31.21 |       2.395 |          0 |           0 |
| YOLO detection                   |         0.258 |        2.062 |           1 |      0.509 |       0.046 |
| BLIP + YOLO + AST + ASR into LLM |          5.93 |        0.056 |           0 |      0.019 |           0 |

**Footnote:**
`total_process_sec` is **3.45× longer** than `video_length` of 255.76666666666668s.
**43 scenes** were detected in `Videos\Cartastrophe.mp4`
**`get_scene_list`, `ast_timings`, and `asr_timings` are measured per minute of video, whereas the remaining processes are measured per scenes.

## Watch Malala Yousafzai's Nobel Peace Prize acceptance speech.mp4

| step                             | wall_time_sec | cpu_time_sec | ram_used_MB | io_read_MB | io_write_MB |
| :------------------------------- | ------------: | -----------: | ----------: | ---------: | ----------: |
| PySceneDetect*                   |         1.296 |        4.964 |       1.755 |     84.872 |           0 |
| AST sound descriptions*          |         5.316 |       41.436 |       1.536 |     84.426 |           0 |
| ASR speech transcription*        |        50.699 |      394.848 |      20.404 |    286.864 |           0 |
| Masked clips saving              |          0.08 |        0.004 |           0 |      0.003 |           0 |
| Frame sampling                   |         0.126 |        0.858 |       0.273 |     11.495 |       0.057 |
| BLIP caption                     |         3.828 |       30.477 |       6.545 |          0 |           0 |
| YOLO detection                   |         0.258 |        2.043 |       0.273 |      0.994 |       0.067 |
| BLIP + YOLO + AST + ASR into LLM |         4.784 |        0.067 |           0 |      0.037 |           0 |

**Footnote:**
`total_process_sec` is **2.09× longer** than `video_length` of 273.4732s.
**22 scenes** were detected in `Videos\Watch Malala Yousafzai's Nobel Peace Prize acceptance speech.mp4`
**`get_scene_list`, `ast_timings`, and `asr_timings` are measured per minute of video, whereas the remaining processes are measured per scenes.

## Spain Vlog.mp4

| step                             | wall_time_sec | cpu_time_sec | ram_used_MB | io_read_MB | io_write_MB |
| :------------------------------- | ------------: | -----------: | ----------: | ---------: | ----------: |
| PySceneDetect*                   |        10.517 |       33.502 |       0.244 |    137.374 |           0 |
| AST sound descriptions*          |        10.631 |       83.586 |      -1.465 |    137.323 |           0 |
| ASR speech transcription*        |        52.042 |      404.278 |      14.893 |    362.596 |           0 |
| Masked clips saving              |         0.114 |        0.003 |           0 |      0.003 |           0 |
| Frame sampling                   |         1.187 |        2.511 |       1.488 |     11.475 |       0.063 |
| BLIP caption                     |         3.797 |       30.223 |      -1.116 |          0 |           0 |
| YOLO detection                   |         0.254 |        2.024 |       0.884 |      0.509 |        0.07 |
| BLIP + YOLO + AST + ASR into LLM |          6.23 |        0.053 |       0.023 |      0.019 |           0 |

**Footnote:**
`total_process_sec` is **4.12× longer** than `video_length` of 245.753s.
**43 scenes** were detected in `Videos\Spain Vlog.mp4`
**`get_scene_list`, `ast_timings`, and `asr_timings` are measured per minute of video, whereas the remaining processes are measured per scenes.

## SpongeBob SquarePants - Writing Essay - Some of These - Meme Source.mp4

| step                             | wall_time_sec | cpu_time_sec | ram_used_MB | io_read_MB | io_write_MB |
| :------------------------------- | ------------: | -----------: | ----------: | ---------: | ----------: |
| PySceneDetect*                   |         1.758 |        6.136 |      28.937 |     67.958 |           0 |
| AST sound descriptions*          |        15.113 |       80.997 |     762.686 |     78.211 |           0 |
| ASR speech transcription*        |        80.789 |      606.053 |     514.658 |    1975.56 |           0 |
| Masked clips saving              |         0.036 |            0 |           0 |      0.003 |           0 |
| Frame sampling                   |         0.151 |        0.803 |         0.4 |      8.178 |        0.06 |
| BLIP caption                     |      *7.648* |        7.572 |         205 |      0.007 |           0 |
| YOLO detection                   |         0.299 |        2.225 |         7.6 |      4.412 |       0.064 |
| BLIP + YOLO + AST + ASR into LLM |         5.897 |        0.278 |         0.2 |      0.243 |           0 |

**Footnote:**
`total_process_sec` is **4.94× longer** than `video_length` of 29.029s.
**5 scenes** were detected in `Videos\SpongeBob SquarePants - Writing Essay - Some of These - Meme Source.mp4`
**`get_scene_list`, `ast_timings`, and `asr_timings` are measured per minute of video, whereas the remaining processes are measured per scenes.
