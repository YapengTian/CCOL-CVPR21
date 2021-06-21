# CCoL
Pytorch implementation for CVPR21 "Cyclic Co-Learning of Sounding Object Visual Grounding and Sound Separation".

*This repository is under construction.

![image](doc/ccol_fig.png)

## Environment
The code is developed under the following configurations.
- Hardware: 1-4 GPUs (change ```[--num_gpus NUM_GPUS]``` accordingly)
- Software: Ubuntu 16.04.3 LTS, ***CUDA>=8.0, Python>=3.5, PyTorch>=1.2***

## Evaluation
1. Evaluate the trained separation model.
```bash
./scripts/eval_ccol.sh
```
2. Evaluate the sounding object-aware separation model on videos with silent objects.
```bash
./scripts/eval_ccol_silent.sh
```

## Training
1. Prepare video dataset.

    a. Download MUSIC dataset from: https://github.com/roudimit/MUSIC_dataset
    
    b. Download videos.

2. Preprocess videos. You can do it in your own way as long as the index files are similar.

    a. Extract frames at 8fps and waveforms at 11025Hz from videos. We have following directory structure:
    ```
    data
    ├── audio
    |   ├── acoustic_guitar
    │   |   ├── M3dekVSwNjY.mp3
    │   |   ├── ...
    │   ├── trumpet
    │   |   ├── STKXyBGSGyE.mp3
    │   |   ├── ...
    │   ├── ...
    |
    └── frames
    |   ├── acoustic_guitar
    │   |   ├── M3dekVSwNjY.mp4
    │   |   |   ├── 000001.jpg
    │   |   |   ├── ...
    │   |   ├── ...
    │   ├── trumpet
    │   |   ├── STKXyBGSGyE.mp4
    │   |   |   ├── 000001.jpg
    │   |   |   ├── ...
    │   |   ├── ...
    │   ├── ...
    ```

    b. Make training/validation index files by running:
    ```
    python scripts/create_index_files.py
    ```
    It will create index files ```train.csv```/```val.csv``` with the following format:
    ```
    ./data/audio/acoustic_guitar/M3dekVSwNjY.mp3,./data/frames/acoustic_guitar/M3dekVSwNjY.mp4,1580
    ./data/audio/trumpet/STKXyBGSGyE.mp3,./data/frames/trumpet/STKXyBGSGyE.mp4,493
    ```
    For each row, it stores the information: ```AUDIO_PATH,FRAMES_PATH,NUMBER_FRAMES```

3. Train the grounding-only model for warming up
```bash
./scripts/train_grd.sh
```

4. Train the co-learning model 
```bash
./scripts/train_col.sh
```

3. Train the cyclic co-learning model
```bash
./scripts/train_ccol.sh
```

4. During training, visualizations are saved in HTML format under ```ckpt/MODEL_ID/visualization/```.



## Reference
If you use the code from the project, please cite:
```bibtex
    @InProceedings{tian2021cyclic,
        title={Cyclic Co-Learning of Sounding Object Visual Grounding and Sound Separation},
        author={Tian, Yapeng and Hu, Di and Xu, Chenliang},
        booktitle = {CVPR},
        year = {2021}
    }
```

### Acknowledgements
We borrowed a lot of code from [SoP](https://github.com/hangzhaomit/Sound-of-Pixels). We thank the authors for sharing their codes. If you use our codes, please also cite their nice work.

