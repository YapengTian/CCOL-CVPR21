import os
import glob
import argparse
import random
import fnmatch
import librosa


def find_recursive(root_dir, ext='.wav'):
    files = []
    for root, dirnames, filenames in os.walk(root_dir):
        for filename in fnmatch.filter(filenames, '*' + ext):
            files.append(os.path.join(root, filename))
    return files


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_audio', default='/mnt/disk0/datasets/ytian21/AVSS_data/VEGAS_dataset/data/audio',
                        help="root for extracted audio files")
    parser.add_argument('--root_frame', default='/mnt/disk0/datasets/ytian21/AVSS_data/VEGAS_dataset/data/frames',
                        help="root for extracted video frames")
    parser.add_argument('--fps', default=8, type=int,
                        help="fps of video frames")
    parser.add_argument('--path_output', default='/mnt/disk0/datasets/ytian21/AVSS_data/VEGAS_dataset/data',
                        help="path to output index files")
    parser.add_argument('--trainset_ratio', default=0.8, type=float,
                        help="80% for training, 20% for validation")
    args = parser.parse_args()

    # find all audio/frames pairs
    infos = []
    audio_files = find_recursive(args.root_audio, ext='.wav')

    for audio_path in audio_files:

        frame_path = audio_path.replace(args.root_audio, args.root_frame) \
                               .replace('.wav', '.mp4')

        try:
            audio_raw, rate = librosa.load(audio_path, sr=None, mono=True)
            if type(rate) is not int:
                print(type(rate))
                os.remove(audio_path)
        except Exception as e:
            print('Failed loading frame/audio: {}'.format(e))
            os.remove(audio_path)

    print('Done!')
