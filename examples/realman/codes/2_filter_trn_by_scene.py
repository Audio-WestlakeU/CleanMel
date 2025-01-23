'''
Author: FnoY fangying@westlake.edu.cn
LastEditors: FnoY0723 fangying@westlake.edu.cn
LastEditTime: 2024-10-10 20:03:58
FilePath: /InASR/examples/realman/codes/2_filter_trn_by_scene.py
'''
import os
import sys
sys.path.append('/data/home/fangying/InASR')

# Get a reference transcript for each scene in the test set
def filter_references(long_path, short_path, output_path):
    with open(short_path, 'r', encoding='utf8') as f:
        short_lines = f.readlines()
    
    # Extract the ID in parentheses at the end of each line of the short file 
    # and convert it to a collection for easy lookup operations
    short_ids = set([line.split(' (')[-1].strip().strip(')') for line in short_lines])
    print(f'{os.path.basename(short_path)}: \t{len(short_ids)}')

    with open(long_path, 'r', encoding='utf8') as f:
        long_lines = f.readlines()
    
    # Keep the lines in the long file that match the ids in the short file
    filtered_long_lines = [line for line in long_lines if line.split(' (')[-1].strip().strip(')') in short_ids]
    
    with open(output_path, 'w', encoding='utf8') as f:
        f.write(''.join(filtered_long_lines))


if __name__ == '__main__':
    # the transcript file in RealMAN dataset
    long_path = '/data/home/RealisticAudio/RealMAN_modified/transcriptions.trn'
    # name of the transcript file generated by ./1_convert_json_to_trn.py
    transcript_name = "ma_noisy_speech"

    for mode in ['moving', 'static']:
        print(f'\nProcessing {mode} scenes...')
        scenes_path = f'./examples/realman/results/{mode}'
        scenes = [name for name in os.listdir(scenes_path) if os.path.isdir(os.path.join(scenes_path, name))]
        print(scenes)
        print(len(scenes))

        results_path = f'./examples/realman/results/{mode}'

        for scene in scenes:
            scene_path = os.path.join(results_path, scene)
            short_path = f'{scene_path}/{transcript_name}_{scene}.trn'
            output_path = f'{scene_path}/ref_{scene}.trn'
            
            filter_references(long_path, short_path, output_path) 

