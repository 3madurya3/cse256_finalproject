import os
import pandas as pd
import random
import json

def select_half_repeatable(input_list, seed=42):
    random.seed(seed)
    shuffled = input_list[:]
    random.shuffle(shuffled)
    half_size = len(input_list) // 2
    selected_half = shuffled[:half_size]
    return selected_half

def get_file_info():
    path = "/Users/maduryasuresh/Desktop/CSE256/final_project/l2arctic_release_v5.0"
    directories = [
            d for d in os.listdir(path)
            if os.path.isdir(os.path.join(path, d)) and d != "suitcase_corpus"
        ]
    print(directories)
    total_len = 0

    language_files = {} # note that the wav files and transcript file names are the same

    for d in directories:
        p_wav = path + "/" + d + "/wav"
        p_tr = path + "/" + d + "/transcript"
        wav_files = [item for item in os.listdir(p_wav) if item != ".DS_Store"]
        tr_files = [item for item in os.listdir(p_tr) if item != ".DS_Store"]
        print(d, len(wav_files), len(tr_files))

        language_files[d] = select_half_repeatable(wav_files) # randomly pick half

        total_len += len(wav_files)

    new_total_len = 0
    #print(language_files['HJK'])
    for key, value in language_files.items():
        print(key, len(value))
        new_total_len += len(value)

    print(total_len) # cut it in half because of computing power restrictions
    print(new_total_len)
    return language_files

def get_transcript(path):
    first_line = ""

    with open(path, "r") as file:
        first_line = file.readline().strip()

    #print(first_line)
    return first_line

def turn_to_json(language_files):
    json_list = []
    #test = 0
    past_key = ""
    for key, value_list in language_files.items():
        #print(test)
        #if (test == 10):
        #    break
        if key != past_key:
            print('key:', key)
            past_key = key
        for file in value_list:
            audio_file = f"/Users/maduryasuresh/Desktop/CSE256/final_project/l2arctic_release_v5.0/{key}/wav/{file}"
            transcript = get_transcript(f"/Users/maduryasuresh/Desktop/CSE256/final_project/l2arctic_release_v5.0/{key}/transcript/{file[:-4]}.txt")
            json_object = {
                "audio_file": audio_file,  # path to file
                "transcript": transcript 
            }
            json_list.append(json_object)

        #test += 1

    json_string = json.dumps(json_list, indent=2)
    #print(json_string)

    with open('./train_dev_test.json', "w") as output_file:
        output_file.write(json_string)

    return

def split_data(json_list, train_ratio=0.8, dev_ratio=0.1, test_ratio=0.1):
    random.shuffle(json_list)

    total = len(json_list) # split into train dev test
    print(total)
    train_size = 10742
    dev_size = 1343
    test_size = 1343

    print(train_size, dev_size, test_size)

    train_set = json_list[:train_size]
    dev_set = json_list[train_size:train_size+dev_size]
    test_set = json_list[train_size+dev_size:]

    return train_set, dev_set, test_set

def analyze_split():

    with open('./train_sixteenth_noABA.json', "r") as file:
        train = json.load(file)
    with open('./dev_sixteenth_noABA.json', "r") as file:
        dev = json.load(file)
    with open('./test_sixteenth_noABA.json', "r") as file:
        test = json.load(file)

    train_files = [obj["audio_file"] for obj in train]
    dev_files = [obj["audio_file"] for obj in dev]
    test_files = [obj["audio_file"] for obj in test]

    #print(dev_files)

    path = "/Users/maduryasuresh/Desktop/CSE256/final_project/l2arctic_release_v5.0"
    directories = [
            d for d in os.listdir(path)
            if os.path.isdir(os.path.join(path, d)) and d != "suitcase_corpus"
        ]
    print(directories)

    train_count = {}
    dev_count = {}
    test_count = {}

    for d in directories:
        for f in train_files:
            if d in f:
                if d not in train_count:
                    train_count[d] = 1
                else:
                    train_count[d] += 1
        for f in dev_files:
            if d in f:
                if d not in dev_count:
                    dev_count[d] = 1
                else:
                    dev_count[d] += 1
        for f in test_files:
            if d in f:
                if d not in test_count:
                    test_count[d] = 1
                else:
                    test_count[d] += 1

    print('train_count:', train_count)
    print('dev_count:', dev_count)
    print('test_count:', test_count)


    return

def halve():
    with open('./train_eighth_noABA.json', "r") as file:
        train = json.load(file)
    with open('./dev_eighth_noABA.json', "r") as file:
        dev = json.load(file)
    with open('./test_eighth_noABA.json', "r") as file:
        test = json.load(file)

    train_files = [obj["audio_file"] for obj in train]
    dev_files = [obj["audio_file"] for obj in dev]
    test_files = [obj["audio_file"] for obj in test]

    print(len(train_files))
    print(len(dev_files))
    print(len(test_files))

    random.shuffle(train)
    random.shuffle(dev)
    random.shuffle(test)

    train_half_size = 1290
    dev_half_size = 160
    test_half_size = 160

    train_half_set = train[:train_half_size]
    dev_half_set = dev[:dev_half_size]
    test_half_set = test[:test_half_size]

    with open('train_sixteenth_noABA.json', "w") as file:
        json.dump(train_half_set, file, indent=2)
    with open('dev_sixteenth_noABA.json', "w") as file:
        json.dump(dev_half_set, file, indent=2)
    with open('test_sixteenth_noABA.json', "w") as file:
        json.dump(test_half_set, file, indent=2)


    return

def remove_ABA():
    with open('./train_half.json', "r") as file:
        train = json.load(file)
    with open('./dev_half.json', "r") as file:
        dev = json.load(file)
    with open('./test_half.json', "r") as file:
        test = json.load(file)

    # Filter out objects where 'audio_file' contains "ABA"
    filtered_train = [obj for obj in train if "ABA" not in obj.get("audio_file", "")]
    filtered_dev = [obj for obj in dev if "ABA" not in obj.get("audio_file", "")]
    filtered_test = [obj for obj in test if "ABA" not in obj.get("audio_file", "")]

    with open('train_half_noABA.json', "w") as file:
        json.dump(filtered_train, file, indent=2)
    with open('dev_half_noABA.json', "w") as file:
        json.dump(filtered_dev, file, indent=2)
    with open('test_half_noABA.json', "w") as file:
        json.dump(filtered_test, file, indent=2)

    return

def train_testjson():
    with open('./train_half.json', "r") as file:
        train = json.load(file)
    train_small_set = train[:10]

    with open('train_small.json', "w") as file:
        json.dump(train_small_set, file, indent=2)


    return

def main():
    #language_files = get_file_info()
    #get_transcript("/Users/maduryasuresh/Desktop/CSE256/final_project/l2arctic_release_v5.0/ABA/transcript/arctic_a0002.txt")
    #turn_to_json(language_files)

    """ with open('./train_dev_test.json', "r") as file:
        json_list = json.load(file)

    train_set, dev_set, test_set = split_data(json_list, train_ratio=0.8, dev_ratio=0.2, test_ratio=0.2)
    with open('train.json', "w") as file:
        json.dump(train_set, file, indent=2)
    with open('dev.json', "w") as file:
        json.dump(dev_set, file, indent=2)
    with open('test.json', "w") as file:
        json.dump(test_set, file, indent=2) """
        
    #analyze_split()
    #halve()
    #analyze_split()
    train_testjson()

    #remove_ABA()


    return

if __name__ == "__main__":
    main()


"""['ZHAA', 'TLV', 'BWC', 'YDCK', 'SVBI', 'NCC', 'RRBI', 'EBVS', 'HKK', 'TNI', 'YKWK', 'HQTV', 'THV', 'TXHC', 'LXC', 'MBMPS', 'ABA', 'SKA', 'ASI', 'NJS', 'HJK', 'ERMS', 'YBAA', 'PNV']
ZHAA 1132 1132
TLV 1132 1132
BWC 1130 1130
YDCK 1131 1131
SVBI 1132 1132
NCC 1131 1131
RRBI 1130 1130
EBVS 1007 1007
HKK 1131 1131
TNI 1131 1131
YKWK 1131 1131
HQTV 1132 1132
THV 1132 1132
TXHC 1132 1132
LXC 1131 1131
MBMPS 1132 1132
ABA 1129 1129
SKA 974 974
ASI 1131 1131
NJS 1131 1131
HJK 1131 1131
ERMS 1132 1132
YBAA 1130 1130
PNV 1132 1132
ZHAA 566
TLV 566
BWC 565
YDCK 565
SVBI 566
NCC 565
RRBI 565
EBVS 503
HKK 565
TNI 565
YKWK 565
HQTV 566
THV 566
TXHC 566
LXC 565
MBMPS 566
ABA 564
SKA 487
ASI 565
NJS 565
HJK 565
ERMS 566
YBAA 565
PNV 566
26867
13428

"""

"""
train_count: {'ZHAA': 457, 'TLV': 441, 'BWC': 448, 'YDCK': 451, 'SVBI': 452, 'NCC': 456, 'RRBI': 459, 'EBVS': 405, 'HKK': 461, 'TNI': 461, 'YKWK': 449, 'HQTV': 451, 'THV': 466, 'TXHC': 450, 'LXC': 463, 'MBMPS': 454, 'ABA': 446, 'SKA': 401, 'ASI': 454, 'NJS': 439, 'HJK': 451, 'ERMS': 441, 'YBAA': 441, 'PNV': 445}
dev_count: {'ZHAA': 50, 'TLV': 47, 'BWC': 60, 'YDCK': 41, 'SVBI': 56, 'NCC': 55, 'RRBI': 58, 'EBVS': 48, 'HKK': 53, 'TNI': 53, 'YKWK': 54, 'HQTV': 68, 'THV': 53, 'TXHC': 54, 'LXC': 47, 'MBMPS': 64, 'ABA': 57, 'SKA': 42, 'ASI': 65, 'NJS': 58, 'HJK': 67, 'ERMS': 65, 'YBAA': 66, 'PNV': 62}
test_count: {'ZHAA': 59, 'TLV': 78, 'BWC': 57, 'YDCK': 73, 'SVBI': 58, 'NCC': 54, 'RRBI': 48, 'EBVS': 50, 'HKK': 51, 'TNI': 51, 'YKWK': 62, 'HQTV': 47, 'THV': 47, 'TXHC': 62, 'LXC': 55, 'MBMPS': 48, 'ABA': 61, 'SKA': 44, 'ASI': 46, 'NJS': 68, 'HJK': 47, 'ERMS': 60, 'YBAA': 58, 'PNV': 59}

10740
1343
1343
"""

"""
['ZHAA', 'TLV', 'BWC', 'YDCK', 'SVBI', 'NCC', 'RRBI', 'EBVS', 'HKK', 'TNI', 'YKWK', 'HQTV', 'THV', 'TXHC', 'LXC', 'MBMPS', 'ABA', 'SKA', 'ASI', 'NJS', 'HJK', 'ERMS', 'YBAA', 'PNV']
train_count: {'ZHAA': 221, 'TLV': 224, 'BWC': 228, 'YDCK': 232, 'SVBI': 234, 'NCC': 225, 'RRBI': 214, 'EBVS': 212, 'HKK': 222, 'TNI': 233, 'YKWK': 234, 'HQTV': 228, 'THV': 238, 'TXHC': 221, 'LXC': 235, 'MBMPS': 213, 'ABA': 209, 'SKA': 212, 'ASI': 216, 'NJS': 214, 'HJK': 247, 'ERMS': 228, 'YBAA': 215, 'PNV': 215}
dev_count: {'ZHAA': 27, 'TLV': 27, 'BWC': 24, 'YDCK': 24, 'SVBI': 26, 'NCC': 25, 'RRBI': 29, 'EBVS': 27, 'HKK': 30, 'TNI': 28, 'YKWK': 28, 'HQTV': 31, 'THV': 21, 'TXHC': 18, 'LXC': 29, 'MBMPS': 34, 'ABA': 30, 'SKA': 22, 'ASI': 30, 'NJS': 31, 'HJK': 33, 'ERMS': 39, 'YBAA': 30, 'PNV': 28}
test_count: {'ZHAA': 28, 'TLV': 39, 'BWC': 29, 'YDCK': 33, 'SVBI': 28, 'NCC': 25, 'RRBI': 26, 'EBVS': 27, 'HKK': 21, 'TNI': 27, 'YKWK': 33, 'HQTV': 20, 'THV': 22, 'TXHC': 32, 'LXC': 28, 'MBMPS': 26, 'ABA': 39, 'SKA': 25, 'ASI': 18, 'NJS': 33, 'HJK': 27, 'ERMS': 31, 'YBAA': 27, 'PNV': 27}
"""

"""ABA: 0

['ZHAA', 'TLV', 'BWC', 'YDCK', 'SVBI', 'NCC', 'RRBI', 'EBVS', 'HKK', 'TNI', 'YKWK', 'HQTV', 'THV', 'TXHC', 'LXC', 'MBMPS', 'ABA', 'SKA', 'ASI', 'NJS', 'HJK', 'ERMS', 'YBAA', 'PNV']
train_count: {'ZHAA': 221, 'TLV': 224, 'BWC': 228, 'YDCK': 232, 'SVBI': 234, 'NCC': 225, 'RRBI': 214, 'EBVS': 212, 'HKK': 222, 'TNI': 233, 'YKWK': 234, 'HQTV': 228, 'THV': 238, 'TXHC': 221, 'LXC': 235, 'MBMPS': 213, 'SKA': 212, 'ASI': 216, 'NJS': 214, 'HJK': 247, 'ERMS': 228, 'YBAA': 215, 'PNV': 215}
dev_count: {'ZHAA': 27, 'TLV': 27, 'BWC': 24, 'YDCK': 24, 'SVBI': 26, 'NCC': 25, 'RRBI': 29, 'EBVS': 27, 'HKK': 30, 'TNI': 28, 'YKWK': 28, 'HQTV': 31, 'THV': 21, 'TXHC': 18, 'LXC': 29, 'MBMPS': 34, 'SKA': 22, 'ASI': 30, 'NJS': 31, 'HJK': 33, 'ERMS': 39, 'YBAA': 30, 'PNV': 28}
test_count: {'ZHAA': 28, 'TLV': 39, 'BWC': 29, 'YDCK': 33, 'SVBI': 28, 'NCC': 25, 'RRBI': 26, 'EBVS': 27, 'HKK': 21, 'TNI': 27, 'YKWK': 33, 'HQTV': 20, 'THV': 22, 'TXHC': 32, 'LXC': 28, 'MBMPS': 26, 'SKA': 25, 'ASI': 18, 'NJS': 33, 'HJK': 27, 'ERMS': 31, 'YBAA': 27, 'PNV': 27}"""


"""
train_half_size = 2580
dev_half_size = 320
test_half_size = 320
['ZHAA', 'TLV', 'BWC', 'YDCK', 'SVBI', 'NCC', 'RRBI', 'EBVS', 'HKK', 'TNI', 'YKWK', 'HQTV', 'THV', 'TXHC', 'LXC', 'MBMPS', 'ABA', 'SKA', 'ASI', 'NJS', 'HJK', 'ERMS', 'YBAA', 'PNV']
train_count: {'ZHAA': 103, 'TLV': 120, 'BWC': 126, 'YDCK': 108, 'SVBI': 115, 'NCC': 114, 'RRBI': 110, 'EBVS': 97, 'HKK': 119, 'TNI': 121, 'YKWK': 116, 'HQTV': 105, 'THV': 119, 'TXHC': 118, 'LXC': 114, 'MBMPS': 96, 'SKA': 108, 'ASI': 123, 'NJS': 95, 'HJK': 119, 'ERMS': 121, 'YBAA': 107, 'PNV': 106}
dev_count: {'ZHAA': 12, 'TLV': 16, 'BWC': 4, 'YDCK': 9, 'SVBI': 15, 'NCC': 14, 'RRBI': 14, 'EBVS': 12, 'HKK': 16, 'TNI': 15, 'YKWK': 20, 'HQTV': 22, 'THV': 11, 'TXHC': 6, 'LXC': 12, 'MBMPS': 13, 'SKA': 16, 'ASI': 16, 'NJS': 13, 'HJK': 15, 'ERMS': 19, 'YBAA': 16, 'PNV': 14}
test_count: {'ZHAA': 18, 'TLV': 16, 'BWC': 12, 'YDCK': 18, 'SVBI': 9, 'NCC': 13, 'RRBI': 15, 'EBVS': 12, 'HKK': 13, 'TNI': 11, 'YKWK': 19, 'HQTV': 17, 'THV': 13, 'TXHC': 14, 'LXC': 13, 'MBMPS': 14, 'SKA': 10, 'ASI': 9, 'NJS': 13, 'HJK': 13, 'ERMS': 17, 'YBAA': 16, 'PNV': 15}"""

"""['ZHAA', 'TLV', 'BWC', 'YDCK', 'SVBI', 'NCC', 'RRBI', 'EBVS', 'HKK', 'TNI', 'YKWK', 'HQTV', 'THV', 'TXHC', 'LXC', 'MBMPS', 'ABA', 'SKA', 'ASI', 'NJS', 'HJK', 'ERMS', 'YBAA', 'PNV']
train_count: {'ZHAA': 44, 'TLV': 63, 'BWC': 65, 'YDCK': 60, 'SVBI': 52, 'NCC': 53, 'RRBI': 53, 'EBVS': 45, 'HKK': 61, 'TNI': 52, 'YKWK': 59, 'HQTV': 56, 'THV': 56, 'TXHC': 58, 'LXC': 66, 'MBMPS': 52, 'SKA': 65, 'ASI': 65, 'NJS': 47, 'HJK': 53, 'ERMS': 58, 'YBAA': 49, 'PNV': 58}
dev_count: {'ZHAA': 7, 'TLV': 11, 'YDCK': 2, 'SVBI': 6, 'NCC': 6, 'RRBI': 9, 'EBVS': 8, 'HKK': 9, 'TNI': 7, 'YKWK': 10, 'HQTV': 14, 'THV': 5, 'TXHC': 2, 'LXC': 7, 'MBMPS': 3, 'SKA': 10, 'ASI': 10, 'NJS': 5, 'HJK': 6, 'ERMS': 10, 'YBAA': 7, 'PNV': 6}
test_count: {'ZHAA': 8, 'TLV': 6, 'BWC': 9, 'YDCK': 13, 'SVBI': 5, 'NCC': 9, 'RRBI': 8, 'EBVS': 6, 'HKK': 6, 'TNI': 5, 'YKWK': 4, 'HQTV': 6, 'THV': 4, 'TXHC': 5, 'LXC': 11, 'MBMPS': 7, 'SKA': 6, 'ASI': 4, 'NJS': 8, 'HJK': 5, 'ERMS': 9, 'YBAA': 6, 'PNV': 10}"""